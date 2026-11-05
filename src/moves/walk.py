# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import os
from enum import Enum, auto

import onnxruntime as ort
import numpy as np

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState
from onnx_obs_stats import load_obs_normalizer_stats


# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False

# Set to True (or MICROBAN_WALK_DEBUG=1) to print projected_gravity/gyro/lean/velocity
# a few times a second while walking is active — useful for diagnosing balance loss
# (growing lean, fall) or a bad observation channel (e.g. an out-of-scale velocity
# reading, see docs/dev/zubr_real_hardware.md) without a full LOGGING dump.
DEBUG_PRINT = os.environ.get("MICROBAN_WALK_DEBUG", "0") == "1"
DEBUG_PRINT_INTERVAL_S = 0.3

# Policy name
AGENT_NAME = "walk.onnx"

# Out-of-domain (OOD) detector: per-tick, how many standard deviations the live
# observation's worst channel is from walk.onnx's own training-time per-channel
# mean/std (see onnx_obs_stats.py) — a cheap, always-available signal of "has the
# robot's actual state drifted somewhere the policy never saw in training", with no
# extra model or training data needed. Not yet tuned against real logged data —
# override via env var while calibrating on real hardware without a code change.
OOD_Z_THRESHOLD = float(os.environ.get("MICROBAN_OOD_Z_THRESHOLD", "6.0"))
# Consecutive ticks required to flip the safety state, in either direction (raise ->
# ramp to the policy's default pose; clear -> ramp back into live inference) — fixed
# at 8 per spec, not a tuning knob.
OOD_CONSECUTIVE_TICKS = 8

# Secondary OOD signal, layered on top of the z-score above. The z-score is a
# *marginal* per-channel statistic pooled over the whole training distribution, so
# joint velocities typical of walking at some other commanded speed don't look
# anomalous even when they occur at a near-zero velocity command — e.g. a robot
# held in the air with its feet off the ground "walks on air" trying to balance,
# which training never saw, but the marginal z-score alone doesn't catch it. This
# check instead looks for the direct behavioral inconsistency: active leg motion
# while the robot is being commanded to stand still. Not yet tuned against real
# logged data — override via env var while calibrating on real hardware.
STANDING_COMMAND_DEADZONE = float(os.environ.get("MICROBAN_STANDING_DEADZONE", "0.05"))
STANDING_MAX_JOINT_VEL = float(os.environ.get("MICROBAN_STANDING_MAX_JOINT_VEL", "1.0"))


class _SafetyState(Enum):
    """WalkMove's own internal OOD-response state, layered on top of the scheduler's
    MoveState (which only tracks user-driven enable/disable toggling, not whether the
    policy's output is currently trusted)."""

    NORMAL = auto()           # running the policy normally every tick
    RAMPING_TO_SAFE = auto()  # OOD flagged for OOD_CONSECUTIVE_TICKS; ramping to the policy's own default pose
    SAFE_HOLD = auto()        # holding default pose; inference paused; watching for OOD to clear
    RECOVERING = auto()       # OOD clear for OOD_CONSECUTIVE_TICKS; blending live inference back in


class WalkMove(Move):
    """Walk using a RL policy trained in simulation."""

    def __init__(self, controller: ControllerProtocol | None = None, start_lerp_duration: float = 1.5) -> None:
        super().__init__()
        self._controller = controller

        # Load ONNX policy
        self._ort_session = ort.InferenceSession(f"src/agents/{AGENT_NAME}")

        # Reference pose and per-joint order: read from ONNX metadata. joint_names is the
        # exact order the policy was trained on for joint_pos/joint_vel/actions — observations
        # must be built and actions decoded in this order, NOT the constants.py
        # OBSERVATION_DOF_ORDER, which lists joints in a different order (shoulders before
        # hips) and previously caused action[i] to be applied to the wrong joint (e.g. a
        # shoulder-sized action landing on a hip), producing wildly out-of-range targets that
        # tripped the overcurrent safety.
        meta = self._ort_session.get_modelmeta().custom_metadata_map
        self._dof_order = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(self._dof_order, positions))
        self._last_action = [0.0] * len(self._dof_order)

        # action_scale is per-joint in newer exports (e.g. a reduced 0.3 scale on the
        # shoulder joints only, to curb arm swing) but a single bare scalar in older ones
        # (every checkpoint through walk0930.onnx) — a hardcoded scalar here silently
        # applied the WRONG per-joint scale on any checkpoint using the newer format
        # (0.8 instead of 0.3 on the shoulders — 2.67x too large), which is exactly what
        # produced visibly exaggerated, mirrored shoulder-pitch twitching at a near-zero
        # command: mjlab's own JointPositionAction reads this per-joint, so it never
        # showed the fault there. Broadcast a single value across every joint for the
        # old format; use the array as-is for the new one.
        scales = [float(v) for v in meta["action_scale"].split(",")]
        if len(scales) == 1:
            scales = scales * len(self._dof_order)
        self._action_scale: dict[str, float] = dict(zip(self._dof_order, scales))

        # on_start ramps target_angles from wherever the robot was holding to
        # _default_pose over this many seconds, instead of jumping there in one
        # tick — a one-tick jump from e.g. NEUTRAL_POSE can move many joints at
        # once and spike the estimated pack current enough to trip the
        # overcurrent safety (see Scheduler._check_overcurrent).
        self._start_lerp_duration = start_lerp_duration
        self._start_lerp_time_s: float | None = None
        self._start_lerp_angles: dict[str, float] = {}

        # Detect reference phase from model input size:
        # base_obs = gyro(3) + proj_grav(3) + pos(N) + vel(N) + action(N) + cmd(3)
        # phase_obs = base_obs + phase(2)
        base_obs_size = 3 + 3 + 3 * len(self._dof_order) + 3
        self._use_reference_phase: bool = self._ort_session.get_inputs()[0].shape[1] > base_obs_size
        self._phase_step = 0
        self._phase_total_steps = 20

        # Safety parameters
        self._projected_gravity_z_threshold = -0.5  # Threshold for detecting a fall based on projected gravity

        # OOD detector stats (see OOD_Z_THRESHOLD above / onnx_obs_stats.py). Sliced to
        # the sensor/state channels only — gyro + projected_gravity + joint_pos +
        # joint_vel + last_action — excluding command (user/agent-driven intent, not a
        # signal of the robot's own state) and reference_phase (a deterministic timing
        # signal, same reasoning), which both sit after this prefix in build_observation().
        obs_mean, obs_std = load_obs_normalizer_stats(f"src/agents/{AGENT_NAME}")
        self._ood_obs_len = 3 + 3 + 3 * len(self._dof_order)
        self._ood_mean = obs_mean[: self._ood_obs_len]
        self._ood_std = obs_std[: self._ood_obs_len]

        self._safety_state = _SafetyState.NORMAL
        self._ood_consecutive = 0
        self._clear_consecutive = 0
        self._safe_ramp_time_s: float | None = None
        self._safe_ramp_angles: dict[str, float] = {}
        self._recover_time_s: float | None = None
        self._last_ood_warn_s: float = 0.0

        # Debug print throttle (see DEBUG_PRINT above)
        self._last_debug_print_s: float = 0.0

        # Logging
        self.position = {
            "head": [],
            "left_hip_yaw": [],
            "left_hip_roll": [],
            "left_hip_pitch": [],
            "left_knee": [],
            "left_ankle_pitch": [],
            "left_ankle_roll": [],
            "right_hip_yaw": [],
            "right_hip_roll": [],
            "right_hip_pitch": [],
            "right_knee": [],
            "right_ankle_pitch": [],
            "right_ankle_roll": [],
            "left_shoulder_pitch": [],
            "left_shoulder_roll": [],
            "left_elbow": [],
            "right_shoulder_pitch": [],
            "right_shoulder_roll": [],
            "right_elbow": [],
        }
        self.voltage = {
            "head": [],
            "left_hip_yaw": [],
            "left_hip_roll": [],
            "left_hip_pitch": [],
            "left_knee": [],
            "left_ankle_pitch": [],
            "left_ankle_roll": [],
            "right_hip_yaw": [],
            "right_hip_roll": [],
            "right_hip_pitch": [],
            "right_knee": [],
            "right_ankle_pitch": [],
            "right_ankle_roll": [],
            "left_shoulder_pitch": [],
            "left_shoulder_roll": [],
            "left_elbow": [],
            "right_shoulder_pitch": [],
            "right_shoulder_roll": [],
            "right_elbow": [],
        }
        
    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        # Defensive: on_stop() already resets the OOD state machine on every normal
        # deactivation, but a fresh start should never inherit a stale non-NORMAL
        # state regardless of how the previous run ended.
        self._safety_state = _SafetyState.NORMAL
        self._ood_consecutive = 0
        self._clear_consecutive = 0

        # Ramp the pose to _default_pose while still holding the stiffer,
        # pre-walk kp (whatever was in effect, typically KP_DEFAULT) — the
        # robot stays rigid while it moves to the walk starting stance. Kp is
        # only dropped to KP_RL right as we hand off to step() below, since
        # softening the joints while they're still away from the pose the RL
        # policy was trained to start from (rather than at it) is what let
        # gravity yank them, spiking the estimated pack current enough to
        # trip the overcurrent safety (see Scheduler._check_overcurrent).
        if self._start_lerp_time_s is None:
            self._start_lerp_time_s = obs.robot_state.time_s
            self._start_lerp_angles = {
                name: obs.robot_state.motor_positions.get(name, 0.0) for name in self._default_pose
            }

        t = min((obs.robot_state.time_s - self._start_lerp_time_s) / self._start_lerp_duration, 1.0)
        for name, target in self._default_pose.items():
            command.target_angles[name] = self._start_lerp_angles[name] * (1.0 - t) + target * t

        if DEBUG_PRINT and (obs.robot_state.time_s - self._last_debug_print_s) >= DEBUG_PRINT_INTERVAL_S:
            self._last_debug_print_s = obs.robot_state.time_s
            pg = obs.robot_state.projected_gravity
            lean_deg = np.degrees(np.arccos(max(-1.0, min(1.0, -pg[2])))) if pg else float("nan")
            print(
                f"Walk: RAMPING t={t:.2f} proj_grav={tuple(round(v, 3) for v in pg)} lean={lean_deg:.1f}deg",
                end="\r\n", flush=True,
            )

        if t >= 1.0:
            if self._controller is not None:
                ids = list(MOTOR_TO_ID.values())
                self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
            self._start_lerp_time_s = None
            self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Update reference phase
        if self._use_reference_phase:
            commanded_vel = np.mean([np.abs(obs.user_input.velocity["vx"]), np.abs(obs.user_input.velocity["vy"]), np.abs(obs.user_input.velocity["vtheta"])])
            if commanded_vel > 0.01:
                self._phase_step += 1
            else:
                self._phase_step = 0

        # Safety check: if the robot is fallen, stop the policy. NOTE: when this fires,
        # step() returns WITHOUT touching command.target_angles, which the scheduler
        # freshly initializes to NEUTRAL_POSE every tick — so a "fallen" robot gets
        # commanded straight to NEUTRAL_POSE (not held in place), which can look like
        # abrupt/repetitive leg motion if this flips on and off near the threshold.
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            if DEBUG_PRINT:
                print(
                    f"Walk: FALL DETECTED, policy suspended (falling back to NEUTRAL_POSE) — "
                    f"proj_grav={tuple(round(v, 3) for v in obs.robot_state.projected_gravity)}",
                    end="\r\n", flush=True,
                )
            return

        # OOD check: always run, regardless of safety state — needed both to decide
        # NORMAL -> RAMPING_TO_SAFE and to know when SAFE_HOLD -> RECOVERING is due.
        input_obs = self.build_observation(obs)
        ood_z = self._ood_z_score(input_obs)
        standing_activity_ood = self._standing_activity_is_ood(obs, input_obs)
        is_ood = ood_z > OOD_Z_THRESHOLD or standing_activity_ood
        if is_ood:
            self._ood_consecutive += 1
            self._clear_consecutive = 0
        else:
            self._clear_consecutive += 1
            self._ood_consecutive = 0

        if DEBUG_PRINT and is_ood and (obs.robot_state.time_s - self._last_ood_warn_s) >= DEBUG_PRINT_INTERVAL_S:
            self._last_ood_warn_s = obs.robot_state.time_s
            reason = "standing-activity" if standing_activity_ood else f"z={ood_z:.1f}"
            print(
                f"Walk: OOD {reason} (z threshold {OOD_Z_THRESHOLD:.1f}) "
                f"consecutive={self._ood_consecutive} state={self._safety_state.name}",
                end="\r\n", flush=True,
            )

        if self._safety_state == _SafetyState.NORMAL and self._ood_consecutive >= OOD_CONSECUTIVE_TICKS:
            self._enter_safe_ramp(obs)

        if self._safety_state == _SafetyState.RAMPING_TO_SAFE:
            self._tick_safe_ramp(obs, command)
            return

        if self._safety_state == _SafetyState.SAFE_HOLD:
            for name, target in self._default_pose.items():
                command.target_angles[name] = target
            if self._clear_consecutive >= OOD_CONSECUTIVE_TICKS:
                self._enter_recovery(obs)
            return

        if self._safety_state == _SafetyState.RECOVERING:
            self._tick_recovery(obs, command, input_obs)
            return

        # _SafetyState.NORMAL, and not (yet) tripped: run inference as usual.
        self._run_inference(obs, command, input_obs)

    def _ood_z_score(self, input_obs: list[float]) -> float:
        """Max |z-score|, over the sensor/state channels only, of input_obs (the
        observation about to be — or that was just — fed to the policy) against
        walk.onnx's own training-time per-channel mean/std. See OOD_Z_THRESHOLD."""
        vec = np.asarray(input_obs[: self._ood_obs_len], dtype=np.float64)
        z = (vec - self._ood_mean) / self._ood_std
        return float(np.max(np.abs(z)))

    def _standing_activity_is_ood(self, obs: Observation, input_obs: list[float]) -> bool:
        """True if vx/vy/vtheta are all commanded ~0 but a joint is moving faster than
        quiet standing ever does — see STANDING_MAX_JOINT_VEL above for why the
        marginal z-score alone can't catch this."""
        vx, vy, vtheta = (obs.user_input.velocity[k] for k in ("vx", "vy", "vtheta"))
        if max(abs(vx), abs(vy), abs(vtheta)) > STANDING_COMMAND_DEADZONE:
            return False
        vel_start = 6 + len(self._dof_order)
        vel_end = vel_start + len(self._dof_order)
        joint_vel = input_obs[vel_start:vel_end]
        return max(abs(v) for v in joint_vel) > STANDING_MAX_JOINT_VEL

    def _run_inference(
        self, obs: Observation, command: MotorCommand, input_obs: list[float], blend: float = 1.0
    ) -> None:
        """Run the policy and write command.target_angles. `blend` scales the action's
        contribution (1.0 = normal; <1.0 while RECOVERING ramps it back in) — the
        default_pose offset is always applied in full regardless, so blend=0 reduces
        exactly to holding default_pose, matching SAFE_HOLD."""
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        self._last_action = action.tolist()

        for i, name in enumerate(self._dof_order):
            command.target_angles[name] = (
                self._default_pose[name] + action[i] * self._action_scale[name] * blend
            )

        if DEBUG_PRINT and (obs.robot_state.time_s - self._last_debug_print_s) >= DEBUG_PRINT_INTERVAL_S:
            self._last_debug_print_s = obs.robot_state.time_s
            pg = obs.robot_state.projected_gravity
            gyro = obs.robot_state.gyro
            lean_deg = np.degrees(np.arccos(max(-1.0, min(1.0, -pg[2])))) if pg else float("nan")
            velocities = [obs.robot_state.motor_velocities[name] for name in self._dof_order]
            max_vel_name = self._dof_order[int(np.argmax(np.abs(velocities)))]
            blend_str = f" blend={blend:.2f}" if blend < 1.0 else ""
            print(
                f"Walk: proj_grav={tuple(round(v, 3) for v in pg)} gyro={tuple(round(v, 3) for v in gyro)} "
                f"lean={lean_deg:.1f}deg max|action|={float(np.max(np.abs(action))):.2f} "
                f"max|vel|={float(np.max(np.abs(velocities))):.2f} rad/s ({max_vel_name}){blend_str}",
                end="\r\n", flush=True,
            )

        # Log positions and voltages
        if LOGGING:
            for name in MOTOR_TO_ID.keys():
                self.position[name].append(obs.robot_state.motor_positions[name])
                self.voltage[name].append(obs.robot_state.motor_voltages[name])

    def _enter_safe_ramp(self, obs: Observation) -> None:
        self._safety_state = _SafetyState.RAMPING_TO_SAFE
        self._safe_ramp_time_s = obs.robot_state.time_s
        self._safe_ramp_angles = {
            name: obs.robot_state.motor_positions.get(name, self._default_pose[name])
            for name in self._default_pose
        }
        # Inference is suspended for the rest of this safety episode (ramp + hold), so
        # nothing would otherwise update _last_action — left alone, it stays frozen at
        # whatever the policy last output right before the trip (plausibly itself an
        # extreme value, since that's often what triggered the OOD flag in the first
        # place), which then feeds straight back into the OOD check's own last_action
        # channel every subsequent tick and can keep it OOD indefinitely regardless of
        # the robot's actual physical state. target = default_pose + action*scale, and
        # the ramp converges to (then holds at) target == default_pose exactly, so the
        # action that's actually consistent with what's being commanded is zero —
        # reporting that (instead of the stale value) keeps the observation internally
        # consistent with the real command for the rest of this episode.
        self._last_action = [0.0] * len(self._dof_order)
        if DEBUG_PRINT:
            print(
                f"Walk: OOD confirmed for {OOD_CONSECUTIVE_TICKS} consecutive ticks — "
                "ramping to default pose, inference suspended",
                end="\r\n", flush=True,
            )

    def _tick_safe_ramp(self, obs: Observation, command: MotorCommand) -> None:
        assert self._safe_ramp_time_s is not None
        t = min((obs.robot_state.time_s - self._safe_ramp_time_s) / self._start_lerp_duration, 1.0)
        for name, target in self._default_pose.items():
            command.target_angles[name] = self._safe_ramp_angles[name] * (1.0 - t) + target * t
        if t >= 1.0:
            self._safety_state = _SafetyState.SAFE_HOLD
            self._safe_ramp_time_s = None
            if DEBUG_PRINT:
                print("Walk: reached default pose — holding, inference paused", end="\r\n", flush=True)

    def _enter_recovery(self, obs: Observation) -> None:
        self._safety_state = _SafetyState.RECOVERING
        self._recover_time_s = obs.robot_state.time_s
        if DEBUG_PRINT:
            print(
                f"Walk: OOD clear for {OOD_CONSECUTIVE_TICKS} consecutive ticks — "
                "blending inference back in",
                end="\r\n", flush=True,
            )

    def _tick_recovery(self, obs: Observation, command: MotorCommand, input_obs: list[float]) -> None:
        assert self._recover_time_s is not None
        blend = min((obs.robot_state.time_s - self._recover_time_s) / self._start_lerp_duration, 1.0)
        self._run_inference(obs, command, input_obs, blend=blend)
        if blend >= 1.0:
            self._safety_state = _SafetyState.NORMAL
            self._recover_time_s = None
            if DEBUG_PRINT:
                print("Walk: recovery ramp complete — back to normal inference", end="\r\n", flush=True)

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state."""
        input_obs = []
        
        # IMU data: gyroscope and projected gravity in body frame
        input_obs.extend(obs.robot_state.gyro)
        input_obs.extend(obs.robot_state.projected_gravity)
        
        # Motor positions
        for name in self._dof_order:
            input_obs.append(obs.robot_state.motor_positions[name] - self._default_pose[name])
        
        # Motor velocities
        for name in self._dof_order:
            input_obs.append(obs.robot_state.motor_velocities[name])
        
        # Last action
        input_obs.extend(self._last_action)

        # Command
        input_obs.append(obs.user_input.velocity["vx"])
        input_obs.append(obs.user_input.velocity["vy"])
        input_obs.append(obs.user_input.velocity["vtheta"])

        # Reference phase
        if self._use_reference_phase:
            reference_phase = (self._phase_step % self._phase_total_steps) / self._phase_total_steps * 2 * np.pi
            input_obs.append(np.cos(reference_phase))
            input_obs.append(np.sin(reference_phase))

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE

        # So the next on_start()/step() always begins fresh rather than resuming
        # mid-ramp or mid-hold from before this deactivation.
        self._safety_state = _SafetyState.NORMAL
        self._ood_consecutive = 0
        self._clear_consecutive = 0
        self._safe_ramp_time_s = None
        self._recover_time_s = None

        # Save json logs
        if LOGGING:
            import json
            with open("walk_log.json", "w") as f:
                json.dump({
                    "position": self.position,
                    "voltage": self.voltage,
                }, f, indent=4)