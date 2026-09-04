# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import onnxruntime as ort
import numpy as np

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState


# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False

# Policy name
AGENT_NAME = "walk.onnx"


class WalkMove(Move):
    """Walk using a RL policy trained in simulation."""

    def __init__(self, controller: ControllerProtocol | None = None, start_lerp_duration: float = 1.5) -> None:
        super().__init__()
        self._controller = controller

        # Load ONNX policy
        self._ort_session = ort.InferenceSession(f"src/agents/{AGENT_NAME}")

        self.action_scale = 1.0

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

        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            return
        
        # Run policy
        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        self._last_action = action.tolist()

        # Update command
        for i, name in enumerate(self._dof_order):
            command.target_angles[name] = self._default_pose[name] + action[i] * self.action_scale

        # Log positions and voltages
        if LOGGING:
            for name in MOTOR_TO_ID.keys():
                self.position[name].append(obs.robot_state.motor_positions[name])
                self.voltage[name].append(obs.robot_state.motor_voltages[name])

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

        # Save json logs
        if LOGGING:
            import json
            with open("walk_log.json", "w") as f:
                json.dump({
                    "position": self.position,
                    "voltage": self.voltage,
                }, f, indent=4)