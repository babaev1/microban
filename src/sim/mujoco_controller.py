# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import ctypes
import math
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer

if TYPE_CHECKING:
    from input.keyboard_input import KeyboardInputSource
    from sim.mujoco_input import MuJoCoInputSource
    from sim.state_stream import StateSender

from bam.model import load_model as bam_load_model
from bam.mujoco import MujocoController as BamController

from constants import MOTOR_TO_ID, ID_TO_MOTOR, NEUTRAL_POSE, KP_DEFAULT, BAM_VIN, BAM_VOLTAGE_DROP_GAIN, BAM_VIN_MIN, BAM_MAX_CURRENT

# GLFW key codes matching sim.mujoco_input's arrow-key handling (no glfw
# import needed here either — values are stable).
_GLFW_KEY_UP = 265
_GLFW_KEY_DOWN = 264
_GLFW_KEY_RIGHT = 262
_GLFW_KEY_LEFT = 263


def _can_create_gl_window() -> bool:
    """Probe whether MuJoCo's real GLX viewer can actually render here.

    ``mujoco.viewer.launch_passive`` calls glfwInit/glfwCreateWindow itself,
    and on failure that aborts the whole process instead of raising a
    catchable Python exception — so we can't just try it and fall back.

    A plain "did glfwCreateWindow succeed" check isn't enough either: over
    indirect GLX (e.g. forwarded through `ssh -X`), window creation commonly
    succeeds with a basic pixel format, but MuJoCo's MjrContext then fails
    creating its depth+stencil framebuffer ("Default framebuffer is not
    complete") — the exact same class of failure hit locally with the OSMesa
    fallback, just via a different path. Indirect GLX frequently can't
    negotiate the modern framebuffer config MuJoCo needs, independent of
    anything on this end.

    So this replicates MuJoCo's own check: create a hidden window with
    matching depth/stencil hints, make its context current, and confirm
    ``glCheckFramebufferStatus`` reports the default framebuffer complete —
    then tears it fully down before the real viewer touches GLFW at all.

    This is what lets MuJoCoController pick the right viewer automatically:
    a display that actually supports MuJoCo's GL requirements gets the full
    3D viewer; anything that doesn't (no display, a GPU that only does GLES
    like this board's PowerVR part, or indirect GLX that can't negotiate the
    needed framebuffer) gets _SoftwareViewer.
    """
    import glfw
    from OpenGL import GL

    try:
        if not glfw.init():
            return False
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.DEPTH_BITS, 24)
            glfw.window_hint(glfw.STENCIL_BITS, 8)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
            window = glfw.create_window(4, 4, "probe", None, None)
            if not bool(ctypes.cast(window, ctypes.c_void_p).value):
                return False
            try:
                glfw.make_context_current(window)
                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                return status == GL.GL_FRAMEBUFFER_COMPLETE
            except Exception:
                return False
            finally:
                glfw.destroy_window(window)
        finally:
            glfw.terminate()
    except Exception:
        return False


class _SoftwareViewer:
    """Live MuJoCo picture window that needs no OpenGL at all.

    ``mujoco.viewer.launch_passive`` opens a GLFW/GLX window, which needs a
    GPU driver that supports desktop OpenGL — this board's PowerVR part only
    exposes OpenGL ES and has no such driver, so GLFW fails outright ("could
    not initialize GLFW"). MuJoCo's own offscreen ``Renderer`` (OSMesa) was
    tried as a fallback, but this machine's Mesa (20.3.5) hits a known bug
    creating the default OSMesa framebuffer ("Default framebuffer is not
    complete"), independent of GL version overrides.

    So this draws a simple 2D stick-figure side view directly from the
    physics state (body positions, no GL/GLX/EGL/OSMesa involved) and shows
    it in a plain Tk window, which paints via ordinary X11. It's a crude
    picture, but it works regardless of the GPU driver situation.

    Selected automatically when no working GLX display is found, or when
    MUJOCO_GL=osmesa forces it (see _can_create_gl_window/MuJoCoController).
    """

    _BG_COLOR = (24, 26, 32)
    _FLOOR_COLOR = (70, 90, 60)
    _BONE_COLOR = (210, 210, 220)
    _JOINT_COLOR = (255, 190, 60)
    _TRUNK_COLOR = (90, 170, 255)
    _IMU_COLOR = (255, 80, 80)

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        key_callback: Callable[[int], None] | None = None,
        width: int = 640,
        height: int = 480,
        px_per_m: float = 550.0,
        # Redraw every Nth sync() call — the scheduler ticks at 50 Hz, but a
        # plain-Tk picture doesn't need redrawing that often, and skipping
        # most of the PIL render cost keeps sync() cheap enough not to blow
        # the control loop's timing budget. Window events (close/keys) are
        # still pumped on every call so the window stays responsive.
        redraw_every: int = 2,
    ) -> None:
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageTk = ImageTk
        self._model = model
        self._data = data
        self._width = width
        self._height = height
        self._px_per_m = px_per_m
        self._trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.opt = mujoco.MjvOption()
        self._running = True
        self._photo = None
        self._redraw_every = max(1, redraw_every)
        self._sync_count = 0

        self._root = tk.Tk()
        self._root.title("MuJoCo (software render)")
        self._label = tk.Label(self._root, bg="black")
        self._label.pack()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        if key_callback is not None:
            self._root.bind("<KeyPress>", lambda e: self._on_key(e, key_callback))

        self.sync()

    def _on_close(self) -> None:
        self._running = False
        self._root.destroy()

    @staticmethod
    def _on_key(event, key_callback: Callable[[int], None]) -> None:
        arrow_keycodes = {
            "Up": _GLFW_KEY_UP,
            "Down": _GLFW_KEY_DOWN,
            "Right": _GLFW_KEY_RIGHT,
            "Left": _GLFW_KEY_LEFT,
        }
        if event.keysym in arrow_keycodes:
            key_callback(arrow_keycodes[event.keysym])
        elif event.char:
            key_callback(ord(event.char.upper()))

    def is_running(self) -> bool:
        return self._running

    def _project(self, pos, cx: float, cz: float) -> tuple[float, float]:
        """Oblique (cabinet) projection: x -> screen x, z (up) -> screen y

        (flipped), with world y (depth, i.e. left/right) also shifting both
        screen axes a bit. A strict side view collapses the left and right
        limbs onto the same line since they only differ in y — the diagonal
        depth shift pulls them apart into a legible, if crude, 3D-ish figure.
        """
        depth = pos[1] * 0.6 * self._px_per_m
        sx = self._width / 2 + (pos[0] - cx) * self._px_per_m + depth
        sy = self._height / 2 - (pos[2] - cz) * self._px_per_m + depth
        return sx, sy

    @staticmethod
    def _dot(center: tuple[float, float], r: float) -> list[float]:
        x, y = center
        return [x - r, y - r, x + r, y + r]

    def _render_frame(self):
        img = self._Image.new("RGB", (self._width, self._height), self._BG_COLOR)
        draw = self._ImageDraw.Draw(img)

        # Chase camera centered on the trunk so the robot stays in frame.
        trunk_pos = self._data.xpos[self._trunk_id]
        cx, cz = float(trunk_pos[0]), float(trunk_pos[2])

        floor_y = self._project((0.0, 0.0, 0.0), cx, cz)[1]
        draw.line([(0, floor_y), (self._width, floor_y)], fill=self._FLOOR_COLOR, width=2)

        for body_id in range(1, self._model.nbody):
            parent_id = self._model.body_parentid[body_id]
            p0 = self._project(self._data.xpos[parent_id], cx, cz)
            p1 = self._project(self._data.xpos[body_id], cx, cz)
            draw.line([p0, p1], fill=self._BONE_COLOR, width=4)
            draw.ellipse(self._dot(p1, 5), fill=self._JOINT_COLOR)

        draw.ellipse(self._dot(self._project(trunk_pos, cx, cz), 9), fill=self._TRUNK_COLOR)

        if self._imu_site >= 0 and self.opt.frame == mujoco.mjtFrame.mjFRAME_SITE:
            p = self._project(self._data.site_xpos[self._imu_site], cx, cz)
            draw.ellipse(self._dot(p, 7), outline=self._IMU_COLOR, width=2)

        return img

    def sync(self) -> None:
        if not self._running:
            return
        if self._sync_count % self._redraw_every == 0:
            img = self._render_frame()
            self._photo = self._ImageTk.PhotoImage(img)
            self._label.configure(image=self._photo)
        self._sync_count += 1
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._running = False


class _NullViewer:
    """No-op viewer for a headless master that streams state instead of rendering.

    Gives MuJoCoController the same interface it drives on the other two viewers
    (``is_running``, ``opt``, ``sync``) so nothing downstream needs to special-case
    streaming mode. Always reports itself as running: this master keeps stepping
    physics regardless of whether a remote slave viewer is even listening, let alone
    whether its window is open — closing a display-only viewer must never be able to
    stop the physics side. See sim.state_stream and docs/dev/sim_stream.md.
    """

    def __init__(self) -> None:
        self.opt = mujoco.MjvOption()

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


# Spawn pose of the trunk free joint.
#
# NEUTRAL_POSE pitches the hips by -10 deg with the knees and ankles at zero, so
# the legs — and with them the foot soles — are rotated 10 deg relative to the
# trunk. Spawning with the trunk upright therefore leaves the robot toe-up,
# balanced on its heels, and it tips forward before the walk policy can engage.
# Pitching the trunk back by the same angle puts the soles flat on the floor
# (measured flatness: 0.1 mm across all 12 foot collision geoms).
SPAWN_TRUNK_PITCH: float = -NEUTRAL_POSE["left_hip_pitch"]
SPAWN_TRUNK_QUAT: tuple[float, float, float, float] = (
    math.cos(SPAWN_TRUNK_PITCH / 2), 0.0, math.sin(SPAWN_TRUNK_PITCH / 2), 0.0
)
# Trunk height [m] at which the flat soles just touch the floor (0.1701), plus
# 2 mm clearance. Too low and MuJoCo resolves the penetration by ejecting the
# robot on the first step. It settles to ~0.165 under its own weight.
SPAWN_TRUNK_Z: float = 0.1721


class _DelayBuffer:
    """Returns values delayed by n_steps ticks (0 = no delay)."""

    def __init__(self, initial, n_steps: int) -> None:
        size = max(1, n_steps + 1)
        self._buf: deque = deque([initial] * size, maxlen=size)

    def push_and_read(self, value):
        self._buf.appendleft(value)
        return self._buf[-1]

    def fill(self, value) -> None:
        for i in range(len(self._buf)):
            self._buf[i] = value


class MuJoCoController:
    """MuJoCo-backed controller."""

    def __init__(
        self,
        mjcf_path: str,
        key_callback: Callable[[int, int, int, int], None] | None = None,
        stop_flag_path: str = "/tmp/microban_scheduler.stop",
        reset_source: "MuJoCoInputSource | KeyboardInputSource | None" = None,
        # Master/slave display split (see sim.state_stream, docs/dev/sim_stream.md):
        # when set, no local viewer is opened at all — this instance runs headless
        # and broadcasts (qpos, qvel) to a remote slave viewer after every tick
        # instead of rendering. key_callback then goes unused; drive input from the
        # terminal instead (KeyboardInputSource), since there's no window to bind to.
        state_sender: "StateSender | None" = None,
        # Actuation delay (command → motor), in simulator steps (default timestep: 0.005 s)
        delay_act_steps: int = 0,
        # Sensor delays (motor/IMU → observation), in scheduler ticks (default: 0.02 s)
        delay_pos_ticks: int = 0,
        delay_vel_ticks: int = 0,
        delay_gyro_ticks: int = 0,
        delay_quat_ticks: int = 0,
        trunk_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._stop_flag_path = Path(stop_flag_path)
        self._reset_source = reset_source
        self._state_sender = state_sender
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)

        self._name_to_actuator_idx: dict[str, int] = {}
        self._name_to_qpos_idx: dict[str, int] = {}
        self._name_to_qvel_idx: dict[str, int] = {}
        for name in MOTOR_TO_ID:
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in MJCF model {mjcf_path!r}")
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MJCF model {mjcf_path!r}")
            self._name_to_actuator_idx[name] = actuator_id
            self._name_to_qpos_idx[name] = self._model.jnt_qposadr[joint_id]
            self._name_to_qvel_idx[name] = self._model.jnt_dofadr[joint_id]
        # Number of physics sub-steps per scheduler tick (scheduler runs at 50 Hz)
        self._steps_per_tick = max(1, round(0.02 / self._model.opt.timestep))
        self._torque_interval = 0.1
        self._last_torque_print = 0.0

        # Apply CoM offset on trunk body (simulates inertial model error)
        if any(v != 0.0 for v in trunk_com_offset):
            trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            self._model.body_ipos[trunk_id, 0] += trunk_com_offset[0]
            self._model.body_ipos[trunk_id, 1] += trunk_com_offset[1]
            self._model.body_ipos[trunk_id, 2] += trunk_com_offset[2]

        # BAM motor model — XL330 m6 (DC motor + Stribeck + load-dependent friction)
        # Built before the initial pose is applied: the BamController constructor
        # calls mj_setConst, which resets mjData back to qpos0. Setting the pose
        # first would leave the robot at the origin, half-buried in the floor.
        bam_model = bam_load_model(motor_name="xl330", model="m6")
        bam_model.actuator.kp = KP_DEFAULT
        bam_model.actuator.vin = BAM_VIN
        # The firmware current limit lives on the actuator, applied by BAM as a
        # duty-cycle constraint inside compute_control.
        bam_model.actuator.max_current = BAM_MAX_CURRENT
        self._bam = BamController(
            bam_model,
            list(MOTOR_TO_ID.keys()),
            self._model,
            self._data,
            vin_drop_resistance=BAM_VOLTAGE_DROP_GAIN,
            vin_min=BAM_VIN_MIN,
        )

        # Set initial pose to neutral so the robot starts upright
        self._apply_neutral_pose()

        # Delay buffers — simulate sensor/communication latency
        self._delay_pos = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_pos_ticks,
            )
            for mid in MOTOR_TO_ID.values()
        }
        self._delay_vel = {
            mid: _DelayBuffer(0.0, delay_vel_ticks)
            for mid in MOTOR_TO_ID.values()
        }
        self._delay_gyro = _DelayBuffer((0.0, 0.0, 0.0), delay_gyro_ticks)
        self._delay_quat = _DelayBuffer((1.0, 0.0, 0.0, 0.0), delay_quat_ticks)
        self._delay_act = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_act_steps,
            )
            for mid in MOTOR_TO_ID.values()
        }

        self._bam_reset_targets()

        if self._state_sender is not None:
            # Master/slave split: this instance never renders locally at all — a
            # remote slave viewer does, from the state broadcast after every tick.
            self._viewer = _NullViewer()
        else:
            # MUJOCO_GL unset: auto-detect (real 3D viewer wherever GLX actually
            # works — e.g. over `ssh -X` to a real desktop — else the software
            # fallback). MUJOCO_GL=osmesa forces the fallback explicitly; any
            # other explicit value is trusted as-is and skips the probe.
            mujoco_gl = os.environ.get("MUJOCO_GL")
            if mujoco_gl:
                use_software_viewer = mujoco_gl == "osmesa"
            else:
                use_software_viewer = not _can_create_gl_window()

            if use_software_viewer:
                self._viewer = _SoftwareViewer(self._model, self._data, key_callback=key_callback)
            else:
                self._viewer = mujoco.viewer.launch_passive(
                    self._model, self._data, key_callback=key_callback
                )

        # Sensor indices for IMU readout
        self._sensor_orientation = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self._sensor_gyro = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")

    @property
    def viewer_opt(self) -> mujoco.MjvOption:
        return self._viewer.opt

    def set_kp(self, kp: float) -> None:
        self._bam.model.actuator.kp = kp

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        kp = int(self._bam.model.actuator.kp)
        return [kp] * len(ids)

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._bam.model.actuator.kp = gains[0]

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        pass

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        pass

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if not self._viewer.is_running():
            self._stop_flag_path.write_text("stop\n", encoding="ascii")
            return

        cmd = dict(zip(ids, positions))

        for _ in range(self._steps_per_tick):
            for motor_id, pos in cmd.items():
                delayed_pos = self._delay_act[motor_id].push_and_read(pos)
                self._bam.set_q_target(ID_TO_MOTOR[motor_id], delayed_pos)
            self._bam.update()
            mujoco.mj_step(self._model, self._data)

        if self._state_sender is not None:
            self._state_sender.send(self._model, self._data)

        if self._reset_source is not None and self._reset_source.consume_reset():
            self.reset()
            return

        if self._reset_source is not None and self._reset_source.show_torque:
            now = time.monotonic()
            if now - self._last_torque_print >= self._torque_interval:
                total = float(sum(abs(f) for f in self._data.actuator_force))
                print(f"Torque sum: {total:.3f} Nm")
                self._last_torque_print = now

        self._viewer.sync()

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        return [
            self._delay_pos[mid].push_and_read(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_position(self, motor_id: int) -> float:
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_pos[motor_id].push_and_read(
            self._data.qpos[self._name_to_qpos_idx[name]]
        ))

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        return [
            self._delay_vel[mid].push_and_read(
                self._data.qvel[self._name_to_qvel_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_velocity(self, motor_id: int) -> float:
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_vel[motor_id].push_and_read(
            self._data.qvel[self._name_to_qvel_idx[name]]
        ))

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        # Motor current from the torque applied by the bam model: I = torque / kt.
        # ctrl holds the (current-clipped) torque set in MujocoController.update().
        kt = self._bam.model.kt.value
        return [
            float(self._data.ctrl[self._name_to_actuator_idx[ID_TO_MOTOR[mid]]] / kt)
            for mid in ids
        ]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        return [80.0] * len(ids)

    def read_present_input_voltage(self, motor_id: int) -> float:
        return 80.0

    def read_acc(self) -> tuple[float, float, float]:
        """Return pseudo-accelerometer (ax, ay, az) in g from the 'orientation' sensor."""
        if self._sensor_orientation < 0:
            return 0.0, 0.0, -1.0
        adr = self._model.sensor_adr[self._sensor_orientation]
        w, x, y, z = self._data.sensordata[adr:adr + 4]
        # Gravity in world is (0, 0, -1) g; rotate into IMU frame using conjugate quat
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = w * w - x * x - y * y + z * z
        return float(gx), float(gy), float(-gz)

    def read_gyro(self) -> tuple[float, float, float]:
        if self._sensor_gyro < 0:
            current = (0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_gyro]
            gx, gy, gz = self._data.sensordata[adr:adr + 3]
            current = (float(gx), float(gy), float(gz))
        return self._delay_gyro.push_and_read(current)

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        if self._sensor_orientation < 0:
            current = (1.0, 0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_orientation]
            w, x, y, z = self._data.sensordata[adr:adr + 4]
            current = (float(w), float(x), float(y), float(z))
        return self._delay_quat.push_and_read(current)

    def _apply_neutral_pose(self) -> None:
        """Place the robot in the neutral pose, soles flat just above the floor."""
        self._data.qpos[2] = SPAWN_TRUNK_Z
        self._data.qpos[3:7] = SPAWN_TRUNK_QUAT
        for name, angle in NEUTRAL_POSE.items():
            if name in self._name_to_qpos_idx:
                self._data.qpos[self._name_to_qpos_idx[name]] = angle
        mujoco.mj_forward(self._model, self._data)

    def _bam_reset_targets(self) -> None:
        """Seed the BAM targets from the current joint positions.

        BAM initialises its targets to zero, so without this the first update()
        would drive every joint away from the neutral pose.
        """
        self._bam.model.actuator.reset()
        for name, qpos_idx in self._name_to_qpos_idx.items():
            self._bam.set_q_target(name, self._data.qpos[qpos_idx])

    def reset(self) -> None:
        """Reset the simulation to the initial neutral standing pose."""
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        self._apply_neutral_pose()
        self._bam_reset_targets()
        for mid in MOTOR_TO_ID.values():
            neutral = self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            self._delay_act[mid].fill(neutral)
        if self._state_sender is not None:
            self._state_sender.send(self._model, self._data)
        self._viewer.sync()