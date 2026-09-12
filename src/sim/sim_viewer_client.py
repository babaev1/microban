# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Display-only slave viewer for the master/slave MuJoCo split — never deployed to the
robot, never touches motors, and steps no physics of its own.

Meant to run on a machine with real desktop OpenGL (e.g. your laptop), receiving state
from a headless `sim_main.py --stream-to ...` master (e.g. the Orange Pi). See
docs/dev/sim_stream.md for the full picture.

When the master is `hw_state_stream.py` (real robot, not a physics step) instead, its
packets also carry a raw IMU reading (see state_stream.py's `_IMU_TRAILER`) — rendered
here as three arrows rooted at a point 10cm above the "imu" site (the best stand-in
for "the robot's head" this model currently has — its actual head body/joint is
commented out of robot.xml), all in world frame:
  - blue:  the IMU's own reported "up" (unit length) — visualizes orientation directly.
  - red:   the raw accelerometer vector, rotated into world frame by that same
           orientation — should point straight down (aligned with the blue arrow's
           opposite, i.e. gravity) when the robot is still and the fusion is healthy;
           divergence between them is a visible sanity signal, not just decoration.
  - green: the raw gyro vector (rotation axis), likewise rotated into world frame —
           near-zero length at rest, longer while turning.
See docs/dev/hw_stream.md — magnitudes are uncalibrated raw sensor units (scaled and
clamped purely for a legible arrow, not physical units), and orientation is taken
as-is with no mounting correction; only directions carry real information.

Usage:
    uv run --group sim src/sim/sim_viewer_client.py --listen 0.0.0.0:9761
    make sim-viewer PORT=9761
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim.state_stream import DEFAULT_STREAM_PORT, StateReceiver, parse_host_port

# How far above the head-proxy site the arrows are rooted, in meters (world Z, i.e.
# straight up regardless of the robot's own tilt).
_ARROW_ANCHOR_HEIGHT = 0.20

_ARROW_ORIENTATION_RGBA = np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32)
_ARROW_ACCEL_RGBA = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)
_ARROW_GYRO_RGBA = np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32)
_ARROW_SHAFT_WIDTH = 0.004
_ARROW_ORIENTATION_LENGTH = 0.08  # fixed — it's a pure direction, magnitude is meaningless
# Raw-unit-to-meters scale and a length cap for the accel/gyro arrows: their true LSB
# scale isn't known (see hw_state_stream.py's _imu_from_telemetry), so this is tuned
# only to keep the arrows legible next to the orientation arrow above, not to represent
# real physical units (e.g. "1 arrow-meter = 1 g" is not a claim this makes).
_ARROW_VECTOR_GAIN = 1.0 / 4096.0
_ARROW_VECTOR_MAX_LENGTH = 0.12


def _quat_rotate(quat_wxyz: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    """Rotate world/body vector ``v`` by unit quaternion ``quat_wxyz`` (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _set_arrow(scn, origin: np.ndarray, vec: np.ndarray, rgba: np.ndarray, width: float) -> None:
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, origin + vec)
    scn.ngeom += 1


def _draw_imu_arrows(scn, anchor: np.ndarray, imu: tuple) -> None:
    """Populate ``scn`` (a viewer's ``user_scn``) with the three IMU arrows at ``anchor``.

    Quaternion is normalized here (not assumed unit-length): the raw reading's true
    LSB scale is unknown (see hw_state_stream.py), but normalizing sidesteps that
    entirely for anything used purely as a rotation, which every use here is.
    """
    scn.ngeom = 0
    gyro_raw, accel_raw, quat_raw = imu
    quat = np.array(quat_raw, dtype=np.float64)
    norm = np.linalg.norm(quat)
    quat = (1.0, 0.0, 0.0, 0.0) if norm < 1e-9 else tuple(quat / norm)

    up_world = _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
    _set_arrow(scn, anchor, up_world * _ARROW_ORIENTATION_LENGTH, _ARROW_ORIENTATION_RGBA, _ARROW_SHAFT_WIDTH)

    accel_world = _quat_rotate(quat, np.array(accel_raw, dtype=np.float64))
    accel_len = min(np.linalg.norm(accel_world) * _ARROW_VECTOR_GAIN, _ARROW_VECTOR_MAX_LENGTH)
    if np.linalg.norm(accel_world) > 1e-9:
        _set_arrow(scn, anchor, accel_world / np.linalg.norm(accel_world) * accel_len, _ARROW_ACCEL_RGBA, _ARROW_SHAFT_WIDTH)

    gyro_world = _quat_rotate(quat, np.array(gyro_raw, dtype=np.float64))
    gyro_len = min(np.linalg.norm(gyro_world) * _ARROW_VECTOR_GAIN, _ARROW_VECTOR_MAX_LENGTH)
    if np.linalg.norm(gyro_world) > 1e-9:
        _set_arrow(scn, anchor, gyro_world / np.linalg.norm(gyro_world) * gyro_len, _ARROW_GYRO_RGBA, _ARROW_SHAFT_WIDTH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display-only MuJoCo viewer for a remote state stream.")
    parser.add_argument(
        "--listen",
        default=f"0.0.0.0:{DEFAULT_STREAM_PORT}",
        metavar="HOST[:PORT]",
        help=f"Address to receive state on (default: 0.0.0.0:{DEFAULT_STREAM_PORT})",
    )
    parser.add_argument(
        "--mjcf-path",
        default="src/model/mjcf/scene.xml",
        metavar="PATH",
        help="Same MJCF the master loaded — must match exactly (default: %(default)s)",
    )
    args = parser.parse_args()

    host, port = parse_host_port(args.listen, DEFAULT_STREAM_PORT)

    model = mujoco.MjModel.from_xml_path(args.mjcf_path)
    data = mujoco.MjData(model)
    receiver = StateReceiver(host, port)

    # Best available stand-in for "the robot's head" — see the module docstring: the
    # actual head body/joint is commented out of robot.xml, so there's nothing closer.
    imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

    print(f"Listening for state on {host}:{port} — waiting for the master...", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        got_first_packet = False
        while viewer.is_running():
            if receiver.poll(data):
                if not got_first_packet:
                    print("Receiving state from master.", flush=True)
                    got_first_packet = True
                # Recompute everything the renderer needs (body poses, contacts,
                # sensors) from the received qpos/qvel — this side never integrates.
                mujoco.mj_forward(model, data)
                if receiver.last_imu is not None and imu_site >= 0:
                    anchor = data.site_xpos[imu_site] + np.array([0.0, 0.0, _ARROW_ANCHOR_HEIGHT])
                    _draw_imu_arrows(viewer.user_scn, anchor, receiver.last_imu)
                else:
                    # No IMU in this packet (e.g. sim_main.py's physics stream) — drop
                    # any arrows left over from a previous hw_state_stream.py session
                    # rather than freezing them on screen.
                    viewer.user_scn.ngeom = 0
            viewer.sync()
            # Nothing to do between packets besides keep the window responsive.
            time.sleep(0.001)


if __name__ == "__main__":
    main()
