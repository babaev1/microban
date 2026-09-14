# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Display-only slave viewer for the master/slave MuJoCo split — never deployed to the
robot, never touches motors, and steps no physics of its own.

Meant to run on a machine with real desktop OpenGL (e.g. your laptop), receiving state
from a headless `sim_main.py --stream-to ...` master (e.g. the Orange Pi). See
docs/dev/sim_stream.md for the full picture.

When the master is `hw_state_stream.py` (real robot, not a physics step) instead, its
packets also carry an IMU observation (see state_stream.py's `_IMU_TRAILER`) —
exactly the two channels moves/walk.py's RL policy observes, gyro and gravity
projected into body frame, already computed sender-side the same way observer.py
computes them for the real BMI088. Rendered here as two arrows rooted at a point
above the "imu" site (the best stand-in for "the robot's head" this model currently
has — its actual head body/joint is commented out of robot.xml). Both vectors are
body-frame, drawn without further rotation: the trunk free joint has no real
orientation source in hw_state_stream.py and is always rendered upright (identity),
so body frame and this render's frame coincide — a real-robot lean shows up as these
arrows tilting away from straight-down/zero, even though the mesh itself stays put.
  - red:   projected gravity (unit vector) — points straight down when the robot is
           level; leans with the robot's real tilt otherwise.
  - green: gyro (rotation axis) — near-zero length at rest, longer while turning.
See docs/dev/hw_stream.md — gyro is uncalibrated raw sensor units (scaled and clamped
purely for a legible arrow, not physical units) and both assume the BHI260's mounting
relative to the trunk is identity (ZUBR_IMU_MOUNT_QUAT, unverified placeholder).

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

_ARROW_GRAVITY_RGBA = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)
_ARROW_GYRO_RGBA = np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32)
_ARROW_SHAFT_WIDTH = 0.004
_ARROW_GRAVITY_LENGTH = 0.08  # fixed — projected_gravity is always unit length by construction
# Raw-unit-to-meters scale and a length cap for the gyro arrow: its true LSB scale
# isn't known (see hw_state_stream.py's _imu_from_telemetry), so this is tuned only to
# keep it legible next to the gravity arrow above, not to represent real physical
# units (e.g. "1 arrow-meter = 1 rad/s" is not a claim this makes).
_ARROW_GYRO_GAIN = 1.0 / 4096.0
_ARROW_GYRO_MAX_LENGTH = 0.12


def _set_arrow(scn, origin: np.ndarray, vec: np.ndarray, rgba: np.ndarray, width: float) -> None:
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, origin + vec)
    scn.ngeom += 1


def _draw_imu_arrows(scn, anchor: np.ndarray, imu: tuple) -> None:
    """Populate ``scn`` (a viewer's ``user_scn``) with the gravity/gyro arrows at ``anchor``.

    ``imu`` is ``(gyro, projected_gravity)`` — already body-frame, already the exact
    RL-observation values (see hw_state_stream.py's _imu_from_telemetry). No rotation
    happens here; see the module docstring for why that's still frame-consistent.
    """
    scn.ngeom = 0
    gyro, projected_gravity = imu

    gravity = np.array(projected_gravity, dtype=np.float64)
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm > 1e-9:
        _set_arrow(scn, anchor, gravity / gravity_norm * _ARROW_GRAVITY_LENGTH, _ARROW_GRAVITY_RGBA, _ARROW_SHAFT_WIDTH)

    gyro_vec = np.array(gyro, dtype=np.float64)
    gyro_norm = np.linalg.norm(gyro_vec)
    if gyro_norm > 1e-9:
        gyro_len = min(gyro_norm * _ARROW_GYRO_GAIN, _ARROW_GYRO_MAX_LENGTH)
        _set_arrow(scn, anchor, gyro_vec / gyro_norm * gyro_len, _ARROW_GYRO_RGBA, _ARROW_SHAFT_WIDTH)


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
