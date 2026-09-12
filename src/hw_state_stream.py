# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Stream the real robot's actuator positions into a remote MuJoCo viewer.

Runs on the Orange Pi. Reads actuator positions from the STM32 motor controller over
/dev/ttyS2 (zubr_link.py) and rebroadcasts them using the exact same UDP wire format
sim_main.py --stream-to uses (sim/state_stream.py) — so the same
`sim_viewer_client.py` / `make sim-viewer` running on a laptop renders it unmodified,
without knowing whether the state came from real hardware or a physics step.

No physics runs here at all: the MuJoCo model is loaded only to get the (qpos, qvel)
layout right (same joint ordering as scene.xml), never stepped. Every poll leaves all
motors relaxed (see zubr_link.RELAX_POSITION) — this script only ever reads, it never
commands the robot, so it's safe to run at any time, including while another process
drives the motors. See docs/dev/hw_stream.md.

Usage (on the Pi):
    cd ~/microban
    PYTHONPATH=src uv run --group sim src/hw_state_stream.py --stream-to <laptop-ip>:9761
    # or: make hw-stream SLAVE=<laptop-ip>:9761

On the laptop first:
    make sim-viewer PORT=9761
"""

import argparse
import time

import mujoco

from constants import MOTOR_SIGN, MOTOR_TO_ID
from sim.state_stream import DEFAULT_STREAM_PORT, StateSender, parse_host_port
from zubr_link import DEFAULT_BAUDRATE, DEFAULT_PORT, MOTOR_COUNT, ZubrLink, ticks_to_rad

# TODO(calibrate on the real board): physical wiring order of the STM32's 16 motor
# slots (both the goal-position command array and the position/velocity telemetry
# array in zubr_link.py's protocol) — index i here is slot i on the board. This is
# NOT derived from MOTOR_TO_ID: that dict's values are Dynamixel-style bus IDs used by
# a different controller (RobotController/rustypot) and its key order is just
# source-file order, neither of which says anything about STM32 wiring — editing
# MOTOR_TO_ID's numbers has no effect here. Edit THIS list directly, in slot order, to
# calibrate. Placeholder below just lists all 16 MOTOR_TO_ID joints in their dict
# order as a starting guess. To verify: move one joint by hand and see which slot's
# telemetry position changes, then put that joint's name at that index.
SLOT_TO_JOINT: tuple[str, ...] = (
    "right_shoulder_pitch",     # slot 0
    "left_shoulder_pitch",      # slot 1
    "right_shoulder_roll",      # slot 2
    "left_shoulder_roll",       # slot 3
    "right_hip_yaw",            # slot 4
    "left_hip_yaw",             # slot 5
    "right_hip_roll",           # slot 6
    "left_hip_roll",            # slot 7
    "right_hip_pitch",          # slot 8
    "left_hip_pitch",           # slot 9
    "right_knee",               # slot 10
    "left_knee",                # slot 11
    "right_ankle_pitch",        # slot 12
    "left_ankle_pitch",         # slot 13
    "right_ankle_roll",         # slot 14
    "left_ankle_roll",          # slot 15
)
assert len(SLOT_TO_JOINT) == MOTOR_COUNT, (
    f"SLOT_TO_JOINT must have exactly {MOTOR_COUNT} entries (one per STM32 motor "
    f"slot), got {len(SLOT_TO_JOINT)}"
)
assert len(set(SLOT_TO_JOINT)) == MOTOR_COUNT, "SLOT_TO_JOINT has a duplicate joint name"
assert set(SLOT_TO_JOINT) <= set(MOTOR_TO_ID), (
    f"SLOT_TO_JOINT names must all be keys of MOTOR_TO_ID (constants.py); "
    f"unknown: {set(SLOT_TO_JOINT) - set(MOTOR_TO_ID)}"
)

# Spawn pose of the trunk free joint — matches sim/mujoco_controller.py's
# SPAWN_TRUNK_Z/SPAWN_TRUNK_QUAT for NEUTRAL_POSE's current all-zero angles (identity
# orientation). This script has no base position/orientation sensor, so the trunk
# stays fixed here the whole run; only the 16 measured joint angles move. Recompute
# (see mujoco_controller.py's comment) if NEUTRAL_POSE's hip pitch ever becomes nonzero.
_SPAWN_TRUNK_Z = 0.2486
_SPAWN_TRUNK_QUAT = (1.0, 0.0, 0.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stream-to",
        required=True,
        metavar="HOST[:PORT]",
        help=f"sim_viewer_client.py address to stream to (default port {DEFAULT_STREAM_PORT})",
    )
    parser.add_argument("--port", default=DEFAULT_PORT, metavar="DEV", help="STM32 serial port (default: %(default)s)")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="default: %(default)s")
    parser.add_argument(
        "--mjcf-path",
        default="src/model/mjcf/scene.xml",
        metavar="PATH",
        help="Same MJCF the viewer loads — must match exactly (default: %(default)s)",
    )
    parser.add_argument("--hz", type=float, default=50.0, metavar="FREQ", help="Polling/streaming rate (default: %(default)s)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf_path)
    data = mujoco.MjData(model)

    joint_qpos_idx: dict[str, int] = {}
    for name in SLOT_TO_JOINT:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in MJCF model {args.mjcf_path!r}")
        joint_qpos_idx[name] = model.jnt_qposadr[joint_id]

    data.qpos[2] = _SPAWN_TRUNK_Z
    data.qpos[3:7] = _SPAWN_TRUNK_QUAT
    mujoco.mj_forward(model, data)

    host, port = parse_host_port(args.stream_to, DEFAULT_STREAM_PORT)
    sender = StateSender(host, port)
    link = ZubrLink(args.port, baudrate=args.baudrate)

    period = 1.0 / args.hz
    dropped = 0
    sent = 0
    print(f"Streaming real actuator positions from {args.port} to {host}:{port} (Ctrl-C to stop) ...", flush=True)
    try:
        while True:
            tick_start = time.perf_counter()

            telemetry = link.poll()
            if telemetry is None:
                dropped += 1
            else:
                for slot, name in enumerate(SLOT_TO_JOINT):
                    # MOTOR_SIGN is the hardware->sim direction convention the rest of
                    # the codebase already uses for these joint names (RobotController
                    # applies it the same way for the rustypot/Dynamixel bus) — reused
                    # here as the best available guess, not yet reverified against this
                    # STM32/zubr board specifically.
                    angle = MOTOR_SIGN[name] * ticks_to_rad(telemetry.motor_positions[slot])
                    data.qpos[joint_qpos_idx[name]] = angle
                mujoco.mj_forward(model, data)
                sender.send(model, data)
                sent += 1

            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Stopped. Sent {sent} frames, dropped {dropped}.", flush=True)
        sender.close()
        link.close()


if __name__ == "__main__":
    main()
