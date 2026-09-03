# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Local simulation entry point — never deployed to the robot.

Usage:
    uv run --group sim src/sim/sim_main.py --hz 50
    make sim

Master/slave display split (see docs/dev/sim_stream.md): pass --stream-to to run
headless and broadcast physics state to a remote `sim_viewer_client.py` instead of
opening a local viewer window — e.g. physics on the Orange Pi, real 3D rendering on
your laptop's own GPU:
    uv run --group sim src/sim/sim_main.py --hz 50 --stream-to 192.168.1.42:9761
"""

import argparse

from scheduler import Scheduler
from input.keyboard_input import KeyboardInputSource
from sim.mujoco_input import MuJoCoInputSource
from sim.mujoco_controller import MuJoCoController
from sim.state_stream import DEFAULT_STREAM_PORT, StateSender, parse_host_port
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove

MOVE_KEYS = {"h": "head", "s": "squat", "v": "walk"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run microban scheduler in MuJoCo simulation.")
    parser.add_argument("--hz", type=float, default=50.0, metavar="FREQ", help="Scheduler frequency in Hz (default: 50)")
    parser.add_argument("--delay-act", type=int, default=2, metavar="STEPS", help="Actuation delay in simulator steps (1 step = 0.005 s)")
    parser.add_argument("--delay-pos", type=int, default=0, metavar="TICKS", help="Motor position read delay in scheduler ticks (1 tick = 20 ms at 50 Hz)")
    parser.add_argument("--delay-vel", type=int, default=1, metavar="TICKS", help="Motor velocity read delay in ticks")
    parser.add_argument("--delay-gyro", type=int, default=3, metavar="TICKS", help="Gyro read delay in ticks")
    parser.add_argument("--delay-quat", type=int, default=4, metavar="TICKS", help="Quaternion (projected gravity) read delay in ticks")
    parser.add_argument("--trunk-com-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="CoM offset on trunk body in meters (body frame)")
    parser.add_argument(
        "--stream-to",
        metavar="HOST[:PORT]",
        help=(
            f"Run headless and broadcast state to a sim_viewer_client.py listening at "
            f"HOST:PORT (default port {DEFAULT_STREAM_PORT}) instead of opening a local "
            "viewer window. Keyboard control moves to this terminal, since there's no "
            "window left to bind it to."
        ),
    )
    args = parser.parse_args()

    state_sender = None
    if args.stream_to:
        host, port = parse_host_port(args.stream_to, DEFAULT_STREAM_PORT)
        state_sender = StateSender(host, port)
        # No viewer window in this mode, so keyboard input comes from the terminal
        # (raw stdin) rather than from a GLFW/Tk key callback.
        input_source = KeyboardInputSource(move_keys=MOVE_KEYS)
        key_callback = None
    else:
        mujoco_input_source = MuJoCoInputSource(move_keys=MOVE_KEYS)
        input_source = mujoco_input_source
        key_callback = mujoco_input_source.key_callback

    controller = MuJoCoController(
        mjcf_path="src/model/mjcf/scene.xml",
        key_callback=key_callback,
        reset_source=input_source,
        state_sender=state_sender,
        delay_act_steps=args.delay_act,
        delay_pos_ticks=args.delay_pos,
        delay_vel_ticks=args.delay_vel,
        delay_gyro_ticks=args.delay_gyro,
        delay_quat_ticks=args.delay_quat,
        trunk_com_offset=tuple(args.trunk_com_offset),
    )
    if isinstance(input_source, MuJoCoInputSource):
        input_source.set_viewer_opt(controller.viewer_opt)

    scheduler = Scheduler(
        frequency_hz=args.hz,
        controller=controller,
        input_source=input_source,
        moves={
            "head": RotateHeadMove(),
            "squat": SquatMove(),
            "walk": WalkMove(controller=controller),
        },
    )
    scheduler.run()


if __name__ == "__main__":
    main()