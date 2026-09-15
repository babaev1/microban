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

Add --joystick to drive vx/vy/vtheta from the STM32/zubr board's own remote-control
joysticks instead of typed keyboard velocity steps — see
input/zubr_joystick_input.py and docs/dev/hw_stream.md:
    uv run --group sim src/sim/sim_main.py --hz 50 --stream-to 192.168.1.42:9761 --joystick
    make sim-master SLAVE=192.168.1.42:9761 JOYSTICK=1
"""

import argparse

from scheduler import Scheduler
from input.keyboard_input import KeyboardInputSource
from input.zubr_joystick_input import ZubrJoystickInputSource
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
    # Every --delay-* value below is a MAX: each channel samples its lag uniformly
    # from [0, MAX] (mjlab's own delay_min_lag is 0 for every delayed term training
    # uses), resampled continuously rather than held fixed — see mujoco_controller.py's
    # _DelayBuffer and docs/dev/sim_training_parity.md. Defaults match training exactly:
    # gyro/quat sample every tick, resampled every 64 ticks (~1.28 s); vel/pos/act
    # resample every tick/step (mjlab's delay_update_period=0 default).
    parser.add_argument("--delay-act", type=int, default=0, metavar="MAX_STEPS", help="Actuation delay upper bound, in simulator steps (1 step = 0.005 s). Training: 0 (none).")
    parser.add_argument("--delay-pos", type=int, default=0, metavar="MAX_TICKS", help="Motor position read delay upper bound, in scheduler ticks (1 tick = 20 ms at 50 Hz). Training: 0 (none).")
    parser.add_argument("--delay-vel", type=int, default=1, metavar="MAX_TICKS", help="Motor velocity read delay upper bound, in ticks. Training: 1.")
    parser.add_argument("--delay-gyro", type=int, default=3, metavar="MAX_TICKS", help="Gyro read delay upper bound, in ticks, resampled every 64 ticks. Training: 3.")
    parser.add_argument("--delay-quat", type=int, default=3, metavar="MAX_TICKS", help="Quaternion (projected gravity) read delay upper bound, in ticks, resampled every 64 ticks. Training: 3.")
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
    parser.add_argument(
        "--joystick",
        action="store_true",
        help=(
            "Drive vx/vy/vtheta from the STM32/zubr board's remote-control joysticks "
            "(over /dev/ttyS2) instead of typed keyboard velocity steps. Move "
            "toggling/reset/torque-display/stop still come from the keyboard, same "
            "terminal as --stream-to uses; 'walk' is force-enabled regardless, since "
            "there's no known remote button to toggle it with instead. See "
            "input/zubr_joystick_input.py and docs/dev/hw_stream.md."
        ),
    )
    args = parser.parse_args()

    state_sender = None
    if args.stream_to:
        host, port = parse_host_port(args.stream_to, DEFAULT_STREAM_PORT)
        state_sender = StateSender(host, port)

    if args.joystick:
        # No viewer window either way once this is in play — see the module
        # docstring; only meaningful alongside --stream-to in practice, but nothing
        # here actually requires it.
        input_source = ZubrJoystickInputSource(move_keys=MOVE_KEYS)
        key_callback = None
    elif args.stream_to:
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