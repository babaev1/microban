# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Display-only slave viewer for the master/slave MuJoCo split — never deployed to the
robot, never touches motors, and steps no physics of its own.

Meant to run on a machine with real desktop OpenGL (e.g. your laptop), receiving state
from a headless `sim_main.py --stream-to ...` master (e.g. the Orange Pi). See
docs/dev/sim_stream.md for the full picture.

Usage:
    uv run --group sim src/sim/sim_viewer_client.py --listen 0.0.0.0:9761
    make sim-viewer PORT=9761
"""

import argparse
import time

import mujoco
import mujoco.viewer

from sim.state_stream import DEFAULT_STREAM_PORT, StateReceiver, parse_host_port


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
            viewer.sync()
            # Nothing to do between packets besides keep the window responsive.
            time.sleep(0.001)


if __name__ == "__main__":
    main()
