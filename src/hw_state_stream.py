# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Stream the real robot's actuator positions and IMU observation into a remote MuJoCo viewer.

Runs on the Orange Pi. Reads actuator positions and the STM32's onboard BHI260
gyro/quaternion from the motor controller over /dev/ttyS2 (zubr_link.py), reduces the
IMU reading to exactly the two channels moves/walk.py's RL policy actually observes —
gyro and gravity projected into body frame — using the *same* math observer.py uses
for the real BMI088 (imu_reader.imu_quat_to_body then quat_apply_inverse), and
rebroadcasts everything using the same UDP wire format sim_main.py --stream-to uses
(sim/state_stream.py), plus an IMU trailer that format supports — so the same
`sim_viewer_client.py` / `make sim-viewer` running on a laptop renders both the pose
and (as two arrows above the robot's head) that observation, without knowing whether
any of it came from real hardware or a physics step.

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
import numpy as np

from constants import MOTOR_SIGN, MOTOR_TO_ID, ZUBR_IMU_MOUNT_QUAT
from imu_reader import imu_quat_to_body, quat_apply_inverse
from sim.state_stream import DEFAULT_STREAM_PORT, StateSender, parse_host_port
from zubr_link import DEFAULT_BAUDRATE, DEFAULT_PORT, MOTOR_COUNT, ZubrLink, ticks_to_rad

# STM32 slot order (which of the 16 telemetry/command array slots is which joint) is
# read from MOTOR_TO_ID's values directly (constants.py) — RobotController's rustypot
# bus is dead code on this robot now, so those numbers are free to mean "STM32 slot
# index" instead of a Dynamixel bus ID. This is calibrated by moving one joint by hand
# and editing MOTOR_TO_ID[name] to whichever slot's telemetry position changed — that
# one dict is now the single place to edit, nothing in this file needs touching.
#
# Calibration is expected to be incremental: build_slot_map() below tolerates joints
# whose MOTOR_TO_ID value isn't a valid, unique slot yet (prints a warning and leaves
# that joint out of the stream — its qpos just stays 0 — rather than refusing to run).
def build_slot_map(motor_to_id: dict[str, int]) -> dict[int, str]:
    slot_to_joint: dict[int, str] = {}
    claimed_by: dict[int, str] = {}
    for name, slot in motor_to_id.items():
        if not (0 <= slot < MOTOR_COUNT):
            print(f"hw_state_stream: {name!r} has MOTOR_TO_ID value {slot} — out of STM32 slot range [0, {MOTOR_COUNT}); not streamed until recalibrated.", flush=True)
            continue
        if slot in claimed_by:
            print(f"hw_state_stream: {name!r} and {claimed_by[slot]!r} both claim slot {slot} in MOTOR_TO_ID — neither is streamed until this is resolved.", flush=True)
            slot_to_joint.pop(slot, None)
            continue
        claimed_by[slot] = name
        slot_to_joint[slot] = name
    missing = MOTOR_COUNT - len(slot_to_joint)
    if missing:
        print(f"hw_state_stream: {missing} of {MOTOR_COUNT} STM32 slots have no (valid, unique) joint in MOTOR_TO_ID yet — those joints will stay at 0 in the viewer.", flush=True)
    return slot_to_joint

# Spawn pose of the trunk free joint — matches sim/mujoco_controller.py's
# SPAWN_TRUNK_Z/SPAWN_TRUNK_QUAT for NEUTRAL_POSE's current all-zero angles (identity
# orientation). This script has no base position/orientation sensor, so the trunk
# stays fixed here the whole run; only the 16 measured joint angles move. Recompute
# (see mujoco_controller.py's comment) if NEUTRAL_POSE's hip pitch ever becomes nonzero.
_SPAWN_TRUNK_Z = 0.2486
_SPAWN_TRUNK_QUAT = (1.0, 0.0, 0.0, 0.0)


def _imu_from_telemetry(telemetry) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build the (gyro, projected_gravity) trailer state_stream.StateSender.send() expects.

    This reads gyro and quaternion off the STM32's onboard BHI260 (already a fused
    orientation estimate — chip does its own sensor fusion, no Madgwick step needed
    here) and reduces them to exactly the two channels moves/walk.py's RL policy
    observes, the *same way* observer.py.read_state() derives them from the real
    BMI088: rotate the quaternion into body frame (imu_quat_to_body), then project
    world gravity through it (quat_apply_inverse). Raw accelerometer is read by
    zubr_link.py but not used here — the RL pipeline this mirrors never uses it either
    (see moves/walk.py's build_observation()), only the gravity direction the fused
    quaternion implies.

    TODO(calibrate on the real board):
    - Gyro stays in the STM32's raw register units, unconverted — no LSB scale factor
      for it is documented anywhere this project has access to (zubr.py just says
      "int16 - imu.gyro.x", no units).
    - ZUBR_IMU_MOUNT_QUAT (constants.py) — this BHI260's mounting rotation relative to
      the trunk — is an unverified identity placeholder, unlike IMU_MOUNT_QUAT (the
      *separate* I2C BMI088 imu_reader.py reads, already reverified against real
      mounting). The quaternion itself is normalized before use, which sidesteps its
      own unknown LSB scale for any purpose that's purely a rotation (true here).
    """
    gyro = tuple(float(v) for v in telemetry.gyro_raw)

    qx, qy, qz, qw = (float(v) for v in telemetry.quat_raw)
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    quat_wxyz = (1.0, 0.0, 0.0, 0.0) if norm < 1e-9 else tuple(quat / norm)

    body_quat = imu_quat_to_body(quat_wxyz, mount_quat=ZUBR_IMU_MOUNT_QUAT)
    projected_gravity = tuple(float(v) for v in quat_apply_inverse(list(body_quat), [0.0, 0.0, -1.0]))

    return gyro, projected_gravity


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

    slot_to_joint = build_slot_map(MOTOR_TO_ID)
    joint_qpos_idx: dict[str, int] = {}
    for name in slot_to_joint.values():
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
                for slot, name in slot_to_joint.items():
                    # MOTOR_SIGN is the hardware->sim direction convention the rest of
                    # the codebase already uses for these joint names (RobotController
                    # applies it the same way for the rustypot/Dynamixel bus) — reused
                    # here as the best available guess, not yet reverified against this
                    # STM32/zubr board specifically.
                    angle = MOTOR_SIGN[name] * ticks_to_rad(telemetry.motor_positions[slot])
                    data.qpos[joint_qpos_idx[name]] = angle
                mujoco.mj_forward(model, data)
                sender.send(model, data, imu=_imu_from_telemetry(telemetry))
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
