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
import math
import time

import mujoco
import numpy as np

from constants import MOTOR_SIGN, MOTOR_TO_ID, MOTOR_ZERO_TICKS, ZUBR_IMU_MOUNT_QUAT
from imu_reader import imu_quat_to_body, quat_apply_inverse
from sim.state_stream import DEFAULT_STREAM_PORT, StateSender, parse_host_port
from zubr_link import DEFAULT_BAUDRATE, DEFAULT_PORT, ZubrLink, ticks_to_rad
from zubr_motor_map import build_slot_map

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

    Gyro is reported by the chip in its own local sensor axes, same as accelerometer —
    it needs the *same* fixed mounting correction as the quaternion before it means
    anything in body-frame terms, just applied directly to the raw vector rather than
    composed into an orientation: since the sensor is rigidly co-moving with the
    trunk, v_body = R(mount) @ v_sensor is a single fixed rotation, not the
    time-varying body_quat gravity needs (that one's projecting a *world*-frame
    reference through the current orientation; this is converting one *body-fixed*
    frame's coordinates into another's, which stays constant regardless of how the
    whole assembly is currently oriented) — verified against a synthetic sensor/body
    pair: quat_apply_inverse(conjugate(mount), v_sensor) recovers the known true
    body-frame vector exactly; the non-conjugated form does not.

    TODO(calibrate on the real board): gyro stays in the STM32's raw register units,
    unconverted (only its *direction* is corrected here, not its scale) — no LSB
    scale factor for it is documented anywhere this project has access to (zubr.py
    just says "int16 - imu.gyro.x", no units). The quaternion's own unknown LSB scale
    is sidestepped by normalizing it before use, valid for anything that's purely a
    rotation (true for both uses here).
    """
    qx, qy, qz, qw = (float(v) for v in telemetry.quat_raw)
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    quat_wxyz = (1.0, 0.0, 0.0, 0.0) if norm < 1e-9 else tuple(quat / norm)

    body_quat = imu_quat_to_body(quat_wxyz, mount_quat=ZUBR_IMU_MOUNT_QUAT)
    projected_gravity = tuple(float(v) for v in quat_apply_inverse(list(body_quat), [0.0, 0.0, -1.0]))

    mount_conj = _qconj(np.array(ZUBR_IMU_MOUNT_QUAT))
    gyro_raw = [float(v) for v in telemetry.gyro_raw]
    gyro_body = tuple(float(v) for v in quat_apply_inverse(list(mount_conj), gyro_raw))

    return gyro_body, projected_gravity


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _average_quat(quats: list[np.ndarray]) -> np.ndarray:
    """Hemisphere-align (quaternions are double-covered: q and -q are the same
    rotation, so naive averaging across a sign flip would cancel instead of average)
    then average and renormalize."""
    aligned = [q if not quats or np.dot(q, quats[0]) >= 0.0 else -q for q in quats]
    mean = np.mean(aligned, axis=0)
    return mean / np.linalg.norm(mean)


def solve_mount(q_level: np.ndarray, q_forward: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Derive ZUBR_IMU_MOUNT_QUAT from two captured raw quaternions (both w,x,y,z,
    already normalized): one with the robot standing level, one with it pitched
    forward (front/chest down) from that same level pose. Returns (mount, warning).

    q_imu = q_body * mount (see constants.py's ZUBR_IMU_MOUNT_QUAT comment for the
    fuller derivation), and q_body_level == identity by definition, so mount == q_level
    exactly — that alone reproduces the old, yaw-blind single-pose calibration (correct
    "down" at rest, but the tilt *direction* comes out wrong by whatever heading the
    robot happened to face during capture — gravity alone can never observe yaw,
    since rotating a vector about the axis it's already aligned with is the identity).

    What the second pose adds: q_forward * conjugate(q_level) equals q_body_forward
    directly (q_body_level cancels to identity), i.e. exactly the pitch that was
    physically applied — independent of whatever arbitrary heading q_level was
    captured at. Its rotation axis reveals how much extra yaw is needed to line
    "forward pitch" up with the model's own lateral (Y) axis, which is the one piece
    a level-only capture can never supply.
    """
    delta = _qmul(q_forward, _qconj(q_level))
    axis = delta[1:4]
    horiz = axis[:2]  # the axis's component within the horizontal (X, Y) plane
    horiz_norm = float(np.linalg.norm(horiz))
    axis_norm = float(np.linalg.norm(axis))

    if horiz_norm < 1e-4:
        return q_level, (
            "the forward-tilt capture barely rotated the sensor (or wasn't a clean "
            "pitch) — yaw can't be determined from it. Re-run with a larger, more "
            "deliberate forward tilt. Falling back to the level-only result (yaw "
            "uncorrected, same limitation as before)."
        )

    warning = None
    if axis_norm > 1e-9 and abs(axis[2]) / axis_norm > 0.3:
        warning = (
            f"step 2's rotation axis has a large out-of-plane component "
            f"({abs(axis[2]) / axis_norm:.0%}) — was it a clean forward pitch, with no "
            "roll or turning mixed in? Proceeding anyway; re-run if the result still "
            "looks wrong."
        )

    # Target: this axis should point along the model's own +Y once yaw is correctly
    # aligned (matches how this codebase signs its pitch joints). The sign of that
    # convention doesn't actually matter for correctness — a consistently flipped
    # axis and rotation angle cancel out in the final gravity direction — only
    # alignment to the Y *line* does.
    current_angle = float(np.arctan2(horiz[1], horiz[0]))
    target_angle = float(np.arctan2(1.0, 0.0))  # angle of +Y
    yaw_correction = target_angle - current_angle
    yaw_quat = np.array([np.cos(yaw_correction / 2), 0.0, 0.0, np.sin(yaw_correction / 2)])

    mount = _qmul(yaw_quat, q_level)
    return mount / np.linalg.norm(mount), warning


def calibrate_mount(link: ZubrLink, n_samples: int) -> None:
    """Capture ZUBR_IMU_MOUNT_QUAT from the real board (level + forward-tilt poses,
    see solve_mount()) and print it, then return."""

    def capture(prompt: str) -> np.ndarray:
        print(f"{prompt} Capturing {n_samples} samples...", flush=True)
        quats: list[np.ndarray] = []
        dropped = 0
        while len(quats) < n_samples:
            telemetry = link.poll()
            if telemetry is None:
                dropped += 1
                continue
            qx, qy, qz, qw = (float(v) for v in telemetry.quat_raw)
            q = np.array([qw, qx, qy, qz], dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm < 1e-9:
                continue
            quats.append(q / norm)
        if dropped:
            print(f"({dropped} dropped/corrupt frames along the way.)", flush=True)
        return _average_quat(quats)

    print("Step 1/2: stand the robot level and still (any heading is fine).", flush=True)
    q_level = capture("Ready —")

    print(
        "Step 2/2: now tip the robot forward (front/chest down) by a clear, "
        "deliberate amount — pitch only, no roll or turning — and hold it steady.",
        flush=True,
    )
    input("Press Enter once it's tilted and held steady... ")
    q_forward = capture("Ready —")

    mount, warning = solve_mount(q_level, q_forward)
    if warning:
        print(f"Warning: {warning}", flush=True)

    w, x, y, z = (float(v) for v in mount)
    print("Paste this into constants.py, replacing the current ZUBR_IMU_MOUNT_QUAT line:", flush=True)
    print(f"ZUBR_IMU_MOUNT_QUAT: tuple[float, float, float, float] = ({w!r}, {x!r}, {y!r}, {z!r})", flush=True)


def calibrate_zero(link: ZubrLink, n_samples: int) -> None:
    """Capture MOTOR_ZERO_TICKS from the real board — the robot must be held/standing
    in its true NEUTRAL_POSE (all joint angles as defined in constants.py, currently
    all 0) while this runs. Averages n_samples raw ticks per resolved slot and prints
    a dict literal to paste into constants.py, then returns."""
    slot_to_joint = build_slot_map(MOTOR_TO_ID)
    print(
        f"Hold the robot in its true NEUTRAL_POSE (standing, all joints at their "
        f"designed zero) and keep it still. Capturing {n_samples} samples...",
        flush=True,
    )
    sums = {slot: 0 for slot in slot_to_joint}
    got = 0
    dropped = 0
    while got < n_samples:
        telemetry = link.poll()
        if telemetry is None:
            dropped += 1
            continue
        for slot in slot_to_joint:
            sums[slot] += telemetry.motor_positions[slot]
        got += 1
    if dropped:
        print(f"({dropped} dropped/corrupt frames along the way.)", flush=True)

    print("Paste this into constants.py, replacing the current MOTOR_ZERO_TICKS line:", flush=True)
    print("MOTOR_ZERO_TICKS: dict[str, int] = {", flush=True)
    for slot in sorted(slot_to_joint):
        name = slot_to_joint[slot]
        print(f"    {name!r}: {round(sums[slot] / n_samples)},", flush=True)
    print("}", flush=True)


def calibrate_gyro_scale(link: ZubrLink, duration_s: float) -> None:
    """Derive a raw-gyro-count -> rad/s scale factor from a timed physical rotation.

    Protocol: you rotate the robot's trunk continuously about ONE axis, as steadily
    as you can, for exactly `duration_s` seconds (the script times this itself —
    accurate elapsed time is what actually matters, not how precisely you hit the
    duration), then report the total rotation you completed in degrees. The scale
    factor is derived only from whichever axis shows the largest average raw
    reading (the one you actually rotated about) and assumed to apply equally to
    all three — standard for MEMS gyro chips, which use one shared full-scale-range
    setting, not a separately-tuned sensitivity per axis.

    This is a single, rough measurement — repeat it a few times (ideally at
    different speeds) and compare/average before trusting the result.
    """
    input(
        f"Get ready to rotate the robot's trunk continuously about ONE axis (e.g. spin "
        f"it around its vertical/yaw axis) as steadily as you can, for about "
        f"{duration_s:.0f}s. Press Enter to start the capture, then begin rotating "
        f"immediately... "
    )

    samples: list[tuple[float, float, float]] = []
    dropped = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        telemetry = link.poll()
        if telemetry is None:
            dropped += 1
            continue
        samples.append(tuple(float(v) for v in telemetry.gyro_raw))
    elapsed = time.perf_counter() - start

    if not samples:
        print("No samples captured — check the link and try again.", flush=True)
        return
    if dropped:
        print(f"({dropped} dropped/corrupt frames along the way.)", flush=True)

    avg = [sum(s[i] for s in samples) / len(samples) for i in range(3)]
    dominant_axis = max(range(3), key=lambda i: abs(avg[i]))
    axis_name = ("x", "y", "z")[dominant_axis]
    print(f"Captured {len(samples)} samples over {elapsed:.2f}s.", flush=True)
    print(
        f"Average raw gyro: x={avg[0]:+.2f} y={avg[1]:+.2f} z={avg[2]:+.2f} "
        f"(dominant axis: {axis_name}, avg={avg[dominant_axis]:+.2f})",
        flush=True,
    )

    answer = input(
        "How much total rotation did you complete during that capture, in degrees? "
        "(e.g. '360' for one full turn, '1080' for three turns): "
    )
    try:
        total_degrees = float(answer)
    except ValueError:
        print("Not a number — aborting without computing a scale.", flush=True)
        return

    if abs(avg[dominant_axis]) < 1e-6:
        print(
            "Dominant-axis average is ~0 — this capture doesn't look like it caught "
            "real rotation. Try again with a faster/cleaner spin.",
            flush=True,
        )
        return

    true_omega = math.radians(total_degrees) / elapsed
    scale = abs(true_omega / avg[dominant_axis])

    print(
        f"\nEstimated scale: {scale:.6e} rad/s per raw count (on the {axis_name} axis, "
        f"assumed the same for all three axes).",
        flush=True,
    )
    print(
        "Paste this into constants.py, replacing the current ZUBR_GYRO_SCALE line:",
        flush=True,
    )
    print(f"ZUBR_GYRO_SCALE: float = {scale!r}", flush=True)
    print(
        "This is a single rough measurement — repeat a few times (ideally at "
        "different rotation speeds) and compare before trusting it for a real walk.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stream-to",
        metavar="HOST[:PORT]",
        help=f"sim_viewer_client.py address to stream to (default port {DEFAULT_STREAM_PORT}). Required unless --calibrate-mount/--calibrate-zero is given.",
    )
    parser.add_argument(
        "--calibrate-mount",
        type=int,
        metavar="N",
        help=(
            "Instead of streaming: capture N onboard-IMU quaternion samples while the "
            "robot stands level and still, print the resulting ZUBR_IMU_MOUNT_QUAT to "
            "paste into constants.py, then exit. See docs/dev/hw_stream.md."
        ),
    )
    parser.add_argument(
        "--calibrate-zero",
        type=int,
        metavar="N",
        help=(
            "Instead of streaming: capture N raw-position samples per joint while the "
            "robot is held in its true NEUTRAL_POSE, print the resulting "
            "MOTOR_ZERO_TICKS to paste into constants.py, then exit. See "
            "docs/dev/hw_stream.md."
        ),
    )
    parser.add_argument(
        "--calibrate-gyro-scale",
        type=float,
        metavar="SECONDS",
        help=(
            "Instead of streaming: capture raw gyro for SECONDS while you physically "
            "rotate the robot's trunk, then prompt for the total rotation you "
            "completed (in degrees) and print the resulting ZUBR_GYRO_SCALE to paste "
            "into constants.py, then exit. See docs/dev/zubr_real_hardware.md."
        ),
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

    if args.calibrate_mount is not None:
        link = ZubrLink(args.port, baudrate=args.baudrate)
        try:
            calibrate_mount(link, args.calibrate_mount)
        finally:
            link.close()
        return
    if args.calibrate_zero is not None:
        link = ZubrLink(args.port, baudrate=args.baudrate)
        try:
            calibrate_zero(link, args.calibrate_zero)
        finally:
            link.close()
        return
    if args.calibrate_gyro_scale is not None:
        link = ZubrLink(args.port, baudrate=args.baudrate)
        try:
            calibrate_gyro_scale(link, args.calibrate_gyro_scale)
        finally:
            link.close()
        return
    if args.stream_to is None:
        parser.error("--stream-to is required unless --calibrate-mount/--calibrate-zero is given")

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
                    zero = MOTOR_ZERO_TICKS.get(name, 0)
                    angle = MOTOR_SIGN[name] * ticks_to_rad(telemetry.motor_positions[slot] - zero)
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
