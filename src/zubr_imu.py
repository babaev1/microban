# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Standalone IMU reader for the STM32/zubr board's onboard BHI260 — prints roll,
pitch, yaw and gyroscope data at 2 Hz, mirroring imu.py's output but for this
different physical sensor (imu.py is the separate I2C BMI088; see
docs/dev/hw_stream.md for how the two relate).

Every poll leaves all motors relaxed (see zubr_link.RELAX_POSITION) — this script
only ever reads, it never commands the robot.

Usage:
    uv run src/zubr_imu.py
    make zubr-imu
"""

import math
import time

import numpy as np

from constants import ZUBR_IMU_MOUNT_QUAT
from imu_reader import imu_quat_to_body, quat_apply_inverse
from zubr_link import ZubrLink


def _normalized_quat(quat_raw: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """quat_raw is (x, y, z, w); returns a normalized (w, x, y, z) — see
    hw_state_stream.py's _imu_from_telemetry for why normalizing sidesteps the
    unknown raw LSB scale for anything used purely as a rotation."""
    qx, qy, qz, qw = (float(v) for v in quat_raw)
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in (quat / norm))


def main() -> None:
    link = ZubrLink()
    print("IMU (STM32/zubr onboard BHI260) — Ctrl-C to stop.")
    try:
        while True:
            telemetry = link.poll()
            if telemetry is None:
                print("(dropped/corrupt frame)")
                time.sleep(0.5)
                continue

            gx, gy, gz = (float(v) for v in telemetry.gyro_raw)
            ax, ay, az = (float(v) for v in telemetry.acc_raw)
            w, x, y, z = _normalized_quat(telemetry.quat_raw)
            bw, bx, by, bz = imu_quat_to_body((w, x, y, z), mount_quat=ZUBR_IMU_MOUNT_QUAT)
            px, py, pz = quat_apply_inverse((bw, bx, by, bz), (0.0, 0.0, -1.0))

            roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

            b_roll = math.degrees(math.atan2(2 * (bw * bx + by * bz), 1 - 2 * (bx * bx + by * by)))
            b_pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (bw * by - bz * bx)))))
            b_yaw = math.degrees(math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz)))

            print("--------------------------------------------")
            print(f"Gyro (raw, uncalibrated units): gx={gx:+.1f}  gy={gy:+.1f}  gz={gz:+.1f}")
            print(f"Acc  (raw, uncalibrated units): ax={ax:+.1f}  ay={ay:+.1f}  az={az:+.1f}")
            print(f"IMU:  roll={roll:+.1f}°  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°")
            print(f"Body: roll={b_roll:+.1f}°  pitch={b_pitch:+.1f}°  yaw={b_yaw:+.1f}°")
            print(f"Projected Gravity: px={px:+.3f}  py={py:+.3f}  pz={pz:+.3f}")

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


if __name__ == "__main__":
    main()
