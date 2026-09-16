# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Read and print the bus voltage from one or all motors.

STATUS: likely non-functional on the current Roki4 hardware. This talks directly to
the old rustypot/XL330 Dynamixel bus on /dev/ttyAMA0 — the same bus RobotController
used, confirmed dead code now that the robot runs on the STM32/zubr board instead
(see docs/dev/hw_stream.md). Two separate problems, either of which breaks this:
  1. /dev/ttyAMA0 may no longer be connected/powered at all on the current hardware.
  2. Even if it is, MOTOR_TO_ID's values have been repurposed to mean STM32 slot
     indices (0-15), not Dynamixel bus IDs — so motor_ids below would address the
     wrong (or nonexistent) servos even if the bus itself still respected them.
The zubr protocol's telemetry has no per-motor voltage field at all (see
zubr_link.Telemetry) — there is currently no way to build a working equivalent for
the STM32/zubr board without more hardware documentation (e.g. whether pack voltage
is readable through the protocol's 2 generic named-variable read slots, and which
index). Left as-is rather than silently producing wrong numbers.
"""

import sys
from rustypot import Xl330PyController

from constants import MOTOR_TO_ID, ID_TO_MOTOR


def main() -> None:
    controller = Xl330PyController(
        serial_port="/dev/ttyAMA0", baudrate=1_000_000, timeout=0.1
    )

    if len(sys.argv) > 1:
        motor_ids = [int(sys.argv[1])]
    else:
        motor_ids = list(MOTOR_TO_ID.values())

    controller.sync_write_status_return_level(motor_ids, [1] * len(motor_ids))

    for motor_id in motor_ids:
        raw = controller.read_present_input_voltage(motor_id)
        voltage_raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        voltage_v = voltage_raw * 0.1
        name = ID_TO_MOTOR.get(motor_id, str(motor_id))
        print(f"  Motor {motor_id:2d}: {voltage_v:.2f} V ({name})")


if __name__ == "__main__":
    main()