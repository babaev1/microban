# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Maps STM32/zubr motor slots (0-15, the position in the telemetry/command arrays —
see zubr_link.py) to joint names, from constants.MOTOR_TO_ID's values.

RobotController's rustypot/Dynamixel bus (the only other consumer of MOTOR_TO_ID's
values) is dead code on this robot, so those numbers were repurposed to mean "STM32
slot index" instead of a Dynamixel bus ID — see docs/dev/hw_stream.md for the full
story and how this was calibrated against the real board. Shared by
hw_state_stream.py (visualization, tolerates an incomplete table) and
zubr_robot_controller.py (real motor control, requires a complete one) so both use
the exact same validated mapping rather than two copies that could drift apart.
"""

from zubr_link import MOTOR_COUNT


def build_slot_map(motor_to_id: dict[str, int]) -> dict[int, str]:
    """Validate and invert MOTOR_TO_ID into {slot: joint_name}.

    Tolerates an incomplete or partially-wrong table rather than raising: a value
    outside [0, MOTOR_COUNT) or a slot claimed by two joints is logged and that
    joint(s) simply doesn't appear in the returned map. Callers that need every slot
    resolved (e.g. before commanding real motors) should check the result's
    completeness themselves — see zubr_robot_controller.py.
    """
    slot_to_joint: dict[int, str] = {}
    claimed_by: dict[int, str] = {}
    for name, slot in motor_to_id.items():
        if not (0 <= slot < MOTOR_COUNT):
            print(f"zubr_motor_map: {name!r} has MOTOR_TO_ID value {slot} — out of STM32 slot range [0, {MOTOR_COUNT}); not resolved until recalibrated.", flush=True)
            continue
        if slot in claimed_by:
            print(f"zubr_motor_map: {name!r} and {claimed_by[slot]!r} both claim slot {slot} in MOTOR_TO_ID — neither is resolved until this is fixed.", flush=True)
            slot_to_joint.pop(slot, None)
            continue
        claimed_by[slot] = name
        slot_to_joint[slot] = name
    missing = MOTOR_COUNT - len(slot_to_joint)
    if missing:
        print(f"zubr_motor_map: {missing} of {MOTOR_COUNT} STM32 slots have no (valid, unique) joint in MOTOR_TO_ID yet.", flush=True)
    return slot_to_joint
