# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Serial link to the robot's STM32 motor controller, over /dev/ttyS2.

Formalizes the wire protocol prototyped in ~/zubr.py into a reusable link: one
fixed-size request/response transaction per call, CRC16 (CCITT-FALSE, poly 0x1021)
guarded in both directions. Byte layout (see _DIRECTIVE/_STATE below) and CRC are
copied from that script unchanged.
"""

import math
import struct
from dataclasses import dataclass

import serial

MOTOR_COUNT = 16
TICKS_PER_REV = 16384  # encoder ticks per motor revolution (16384 ticks == 2*pi rad)
RELAX_POSITION = 32767  # goal-position value meaning "release this motor" (no torque)

DEFAULT_PORT = "/dev/ttyS2"
DEFAULT_BAUDRATE = 1_500_000

# Directive (host -> STM32): 2 arbitrary variable writes, 2 arbitrary variable reads,
# then 16 goal positions (RELAX_POSITION to leave a motor free) — see ~/zubr.py.
_DIRECTIVE = struct.Struct("<hihihh16h")

# State (STM32 -> host): imu accel/gyro/quaternion (10 int16), remote control (buttons
# + 4 joystick axes), the 2 requested variable reads, then (position, velocity) per
# motor for all 16 motors, then a trailing CRC16 — see ~/zubr.py.
_STATE = struct.Struct("<10hi4bhihi32hh")


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def ticks_to_rad(ticks: int) -> float:
    """Convert raw encoder ticks to radians (16384 ticks == 2*pi rad)."""
    return ticks * (2.0 * math.pi / TICKS_PER_REV)


@dataclass
class Telemetry:
    """One decoded state frame from the STM32."""

    acc_raw: tuple[int, int, int]
    gyro_raw: tuple[int, int, int]
    quat_raw: tuple[int, int, int, int]  # (x, y, z, w), raw units — not yet calibrated
    remote_buttons: int
    left_joystick: tuple[int, int]
    right_joystick: tuple[int, int]
    read_values: tuple[int, int]  # values for the 2 variable indices requested
    motor_positions: tuple[int, ...]  # 16 raw encoder ticks, in STM32 slot order
    motor_velocities: tuple[int, ...]  # 16 raw encoder ticks/s, in STM32 slot order

    @classmethod
    def _from_fields(cls, fields: tuple) -> "Telemetry":
        motor_pairs = fields[19:19 + 2 * MOTOR_COUNT]
        return cls(
            acc_raw=fields[0:3],
            gyro_raw=fields[3:6],
            quat_raw=fields[6:10],
            remote_buttons=fields[10],
            left_joystick=fields[11:13],
            right_joystick=fields[13:15],
            read_values=(fields[16], fields[18]),
            motor_positions=motor_pairs[0::2],
            motor_velocities=motor_pairs[1::2],
        )


class ZubrLink:
    """One request/response transaction per `poll()` call — this protocol is strictly
    query/response, so a frame is only ever produced in reaction to sending a
    directive (there is no free-running telemetry stream to just listen on)."""

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 0.025,
    ) -> None:
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def poll(
        self,
        goal_positions: list[int] | None = None,
        read_idx: tuple[int, int] = (0, 1),
        write: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Telemetry | None:
        """Send one directive and return the parsed response, or None on a
        dropped/short/CRC-mismatched frame (matches ~/zubr.py's "Frame dropped or
        timeout" case).

        Defaults to leaving every motor relaxed (RELAX_POSITION) and writing nothing —
        safe to call in a read-only polling loop.
        """
        if goal_positions is None:
            goal_positions = [RELAX_POSITION] * MOTOR_COUNT
        if len(goal_positions) != MOTOR_COUNT:
            raise ValueError(f"goal_positions must have {MOTOR_COUNT} entries, got {len(goal_positions)}")

        write_idx1, write_val1, write_idx2, write_val2 = write
        read_idx1, read_idx2 = read_idx
        control = [write_idx1, write_val1, write_idx2, write_val2, read_idx1, read_idx2, *goal_positions]
        payload = _DIRECTIVE.pack(*control)
        payload += struct.pack("<H", _crc16(payload))
        self._ser.write(payload)

        raw = self._ser.read(_STATE.size)
        if len(raw) != _STATE.size:
            return None
        fields = _STATE.unpack(raw)
        if fields[-1] != _crc16(raw[:-2]):
            return None
        return Telemetry._from_fields(fields)

    def close(self) -> None:
        self._ser.close()
