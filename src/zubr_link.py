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
import time
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
_STATE = struct.Struct("<10hi4bhihi32hH")


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


def rad_to_ticks(rad: float) -> int:
    """Inverse of ticks_to_rad — convert a target angle in radians to raw encoder ticks."""
    return round(rad * (TICKS_PER_REV / (2.0 * math.pi)))


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
        # Diagnostics from the most recent poll() call — see docs/dev/zubr_real_hardware.md's
        # timing investigation. last_read_ms times only the self._ser.read() call itself
        # (write + unpack/CRC are negligible next to it), so it directly answers "did this
        # attempt time out waiting (~= self._ser.timeout), or come back fast with wrong
        # content?" — the two failure modes read identically as "None" to callers otherwise,
        # but point at very different root causes (STM32-side processing latency vs. a
        # framing/stale-buffer bug on either end).
        self.last_failure_reason: str | None = None
        self.last_read_ms: float = 0.0
        self.last_sent: bytes = b""
        self.last_raw: bytes = b""

    def poll(
        self,
        goal_positions: list[int] | None = None,
        read_idx: tuple[int, int] = (0, 1),
        write: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Telemetry | None:
        """Send one directive and return the parsed response, or None on a
        dropped/short/CRC-mismatched frame (matches ~/zubr.py's "Frame dropped or
        timeout" case). See last_failure_reason/last_read_ms for which and how long.

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

        # Discard anything already sitting in the receive buffer before sending a new
        # directive. Without this, a response that arrived a little later than this
        # method assumed (e.g. the previous call's read() gave up right as it landed)
        # sits in the OS buffer until the *next* read() call consumes it first —
        # producing a byte sequence that straddles two frames: full-length, but
        # misaligned, so it fails CRC even though nothing was actually corrupted.
        # Diagnosed from a real run where every single failure was fast (~7-10ms,
        # nowhere near the 25ms timeout) and crc_mismatch, never short_read/timeout —
        # exactly what stale-buffer misalignment looks like, as opposed to the STM32
        # genuinely being slow to respond. See docs/dev/zubr_real_hardware.md.
        self._ser.reset_input_buffer()
        self._ser.write(payload)
        self.last_sent = payload

        read_start = time.perf_counter()
        raw = self._ser.read(_STATE.size)
        self.last_read_ms = (time.perf_counter() - read_start) * 1000.0
        self.last_raw = raw

        if len(raw) != _STATE.size:
            self.last_failure_reason = f"short_read:{len(raw)}/{_STATE.size}B"
            return None
        fields = _STATE.unpack(raw)
        if fields[-1] != _crc16(raw[:-2]):
            self.last_failure_reason = "crc_mismatch"
            return None
        self.last_failure_reason = None
        return Telemetry._from_fields(fields)

    def close(self) -> None:
        self._ser.close()
