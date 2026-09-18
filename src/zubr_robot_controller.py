# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""ControllerProtocol implementation for the real Roki4 hardware's STM32/zubr motor
controller board — see zubr_link.py for the wire protocol, zubr_motor_map.py for the
slot<->joint calibration (MOTOR_TO_ID's values, verified complete before this class
will even construct), and docs/dev/hw_stream.md for how all of it (servo
identification, motion direction, IMU mounting) was reverse-engineered and validated
by streaming to a remote MuJoCo viewer before any of this wrote a single real motor
command.

Read/write model: every ZubrLink.poll() call is ONE request/response transaction that
simultaneously sends the 16 motors' current goal positions AND returns fresh
telemetry — there's no way to do only one half. This class handles that with two
different tolerances for a dropped/corrupt frame:

  - sync_write_goal_position() updates the in-memory goal-position table and then
    polls, but a dropped frame there is NOT fatal: the new targets are already
    stored, so the very next write *or* read retries the same values a tick (or
    less) later — this is what lets main.py's ramp_to_neutral() (a tight
    write-only loop, no interleaved reads) actually reach the motors every
    iteration instead of only updating local state.
  - sync_read_present_position() also polls (sending whatever goal positions are
    currently held) but DOES raise RuntimeError on a dropped frame, matching
    Scheduler.run()'s existing retry-with-backoff handling for exactly this
    exception from a read.

sync_read_present_position() is relied on to be the *first* controller call each
tick (true of Observer.read_state(), which calls it before position/velocity/
acc/gyro/quat) — every other read method below reuses that one cached telemetry
rather than polling again itself, avoiding a redundant serial round trip per tick
for channels that don't need their own fresh transaction.
"""

import math
import os
import time

import numpy as np

from constants import IMU_MOUNT_QUAT, KP_DEFAULT, MOTOR_SIGN, MOTOR_TO_ID, MOTOR_ZERO_TICKS, ZUBR_GYRO_SCALE, ZUBR_IMU_MOUNT_QUAT
from imu_reader import quat_apply_inverse
from zubr_link import DEFAULT_BAUDRATE, DEFAULT_PORT, MOTOR_COUNT, RELAX_POSITION, Telemetry, ZubrLink, rad_to_ticks, ticks_to_rad
from zubr_motor_map import build_slot_map

# read_gyro()/read_acc() below can't just pass the raw sensor reading through the way
# RobotController does for the BMI088. That works there ONLY because the real BMI088
# happens to be physically mounted at exactly IMU_MOUNT_QUAT relative to the trunk —
# the same rotation the mjcf's "imu" site (which mujoco_controller.py's read_gyro()
# reads from) assumes — so "raw BMI088 reading" and "what training's imu-site sensor
# would read" are the same frame by construction. The BHI260 is mounted at a
# *different* rotation (ZUBR_IMU_MOUNT_QUAT != IMU_MOUNT_QUAT — confirmed, they were
# calibrated independently and came out different), so its raw reading is a different,
# wrong frame for the policy — this bit a real run: gyro fed to the walk policy in
# the wrong frame, fine while standing still (gyro ~= 0 regardless of frame) but
# producing wildly wrong actions the moment real rotation appeared once walking
# started, which is exactly the "correct pose, then chaotic motion a second later"
# failure this fixes.
#
# GYRO_FRAME_ADAPTER carries a raw BHI260-frame vector into the SAME "as if mounted
# like a BMI088 at IMU_MOUNT_QUAT" frame training expects — one fixed rotation,
# composing the two mounts: conj(IMU_MOUNT_QUAT) * ZUBR_IMU_MOUNT_QUAT. Verified
# numerically (not just by this derivation) against ground truth before shipping:
# when ZUBR_IMU_MOUNT_QUAT == IMU_MOUNT_QUAT this reduces to the identity (recovering
# RobotController's plain raw-passthrough case exactly), and for the current real
# mount values, correcting a raw BHI260 reading through this exactly recovers what
# the equivalent BMI088-mounted reading would have been for the same true rotation —
# projected_gravity needs no equivalent adapter (verified separately): it's computed
# from imu_quat_to_body()'s q_body output, which is the trunk's actual orientation
# and therefore already mount-independent, unlike gyro's raw, frame-relative vector.
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _qconj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


_GYRO_FRAME_ADAPTER = _qmul(_qconj(IMU_MOUNT_QUAT), ZUBR_IMU_MOUNT_QUAT)
_gyro_adapter_norm = math.sqrt(sum(c * c for c in _GYRO_FRAME_ADAPTER))
_GYRO_FRAME_ADAPTER = tuple(c / _gyro_adapter_norm for c in _GYRO_FRAME_ADAPTER)


def _adapt_sensor_frame_vector(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a raw BHI260-frame vector (gyro or accel) into the "as if BMI088-mounted"
    frame training expects — see _GYRO_FRAME_ADAPTER above."""
    return tuple(float(v) for v in quat_apply_inverse(list(_qconj(_GYRO_FRAME_ADAPTER)), list(vec)))

# Hard sanity bound on any single commanded joint angle, applied just before it's
# converted to ticks and written to the wire — independent of and far tighter than
# the int16 wire format's own overflow point (~+/-12.57 rad). No joint on this robot
# legitimately needs anywhere near +/-180 deg from zero; a value past this is a bug
# upstream (e.g. a bad present_position reading feeding a move's ramp-in — see
# docs/dev/zubr_real_hardware.md), not a real target. Caught here rather than left to
# either crash the whole control loop (a struct.error packing an out-of-range tick
# count) or, worse, silently send whatever garbage tick value the overflow wrapped to.
_MAX_SAFE_RAD = math.pi

# Set MICROBAN_ZUBR_TIMING_DEBUG=1 to log every _poll() that needed more than one
# attempt: "zubr: poll needed N attempts (X.X ms)". A tick that needs several
# retries can silently stretch from ~1ms to ~200ms (retries * ZubrLink's 25ms serial
# timeout) with NO warning otherwise — sync_read_present_position() only ever prints
# something if ALL _READ_RETRIES attempts fail, so a tick that succeeds on, say, the
# 6th attempt costs ~150ms and nothing shows it. Scheduler.run() has no compensation
# for an overlong tick (it just starts the next one immediately), and the walk
# policy's reference-phase counter (moves/walk.py) advances once per *call*, not
# once per real 20ms — so a stretched tick desyncs it from real elapsed time. Added
# to directly measure whether this correlates with observed instability (a real,
# on-ground test diverged specifically once vx increased, which is also when
# reference-phase tracking engages) rather than continuing to guess.
_TIMING_DEBUG = os.environ.get("MICROBAN_ZUBR_TIMING_DEBUG", "0") == "1"


class ZubrRobotController:
    """Wraps ZubrLink to satisfy ControllerProtocol for the real Roki4 hardware."""

    mount_quat = ZUBR_IMU_MOUNT_QUAT

    # This link drops or corrupts a real, non-negligible fraction of frames in
    # practice (observed anywhere from ~0% to >50% depending on the session — see
    # docs/dev/hw_stream.md's live data points) — a single miss must not be fatal.
    # Reads retry immediately in a loop (each attempt bounded by ZubrLink's own
    # 25 ms serial timeout, so even the worst case here is small next to a 20 ms
    # scheduler tick budget only if it resolves within a couple of attempts, which
    # it does in practice) rather than surfacing on the very first miss — this is
    # what Scheduler.run()'s own retry-with-backoff around Observer.read_state()
    # assumes upstream, but main.py's ramp_to_neutral() (and the initial read right
    # after construction) call sync_read_present_position with no such wrapper of
    # their own, so without this a single startup blip crashed main.py outright.
    _READ_RETRIES = 8
    # Writes are already best-effort (a miss just gets resent next tick), so fewer
    # retries here — this only smooths out ramp_to_neutral()'s tight write-only loop.
    _WRITE_RETRIES = 2

    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUDRATE, dry_run: bool = False) -> None:
        # dry_run: sync_write_goal_position() logs what it would send (name, target
        # rad, computed ticks) but never actually updates _goal_ticks — every motor
        # slot stays at RELAX_POSITION for the controller's entire lifetime, and per
        # the protocol's own firmware behavior (not just this class cooperating: see
        # zubr_link.RELAX_POSITION / ~/zubr.py) that value unconditionally means
        # "release this motor," regardless of anything sync_write_torque_enable did.
        # Reads/telemetry (position, velocity, gyro, quat) still poll and update
        # completely normally — this is for safely observing what the walk policy
        # *would* command (see docs/dev/zubr_real_hardware.md's incident writeup)
        # without the robot ever being able to actually move.
        self._dry_run = dry_run
        self._link = ZubrLink(port, baudrate=baudrate)

        self._slot_to_joint = build_slot_map(MOTOR_TO_ID)
        unresolved = set(MOTOR_TO_ID) - set(self._slot_to_joint.values())
        if unresolved:
            raise RuntimeError(
                f"ZubrRobotController requires every joint in MOTOR_TO_ID to resolve to a "
                f"valid, unique STM32 slot before it will command real motors; unresolved: "
                f"{sorted(unresolved)!r}. See the warnings printed above and "
                f"docs/dev/hw_stream.md's slot-calibration section."
            )

        # Every motor starts relaxed and disabled — matches the robot's natural
        # power-on state (nothing commanded yet) until sync_write_torque_enable(True)
        # is called explicitly, same as main.py already does for RobotController.
        self._goal_ticks: list[int] = [RELAX_POSITION] * MOTOR_COUNT
        self._enabled: list[bool] = [False] * MOTOR_COUNT
        self._latest: Telemetry | None = None
        self._dropped_count = 0

        self._kp_warned = False
        self._last_unsafe_warn_s: float = 0.0
        self._last_dry_run_print_s: float = 0.0

        if self._dry_run:
            print(
                "ZubrRobotController: DRY RUN — every motor stays relaxed for this "
                "entire session; sync_write_goal_position() will log intended targets "
                "instead of sending them.",
                flush=True,
            )

        if ZUBR_GYRO_SCALE == 1.0 and not self._dry_run:
            print(
                "ZubrRobotController: WARNING — ZUBR_GYRO_SCALE is still its "
                "uncalibrated placeholder (1.0), feeding the walk policy raw sensor "
                "noise at an unknown, likely-huge scale. This alone reproduced real "
                "instability before (see docs/dev/zubr_real_hardware.md). Calibrate "
                "with `hw_state_stream.py --calibrate-gyro-scale` before trusting "
                "walk on real hardware.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Actuator writes

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        for motor_id, enabled in zip(ids, values):
            self._enabled[motor_id] = enabled
            if not enabled:
                self._goal_ticks[motor_id] = RELAX_POSITION
        self._poll(retries=self._WRITE_RETRIES)  # best-effort — see module docstring

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        # No equivalent concept in the zubr protocol: every transaction returns full
        # telemetry unconditionally, regardless of what was written. Nothing to configure.
        pass

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if self._dry_run:
            self._log_dry_run_targets(ids, positions)
            self._poll(retries=self._WRITE_RETRIES)  # still refresh telemetry; _goal_ticks untouched
            return

        for motor_id, pos in zip(ids, positions):
            name = self._slot_to_joint.get(motor_id)
            if name is None or not self._enabled[motor_id]:
                continue
            if not (-_MAX_SAFE_RAD <= pos <= _MAX_SAFE_RAD):
                self._warn_unsafe_target(name, pos)
                continue  # leave this one motor's last (known-safe) goal in place
            zero = MOTOR_ZERO_TICKS.get(name, 0)
            self._goal_ticks[motor_id] = rad_to_ticks(MOTOR_SIGN[name] * pos) + zero
        self._poll(retries=self._WRITE_RETRIES)  # best-effort — see module docstring

    def _log_dry_run_targets(self, ids: list[int], positions: list[float]) -> None:
        # Rate-limited — the scheduler calls this every tick (50 Hz).
        now = time.perf_counter()
        if (now - self._last_dry_run_print_s) < 0.3:
            return
        self._last_dry_run_print_s = now
        parts = []
        for motor_id, pos in zip(ids, positions):
            name = self._slot_to_joint.get(motor_id, f"slot{motor_id}")
            flag = " !!" if not (-_MAX_SAFE_RAD <= pos <= _MAX_SAFE_RAD) else ""
            parts.append(f"{name}={pos:+.3f}{flag}")
        print("[dry-run] would command: " + " ".join(parts), flush=True)

    def _warn_unsafe_target(self, name: str, pos: float) -> None:
        # Rate-limited: a persistent bad upstream value (e.g. a move stuck feeding a
        # bad reading into its own ramp-in) would otherwise spam this every tick.
        now = time.perf_counter()
        if (now - self._last_unsafe_warn_s) >= 1.0:
            print(
                f"ZubrRobotController: refusing to command {name!r} to {pos:+.3f} rad "
                f"(outside +/-{_MAX_SAFE_RAD:.3f} rad safety bound) — holding its last "
                "commanded position instead. This is a bug upstream (see "
                "docs/dev/zubr_real_hardware.md), not a real target.",
                flush=True,
            )
            self._last_unsafe_warn_s = now

    # ------------------------------------------------------------------
    # Actuator reads

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        telemetry = self._poll(retries=self._READ_RETRIES)
        if telemetry is None:
            raise RuntimeError(
                f"zubr: {self._READ_RETRIES} consecutive dropped/corrupt frames from "
                "STM32 controller"
            )
        return [self._present_position(telemetry, motor_id) for motor_id in ids]

    def read_present_position(self, motor_id: int) -> float:
        return self.sync_read_present_position([motor_id])[0]

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        telemetry = self._require_latest()
        return [self._present_velocity(telemetry, motor_id) for motor_id in ids]

    def read_present_velocity(self, motor_id: int) -> float:
        return self.sync_read_present_velocity([motor_id])[0]

    def _present_position(self, telemetry: Telemetry, motor_id: int) -> float:
        name = self._slot_to_joint[motor_id]
        zero = MOTOR_ZERO_TICKS.get(name, 0)
        return MOTOR_SIGN[name] * ticks_to_rad(telemetry.motor_positions[motor_id] - zero)

    def _present_velocity(self, telemetry: Telemetry, motor_id: int) -> float:
        name = self._slot_to_joint[motor_id]
        # TODO(verify on real hardware): assumes velocity is reported in the same
        # ticks-per-revolution unit as position ("ticks/s") — zubr.py documents no
        # units for this field, same caveat as gyro (see hw_state_stream.py).
        return MOTOR_SIGN[name] * ticks_to_rad(telemetry.motor_velocities[motor_id])

    # ------------------------------------------------------------------
    # Current / voltage — not available from this protocol

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        raise NotImplementedError(
            "ZubrRobotController: the zubr protocol's telemetry has no per-motor "
            "current field (see zubr_link.Telemetry) — Observer.observe_current must "
            "stay False with this controller."
        )

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        raise NotImplementedError(
            "ZubrRobotController: the zubr protocol's telemetry has no per-motor "
            "voltage field (see zubr_link.Telemetry) — Observer.observe_voltage must "
            "stay False with this controller."
        )

    def read_present_input_voltage(self, motor_id: int) -> float:
        raise NotImplementedError(
            "ZubrRobotController: the zubr protocol's telemetry has no per-motor "
            "voltage field (see zubr_link.Telemetry)."
        )

    # ------------------------------------------------------------------
    # Kp — not exposed by this protocol either

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        self._warn_kp_unsupported()
        return [KP_DEFAULT] * len(ids)

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._warn_kp_unsupported()

    def _warn_kp_unsupported(self) -> None:
        if not self._kp_warned:
            print(
                "ZubrRobotController: per-motor kp is not exposed by the zubr protocol "
                "(only 2 generic named-variable read/write slots exist, with no "
                "documented register map) — gain is whatever is fixed in firmware; "
                "ignoring.",
                flush=True,
            )
            self._kp_warned = True

    # ------------------------------------------------------------------
    # IMU
    #
    # read_quat() returns the raw (normalized) orientation uncorrected — matching
    # RobotController's/MuJoCoController's own read_quat(), since Observer.read_state()
    # applies the mounting correction externally via imu_quat_to_body(), and that
    # output (q_body) is mount-independent regardless of which sensor produced it
    # (verified — see the module-level comment above _GYRO_FRAME_ADAPTER).
    #
    # read_gyro()/read_acc() do NOT get the same free pass: they're raw, sensor-frame
    # vectors, not an absolute orientation, so they need _adapt_sensor_frame_vector()
    # to land in the same frame training's imu-site-mounted sensor produces — see
    # that comment for why plain passthrough (correct for RobotController's BMI088)
    # is wrong for this different sensor's different mounting.

    def read_acc(self) -> tuple[float, float, float]:
        telemetry = self._require_latest()
        return _adapt_sensor_frame_vector(tuple(float(v) for v in telemetry.acc_raw))

    def read_gyro(self) -> tuple[float, float, float]:
        telemetry = self._require_latest()
        adapted = _adapt_sensor_frame_vector(tuple(float(v) for v in telemetry.gyro_raw))
        return tuple(v * ZUBR_GYRO_SCALE for v in adapted)

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        _ = dt
        telemetry = self._require_latest()
        qx, qy, qz, qw = (float(v) for v in telemetry.quat_raw)
        quat = np.array([qw, qx, qy, qz], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-9:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(float(v) for v in (quat / norm))

    # ------------------------------------------------------------------
    # Lifecycle

    def shutdown(self) -> None:
        pass  # nothing analogous to imu_reader's background thread to stop here

    def close(self) -> None:
        self.shutdown()
        self._link.close()

    # ------------------------------------------------------------------
    # Internal

    def _poll(self, retries: int = 1) -> Telemetry | None:
        """Try up to ``retries`` transactions, returning the first successful one (or
        None if every attempt drops/corrupts). Each one sends the *same* current
        _goal_ticks — a retried attempt is not a new command, just resending the
        one that didn't get an answer."""
        start = time.perf_counter() if _TIMING_DEBUG else 0.0
        failures: list[str] = []
        for attempt in range(1, retries + 1):
            telemetry = self._link.poll(goal_positions=list(self._goal_ticks))
            if telemetry is not None:
                self._latest = telemetry
                if _TIMING_DEBUG and attempt > 1:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    print(
                        f"zubr: poll needed {attempt} attempts ({elapsed_ms:.1f} ms) "
                        f"[{', '.join(failures)}]",
                        flush=True,
                    )
                return telemetry
            self._dropped_count += 1
            if _TIMING_DEBUG:
                failures.append(f"{self._link.last_failure_reason}@{self._link.last_read_ms:.1f}ms")
        if _TIMING_DEBUG:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"zubr: poll FAILED all {retries} attempts ({elapsed_ms:.1f} ms) [{', '.join(failures)}]", flush=True)
        return None

    @property
    def latest_telemetry(self) -> Telemetry | None:
        """Most recent successfully decoded state frame, or None before the first
        one arrives. Exposed so other components (e.g. a remote-joystick input
        source) can read fields like the zubr remote's joystick axes without
        opening a second connection to this strictly one-master-at-a-time serial
        link — see input/zubr_remote_input.py."""
        return self._latest

    def _require_latest(self) -> Telemetry:
        if self._latest is None:
            raise RuntimeError("zubr: no telemetry received yet (dropped/corrupt frame)")
        return self._latest
