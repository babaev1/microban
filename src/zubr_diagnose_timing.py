# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Standalone diagnostic: does the STM32/zubr link's near-universal "first attempt
fails, immediate retry succeeds" pattern depend on the delay between requests?

Context (see docs/dev/zubr_real_hardware.md's Update 6/7): every poll attempt —
succeeding or failing — takes an almost identical ~8-10ms, regardless of content.
That's consistent with the STM32 responding on its own fixed internal cycle rather
than instantly on receipt, and with our polling cadence (a clean 20ms = 2x10ms
multiple) landing in the same relative phase of that cycle on every regularly-timed
request — a phase that's consistently bad, while an immediate retry (sent at a
different, unsynchronized phase) usually isn't. A host-side receive-buffer flush
(tried first) had zero measurable effect, which fits: if this is really about the
STM32's own response timing, nothing done to our receive buffer touches it.

This script tests that directly: for each of several fixed delays between the end
of one transaction and the start of the next, it fires N single-attempt polls (no
retry logic — we want the raw per-delay failure rate) and reports the failure rate
and mean latency at that delay. If there's a phase-lock, failure rate should vary
clearly with delay rather than sitting flat — and if some delay is much better,
that tells us how to fix the real control loop's timing instead of guessing further.

Every poll leaves all motors relaxed — this script only ever reads, it never
commands the robot.

Usage:
    PYTHONPATH=src uv run --group sim src/zubr_diagnose_timing.py
"""

import argparse
import time

from zubr_link import DEFAULT_BAUDRATE, DEFAULT_PORT, ZubrLink


def _diff_marks(a: bytes, b: bytes) -> str:
    n = max(len(a), len(b))
    marks = []
    for i in range(n):
        ba = a[i] if i < len(a) else None
        bb = b[i] if i < len(b) else None
        marks.append("  " if ba == bb else "^^")
    return " ".join(marks)


def dump_failures(link: ZubrLink, count: int, delay_ms: float) -> None:
    """Capture raw bytes around the first `count` failures, to tell apart a
    bit-level corruption pattern (differs from the previous good frame almost
    everywhere) from a shift/framing error (differs by a fixed byte offset)
    from a stale-frame repeat (matches a previous frame's bytes exactly)."""
    delay_s = delay_ms / 1000.0
    last_good_raw: bytes | None = None
    last_good_sent: bytes | None = None
    found = 0
    attempts = 0
    print(f"Capturing {count} failures at delay={delay_ms}ms (sent/prev-good/this-frame, hex)...", flush=True)
    while found < count and attempts < count * 50:
        attempts += 1
        telemetry = link.poll()
        if telemetry is None:
            found += 1
            print(f"\n--- failure #{found} ({link.last_failure_reason}, {link.last_read_ms:.1f} ms) ---")
            print("sent:      ", link.last_sent.hex(" "))
            if last_good_sent is not None:
                print("prev sent: ", last_good_sent.hex(" "))
            if last_good_raw is not None:
                print("prev good: ", last_good_raw.hex(" "))
            print("this frame:", link.last_raw.hex(" "))
            if last_good_raw is not None:
                print("diff:      ", _diff_marks(last_good_raw, link.last_raw))
        else:
            last_good_raw = link.last_raw
            last_good_sent = link.last_sent
        if delay_s > 0:
            time.sleep(delay_s)
    if found < count:
        print(f"\nOnly saw {found} failures in {attempts} attempts.")


def sweep(link: ZubrLink, delays_ms: list[float], samples_per_delay: int) -> None:
    print(f"{'delay_ms':>10} {'fail_rate':>10} {'mean_ok_ms':>11} {'mean_fail_ms':>13}", flush=True)
    for delay_ms in delays_ms:
        delay_s = delay_ms / 1000.0
        failures = 0
        ok_total_ms = 0.0
        ok_count = 0
        fail_total_ms = 0.0
        fail_count = 0
        for _ in range(samples_per_delay):
            telemetry = link.poll()
            if telemetry is None:
                failures += 1
                fail_total_ms += link.last_read_ms
                fail_count += 1
            else:
                ok_total_ms += link.last_read_ms
                ok_count += 1
            if delay_s > 0:
                time.sleep(delay_s)
        fail_rate = failures / samples_per_delay
        mean_ok = (ok_total_ms / ok_count) if ok_count else float("nan")
        mean_fail = (fail_total_ms / fail_count) if fail_count else float("nan")
        print(f"{delay_ms:10.1f} {fail_rate:10.0%} {mean_ok:11.2f} {mean_fail:13.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=DEFAULT_PORT, metavar="DEV", help="default: %(default)s")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="default: %(default)s")
    parser.add_argument(
        "--samples", type=int, default=200, metavar="N",
        help="polls per delay value (default: %(default)s)",
    )
    parser.add_argument(
        "--delays", type=float, nargs="+",
        default=[0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 25, 30],
        metavar="MS",
        help="delays (ms) to sweep between the end of one poll and the start of the "
             "next (default: %(default)s)",
    )
    parser.add_argument(
        "--dump-failures", type=int, default=0, metavar="N",
        help="instead of sweeping, capture raw bytes around the first N failures "
             "(at a single fixed delay, see --delays' first value) to inspect for a "
             "corruption pattern",
    )
    args = parser.parse_args()

    link = ZubrLink(args.port, baudrate=args.baudrate)
    try:
        if args.dump_failures > 0:
            dump_failures(link, args.dump_failures, args.delays[0])
        else:
            print(f"Sweeping {len(args.delays)} delays x {args.samples} samples each — this will take a "
                  f"couple of minutes...", flush=True)
            sweep(link, args.delays, args.samples)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


if __name__ == "__main__":
    main()
