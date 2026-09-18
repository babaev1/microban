# Running the real Roki4 (RL walk included) on the STM32/zubr board

Everything up to this point (servo identification/motion direction, IMU mounting,
remote joystick control) was validated read-only, streamed to a remote MuJoCo viewer
— see [hw_stream.md](hw_stream.md) and [sim_joystick_control.md](sim_joystick_control.md).
This is the step after that: **`make run` now actually commands the real motors**
through the STM32/zubr board, using everything calibrated in those earlier passes.

## Incident: walk policy went chaotic ~1s after starting — gyro was in the wrong frame

First real attempt: `main.py` ramped to neutral correctly, walking was toggled on,
the robot reached the walk's starting pose with correct posture on every servo — then
about a second later began commanding random poses at dangerous speed, and the robot
was power-cycled to stop it before any damage.

**Root cause**: `read_gyro()` was returning the BHI260's raw sensor-frame reading
unmodified, copying `RobotController`'s pattern for the real BMI088. That pattern is
only correct there because the real BMI088 happens to be physically mounted at
exactly `IMU_MOUNT_QUAT` relative to the trunk — the same rotation the mjcf's "imu"
site (which the simulated/training gyro reads from) assumes — so "raw BMI088 reading"
and "what training's sensor would read" are the same frame by construction. The
BHI260 is mounted at a *different* rotation (`ZUBR_IMU_MOUNT_QUAT != IMU_MOUNT_QUAT` —
they were calibrated independently against the real board and came out different), so
its raw reading was a different, wrong frame for the policy's gyro observation.

This explains the exact timeline: at rest, gyro is ~0 in *any* frame, so the wrong
frame was invisible during the ramp-in and initial pose. The instant real rotation
appeared (the policy's own first stepping motion), the gyro observation became
actively wrong — feeding the network a rotation-axis signal that didn't correspond to
reality, which is exactly the kind of out-of-distribution input that produces
erratic, divergent actions from a network never trained on it. `projected_gravity`
was verified unaffected (see below) — it's the walk policy's *other* IMU channel, and
looked fine, consistent with the robot's posture staying correct right up until the
gyro-dependent instability hit.

**Fix**, in `zubr_robot_controller.py`: `_GYRO_FRAME_ADAPTER`, a single fixed rotation
composing both mounts (`conj(IMU_MOUNT_QUAT) * ZUBR_IMU_MOUNT_QUAT`), applied to
`read_gyro()`'s (and, for consistency, `read_acc()`'s) raw reading before returning
it — carrying it into the same "as if mounted like a BMI088" frame training's sensor
produces. Verified two ways before shipping: (1) when the two mount constants are
equal, the adapter reduces to the identity, recovering `RobotController`'s plain
passthrough exactly; (2) for the actual current mount values, correcting a raw
BHI260 reading through it exactly recovers what an equivalently-mounted BMI088 would
have reported for the same true rotation (checked against a ground-truth synthetic
scenario, not just algebra).

**`projected_gravity` does *not* need an equivalent adapter** — verified separately:
it's derived from `imu_quat_to_body()`'s `q_body` output, the trunk's actual absolute
orientation, which came out numerically identical whether computed via the BMI088's
mount+reading or the BHI260's — unlike gyro's raw, sensor-frame-relative vector, an
absolute orientation is mount-independent once correctly converted once.

This was caught and fixed from your incident report alone — I have not re-run this
against real hardware since motor control on this board is intentionally something I
avoid triggering unsupervised.

### Update: the same behavior repeated after the gyro fix — switched to safe diagnosis

The gyro-frame fix above is verified correct for what it fixes, but a second real
attempt with it applied reproduced the same chaotic behavior — so either it wasn't
the only bug, or something else is *also* wrong. Rather than ship another
single-hypothesis fix and ask for a third at-risk test, added a way to see exactly
what the policy is doing with **zero ability for the robot to actually move**:

- **`ZubrRobotController(dry_run=True)`** (`MICROBAN_DRY_RUN=1` for `main.py`):
  `sync_write_goal_position()` logs the target it would have sent instead of
  applying it — every motor slot stays at `RELAX_POSITION` for the controller's
  entire life. This is a hardware-level guarantee, not just this class cooperating:
  per the protocol's own firmware behavior (`~/zubr.py`), that value unconditionally
  means "release this motor," regardless of anything `sync_write_torque_enable` did.
  Reads/telemetry still poll and update completely normally, so the rest of the
  control loop (including real RL inference) runs exactly as it would live.
- **`moves/walk.py`'s existing `DEBUG_PRINT`** (already in the codebase, previously
  a hardcoded `False`) is now `MICROBAN_WALK_DEBUG=1`-controllable, and now also
  reports `max|vel|` — the largest per-joint velocity observation the policy is
  seeing, and which joint — alongside the existing `proj_grav`/`gyro`/`lean`/
  `max|action|`. Chosen because the next leading suspect is `_present_velocity()`'s
  units: it assumes STM32 velocity is reported in the same ticks-per-revolution unit
  as position, a guess that has never been verified against real hardware (unlike
  gyro's frame, which now has been) — an out-of-scale velocity feeding the policy
  would produce exactly this kind of out-of-distribution, erratic output, and would
  look identical to the gyro bug from the outside (fine while still, wrong once
  actually moving).

**To diagnose safely:**
```bash
MICROBAN_DRY_RUN=1 MICROBAN_WALK_DEBUG=1 PYTHONPATH=src .venv/bin/python src/main.py
```
Toggle walk with `v` as before — the robot cannot move — and watch the printed
`proj_grav`/`gyro`/`lean`/`max|action|`/`max|vel|` line (~3/s). Report back what
these look like, especially in the first couple of seconds after `step()` takes
over from the ramp-in: a `max|action|` or `max|vel|` that's small and stable points
elsewhere; either one growing large or erratic right at that transition points at
the channel it names.

### Update 2: the dry-run log found two more real bugs, independent of the gyro-frame fix

The captured log (`MICROBAN_DRY_RUN=1 MICROBAN_WALK_DEBUG=1`) was extremely useful —
it shows `proj_grav ~= (0, 0, -1)` throughout (confirms the gravity/gravity-mount
work is correct) but two other things clearly wrong:

1. **All four shoulder joints have a large, uncorrected raw-tick zero offset.**
   `sync_write_goal_position`'s dry-run log shows `left_shoulder_pitch`,
   `left_shoulder_roll`, `right_shoulder_pitch`, `right_shoulder_roll` starting each
   ramp (`ramp_to_neutral()` and `WalkMove.on_start()` alike) around ±2.4-2.8 rad
   (~140-160°) and decaying toward 0 as the ramp interpolates toward its target —
   meaning `sync_read_present_position()` is reporting the shoulders as being
   ~140-160° away from wherever they actually are. This matches an early raw-telemetry
   observation almost exactly: slots 0-3 (the current `MOTOR_TO_ID` values for exactly
   these four joints) once read 6693-7373 ticks (`ticks_to_rad(7373) ~= 2.83 rad`)
   while otherwise stationary — the "assumes raw tick 0 == mechanical zero" limitation
   flagged since the very first hardware pass, now confirmed wrong for these four.
   Every tick this ran, the walk policy's joint-position-error observation for these
   four joints was wrong by ~140-160°, independent of gyro — on its own, easily
   enough to destabilize the policy.

   **Fixed**: `constants.MOTOR_ZERO_TICKS` (a new dict, empty/no-op by default — same
   incremental philosophy as `MOTOR_TO_ID`/`ZUBR_IMU_MOUNT_QUAT`) is now subtracted
   from the raw ticks (in the raw-tick domain, before `MOTOR_SIGN`/`ticks_to_rad`) in
   both `zubr_robot_controller.py`'s read *and* write paths — verified as a matched
   inverse pair, not just the read side: commanding 0 rad for a joint with a
   calibrated offset now writes that joint's *offset* tick value (its true mechanical
   zero), not raw tick 0, and reading it back recovers 0 rad exactly. Calibrate with:
   ```bash
   PYTHONPATH=src uv run --group sim src/hw_state_stream.py --calibrate-zero 100
   ```
   while the robot is held/standing in its true `NEUTRAL_POSE` (all joints at 0)
   — paste the printed dict over the current `MOTOR_ZERO_TICKS` in `constants.py`.
   `hw_state_stream.py`'s own visualization also applies this now, for the same
   accuracy benefit.

2. **Gyro is never scale-converted at all — only rotated.** With the robot completely
   motionless (dry-run guarantees this), the debug log's `gyro=` values swing between
   roughly -6 and +5 across successive ~0.3s samples. A real stationary robot's
   angular velocity is ~0 rad/s; these are raw register counts with ordinary sensor
   noise on them, and that noise floor alone is already large relative to any
   plausible rad/s scale — meaning even before any real rotation, the policy's gyro
   observation is dominated by noise blown up by an unknown (and almost certainly
   large) missing scale factor. `~/zubr.py` documents no LSB-to-real-units
   conversion for this field, and unlike the frame (fixable purely from two mount
   quaternions already calibrated), the *scale* genuinely cannot be derived without
   either hardware documentation or a physical calibration measurement.

   (Shoulder zero-offset above deprioritized per your call — not fixed further for
   now.)

### Update 3: gyro-scale — resolved via documented sensitivity

Two paths to `ZUBR_GYRO_SCALE` were built:

- **`hw_state_stream.py --calibrate-gyro-scale SECONDS`**: physically rotate the
  robot's trunk about one axis for the prompted duration (the script times the
  actual elapsed wall-clock time itself), report the total degrees completed, get
  a scale factor back. Verified the recovery math against a synthetic rotation with
  a known injected scale before handing it over. Still available as an independent
  cross-check if the value below is ever in doubt.
- **The actual fix used**: the firmware's gyro sensitivity is documented as raw
  count / 16384 == degrees/second. `ZUBR_GYRO_SCALE` is now
  `deg2rad(1/16384)` — the exact rad/s-per-count factor, no physical measurement
  needed. Sanity check: at this scale, the ~-6..+5 raw-count noise observed while
  the robot was motionless becomes ~+-6e-6 rad/s — negligible, exactly what a
  stationary gyro should read, confirming the earlier instability really was
  "raw counts fed in as if they were already rad/s" and not a hardware fault.

**This resolves the "do not run real walk" gate** — `ZubrRobotController`'s
startup warning (which fired whenever `ZUBR_GYRO_SCALE` was still its old `1.0`
placeholder) no longer triggers, since the constant is now a real value. Still,
given two prior incidents, re-run the dry-run diagnostic below before trusting a
real walk again:
```bash
MICROBAN_DRY_RUN=1 MICROBAN_WALK_DEBUG=1 PYTHONPATH=src .venv/bin/python src/main.py
```
Toggle `v`, confirm `gyro=` sits near 0 while stationary and `max|action|`/
`max|vel|` look small and stable through the ramp-in→policy-takeover transition,
*then* consider a real (non-dry-run) attempt — robot secured, ready to cut power,
same as every real test on this backend so far.

### Update 4: real tests are stable at low speed, still diverge at higher vx — suspect timing

With the gyro fix in place, `vx=0`/`0.1` real tests (held in the air) came out sane
and bounded (`max|action|` 0.08-0.3, small lean) — a real improvement. But `vx=0.2`
consistently destabilized within a few hundred ms, both held in the air and (worse,
faster: ~16° lean in ~120ms) on the ground with no spotter — this last one should
not be repeated without support/spotting; an unsupported real fall risk was reached.

Two hypotheses considered:
- **No ground contact** (held-in-air tests only) — plausible there, but the
  on-ground test diverged the same way with real ground contact present, which
  weakens this as the sole explanation.
- **Variable, uncompensated control-loop latency** — the more likely remaining
  cause. `sync_read_present_position()`'s retry loop (added so a single dropped
  frame wouldn't crash `main.py` — see Update 1) can silently stretch a nominal
  20ms tick to ~200ms (up to 8 retries * 25ms serial timeout) *without* printing
  anything, since the "serial read error" warning only fires when every retry in
  a call fails. `Scheduler.run()` has no compensation for an overlong tick, and
  `moves/walk.py`'s reference-phase counter advances once per *call*, not once
  per real 20ms — so a stretched tick desyncs it from real elapsed time. This
  would plausibly get worse under load if higher motor current increases the
  frame-drop rate (the EMI hypothesis from Update 3's context), which would
  produce exactly "stable at rest, destabilizes once it moves harder."

**Added, not yet correlated with real data**: `MICROBAN_ZUBR_TIMING_DEBUG=1`
makes `ZubrRobotController._poll()` print `zubr: poll needed N attempts (X.X ms)`
every time a read needed more than one attempt — direct visibility into exactly
the latency spikes described above, which were previously invisible. Verified the
instrumentation fires correctly against a synthetic flaky link before handing it
over. Run alongside `MICROBAN_WALK_DEBUG=1` on the next test and check whether
these lines cluster right before/during a lean/action spike — that would confirm
the timing hypothesis directly rather than leaving it as speculation.

### Update 5: real data — retries are the baseline, not an exception

Result of Update 4's instrumentation: on a real run, **nearly every single poll**
needed a retry, including throughout the completely static `ramp_to_neutral()`
phase before walk was even toggled — `zubr: poll needed 2 attempts (~20ms)`,
extremely consistently (19.6-20.3ms), the *entire time*, motion or not. This
rules out motor-current/EMI as the primary cause of the *baseline* overhead (it's
present at zero current draw) and points instead at a systematic timing/protocol
mismatch — most likely the STM32 only produces a fresh valid response once per
its own internal cycle, and the host's fixed-cadence polling is frequently out of
phase with it, so the first attempt catches a stale/not-yet-ready frame.

Separately, during actual walking — especially approaching a fall — attempt
counts got measurably worse (3, 4, 5, 6, up to 8 attempts, ~30-80ms) and these
spikes clustered in the same stretches as the biggest lean/action growth, though
not as one isolated smoking-gun spike — more a generally noisier window
overlapping the escalation. Consistent with, not proof of, a reinforcing loop
(worse timing → worse tracking → harder corrective motion → more load → worse
timing).

Net effect: **the control loop has not actually run at a clean 50Hz/20ms in any
test so far** — every test, including the ones that looked stable at low `vx`,
ran under this same persistent baseline overhead. That reframes today's results:
even the "good" runs weren't testing the policy under its trained assumptions.

**Added finer-grained instrumentation** to distinguish *why* an attempt failed —
previously both failure modes were indistinguishable ("dropped/corrupt"):
- `zubr_link.ZubrLink` now exposes `last_failure_reason` (`"short_read:N/92B"` —
  fewer bytes than expected arrived before the 25ms timeout, i.e. object
  genuinely wasn't ready in time; or `"crc_mismatch"` — a full frame arrived but
  its content was wrong, i.e. more consistent with stale/misaligned buffer
  content than a slow response) and `last_read_ms` (how long that specific
  `read()` call actually took — near-instant vs. near the full 25ms timeout is
  itself diagnostic of which failure mode occurred, independent of the label).
- `ZubrRobotController._poll()`'s `MICROBAN_ZUBR_TIMING_DEBUG=1` output now
  includes both per failed attempt: `zubr: poll needed 3 attempts (29.7 ms)
  [short_read:0/92B@25.0ms, crc_mismatch@1.2ms]`. Also now prints when *every*
  attempt in a call fails (previously silent under timing-debug specifically,
  though `Scheduler`'s own "serial read error" warning still fires separately).
  Verified against a synthetic link producing each failure mode before handing
  this over.

Run the same `MICROBAN_ZUBR_TIMING_DEBUG=1 MICROBAN_WALK_DEBUG=1` command again —
the reasons/timings in the new output should show whether it's consistently
`short_read` near the full timeout (points to genuine STM32-side response
latency) or consistently `crc_mismatch` arriving fast (points to a framing/stale-
buffer bug, possibly fixable in software without needing STM32 firmware changes).

### Update 6: root cause found — stale bytes in the receive buffer, fixed

Result of Update 5's finer-grained instrumentation: **every single failure across
an entire real run — hundreds of polls — was `crc_mismatch`, arriving in ~7-10ms**.
Not one `short_read`/timeout anywhere. That settles the question: the STM32 is not
slow. A full-length frame arriving fast but failing its own CRC, this consistently,
means we're reading a byte sequence that straddles two frames — a stale-but-valid
frame would still pass its own CRC, so this is misalignment, not corruption or
STM32-side latency.

**Root cause**: `ZubrLink.poll()` never flushed the serial receive buffer before
sending a new directive. If a response arrived a little later than a previous
`read()` call assumed (that call already gave up, or moved on), its bytes sat in
the OS buffer until the *next* `read()` consumed them first — a leftover tail from
one frame followed by the head of the next, full length, wrong content. This
recurred almost every tick, explaining the near-universal "needs 2 attempts, first
one crc_mismatch" baseline from Update 5, entirely independent of motor activity —
consistent with it being present from the very start of every test today,
including the completely static `ramp_to_neutral()` phase.

**Fixed**: `poll()` now calls `self._ser.reset_input_buffer()` immediately before
writing each new directive, discarding anything left over from before. Safe to do
here specifically because the protocol is strictly synchronous request/response —
by the time we're about to send the next request, any response related to the
previous one has long since either been consumed or given up on.

This is the first concrete, non-speculative root cause found for the retry
pattern itself (as opposed to the RL instability, which had its own separate,
already-fixed causes — gyro frame, gyro scale). If it holds, `MICROBAN_ZUBR_TIMING_DEBUG=1`
on the next run should show mostly `1 attempt` (no line printed at all, since the
debug line only fires when `attempt > 1`) instead of the near-universal `2
attempts` seen before — a large, real reduction in per-tick timing jitter, which
was one of the two live hypotheses (alongside no-ground-contact dynamics and the
shoulder offset) for why higher-`vx` walks kept destabilizing. Not yet tested
against real hardware.

### Update 7: the buffer-flush fix did nothing — real root cause is a firmware-side race

Re-ran with `reset_input_buffer()` in place: **identical pattern, no improvement at
all** — same near-universal 2-attempts-per-poll, same `crc_mismatch@~9ms`. That
falsifies the Update 6 diagnosis. A host-side receive-buffer flush cannot explain a
result that didn't change once one was added.

Two follow-up diagnostics, both read-only ([`src/zubr_diagnose_timing.py`](../../src/zubr_diagnose_timing.py),
motors always left relaxed):

1. **Delay sweep** — single-attempt polls at fixed inter-request delays from 0 to
   30ms. Failure rate sat flat at **~44-55% for every delay tested**, including
   values far from any plausible alias of a ~10ms STM32 cycle. This rules out a
   phase-lock between our polling cadence and an STM32-internal loop (the Update 6
   framing's alternative explanation) — if that were the cause, some delay should
   have clearly escaped it, and none did.
2. **Byte-level dump of failing frames** (`--dump-failures`) — the actual finding.
   Failing frames are **not corrupted**: every field is internally coherent and
   physically plausible. Diffed against the immediately preceding good frame, the
   large static block (all 16 motor positions/velocities — motors were at rest,
   relaxed) is **byte-identical every single time**. The only bytes that differ are
   exactly the fields expected to change fast on their own — accel/gyro (sensor
   jitter) and a monotonically-incrementing counter — each incremented by a
   plausible, small step, never garbage.

**Revised root cause (firmware-side, not host-side)**: this looks like the STM32
computes the response CRC over its buffer, but a separate, faster interrupt (the one
updating IMU/accel/gyro and that counter) is still free to write into the *same*
buffer between when the CRC is computed and when the frame actually finishes
transmitting. The bytes that go out are a genuine torn mix — mostly the snapshot the
CRC was computed against, plus a couple of fields already bumped to their next value
by the time transmission caught up — which is why the content reads as completely
sane telemetry that simply doesn't hash to the CRC that was appended to it.

If correct, this is not fixable from the host side (short of not checking CRC at
all, which was explicitly rejected as too risky for a path that feeds real motor
commands — see decision below). `reset_input_buffer()` is harmless and stays in
`zubr_link.py`, but it does not address this.

**Decision (user, 2026-09-17)**: pursue getting access to the STM32 firmware source
or its author, rather than a host-side workaround. This is now a concrete, evidenced
bug report, not a guess: *"CRC appears to be computed against the response buffer
before it's fully frozen against a concurrent sensor-update ISR — frames that fail
CRC contain coherent, plausible values (not corruption), and the only fields that
differ from the previous good frame are the fast-changing ones (accel/gyro, a
counter), never the slow ones (motor position/velocity, unchanged when at rest).
Suggests the CRC should be computed over a snapshot taken atomically (e.g. with the
relevant interrupt briefly disabled) rather than over the live buffer."* Until this
is fixed upstream, expect the ~50% retry rate to persist regardless of anything
done in this repo.

## What changed

- **New: [`src/zubr_robot_controller.py`](../../src/zubr_robot_controller.py)** —
  `ZubrRobotController`, a `ControllerProtocol` implementation for the real hardware
  (parallel to `RobotController`, the now-dead rustypot/XL330 backend). `src/main.py`
  now constructs this instead of `RobotController` — `make run`/`make stop`/the
  gamepad daemon all pick it up automatically, no other changes needed there.
- **New: [`src/zubr_motor_map.py`](../../src/zubr_motor_map.py)** — `build_slot_map()`
  moved out of `hw_state_stream.py` so both it and `zubr_robot_controller.py` share
  the exact same validated slot-mapping logic instead of two copies that could drift.
- **`imu_reader.imu_quat_to_body()` / `ControllerProtocol` / `observer.py` /
  `scheduler.py`**: every controller now exposes its own `mount_quat` (`RobotController`
  → `IMU_MOUNT_QUAT`, `MuJoCoController` → `IMU_MOUNT_QUAT`, `ZubrRobotController` →
  `ZUBR_IMU_MOUNT_QUAT`), and `Observer.read_state()`/`Scheduler`'s IMU debug print
  both read it from the controller instead of assuming the BMI088's constant. This was
  a latent bug waiting to happen: without it, plugging `ZubrRobotController` in would
  have silently applied the *wrong* IMU's mounting correction to `projected_gravity` —
  the walk policy's single most safety-critical observation.
- **New: [`src/zubr_imu.py`](../../src/zubr_imu.py)** / `make zubr-imu` — mirrors
  `imu.py`'s output for the STM32 board's onboard BHI260 instead of the separate I2C
  BMI088 `imu.py` reads.
- **`pyproject.toml`**: `pyserial` moved from the `sim` dependency group to base
  dependencies. It was sim-group-only because it used to be needed only by
  `hw_state_stream.py`; now `main.py` needs it too, and `make run` deploys via `make
  setup`'s `uv sync --frozen` (no `--group sim`) — leaving it sim-only would have made
  `main.py` fail at import with `No module named 'serial'` the moment this shipped.
  Caught by actually reproducing that exact deploy command before calling this done.
- **New: [`src/input/zubr_remote_input.py`](../../src/input/zubr_remote_input.py)** —
  `main.py`'s counterpart to `input/zubr_joystick_input.py` (which only ever wired
  into `sim_main.py --joystick`). Same axis mapping/signs/"velocity from the
  remote, everything else from the keyboard" design, but **not** the same
  implementation: `zubr_joystick_input.py` opens and polls its own `ZubrLink`,
  which is fine in sim (nothing else talks to the STM32 there) but would corrupt
  both streams on real hardware, where `ZubrRobotController` already owns the one
  connection to `/dev/ttyS2` and this protocol only tolerates one poller at a
  time. `ZubrRemoteInputSource` instead takes a reference to that controller and
  reads its new `latest_telemetry` property — the same frame the controller just
  fetched for motor/IMU state, at no extra STM32 traffic. Opt in with
  `MICROBAN_INPUT=zubr_remote` (see [usage.md](../usage.md)); never
  auto-detected, unlike the gamepad.

## Read/write model and safety design

`ZubrLink.poll()` is one request/response transaction that *simultaneously* sends the
16 motors' goal positions and returns fresh telemetry — there's no way to do only one
half (see `zubr_link.py`). `ZubrRobotController` handles that with two different
tolerances for a dropped/corrupt frame, matching what each direction actually needs:

- `sync_write_goal_position()` updates its in-memory goal-position table, then polls
  — but a dropped frame there is **not** fatal, since the new target is already
  stored and the very next write or read resends it. This matters concretely:
  `main.py`'s `ramp_to_neutral()` is a tight write-only loop with no interleaved
  reads, so every write must actually reach the hardware on its own.
- `sync_read_present_position()` also polls, and **does** raise `RuntimeError` — but
  only after `_READ_RETRIES` (8) consecutive dropped/corrupt frames in its own
  internal retry loop, not on the first one. This link drops a real, non-negligible
  fraction of frames in practice (observed anywhere from ~0% to >50% depending on the
  session), and the first real run hit exactly this: `main.py`'s `ramp_to_neutral()`
  and the initial read right after construction call this method with no wrapper of
  their own (unlike `Scheduler.run()`'s main loop, which has its own separate
  retry-with-backoff around `Observer.read_state()`), so a single startup blip
  crashed `main.py` outright before this internal retry was added. Writes get a
  smaller retry budget (`_WRITE_RETRIES`, 2) for the same reason, tolerated rather
  than raised, since a miss there just gets resent on the next write anyway.
- `sync_read_present_position()` is relied on to be the *first* controller call each
  tick (true of `Observer.read_state()`) — every other read method reuses that one
  cached telemetry rather than polling again, avoiding a redundant round trip per
  tick for velocity/acc/gyro/quat.
- **Torque enable has no protocol equivalent** — `sync_write_torque_enable(False)`
  is implemented by substituting `RELAX_POSITION` into that motor's slot on every
  subsequent write until re-enabled, rather than any firmware register.
- **Every commanded angle is sanity-checked against `_MAX_SAFE_RAD` (±π rad) before
  it's converted to ticks and written.** The second real run hit this directly:
  `WalkMove.on_start()` seeds its ramp-in from `obs.robot_state.motor_positions.get(name,
  0.0)` — the real, current position read straight off the hardware — and at t≈0
  commands close to that value verbatim. If that reading is itself far out of range
  for some joint, this fed a multi-radian target straight through, which either blew
  past the wire format's own int16 range entirely (a `struct.error` that crashed the
  whole control loop, since `sync_write_goal_position` isn't wrapped in
  `Scheduler.run()`'s read-retry logic) or, worse, would have silently sent whatever
  garbage tick value that overflow happened to wrap to. A value outside the bound now
  gets a rate-limited warning and that one motor's *previous* commanded position is
  held instead — every other motor is unaffected, and the loop keeps running rather
  than crashing outright. **If you see this warning, the actual bug is almost
  certainly the "zero offset" gap already flagged in [hw_stream.md](hw_stream.md)**:
  `hw_state_stream.py`/this controller both assume raw tick 0 is each joint's
  mechanical zero, with no per-joint calibration table — if some joint's real
  absolute-encoder reading is far from 0 at its true zero (plausible for a
  multi-turn encoder never re-zeroed since assembly), every angle computed from it
  inherits that offset. Check which joint the warning names and compare its
  `sync_read_present_position` reading against the robot's actual pose.
- **Startup requires full slot calibration**: the constructor calls
  `build_slot_map()` and raises immediately if any of `MOTOR_TO_ID`'s 16 joints
  doesn't resolve to a valid, unique slot — unlike `hw_state_stream.py` (visualization
  only), commanding real motors with an ambiguous mapping is not something to run
  with partial calibration.

All of the above was verified with a fake stand-in link (no real hardware touched) —
torque-disable forcing RELAX, sign/tick round-trip through a real joint, dropped-frame
behavior on both read (raises) and write (doesn't), raw IMU passthrough, and the kp/
current/voltage stubs below — before this was ever pointed at `/dev/ttyS2`.

## Known hardware/protocol limitations — read before your first real walk test

- **Per-motor kp is not exposed by this protocol at all.** `sync_read_kp`/
  `sync_write_kp` are no-ops (one console warning, then silent) — `sync_read_kp`
  returns `KP_DEFAULT` as a placeholder so nothing crashes, but nothing you write
  ever reaches the hardware. **This means the walk policy's kp transition
  (`KP_DEFAULT` while holding the ramp-in pose → `KP_RL` once walking starts, see
  `moves/walk.py`) is completely inert on this hardware** — the real robot will run
  at whatever gain is fixed in the STM32 firmware for its entire session, not the
  softer `KP_RL` the policy was tuned against. If the walk looks stiffer/different
  than sim, this is the first thing to account for, not a policy bug.
- **Per-motor current and voltage are not in this protocol's telemetry either**
  (`zubr_link.Telemetry` has no such fields). `sync_read_present_current`/
  `sync_read_present_input_voltage`/`read_present_input_voltage` raise
  `NotImplementedError` — safe only because `Observer.observe_current` and
  `observe_voltage` both default to `False`, so nothing calls them in `main.py`'s
  normal flow. **Do not flip either flag on with this controller** — the overcurrent
  safety already has a fallback (`Scheduler._per_motor_currents`'s proxy estimate from
  position error, used automatically whenever `observe_current` is `False`), so this
  costs no safety coverage, just don't override the default.
- **`make voltage` doesn't work with this hardware** — see the comment atop
  `src/voltage.py`. It targets the old rustypot/XL330 bus directly (never went through
  `ControllerProtocol`), which is dead code now, and there's no zubr equivalent to
  build without more hardware documentation (does the STM32 expose pack voltage
  through its 2 generic named-variable read slots, and at which index?).
- **Velocity units are an unverified guess**: `_present_velocity()` assumes the
  STM32 reports motor velocity in the same ticks-per-revolution unit as position
  (i.e. "ticks/s") — `zubr.py` documents no units for this field, same caveat already
  flagged for gyro in `hw_state_stream.py`.

## Before running `make run` for the first time on this backend

Everything in [usage.md](../usage.md)'s safety note still applies unchanged — place
the robot on a stable surface or hold it securely; it enables torque and ramps to
neutral pose on start. Additionally specific to this changeover:

- The slot mapping (`MOTOR_TO_ID`), sign convention (`MOTOR_SIGN`), and IMU mount
  (`ZUBR_IMU_MOUNT_QUAT`) this controller relies on are exactly what was calibrated
  and verified via `hw_state_stream.py`/the sim viewer in the prior passes — if
  anything about the physical assembly changed since then (re-wiring, a swapped
  servo, a re-mounted IMU), re-verify with that read-only tool before trusting this
  one to write real commands.
- Expect the walk to behave differently from sim specifically because of the kp
  limitation above — that's a known, explained gap, not necessarily a new bug to
  chase if the gait looks stiffer or softer than expected.
