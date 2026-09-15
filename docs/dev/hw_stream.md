# Visualizing the real robot's pose and IMU observation in a remote MuJoCo viewer

Same idea as the [master/slave physics stream](sim_stream.md), but the Orange Pi's
"master" side is the real robot instead of a physics step: it reads actual actuator
positions and the STM32's onboard BHI260 IMU off the motor controller, reduces the IMU
reading to the same two-channel observation `moves/walk.py`'s RL policy actually uses,
and rebroadcasts everything over the same wire format (plus an optional trailer, see
below), so the unmodified `sim_viewer_client.py` renders live hardware state instead
of a simulation.

This never sends motor commands (every poll leaves all 16 motors relaxed — see
`zubr_link.RELAX_POSITION`), so it's safe to run at any time, including while the
robot is being moved by hand or driven by another process.

## Files

- [`src/zubr_link.py`](../../src/zubr_link.py) — the serial link: struct layouts, CRC16,
  `ZubrLink.poll()`. Formalizes the protocol prototyped in `~/zubr.py` (not part of this
  repo) into something importable.
- [`src/hw_state_stream.py`](../../src/hw_state_stream.py) — the entry-point script
  described here.
- [`src/sim/state_stream.py`](../../src/sim/state_stream.py) — shared UDP wire format;
  gained an optional IMU trailer (see "Wire format" below) that only this script sends.
- [`src/sim/sim_viewer_client.py`](../../src/sim/sim_viewer_client.py) — unchanged for
  a physics master; when the packet carries an IMU trailer, also draws the two arrows
  described below.
- [`src/imu_reader.py`](../../src/imu_reader.py) — `imu_quat_to_body()` gained an
  optional `mount_quat` parameter (default unchanged) so `hw_state_stream.py` can reuse
  it with `ZUBR_IMU_MOUNT_QUAT` instead of the BMI088's `IMU_MOUNT_QUAT`.
- `src/constants.py`: added `ZUBR_IMU_MOUNT_QUAT` (identity placeholder, see below).
- `Makefile`'s `hw-stream` target.
- `pyproject.toml`: added `pyserial` to the `sim` dependency group (this script needs
  both `serial` and `mujoco`, so it rides along with `--group sim` like `sim-master`
  does) — pulled in by `uv sync --group sim`.

## Setup

```bash
cd ~/microban
uv sync --group sim   # installs mujoco + pyserial into .venv
```

## Procedure

1. **On your laptop**, start the slave viewer first:
   ```bash
   make sim-viewer PORT=9761
   ```

2. **On the Orange Pi** (`~/microban`), point the stream at your laptop:
   ```bash
   make hw-stream SLAVE=<laptop-ip>:9761
   # or directly:
   PYTHONPATH=src:vendor/bam uv run --group sim src/hw_state_stream.py --stream-to <laptop-ip>:9761
   ```

3. Move a joint by hand (or have the control loop move it) — the laptop's viewer
   should show it move within one poll (~20 ms at the default 50 Hz).

## What it does and doesn't do

- Reads all 16 actuator positions plus the onboard BHI260's gyro/quaternion once per
  tick over `/dev/ttyS2` ([`src/zubr_link.py`](../../src/zubr_link.py), formalizing
  the protocol prototyped in `~/zubr.py`: one fixed-size, CRC16-guarded
  request/response transaction per call). The BHI260 does its own onboard sensor
  fusion (like the BMI088 below, but chip-side rather than the software Madgwick
  filter `imu_reader.py` runs) — `quat_raw` arrives already fused, no extra filtering
  needed here.
- Converts raw encoder ticks to radians (16384 ticks = 2π rad) and writes them into a
  local, never-stepped `MjData`'s `qpos` — same joint layout as `scene.xml` — then
  sends it with `StateSender`, identical to what `sim_main.py --stream-to` sends, plus
  an IMU observation as an extra trailer that format now supports (see "IMU arrows"
  below).
- Does **not** run any physics (`mj_step` is never called) and does **not** command
  the motors — it only issues the STM32 protocol's mandatory read/relax directive to
  get a telemetry frame back.
- Has no base position/orientation sensor, so the trunk free joint is left at a fixed
  neutral spawn pose the whole run; only the 16 measured joint angles animate.
- Velocity (`qvel`) is sent as all-zero — only positions are read from the robot.

## STM32 slot mapping — now driven entirely by `MOTOR_TO_ID`

Which of the STM32's 16 telemetry/command slots is which joint is **not** a table
inside `hw_state_stream.py` — it's read straight from `MOTOR_TO_ID`'s values in
`src/constants.py`. `RobotController`'s rustypot/Dynamixel bus (the only other consumer
of those numbers) is dead code on this robot now, so `MOTOR_TO_ID[name]` was
repurposed to mean "STM32 slot index" instead of a Dynamixel bus ID — there's now a
single dict to edit, not two.

To calibrate a joint: move it by hand while `hw-stream` is running, watch which of the
16 numbers in the telemetry changes, and set `MOTOR_TO_ID[that_joint] = that_slot`.

`build_slot_map()` in `hw_state_stream.py` validates this on every run rather than
trusting it blindly — calibration is expected to happen incrementally, one joint at a
time, so it tolerates an unfinished table instead of refusing to run:
- A value outside `[0, 16)` (leftover from before this repurposing, or a typo) is
  logged and that joint is left out of the stream (`qpos` stays 0) until fixed.
- Two joints claiming the same slot number is logged and **both** are left out until
  the conflict is resolved (silently picking one would hide the mistake).
- Whatever remains resolves normally; a startup line reports how many of the 16 slots
  are still unaccounted for.

**Sign convention** (`MOTOR_SIGN` in `src/constants.py`) is a separate table, keyed by
joint name (not slot), reused as-is from the rustypot/Dynamixel backend since it's the
only convention this codebase has for these joint names — it hasn't been reverified
against the STM32/zubr board specifically.

**Zero offset**: assumes raw ticks map directly to the joint's mechanical zero (tick 0
== `NEUTRAL_POSE`'s 0 rad), i.e. no per-joint calibration table. If a joint looks
visibly offset from the real robot's pose at rest, that's the next thing to check.

## IMU arrows

The values streamed and drawn are **exactly the two channels `moves/walk.py`'s RL
policy observes** — gyro and gravity projected into body frame — computed
`hw_state_stream.py`-side (`_imu_from_telemetry()`) the *same way*
[`observer.py`](../../src/observer.py) computes them for the real BMI088:
`imu_reader.imu_quat_to_body()` (BHI260 quaternion → body frame, using a
BHI260-specific mount constant) then `imu_reader.quat_apply_inverse()` (world gravity
`(0,0,-1)` projected through that orientation). Raw accelerometer is read off the
board but not used — the RL pipeline this mirrors doesn't use it either.

Gyro gets its own, simpler correction: the chip reports it in its own local sensor
axes, same as the quaternion, but it's a body-fixed vector rather than a world-frame
reference — the mounting is a single *fixed* rotation regardless of how the whole
assembly is currently oriented, so it's `quat_apply_inverse(conjugate(mount), gyro_raw)`
rather than anything involving the (time-varying) `body_quat` gravity needs. (Verified
against a synthetic sensor/body pair before landing — the non-conjugated form gives a
plausible-looking but wrong vector, easy to get backwards by hand.)

Two arrows, rooted at a point above the "imu" site (the best stand-in for "the robot's
head" the model currently has — its actual head body/joint is commented out of
`robot.xml`), drawn by `sim_viewer_client.py` whenever a packet carries the trailer.
Both are body-frame vectors drawn **without any further rotation**: `hw_state_stream.py`
has no base-orientation sensor for the trunk itself, so the rendered trunk is always
upright (identity) — meaning body frame and this render's frame coincide, and a real
lean shows up as these arrows tilting away from straight-down/zero even though the
mesh doesn't move:

| Color | Source | Meaning |
|---|---|---|
| red | projected gravity | Unit vector; points straight down when the robot is level, leans with real tilt otherwise. |
| green | gyro | Raw gyro vector (rotation axis) — near-zero length at rest, longer while turning. |

Caveats, in `hw_state_stream.py`'s `_imu_from_telemetry()`:
- **Gyro units are raw STM32 register values, not SI** — `~/zubr.py` documents no LSB
  scale for it. Its arrow is length-scaled and clamped purely to stay legible next to
  the (always unit-length) gravity arrow (`_ARROW_GYRO_GAIN`/`_ARROW_GYRO_MAX_LENGTH`
  in `sim_viewer_client.py`) — not a physical units claim.
- **Quaternion scale** is sidestepped by normalizing it before use — valid for a
  rotation regardless of the true LSB scale factor.
- **Mounting** (`constants.ZUBR_IMU_MOUNT_QUAT`) has been calibrated against the real
  board (see below) — not an identity placeholder anymore. Don't wire the gyro channel
  into anything that assumes calibrated units without fixing the units issue above
  first; the gravity channel's direction is already meaningful (same caveats as the
  real RL pipeline's own `projected_gravity` has for the BMI088).

### Calibrating `ZUBR_IMU_MOUNT_QUAT`

`hw_state_stream.py --calibrate-mount N` captures this from the real board in two
poses rather than one — level alone isn't enough:

```bash
PYTHONPATH=src uv run --group sim src/hw_state_stream.py --calibrate-mount 100
```

1. **Level** — stand the robot level and still, any heading. This alone fixes "down"
   at rest (`mount = quat_raw` at that pose makes `body_quat` come out as identity —
   see the derivation in `solve_mount()`'s docstring and in `constants.py`'s
   `ZUBR_IMU_MOUNT_QUAT` comment) but **cannot** fix yaw: a gravity/accelerometer
   reading is mathematically blind to rotation about the vertical axis (rotating a
   vector about the axis it's already aligned with is the identity), so a level-only
   capture silently bakes in whatever heading the robot happened to face — which then
   makes the arrow's *tilt direction* wrong even though "down at rest" looks fine.
   (This is exactly what happened the first time this was calibrated: forward tilt
   showed as a rightward lean, a rightward tilt showed as a backward lean — both
   explained by one fixed, uncorrected +90° yaw baked in from that capture's heading.)
2. **Forward tilt** — from that same level pose, pitch the robot forward (front/chest
   down) by a clear amount, no roll or turning mixed in, and hold it. The *change*
   between the two captures is exactly the physical pitch applied, independent of
   whatever heading step 1 used — its rotation axis reveals how much yaw correction is
   needed to line "forward pitch" up with the model's own lateral axis, which a
   level-only capture can never supply.

Re-verify by tilting the robot forward/back/left/right and checking the red arrow
leans the same way the robot actually tilted. If step 2 warns about a large
out-of-plane component, the tilt wasn't clean pitch (roll/yaw crept in) — redo it.

### Checking the gyro (green) arrow

Gyro shares `ZUBR_IMU_MOUNT_QUAT` with gravity (see above), so nothing extra needs
calibrating — but it's a genuinely different reading (a *rotation rate*, not a static
tilt), so it needs its own check with the robot actually spinning, not just tilted.

With `hw-stream` running, **spin the robot about one axis at a time** and check the
green arrow's direction against the right-hand rule (curl your right hand's fingers in
the direction of the spin — your thumb points the way the arrow should):
- **Yaw** the robot (spin it flat, like turning in place): counterclockwise viewed
  from above should show the arrow pointing straight **up**; clockwise, straight
  **down**.
- **Pitch** it (rock it nose-down then nose-up, i.e. rotate about the left-right
  axis): tipping the nose *down* should show the arrow pointing out the robot's
  **left** side; nose *up*, its **right** side. (Opposite of what you might expect if
  you picture the rotation "arrow" as pointing where the front is heading — it's the
  spin *axis*, not the direction of motion.)
- **Roll** it (rock it side-to-side, rotating about the front-back axis): rolling so
  the right side dips down should show the arrow pointing out the **front**; left side
  down, out the **back**.

The arrow should stay near-zero length whenever the robot is held still (any nonzero
reading at rest is just noise) and grow with faster spinning — it only needs to point
the *right way*, since its length is an uncalibrated, arbitrarily-scaled raw reading
(see the caveats above), not an actual rotation rate.

### Wire format addition

[`state_stream.py`](../../src/sim/state_stream.py)'s `_IMU_TRAILER`: 6 float64s —
gyro `(x, y, z)`, projected gravity `(x, y, z)` — appended after qpos/qvel. Presence is
inferred from packet length, not a header flag, so `sim_main.py`'s IMU-less packets
are untouched; `StateReceiver.last_imu` is `None` whenever the most recently applied
packet didn't carry one (which is also how the viewer knows to clear stale arrows if a
physics master takes over the port).
