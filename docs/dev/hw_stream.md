# Visualizing the real robot's pose and IMU in a remote MuJoCo viewer

Same idea as the [master/slave physics stream](sim_stream.md), but the Orange Pi's
"master" side is the real robot instead of a physics step: it reads actual actuator
positions and the STM32's onboard IMU off the motor controller and rebroadcasts them
over the same wire format (plus an optional trailer, see below), so the unmodified
`sim_viewer_client.py` renders live hardware state instead of a simulation.

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
  a physics master; when the packet carries an IMU trailer, also draws the three arrows
  described below.
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

- Reads all 16 actuator positions plus the onboard gyro/accelerometer/quaternion once
  per tick over `/dev/ttyS2` ([`src/zubr_link.py`](../../src/zubr_link.py), formalizing
  the protocol prototyped in `~/zubr.py`: one fixed-size, CRC16-guarded
  request/response transaction per call).
- Converts raw encoder ticks to radians (16384 ticks = 2π rad) and writes them into a
  local, never-stepped `MjData`'s `qpos` — same joint layout as `scene.xml` — then
  sends it with `StateSender`, identical to what `sim_main.py --stream-to` sends, plus
  the raw IMU reading as an extra trailer that format now supports.
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

Three arrows, rooted at a point 10cm above the "imu" site (the best stand-in for "the
robot's head" the model currently has — its actual head body/joint is commented out of
`robot.xml`), drawn by `sim_viewer_client.py` whenever a packet carries the IMU
trailer:

| Color | Source | Meaning |
|---|---|---|
| blue | quaternion | The IMU's own reported "up," in world frame — visualizes orientation directly. Fixed length (it's a pure direction). |
| red | accelerometer | Raw accel vector rotated into world frame by that same orientation — should point straight down (opposite the blue arrow) when the robot is still and the fusion is healthy; divergence between them is a visible sanity signal. |
| green | gyro | Raw gyro vector (rotation axis), likewise rotated into world frame — near-zero length at rest, longer while turning. |

Caveats, both in `hw_state_stream.py`'s `_imu_from_telemetry()`:
- **Units are raw STM32 register values, not SI** — `~/zubr.py` documents no LSB scale
  for gyro/accel, and the quaternion's scale is sidestepped entirely by normalizing it
  before use (valid for a rotation regardless of the true scale factor). The red/green
  arrows are length-scaled and clamped purely to stay legible next to the blue one
  (`_ARROW_VECTOR_GAIN`/`_ARROW_VECTOR_MAX_LENGTH` in `sim_viewer_client.py`) — that is
  **not** a physical units claim.
- **Mounting is assumed identity** — this onboard IMU's rotation relative to the trunk
  is unknown (`constants.IMU_MOUNT_QUAT` is for the *separate* I2C BMI088
  `imu_reader.py` reads, not necessarily this one). If the blue arrow doesn't point up
  when the robot is genuinely upright, that's this assumption, not a bug.
- Only directions carry real information; don't wire this into anything that assumes
  calibrated units or a known mounting without fixing the above first.

### Wire format addition

[`state_stream.py`](../../src/sim/state_stream.py)'s `_IMU_TRAILER`: 10 float64s —
gyro `(x, y, z)`, accel `(x, y, z)`, quat `(w, x, y, z)` — appended after qpos/qvel.
Presence is inferred from packet length, not a header flag, so `sim_main.py`'s
IMU-less packets are untouched; `StateReceiver.last_imu` is `None` whenever the most
recently applied packet didn't carry one (which is also how the viewer knows to clear
stale arrows if a physics master takes over the port).
