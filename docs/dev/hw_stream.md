# Visualizing the real robot's pose in a remote MuJoCo viewer

Same idea as the [master/slave physics stream](sim_stream.md), but the Orange Pi's
"master" side is the real robot instead of a physics step: it reads actual actuator
positions off the STM32 motor controller and rebroadcasts them over the same wire
format, so the unmodified `sim_viewer_client.py` renders live hardware state instead
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

- Reads all 16 actuator positions once per tick over `/dev/ttyS2`
  ([`src/zubr_link.py`](../../src/zubr_link.py), formalizing the protocol prototyped in
  `~/zubr.py`: one fixed-size, CRC16-guarded request/response transaction per call).
- Converts raw encoder ticks to radians (16384 ticks = 2π rad) and writes them into a
  local, never-stepped `MjData`'s `qpos` — same joint layout as `scene.xml` — then
  sends it with [`sim/state_stream.py`](../../src/sim/state_stream.py)'s `StateSender`,
  identical to what `sim_main.py --stream-to` sends.
- Does **not** run any physics (`mj_step` is never called) and does **not** command
  the motors — it only issues the STM32 protocol's mandatory read/relax directive to
  get a telemetry frame back.
- Has no base position/orientation sensor, so the trunk free joint is left at a fixed
  neutral spawn pose the whole run; only the 16 measured joint angles animate.
- Velocity (`qvel`) is sent as all-zero — only positions are read from the robot.

## Known placeholders — verify before trusting the pose

- **STM32 motor slot order** (`SLOT_TO_JOINT` in
  [`src/hw_state_stream.py`](../../src/hw_state_stream.py)): the physical wiring order
  of the 16 motor slots in zubr.py's protocol hasn't been confirmed against the real
  board. It's a plain, explicitly-indexed tuple — edit it directly, entry `i` is slot
  `i` on the board. **It is intentionally independent of `MOTOR_TO_ID` in
  `constants.py`**: that dict's values are Dynamixel-style bus IDs for a different
  controller (`RobotController`/rustypot) and its key order is just source-file order —
  neither has anything to do with STM32 wiring, so editing `MOTOR_TO_ID`'s numbers has
  no effect on this mapping. Verify by moving one joint by hand and checking which
  slot's telemetry position changes, then put that joint's name at that index.
- **Sign convention** (`MOTOR_SIGN` in `src/constants.py`): reused as-is from the
  rustypot/Dynamixel backend (`RobotController`) since it's the only convention this
  codebase has for these joint names, but it hasn't been reverified against the
  STM32/zubr board specifically.
- **Zero offset**: assumes raw ticks map directly to the joint's mechanical zero (tick
  0 == `NEUTRAL_POSE`'s 0 rad), i.e. no per-joint calibration table. If a joint looks
  visibly offset from the real robot's pose at rest, that's the first thing to check.

### Live data point (2026-09-09, board connected on `/dev/ttyS2`)

A one-off poll (`ZubrLink().poll()`) returned real, stable telemetry — confirming the
link itself works — but with a pattern worth checking before trusting the mapping:

```
positions: (6693, 6713, 7358, 7373, -16, 16, -29, -41, -14, -22, 12, 31, -4, -58, -31, 17)
```

Slots 0–3 read large, stable values (6693–7373 ticks ≈ 147–162°); slots 4–15 all sat
within ±60 ticks of zero (≈ ±1.3°). Either slots 0–3 are joints with a genuinely large
zero-offset (not yet in a table anywhere), or `SLOT_TO_JOINT`'s assumed order is wrong
and those 4 slots aren't the joints it currently thinks they are. Worth resolving
before relying on the visualized pose for anything beyond "is it moving."
