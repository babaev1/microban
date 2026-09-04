# Debugging the sim walk move: overcurrent trips, instant falls, and one self-inflicted regression

This documents a debugging session on `WalkMove` (`src/moves/walk.py`) in `make sim`,
starting from "overcurrent safety triggers as soon as walk is enabled" and ending at a
robot that stands, walks at moderate commanded speed, and falls over in a physically
sane way if pushed too fast. Kept mainly for the last section — the mistake that cost
the most time and is easy to repeat.

## Bugs found and fixed, in the order they were found

### 1. Action/observation DoF order mismatch (`walk.py`)

`WalkMove` used a hardcoded `OBSERVATION_DOF_ORDER` list from `constants.py` to build
the policy's observation vector and to decode its action vector back into per-joint
targets. That list ordered joints shoulders-first, hips-last. The ONNX model's actual
training order (`joint_names` in its metadata) is hips-first, shoulders-last. Result:
`action[i]` was applied to the wrong joint — e.g. a shoulder-sized action landing on a
hip, producing targets like `2.5 rad` (143°) on a leg joint.

**Fix:** `WalkMove.__init__` now reads `self._dof_order` straight from the ONNX
metadata's `joint_names` and uses it everywhere (observation building, action decoding,
`_last_action` sizing). `OBSERVATION_DOF_ORDER` was deleted from `constants.py` — it's
dead now, and keeping it around would just invite the same drift again for a future
retrained agent.

### 2. One-tick pose jump on walk start (`walk.py`)

`WalkMove.on_start` used to just set Kp and immediately go `ACTIVE`, so `step()` jumped
`command.target_angles` straight from whatever pose preceded it (typically
`NEUTRAL_POSE`) to the RL policy's trained `default_pose` in a single 20ms tick — a
large, simultaneous multi-joint step.

**Fix:** `on_start` now linearly ramps every joint from wherever it was to
`default_pose` over `start_lerp_duration` (1.5s), matching the lerp pattern already used
in `SquatMove.on_start`/`on_stop`.

### 3. Kp dropped before reaching the trained pose (`walk.py`)

Softening to `KP_RL` used to happen on the very first `STARTING` tick, i.e. *before* the
ramp above existed / before the ramp finished — so the robot went compliant while still
away from the pose the RL policy expects to start from, and gravity could yank joints
around during the transition.

**Fix:** the Kp switch now happens only once the ramp reaches `t >= 1.0`, i.e. right as
control hands off to `step()`, at the correct starting pose.

### 4. Two latent `KeyError` crashes from the "roki" motor-set change

An earlier commit ("changing to roki in sim") dropped `left_elbow`/`right_elbow`/`head`
from `MOTOR_TO_ID` and `NEUTRAL_POSE`, but didn't touch two other moves that still
assumed those keys existed:

- `SquatMove._UPPER_JOINTS` still listed `"left_elbow"`/`"right_elbow"` →
  `NEUTRAL_POSE[name]` raised `KeyError` the instant squat was enabled. Fixed by
  dropping them from the list (`squat.py`).
- `RotateHeadMove` unconditionally wrote `command.target_angles["head"]` →
  `Scheduler._send_to_motors`'s `MOTOR_TO_ID[name]` lookup would have raised the same
  kind of `KeyError` the instant the head move was enabled (not yet hit when found;
  fixed proactively). Fixed by guarding all three writes with
  `if "head" in MOTOR_TO_ID` (`rotate_head.py`).

### 5. Overcurrent safety diagnostics (`scheduler.py`)

`Scheduler._check_overcurrent` now prints the top 6 per-motor contributors (name,
estimated current, target, position, raw error) whenever it trips, not just the total.
This was the tool that made every later diagnosis possible — worth keeping permanently,
it's cheap and only fires on an actual trip.

### 6. Live balance telemetry (`walk.py`)

`DEBUG_PRINT` (default `False`) gates two throttled prints: one inside `on_start`'s ramp
and one inside `step()`, both showing `projected_gravity`, `gyro`, and a computed lean
angle in degrees. `step()`'s print only fires *after* a policy run succeeds, which
turned out to be diagnostic in itself — see below. Flip `DEBUG_PRINT = True` when
chasing balance/gait issues again.

## The big one: a self-inflicted `IMU_MOUNT_QUAT` regression

After fixing the four bugs above, the robot still instantly tipped over — `WalkMove`'s
own fall-check (`projected_gravity[2] > -0.5`) was firing on literally the first tick
`step()` ever ran, meaning the robot was already past the fall threshold before the RL
policy touched it even once.

The investigation chased this as a "gravity axis is wrong" bug: `IMU_MOUNT_QUAT`
(`constants.py`) converts the raw MuJoCo `orientation` sensor reading into the trunk's
body frame, and at what was believed to be the neutral standing pose it produced
`projected_gravity ≈ (1, 0, 0)` — gravity on the trunk's X axis instead of Z, which a
policy trained on a Z-up convention reads as "robot lying on its side" and reacts to
with (physically real, in sim) violent corrective actions. `IMU_MOUNT_QUAT` was changed
from `(0.5, -0.5, -0.5, 0.5)` to `(0.70710678, 0, 0, 0.70710678)` to compensate, and a
battery of headless MuJoCo+BAM+ONNX repro scripts run outside the interactive viewer
"confirmed" this fixed it, self-consistently, several times over — including while
separately tuning `KP_RL` and `OVERCURRENT_CUTOFF_A` (both got bumped up chasing a
"robot is too soft to hold the crouch" symptom this same regression was causing) and
while re-verifying the new value's sign convention across a wide sweep of simulated lean
angles.

**The actual bug**: `SPAWN_TRUNK_QUAT` in `sim/mujoco_controller.py` is *not* a fixed
literal. It's derived from `NEUTRAL_POSE`:

```python
SPAWN_TRUNK_PITCH = -NEUTRAL_POSE["left_hip_pitch"]
SPAWN_TRUNK_QUAT = (math.cos(SPAWN_TRUNK_PITCH / 2), 0.0, math.sin(SPAWN_TRUNK_PITCH / 2), 0.0)
```

Every one of the headless repro scripts hardcoded `SPAWN_TRUNK_QUAT = (0.70710678, 0,
0.70710678, 0)` — a stale literal that would only be correct if
`NEUTRAL_POSE["left_hip_pitch"]` were ±90°. In the current (roki) `NEUTRAL_POSE` it's
`0°`, so the real spawn orientation is identity, not a 90° rotation. Because every test
script made the *same* wrong assumption, they were internally consistent with each other
while being wrong relative to the actual simulator — nothing caught it.

Recomputing `IMU_MOUNT_QUAT`'s correctness properly (deriving `SPAWN_TRUNK_QUAT` from
the real formula instead of a literal) showed the *original* value,
`(0.5, -0.5, -0.5, 0.5)` — the mjcf `imu` site's own local quat in `robot.xml` — already
gives exactly `projected_gravity = (0, 0, -1)` at a true upright spawn. It was correct
the whole time. `IMU_MOUNT_QUAT`, `KP_RL`, and `OVERCURRENT_CUTOFF_A` were all reverted
to their original values (`(0.5,-0.5,-0.5,0.5)`, `125`, `15.0`).

**Lesson for next time**: `SPAWN_TRUNK_QUAT` depends on `NEUTRAL_POSE["left_hip_pitch"]`.
Any headless repro script touching robot orientation must either import/recompute it
from that formula, or import it directly from `sim.mujoco_controller`, never hardcode
the literal — it silently goes stale the moment `NEUTRAL_POSE` changes (as it did in the
roki swap).

## State after this session

- Ramp into the crouch: stays within ~0.3° of perfectly upright the entire 1.5s.
- Walking at moderate commanded velocity (`vx` up to ~0.3-0.4): stable, current draw
  mostly under the 15A cutoff, gyro shows normal gait oscillation.
- Pushing `vx` up toward `0.5`: lean grows progressively and the robot eventually falls
  forward — this looks like a genuine gait-robustness/training limit of `walk.onnx`
  at that commanded speed, not a code bug. The fall-check and overcurrent safety both
  fire in the right order when it happens (fall detected first, current spike from legs
  snapping toward `NEUTRAL_POSE` while collapsing catches the overcurrent trip
  afterward — exactly the scenario `OVERCURRENT_CUTOFF_A`'s design comment describes).

## Open items

- `OVERCURRENT_CUTOFF_A = 15.0` and `KP_RL = 125` are the original, pre-session values —
  not independently re-validated against real hardware in this session, just restored
  after ruling out the regression that made them look wrong.
- `IMU_MOUNT_QUAT` is shared between sim and the real BMI088 driver
  (`src/imu_reader.py`). It was re-derived from sim's own `imu` site definition, not
  re-verified against the physical IMU mounting — worth checking before trusting it
  unattended on real hardware, per the comment in `constants.py`.
- The `vx≈0.5` fall is unexplored beyond confirming it's plausible/physical — whether
  it's worth retraining, gain-tuning, or just documenting as a speed limit is an open
  call.
