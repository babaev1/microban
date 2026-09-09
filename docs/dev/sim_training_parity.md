# Why `make sim` walks badly while mjlab `play` walks well

`walk.onnx` (exported from `mjlab_roki6` run `2026-08-16_08-45-39`, `model_14999.pt`) is
balanced and stable when replayed inside its training environment:

```
uv run play Mjlab-Velocity-Microban --checkpoint-file \
  /home/a/mjlab_roki6/logs/rsl_rl/mjlab_microban_velocity/2026-08-16_08-45-39/model_14999.pt
```

The same policy in `make sim` falls over within ~1.5 s. This documents the root cause,
the evidence for it, and what still needs doing. The short version: **the policy and the
whole observation/action pipeline in this repo are correct; `make sim`'s physics is not.**

**Status: the sim half (Bugs 1, 3, 4 and the SKS2401 motor model half of Bug 2) is fixed
and verified — see "Sim half — done" below.** `KP_RL` in `constants.py` is now `7334`
(changed directly by the user after seeing the numbers below, not by this doc's original
recommendation, which had held it back) — see "the `KP_RL` question" under the hardware
half for what that decision does and does not cover: it's correct for `make sim`, but
`main.py` writes the same constant straight to real servo registers, and the overcurrent
proxy meant to protect those servos still models the wrong motor family. **Do not run
`make run` against real hardware until that proxy is rewritten for the SKS2401** (see the
hardware TODO list).

## Method

The diagnosis is anchored on ground truth dumped from the training environment itself, so
nothing here rests on re-deriving what training "probably" did. The dump script lives in
the other repo (untracked): `/home/a/mjlab_roki6/dump_groundtruth.py`. It loads the play
env plus the checkpoint, rolls out headless with `vx = 0.3` fixed, and saves per-step
`obs`, `act`, `qpos`, `qvel`, `projected_gravity`, `root_z`, `root_quat`, `command` to
`/tmp/groundtruth.npz`.

Reference behaviour from that rollout: **300 steps (6 s), no termination, lean mean 7.6°
max 14.1°, `root_z` mean 0.2440, max `|action|` 2.48 rad.**

This is the number to compare any sim change against. It is also worth re-reading the
lesson at the end of `walk_debugging.md` before writing any new test harness: a battery of
self-consistent scripts that all share one wrong assumption will confirm each other
indefinitely. Dumping from the real training env is what avoids that.

## Ruled out — verified correct, do not re-investigate

| what | evidence |
| --- | --- |
| ONNX export fidelity | `ONNX(obs)` vs the checkpoint's own action: max abs diff **1.4e-06** |
| Observation normalisation | embedded in the graph: `Sub(obs, obs_normalizer._mean)` → `Div` → `Gemm` |
| Observation term order | metadata `observation_names` = `base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command`, term dims `(3,3,16,16,16,3)` = 57, confirmed against `env.observation_manager.active_terms["actor"]` |
| `build_observation` slice layout | `joint_pos` diff 4.0e-04, previous action **0.0**, command **0.0** |
| `projected_gravity` / `joint_vel` slices | diffs of 1.2e-01 / 1.13e+01 are training's *own* delay buffers, not a layout error |
| Joint / action ordering | metadata `joint_names` == `robot.joint_names` == action term `_target_names`, all right-leg → left-leg → shoulders |
| `action_scale` and offset | `scale = 1.0`, offset = `default_joint_pos`; `target = default + action` is correct |
| Default pose | `hip_pitch -15°, knee +30°, ankle_pitch -15°`, matches `LEARNING_ZERO_JOINT_POS` |
| Control rate | training `decimation=4 × 0.005 s` = 50 Hz; `make sim --hz 50` matches |
| `IMU_MOUNT_QUAT` | `(0.5, -0.5, -0.5, 0.5)` yields exactly `projected_gravity = (0, 0, -1)` upright — correct, as `walk_debugging.md` already concluded |
| IMU frames | training's `base_ang_vel` is the gyro at the `imu` **site** (site frame) and `projected_gravity` is **trunk** frame; the playout reproduces both, including the frame mix |
| Robot model | `src/model/mjcf/robot.xml` is **byte-identical** to `mjlab_roki6/.../Roki_4_MJCF/robot.xml` |
| Velocity command limits | `VX_MAX 0.7`, `VY_MAX 0.3`, `VTHETA 1.5/3.0` match the stage-1 curriculum ranges |

`MOTOR_SIGN` is unused in the sim path (only `robot_controller.py` reads it), so it cannot
affect `make sim` either way.

## Bug 1 — BAM torques are written into MuJoCo `<position>` actuators — FIXED

**The dominant bug.** `bam.mujoco.MujocoController.update()` writes a **torque** into
`data.ctrl`. That is only valid if the actuators are `<motor>` type.

During training, `bam.mjlab` rewrites them before compiling: `create_motor_actuator` /
`mjact.set_to_motor()` sets `gaintype=FIXED`, `gainprm[0]=1`, `biastype=NONE`. The ONNX
metadata confirms it — `joint_stiffness = 1.000` and `joint_damping = 0.000` for all 16
joints.

`mujoco_controller.py` calls `MjModel.from_xml_path()` and gets no such rewrite, so the
actuators stay as `robot.xml` declares them:

```xml
<default class="SKS2401">
  <joint damping="0.041" frictionloss="0.013" armature="0.023" />
  <position kp="0.78" kv="0.0" forcerange="-2.5 2.5"/>
</default>
```

Measured on the compiled model: `gainprm[0] = 0.78`, `biasprm = [0, -0.78, 0]`,
`forcerange = ±2.5`, and `ctrlrange` = **the joint's angle range** (because the actuators
use `inheritrange="1"`). So the applied torque is

```
force = 0.78 * (tau_BAM - q)      clipped to +-2.5 Nm
```

Holding the policy's default pose, measured `ctrl` vs applied force:

| joint | `ctrl` (BAM torque) | actually applied |
| --- | --- | --- |
| right_knee | **+0.138 Nm** | **-0.281 Nm** |
| left_knee | +0.134 Nm | -0.284 Nm |
| right_hip_pitch | -0.336 Nm | -0.109 Nm |
| right_ankle_pitch | -0.405 Nm | -0.172 Nm |

The knee torque is **sign-inverted** (`0.78 * (0.138 - 0.497) = -0.280`). For any joint
held away from zero — precisely the crouch joints the policy lives in — the `-0.78 * q`
term dominates and drags the leg toward `q = 0`.

This is the real cause of the "robot is too soft to hold the crouch" symptom that
`walk_debugging.md` chased into `IMU_MOUNT_QUAT`, `KP_RL` and `OVERCURRENT_CUTOFF_A`. Held
statically it looks survivable — the joints just settle short of target:

| joint | target | as-is `<position>` | as `<motor>` |
| --- | --- | --- | --- |
| hip_pitch | -15.0° | -12.4° | -14.6° |
| knee | +30.0° | +27.0° | +32.1° |
| ankle_pitch | -15.0° | -12.5° | -14.6° |

Dynamically it is fatal.

### Fix

Load via `MjSpec`, convert, then compile. `set_to_motor()` **is** available in microban's
MuJoCo (3.12.0), so this is the identical call training makes. The force limit must be
derived from the BAM model exactly as `bam.mjlab` does (`force_limit = vin * kt / R`),
which means the BAM model has to be built *before* the MuJoCo model — move the
`bam_load_model(...)` block above the model load, keeping `BamController(...)` where it is.

```python
force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
spec = mujoco.MjSpec.from_file(mjcf_path)
for act in spec.actuators:
    if act.name not in MOTOR_TO_ID:
        continue
    act.set_to_motor()
    act.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    act.forcelimited = True
    act.forcerange = (-force_limit, force_limit)
    act.ctrllimited = True
    act.ctrlrange = (-force_limit, force_limit)
self._model = spec.compile()
```

Verified after this change: **`max |applied_force - ctrl| = 0`** and the crouch holds to
0.1° instead of drifting 3° toward zero.

Do this in code, not by editing the XML: `robot.xml` being byte-identical to training's
copy is a useful invariant (it is how the robot model was ruled out above), and the force
limit is a runtime value from the BAM model anyway.

Training also zeroes `joint.damping` / `joint.frictionloss` and sets `joint.armature` from
BAM. `bam.mujoco` already overwrites `dof_frictionloss` / `dof_damping` on every `update()`
and sets `dof_armature` in its constructor, so only the first tick differs — negligible.
Training's `stiff_frictionloss=True` does **not** need replicating: it is a MuJoCo-Warp
substitute for the missing noslip solver and is irrelevant on CPU MuJoCo.

## Bug 2 — wrong motor model: XL330 instead of SKS2401 — FIXED (sim side)

`mujoco_controller.py` loads `bam_load_model(motor_name="xl330", model="m6")`. The Roki4
servos are **SKS2401**, and training fits them with `roki4_actuator/m3.json` (steel) and
`m3_al.json` (aluminium), split by joint group:

- **steel** (`m3.json`) — `shoulder_pitch`, `shoulder_roll`, `knee`
- **aluminium** (`m3_al.json`) — `hip_yaw`, `hip_roll`, `hip_pitch`, `ankle_pitch`, `ankle_roll`

| | kt | R | armature | current_limit | ceiling `vin·kt/R` | current-limited |
| --- | --- | --- | --- | --- | --- | --- |
| microban sim (xl330 m6) | 0.36601 | 2.81139 | 0.0018077 | 6.8 (overridden) | **1.43 Nm** | 2.5 (XML forcerange) |
| training steel (m3) | 1.53801 | 0.10852 | 0.0185654 | 6.78177 | 155.90 Nm | 10.43 Nm |
| training alu (m3_al) | 2.01433 | 1.00426 | 0.0099169 | 3.19651 | 22.06 Nm | 6.44 Nm |

A **5–7× torque deficit**, plus ~10× too little reflected inertia. Training's
`actuator_forcerange` reads `±155.895` / `±22.064`; `dof_armature` in joint order is
`0.00992` for hips/ankles and `0.01857` for knees/shoulders. The policy commands knee
targets up to 172° and `|action|` up to 2.48 rad — it needs that torque.

Other training values to match: `vin = 11.0` (`ROKI4_VIN` default; note the `Roki4Actuator`
class default of 12.0 is overridden), `kp_fw = 7333.9`.

### The packaging problem

This is not a one-line config change. microban's `sim` group depends on
`better-actuator-models` (PyPI), whose `bam` has **no SKS2401 actuator at all**:

```
KeyError: 'SKS2401'   # bam/model.py: actuators[data["actuator"]]
```

Its `params/` contains only `erob80_100, erob80_50, feetech_sts3215_7_4V, mx106, mx64,
xl320, xl330`. Where the roki4-capable bam actually lives:

- **`/home/a/mjlab_roki6/.venv/lib/python3.13/site-packages/bam/`** — has `roki4/actuator.py`,
  `roki4/testbench.py`, `params/roki4_actuator/{m3,m3_al}.json`, and `"SKS2401"` registered
  in `actuators.py`. **No `dist-info`** — it was hand-placed, not installed. Files dated
  Jul 13 – Aug 16.
- **`/home/a/bam/`** — a real git clone of `github.com/Rhoban/bam`, but **no remote ref
  contains `bam/roki4/actuator.py`**: all 27 remote refs checked individually, zero hits,
  and none has `SKS2401` in `bam/actuators.py` either. Only `params/roki4_actuator/`
  exists, added locally at the top level on Aug 9. Its checked-out `main` has neither the
  `roki4` module nor the registration.
- `mjlab_roki6/pyproject.toml` declares `bam = { git = "ssh://git@github.com/rhoban/bam", branch = "doc" }`,
  but `origin/doc` has no roki4 module either — consistent with that GitHub branch being
  outdated.

**So there is no git ref to depend on.** Resolved by vendoring: the modules
`bam.model`/`bam.mujoco` actually reach (transitively) plus the two SKS2401 JSONs are
copied into `vendor/bam/bam/` (172 KB, see `vendor/bam/README.md` for exactly what was
and wasn't copied and why). `make sim`/`make sim-master` put `vendor/bam` on
`PYTHONPATH` directly — it is not installed as a uv dependency, so `pyproject.toml`'s
`sim` group no longer lists `better-actuator-models` at all.

## Bug 3 — `SPAWN_TRUNK_Z` is 74 mm too low — FIXED

`mujoco_controller.py` sets `SPAWN_TRUNK_Z = 0.1721`, commented as "soles just touch the
floor (0.1701), measured flatness 0.1 mm across all 12 foot collision geoms". Measured on
the *current* (Roki4) robot:

- required trunk z, all-zero `NEUTRAL_POSE`: **0.2466**
- required trunk z, policy default pose: **0.2423**
- training `HOME_FRAME` pos z: 0.255

At spawn the robot is therefore **74.5 mm inside the floor with 64 contacts**
(`worst contact dist = -0.0745`), gets ejected upward at 1.4 m/s to z = 0.2856, and only
settles at 0.2465 after ~0.4 s. Every `make sim` run launches the robot at t = 0.

Same staleness class the previous session warned about — the value is left over from the
pre-Roki4 robot. The surrounding comment is stale too: it explains a "-10 deg" hip pitch
that matches neither `NEUTRAL_POSE` definition.

Fixed to `0.2486` (0.246617 flush + 2 mm clearance, matching the original convention).
Verified: constructing `MuJoCoController` now spawns with `ncon = 0` and the robot settles
to 0.2465 within a fraction of a second, no ejection.

### Related: `NEUTRAL_POSE` is defined twice — left alone, deliberately

`constants.py` defines `NEUTRAL_POSE` twice; the second (all zeros) wins. The dead first
one (`hip_pitch -15°, knee 30°, ankle_pitch -15°`) is **exactly the policy's trained
default pose**. This was left untouched for the sim-half fix — it's a design decision
(spawn at the policy's stance vs. spawn upright and let `WalkMove`'s ramp carry it there),
not a bug, and `SPAWN_TRUNK_Z` above was fixed for the pose that's actually active
(all-zero) rather than pre-empting that decision.

## Bug 4 — the delay stack sits at and past training's worst case — FIXED

mjlab's lag unit is the **control step**, not the physics step (`ObservationTermCfg`:
"Convert to ms: lag * (1000 / control_hz)"), so 1 lag = 20 ms.

| channel | training | `make sim` default |
| --- | --- | --- |
| `joint_pos` | 0 – 0 | 0 ticks |
| `joint_vel` | 0 – 1 | 1 tick |
| `base_ang_vel` (gyro) | 0 – 3, resampled every 64 steps | **3 ticks fixed** (60 ms) |
| `projected_gravity` (quat) | 0 – 3, resampled every 64 steps | **4 ticks fixed** (80 ms) |
| actuation | none (`BamActuatorCfg` has no delay set) | **2 physics steps** (10 ms) |

So the quat delay *exceeds* the trained maximum, every channel is pinned at its worst case
simultaneously and constantly rather than sampled, and there is an actuation delay training
never had.

**Update: fixed properly, not just re-tuned.** The first pass at this (see the closed-loop
A/B further down) only changed the *fixed* gyro/quat values from 3/4 down to 2/2 ticks — a
constant compromise, not a reproduction of what training actually does. `_DelayBuffer` in
`mujoco_controller.py` now samples a fresh lag uniformly from `[0, max_lag]` (every delayed
term training uses has `delay_min_lag=0`, so min is hardcoded rather than exposed),
resampled every tick by default or every 64 ticks for gyro/quat specifically — exactly
matching `delay_update_period`. `sim_main.py`'s `--delay-*` flags are now upper bounds, not
fixed values, and their defaults are training's own numbers verbatim: gyro/quat `3`
(resampled every 64 ticks), vel `1`, pos/act `0`. See "Sim half — done" below for the
verification — pinning every channel at a constant worst case, not the worst-case value
itself, was what actually destabilized the walk.

## Bug 5 — solver settings (Euler, not implicitfast) — FIXED

Found while investigating a follow-up report: even after Bugs 1-4, `make sim`'s gait
still looked "sharper and shakier" than `mjlab_roki6`'s own `play`/`play_keyboard.py`,
with visible unexplained rotation while walking straight. Two candidates were
investigated; one held up, one didn't (see "Dead end" below) — recorded here rather
than deleted, since the reasoning error is worth not repeating.

### Solver settings — real, fixed

`robot.xml` has no `<option>` block, so the compiled model uses MuJoCo's own defaults —
confirmed directly (`MjModel.from_xml_path`): `integrator=Euler`, `timestep=0.002`,
`iterations=100`, `ls_iterations=50`, `ccd_iterations=35`. Training's `SIM_CFG`
(`microban_velocity_env_cfg.py`) only overrides `timestep=0.005`, `iterations=10`,
`ls_iterations=20`, `ccd_iterations=100` — the rest come from mjlab's `MujocoCfg`
dataclass defaults, confirmed by reading `mjlab/sim/sim.py`: **`integrator="implicitfast"`**
(not Euler), `cone="pyramidal"`, `jacobian="auto"`, `solver="newton"`, `impratio=1.0` (the
last four already happened to match MuJoCo's own compiled defaults, so only integrator
and the four iteration/timestep fields were actually wrong).

Euler vs implicitfast is the standout mismatch: implicitfast exists specifically to be
more stable than explicit Euler for damped, stiff dynamics — exactly this system's
profile (joint friction recomputed every tick by BAM, SKS2401 force limits up to
155.9 Nm). Running that combination under explicit Euler is a plausible direct cause of
a jerkier gait and spurious per-step yaw noise.

Fixed by setting `spec.option.*` before compiling (not in the XML, same reasoning as
Bug 1 — keeps `robot.xml` byte-identical to training's copy): `timestep=0.005`,
`integrator=implicitfast`, `iterations=10`, `ls_iterations=20`, `ccd_iterations=100`,
plus the other five fields set explicitly even though unchanged, so this can't silently
drift if a MuJoCo version ever changes its own defaults. `steps_per_tick` (`round(0.02 /
timestep)`) becomes `4` at the new timestep — exactly training's own `decimation=4`,
confirming the two were always meant to line up.

### Dead end — foot friction looked inverted, but the effect never actually happens

Training's `FULL_COLLISION` (`roki_4_constants.py`) is written to give feet
`condim=3, priority=1, friction=(1.0,)` (real, grippy foot-ground contact) and
everything else `condim=1` (frictionless). Its foot-specific regex,
`r"^(left|right)_foot_collision$"`, requires an **exact** match on the whole geom name —
but the real geoms are `right_foot_collision_1` through `_6` (numbered), so it never
matches a single one. Measured directly on the live training env
(`env.sim.mj_model.geom_condim`/`geom_priority`): feet end up `condim=1, priority=0`;
non-foot geoms are simply never touched by `FULL_COLLISION` at all (their names don't
contain `"_collision"` as a substring — e.g. `Trunk_primitive1` — so they keep the XML's
own untouched default of `condim=3`, already matching microban).

This looked like a strong, direct explanation — a policy trained with feet that slide
completely freely, suddenly given real grip at every footfall — so a first pass set
`condim=1` on microban's 12 foot geoms too, matching the (accidental) training value.

**That fix was inert — the reasoning stopped one step too early.** A geom's own
`condim` isn't what determines a contact's actual behavior; MuJoCo resolves a contact
*pair* to `max(condim_1, condim_2)` when both geoms have equal `priority` (both are 0
here, untouched by either project). The floor/terrain geom is `condim=3` in **both**
projects — never touched by anything discussed above. So:

```
max(foot condim=1, floor condim=3) = 3   →   full friction, regardless of the foot's own condim
```

Verified by inspecting the actual resolved contacts (`data.contact[i].dim`, not just the
static geom attribute) in **both** projects after real physics steps: every foot-ground
contact resolves to `dim=3` in training too. The feet were never frictionless in
training — the geom-level attribute was real (and the regex bug is real), but it never
once affected an actual foot-ground contact, in training or here. Setting `condim=1` on
microban's foot geoms to match was consequently a no-op — reverted rather than left in
as misleading dead code with an incorrect justifying comment.

**Lesson**: a geom's own contact parameters are not its resolved contact behavior — check
`data.contact[i].dim`/`.friction`, the actual output of MuJoCo's pairwise resolution, not
just `model.geom_condim[i]`, before drawing a physical conclusion from it.

### Verification

Closed-loop, real `walk.onnx`, `KP_RL=7334`, `vx=0.3`, 5 stochastic trials, 10 s each,
training-matched delays (`2/2/1/2` upper bounds, per Bug 4), **with the solver-settings
fix only** (the foot-condim change was reverted, see above — these numbers still stand,
since the solver fix was applied in the same test run and the condim change had no
effect to begin with):

```
trial 0: walked full 10s  lean mean=5.2 max=11.3  gyro_z rms=54.2 deg/s  x=+2.56m y=-0.38m
trial 1: walked full 10s  lean mean=6.1 max=12.7  gyro_z rms=52.7 deg/s  x=+2.30m y=-1.39m
trial 2: walked full 10s  lean mean=5.3 max=12.5  gyro_z rms=56.9 deg/s  x=+2.47m y=-0.35m
trial 3: walked full 10s  lean mean=5.3 max=12.3  gyro_z rms=53.7 deg/s  x=+2.22m y=-1.21m
trial 4: walked full 10s  lean mean=5.5 max=12.6  gyro_z rms=56.4 deg/s  x=+2.43m y=-0.99m
```

Lean `max` tightened noticeably (was 11.0-18.0° across trials before this fix, now a
consistent 11.3-12.7°) — less variance, i.e. a visibly steadier gait.

**Important, reassuring cross-check**: `gyro_z rms` (~53-57 deg/s here) is *not* elevated
relative to training. The same metric measured directly on the ground-truth training env
at the identical command (`vx=0.3, vtheta=0`, real checkpoint, real physics) came out to
**67.8 deg/s** — *higher* than microban's post-fix number. So the residual gyro
oscillation visible in `make sim` is not a sim bug at all: it's an inherent characteristic
of this policy's gait cycle that shows up at least as strongly in its own native training
environment. Likewise the net yaw *drift* while walking dead straight (`vtheta=0`) —
measured earlier at about -8.9 deg/s average in the training env itself — is a real trait
of this checkpoint, not a sim/training discrepancy. Neither is fixable by further sim
parity work; both would need retraining (same category as the vtheta-tracking finding
below).

## Evidence: closed-loop A/B

Real `walk.onnx`, microban's own observation/action pipeline, commanded `vx = 0.3`.
Reference (training env): 6 s, no fall, lean mean 7.6° max 14.1°.

| config | outcome | lean mean / max | x travel |
| --- | --- | --- | --- |
| A — as-is: `<position>`, xl330, `kp=1334`, sim delays | **FELL at 1.52 s** | 18.3° / 61.5° | +0.30 m |
| B — A + actuators as `<motor>` | FELL at 1.82 s | 25.7° / 60.4° | +0.60 m |
| C — B + `kp_fw=7334` | FELL at 1.34 s | 25.6° / 62.9° | +0.03 m |
| D — C + SKS2401-like kt/R/current/armature | FELL at 1.64 s | 20.4° / 63.0° | +0.51 m |
| E — D + no sensor/actuation delays | **walked 8 s, no fall** | **7.1° / 16.1°** | +0.79 m |
| F — A but with no delays | FELL at 1.30 s | 18.0° / 62.5° | +0.30 m |

**The fix is a conjunction.** Actuator corrections alone (B/C/D) do not help; removing
delays alone (F) does not help; together (E) the behaviour matches training.

### Which delay, with the actuators already correct (10 s runs)

| gyro / quat / vel / act | outcome | lean mean / max | x travel |
| --- | --- | --- | --- |
| 0 / 0 / 0 / 0 | walked | 7.0° / 16.1° | +0.47 m |
| 3 / 3 / 0 / 0 | walked | 7.6° / 20.2° | -0.65 m |
| 3 / 4 / 0 / 0 | FELL at 4.86 s | 12.6° / 62.1° | +1.73 m |
| 3 / 4 / 1 / 0 | walked | 10.0° / 24.1° | +1.74 m |
| 3 / 3 / 1 / 0 | walked | 9.3° / 20.7° | +1.62 m |
| 3 / 3 / 1 / 2 | FELL at 0.88 s | 19.8° / 63.0° | -0.05 m |
| **2 / 2 / 1 / 2** | **walked** | **8.8° / 19.0°** | **+2.01 m** |
| 1 / 1 / 1 / 2 | walked | 8.3° / 24.1° | +2.14 m |
| 3 / 4 / 1 / 2 (`make sim` default) | FELL at 1.64 s | 20.4° / 63.0° | +0.51 m |

Individually each delay is survivable; stacked they are not, and the results near the
maximum are marginal and noisy — that in itself says there is no margin left there.
`2/2/1/2` walks reliably and tracks the command well (+2.01 m in 10 s ≈ 0.20 m/s against a
commanded 0.3).

## Sim half — done

All four items implemented in `src/sim/mujoco_controller.py`, `src/sim/sim_main.py`,
`Makefile`, `pyproject.toml`, plus the new `vendor/bam/`:

1. Actuators converted to `<motor>` via `MjSpec` + `set_to_motor()` per-actuator, with
   `force_limit = vin·kt/R` computed from whichever BAM model (steel/alu) that joint
   belongs to. `bam_load_model(...)` moved above the model compile so the force limit is
   available before `spec.compile()`.
2. The single xl330 BAM model replaced by two SKS2401 groups (`_STEEL_JOINTS`:
   shoulders + knees, `_ALU_JOINTS`: hips + ankles — matching
   `ROKI4_MOTOR_JOINT_EXPR_STEEL`/`_ALU` exactly, with an `assert` that every
   `MOTOR_TO_ID` joint lands in one of the two groups), `kp_fw = 7333.9` (`KP_DEFAULT`,
   already correct), `vin = 11.0` (`BAM_VIN`, already correct). Per-group current limits
   are **not** set explicitly — `Roki4Actuator` (`CurrentControlledActuator`) enforces
   `model.current_limit` loaded straight from each JSON (6.78177 / 3.19651), unlike the
   XL330's `actuator.max_current`, which this model doesn't use at all. Two
   `BamController` instances now exist (`_bam_steel`, `_bam_alu`); every method that used
   to touch a single `self._bam` (`set_kp`, `sync_read_kp`, `sync_write_kp`,
   `sync_write_goal_position`, `sync_read_present_current`, `_bam_reset_targets`) now
   routes per-joint through `_name_to_bam` / per-group through `_bams`, and current
   estimation uses a per-joint `_name_to_kt` instead of one shared `kt`.
   Unblocked by vendoring `bam` (see Bug 2 above and `vendor/bam/README.md`) instead of
   depending on a nonexistent git ref.
3. `SPAWN_TRUNK_Z` → `0.2486`.
4. `_DelayBuffer` (`mujoco_controller.py`) redesigned to sample each channel's lag
   uniformly from `[0, max_lag]`, resampled every tick by default or every 64 ticks for
   gyro/quat — matching mjlab's `delay_min_lag=0`/`delay_update_period` semantics exactly
   — instead of holding a fixed compromise value. `sim_main.py`'s `--delay-*` flags
   became upper bounds with training's own defaults restored: gyro/quat `3` (was
   temporarily dropped to `2` as a fixed value in an earlier pass, no longer needed
   now that it's sampled), vel `1`, pos/act `0`.
5. Solver settings (`spec.option.*`: implicitfast integrator, timestep 0.005,
   iterations/ls_iterations/ccd_iterations), set in the same `MjSpec` pass — see Bug 5.
   (A foot-geom `condim` change was also tried and reverted — verified inert, see Bug 5's
   "Dead end" for why.)

### Verification

Confirmed clean end-to-end:

- `MuJoCoController(...)` constructs successfully; `spec.actuators` conversion produces
  the expected per-group `forcerange` (`±155.9` steel / `±22.1` alu) and
  `gaintype == FIXED` for all 16 actuators.
- Spawns with `ncon = 0` (no penetration) at `qpos[2] = 0.2486`, settles to `0.2465`
  within a fraction of a second — no ejection.
- `make sim` itself runs clean through the real entrypoint: imports resolve, the GLFW
  viewer opens, "Starting control loop at 50.0 Hz" prints, no exceptions, runs
  indefinitely until killed.
- Full closed-loop test through the **actual** `MuJoCoController` + `WalkMove` +
  `Observation`/`MotorCommand` classes (only the outer scheduler-loop plumbing was
  hand-rolled, to drive it without a keyboard/GLFW window), `walk.onnx`, `vx = 0.3`,
  10 s, delays `2/2/1/2` (an earlier, fixed-value pass — see below for the properly
  sampled re-run):

  | Kp used once `WalkMove` goes active | outcome | lean mean / max | x travel |
  | --- | --- | --- | --- |
  | `KP_RL = 1334` (current `constants.py` value) | walked full 10 s, no fall | 13.4° / 44.5° | +1.15 m |
  | `KP_RL = 7334` (SKS2401 `kp_fw`, training's value) | walked full 10 s, no fall | **5.7° / 13.6°** | -0.17 m |

  Training reference: 6 s, no fall, lean mean 7.6° max 14.1°. At `Kp = 7334` the match is
  close to exact. At `vx = 0.5` the same pattern holds (1334: 12.2°/23.5°; 7334:
  5.8°/14.8°) — no sign of the `vx≈0.5` speed-limit fall `walk_debugging.md` reported
  actually being a training-limit; more likely it too was a symptom of the same
  actuator/Kp mismatch, now resolved.

  **This is not applied to `constants.py`.** `KP_RL` is shared with `main.py`/`walk.py`,
  which write it straight to real servo P-gain registers — see "the `KP_RL` question"
  under the hardware half below before changing it.

- Re-ran the same closed-loop test with `_DelayBuffer`'s proper sampling (gyro/quat
  max 3, resampled every 64 ticks; vel max 1; act max 0 — training's own numbers, not
  the earlier fixed 2/2/1/2 compromise), `KP_RL = 7334`, `vx = 0.3`, 10 s, 6 independent
  trials (stochastic now, so run repeatedly rather than once):

  ```
  trial 0: walked full 10s, no fall   lean mean=5.6 max=14.4  x=+2.06m
  trial 1: walked full 10s, no fall   lean mean=5.7 max=12.5  x=+1.33m
  trial 2: walked full 10s, no fall   lean mean=5.6 max=12.0  x=+1.34m
  trial 3: walked full 10s, no fall   lean mean=6.5 max=18.0  x=+2.32m
  trial 4: walked full 10s, no fall   lean mean=5.6 max=11.0  x=+1.78m
  trial 5: walked full 10s, no fall   lean mean=5.4 max=15.6  x=+2.46m
  ```

  All 6 trials walk the full duration; lean mean 5.4–6.5°, max 11.0–18.0° — matching
  the training reference (7.6°/14.1°) at least as closely as the fixed-2/2/1/2 pass did,
  now for the right reason (sampled the way training samples it) rather than a value
  picked by trial and error.

## The hardware half — open

Confirmed with the user: **the physical robot is a Roki4 with SKS2401 servos**, so
`walk.onnx` does match the hardware. But this repo's hardware path is still written for the
XL330 microban, and the SKS2401 is a **different control architecture**, not just different
numbers.

### The `KP_RL` question — decided for sim, still open for hardware

The sim-half verification found that `KP_RL = 1334` (the value at the time) does not stop
the robot walking, but produces a visibly rougher gait than training's own
`kp_fw = 7333.9` (lean mean/max roughly 2× and 3× worse). Since `WalkMove` writes whatever
`KP_RL` is straight through `sync_write_kp` — the exact same call
`main.py`/`robot_controller.py` use to set real servo P-gain registers — this constant is
not sim-only, so it was deliberately left unchanged as part of the sim-half work and
written up here instead as a decision for the user to make.

**The user has since changed `KP_RL` to `7334` directly in `constants.py`.** For `make
sim`, this is correct and already verified (see the 6-trial re-run above, all with
`KP_RL = 7334`). It does **not** by itself make it safe to run `main.py` against the real
robot: the overcurrent proxy meant to protect the real servos (`scheduler.py`'s
`_per_motor_currents`) still models the wrong motor family entirely (XL330's
voltage-controlled duty cycle, not the SKS2401's current-controlled firmware — see the
comparison table below), so raising the real P-gain to 7334 before that proxy is rewritten
could let a genuine overcurrent condition go undetected. **Do not run `make run` /
deploy to the physical robot until the hardware-half current-proxy rewrite (below) is
done** — nothing about the sim fix changes that.

From `bam/roki4/actuator.py` (`Roki4Actuator`, registered as `"SKS2401"`):

```python
PWM_LIMIT   = 2999
DEFAULT_KP  = 7333.9   # PWM / rad
VELOCITY_GAIN = -274.0 # PWM / (rad/s)
CURRENT_LIMIT = 1.5    # A  (class default; the fitted JSONs override this)
error_gain  = CURRENT_LIMIT / PWM_LIMIT

I = (q_target - q) * kp * error_gain * control_gain_ratio
  + dq * VELOCITY_GAIN / PWM_LIMIT * CURRENT_LIMIT * velocity_gain_ratio
# then clamped by voltage headroom (back_emf = kt*dq, R), then by current_limit
```

| | XL330 (what this repo assumes) | SKS2401 (actual) |
| --- | --- | --- |
| BAM class | `VoltageControlledActuator` | `CurrentControlledActuator` |
| firmware loop | `duty = kp·error_gain·err`, `I = (vin·duty - kt·dq)/R` | current commanded directly, as above |
| damping term | none | **`VELOCITY_GAIN = -274.0`** |
| `kp` units | Dynamixel P register | **PWM/rad** |
| interface | Dynamixel protocol | IronArt: target position, measured position, signed PWM |

The docstring also warns that the fitted constants are *joint-level* and include the
transmission ratio — they are not bare DC-motor datasheet values.

### What crosses over from sim to hardware

The `bam` **package** is sim-only: it is imported by `src/sim/mujoco_controller.py` alone
(`scheduler.py` imports only the `BAM_MAX_CURRENT` *constant*), `src/sim` is excluded from
`make sync`, and `better-actuator-models` is in the `sim` dependency group so it is never
installed on the Pi. Changing the motor model cannot affect hardware.

But these `constants.py` values are synced and **do** run on the robot:

| constant | current value | used by | consequence |
| --- | --- | --- | --- |
| `PROXY_KT` / `PROXY_R` | 0.366 / 2.811 | `scheduler.py` | XL330 values in the XL330 *voltage* equation |
| `PROXY_ERROR_GAIN`, `PROXY_KP`, `PROXY_VIN` | XL330 scaling | `scheduler.py` | same |
| `BAM_MAX_CURRENT` | 6.8 | `scheduler.py` + sim | matches steel (6.78) but not alu (3.20) |
| `PRESENT_CURRENT_UNIT_A` | 0.001 | `robot_controller.py` | XL330 register unit, scales the *measured* current |
| `KP_DEFAULT` | 7334 | `main.py`, `walk.py`, sim | **already correct** — this is `Roki4Actuator.DEFAULT_KP` in PWM/rad |
| `KP_RL` | now `7334` (was 1334) | `walk.py` | now matches training; the real-servo risk is the overcurrent proxy below, not this value itself |
| `KP_GAIN_PRM` | 0.0022 | *unused* | XL330-specific dead constant |
| `MOTOR_SIGN` | — | `robot_controller.py` | still lists `head` / `left_elbow` / `right_elbow` |

The overcurrent proxy in `scheduler.py::_per_motor_currents` is therefore **structurally**
wrong, not merely miscalibrated: it reproduces a voltage-controlled duty cycle for a
current-controlled servo, and has no velocity term. That is the likely reason
`OVERCURRENT_CUTOFF_A` had to be inflated from 15 A to 105 A — the *estimate* was out of
range, not the robot.

### Unresolved

`robot_controller.py` and `voltage.py` drive the servos with `rustypot.Xl330PyController`.
`rustypot` 1.6.0 exports only `Ax, Mx, Orbita2dFoc, Orbita2dPoulpe, Orbita3dFoc,
Orbita3dPoulpe, Scs0009, Scs0043, Sts3215, Xl320, Xl330, Xl430` — **no SKS2401** — and the
BAM model describes an IronArt PWM interface rather than Dynamixel. Either the servos speak
a Dynamixel-compatible protocol despite the SKS2401 label, or `make run` cannot be driving
them correctly. **Needs confirmation against the actual wiring before anything on the
hardware path is changed.**

### Hardware TODO, once that is settled

- resolve the `Xl330PyController` question above
- ~~`KP_RL` → 7334~~ **done** (`constants.py` now has 7334) — but this alone does not make `make run` safe; the item below does
- rewrite the current proxy as SKS2401 current-control, including the velocity term
- per-group `BAM_MAX_CURRENT` (6.78 steel / 3.20 alu)
- re-derive `OVERCURRENT_CUTOFF_A` once the proxy is correct — do not keep 105 A
- fix `PRESENT_CURRENT_UNIT_A` for the SKS2401/IronArt register
- delete the unused XL330 `KP_GAIN_PRM`
- prune dead `head` / `*_elbow` entries from `MOTOR_SIGN`
- reconcile the stale comments on `KP_RL` and `OVERCURRENT_CUTOFF_A`, which claim values
  (125, 15.0) that the code does not have (1334, 105.0)

## Incidental notes

- `src/model/mjcf/robot.xml` (Roki4, `class="SKS2401"`, 16 DoF) was swapped in by commit
  `a94ea71` "changing to roki in sim". `src/model/mjcf/robot_m.xml` is the older microban
  robot (`class="xl330"`, 19 DoF with head and elbows), still present.
- `src/agents/walk_microban.onnx` is the older 19-DoF XL330 policy (run
  `2026-06-26_16-03-28`, 63 obs / 18 actions). `walk.onnx` is the Roki4 one
  (`2026-08-16_08-45-39`, 57 obs / 16 actions).
- Solver settings and foot contact condim were both real, implicated differences —
  see Bug 5 above (fixed).
- Training randomises `encoder_bias` (~±0.012 rad) and subtracts it from the position
  target. Zero-mean domain randomisation; the observation itself is unbiased
  (`joint_pos_rel` with `biased=False`). Nothing to replicate.
