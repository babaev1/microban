# Vendored `bam` (subset)

`bam` (https://github.com/Rhoban/bam, Apache-2.0 — see `LICENSE`) is the actuator model
used by `src/sim/mujoco_controller.py` for the Roki4's SKS2401 servos (via the
`"SKS2401"` / `bam.roki4.actuator.Roki4Actuator` model and its fitted parameters).

## Why this is vendored instead of a normal dependency

- PyPI's `better-actuator-models` (what `pyproject.toml`'s `sim` group used to depend on)
  has **no SKS2401 support at all** — its `params/` only covers
  `erob80_100, erob80_50, feetech_sts3215_7_4V, mx106, mx64, xl320, xl330`, and its
  `bam/actuators.py` has no `"SKS2401"` entry. Loading `roki4_actuator/m3.json` against
  that package raises `KeyError: 'SKS2401'`.
- `github.com/Rhoban/bam` does not have it either. Checked exhaustively — all 27 remote
  refs (branches + tags), including the `doc` branch that `mjlab_roki6/pyproject.toml`
  points its own `bam` dependency at — none contains `bam/roki4/actuator.py`, and none
  registers `"SKS2401"` in `bam/actuators.py`.
- The only place this code exists is a hand-patched copy sitting in
  `mjlab_roki6/.venv/lib/python3.13/site-packages/bam` (no `dist-info`, so it was not
  installed via pip/uv from any of the above — someone dropped newer files directly into
  the venv). Copied from there on 2026-09-06.

Because that source lives inside another project's venv (which `uv sync` can blow away
and rebuild at any time) rather than in any git-tracked location, there is no dependency
reference (`git`, `path`, or otherwise) that reliably points at it. Vendoring a copy is the
only way to make `make sim` reproducible without depending on the state of that other
project's environment.

## What's here vs. what's not

Only the modules actually reached by `bam.model.load_model` / `bam.mujoco.MujocoController`
(what `mujoco_controller.py` imports) and their transitive imports:

```
bam/{__init__,model,actuator,actuators,parameter,testbench,mujoco,message}.py
bam/{dynamixel,erob,feetech,unitree}/{__init__,actuator}.py   # actuators.py imports all of these eagerly
bam/roki4/{__init__,actuator,testbench}.py
bam/params/roki4_actuator/{m3,m3_al}.json                     # the two SKS2401 fits actually used
```

Not vendored (present in the source tree but unused by `mujoco_controller.py`, and in
several cases unimportable here without extra dependencies like `torch`/`mujoco_warp`):
`bam.mjlab` (mjlab/training-only actuator wrapper), `bam.fit`, `bam.simulate`,
`bam.animate`, `bam.jitter`, `bam.logs`, `bam.process`, `bam.plot`, `bam.trajectory*`,
`bam.drive_backdrive`, `bam.to_mujoco`, every `*_old*.py` file, and every other bundled
motor's params directory.

## Usage

```python
from bam.model import load_model
steel = load_model(motor_name="roki4_actuator", model="m3")     # shoulders + knees
alu   = load_model(motor_name="roki4_actuator", model="m3_al")  # hips + ankles
```

See `docs/dev/sim_training_parity.md` for the full story (why xl330 was wrong, why the
actuators need converting to `<motor>`, etc.) and `Makefile`'s `sim`/`sim-master` targets
for how this directory reaches `PYTHONPATH`.

## Updating

If a newer bam fit for the SKS2401 (or a fix to `bam/roki4/actuator.py`) ever lands in a
real, citable place (an upstream release, a proper fork), replace the files here from
that source and update this note — don't hand-edit the vendored copy in place.
