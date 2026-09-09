# `make sim` keyboard controls: why arrow keys silently did nothing

Reported symptom: with the walk fixed to a reliable gait (see
`sim_training_parity.md`), pressing the Left/Right arrow keys in the `make sim` window
never turned the robot (`vtheta` never changed) — and neither would Up/Down have moved
`vx`, though that wasn't separately noticed.

## Root cause

`mujoco.viewer.launch_passive` (used by `MuJoCoController` whenever a real GLX display is
available — the normal case) opens MuJoCo's own bundled `simulate` GUI. That GUI reserves
a fixed set of keys for its own playback controls **before** anything reaches the
`key_callback` passed in — for keys it reserves, `key_callback` is never called at all,
there's no event to intercept or pass through.

Confirmed directly from the compiled extension's own embedded help text
(`strings mujoco/_simulate.cpython-*.so`):

```
Play / Pause
Speed Up / Down
Step Back / Forward
Toggle Left / Right UI
```

This is MuJoCo's standard keybinding table: Space = Play/Pause, **Up/Down = playback
speed**, **Left/Right = step back/forward** (single-step through history), Tab = toggle
UI panels. All four arrow keys are reserved. This is a property of MuJoCo's bundled
viewer, not something the app's Python code can opt out of or reconfigure.

`sim.mujoco_input.MuJoCoInputSource.key_callback` had been mapping vx to Up/Down and
vtheta to Left/Right — i.e., entirely onto keys the viewer never forwards. The move-toggle
keys (`h`/`s`/`v`) and the function keys (`x`/`r`/`i`/`t`/`q`) all worked fine throughout,
because plain letter keys aren't part of MuJoCo's reserved set — which is exactly why the
fix is to move velocity control onto letters too, not to look for a way to un-reserve the
arrows.

## Fix

`vx`/`vtheta` now use plain letter keys instead, in `src/sim/mujoco_input.py`:

| key | action |
| --- | --- |
| `w` | vx + (forward) |
| `z` | vx − (back) |
| `d` | vtheta + (turn right) |
| `a` | vtheta − (turn left) |

Chosen to avoid every key already bound in this file (`h`/`s`/`v` move toggles,
`x`/`r`/`i`/`t`/`q` functions). The old `_GLFW_KEY_UP/_DOWN/_LEFT/_RIGHT` branches (and the
matching dead-code path in `mujoco_controller.py`'s `_SoftwareViewer._on_key`, which
translated Tk arrow keysyms into those same GLFW codes purely to feed this callback) were
removed rather than left in place — they can never fire again since `key_callback` no
longer reacts to those codes.

## Scope: this affects only `MuJoCoInputSource`, not the real robot or `--stream-to`

`src/input/keyboard_input.py` (`KeyboardInputSource` — used by `main.py` on real hardware,
and by `sim_main.py --stream-to`) implements arrow keys itself via raw terminal ANSI
escape sequences (`\x1b[A` etc.), read directly from stdin in raw mode. There is no GLFW
window in that path at all, so MuJoCo's reserved-key behavior never applies there — its
arrow-key handling was, and still is, unaffected. `docs/usage.md`'s documented arrow-key
controls describe that path and remain accurate.

Only the windowed, default `make sim` path (`MuJoCoInputSource`, driven through
`mujoco.viewer.launch_passive`'s `key_callback`) was broken, and only that file needed
changing.
