# Driving `make sim-master` from the robot's own remote-control joysticks

`make sim-master` (see [sim_stream.md](sim_stream.md)) normally takes velocity
commands from the SSH terminal's keyboard, since headless mode has no window to bind
a key callback to. `--joystick` (`JOYSTICK=1` for the Makefile target) replaces that
with the STM32/zubr board's own handheld remote — read over the same `/dev/ttyS2` link
[`hw_state_stream.py`](hw_stream.md) uses — so you can walk the *simulated* robot
around with the real remote instead of typing velocity steps.

```bash
# On the Orange Pi:
make sim-master SLAVE=<laptop-ip>:9761 JOYSTICK=1
# or directly:
PYTHONPATH=src:vendor/bam uv run --group sim src/sim/sim_main.py --hz 50 --stream-to <laptop-ip>:9761 --joystick

# On the laptop, as usual:
make sim-viewer PORT=9761
```

## Axis mapping

[`src/input/zubr_joystick_input.py`](../../src/input/zubr_joystick_input.py):

| Stick | Axis | Command |
|---|---|---|
| Right | Y | `vx` (forward / backward) |
| Right | X | `vy` (left / right strafe) |
| Left  | X | `vtheta` (rotate clockwise / counterclockwise) |

Left stick Y is unused. Read via `zubr_link.ZubrLink.poll()` — the same read/relax
transaction `hw_state_stream.py` uses, so this never sends the robot any motor
commands either (it only ever *reads* the remote's joystick fields off the telemetry
frame).

Confirmed against real hardware: this remote reports all three axes opposite to
`gamepad_input.py`'s convention, so `VX_SIGN`/`VY_SIGN`/`VTHETA_SIGN` in
`zubr_joystick_input.py` are all `-1.0`.

## Why moves/reset/stop still come from the keyboard

`zubr.py` documents no bit layout for `remote.buttons` — which physical button on the
handheld remote is which bit isn't known, and guessing risks silently binding the
wrong one. Rather than guess, `ZubrJoystickInputSource` only reads the joystick axes
from the remote and composes a normal `KeyboardInputSource` for everything else (move
toggling, reset, torque display, stop) — typed into the same SSH terminal, exactly as
plain `--stream-to` already works. Since there's then no known remote button to toggle
the walk move with either, and driving speed via joystick is meaningless without it,
`'walk'` is force-enabled the entire time this input source runs (see `FORCED_MOVE` in
`zubr_joystick_input.py`).

## Unrelated fix bundled with this: `colorama` was missing from `--group sim`

While testing this, `sim_main.py` failed to import at all
(`ModuleNotFoundError: No module named 'colorama'`) — unrelated to the joystick work.
`vendor/bam/bam/message.py` imports `colorama` unconditionally, but nothing in
`pyproject.toml` ever declared it: it used to arrive only as an incidental transitive
dependency of the now-removed `better-actuator-models` PyPI package, and silently
stopped resolving once a `uv sync --group sim` ran without anything left to pull it in.
Added directly to the `sim` dependency group with a comment explaining why. If `make
sim`/`sim-master` ever throws this same error again, `uv sync --group sim` is the fix.
