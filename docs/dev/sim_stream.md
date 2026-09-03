# Running the sim's physics on the Pi with real 3D rendering on your laptop

A master/slave split: the Orange Pi runs physics headless and broadcasts state over the
network; your laptop runs a display-only MuJoCo viewer against its own GPU. This gets
you the real 3D viewer instead of the [X11-forwarded stick figure](sim_display.md) —
`mujoco.viewer.launch_passive` on your laptop uses your laptop's own desktop OpenGL, so
none of the GLX/framebuffer problems documented there apply.

## Why not just X11-forward the real viewer?

Because there's no real GLX viewer to forward on the Pi's end. Its GPU only exposes
OpenGL ES, and — confirmed in the [X11-forwarding session](sim_display.md) — even
indirect GLX over `ssh -X` to a real desktop can't negotiate the framebuffer MuJoCo
needs, only a basic pixel format. That's why `MuJoCoController` falls back to
`_SoftwareViewer`, a crude 2D stick figure with no GL involved at all.

This approach sidesteps the problem instead of working around it: the Pi never touches
OpenGL. It only does physics (`mj_step`) and sends the resulting `qpos`/`qvel` over UDP.
Your laptop loads the *same* MJCF, writes the received `qpos`/`qvel` into its own
`MjData`, calls `mj_forward` (recomputes body poses, contacts, sensor data — no
integration), and hands that to the real `mujoco.viewer.launch_passive`, which runs
entirely against your laptop's own GPU.

## Procedure

1. **Both machines need the same model.** They must load byte-identical `scene.xml` (and
   whatever it includes) — `qpos`/`qvel` are plain arrays streamed by index, with no
   names attached, so a mismatched model silently produces a garbled or wrong pose (a
   size mismatch is caught and logged; a same-size-but-different model isn't). Simplest:
   same git commit on both sides — `git rev-parse HEAD` on each and compare.

2. **On your laptop, start the slave viewer first** (it just waits for state):
   ```bash
   PYTHONPATH=src uv run --group sim src/sim/sim_viewer_client.py --listen 0.0.0.0:9761
   # or: make sim-viewer PORT=9761
   ```
   This opens the real 3D MuJoCo viewer — controllable with mouse/scroll like any local
   sim, but display-only: keyboard robot controls (moves, velocity, reset) don't work
   in this window (see below).

3. **On the Pi (plain `ssh`, no `-X` needed — nothing is forwarded)**, run the sim
   headless, pointed at your laptop's IP:
   ```bash
   cd ~/microban
   PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50 --stream-to <laptop-ip>:9761
   # or: make sim-master SLAVE=<laptop-ip>:9761
   ```
   `MuJoCoController` opens no viewer at all in this mode (see `_NullViewer` in
   [mujoco_controller.py](../../src/sim/mujoco_controller.py)) — it only steps physics
   and calls `StateSender.send()` once per tick.

4. Your laptop's window starts showing the robot within one tick (~20 ms) of the first
   packet arriving.

## Keyboard control stays on the Pi's terminal

There's no window on the Pi to bind a key callback to, so `--stream-to` switches input
from `MuJoCoInputSource` (GLFW/Tk window key events) to
[`KeyboardInputSource`](../../src/input/keyboard_input.py) — the same raw-terminal
reader `viewer_main.py` and the real robot use. Controls are the same keys, just typed
into the SSH session running `sim_main.py` instead of clicked into a viewer window:
moves, arrow-key velocity, `[i]` IMU display, `[t]` torque display, `[r]` reset, `[q]`
quit.

## Wire format

[`sim/state_stream.py`](../../src/sim/state_stream.py) — plain UDP, "latest wins," never
TCP: every packet is a full `(qpos, qvel)` snapshot, not a delta, so a dropped packet
just costs one skipped redraw. `StateReceiver.poll()` drains any backlog and applies
only the newest packet, so a slow slave never renders a queue of stale state. At 50 Hz
with this robot's ~40 `qpos` + ~40 `qvel` float64 values, bandwidth is trivial even over
Wi-Fi — the header carries `nq`/`nv` so a model mismatch is logged instead of silently
corrupting the unpack.

Assumes both machines are little-endian (true for the Pi's ARM Linux and any
x86_64/ARM laptop) — no byte-swapping is done.

## Choosing between this and X11 forwarding

- **This (master/slave streaming)**: real 3D viewer, needs your laptop and the Pi on
  the same network (or reachable over one), keyboard control moves to the Pi's
  terminal.
- **[X11 forwarding](sim_display.md)**: single `ssh -X` session, no second machine-side
  script to run, but you're limited to the 2D stick figure and it needs an X server on
  non-Linux laptops (XQuartz/VcXsrv).
