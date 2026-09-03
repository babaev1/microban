# Running the sim on the Pi with the graphic window on your laptop

Procedure for running the MuJoCo simulation on the Orange Pi Zero 3W
(`orangepizero3w`) over SSH while seeing the live graphic window on your own
laptop, plus the troubleshooting notes behind why it's built this way and a
record of today's session.

> Want the real 3D viewer instead of the 2D stick figure this produces? See
> [sim_stream.md](sim_stream.md) — a master/slave split that runs physics headless
> on the Pi and renders on your laptop's own GPU.

## Procedure

1. **Connect with X11 forwarding** (not a plain `ssh`) — this is what carries the
   graphic window's protocol back to your laptop instead of the Pi's own monitor:
   ```bash
   ssh -X orangepi@<pi-ip>      # -Y instead of -X if -X is rejected/too restrictive
   ```
   On Ubuntu, `ssh` and X11 support are already installed; nothing extra to set up
   locally. Requires `xauth` and `X11Forwarding yes` on the Pi side — both already
   in place on this image (`/etc/ssh/sshd_config`).

2. **Run the sim on the Pi, from that same SSH session** — the physics, scheduler
   and control loop all run remotely on the Pi; only the picture is forwarded.
   No `MUJOCO_GL` or other env var needed — `MuJoCoController` auto-detects the
   right viewer (see below):
   ```bash
   cd ~/microban
   PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50
   ```
   Don't set `DISPLAY` yourself — SSH already pointed it at the forwarded display
   for this session. Setting `DISPLAY=:0` instead sends the window to the Pi's own
   local monitor, not your laptop.

3. The window opens on your laptop as a live 2D stick figure (arms, legs, trunk,
   floor line) redrawn from the physics each tick — confirmed working end-to-end
   in today's session. All keyboard controls (moves, velocity, reset, `[i]` IMU
   marker, `[q]` quit) work in it exactly as documented in
   [usage.md](../usage.md).

## Why: the two viewers, and how MuJoCoController picks one

`mujoco.viewer.launch_passive` needs a real desktop-OpenGL (GLX) window, and a
failed `launch_passive` **kills the whole process** (it isn't a catchable Python
exception — confirmed directly: even `except BaseException` doesn't stop it).
So `MuJoCoController.__init__` never calls it blind; it probes first via
`_can_create_gl_window()` and picks:

- Probe succeeds → real `mujoco.viewer.launch_passive` (full 3D viewer).
- Probe fails → `_SoftwareViewer`, a fallback needing no OpenGL at all: it draws
  the robot's body positions as a simple 2D stick figure with PIL and shows it in
  a plain Tk window (Tk paints through ordinary X11, not GLX), throttling
  redraws (`redraw_every`) to keep the control loop's timing budget reasonable.

Three separate GL failure modes were hit and diagnosed while building this, all
variants of the same underlying problem — MuJoCo's context needs a fully
negotiated depth+stencil framebuffer, and several paths on this system can't
provide one:

1. **Pi's own local monitor (`DISPLAY=:0`)**: this board's GPU (PowerVR
   B-Series, via a custom `sunxi-drm`/`pvr` driver under `/usr/local/lib/dri`)
   only exposes OpenGL ES, not desktop GL — GLX can't get a context at all
   (`glxinfo`: "couldn't find RGB GLX visual or fbconfig").
2. **MuJoCo's own OSMesa offscreen renderer** (`mujoco.Renderer`, tried as a
   GPU-independent alternative before writing `_SoftwareViewer`): this system's
   Mesa (`20.3.5`, Debian bullseye-era) hits a known bug creating the OSMesa
   framebuffer (`mujoco.FatalError: Default framebuffer is not complete, error
   0x0`), confirmed independent of `MESA_GL_VERSION_OVERRIDE`. Upgrading system
   Mesa was ruled out as too risky — it's patched in-place for the PowerVR
   driver and shared with the desktop GUI.
3. **Indirect GLX over `ssh -X`**: hit live in today's session. A first version
   of the probe only checked that `glfwCreateWindow` returned a handle at all —
   that succeeded over `ssh -X` (a basic pixel format is easy to get), but the
   real `launch_passive` then crashed with the exact same "Default framebuffer
   is not complete, error 0x0" as #2, just via a different path: indirect GLX
   frequently can't negotiate the modern framebuffer config MuJoCo needs,
   independent of anything fixable on this end. **Fix**: `_can_create_gl_window`
   now replicates MuJoCo's own check instead of a shallow one — it creates a
   hidden window with matching depth/stencil hints, makes the context current,
   and calls `glCheckFramebufferStatus` itself before deciding. Re-verified
   against both known-broken local cases after the fix; the user then confirmed
   the stick figure renders correctly over their real `ssh -X` session.

`MUJOCO_GL=osmesa` still forces `_SoftwareViewer` explicitly and skips the
probe — useful on a slow link where the real 3D viewer's extra bandwidth isn't
worth it, or as a known-good fallback if the auto-detected real viewer ever
misbehaves. Any other `MUJOCO_GL` value (e.g. `glfw`, `egl`) is trusted as-is
and also skips the probe.

Your laptop needs an X server to receive the forwarded window if you're not on
Linux:
- macOS: install [XQuartz](https://www.xquartz.org/), log out/in once, then `ssh -X`.
- Windows: run an X server (e.g. VcXsrv) or use WSL, then `ssh -X`.

## Other change from today's session: overrun warnings removed

`Scheduler.run()` (`src/scheduler.py`) used to print `Warning: control loop
overrun by N ms` on every tick that ran past its budget. Rendering the sim
viewer's window each tick pushed the 50 Hz control loop over budget often
enough that this flooded the terminal, so the print was removed outright.

This is scheduler-wide, not sim-specific — it also silences the same warning
when running on the real robot. If that timing signal is ever needed again for
hardware debugging, either restore the removed `print` in the `else` branch of
the sleep-timing block, or reintroduce it rate-limited (e.g. once/second)
instead of every tick.
