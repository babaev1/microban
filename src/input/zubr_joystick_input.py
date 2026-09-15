# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Drive the walk policy's velocity command from the STM32/zubr board's own
remote-control joysticks, read over the same /dev/ttyS2 link zubr_link.py/
hw_state_stream.py use — so `make sim-master` can be driven from the robot's own
handheld remote instead of typing velocity steps into the SSH session.

Axis mapping (per the physical remote, confirmed by inspection — not by testing the
sign convention against real hardware, see VX_SIGN/VY_SIGN/VTHETA_SIGN below):
    right stick Y -> vx     (forward / backward)
    right stick X -> vy     (left / right strafe)
    left  stick X -> vtheta (rotate clockwise / counterclockwise)
Left stick Y is unused — nothing in this project needed a fourth axis.

Move toggling, reset, torque display and stop are deliberately NOT read from the
remote's buttons: zubr.py documents no bit layout for remote.buttons (which physical
button is which bit), and guessing one risks silently binding the wrong button on
real hardware. Instead this composes a plain KeyboardInputSource for all of that, on
the same SSH terminal sim_main.py already reads in --stream-to mode — only the
velocity axes come from the joystick. Since there's then no known button to toggle
the walk move with either, and driving speed via joystick is meaningless without it,
'walk' is force-enabled for as long as this input source runs.

See docs/dev/hw_stream.md.
"""

import threading

from input.input_source import InputSource, UserInput
from input.keyboard_input import KeyboardInputSource
from zubr_link import DEFAULT_BAUDRATE, DEFAULT_PORT, ZubrLink

# Stick fraction below which the axis is treated as centered (drift rejection).
# Same idea as gamepad_input.py's DEADZONE; not verified against this specific
# remote's actual center-position noise.
DEADZONE = 0.1

# zubr.py's remote joystick fields are signed int8 (see zubr_link.Telemetry).
_AXIS_FULL_SCALE = 127.0

# Sign per axis, so the mapping can be flipped without touching the logic below.
# Confirmed against real hardware: this remote reports all three axes opposite to
# gamepad_input.py's convention (stick pushed away from center in the labeled
# positive direction reads as a *negative* raw value here), so all three are flipped.
VX_SIGN = -1.0
VY_SIGN = -1.0
VTHETA_SIGN = -1.0

# The one move this input source can activate — see the module docstring for why
# it's unconditional rather than button-triggered.
FORCED_MOVE = "walk"


class ZubrJoystickInputSource(InputSource):
    """Velocity from the zubr remote's joysticks; everything else from the keyboard.

    Args:
        move_keys: passed straight through to the internal KeyboardInputSource (move
            toggling still works from the keyboard; 'walk' is force-enabled
            regardless — see FORCED_MOVE).
        port, baudrate: STM32 serial link, same defaults as zubr_link.ZubrLink.
        stop_flag_path: passed straight through to the internal KeyboardInputSource.
    """

    def __init__(
        self,
        move_keys: dict[str, str] | None = None,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        stop_flag_path: str = "/tmp/microban_scheduler.stop",
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._keyboard = KeyboardInputSource(move_keys=move_keys or {}, stop_flag_path=stop_flag_path)

        self._link: ZubrLink | None = None
        self._lock = threading.Lock()
        self._velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._link = ZubrLink(self._port, baudrate=self._baudrate)
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._keyboard.start()
        self._print_help()

    def stop(self) -> None:
        self._running = False
        self._keyboard.stop()
        if self._link is not None:
            self._link.close()
            self._link = None

    def read(self) -> UserInput:
        state = self._keyboard.read()
        with self._lock:
            state.velocity = dict(self._velocity)
        state.active_moves.add(FORCED_MOVE)
        return state

    @property
    def show_torque(self) -> bool:
        return self._keyboard.show_torque

    def consume_reset(self) -> bool:
        return self._keyboard.consume_reset()

    # ------------------------------------------------------------------
    # Internal

    def _poll_loop(self) -> None:
        assert self._link is not None
        while self._running:
            telemetry = self._link.poll()
            if telemetry is None:
                continue  # dropped/corrupt frame — keep the last good velocity
            right_x, right_y = telemetry.right_joystick
            left_x, _left_y = telemetry.left_joystick
            velocity = {
                "vx": VX_SIGN * self._normalize(right_y),
                "vy": VY_SIGN * self._normalize(right_x),
                "vtheta": VTHETA_SIGN * self._normalize(left_x),
            }
            with self._lock:
                self._velocity = velocity

    @staticmethod
    def _normalize(value: int) -> float:
        norm = max(-1.0, min(1.0, value / _AXIS_FULL_SCALE))
        return 0.0 if abs(norm) < DEADZONE else norm

    def _print_help(self) -> None:
        print("Zubr remote joystick controls:", end="\r\n", flush=True)
        print("  right stick  vx (up/down), vy (left/right)", end="\r\n", flush=True)
        print("  left stick   vtheta (X axis only)", end="\r\n", flush=True)
        print(f"  '{FORCED_MOVE}' move is always enabled while this input source runs", end="\r\n", flush=True)
