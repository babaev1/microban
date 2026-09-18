# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Drive the walk policy's velocity command from the STM32/zubr board's own
remote-control joysticks, on real hardware — the `main.py` counterpart of
`input/zubr_joystick_input.py` (used by `sim_main.py --joystick`).

Same axis mapping, signs and "velocity from the remote, everything else from the
keyboard" design as the sim version — see that module's docstring for the full
rationale. The one real difference: `sim_main.py` has nothing else talking to the
STM32, so its input source can freely open and poll its own ZubrLink. On real
hardware, `ZubrRobotController` already owns the one connection to /dev/ttyS2 and
polls it every control tick (this protocol is strictly one request/response
transaction at a time — a second independent poller on the same link would
interleave writes and corrupt both streams). So this input source takes a
reference to that controller and reads whatever telemetry it most recently
fetched instead of opening a second connection — no extra STM32 traffic, and
naturally in sync with the same control-loop cadence everything else runs at.
"""

from input.input_source import InputSource, UserInput
from input.keyboard_input import KeyboardInputSource
from zubr_robot_controller import ZubrRobotController

# Stick fraction below which the axis is treated as centered (drift rejection).
DEADZONE = 0.1

# zubr.py's remote joystick fields are signed int8 (see zubr_link.Telemetry).
_AXIS_FULL_SCALE = 127.0

# Confirmed against real hardware (see zubr_joystick_input.py): this remote reports
# all three axes opposite to gamepad_input.py's convention, so all three are flipped.
VX_SIGN = -1.0
VY_SIGN = -1.0
VTHETA_SIGN = -1.0

# The one move this input source can activate — see the module docstring for why
# it's unconditional rather than button-triggered.
FORCED_MOVE = "walk"


class ZubrRemoteInputSource(InputSource):
    """Velocity from the zubr remote's joysticks; everything else from the keyboard.

    Args:
        controller: the ZubrRobotController already polling the STM32 link — this
            reads its latest_telemetry rather than opening a second connection.
        move_keys: passed straight through to the internal KeyboardInputSource (move
            toggling still works from the keyboard; 'walk' is force-enabled
            regardless — see FORCED_MOVE).
        stop_flag_path: passed straight through to the internal KeyboardInputSource.
    """

    def __init__(
        self,
        controller: ZubrRobotController,
        move_keys: dict[str, str] | None = None,
        stop_flag_path: str = "/tmp/microban_scheduler.stop",
    ) -> None:
        self._controller = controller
        self._keyboard = KeyboardInputSource(move_keys=move_keys or {}, stop_flag_path=stop_flag_path)

    def start(self) -> None:
        self._keyboard.start()
        self._print_help()

    def stop(self) -> None:
        self._keyboard.stop()

    def read(self) -> UserInput:
        state = self._keyboard.read()
        state.velocity = self._read_velocity()
        state.active_moves.add(FORCED_MOVE)
        return state

    @property
    def show_torque(self) -> bool:
        return self._keyboard.show_torque

    def consume_reset(self) -> bool:
        return self._keyboard.consume_reset()

    # ------------------------------------------------------------------
    # Internal

    def _read_velocity(self) -> dict[str, float]:
        telemetry = self._controller.latest_telemetry
        if telemetry is None:
            return {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        right_x, right_y = telemetry.right_joystick
        left_x, _left_y = telemetry.left_joystick
        return {
            "vx": VX_SIGN * self._normalize(right_y),
            "vy": VY_SIGN * self._normalize(right_x),
            "vtheta": VTHETA_SIGN * self._normalize(left_x),
        }

    @staticmethod
    def _normalize(value: int) -> float:
        norm = max(-1.0, min(1.0, value / _AXIS_FULL_SCALE))
        return 0.0 if abs(norm) < DEADZONE else norm

    def _print_help(self) -> None:
        print("Zubr remote joystick controls:", end="\r\n", flush=True)
        print("  right stick  vx (up/down), vy (left/right)", end="\r\n", flush=True)
        print("  left stick   vtheta (X axis only)", end="\r\n", flush=True)
        print(f"  '{FORCED_MOVE}' move is always enabled while this input source runs", end="\r\n", flush=True)
