"""BAM model for a Roki4 right-shoulder actuator.

IronArt exposes a target position, measured joint position and a signed PWM
command.  Here that command is used as a proxy for the motor current requested
by the firmware: ``I = kp * position_error + kd * velocity``.  A full-scale PWM
command maps to the configured current limit.
This is an effective *joint-level* model: fitted motor constants include the
transmission ratio and should not be interpreted as bare DC-motor datasheet
values.
"""

from bam.actuator import ArrayLike, CurrentControlledActuator
from bam.parameter import Parameter
from bam.testbench import Pendulum, Testbench


class Roki4Actuator(CurrentControlledActuator):
    """Current-controlled model of a Roki4 actuator.

    ``kp`` is read from each BAM log in PWM/rad.  The derivative gain is the
    measured controller value in PWM/(rad/s).  Small optimisable ratios allow
    for calibration error without making the controller gains free parameters.
    """

    PWM_LIMIT = 2999
    DEFAULT_KP = 7333.9  # PWM / rad
    VELOCITY_GAIN = -274.0  # PWM / (rad/s)
    CURRENT_LIMIT = 1.5  # A

    def __init__(self, testbench_class: Testbench = Pendulum):
        super().__init__(
            testbench_class,
            vin=12.0,
            kp=self.DEFAULT_KP,
            error_gain=self.CURRENT_LIMIT / self.PWM_LIMIT,
        )

    def initialize(self) -> None:
        # Joint-level effective parameters.  The optimiser refines them from
        # logs recorded with known pendulum masses and lengths.
        self.model.kt = Parameter(0.5, 0.02, 5.0)
        self.model.R = Parameter(2.0, 0.05, 20.0)
        self.model.armature = Parameter(0.001, 0.0, 0.05)
        self.model.current_limit = Parameter(self.CURRENT_LIMIT, 0.1, 10.0)

        # Calibration corrections around the measured controller gains.
        self.model.control_gain_ratio = Parameter(1.0, 0.8, 1.2)
        self.model.velocity_gain_ratio = Parameter(1.0, 0.8, 1.2)

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        current = (
            (q_target - q)
            * self.kp
            * self.error_gain
            * self.model.control_gain_ratio.value
            + dq
            * self.VELOCITY_GAIN
            / self.PWM_LIMIT
            * self.CURRENT_LIMIT
            * self.model.velocity_gain_ratio.value
        )

        # The current loop is still constrained by the voltage the H-bridge can
        # produce at the present speed, then by its configured thermal limit.
        resistance = self.model.R.value
        back_emf = self.model.kt.value * dq
        current = self.backend.clamp(
            current,
            (-self.vin - back_emf) / resistance,
            (self.vin - back_emf) / resistance,
        )
        return self.backend.clamp(
            current,
            -self.model.current_limit.value,
            self.model.current_limit.value,
        )

    def get_extra_inertia(self) -> float:
        return self.model.armature.value
