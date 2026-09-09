"""Pendulum test bench for Roki4 identification experiments."""

import numpy as np

from bam.testbench import Testbench


class Roki4Pendulum(Testbench):
    """Single-axis pendulum with optional manually calculated inertia.

    Required log fields are ``mass``, ``arm_mass`` and ``length``.  When the
    log also contains ``inertia`` [kg m²], it replaces BAM's point-mass plus
    uniform-rod calculation.  ``gravity_coefficient`` [Nm] may likewise be
    provided to replace the default ``(mass + arm_mass/2) * g * length``.

    Angle zero must be the arm pointing down.  Positive angle convention must
    agree with the measured encoder data.
    """

    def __init__(self, log: dict):
        self.mass = log["mass"]
        self.arm_mass = log["arm_mass"]
        self.length = log["length"]
        self.inertia = log.get("inertia")
        self.gravity_coefficient = log.get(
            "gravity_coefficient",
            (self.mass + self.arm_mass / 2.0) * 9.80665 * self.length,
        )

    def compute_mass(self, q: float, dq: float) -> float:
        if self.inertia is not None:
            return self.inertia
        return self.mass * self.length**2 + (self.arm_mass / 3.0) * self.length**2

    def compute_bias(self, q: float, dq: float) -> float:
        return -self.gravity_coefficient * np.sin(q)
