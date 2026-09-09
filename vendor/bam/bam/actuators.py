# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from .testbench import Pendulum
from .erob.actuator import ErobActuator
from .dynamixel.actuator import (
    MXActuator,
    XL320Actuator,
    XL330Actuator,
    XL330CurrentActuator,
)
from .feetech.actuator import STS3215Actuator
from .unitree.actuator import UnitreeGo1Actuator
from .roki4.actuator import Roki4Actuator
from .roki4.testbench import Roki4Pendulum

actuators = {
    # Dynamixel MX series
    "mx64": lambda: MXActuator(Pendulum),
    "mx106": lambda: MXActuator(Pendulum),
    # Dynamixel XL series
    "xl320": lambda: XL320Actuator(Pendulum),
    "xl330": lambda: XL330Actuator(Pendulum),
    "xl330i": lambda: XL330CurrentActuator(Pendulum),
    # eRob actuators with custom PD controller
    "erob80_100": lambda: ErobActuator(Pendulum, damping=2.0),
    "erob80_50": lambda: ErobActuator(Pendulum, damping=1.0),
    # Feetech STS3215
    "sts3215": lambda: STS3215Actuator(Pendulum),
    # Unitree Go1
    "unitree_go1": lambda: UnitreeGo1Actuator(Pendulum),
    # IronArt Roki4 controller; recorded through the right shoulder channel.
    "SKS2401": lambda: Roki4Actuator(Roki4Pendulum),
}
