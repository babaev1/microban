# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import numpy as np

MOTOR_TO_ID = {
    "left_hip_yaw": 11,
    "left_hip_roll": 12,
    "left_hip_pitch": 13,
    "left_knee": 14,
    "left_ankle_pitch": 15,
    "left_ankle_roll": 16,
    "right_hip_yaw": 21,
    "right_hip_roll": 22,
    "right_hip_pitch": 23,
    "right_knee": 24,
    "right_ankle_pitch": 25,
    "right_ankle_roll": 26,
    "left_shoulder_pitch": 31,
    "left_shoulder_roll": 32,
    #"left_elbow": 33,
    "right_shoulder_pitch": 41,
    "right_shoulder_roll": 42,
    #"right_elbow": 43,
    #"head": 51,
}

ID_TO_MOTOR = {v: k for k, v in MOTOR_TO_ID.items()}

NEUTRAL_POSE = {
    "left_hip_yaw": float(np.deg2rad(0.0)),
    "left_hip_roll": float(np.deg2rad(0.0)),
    "left_hip_pitch": float(np.deg2rad(-15.0)),
    "left_knee": float(np.deg2rad(30.0)),
    "left_ankle_pitch": float(np.deg2rad(-15.0)),
    "left_ankle_roll": float(np.deg2rad(0.0)),
    "right_hip_yaw": float(np.deg2rad(0.0)),
    "right_hip_roll": float(np.deg2rad(0.0)),
    "right_hip_pitch": float(np.deg2rad(-15.0)),
    "right_knee": float(np.deg2rad(30.0)),
    "right_ankle_pitch": float(np.deg2rad(-15.0)),
    "right_ankle_roll": float(np.deg2rad(0.0)),
    "left_shoulder_pitch": float(np.deg2rad(0.0)),
    "left_shoulder_roll": float(np.deg2rad(0.0)),
    #"left_elbow": float(np.deg2rad(0.0)),
    "right_shoulder_pitch": float(np.deg2rad(0.0)),
    "right_shoulder_roll": float(np.deg2rad(0.0)),
    #"right_elbow": float(np.deg2rad(0.0)),
    #"head": float(np.deg2rad(0.0)),
}
NEUTRAL_POSE = {
    "left_hip_yaw": float(np.deg2rad(0.0)),
    "left_hip_roll": float(np.deg2rad(0.0)),
    "left_hip_pitch": float(np.deg2rad(0.0)),
    "left_knee": float(np.deg2rad(0.0)),
    "left_ankle_pitch": float(np.deg2rad(0.0)),
    "left_ankle_roll": float(np.deg2rad(0.0)),
    "right_hip_yaw": float(np.deg2rad(0.0)),
    "right_hip_roll": float(np.deg2rad(0.0)),
    "right_hip_pitch": float(np.deg2rad(0.0)),
    "right_knee": float(np.deg2rad(0.0)),
    "right_ankle_pitch": float(np.deg2rad(0.0)),
    "right_ankle_roll": float(np.deg2rad(0.0)),
    "left_shoulder_pitch": float(np.deg2rad(0.0)),
    "left_shoulder_roll": float(np.deg2rad(0.0)),
    #"left_elbow": float(np.deg2rad(0.0)),
    "right_shoulder_pitch": float(np.deg2rad(0.0)),
    "right_shoulder_roll": float(np.deg2rad(0.0)),
    #"right_elbow": float(np.deg2rad(0.0)),
    #"head": float(np.deg2rad(0.0)),
}

MOTOR_SIGN = {
    "left_hip_yaw": -1.0,
    "left_hip_roll": 1.0,
    "left_hip_pitch": -1.0,
    "left_knee": 1.0,
    "left_ankle_pitch": 1.0,
    "left_ankle_roll": -1.0,
    "right_hip_yaw": -1.0,
    "right_hip_roll": 1.0,
    "right_hip_pitch": 1.0,
    "right_knee": -1.0,
    "right_ankle_pitch": -1.0,
    "right_ankle_roll": -1.0,
    "left_shoulder_pitch": 1.0,
    "left_shoulder_roll": -1.0,
    "left_elbow": 1.0,
    "right_shoulder_pitch": -1.0,
    "right_shoulder_roll": -1.0,
    "right_elbow": -1.0,
    "head": 1.0,
}

# Position P Gain (Dynamixel register value)
KP_DEFAULT: int = 7334       #400        # ~0.886 Nm/rad in MuJoCo
# Reverted back to 125: it was raised to 1000 (and OVERCURRENT_CUTOFF_A to 60A alongside
# it) while chasing what looked like the robot being too soft to hold the walk policy's
# crouch — but that whole symptom was very likely a side effect of the IMU_MOUNT_QUAT
# regression documented further down this file (now reverted), feeding the policy a badly
# wrong orientation, not an actual torque shortfall. Re-test with the IMU fix reverted
# before touching this again.
KP_RL: int = 1334            #125             # ~0.277 Nm/rad in MuJoCo
KP_GAIN_PRM: float = 0.0022  # Nm/rad per register unit (for Xl330)

# BAM motor model (bam package, XL330 m6)
BAM_VIN: float = 11.0       #7.5
BAM_VIN_MIN: float = 9.0    #6.0
BAM_VOLTAGE_DROP_GAIN: float = 0.2
BAM_MAX_CURRENT: float = 6.8   #1.75 # XL330 firmware current limit [A]: clips motor torque to ±BAM_MAX_CURRENT * kt

# Overcurrent safety: emergency torque-off when the summed |present_current| of all
# motors stays above OVERCURRENT_CUTOFF_A for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.
# Goal: cut the robot before a current spike (e.g. all motors snapping during a fall) trips the BMS.
PRESENT_CURRENT_UNIT_A: float = 0.001   # XL330 present_current register unit (1.0 mA/LSB)
# Reverted back to 15.0 alongside KP_RL — was raised to 60A to cover a "sustained high
# current" reading that was itself measured under the (now-reverted) bad IMU_MOUNT_QUAT,
# so that data point is no longer trustworthy. Re-test with the IMU fix reverted first.
OVERCURRENT_CUTOFF_A: float = 105.0      # total pack current threshold (CALIBRATE: below BMS trip, above normal walk peak)
OVERCURRENT_DEBOUNCE_TICKS: int = 2     # consecutive over-threshold ticks before cutting

# Current proxy used when present_current is NOT read (Observer.observe_current = False), so the
# safety needs no extra bus transaction. Reproduces the bam XL330 m6 voltage-controlled model from
# data already read (present_position, present_velocity) and the command target:
#   duty = clip(PROXY_KP * PROXY_ERROR_GAIN * (target - q), ±PROXY_MAX_PWM)
#   I    = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R      then |I| capped at BAM_MAX_CURRENT
PROXY_KT: float = 0.366                  # XL330 m6 torque constant [Nm/A]
PROXY_R: float = 2.811                   # XL330 m6 motor resistance [Ohm]
PROXY_VIN: float = BAM_VIN               # supply voltage [V]
PROXY_ERROR_GAIN: float = 0.0028773775   # duty cycle per (kp * rad), XL330 encoder/gain scaling
PROXY_MAX_PWM: float = 1.0               # max duty cycle magnitude
PROXY_KP: int = KP_RL                    # firmware P gain assumed by the proxy (walking regime)
OVERCURRENT_PROXY_DELAY_TICKS: int = 3   # number of ticks to delay the proxy current estimate

# Velocity command limits [m/s, m/s, rad/s], applied centrally to every input source.
# Input sources emit normalized commands in [-1, 1]; scale_velocity() maps them to these.
# Rotation gets a wider range when turning in place (vx = vy = 0) than while translating.
VX_MAX: float = 0.7
VX_MAX_BACKWARD: float = 0.5  # backward (vx < 0) is capped lower than forward
VY_MAX: float = 0.3
VTHETA_MAX_STATIONARY: float = 3.0
VTHETA_MAX_MOVING: float = 1.5

# IMU (BMI088) I2C bus number on the Raspberry Pi
IMU_I2C_BUS: int = 1

# Rotation from trunk frame (body) to IMU sensor frame — the mjcf "imu" site's own local
# quat (robot.xml). This was briefly (and wrongly) changed to (0.70710678, 0, 0, 0.70710678)
# while debugging an apparent gravity-axis mismatch; that diagnosis was itself based on a
# test harness that hardcoded a stale SPAWN_TRUNK_QUAT literal instead of computing it from
# NEUTRAL_POSE (mujoco_controller.py derives it as -NEUTRAL_POSE["left_hip_pitch"], which is
# 0 in the current all-zero NEUTRAL_POSE, i.e. spawn orientation is identity) — recomputed
# correctly, THIS value already gives projected_gravity=(0,0,-1) at a true upright spawn, so
# it was correct all along. Re-verify against the real BMI088 mounting if that hasn't been
# done — this constant is shared by sim and real hardware — but don't re-derive it from sim
# without re-deriving SPAWN_TRUNK_QUAT the same way mujoco_controller.py does.
IMU_MOUNT_QUAT: tuple[float, float, float, float] = (0.5, -0.5, -0.5, 0.5)

# NOTE: the walk RL policy's DoF ordering used to be hardcoded here as
# OBSERVATION_DOF_ORDER, but it didn't match the order the policy was actually
# trained on (its ONNX metadata's joint_names) — that mismatch misapplied
# actions to the wrong joints and was the root cause of the walk-start
# overcurrent trips. WalkMove now reads the order from the model's own
# metadata (self._dof_order in moves/walk.py) instead of a separate constant
# that can drift out of sync with whatever agent .onnx file is loaded.