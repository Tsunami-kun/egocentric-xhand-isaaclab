"""Constants and configuration for dual-arm UR5E + XHand Isaac Lab environment."""

import math
import os

import numpy as np


# =============================================================================
# Paths
# =============================================================================

# Base path: parent of this package
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PACKAGE_DIR)

ASSET_DIR = os.path.join(_PROJECT_DIR, "assets")
URDF_DIR = os.path.join(ASSET_DIR, "ur5e_with_xhand_urdf_offset_sim2real")
USD_DIR = os.path.join(ASSET_DIR, "ur5e_with_xhand_usd")

LEFT_URDF = os.path.join(URDF_DIR, "ur5e_with_xhand_left_limited_joint.urdf")
RIGHT_URDF = os.path.join(URDF_DIR, "ur5e_with_xhand_right_limited_joint.urdf")

LEFT_USD = os.path.join(USD_DIR, "ur5e_with_xhand_left.usd")
RIGHT_USD = os.path.join(USD_DIR, "ur5e_with_xhand_right.usd")

OBJECT_MESH = os.path.join(ASSET_DIR, "simplified.obj")


# =============================================================================
# Physics & Control
# =============================================================================

TABLE_HEIGHT = 0.714
PHYSICS_DT = 1.0 / 240.0  # 240 Hz physics
CONTROL_FREQ = 20  # 20 Hz control
FRAME_SKIP = 12  # 240 / 20

JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 100.0


# =============================================================================
# Joint Configuration
# =============================================================================

# UR5E Home Joint Positions (radians)
UR5E_LEFT_HOME = np.array([
    -math.pi / 2,           # shoulder_pan_joint
    -math.pi * 9 / 16,      # shoulder_lift_joint
    -math.pi * 7 / 16,      # elbow_joint
    math.pi * 7 / 8,        # wrist_1_joint
    -math.pi / 2,           # wrist_2_joint
    -math.pi / 4,           # wrist_3_joint
])

UR5E_RIGHT_HOME = np.array([
    math.pi / 2,
    -math.pi * 7 / 16,
    math.pi * 7 / 16,
    math.pi / 8,
    math.pi / 2,
    math.pi / 4,
])

# XHand Home (12 DOF, DEFAULT order)
XHAND_HOME = np.array([
    20, 20, 20,   # thumb: bend, rota1, rota2
    5, 20, 20,    # index: bend, joint1, joint2
    20, 20,       # mid: joint1, joint2
    20, 20,       # ring: joint1, joint2
    20, 20,       # pinky: joint1, joint2
]) * math.pi / 180

# Full 18-DOF home positions per arm
LEFT_HOME_QPOS = np.concatenate([UR5E_LEFT_HOME, XHAND_HOME])
RIGHT_HOME_QPOS = np.concatenate([UR5E_RIGHT_HOME, XHAND_HOME])


# =============================================================================
# Joint Names
# =============================================================================

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

XHAND_LEFT_JOINT_NAMES = [
    "left_hand_thumb_bend_joint",
    "left_hand_thumb_rota_joint1",
    "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint",
    "left_hand_index_joint1",
    "left_hand_index_joint2",
    "left_hand_mid_joint1",
    "left_hand_mid_joint2",
    "left_hand_ring_joint1",
    "left_hand_ring_joint2",
    "left_hand_pinky_joint1",
    "left_hand_pinky_joint2",
]

XHAND_RIGHT_JOINT_NAMES = [
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
]


# =============================================================================
# Camera Configuration
# =============================================================================

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1440
CAMERA_FOVY_RAD = math.radians(70.0)

# Virtual robot-head frame centered between the two arm bases.
# Move the eye point substantially closer to the manipulation zone while
# keeping a high human-over-desk viewing angle.
FIRST_PERSON_CAMERA_POS = (-0.02, 0.0, TABLE_HEIGHT + 1.32)
FIRST_PERSON_CAMERA_TARGET = (0.34, 0.0, TABLE_HEIGHT - 0.01)
FIRST_PERSON_CAMERA_UP = (0.0, 0.0, 1.0)
