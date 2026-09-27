"""Default initial poses for VLA inference.

These arrays are sent to the C++ control loop when the user presses 'i'
to move the robot to a known starting configuration before inference begins.

WARNING: The initial motion token below is specific to the SONIC checkpoint used
during training. Different SONIC checkpoints encode different latent spaces, so
this token will produce a different (and likely incorrect) pose if you switch to
a different SONIC checkpoint. When changing the SONIC checkpoint, you MUST update
LATENT_INITIAL_MOTION_TOKEN to a value that corresponds to a known safe standing
pose in the new checkpoint's latent space.
"""

import numpy as np


def sample_upper_body_transition(start, target, elapsed: float, duration: float):
    """Cubic position reference with consistent velocity and zero endpoint speed."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("transition duration must be positive and finite")
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if start.shape != (17,) or target.shape != (17,):
        raise ValueError("upper-body transition requires 17 joint angles")
    if not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("upper-body joint angles must be finite")
    t = float(np.clip(elapsed / duration, 0.0, 1.0))
    delta = target - start
    return start + t*t*(3.0-2.0*t)*delta, (6.0*t*(1.0-t)/duration)*delta

# 64-dim motion token for a stable standing pose.
# CHECKPOINT-SPECIFIC: this value must be updated if the SONIC checkpoint changes.
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [
        -0.0625,  0.0000, -0.0625, -0.1250, -0.1875, -0.0625,  0.1875,
         0.2500,  0.1875, -0.1250,  0.0625, -0.0625, -0.2500, -0.2500,
        -0.3125, -0.0625,  0.0000, -0.0625, -0.1250, -0.1875,  0.0000,
        -0.2500,  0.0000, -0.2500, -0.0625,  0.0625,  0.1250, -0.1250,
         0.2500,  0.1875,  0.2500, -0.1250,  0.1250,  0.1875, -0.0625,
         0.0000, -0.1875, -0.1875,  0.2500,  0.0000,  0.0000, -0.1250,
         0.0625,  0.0000, -0.0625, -0.0625,  0.1875, -0.0625,  0.0000,
         0.0625,  0.1250,  0.0625,  0.1250,  0.0625,  0.1250,  0.0000,
         0.1250,  0.1875,  0.0000,  0.0000,  0.0625,  0.0625,  0.1875,
         0.0625,
    ],
    dtype=np.float32,
)

# User-selected measured pose, captured 2026-09-22 15:27 Asia/Shanghai.
# Median of 201 body_q_measured frames; absolute radians in planner order.
# Raw capture: outputs/pose_snapshots/20260922_152740/
VLA_INITIAL_UPPER_BODY_RAD: list[float] = [
    -0.01566852, -0.00589521, 0.15957156,  # waist yaw, roll, pitch
    0.29736459, 0.25334653,    # shoulder pitch (left, right)
    0.20228174, -0.18484469,   # shoulder roll
    -0.24355541, 0.23568177,   # shoulder yaw
    0.06701579, 0.14107829,    # elbow
    -0.74939758, 0.46938616,   # wrist roll
    0.04104597, -0.08275107,   # wrist pitch
    -0.05149622, -0.00100667,  # wrist yaw
]

SONIC_STAND_UPPER_BODY_RAD: list[float] = [
    0.0, 0.0, 0.0,
    0.2, 0.2,
    0.2, -0.2,
    0.0, 0.0,
    0.6, 0.6,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
]

UPPER_BODY_MUJOCO_INDICES = [
    12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
]


def measured_upper_body_from_state(state):
    """Use absolute hardware/MuJoCo angles, never normalized policy body_q."""
    if state is None or "body_q_measured" not in state:
        raise ValueError("missing body_q_measured for POSE entry")
    joints = np.asarray(state["body_q_measured"], dtype=np.float32)
    if joints.shape != (29,) or not np.isfinite(joints).all():
        raise ValueError("body_q_measured must contain 29 finite joint angles")
    return joints[UPPER_BODY_MUJOCO_INDICES].copy()
