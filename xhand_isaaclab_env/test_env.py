#!/usr/bin/env python3
"""
Test script: launch headless, save RGBD screenshots.

Usage:
    cd /path/to/IsaacLab
    ./isaaclab.sh -p /path/to/egocentric-xhand-isaaclab/xhand_isaaclab_env/test_env.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure package importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xhand_isaaclab_env.env import _launch_app

# Launch app (parses args, starts Isaac Sim)
launcher, args = _launch_app()
simulation_app = launcher.app

# Now safe to import the rest
from xhand_isaaclab_env.env import DualArmXHandEnv
from xhand_isaaclab_env.config import LEFT_HOME_QPOS, RIGHT_HOME_QPOS

print("=" * 60)
print("RGBD Screenshot Test")
print("=" * 60)

env = DualArmXHandEnv(
    sim=simulation_app,
    args=args,
    with_object=not getattr(args, "no_object", False),
)

# Step a few more times to let rendering settle
home_action = np.concatenate([LEFT_HOME_QPOS, RIGHT_HOME_QPOS])
for _ in range(10):
    env.step(action=home_action, get_obs=False)

obs = env.get_obs()

# Save outputs
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
os.makedirs(out_dir, exist_ok=True)

from PIL import Image

if "rgb" in obs:
    rgb = obs["rgb"]
    Image.fromarray(rgb).save(os.path.join(out_dir, "rgb.png"))
    print(f"Saved rgb.png  shape={rgb.shape} dtype={rgb.dtype}")

if "depth" in obs:
    depth = obs["depth"]
    # Normalize depth for visualization (clip to 3m, map to 0-255)
    depth_vis = depth.copy()
    depth_vis[depth_vis > 3.0] = 3.0
    depth_vis[depth_vis < 0.01] = 0.0
    if depth_vis.max() > 0:
        depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
    Image.fromarray(depth_vis).save(os.path.join(out_dir, "depth.png"))
    # Also save raw depth as .npy
    np.save(os.path.join(out_dir, "depth_raw.npy"), depth)
    print(f"Saved depth.png + depth_raw.npy  shape={depth.shape} range=[{depth.min():.3f}, {depth.max():.3f}]m")

# Print joint state summary
print(f"\nqpos_0 (left):  {obs['qpos_0'][:6]}  (arm)")
print(f"qpos_1 (right): {obs['qpos_1'][:6]}  (arm)")
print(f"\nOutputs saved to: {out_dir}")

simulation_app.close()
