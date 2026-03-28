"""
Isaac Lab Dual-Arm UR5E + XHand Environment

Self-contained environment for dual-arm manipulation with:
- Two UR5E robotic arms with 12-DOF XHand grippers
- Third-person RGBD camera
- Table workspace with optional object

Usage:
    # Must be launched via Isaac Lab:
    cd /path/to/IsaacLab
    ./isaaclab.sh -p /path/to/egocentric-xhand-isaaclab/xhand_isaaclab_env/env.py
    ./isaaclab.sh -p /path/to/egocentric-xhand-isaaclab/xhand_isaaclab_env/env.py --headless
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# Ensure package is importable when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xhand_isaaclab_env.config import (
    ASSET_DIR,
    CAMERA_FOVY_RAD,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROL_FREQ,
    FRAME_SKIP,
    JOINT_DAMPING,
    JOINT_STIFFNESS,
    LEFT_HOME_QPOS,
    LEFT_USD,
    LEFT_URDF,
    MAIN_CAMERA_POS,
    MAIN_CAMERA_TARGET,
    MAIN_CAMERA_UP,
    OBJECT_MESH,
    PHYSICS_DT,
    RIGHT_HOME_QPOS,
    RIGHT_USD,
    RIGHT_URDF,
    TABLE_HEIGHT,
    UR5E_JOINT_NAMES,
    USD_DIR,
    XHAND_LEFT_JOINT_NAMES,
    XHAND_RIGHT_JOINT_NAMES,
)


def _launch_app():
    """Parse args and launch Isaac Sim. Must be called before any Isaac imports."""
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Dual-Arm XHand Isaac Lab Env")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--no_object", action="store_true", help="Skip loading object")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True)
    args = parser.parse_args()

    launcher = AppLauncher(args)
    return launcher, args


# =============================================================================
# URDF-to-USD Conversion
# =============================================================================

def _convert_urdf_to_usd(urdf_path: str, usd_path: str) -> str:
    """Convert URDF to USD if not already cached."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    if os.path.exists(usd_path):
        return usd_path

    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    print(f"Converting URDF -> USD: {os.path.basename(urdf_path)}")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=JOINT_STIFFNESS,
                damping=JOINT_DAMPING,
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    return converter.usd_path


def _ensure_usd_assets() -> tuple[str, str]:
    """Ensure USD assets exist, converting from URDF if needed."""
    for path in (LEFT_URDF, RIGHT_URDF):
        if not os.path.exists(path):
            raise FileNotFoundError(f"URDF not found: {path}")

    left = _convert_urdf_to_usd(LEFT_URDF, LEFT_USD)
    right = _convert_urdf_to_usd(RIGHT_URDF, RIGHT_USD)
    return left, right


# =============================================================================
# Camera Utilities
# =============================================================================

def _compute_camera_quat(eye: tuple, target: tuple, up: tuple) -> tuple:
    """Compute a world-convention camera quaternion from a look-at pose."""
    from scipy.spatial.transform import Rotation as R

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera eye and target must be different.")
    forward = forward / forward_norm

    up = up / np.linalg.norm(up)
    left = np.cross(up, forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-8:
        raise ValueError("Camera up vector is parallel to the viewing direction.")
    left = left / left_norm
    up = np.cross(forward, left)

    rot = np.stack([forward, left, up], axis=1)
    q = R.from_matrix(rot).as_quat()  # (x, y, z, w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _compute_horizontal_aperture(fovy_rad: float, width: int, height: int) -> float:
    """Compute horizontal aperture from vertical FOV and aspect ratio."""
    aspect = width / height
    fovx_rad = 2 * math.atan(aspect * math.tan(fovy_rad / 2))
    # focal_length = horizontal_aperture / (2 * tan(fovx/2))
    # We pick focal_length=24mm (standard), then solve for aperture
    focal_length = 24.0
    return 2 * focal_length * math.tan(fovx_rad / 2)


# =============================================================================
# Environment
# =============================================================================

class DualArmXHandEnv:
    """
    Isaac Lab environment for dual-arm UR5E + XHand.

    Provides:
    - First-person RGBD observation from a robot-head viewpoint
    - Joint state observations for both arms (18 DOF each)
    - Position-controlled action interface (36 DOF total)
    """

    def __init__(self, sim, args, num_envs: int = 1, with_object: bool = True):
        """
        Initialize the environment.

        Args:
            sim: Isaac Sim SimulationApp instance
            args: Parsed CLI arguments
            num_envs: Number of parallel environments
            with_object: Whether to place an object on the table
        """
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        self.num_envs = num_envs
        self.with_object = with_object
        self._torch = torch
        self._sim_utils = sim_utils

        # Store class refs for later use
        self._ArticulationCfg = ArticulationCfg
        self._ImplicitActuatorCfg = ImplicitActuatorCfg
        self._RigidObjectCfg = RigidObjectCfg
        self._RigidObject = RigidObject
        self._CameraCfg = CameraCfg
        self._Camera = Camera

        # Convert/load USD assets
        print("Loading robot assets...")
        self.left_usd, self.right_usd = _ensure_usd_assets()

        # Setup simulation context
        device = getattr(args, "device", "cuda:0")
        sim_cfg = SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=FRAME_SKIP,
            device=device,
        )
        self.sim = SimulationContext(sim_cfg)
        self.device = device

        # Set viewport to the main third-person view
        try:
            self.sim.set_camera_view(
                eye=MAIN_CAMERA_POS,
                target=MAIN_CAMERA_TARGET,
            )
        except Exception:
            pass

        # Build scene
        self._build_scene()

        # Warm up
        print("Warming up simulation...")
        home_action = np.concatenate([LEFT_HOME_QPOS, RIGHT_HOME_QPOS])
        for _ in range(30):
            self.step(action=home_action, get_obs=False)

        print("Environment ready.")

    # -----------------------------------------------------------------
    # Scene construction
    # -----------------------------------------------------------------

    def _build_scene(self):
        sim_utils = self._sim_utils

        # Dark floor slab instead of the default ground-plane mesh.
        floor_cfg = sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.20),
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
        )
        floor_cfg.func("/World/Floor", floor_cfg, translation=(0.75, 0.0, -0.01))

        # Lights
        dome = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
        dome.func("/World/DomeLight", dome)
        pt = sim_utils.SphereLightCfg(intensity=50000.0, color=(1.0, 1.0, 1.0), radius=0.1)
        pt.func("/World/PointLight1", pt, translation=(2.0, 2.0, 4.0 + TABLE_HEIGHT))
        pt.func("/World/PointLight2", pt, translation=(2.0, -2.0, 4.0 + TABLE_HEIGHT))

        # Table
        table_cfg = sim_utils.CuboidCfg(
            size=(1.8, 2.1, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.97, 0.89, 0.66),
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
        )
        table_cfg.func("/World/Table", table_cfg,
                        translation=(0.75, 0.0, TABLE_HEIGHT - 0.03))

        # Robots
        from isaaclab.assets import Articulation
        self.robot_left = Articulation(self._make_robot_cfg(
            self.left_usd, "/World/RobotLeft", (0.0, 0.45, TABLE_HEIGHT), is_left=True))
        self.robot_right = Articulation(self._make_robot_cfg(
            self.right_usd, "/World/RobotRight", (0.0, -0.45, TABLE_HEIGHT), is_left=False))

        # Main third-person RGBD camera
        self.camera = self._make_camera()

        # Object
        self.object = None
        if self.with_object:
            self._spawn_object()

        # Reset sim to apply all prims
        self.sim.reset()
        self.robot_left.reset()
        self.robot_right.reset()
        self.camera.reset()

        print("Scene built.")

    def _make_robot_cfg(self, usd_path, prim_path, pos, is_left):
        sim_utils = self._sim_utils
        home = LEFT_HOME_QPOS if is_left else RIGHT_HOME_QPOS
        hand_names = XHAND_LEFT_JOINT_NAMES if is_left else XHAND_RIGHT_JOINT_NAMES
        all_names = UR5E_JOINT_NAMES + hand_names
        joint_pos = {n: float(home[i]) for i, n in enumerate(all_names)}

        return self._ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=25,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=self._ArticulationCfg.InitialStateCfg(
                pos=pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=joint_pos,
            ),
            actuators={
                "arm": self._ImplicitActuatorCfg(
                    joint_names_expr=UR5E_JOINT_NAMES,
                    stiffness=JOINT_STIFFNESS,
                    damping=JOINT_DAMPING,
                ),
                "hand": self._ImplicitActuatorCfg(
                    joint_names_expr=hand_names,
                    stiffness=JOINT_STIFFNESS,
                    damping=JOINT_DAMPING,
                ),
            },
        )

    def _make_camera(self):
        sim_utils = self._sim_utils
        CameraCfg = self._CameraCfg

        h_aperture = _compute_horizontal_aperture(CAMERA_FOVY_RAD, CAMERA_WIDTH, CAMERA_HEIGHT)
        focal_length = 24.0

        quat = _compute_camera_quat(
            MAIN_CAMERA_POS,
            MAIN_CAMERA_TARGET,
            MAIN_CAMERA_UP,
        )

        cfg = CameraCfg(
            prim_path="/World/Cameras/MainView",
            update_period=0.0,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=h_aperture,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=MAIN_CAMERA_POS,
                rot=quat,
                convention="world",
            ),
        )
        return self._Camera(cfg)

    def _spawn_object(self):
        sim_utils = self._sim_utils

        # Use placeholder sphere (OBJ -> USD conversion not yet implemented)
        cfg = self._RigidObjectCfg(
            prim_path="/World/Object",
            spawn=sim_utils.SphereCfg(
                radius=0.05,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=20.0,
                    angular_damping=20.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2),
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                ),
            ),
            init_state=self._RigidObjectCfg.InitialStateCfg(
                pos=(0.80, 0.0, TABLE_HEIGHT + 0.05),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        try:
            self.object = self._RigidObject(cfg)
        except Exception as e:
            print(f"Warning: Could not spawn object: {e}")
            self.with_object = False

    # -----------------------------------------------------------------
    # Step & Observations
    # -----------------------------------------------------------------

    def apply_action(self, action: np.ndarray):
        """
        Apply joint position targets.

        Args:
            action: (36,) array [left_arm6 + left_hand12 + right_arm6 + right_hand12]
        """
        torch = self._torch
        left = torch.tensor(action[:18], dtype=torch.float32, device=self.device).unsqueeze(0)
        right = torch.tensor(action[18:], dtype=torch.float32, device=self.device).unsqueeze(0)

        self.robot_left.set_joint_position_target(left)
        self.robot_right.set_joint_position_target(right)
        self.robot_left.write_data_to_sim()
        self.robot_right.write_data_to_sim()

    def step(self, action: np.ndarray | None = None, get_obs: bool = True) -> dict:
        """
        Step simulation. Returns observations if get_obs=True.

        Args:
            action: Optional (36,) joint position targets
            get_obs: Whether to return observations

        Returns:
            Dict with 'qpos_0', 'qpos_1', 'rgb', 'depth' keys
        """
        if action is not None:
            self.apply_action(action)

        self.sim.step()
        self.robot_left.update(self.sim.get_physics_dt())
        self.robot_right.update(self.sim.get_physics_dt())

        if get_obs:
            return self.get_obs()
        return {}

    def get_obs(self) -> dict:
        """
        Get observations.

        Returns:
            Dict with:
            - 'qpos_0': (18,) left arm joint positions
            - 'qpos_1': (18,) right arm joint positions
            - 'rgb': (H, W, 3) uint8 color image
            - 'depth': (H, W) float32 depth in meters
        """
        obs = {}

        # Joint positions
        obs["qpos_0"] = self.robot_left.data.joint_pos[0].cpu().numpy()
        obs["qpos_1"] = self.robot_right.data.joint_pos[0].cpu().numpy()

        # Camera RGBD
        self.camera.update(self.sim.get_physics_dt())

        if hasattr(self.camera.data, "output") and self.camera.data.output is not None:
            output = self.camera.data.output

            # RGB
            if "rgb" in output:
                rgb = output["rgb"][0].cpu().numpy()
                if rgb.shape[-1] == 4:
                    rgb = rgb[:, :, :3]
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                else:
                    rgb = rgb.clip(0, 255).astype(np.uint8)
                obs["rgb"] = rgb

            # Depth
            if "distance_to_image_plane" in output:
                depth = output["distance_to_image_plane"][0].cpu().numpy()
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                obs["depth"] = depth.astype(np.float32)

        return obs

    def reset(self) -> dict:
        """Reset robots to home positions and return observations."""
        torch = self._torch
        left_pos = torch.tensor(LEFT_HOME_QPOS, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        right_pos = torch.tensor(RIGHT_HOME_QPOS, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        zeros_l = torch.zeros_like(left_pos)
        zeros_r = torch.zeros_like(right_pos)

        self.robot_left.write_joint_state_to_sim(left_pos, zeros_l)
        self.robot_right.write_joint_state_to_sim(right_pos, zeros_r)

        self.sim.step()
        self.robot_left.update(self.sim.get_physics_dt())
        self.robot_right.update(self.sim.get_physics_dt())

        return self.get_obs()

    def run_viewer(self, simulation_app):
        """Run interactive viewer loop (for non-headless mode)."""
        print("\nViewer running. Close window or press ESC to exit.")
        home_action = np.concatenate([LEFT_HOME_QPOS, RIGHT_HOME_QPOS])
        step_count = 0
        while simulation_app.is_running():
            self.step(action=home_action, get_obs=False)
            step_count += 1
            if step_count % 200 == 0:
                print(f"  step {step_count}")


# =============================================================================
# Script entry point
# =============================================================================

def main():
    launcher, args = _launch_app()
    simulation_app = launcher.app

    env = DualArmXHandEnv(
        sim=simulation_app,
        args=args,
        num_envs=args.num_envs,
        with_object=not args.no_object,
    )

    if args.headless:
        # Headless: take one observation and print summary
        obs = env.get_obs()
        print("\nObservation summary:")
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        print("\nDone.")
    else:
        env.run_viewer(simulation_app)

    simulation_app.close()


if __name__ == "__main__":
    main()
