# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import M20_Piper.tasks.manager_based.m20_piper.mdp as mdp
from M20_Piper.assets.m20_piper import M20_PIPER_CFG


##
# Scene definition
##

@configclass
class M20PiperSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = M20_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.0), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    final_goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FinalGoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.16,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.25), roughness=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.16)),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + mdp.base_link_name,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + mdp.base_link_name,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##



@configclass
class ActionsCfg:
    """PPO controls wheels + Piper arm.

    Action dimension:
        4 wheel velocity actions
        6 arm joint position actions
        total = 10
    """

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=mdp.wheel_joint_names,
        scale=0.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )

    # arm_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=mdp.arm_joint_names,
    #     use_default_offset=True,
    #     preserve_order=True,

    #     # These are action-to-joint-position scales around default pose.
    #     # Default arm pose from m20_piper.py:
    #     # joint1=0.0, joint2=0.2, joint3=-0.35, joint4=0.0, joint5=0.2, joint6=0.0
    #     #
    #     # Sweep target should be reachable:
    #     # joint1≈-0.9, joint2≈0.35, joint3≈-1.10, joint5≈0.80
    #     scale={
    #         "joint1": 1.2,
    #         "joint2": 0.5,
    #         "joint3": 1.0,
    #         "joint4": 1.2,
    #         "joint5": 0.8,
    #         "joint6": 1.20,
    #     },

    #     clip={
    #         "joint1": (-1.20, 1.20),
    #         "joint2": (-1.00, 1.00),
    #         "joint3": (-1.00, 1.00),
    #         "joint4": (-1.00, 1.00),
    #         "joint5": (-1.00, 1.00),
    #         "joint6": (-1.00, 1.00),
    #     },
    # )
    arm_pos = mdp.PiperSweepActionCfg(
        asset_name="robot",
        joint_names=mdp.arm_joint_names,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-2.0, 2.0),
            scale=1.0,
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        arm_joint_pos = ObsTerm(
            func=mdp.arm_joint_pos_rel_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-3.0, 3.0),
            scale=1.0,
        )

        arm_joint_vel = ObsTerm(
            func=mdp.arm_joint_vel_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        front_scan = ObsTerm(
            func=mdp.front_lidar_scan_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "num_rays": 181,
                "fov_deg": 270.0,
                "max_range": 5.0,
                "normalize": True,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        front_blocked = ObsTerm(
            func=mdp.lidar_sector_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "num_rays": 181,
                "fov_deg": 270.0,
                "max_range": 5.0,
                "trigger_range": 1.50,
                "sector_center_deg": 0.0,
                "sector_half_width_deg": 20.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        wide_blocked = ObsTerm(
            func=mdp.lidar_sector_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "num_rays": 181,
                "fov_deg": 270.0,
                "max_range": 5.0,
                "trigger_range": 1.75,
                "sector_center_deg": 0.0,
                "sector_half_width_deg": 120.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        path_corridor_blocked = ObsTerm(
            func=mdp.path_corridor_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "min_ahead_m": 0.30,
                "max_ahead_m": 3.00,
                "corridor_half_width": 0.30,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        arm_reach_blocked = ObsTerm(
            func=mdp.arm_reach_path_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "arm_min_m": 0.25,
                "arm_max_m": 1.10,
                "corridor_half_width": 0.30,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        # --- Path tracking observations ---
        path_window = ObsTerm(
            func=mdp.local_path_window,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_points": 8,
                "normalize_dist": 4.0,
            },
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        path_heading = ObsTerm(
            func=mdp.path_heading_error,
            params={"asset_cfg": SceneEntityCfg("robot"), "lookahead": 4},
            clip=(-math.pi, math.pi),
            scale=1.0,
        )
        path_cte = ObsTerm(
            func=mdp.path_cross_track_error,
            params={"asset_cfg": SceneEntityCfg("robot"), "lookahead": 4, "normalize_dist": 2.0},
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        dist_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"asset_cfg": SceneEntityCfg("robot"), "normalize_dist": 8.0},
            clip=(0.0, 2.0),
            scale=1.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        arm_joint_pos = ObsTerm(
            func=mdp.arm_joint_pos_rel_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-3.0, 3.0),
            scale=1.0,
        )

        arm_joint_vel = ObsTerm(
            func=mdp.arm_joint_vel_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        front_scan = ObsTerm(
            func=mdp.front_lidar_scan_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "num_rays": 181,
                "fov_deg": 270.0,
                "max_range": 5.0,
                "normalize": True,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        front_blocked = ObsTerm(
            func=mdp.lidar_sector_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "num_rays": 181,
                "fov_deg": 270.0,
                "max_range": 5.0,
                "trigger_range": 1.50,
                "sector_center_deg": 0.0,
                "sector_half_width_deg": 20.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        wide_blocked = ObsTerm(
            func=mdp.lidar_sector_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "num_rays": 181,
                "fov_deg": 270.0,
                "max_range": 5.0,
                "trigger_range": 1.75,
                "sector_center_deg": 0.0,
                "sector_half_width_deg": 120.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        path_corridor_blocked = ObsTerm(
            func=mdp.path_corridor_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "min_ahead_m": 0.30,
                "max_ahead_m": 3.00,
                "corridor_half_width": 0.30,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        arm_reach_blocked = ObsTerm(
            func=mdp.arm_reach_path_blocked_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_name": "obstacle",
                "arm_min_m": 0.25,
                "arm_max_m": 1.10,
                "corridor_half_width": 0.30,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        path_window = ObsTerm(
            func=mdp.local_path_window,
            params={"asset_cfg": SceneEntityCfg("robot"), "num_points": 8, "normalize_dist": 4.0},
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        path_heading = ObsTerm(
            func=mdp.path_heading_error,
            params={"asset_cfg": SceneEntityCfg("robot"), "lookahead": 4},
            clip=(-math.pi, math.pi),
            scale=1.0,
        )
        path_cte = ObsTerm(
            func=mdp.path_cross_track_error,
            params={"asset_cfg": SceneEntityCfg("robot"), "lookahead": 4, "normalize_dist": 2.0},
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        dist_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"asset_cfg": SceneEntityCfg("robot"), "normalize_dist": 8.0},
            clip=(0.0, 2.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # randomize_reset_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    # )

    # randomize_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,  # type: ignore
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.wheel_joint_names),
    #         "stiffness_distribution_params": (0.85, 1.15),
    #         "damping_distribution_params": (0.85, 1.15),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # Path reset — runs AFTER joint/base resets so robot state is valid.
    # The dataset directory is read from env.cfg.nav2_path_dataset_dir.
    reset_path = EventTerm(
        func=mdp.reset_path_tracking,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_path_points": 600,
        },
    )

    reset_stage2_obstacle = EventTerm(
        func=mdp.reset_obstacle_stage2_arm_reach,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_name": "obstacle",
            "x_min": 0.68,
            "x_max": 0.72,
            "y_noise": 0.02,
            "z_height": 0.25,
        },
    )
    # Path + obstacle debug visualisation
    draw_path_reset = EventTerm(
        func=mdp.draw_path_debug,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "path_stride": 3, "max_draw_envs": 4},
    )
    draw_path_interval = EventTerm(
        func=mdp.draw_path_debug,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={"asset_cfg": SceneEntityCfg("robot"), "path_stride": 3, "max_draw_envs": 4},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # Root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[mdp.base_link_name]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.4,
        },
    )

    # Joint penalties (legs only)
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.wheel_joint_names)},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_raw,
        weight=-0.04,
    )
    base_planar_speed_l2 = RewTerm(
        func=mdp.base_planar_speed_l2,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    yaw_rate_l2 = RewTerm(
        func=mdp.yaw_rate_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                 body_names=[r"^(?!.*(_wheel|front_lidar|arm_base_link|gripper_base|link[1-8])).*"],
                # body_names=[f"^(?!.*(_wheel|joint[1-8])).*"], # exclude wheels AND arm links
            ),
            "threshold": 1.0,
        },
    )

    upward = RewTerm(func=mdp.upward, weight=1.0)


    # path_corridor_clearance = RewTerm(
    #     func=mdp.path_corridor_clearance_reward,
    #     weight=30.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "obstacle_name": "obstacle",
    #         "min_arm_deviation": 0.10,
    #         "stopped_speed": 0.06,
    #         "stopped_yaw_rate": 0.12,
    #     },
    # )

    # Teach the PPO arm action to extend and sweep.
    stage2_extend_sweep_action = RewTerm(
        func=mdp.stage2_extend_sweep_action_reward,
        weight=1.5,
        params={
            "reach_start_step": 5,
            "sweep_start_step": 35,
        },
    )

    # Dense physical reward: obstacle moved sideways this step.
    stage2_obstacle_lateral_progress = RewTerm(
        func=mdp.stage2_obstacle_lateral_progress_reward,
        weight=80.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_name": "obstacle",
            "min_ahead_m": 0.10,
            "max_ahead_m": 1.50,
            "max_step_progress": 0.015,
        },
    )

    # Dense physical reward: obstacle is farther from path centerline.
    stage2_obstacle_lateral_distance = RewTerm(
        func=mdp.stage2_obstacle_lateral_distance_reward,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_name": "obstacle",
            "min_ahead_m": 0.10,
            "max_ahead_m": 1.50,
            "start_width": 0.03,
            "target_width": 0.55,
        },
    )

    front_blocked_persistence = RewTerm(
        func=mdp.front_blocked_persistence_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_name": "obstacle",
            "trigger_range": 1.50,
            "max_range": 5.0,
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # illegal_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=mdp.base_link_name), "threshold": 50.0},
    # )
    bad_orientation_2 = DoneTerm(func=mdp.bad_orientation_2)

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact_after_settle,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=mdp.base_link_name),
            "threshold": 150.0,
            "settle_steps": 100,
        },
    )

    non_arm_body_contact = DoneTerm(
        func=mdp.illegal_contact_after_settle,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!.*(_wheel|arm_base_link|gripper_base|link[1-8])).*"
                ],
            ),
            "threshold": 10.0,
            "settle_steps": 20,
        },
    )




@configclass
class M20PiperEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: M20PiperSceneCfg = M20PiperSceneCfg(num_envs=4096, env_spacing=25.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()  # enable after terrain is added

    # Directory containing path_*.npz files for path tracking.
    nav2_path_dataset_dir: str = mdp.nav2_path_dataset_dir
    # Keep False for full 4096-env training.
    debug_draw_enabled: bool = False

    # These only matter when debug_draw_enabled=True.
    debug_draw_path: bool = True
    debug_draw_goal: bool = True
    debug_draw_obstacle: bool = True
    debug_draw_lidar: bool = True

    debug_draw_max_envs: int = 4
    debug_draw_path_stride: int = 3

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 1
