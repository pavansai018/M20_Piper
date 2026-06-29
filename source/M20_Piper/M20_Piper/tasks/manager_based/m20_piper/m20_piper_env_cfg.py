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
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Only legs and wheels are controlled by the RL policy.
    Arm joints (joint1-6) and gripper joints (joint7-8) are intentionally
    excluded so they hold their default position via the PD actuator and do
    not wander randomly.
    """
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=mdp.leg_joint_names + mdp.wheel_joint_names,
    #     scale={".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25},
    #     use_default_offset=True,
    #     clip={".*": (-100.0, 100.0)},
    #     preserve_order=True,
    # )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=mdp.wheel_joint_names, # + mdp.leg_joint_names,
        scale=2.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # velocity_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "base_velocity"},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel_without_wheel,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.joint_names, preserve_order=True),
        #         "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=mdp.wheel_joint_names),
        #     },
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.joint_names, preserve_order=True)},
        #     noise=Unoise(n_min=-1.5, n_max=1.5),
        #     clip=(-100.0, 100.0),
        #     scale=0.05,
        # )
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
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        # velocity_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "base_velocity"},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel_without_wheel,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.joint_names, preserve_order=True),
        #         "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=mdp.wheel_joint_names),
        #     },
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.joint_names, preserve_order=True)},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # )
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

    # startup
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.35, 1.5],
            "dynamic_friction_range": [0.35, 1.5],
            "restitution_range": [0.0, 0.1],
            "num_buckets": 1024,
        },
    )
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=f"^(?!.*{mdp.base_link_name}).*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[mdp.base_link_name]),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[mdp.base_link_name]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )

    # reset
    # randomize_apply_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[mdp.base_link_name]),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-10.0, 10.0),
    #     },
    # )
    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

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

    # Obstacle placed randomly on path — must run AFTER reset_path
    reset_obstacle = EventTerm(
        func=mdp.reset_obstacle_on_path,
        mode="reset",
        params={"obstacle_name": "obstacle", "path_frac_min": 0.3, "path_frac_max": 0.7},
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

    # Arm sub-controller: forward-lidar detection → extend → sweep → retract
    arm_controller = EventTerm(
        func=mdp.arm_obstacle_controller,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_name": "obstacle",
            "detection_range": 0.9,
            "detection_lat_half": 0.35,
            "extend_duration_s": 1.2,
            "sweep_duration_s": 2.0,
            "retract_duration_s": 1.0,
        },
    )

    # interval
    # randomize_push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


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
    # joint_torques_l2 = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-2.5e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.leg_joint_names)},
    # )
    # joint_acc_l2 = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-2e-7,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.leg_joint_names)},
    # )
    # joint_pos_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.leg_joint_names)},
    # )
    # joint_power = RewTerm(
    #     func=mdp.joint_power,
    #     weight=-2e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.leg_joint_names)},
    # )
    # hipx_joint_pos_penalty = RewTerm(
    #     func=mdp.joint_pos_penalty,
    #     weight=-0.4,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.hipx_joint_names),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #         "command_threshold": 0.1,
    #     },
    # )
    # hipy_joint_pos_penalty = RewTerm(
    #     func=mdp.joint_pos_penalty,
    #     weight=-0.1,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.hipy_joint_names),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #         "command_threshold": 0.1,
    #     },
    # )
    # knee_joint_pos_penalty = RewTerm(
    #     func=mdp.joint_pos_penalty,
    #     weight=-0.1,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.knee_joint_names),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #         "command_threshold": 0.1,
    #     },
    # )
    # action_mirror = RewTerm(
    #     func=mdp.action_mirror,
    #     weight=-0.03,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [
    #             ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
    #             ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
    #         ],
    #     },
    # )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*(_wheel|joint[1-8])).*"], ), # exclude wheels AND arm links
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[mdp.foot_link_name]), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp,
    #     weight=0.5,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    # )
    # feet_contact_without_cmd = RewTerm(
    #     func=mdp.feet_contact_without_cmd,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[mdp.foot_link_name]),
    #         "command_name": "base_velocity",
    #     },
    # )
    # stand_still = RewTerm(
    #     func=mdp.stand_still_joint_deviation_l1,
    #     weight=-2.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=mdp.leg_joint_names),
    #     },
    # )
    upward = RewTerm(func=mdp.upward, weight=1.0)

    # --- Path-following rewards ---
    path_progress = RewTerm(
        func=mdp.path_progress,
        weight=35.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_step_reward": 0.05},
    )
    path_cte_penalty = RewTerm(
        func=mdp.path_cross_track_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "lookahead": 4, "max_error": 1.0},
    )
    path_heading = RewTerm(
        func=mdp.path_heading_alignment,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "lookahead": 4},
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4, "bonus": 120.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=mdp.base_link_name), "threshold": 1.0},
    )
    bad_orientation_2 = DoneTerm(func=mdp.bad_orientation_2)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4},
    )


@configclass
class M20PiperEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: M20PiperSceneCfg = M20PiperSceneCfg(num_envs=4096, env_spacing=25.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()  # enable after terrain is added

    # Directory containing path_*.npz files for path tracking.
    nav2_path_dataset_dir: str = "/home/pavan/Downloads/SUTD/DesignProject/navrl-bench/m3_ros2_ws/src/nav_rl_bridge/rl_path_dataset/aws_warehouse"

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 1
