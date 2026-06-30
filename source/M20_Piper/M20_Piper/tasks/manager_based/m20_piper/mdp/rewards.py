# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# re-use path helpers from observations (single source of truth)
from .observations import _robot_xy_local, _robot_yaw, _nearest_path_index, _wrap_to_pi, lidar_path_corridor_blocked


# Global curriculum scalar in [0, 1], updated from terrain-level mean.
gait_level: float = 0.0

def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
    """Update global gait_level from mean terrain level.

    Mapping rule:
    - mean <= 0.0 -> 0.0
    - 0.0 < mean < 3.0 -> 使用 exp 函数映射
    - mean == 3.0 -> 1.0
    - mean >= 3.0 -> 1.0
    """
    global gait_level

    mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
    if mean_tensor.numel() == 0:
        mean_val = 0.0
    else:
        mean_val = float(torch.mean(mean_tensor).item())

    if math.isnan(mean_val) or math.isinf(mean_val):
        mean_val = 0.0

    if mean_val <= 0.0:
        gait_level = 0.0
    elif mean_val < 3.0:
        # exp 映射：mean=0 时接近 0，mean=3 时恰好为 1
        gait_level = math.exp(mean_val - 3.0)
    else:  # mean_val >= 3.0
        gait_level = 1.0

    return gait_level

def get_gait_level_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return global gait_level as tensor matching environment batch size."""
    return torch.full((env.num_envs,), gait_level, device=env.device)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact force violations (curriculum-scaled by gait_level)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold # type: ignore
    reward = torch.sum(violation.clip(min=0.0), dim=1)
    return reward * get_gait_level_tensor(env)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold # type: ignore
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ---------------------------------------------------------------------------
# Path-following rewards
# ---------------------------------------------------------------------------

def path_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_step_reward: float = 0.05,
) -> torch.Tensor:
    """
    Reward for forward progress along the path.

    Measures how much arc-length the robot advanced along the path since the
    last step.  Capped at `max_step_reward` to prevent reward hacking by
    teleporting.  Returns zero if path is not initialised.
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    path = e.navrl_global_path_xy       # [num_envs, max_pts, 2]
    arc = e.navrl_arc_length            # [num_envs, max_pts]
    valid = e.navrl_path_valid_count    # [num_envs]
    prev = e.navrl_prev_progress        # [num_envs]

    nearest = _nearest_path_index(e)
    nearest_clamped = nearest.clamp(max=(valid - 1).clamp(min=0))

    # Arc length at current nearest point
    curr_arc = arc.gather(1, nearest_clamped.unsqueeze(1)).squeeze(1)  # [num_envs]

    delta = (curr_arc - prev).clamp(min=0.0, max=max_step_reward)
    e.navrl_prev_progress = curr_arc.detach()

    return delta

def path_progress_unless_arm_reach_blocked(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    max_step_reward: float = 0.025,
) -> torch.Tensor:
    from .observations import lidar_path_corridor_blocked

    progress = path_progress(
        env,
        asset_cfg=asset_cfg,
        max_step_reward=max_step_reward,
    )

    arm_zone_blocked = lidar_path_corridor_blocked(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        min_ahead_m=0.25,
        max_ahead_m=1.10,
        corridor_half_width=0.30,
    )

    return progress * (~arm_zone_blocked).float()

def path_forward_velocity_unless_arm_reach_blocked(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    lookahead: int = 4,
    max_speed: float = 0.35,
) -> torch.Tensor:

    reward = path_forward_velocity(
        env,
        asset_cfg=asset_cfg,
        lookahead=lookahead,
        max_speed=max_speed,
    )

    arm_zone_blocked = lidar_path_corridor_blocked(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        min_ahead_m=0.25,
        max_ahead_m=1.10,
        corridor_half_width=0.30,
    )

    return reward * (~arm_zone_blocked).float()

def path_cross_track_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
    max_error: float = 1.0,
) -> torch.Tensor:
    """
    Penalise deviation from the path (cross-track error).

    Returns a value in [0, max_error]; use a negative weight in RewardsCfg.
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    path = e.navrl_global_path_xy
    valid = e.navrl_path_valid_count
    nearest = _nearest_path_index(e)
    robot_xy = _robot_xy_local(e)

    ahead_idx = (nearest + lookahead).clamp(max=(valid - 1).clamp(min=0))
    nearest_exp = nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    ahead_exp = ahead_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)

    p0 = path.gather(1, nearest_exp).squeeze(1)
    p1 = path.gather(1, ahead_exp).squeeze(1)

    tangent = p1 - p0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)

    cte = torch.abs(torch.sum((robot_xy - p0) * normal, dim=-1))
    return cte.clamp(max=max_error)


def path_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
) -> torch.Tensor:
    """
    Reward for aligning robot heading with the path tangent.

    Returns cos(heading_error) in [-1, 1]; use a positive weight.
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    path = e.navrl_global_path_xy
    valid = e.navrl_path_valid_count
    nearest = _nearest_path_index(e)

    ahead_idx = (nearest + lookahead).clamp(max=(valid - 1).clamp(min=0))
    nearest_exp = nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    ahead_exp = ahead_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)

    p0 = path.gather(1, nearest_exp).squeeze(1)
    p1 = path.gather(1, ahead_exp).squeeze(1)

    tangent = p1 - p0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)

    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    robot_yaw = _robot_yaw(e)
    err = _wrap_to_pi(path_yaw - robot_yaw)

    return torch.cos(err)


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.4,
    bonus: float = 1.0,
) -> torch.Tensor:
    """
    One-time bonus when the robot reaches within `threshold` metres of the
    last path waypoint.
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    path = e.navrl_global_path_xy
    valid = e.navrl_path_valid_count
    robot_xy = _robot_xy_local(e)

    last_idx = (valid - 1).clamp(min=0)
    last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    goal_xy = path.gather(1, last_exp).squeeze(1)

    dist = torch.norm(goal_xy - robot_xy, dim=-1)
    return (dist < threshold).float() * bonus


def _path_tangent_at_robot(
    env: ManagerBasedRLEnv,
    lookahead: int = 4,
) -> torch.Tensor:
    """Return local path tangent vector in world XY frame. Shape: [N, 2]."""
    e: Any = env

    path = e.navrl_global_path_xy
    valid = e.navrl_path_valid_count
    nearest = _nearest_path_index(e)

    ahead_idx = (nearest + lookahead).clamp(max=(valid - 1).clamp(min=0))

    nearest_exp = nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    ahead_exp = ahead_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)

    p0 = path.gather(1, nearest_exp).squeeze(1)
    p1 = path.gather(1, ahead_exp).squeeze(1)

    tangent = p1 - p0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)

    return tangent


def path_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
    max_speed: float = 1.0,
) -> torch.Tensor:
    """Reward velocity in the correct path direction."""
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    tangent = _path_tangent_at_robot(env, lookahead=lookahead)

    vel_xy_w = asset.data.root_lin_vel_w[:, :2]
    forward_speed = torch.sum(vel_xy_w * tangent, dim=-1)

    return torch.clamp(forward_speed, 0.0, max_speed)


def path_reverse_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
    max_speed: float = 1.0,
) -> torch.Tensor:
    """Penalize velocity opposite to the path direction."""
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    tangent = _path_tangent_at_robot(env, lookahead=lookahead)

    vel_xy_w = asset.data.root_lin_vel_w[:, :2]
    forward_speed = torch.sum(vel_xy_w * tangent, dim=-1)

    return torch.clamp(-forward_speed, 0.0, max_speed)


def action_rate_l2_raw(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action smoothness penalty without gait-level curriculum gating."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def base_planar_speed_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize high XY base speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def yaw_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize fast yaw rotation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def _front_center_min_from_scan(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    num_rays: int = 181,
    fov_deg: float = 270.0,
    max_range: float = 5.0,
    sector_half_width_deg: float = 20.0,
) -> torch.Tensor:
    """Minimum LiDAR range in front sector."""
    from .observations import front_lidar_scan_obs

    ranges = front_lidar_scan_obs(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        num_rays=num_rays,
        fov_deg=fov_deg,
        max_range=max_range,
        normalize=False,
    )

    angles = torch.linspace(
        -0.5 * fov_deg,
        0.5 * fov_deg,
        num_rays,
        device=env.device,
    )

    front_mask = torch.abs(angles) <= sector_half_width_deg
    return torch.min(ranges[:, front_mask], dim=1).values


def front_blocked_persistence_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    trigger_range: float = 1.50,
    max_range: float = 5.0,
) -> torch.Tensor:
    """Penalty for keeping the front path blocked."""
    center_min = _front_center_min_from_scan(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        max_range=max_range,
    )
    return (center_min < trigger_range).float()


def stop_when_front_blocked_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    trigger_range: float = 1.50,
    max_range: float = 5.0,
) -> torch.Tensor:
    """Penalize base motion when the front LiDAR sector is blocked.

    This teaches:
        obstacle in front -> stop base
    """
    asset: Articulation = env.scene[asset_cfg.name]

    center_min = _front_center_min_from_scan(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        max_range=max_range,
    )

    front_blocked = center_min < trigger_range
    base_speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)

    return front_blocked.float() * base_speed

def arm_home_when_clear_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    trigger_range: float = 1.50,
    max_range: float = 5.0,
) -> torch.Tensor:
    """Penalize unnecessary arm movement when the front path is clear.

    This teaches:
        front clear -> keep arm near home
    """
    asset: Articulation = env.scene[asset_cfg.name]

    center_min = _front_center_min_from_scan(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        max_range=max_range,
    )

    front_clear = center_min >= trigger_range

    if not hasattr(env, "_arm_home_penalty_joint_ids"):
        from M20_Piper.tasks.manager_based.m20_piper.mdp.config import arm_joint_names
        arm_ids, _ = asset.find_joints(arm_joint_names)
        env._arm_home_penalty_joint_ids = arm_ids  # type: ignore

    arm_ids = env._arm_home_penalty_joint_ids  # type: ignore

    arm_deviation = torch.norm(
        asset.data.joint_pos[:, arm_ids] - asset.data.default_joint_pos[:, arm_ids],
        dim=1,
    )

    return front_clear.float() * arm_deviation

def path_corridor_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    min_arm_deviation: float = 0.10,
    stopped_speed: float = 0.06,
    stopped_yaw_rate: float = 0.12,
) -> torch.Tensor:
    """Reward arm only when it clears LiDAR obstacle from the path corridor."""
    from .observations import lidar_path_corridor_blocked

    asset: Articulation = env.scene[asset_cfg.name]

    arm_zone_blocked = lidar_path_corridor_blocked(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        min_ahead_m=0.25,
        max_ahead_m=1.10,
        corridor_half_width=0.30,
    )

    wider_path_blocked = lidar_path_corridor_blocked(
        env,
        asset_cfg=asset_cfg,
        obstacle_name=obstacle_name,
        min_ahead_m=0.25,
        max_ahead_m=1.60,
        corridor_half_width=0.35,
    )

    if not hasattr(env, "_prev_arm_zone_blocked"):
        env._prev_arm_zone_blocked = arm_zone_blocked.detach().clone()  # type: ignore

    prev_blocked = env._prev_arm_zone_blocked  # type: ignore
    env._prev_arm_zone_blocked = arm_zone_blocked.detach()  # type: ignore

    if not hasattr(env, "_path_clear_arm_joint_ids"):
        from M20_Piper.tasks.manager_based.m20_piper.mdp.config import arm_joint_names
        arm_ids, _ = asset.find_joints(arm_joint_names)
        env._path_clear_arm_joint_ids = arm_ids  # type: ignore

    arm_ids = env._path_clear_arm_joint_ids  # type: ignore

    arm_dev = torch.norm(
        asset.data.joint_pos[:, arm_ids] - asset.data.default_joint_pos[:, arm_ids],
        dim=1,
    )

    base_speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])

    base_stopped = base_speed < stopped_speed
    yaw_stopped = yaw_rate < stopped_yaw_rate
    arm_active = arm_dev > min_arm_deviation

    # Reward only on transition:
    # previously blocked in arm zone, now not blocked.
    cleared_now = prev_blocked & (~arm_zone_blocked)

    # Additional guard: avoid rewarding if path is still blocked slightly farther ahead.
    truly_clear = ~wider_path_blocked

    return (
        cleared_now.float()
        * truly_clear.float()
        * base_stopped.float()
        * yaw_stopped.float()
        * arm_active.float()
    )