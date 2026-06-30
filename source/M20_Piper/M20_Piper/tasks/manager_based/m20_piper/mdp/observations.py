# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import math
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Path tracking helpers
# ---------------------------------------------------------------------------

def _robot_xy_local(env: Any, asset_name: str = "robot") -> torch.Tensor:
    """Robot (x, y) in the LOCAL env frame (world pos minus env origin)."""
    asset: Articulation = env.scene[asset_name]
    return asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def _robot_yaw(env: Any, asset_name: str = "robot") -> torch.Tensor:
    """Robot yaw angle in world frame. Shape [num_envs]."""
    asset: Articulation = env.scene[asset_name]
    quat = asset.data.root_quat_w  # [num_envs, 4]  (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _nearest_path_index(env: Any) -> torch.Tensor:
    """Brute-force nearest-neighbour search across path points, cached per step.

    Called up to 9 times per step (2× policy obs + 2× critic obs + 3× rewards).
    Without caching each call allocates ~29 MB of GPU temporaries; the CUDA
    caching allocator pool grows over time causing steadily increasing iteration
    time.  One result per sim-step is sufficient since physics state doesn't
    change between observation and reward calls within the same step.

    Returns indices of shape [num_envs] (long).
    """
    # env.common_step_counter is a Python int incremented each env.step() — no GPU sync.
    step = env.common_step_counter
    if getattr(env, "_navrl_nearest_step", -1) == step:
        return env._navrl_nearest_result  # type: ignore[return-value]

    path = env.navrl_global_path_xy           # [num_envs, max_pts, 2]
    valid = env.navrl_path_valid_count        # [num_envs]
    robot_xy = _robot_xy_local(env)           # [num_envs, 2]

    dists = torch.norm(path - robot_xy.unsqueeze(1), dim=-1)  # [num_envs, max_pts]

    max_pts = path.shape[1]
    mask = torch.arange(max_pts, device=env.device).unsqueeze(0) >= valid.unsqueeze(1)
    dists = dists.masked_fill(mask, 1e6)

    result = torch.argmin(dists, dim=1)  # [num_envs]
    env._navrl_nearest_step   = step
    env._navrl_nearest_result = result
    return result


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


# ---------------------------------------------------------------------------
# Path observation functions (registered in env cfg)
# ---------------------------------------------------------------------------

def local_path_window(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_points: int = 8,
    normalize_dist: float = 4.0,
) -> torch.Tensor:
    """
    Next `num_points` path waypoints in the robot body frame, flattened.

    Output shape: [num_envs, num_points * 2]
    Values normalised by `normalize_dist`.
    Returns zeros if path not initialised.
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, num_points * 2, device=env.device)

    path = e.navrl_global_path_xy             # [num_envs, max_pts, 2]
    valid = e.navrl_path_valid_count          # [num_envs]
    robot_xy = _robot_xy_local(e)            # [num_envs, 2]
    yaw = _robot_yaw(e)                      # [num_envs]
    nearest = _nearest_path_index(e)         # [num_envs]

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    num_envs = env.num_envs
    max_pts = path.shape[1]
    window = torch.zeros(num_envs, num_points, 2, device=env.device)

    for k in range(num_points):
        idx = torch.clamp(nearest + k, max=valid - 1)  # [num_envs]
        # Gather waypoint for each env
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        pt = path.gather(1, idx_expanded).squeeze(1)    # [num_envs, 2]

        # Transform to robot body frame
        rel = pt - robot_xy                             # [num_envs, 2]
        bx = cos_y * rel[:, 0] + sin_y * rel[:, 1]
        by = -sin_y * rel[:, 0] + cos_y * rel[:, 1]

        window[:, k, 0] = bx
        window[:, k, 1] = by

    return window.reshape(num_envs, num_points * 2) / normalize_dist


def path_heading_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
) -> torch.Tensor:
    """
    Signed heading error between robot yaw and local path tangent.
    Shape: [num_envs, 1].  Range: [-pi, pi].
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    path = e.navrl_global_path_xy
    valid = e.navrl_path_valid_count
    nearest = _nearest_path_index(e)

    ahead_idx = torch.clamp(nearest + lookahead, max=valid - 1)
    nearest_expanded = nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    ahead_expanded = ahead_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)

    p0 = path.gather(1, nearest_expanded).squeeze(1)   # [num_envs, 2]
    p1 = path.gather(1, ahead_expanded).squeeze(1)     # [num_envs, 2]

    tangent = p1 - p0                                  # [num_envs, 2]
    tangent_len = torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)
    tangent = tangent / tangent_len

    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])  # [num_envs]
    robot_yaw = _robot_yaw(e)
    err = _wrap_to_pi(path_yaw - robot_yaw)

    return err.unsqueeze(-1)


def path_cross_track_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
    normalize_dist: float = 2.0,
) -> torch.Tensor:
    """
    Signed cross-track error (perpendicular distance from robot to path).
    Positive = robot is left of path tangent.
    Shape: [num_envs, 1].
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    path = e.navrl_global_path_xy
    valid = e.navrl_path_valid_count
    nearest = _nearest_path_index(e)
    robot_xy = _robot_xy_local(e)

    ahead_idx = torch.clamp(nearest + lookahead, max=valid - 1)
    nearest_expanded = nearest.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    ahead_expanded = ahead_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)

    p0 = path.gather(1, nearest_expanded).squeeze(1)
    p1 = path.gather(1, ahead_expanded).squeeze(1)

    tangent = p1 - p0
    tangent_len = torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)
    tangent = tangent / tangent_len

    # Normal = perpendicular to tangent (left-hand normal)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)

    cte = torch.sum((robot_xy - p0) * normal, dim=-1)  # [num_envs]

    return (cte / normalize_dist).unsqueeze(-1)


def distance_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_dist: float = 8.0,
) -> torch.Tensor:
    """
    Euclidean distance from robot to the last path waypoint.
    Shape: [num_envs, 1].
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.ones(env.num_envs, 1, device=env.device)

    path = e.navrl_global_path_xy    # [num_envs, max_pts, 2]
    valid = e.navrl_path_valid_count  # [num_envs]
    robot_xy = _robot_xy_local(e)   # [num_envs, 2]

    last_idx = (valid - 1).clamp(min=0)
    last_idx_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    goal_xy = path.gather(1, last_idx_exp).squeeze(1)   # [num_envs, 2]

    dist = torch.norm(goal_xy - robot_xy, dim=-1)        # [num_envs]
    return (dist / normalize_dist).unsqueeze(-1)

def front_lidar_scan_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    num_rays: int = 31,
    fov_deg: float = 80.0,
    max_range: float = 5.0,
    lidar_x: float = 0.37,
    lidar_z: float = 0.05,
    lidar_pitch_deg: float = -7.0,
    obs_hx: float = 0.20,
    obs_hy: float = 0.20,
    obs_hz: float = 0.25,
    normalize: bool = True,
) -> torch.Tensor:
    """Front LiDAR sector scan.

    Sim training:
        Synthetic scan is generated from obstacle pose.

    Real deployment:
        Replace this with real /scan sector preprocessing.
        PPO should still receive the same normalized front scan vector.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if obstacle_name not in env.scene.rigid_objects:
        out = torch.full((env.num_envs, num_rays), max_range, device=env.device)
        return out / max_range if normalize else out

    obstacle = env.scene.rigid_objects[obstacle_name]

    robot_pos = asset.data.root_pos_w
    robot_quat = asset.data.root_quat_w
    obs_pos = obstacle.data.root_pos_w

    q = robot_quat
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    sensor = torch.stack(
        [
            robot_pos[:, 0] + lidar_x * torch.cos(yaw),
            robot_pos[:, 1] + lidar_x * torch.sin(yaw),
            robot_pos[:, 2] + lidar_z,
        ],
        dim=1,
    )

    offsets = torch.linspace(
        -math.radians(fov_deg) * 0.5,
        math.radians(fov_deg) * 0.5,
        num_rays,
        device=env.device,
    )

    pitch = math.radians(lidar_pitch_deg)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    ray_yaw = yaw.unsqueeze(1) + offsets.unsqueeze(0)

    dirs = torch.stack(
        [
            torch.cos(ray_yaw) * cp,
            torch.sin(ray_yaw) * cp,
            torch.full((env.num_envs, num_rays), sp, device=env.device),
        ],
        dim=2,
    )

    half_ext = torch.tensor([obs_hx, obs_hy, obs_hz], device=env.device)

    obs_lo = (obs_pos - half_ext).unsqueeze(1)
    obs_hi = (obs_pos + half_ext).unsqueeze(1)
    sensor_exp = sensor.unsqueeze(1)

    safe_dirs = torch.where(
        dirs.abs() < 1e-9,
        torch.full_like(dirs, 1e-9),
        dirs,
    )

    t0 = (obs_lo - sensor_exp) / safe_dirs
    t1 = (obs_hi - sensor_exp) / safe_dirs

    t_enter = torch.min(t0, t1).max(dim=2).values
    t_exit = torch.max(t0, t1).min(dim=2).values

    hit = (t_enter < t_exit) & (t_enter > 0.05) & (t_enter < max_range)

    ranges = torch.full((env.num_envs, num_rays), max_range, device=env.device)
    ranges = torch.where(hit, t_enter, ranges)

    return ranges / max_range if normalize else ranges

def lidar_sector_blocked_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_name: str = "obstacle",
    num_rays: int = 181,
    fov_deg: float = 270.0,
    max_range: float = 5.0,
    trigger_range: float = 1.50,
    sector_center_deg: float = 0.0,
    sector_half_width_deg: float = 20.0,
) -> torch.Tensor:
    """Return 1 if LiDAR sector is blocked.

    sector_center_deg=0 means straight ahead.
    sector_half_width_deg=20 means front ±20°.
    """
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

    sector_mask = (
        torch.abs(angles - sector_center_deg) <= sector_half_width_deg
    )

    if not torch.any(sector_mask):
        return torch.zeros(env.num_envs, 1, device=env.device)

    sector_min = torch.min(ranges[:, sector_mask], dim=1).values
    blocked = sector_min < trigger_range

    return blocked.float().unsqueeze(1)


def arm_joint_pos_rel_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Arm joint position relative to default pose."""
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_obs_arm_joint_ids"):
        from M20_Piper.tasks.manager_based.m20_piper.mdp.config import arm_joint_names
        ids, _ = asset.find_joints(arm_joint_names)
        env._obs_arm_joint_ids = ids # type: ignore

    arm_ids = env._obs_arm_joint_ids # type: ignore
    return asset.data.joint_pos[:, arm_ids] - asset.data.default_joint_pos[:, arm_ids]


def arm_joint_vel_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Arm joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_obs_arm_vel_joint_ids"):
        from M20_Piper.tasks.manager_based.m20_piper.mdp.config import arm_joint_names
        ids, _ = asset.find_joints(arm_joint_names)
        env._obs_arm_vel_joint_ids = ids # type: ignore

    arm_ids = env._obs_arm_vel_joint_ids # type: ignore
    return asset.data.joint_vel[:, arm_ids]