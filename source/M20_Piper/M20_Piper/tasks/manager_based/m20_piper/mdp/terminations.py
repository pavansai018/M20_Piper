# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from isaaclab.utils.math import quat_apply_inverse

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from .observations import _nearest_path_index, _robot_xy_local # type: ignore

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.4,
) -> torch.Tensor:
    """
    Terminate when the robot is within `threshold` metres of the last path
    waypoint.  Returns False for all envs if path is not initialised.
    """
    e: Any = env
    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    path = e.navrl_global_path_xy     # [num_envs, max_pts, 2]
    valid = e.navrl_path_valid_count  # [num_envs]
    robot_xy = _robot_xy_local(e)    # [num_envs, 2]

    last_idx = (valid - 1).clamp(min=0)
    last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    goal_xy = path.gather(1, last_exp).squeeze(1)  # [num_envs, 2]

    dist = torch.norm(goal_xy - robot_xy, dim=-1)  # [num_envs]
    return dist < threshold

def illegal_contact_after_settle(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 50.0,
    settle_steps: int = 50,
) -> torch.Tensor:
    """Illegal contact, but ignore the reset transient.

    Use this only after fixing the asset. It helps distinguish real base-ground
    crashes from one-frame startup impulses caused by contact history or import
    artifacts.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    net_forces = contact_sensor.data.net_forces_w_history

    forces = torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1)  # type: ignore # [N, H, B]
    bad = torch.any(torch.max(forces, dim=1).values > threshold, dim=1)

    return bad & (env.episode_length_buf >= settle_steps)


def path_deviation_too_large(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 4,
    max_cte: float = 1.25,
    settle_steps: int = 30,
) -> torch.Tensor:
    """Terminate if robot leaves the global path too far."""
    e: Any = env

    if not hasattr(e, "navrl_global_path_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

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

    return (cte > max_cte) & (env.episode_length_buf > settle_steps)

def arm_body_collision_zone(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_body_names: tuple[str, ...] = ("link5", "link6", "link7", "link8"),
    base_half_x: float = 0.38,
    base_half_y: float = 0.30,
    z_min: float = -0.10,
    z_max: float = 0.45,
    settle_steps: int = 20,
) -> torch.Tensor:
    """Terminate if distal arm/gripper links enter robot body safety box."""
    e: Any = env
    asset = env.scene[asset_cfg.name]

    if not hasattr(e, "_arm_body_collision_term_ids"):
        ids, _ = asset.find_bodies(list(arm_body_names))
        e._arm_body_collision_term_ids = ids

    body_ids = e._arm_body_collision_term_ids

    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    arm_pos_w = asset.data.body_pos_w[:, body_ids, :]
    rel_w = arm_pos_w - asset.data.root_pos_w[:, None, :]

    q = asset.data.root_quat_w[:, None, :].expand(-1, len(body_ids), -1)
    rel_b = quat_apply_inverse(
        q.reshape(-1, 4),
        rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, len(body_ids), 3)

    inside_x = torch.abs(rel_b[..., 0]) < base_half_x
    inside_y = torch.abs(rel_b[..., 1]) < base_half_y
    inside_z = (rel_b[..., 2] > z_min) & (rel_b[..., 2] < z_max)

    violation = (inside_x & inside_y & inside_z).any(dim=1)

    return violation & (env.episode_length_buf > settle_steps)