# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from .observations import _robot_xy_local # type: ignore

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