from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class PiperSweepAction(ActionTerm):
    """6D PPO arm action mapped to a safe Piper sweep primitive.

    Keeps action_dim = 6 for checkpoint compatibility.

    Raw PPO arm action:
        a0 = extend        [-1, 1] -> [0, 1]
        a1 = sweep         [-1, 1] -> left/right yaw
        a2 = height        [-1, 1] -> small shoulder/elbow height offset
        a3 = wrist_roll    optional
        a4 = wrist_pitch   optional
        a5 = unused        reserved

    This prevents PPO from learning raw IK from scratch.
    PPO controls sweep parameters; this action term converts them to joint targets.
    """

    cfg: "PiperSweepActionCfg"

    def __init__(self, cfg: "PiperSweepActionCfg", env):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)

        if len(self._joint_ids) != 6:
            raise RuntimeError(
                f"PiperSweepAction expected 6 arm joints, got {len(self._joint_ids)}: {self._joint_names}"
            )

        self._raw_actions = torch.zeros(env.num_envs, 6, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 6, device=env.device)

        self._home = torch.tensor(cfg.home_pose, device=env.device, dtype=torch.float32).view(1, 6)
        self._reach = torch.tensor(cfg.reach_pose, device=env.device, dtype=torch.float32).view(1, 6)

        self._lower = torch.tensor(cfg.lower_limits, device=env.device, dtype=torch.float32).view(1, 6)
        self._upper = torch.tensor(cfg.upper_limits, device=env.device, dtype=torch.float32).view(1, 6)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        raw = torch.clamp(actions, -1.0, 1.0)
        self._raw_actions[:] = raw

        # a0: extend in [0, 1]
        extend = 0.5 * (raw[:, 0:1] + 1.0)

        # a1: sweep left/right, only effective when extended
        sweep = raw[:, 1:2]

        # a2: small reach-height correction
        height = raw[:, 2:3]

        # a3, a4: optional wrist correction
        wrist_roll = raw[:, 3:4]
        wrist_pitch = raw[:, 4:5]

        # Interpolate home -> reach.
        target = self._home + extend * (self._reach - self._home)

        # Sweep using joint1 yaw. Only allow sweep when extended.
        target[:, 0:1] = target[:, 0:1] + extend * sweep * self.cfg.max_sweep_yaw

        # Height/reach adjustment. Small values only.
        target[:, 1:2] = target[:, 1:2] + extend * height * self.cfg.joint2_height_gain
        target[:, 2:3] = target[:, 2:3] - extend * height * self.cfg.joint3_height_gain

        # Optional wrist adjustment.
        target[:, 3:4] = target[:, 3:4] + extend * wrist_roll * self.cfg.joint4_gain
        target[:, 4:5] = target[:, 4:5] + extend * wrist_pitch * self.cfg.joint5_gain

        # joint6 left unchanged unless you later need it.
        target = torch.clamp(target, self._lower, self._upper)

        self._processed_actions[:] = target

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )


@configclass
class PiperSweepActionCfg(ActionTermCfg):
    """Config for PiperSweepAction."""

    class_type: type = PiperSweepAction

    asset_name: str = MISSING # type: ignore
    joint_names: list[str] = MISSING # type: ignore

    # Default pose from your m20_piper.py.
    home_pose: tuple[float, float, float, float, float, float] = (
        0.0, 0.20, -0.35, 0.0, 0.20, 0.0
    )

    # Reach pose, not final sweep. PPO still chooses sweep direction using a1.
    reach_pose: tuple[float, float, float, float, float, float] = (
        0.0, 0.45, -1.05, 0.0, 0.75, 0.0
    )

    lower_limits: tuple[float, float, float, float, float, float] = (
        -1.20, -1.00, -1.20, -1.20, -1.00, -1.20
    )

    upper_limits: tuple[float, float, float, float, float, float] = (
        1.20, 1.00, 0.00, 1.20, 1.00, 1.20
    )

    max_sweep_yaw: float = 0.85
    joint2_height_gain: float = 0.12
    joint3_height_gain: float = 0.18
    joint4_gain: float = 0.25
    joint5_gain: float = 0.25