# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Domain-randomisation helpers
# ---------------------------------------------------------------------------

def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    inertias = asset.root_physx_view.get_inertias()
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    for idx in [0, 4, 8]:
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    asset.root_physx_view.set_coms(com_offsets, env_ids)


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: str,
    distribution: str,
) -> torch.Tensor:
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)  # type: ignore
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]

    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(f"Unknown distribution: '{distribution}'")

    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)  # type: ignore
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)  # type: ignore
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)  # type: ignore
    else:
        raise NotImplementedError(f"Unknown operation: '{operation}'")
    return data


def bad_orientation_2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # Skip first 10 steps so the contact-sensor history and leg springs can settle
    # after each episode reset before we start checking orientation.
    settling = env.episode_length_buf < 100
    asset: RigidObject = env.scene[asset_cfg.name]
    bad = (asset.data.projected_gravity_b[:, 2] > -0.1) | (asset.data.projected_gravity_b[:, :2].abs() > 0.92).any(-1)
    return bad & ~settling


# ---------------------------------------------------------------------------
# Path debug draw
# ---------------------------------------------------------------------------

def _get_debug_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw  # type: ignore
    except Exception:
        from omni.isaac.debug_draw import _debug_draw  # type: ignore
    return _debug_draw.acquire_debug_draw_interface()


def _clear_debug_draw(draw: Any) -> None:
    for fn in ("clear", "clear_points", "clear_lines"):
        if hasattr(draw, fn):
            try:
                getattr(draw, fn)()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Lidar simulation helpers
# ---------------------------------------------------------------------------

# Sensor mount — MUST match the front_lidar joint in M20_Piper.urdf:
#   <joint name="base_to_front_lidar"> <origin xyz="0.37 0.0 0.05" rpy="0 0.12 0"/>
#
# base_link oscillates ~0.45–0.60 m during gait.
# Sensor world height = robot_z + _LIDAR_Z_OFFS ≈ 0.50–0.65 m
# Obstacle top = 0.50 m.  To see the obstacle, rays must be tilted DOWN.
#
# _LIDAR_PITCH_DEG = −7° (matches rpy pitch 0.12 rad in URDF joint).
# At 1.0 m forward: ray drops 0.12 m → hits obstacle side even when sensor
# is 0.10 m above the obstacle top.
#
# To calibrate for your real robot:
#   1. Measure (x, y, z) of the physical lidar centre in base_link frame.
#   2. Update the URDF joint origin AND the three constants below.
#   3. Measure the lidar's downward pitch (or set to 0 if truly horizontal).
_LIDAR_FWD_X      = 0.37   # matches urdf joint x (m)
_LIDAR_Z_OFFS     = 0.05   # matches urdf joint z (m)
_LIDAR_PITCH_DEG  = -7.0   # nose-down pitch in degrees (matches rpy[1]=0.12 rad)
_LIDAR_MAX_RANGE  = 5.0    # max sensing distance (m)

# Obstacle half-extents in XY (matches CuboidCfg size=(0.4, 0.4, 0.5))
_OBS_HX = 0.20
_OBS_HY = 0.20

# Detection ray constants — cached at module level to avoid per-call GPU allocations.
# Recomputed lazily on first call and reused thereafter (device is fixed per process).
import math as _math


def _lidar_ray_dir(yaw: float, azimuth_offset_deg: float, pitch_deg: float):
    """Compute a normalised ray direction for a tilted lidar.

    yaw               – robot heading (rad)
    azimuth_offset_deg – angle offset from forward in the horizontal plane (deg)
    pitch_deg         – downward pitch of the entire lidar plane (deg, negative = down)

    Returns (dx, dy, dz) normalised to unit length.
    """
    import math as _math
    a   = yaw + _math.radians(azimuth_offset_deg)
    cp  = _math.cos(_math.radians(pitch_deg))
    sp  = _math.sin(_math.radians(pitch_deg))   # negative → downward component
    dx  = _math.cos(a) * cp
    dy  = _math.sin(a) * cp
    dz  = sp                                     # same pitch for every beam in the sweep
    # already unit-length: cos²(a)*cos²p + sin²(a)*cos²p + sin²p = 1
    return dx, dy, dz


def _ray_aabb_hit(lx: float, ly: float, lz: float,
                  dx: float, dy: float, dz: float,
                  cx: float, cy: float, cz: float,
                  hx: float, hy: float, hz: float,
                  max_dist: float):
    """3-D ray vs axis-aligned box (slab method).

    Simulates one lidar beam measuring distance to a box obstacle.
    On the real robot: replace with the matching /scan.ranges element.

    Returns (hit: bool, distance: float).
    """
    def slab(o, d, lo, hi):
        if abs(d) > 1e-9:
            t0 = (lo - o) / d
            t1 = (hi - o) / d
            return (t0, t1) if t0 < t1 else (t1, t0)
        return (-1e18, 1e18) if lo <= o <= hi else (1e18, -1e18)

    txn, txf = slab(lx, dx, cx - hx, cx + hx)
    tyn, tyf = slab(ly, dy, cy - hy, cy + hy)
    tzn, tzf = slab(lz, dz, cz - hz, cz + hz)

    t_enter = max(txn, tyn, tzn)
    t_exit  = min(txf, tyf, tzf)

    if t_enter < t_exit and 0.0 < t_enter < max_dist:
        return True, t_enter
    return False, max_dist

def draw_path_debug(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_stride: int = 3,
    max_draw_envs: int = 4,
) -> None:
    """Optional debug draw for path, goal, obstacle wireframe, and lidar rays.

    Controlled from env.cfg:

        debug_draw_enabled
        debug_draw_path
        debug_draw_goal
        debug_draw_obstacle
        debug_draw_lidar
        debug_draw_max_envs
        debug_draw_path_stride

    When debug_draw_enabled=False, this returns before debug-draw import,
    GPU->CPU transfer, or Python drawing loops.
    """
    e: Any = env

    # ----------------------------------------------------------------------
    # Config flags: default is OFF for training speed.
    # ----------------------------------------------------------------------
    cfg = getattr(env, "cfg", None)

    debug_draw_enabled = bool(getattr(cfg, "debug_draw_enabled", False))
    if not debug_draw_enabled:
        return

    draw_path = bool(getattr(cfg, "debug_draw_path", True))
    draw_goal = bool(getattr(cfg, "debug_draw_goal", True))
    draw_obstacle = bool(getattr(cfg, "debug_draw_obstacle", True))
    draw_lidar = bool(getattr(cfg, "debug_draw_lidar", True))

    if not (draw_path or draw_goal or draw_obstacle or draw_lidar):
        return

    path_stride = int(getattr(cfg, "debug_draw_path_stride", path_stride))
    max_draw_envs = int(getattr(cfg, "debug_draw_max_envs", max_draw_envs))

    if not hasattr(e, "navrl_global_path_xy"):
        return

    # Extra safety for real headless mode.
    try:
        import carb as _carb
        if _carb.settings.get_settings().get_as_bool("/app/window/hideUi"):
            return
    except Exception:
        pass

    draw = _get_debug_draw()
    _clear_debug_draw(draw)

    draw_ids = env_ids[: min(len(env_ids), max_draw_envs)]
    n = len(draw_ids)
    if n == 0:
        return

    draw_ids_list = draw_ids.tolist()

    has_obs = "obstacle" in env.scene.rigid_objects

    need_origins = draw_path or draw_goal
    need_robot_pose = draw_lidar
    need_obstacle_pose = (draw_obstacle or draw_lidar) and has_obs

    origins_l = None
    valids_l = None
    goals_l = None
    rp_l = None
    q_l = None
    obs_l: list = []

    # ----------------------------------------------------------------------
    # Transfer only the tensors required by enabled flags.
    # ----------------------------------------------------------------------
    if need_origins:
        origins_l = e.scene.env_origins[draw_ids, :2].cpu().tolist()

    if draw_path:
        valids_l = e.navrl_path_valid_count[draw_ids].cpu().tolist()

    if draw_goal:
        goals_l = e.navrl_final_goal_xy[draw_ids].cpu().tolist()

    robot_asset: Articulation = env.scene[asset_cfg.name]

    if need_robot_pose:
        rp_l = robot_asset.data.root_pos_w[draw_ids].cpu().tolist()
        q_l = robot_asset.data.root_quat_w[draw_ids].cpu().tolist()

    if need_obstacle_pose:
        obs_l = env.scene.rigid_objects["obstacle"].data.root_pos_w[draw_ids].cpu().tolist()

    # ----------------------------------------------------------------------
    # Path lines
    # ----------------------------------------------------------------------
    if draw_path and origins_l is not None and valids_l is not None:
        all_p0: list = []
        all_p1: list = []

        for i, eid in enumerate(draw_ids_list):
            valid = int(valids_l[i])
            if valid <= 1:
                continue

            ox, oy = origins_l[i]

            pts = e.navrl_global_path_xy[eid, :valid:path_stride].cpu().tolist()

            for k in range(len(pts) - 1):
                all_p0.append((pts[k][0] + ox, pts[k][1] + oy, 0.12))
                all_p1.append((pts[k + 1][0] + ox, pts[k + 1][1] + oy, 0.12))

        if all_p0:
            draw.draw_lines(
                all_p0,
                all_p1,
                [(0.0, 0.4, 1.0, 1.0)] * len(all_p0),
                [3.0] * len(all_p0),
            )

    # ----------------------------------------------------------------------
    # Goal marker
    # ----------------------------------------------------------------------
    if draw_goal and origins_l is not None and goals_l is not None:
        goal_pts = [
            (
                goals_l[i][0] + origins_l[i][0],
                goals_l[i][1] + origins_l[i][1],
                0.35,
            )
            for i in range(n)
        ]

        if goal_pts:
            draw.draw_points(
                goal_pts,
                [(0.0, 1.0, 0.0, 1.0)] * n,
                [18.0] * n,
            )

    # ----------------------------------------------------------------------
    # Obstacle wireframe
    # ----------------------------------------------------------------------
    if draw_obstacle and has_obs and obs_l:
        hx, hy, hz = 0.20, 0.20, 0.25

        corners = [
            (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
            (-hx, -hy,  hz), (hx, -hy,  hz), (hx, hy,  hz), (-hx, hy,  hz),
        ]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        box_p0: list = []
        box_p1: list = []

        for i in range(n):
            cx, cy, cz = obs_l[i]
            wc = [(cx + dx, cy + dy, cz + dz) for dx, dy, dz in corners]

            for a, b in edges:
                box_p0.append(wc[a])
                box_p1.append(wc[b])

        if box_p0:
            draw.draw_lines(
                box_p0,
                box_p1,
                [(1.0, 0.4, 0.0, 1.0)] * len(box_p0),
                [2.5] * len(box_p0),
            )

    # ----------------------------------------------------------------------
    # LiDAR rays
    # ----------------------------------------------------------------------
    if draw_lidar and has_obs and obs_l and rp_l is not None and q_l is not None:
        angle_offsets = [135.0 - k * 270.0 / 89 for k in range(90)]
        obs_hz = 0.25

        ray_p0: list = []
        ray_p1: list = []
        ray_colors: list = []

        for i in range(n):
            w_, x_, y_, z_ = q_l[i]
            yaw = _math.atan2(
                2.0 * (w_ * z_ + x_ * y_),
                1.0 - 2.0 * (y_ * y_ + z_ * z_),
            )

            rx, ry, rz = rp_l[i]

            lx = rx + _LIDAR_FWD_X * _math.cos(yaw)
            ly = ry + _LIDAR_FWD_X * _math.sin(yaw)
            lz = rz + _LIDAR_Z_OFFS

            cx, cy, cz = obs_l[i]

            for a_off in angle_offsets:
                dx, dy, dz = _lidar_ray_dir(yaw, a_off, _LIDAR_PITCH_DEG)

                hit, dist = _ray_aabb_hit(
                    lx, ly, lz,
                    dx, dy, dz,
                    cx, cy, cz,
                    _OBS_HX, _OBS_HY, obs_hz,
                    _LIDAR_MAX_RANGE,
                )

                ray_p0.append((lx, ly, lz))
                ray_p1.append((lx + dx * dist, ly + dy * dist, lz + dz * dist))

                if hit:
                    ray_colors.append((1.0, 0.1, 0.1, 0.9))
                else:
                    ray_colors.append((0.1, 0.85, 0.1, 0.12))

        if ray_p0:
            draw.draw_lines(ray_p0, ray_p1, ray_colors, [1.2] * len(ray_p0))

# ---------------------------------------------------------------------------
# Path tracking reset
# ---------------------------------------------------------------------------

def _ensure_path_buffers(env: Any, max_path_points: int) -> None:
    """Initialise path tracking tensors and Nav2PathDataset on first call."""
    if hasattr(env, "navrl_global_path_xy"):
        return

    from .path_dataset import Nav2PathDataset

    if not hasattr(env, "cfg") or not hasattr(env.cfg, "nav2_path_dataset_dir"):
        raise RuntimeError(
            "env.cfg.nav2_path_dataset_dir is not set. "
            "Add  nav2_path_dataset_dir: str  to M20PiperEnvCfg and point it at "
            "your directory of path_*.npz files."
        )

    env.nav2_path_dataset = Nav2PathDataset(
        dataset_dir=env.cfg.nav2_path_dataset_dir,
        device=env.device,
        max_path_points=max_path_points,
    )

    env.navrl_global_path_xy  = torch.zeros(env.num_envs, max_path_points, 2, device=env.device)
    env.navrl_path_valid_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.navrl_arc_length       = torch.zeros(env.num_envs, max_path_points, device=env.device)
    env.navrl_prev_progress    = torch.zeros(env.num_envs, device=env.device)
    env.navrl_final_goal_xy    = torch.zeros(env.num_envs, 2, device=env.device)
    env.navrl_start_pose       = torch.zeros(env.num_envs, 3, device=env.device)
    env.navrl_goal_pose        = torch.zeros(env.num_envs, 3, device=env.device)


def _compute_arc_length(paths: torch.Tensor, valid_counts: torch.Tensor) -> torch.Tensor:
    """Compute cumulative arc length for a batch of paths.

    Args:
        paths:        [N, max_pts, 2]
        valid_counts: [N]  long

    Returns:
        arc: [N, max_pts]
    """
    num_envs, max_pts, _ = paths.shape
    arc = torch.zeros(num_envs, max_pts, device=paths.device)

    diffs = paths[:, 1:, :] - paths[:, :-1, :]          # [N, max_pts-1, 2]
    seg_len = torch.norm(diffs, dim=-1)                  # [N, max_pts-1]
    arc[:, 1:] = torch.cumsum(seg_len, dim=1)

    # Zero out entries beyond valid_count
    idx = torch.arange(max_pts, device=paths.device).unsqueeze(0)  # [1, max_pts]
    arc = arc * (idx < valid_counts.unsqueeze(1)).float()

    return arc


def reset_path_tracking(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_path_points: int = 600,
) -> None:
    """
    Reset path tracking for the given envs.

    Randomly picks one path_*.npz file per env from nav2_path_dataset_dir
    (set on env.cfg).  Each npz must contain:
        start    [3]    — (x, y, yaw) in local env frame
        goal     [3]    — (x, y, yaw) in local env frame
        path_xy  [K, 2] — waypoints in local env frame

    Tensors updated on env:
        navrl_global_path_xy   [num_envs, max_path_points, 2]
        navrl_path_valid_count [num_envs]
        navrl_arc_length       [num_envs, max_path_points]
        navrl_prev_progress    [num_envs]
        navrl_final_goal_xy    [num_envs, 2]
        navrl_start_pose       [num_envs, 3]
        navrl_goal_pose        [num_envs, 3]

    Robot is placed at the start pose from the npz.
    Goal marker is moved to the goal position from the npz.
    """
    e: Any = env
    _ensure_path_buffers(e, max_path_points)

    starts, goals, paths, valid_counts, _ = e.nav2_path_dataset.sample_batch(env_ids)

    # Force goal xy to match actual last valid path point
    last_idx = (valid_counts - 1).clamp(min=0)  # [N]
    last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    last_xy = paths.gather(1, last_exp).squeeze(1)       # [N, 2]
    goals_clone = goals.clone()
    goals_clone[:, :2] = last_xy

    e.navrl_global_path_xy[env_ids]  = paths
    e.navrl_path_valid_count[env_ids] = valid_counts
    e.navrl_arc_length[env_ids]       = _compute_arc_length(paths, valid_counts)
    e.navrl_prev_progress[env_ids]    = 0.0
    e.navrl_start_pose[env_ids]       = starts
    e.navrl_goal_pose[env_ids]        = goals_clone
    e.navrl_final_goal_xy[env_ids]    = goals_clone[:, :2]

    # ---- Place robot at path start ----------------------------------------
    asset: Articulation = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]   # [N, 3]

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = starts[:, 0] + origins[:, 0]   # x
    root_state[:, 1] = starts[:, 1] + origins[:, 1]   # y
    root_state[:, 2] = origins[:, 2] + asset.data.default_root_state[env_ids, 2]

    # Apply start yaw from path + small noise
    # Apply start yaw from actual path tangent, not from dataset yaw.
    # This prevents spawning opposite to the path direction when dataset start yaw is bad.
    look_idx = torch.full_like(valid_counts, 5)
    look_idx = torch.minimum(look_idx, (valid_counts - 1).clamp(min=1))

    p0 = paths[:, 0, :]  # [N, 2]
    p1 = paths.gather(
        1,
        look_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    ).squeeze(1)

    path_start_yaw = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])

    yaw = path_start_yaw + (torch.rand(len(env_ids), device=env.device) - 0.5) * 0.20
    cos_y = torch.cos(yaw / 2)
    sin_y = torch.sin(yaw / 2)
    zeros = torch.zeros_like(yaw)
    root_state[:, 3] = cos_y   # qw
    root_state[:, 4] = zeros   # qx
    root_state[:, 5] = zeros   # qy
    root_state[:, 6] = sin_y   # qz
    root_state[:, 7:] = 0.0

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)  # type: ignore[arg-type]

    # ---- Move goal marker to path goal ------------------------------------
    if "final_goal_marker" in env.scene.rigid_objects:
        goal_marker = env.scene.rigid_objects["final_goal_marker"]
        all_states = goal_marker.data.root_state_w.clone()
        all_states[env_ids, 0] = goals_clone[:, 0] + origins[:, 0]
        all_states[env_ids, 1] = goals_clone[:, 1] + origins[:, 1]
        all_states[env_ids, 2] = 0.16
        all_states[env_ids, 3] = 1.0  # qw
        all_states[env_ids, 4:7] = 0.0
        goal_marker.write_root_state_to_sim(all_states[env_ids], env_ids=env_ids)  # type: ignore[arg-type]



# ---------------------------------------------------------------------------
# Obstacle placement
# ---------------------------------------------------------------------------

def reset_obstacle_on_path(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    obstacle_name: str = "obstacle",
    path_frac_min: float = 0.3,
    path_frac_max: float = 0.7,
) -> None:
    """Place the obstacle at a random waypoint 30-70% along each env's path.

    Must run AFTER reset_path_tracking so navrl_global_path_xy is populated.
    """
    e: Any = env
    if obstacle_name not in env.scene.rigid_objects:
        return
    if not hasattr(e, "navrl_global_path_xy"):
        return

    obstacle = env.scene.rigid_objects[obstacle_name]
    num = len(env_ids)

    valid = e.navrl_path_valid_count[env_ids]
    low   = (path_frac_min * valid.float()).long().clamp(min=1)
    high  = (path_frac_max * valid.float()).long().clamp(min=2)
    span  = (high - low).clamp(min=1)
    idx   = (low + (torch.rand(num, device=env.device) * span.float()).long()).clamp(max=valid - 1)

    idx_exp   = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
    obs_local = e.navrl_global_path_xy[env_ids].gather(1, idx_exp).squeeze(1)  # [N, 2]
    origins   = env.scene.env_origins[env_ids, :2]                             # [N, 2]

    all_states = obstacle.data.root_state_w.clone()
    all_states[env_ids, 0] = obs_local[:, 0] + origins[:, 0]
    all_states[env_ids, 1] = obs_local[:, 1] + origins[:, 1]
    all_states[env_ids, 2] = 0.25    # half-height (box is 0.5 m tall)
    all_states[env_ids, 3] = 1.0    # qw
    all_states[env_ids, 4:7] = 0.0
    all_states[env_ids, 7:]  = 0.0  # zero velocity at spawn

    obstacle.write_root_state_to_sim(all_states[env_ids], env_ids=env_ids)  # type: ignore[arg-type]

def hold_leg_home_position(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Hold hip/knee joints at default stance using position + zero velocity targets."""
    e: Any = env
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(e, "_leg_home_joint_ids"):
        from M20_Piper.tasks.manager_based.m20_piper.mdp.config import leg_joint_names
        ids, _ = asset.find_joints(leg_joint_names)
        e._leg_home_joint_ids = ids

    leg_ids = e._leg_home_joint_ids

    pos_target = asset.data.default_joint_pos[:, leg_ids]
    vel_target = torch.zeros_like(pos_target)

    asset.set_joint_position_target(pos_target, joint_ids=leg_ids)  # type: ignore[arg-type]
    asset.set_joint_velocity_target(vel_target, joint_ids=leg_ids)  # type: ignore[arg-type]

