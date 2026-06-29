from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Nav2PathDataset:
    """Loads Nav2-generated global paths saved as path_*.npz files.

    Each npz file contains:
        start      [3]    — (x, y, yaw) of path start in local env frame
        goal       [3]    — (x, y, yaw) of path goal in local env frame
        path_xy    [N, 2] — dense waypoints in local env frame
        path_length [1]   — total arc length (optional, recomputed if missing)
    """

    def __init__(self, dataset_dir: str, device: str, max_path_points: int = 600):
        self.dataset_dir = Path(dataset_dir)
        self.device = device
        self.max_path_points = max_path_points

        files = sorted(self.dataset_dir.glob("path_*.npz"))
        if not files:
            raise RuntimeError(f"No path_*.npz files found in {self.dataset_dir}")

        # Pre-load all paths into RAM so sample_batch never hits disk.
        # For a typical dataset of 100-1000 paths × 600 points this is <50 MB.
        self._cache: list[dict] = []
        for f in files:
            data = np.load(f)
            xy = data["path_xy"].astype(np.float32)
            diff = xy[1:] - xy[:-1]
            length = float(np.linalg.norm(diff, axis=1).sum()) if len(xy) > 1 else 0.0
            self._cache.append({
                "start":       data["start"].astype(np.float32),
                "goal":        data["goal"].astype(np.float32),
                "path_xy":     xy,
                "path_length": float(data["path_length"][0]) if "path_length" in data else length,
            })
        print(f"[Nav2PathDataset] Loaded {len(self._cache)} paths into RAM.")

    def __len__(self) -> int:
        return len(self._cache)

    def sample_batch(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample one path per env_id from in-memory cache (no disk I/O).

        Returns:
            starts        [N, 3]               — (x, y, yaw) per env
            goals         [N, 3]               — (x, y, yaw) per env
            paths         [N, max_path_points, 2]
            valid_counts  [N]  long
            path_lengths  [N]
        """
        num = len(env_ids)
        starts = torch.zeros(num, 3, device=self.device)
        goals  = torch.zeros(num, 3, device=self.device)
        paths  = torch.zeros(num, self.max_path_points, 2, device=self.device)
        valid_counts = torch.zeros(num, dtype=torch.long, device=self.device)
        path_lengths = torch.zeros(num, device=self.device)

        indices = torch.randint(len(self._cache), (num,)).tolist()

        for i, idx in enumerate(indices):
            entry   = self._cache[idx]
            path_xy = entry["path_xy"]          # numpy [K, 2], already float32
            n       = min(len(path_xy), self.max_path_points)

            starts[i]       = torch.from_numpy(entry["start"]).to(self.device)
            goals[i]        = torch.from_numpy(entry["goal"]).to(self.device)
            paths[i, :n]    = torch.from_numpy(path_xy[:n]).to(self.device)
            valid_counts[i] = n
            path_lengths[i] = entry["path_length"]

        return starts, goals, paths, valid_counts, path_lengths
