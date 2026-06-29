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

        self.files = sorted(self.dataset_dir.glob("path_*.npz"))
        if not self.files:
            raise RuntimeError(f"No path_*.npz files found in {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def sample_batch(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample one path per env_id.

        Returns:
            starts        [N, 3]               — (x, y, yaw) per env
            goals         [N, 3]               — (x, y, yaw) per env
            paths         [N, max_path_points, 2]
            valid_counts  [N]  long
            path_lengths  [N]
        """
        num = len(env_ids)
        starts = torch.zeros(num, 3, device=self.device)
        goals = torch.zeros(num, 3, device=self.device)
        paths = torch.zeros(num, self.max_path_points, 2, device=self.device)
        valid_counts = torch.zeros(num, dtype=torch.long, device=self.device)
        path_lengths = torch.zeros(num, device=self.device)

        file_indices = torch.randint(len(self.files), (num,)).cpu().numpy()

        for i, file_idx in enumerate(file_indices):
            data = np.load(self.files[int(file_idx)])

            start = data["start"].astype(np.float32)     # [3]
            goal = data["goal"].astype(np.float32)       # [3]
            path_xy = data["path_xy"].astype(np.float32) # [K, 2]

            n = min(len(path_xy), self.max_path_points)

            starts[i] = torch.tensor(start, device=self.device)
            goals[i] = torch.tensor(goal, device=self.device)
            paths[i, :n] = torch.tensor(path_xy[:n], device=self.device)
            valid_counts[i] = n

            if "path_length" in data:
                path_lengths[i] = float(data["path_length"][0])
            else:
                diff = path_xy[1:] - path_xy[:-1]
                path_lengths[i] = float(np.linalg.norm(diff, axis=1).sum())

        return starts, goals, paths, valid_counts, path_lengths
