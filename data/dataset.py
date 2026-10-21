"""PyTorch dataset over pairs.csv."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resolve_paths(paths: pd.Series, root: str, legacy_prefix: Optional[str] = None) -> np.ndarray:
    """Map the tile paths stored in a CSV to this machine.

    * paths that start with `legacy_prefix` (e.g. the old Colab/Drive folder) are re-rooted under `root`;
    * other relative paths are taken relative to `root`;
    * other absolute paths are left alone.
    """
    root_p = Path(root)
    out = []
    for p in paths.astype(str):
        if legacy_prefix and p.startswith(legacy_prefix):
            out.append(str(root_p / p[len(legacy_prefix):].lstrip("/")))
        elif Path(p).is_absolute():
            out.append(p)
        else:
            out.append(str(root_p / p))
    return np.array(out)


class JigsawPairDataset(Dataset):
    """Yields (tile_a, tile_b, label) for one split of pairs.csv. Tiles are RGB, passed through `transform`."""

    def __init__(self, pairs_csv: str, split: str, root: str, transform=None, legacy_prefix: Optional[str] = None):
        df = pd.read_csv(pairs_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with split == {split!r} in {pairs_csv}")
        self.data = df
        self.path_a = resolve_paths(df["tile_path_a"], root, legacy_prefix)
        self.path_b = resolve_paths(df["tile_path_b"], root, legacy_prefix)
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _read(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image at path: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        a, b = self._read(self.path_a[idx]), self._read(self.path_b[idx])
        if self.transform is not None:
            a, b = self.transform(a), self.transform(b)
        return a, b, torch.tensor(self.labels[idx], dtype=torch.long)
