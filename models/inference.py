"""Run JigNet on tiles: preprocessing and all-pairs probability matrix."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torchvision import transforms

from utils.constants import MEAN, STD


def normalize_transform(mean: Sequence[float] = MEAN, std: Sequence[float] = STD):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=list(mean), std=list(std))])


def prepare_tiles(tiles: List[np.ndarray], mean=MEAN, std=STD, device="cpu") -> torch.Tensor:
    """List of RGB uint8 (H, W, 3) tiles -> normalised float32 tensor (N, 3, H, W)."""
    tf = normalize_transform(mean, std)
    return torch.stack([tf(t) for t in tiles]).to(device)


@torch.no_grad()
def predict_probs_matrix(model, tiles: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    """probs[i, j] = softmax probabilities of the 5 relations of tile j with respect to tile i.

    Shape (N, N, 5); the diagonal (i == j) is left at zero. Every ordered pair is scored once.
    """
    n = tiles.shape[0]
    ia, ib = np.where(~np.eye(n, dtype=bool))            # row-major order of all (i, j), i != j
    probs_matrix = np.zeros((n, n, 5), dtype=np.float32)
    for s in range(0, len(ia), batch_size):
        a = torch.as_tensor(ia[s:s + batch_size], device=tiles.device)
        b = torch.as_tensor(ib[s:s + batch_size], device=tiles.device)
        p = torch.softmax(model(tiles[a], tiles[b]), dim=1).cpu().numpy()
        probs_matrix[ia[s:s + batch_size], ib[s:s + batch_size]] = p
    return probs_matrix
