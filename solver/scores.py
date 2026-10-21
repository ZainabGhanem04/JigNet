"""Turn JigNet probabilities into the edge scores the ILP maximises."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from utils.constants import DOWN_IDX, LEFT_IDX, RIGHT_IDX, UP_IDX

PairwiseScores = Dict[Tuple[int, int, str], float]


def harden(probs_matrix: np.ndarray) -> np.ndarray:
    """Replace softmax outputs by one-hot argmax predictions (ablation: throw away confidence)."""
    onehot = np.eye(probs_matrix.shape[-1], dtype=np.float32)[probs_matrix.argmax(axis=-1)]
    onehot[np.arange(len(probs_matrix)), np.arange(len(probs_matrix))] = 0.0   # keep diagonal empty
    return onehot


def build_pairwise_scores(probs_matrix: np.ndarray, transform: str = "prob", eps: float = 1e-6) -> PairwiseScores:
    """Edge scores for every ordered tile pair (i, j).

    scores[(i, j, "RIGHT")] : j sits to the right of i   (average of "i->j is RIGHT" and "j->i is LEFT")
    scores[(i, j, "DOWN")]  : j sits below i             (average of "i->j is DOWN"  and "j->i is UP")

    transform:
        "prob" : average probability                        (the setting used for the reported results)
        "log"  : log(average probability + eps)             (log-likelihood style costs, for ablations)
    Note: every full board uses exactly 2*g*(g-1) edges, so adding a constant to all scores never
    changes the optimum; but a non-linear transform such as log does.
    """
    if transform not in ("prob", "log"):
        raise ValueError(f"unknown score transform: {transform}")
    n = probs_matrix.shape[0]
    scores: PairwiseScores = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            right = float((probs_matrix[i, j, RIGHT_IDX] + probs_matrix[j, i, LEFT_IDX]) / 2.0)
            down = float((probs_matrix[i, j, DOWN_IDX] + probs_matrix[j, i, UP_IDX]) / 2.0)
            if transform == "log":
                right, down = float(np.log(right + eps)), float(np.log(down + eps))
            scores[(i, j, "RIGHT")] = right
            scores[(i, j, "DOWN")] = down
    return scores
