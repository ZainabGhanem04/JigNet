"""Evaluation metrics shared by eval.py."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def bootstrap_ci(per_item_values: Sequence[float], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and (1-alpha) percentile bootstrap CI, resampling ITEMS (images), not pairs/tiles --
    pairs from one image are strongly correlated, so resampling them would give over-optimistic intervals."""
    v = np.asarray(per_item_values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = v[rng.randint(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def per_class_table(y_true: np.ndarray, top1: np.ndarray, topk_hit: np.ndarray, label_names: Dict[int, str],
                    top_k: int) -> pd.DataFrame:
    rows = []
    for c, name in label_names.items():
        m = y_true == c
        support = int(m.sum())
        c1, ck = int((top1[m] == c).sum()), int(topk_hit[m].sum())
        rows.append({"class": name, "support": support, "top1_correct": c1,
                     "top1_acc": c1 / support if support else np.nan,
                     f"top{top_k}_correct": ck, f"top{top_k}_acc": ck / support if support else np.nan})
    return pd.DataFrame(rows)


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, label_names: Dict[int, str]) -> dict:
    ids = list(label_names)
    names = [label_names[i] for i in ids]
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.full(len(ids), np.nan), where=support > 0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=ids, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=ids, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, labels=ids, target_names=names, digits=4,
                                        zero_division=0, output_dict=True),
        "report_text": classification_report(y_true, y_pred, labels=ids, target_names=names, digits=4, zero_division=0),
    }


def plot_confusion(cm: np.ndarray, names: List[str], path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title(f"{title} (counts)")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1,
                xticklabels=names, yticklabels=names, ax=axes[1])
    axes[1].set_title(f"{title} (row-normalised = recall)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
