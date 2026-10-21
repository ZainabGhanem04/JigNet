"""Evaluate JigNet. One command per experiment, selected by `mode` in the config:

    mode: pairs     classification report / confusion matrix on a split of pairs.csv
    mode: network   JigNet on every ordered tile pair of full images, for several grid sizes
    mode: pipeline  network + ILP solver: reconstruct shuffled puzzles, for several grid sizes

    python eval.py --config configs/eval_network_test.yaml
    python eval.py --config configs/eval_pipeline_coco.yaml --set grid_sizes=[3,4] solver.time_limit=30

Results (per-image CSVs, metrics.json, confusion matrices, summary.csv, config + environment) are written
to <output.runs_dir>/<name>/.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import JigsawPairDataset
from data.tiling import make_puzzle, split_image_into_tiles
from models.checkpoint import load_jignet
from models.inference import normalize_transform, predict_probs_matrix, prepare_tiles
from solver import (board_score, build_pairwise_scores, evaluate_board, gt_board, harden, neighbor_accuracy,
                    solve_jigsaw_ilp)
from utils.config import load_config, parse_args
from utils.constants import DELTAS, LABEL_NAMES, NO_MATCH_IDX
from utils.metrics import bootstrap_ci, classification_summary, per_class_table, plot_confusion
from utils.runs import make_run_dir
from utils.seed import set_seed

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _dump(obj, path: Path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default))


def list_images(folder: str, max_images=None) -> List[Path]:
    files = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files[:max_images] if max_images else files


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def true_labels(positions: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Ground-truth class of every ordered pair (A, B) from the tiles' grid positions."""
    d = positions[ib] - positions[ia]
    y = np.full(len(ia), NO_MATCH_IDX, dtype=np.int64)
    for cls, (dr, dc) in DELTAS.items():
        y[(d[:, 0] == dr) & (d[:, 1] == dc)] = cls
    return y


# ----------------------------------------------------------------------------- mode: pairs
def eval_pairs(cfg, model, device, run_dir: Path):
    tf = normalize_transform(cfg.data.mean, cfg.data.std)
    ds = JigsawPairDataset(str(Path(cfg.data.root) / cfg.data.pairs_csv), cfg.split, cfg.data.root, tf,
                           cfg.data.get("legacy_prefix"))
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 256), shuffle=False, num_workers=cfg.data.get("num_workers", 2))
    ys, ps = [], []
    with torch.no_grad():
        for a, b, y in loader:
            ps.append(model(a.to(device), b.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    summ = classification_summary(y_true, y_pred, LABEL_NAMES)
    print(f"[{cfg.split}] {len(y_true)} pairs | acc {summ['accuracy']:.2%} | balanced acc {summ['balanced_accuracy']:.2%} "
          f"| macro F1 {summ['macro_f1']:.4f}\n{summ['report_text']}")
    plot_confusion(np.array(summ["confusion_matrix"]), list(LABEL_NAMES.values()), run_dir / f"confusion_{cfg.split}.png",
                   f"pairs / {cfg.split}")
    _dump({k: v for k, v in summ.items() if k != "report_text"}, run_dir / "metrics.json")


# ----------------------------------------------------------------------------- mode: network
def eval_network(cfg, model, device, run_dir: Path):
    images = list_images(cfg.data.images_dir, cfg.get("max_images"))
    top_k = cfg.network.top_k
    summary_rows = []
    for g in cfg.grid_sizes:
        gdir = run_dir / f"grid_{g}"
        gdir.mkdir(exist_ok=True)
        n = g * g
        print(f"\n=== network | grid {g}x{g} | {n} tiles -> {n * (n - 1)} ordered pairs/image | {len(images)} images ===")
        ia, ib = np.where(~np.eye(n, dtype=bool))
        pair_frames, per_image = [], []
        Y, T1, TK = [], [], []
        for k, path in enumerate(images, 1):
            tiles, positions = split_image_into_tiles(read_rgb(path), g, cfg.tiling.tile_size, cfg.tiling.mode)
            pos = np.array(positions)
            tiles_t = prepare_tiles(tiles, cfg.data.mean, cfg.data.std, device)
            probs = predict_probs_matrix(model, tiles_t, cfg.network.pair_batch_size)[ia, ib]    # (P, 5)
            y = true_labels(pos, ia, ib)
            topk = np.argsort(-probs, axis=1)[:, :top_k]
            top1 = topk[:, 0]
            hit_k = (topk == y[:, None]).any(axis=1)
            adj = y != NO_MATCH_IDX
            per_image.append({
                "image": path.name, "pairs": len(y), "top1_acc": float((top1 == y).mean()),
                f"top{top_k}_acc": float(hit_k.mean()), "relationships_total": int(adj.sum()),
                "relationships_correct_top1": int(((top1 == y) & adj).sum()),
                "relationship_acc_top1": float(((top1 == y) & adj).sum() / adj.sum()),
                f"relationships_correct_top{top_k}": int((hit_k & adj).sum()),
                "false_adjacent_pred": int(((top1 != NO_MATCH_IDX) & ~adj).sum()),
            })
            Y.append(y); T1.append(top1); TK.append(hit_k)
            if cfg.network.get("save_pair_predictions", False):
                pair_frames.append(pd.DataFrame({
                    "image": path.name, "tile_a": ia, "tile_b": ib, "row_a": pos[ia, 0], "col_a": pos[ia, 1],
                    "row_b": pos[ib, 0], "col_b": pos[ib, 1], "true": y, "top1": top1,
                    **{f"score_{LABEL_NAMES[c]}": probs[:, c] for c in LABEL_NAMES}}))
            if k % 25 == 0 or k == len(images):
                print(f"  [{k}/{len(images)}] running top-1 acc: {np.mean([r['top1_acc'] for r in per_image]):.2%}")

        y_true, y_pred, hit_k = np.concatenate(Y), np.concatenate(T1), np.concatenate(TK)
        adj = y_true != NO_MATCH_IDX
        summ = classification_summary(y_true, y_pred, LABEL_NAMES)
        img_df = pd.DataFrame(per_image)
        ci = {m: bootstrap_ci(img_df[m], cfg.bootstrap.n, cfg.bootstrap.seed)
              for m in ("top1_acc", "relationship_acc_top1")}
        cls_df = per_class_table(y_true, y_pred, hit_k, LABEL_NAMES, top_k)

        metrics = {
            "grid_size": g, "n_images": len(images), "n_pairs": int(len(y_true)),
            "top1_acc": summ["accuracy"], f"top{top_k}_acc": float(hit_k.mean()),
            "balanced_accuracy": summ["balanced_accuracy"], "macro_f1": summ["macro_f1"], "weighted_f1": summ["weighted_f1"],
            "relationship_acc_top1": float(((y_true == y_pred) & adj).sum() / adj.sum()),
            f"relationship_acc_top{top_k}": float((hit_k & adj).sum() / adj.sum()),
            "false_adjacent_rate": float(((y_pred != NO_MATCH_IDX) & ~adj).sum() / (~adj).sum()),
            "per_image_top1_acc_ci95": ci["top1_acc"], "per_image_relationship_acc_ci95": ci["relationship_acc_top1"],
            "confusion_matrix": summ["confusion_matrix"], "report": summ["report"],
        }
        _dump(metrics, gdir / "metrics.json")
        img_df.to_csv(gdir / "per_image_summary.csv", index=False)
        cls_df.to_csv(gdir / "per_class_summary.csv", index=False)
        if pair_frames:
            pd.concat(pair_frames, ignore_index=True).to_csv(gdir / "pair_predictions.csv", index=False)
        plot_confusion(np.array(summ["confusion_matrix"]), list(LABEL_NAMES.values()), gdir / "confusion_matrix.png",
                       f"grid {g}x{g}")
        print(summ["report_text"])
        print(f"top-1 {metrics['top1_acc']:.2%} | balanced acc {metrics['balanced_accuracy']:.2%} | macro F1 {metrics['macro_f1']:.4f} | "
              f"relationship acc {metrics['relationship_acc_top1']:.2%} "
              f"(per-image 95% CI {ci['relationship_acc_top1'][1]:.2%}-{ci['relationship_acc_top1'][2]:.2%})")
        summary_rows.append({k: v for k, v in metrics.items()
                             if not isinstance(v, (dict, list)) } | {
            "relationship_acc_ci_lo": ci["relationship_acc_top1"][1], "relationship_acc_ci_hi": ci["relationship_acc_top1"][2]})
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)


# ----------------------------------------------------------------------------- mode: pipeline
def eval_pipeline(cfg, model, device, run_dir: Path):
    images = list_images(cfg.data.images_dir, cfg.get("max_images"))
    s = cfg.solver
    summary_rows = []
    for g in cfg.grid_sizes:
        gdir = run_dir / f"grid_{g}"
        gdir.mkdir(exist_ok=True)
        n = g * g
        rng = random.Random(cfg.seed)                       # one stream per grid size, images in sorted order
        print(f"\n=== pipeline | grid {g}x{g} | {n} tiles | {len(images)} images | ILP limit {s.time_limit}s ===")
        rows = []
        for k, path in enumerate(images, 1):
            try:
                tiles, gt_pos = make_puzzle(read_rgb(path), g, cfg.tiling.tile_size, cfg.tiling.mode,
                                            cfg.get("shuffle", True), rng)
                tiles_t = prepare_tiles(tiles, cfg.data.mean, cfg.data.std, device)
                probs = predict_probs_matrix(model, tiles_t, cfg.get("pair_batch_size", 256))
                if s.probs == "hard":
                    probs = harden(probs)
                scores = build_pairwise_scores(probs, s.score)
                res = solve_jigsaw_ilp(scores, g, s.time_limit, s.backend)
            except Exception as e:                         # keep going, but never silently
                print(f"  [{k}/{len(images)}] {path.name}: ERROR ({e})")
                rows.append({"image": path.name, "error": str(e)})
                continue
            exact, correct = evaluate_board(res.board, gt_pos, g)
            gt_obj = board_score(gt_board(gt_pos, g), scores)
            search_err = bool(res.board is not None and res.objective + 1e-6 < gt_obj)
            rows.append({
                "image": path.name, "status": res.status, "proven_optimal": res.proven_optimal,
                "objective": res.objective, "gt_objective": gt_obj, "runtime_s": res.runtime,
                "exact_match": exact, "correct_tiles": correct, "num_tiles": n,
                "neighbor_acc": neighbor_accuracy(res.board, gt_pos, g),
                # solver returned a board worse than the ground truth -> ILP stopped early / failed
                "search_error": search_err,
                # a board at least as good as the ground truth was chosen, but it is wrong -> JigNet's scores are the problem
                "model_error": (not exact) and (not search_err),
            })
            print(f"  [{k}/{len(images)}] {path.name}: {'OK  ' if exact else 'FAIL'} | {res.status} optimal={res.proven_optimal} | "
                  f"obj {res.objective:.3f} (GT {gt_obj:.3f}) | {res.runtime:.2f}s | tiles {correct}/{n}")
        df = pd.DataFrame(rows)
        df.to_csv(gdir / "per_image_results.csv", index=False)
        ok = df[df.get("error").isna()] if "error" in df else df
        m = len(ok)
        ci = {c: bootstrap_ci(ok[c].astype(float), cfg.bootstrap.n, cfg.bootstrap.seed)
              for c in ("exact_match", "neighbor_acc")}
        tile_per_img = ok["correct_tiles"] / ok["num_tiles"]
        tile_ci = bootstrap_ci(tile_per_img, cfg.bootstrap.n, cfg.bootstrap.seed)
        metrics = {
            "grid_size": g, "n_images": len(images), "n_evaluated": m, "n_errors": len(df) - m,
            "perfect_reconstructions": int(ok.exact_match.sum()), "perfect_rate": float(ok.exact_match.mean()),
            "perfect_rate_ci95": ci["exact_match"][1:],
            "tile_accuracy": float(ok.correct_tiles.sum() / (m * n)), "tile_accuracy_ci95": tile_ci[1:],
            "neighbor_accuracy": float(ok.neighbor_acc.mean()), "neighbor_accuracy_ci95": ci["neighbor_acc"][1:],
            "optimality_proven": int(ok.proven_optimal.sum()),
            "search_errors": int(ok.search_error.sum()), "model_errors": int(ok.model_error.sum()),
            "avg_objective": float(ok.objective.mean()), "avg_gt_objective": float(ok.gt_objective.mean()),
            "runtime_mean_s": float(ok.runtime_s.mean()), "runtime_median_s": float(ok.runtime_s.median()),
            "runtime_max_s": float(ok.runtime_s.max()),
        }
        _dump(metrics, gdir / "metrics.json")
        print("\n" + "=" * 60 + f"\n FINAL REPORT | grid {g}x{g} ({n} tiles)\n" + "=" * 60)
        print(f"Images evaluated:            {m}/{len(images)}")
        print(f"Perfect reconstructions:     {metrics['perfect_reconstructions']}/{m} ({metrics['perfect_rate']:.2%}, "
              f"95% CI {ci['exact_match'][1]:.2%}-{ci['exact_match'][2]:.2%})")
        print(f"Tile placement accuracy:     {metrics['tile_accuracy']:.2%}")
        print(f"Neighbour accuracy:          {metrics['neighbor_accuracy']:.2%}")
        print(f"Optimality proven:           {metrics['optimality_proven']}/{m}")
        print(f"Failures: model errors {metrics['model_errors']} | solver search errors {metrics['search_errors']}")
        print(f"Solver time per puzzle:      mean {metrics['runtime_mean_s']:.2f}s, median {metrics['runtime_median_s']:.2f}s, "
              f"max {metrics['runtime_max_s']:.2f}s")
        summary_rows.append({k: (v if not isinstance(v, (list, tuple)) else None) for k, v in metrics.items()} |
                            {"perfect_rate_ci_lo": ci["exact_match"][1], "perfect_rate_ci_hi": ci["exact_match"][2]})
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)


MODES = {"pairs": eval_pairs, "network": eval_network, "pipeline": eval_pipeline}


def main():
    args = parse_args("Evaluate JigNet")
    cfg = load_config(args.config, args.overrides)
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    set_seed(cfg.seed, cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg.output.runs_dir, cfg.name, cfg)
    model = load_jignet(cfg.checkpoint, device, cfg.get("num_classes", 5))
    print(f"mode: {cfg.mode} | device: {device} | checkpoint: {cfg.checkpoint} | results -> {run_dir}")
    t0 = time.time()
    MODES[cfg.mode](cfg, model, device, run_dir)
    print(f"\ndone in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
