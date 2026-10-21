"""Dataset preparation: validate -> split -> tile -> verify -> build pair manifest.

    python -m data.prepare --config configs/prepare_dataset.yaml

Output layout under cfg.root:
    images/{train,val,test}/img_<id>.png
    tiles/{train,val,test}/tile_<k>_img_<id>.png
    metadata/images_splits.csv  tiles.csv  pairs.csv  class_weights.json  pair_distribution.csv
All paths inside the CSVs are RELATIVE to cfg.root, so the dataset can be moved between machines.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import load_config, parse_args, save_config
from utils.constants import CLASS_LABEL, OPPOSITE

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ----------------------------------------------------------------------------- 1. validation
def validate_images(source_dir: str, expected_shape: Tuple[int, int, int]) -> Tuple[List[str], dict]:
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(IMG_EXTS))
    valid, invalid, unreadable = [], [], []
    for name in files:
        img = cv2.imread(os.path.join(source_dir, name))
        if img is None:
            unreadable.append(name)
        elif img.shape != tuple(expected_shape):
            invalid.append({"file": name, "shape": list(img.shape)})
        else:
            valid.append(name)
    report = {"found": len(files), "valid": len(valid), "invalid_shape": invalid, "unreadable": unreadable,
              "expected_shape": list(expected_shape)}
    return valid, report


def clean_name(name: str, strip_suffix: str) -> str:
    """img_12_0.png -> img_12.png (the SDXL generator appends _<k>)."""
    stem, ext = os.path.splitext(name)
    if strip_suffix and stem.endswith(strip_suffix):
        stem = stem[: -len(strip_suffix)]
    return stem + ext


# ----------------------------------------------------------------------------- 2. splitting
def make_splits(names: List[str], seed: int, train_frac: float, existing_csv: str = None) -> Dict[str, List[str]]:
    """Return {split: [file names]} (names as they appear after clean_name).

    If `existing_csv` (an images_splits.csv from an earlier run) is given it is the source of truth: the
    original notebook split `os.listdir()` output, whose order is filesystem dependent, so a fresh split
    is NOT guaranteed to match the one the published model was trained on.
    """
    if existing_csv:
        df = pd.read_csv(existing_csv)
        by_stem = {os.path.splitext(n)[0]: n for n in names}
        splits = {"train": [], "val": [], "test": []}
        missing = []
        for image_id, split in zip(df["image_id"], df["split"]):
            if image_id in by_stem:
                splits[split].append(by_stem[image_id])
            else:
                missing.append(image_id)
        if missing:
            raise FileNotFoundError(f"{len(missing)} images from {existing_csv} not found, e.g. {missing[:3]}")
        return splits
    names = sorted(names)                                           # deterministic order, unlike os.listdir
    train, rest = train_test_split(names, train_size=train_frac, random_state=seed, shuffle=True)
    val, test = train_test_split(rest, train_size=0.5, random_state=seed, shuffle=True)
    return {"train": train, "val": val, "test": test}


# ----------------------------------------------------------------------------- 3. tiling
def split_image(image: np.ndarray, grid_size: int, tile_h: int, tile_w: int) -> List[np.ndarray]:
    return [image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            for i in range(grid_size) for j in range(grid_size)]


def write_tiles(root: Path, splits: Dict[str, List[str]], grid_size: int, tile_h: int, tile_w: int) -> pd.DataFrame:
    rows = []
    for split, names in splits.items():
        out_dir = root / "tiles" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            img = cv2.imread(str(root / "images" / split / name))
            image_name, ext = os.path.splitext(name)
            idx = image_name.split("_")[-1] if "_" in image_name else image_name
            for k, tile in enumerate(split_image(img, grid_size, tile_h, tile_w)):
                tile_name = f"tile_{k}_img_{idx}{ext}"
                if not cv2.imwrite(str(out_dir / tile_name), tile):
                    raise IOError(f"failed to write {tile_name}")
                rows.append({"tile_id": os.path.splitext(tile_name)[0], "img_id": image_name,
                             "row": k // grid_size, "col": k % grid_size,
                             "tile_path": str(Path("tiles") / split / tile_name), "split": split})
    return pd.DataFrame(rows)


def check_reconstruction(root: Path, tiles_df: pd.DataFrame, splits_df: pd.DataFrame, grid_size, tile_h, tile_w):
    """Stitch every image back from its tiles and require pixel-identical results."""
    bad = []
    for img_id, g in tiles_df.groupby("img_id"):
        canvas = np.zeros((grid_size * tile_h, grid_size * tile_w, 3), dtype=np.uint8)
        for r in g.itertuples():
            canvas[r.row * tile_h:(r.row + 1) * tile_h, r.col * tile_w:(r.col + 1) * tile_w] = \
                cv2.imread(str(root / r.tile_path))
        orig = cv2.imread(str(root / splits_df.loc[splits_df.image_id == img_id, "image_path"].iloc[0]))
        if not np.array_equal(canvas, orig):
            bad.append(img_id)
    if bad:
        raise AssertionError(f"{len(bad)} images are not reconstructed exactly from their tiles, e.g. {bad[:3]}")
    print(f"[PASS] all {tiles_df.img_id.nunique()} images are reconstructed pixel-exactly from their tiles")


# ----------------------------------------------------------------------------- 4. pairs
def get_positive_neighbors(img_id, row, col, tile_lookup, max_rows, max_cols) -> Dict[str, Tuple[int, int]]:
    directions = {"LEFT": (row, col - 1), "RIGHT": (row, col + 1), "UP": (row - 1, col), "DOWN": (row + 1, col)}
    return {name: (r, c) for name, (r, c) in directions.items()
            if 0 <= r < max_rows and 0 <= c < max_cols and (img_id, r, c) in tile_lookup}


def get_hard_negatives(row, col, coords, pos_coords_set, num_samples, rng: random.Random):
    """Tiles of the same image that are neither the anchor nor one of its 4-neighbours."""
    cand = [c for c in coords if c not in pos_coords_set and c != (row, col)]
    return rng.sample(cand, min(num_samples, len(cand)))


def build_pairs(tiles_df: pd.DataFrame, grid_size: int, hard_neg: int, easy_neg: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)
    lookup = {(r.img_id, r.row, r.col): r._asdict() for r in tiles_df.itertuples(index=False)}
    image_grids = tiles_df.groupby("img_id")[["row", "col"]].apply(lambda g: list(zip(g["row"], g["col"]))).to_dict()
    split_pool = {s: g.reset_index(drop=True) for s, g in tiles_df.groupby("split")}

    pairs, pid = [], 0

    def add(a, b, label_name, pair_type):
        nonlocal pid
        pairs.append({
            "pair_id": pid, "image_id_a": a["img_id"], "image_id_b": b["img_id"],
            "tile_id_a": a["tile_id"], "tile_id_b": b["tile_id"],
            "tile_path_a": a["tile_path"], "tile_path_b": b["tile_path"],
            "row_a": a["row"], "col_a": a["col"], "row_b": b["row"], "col_b": b["col"],
            "label": CLASS_LABEL[label_name], "label_name": label_name,
            "pair_type": pair_type, "split": a["split"]})
        pid += 1

    for anchor in tiles_df.itertuples(index=False):
        a = anchor._asdict()
        pos = get_positive_neighbors(a["img_id"], a["row"], a["col"], lookup, grid_size, grid_size)
        for rel, (r, c) in pos.items():
            add(a, lookup[(a["img_id"], r, c)], rel, "positive")

        for (r, c) in get_hard_negatives(a["row"], a["col"], image_grids[a["img_id"]], set(pos.values()), hard_neg, rng):
            add(a, lookup[(a["img_id"], r, c)], "no_match", "hard")

        pool = split_pool[a["split"]]
        n_other = (pool["img_id"] != a["img_id"]).sum()
        if n_other < easy_neg:
            raise ValueError(f"split {a['split']!r} has too few tiles from other images for {easy_neg} easy negatives")
        chosen = set()
        while len(chosen) < easy_neg:
            j = int(nprng.randint(len(pool)))
            if pool.at[j, "img_id"] != a["img_id"]:
                chosen.add(j)
        for j in sorted(chosen):
            t = pool.iloc[j]
            add(a, {"img_id": t["img_id"], "tile_id": t["tile_id"], "tile_path": t["tile_path"],
                    "row": int(t["row"]), "col": int(t["col"])}, "no_match", "easy")
    return pd.DataFrame(pairs)


# ----------------------------------------------------------------------------- 5. verification
def verify_pairs(df: pd.DataFrame, strict: bool = True) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    checks["unique pair_id"] = df["pair_id"].is_unique
    checks["no missing values"] = int(df.isnull().sum().sum()) == 0
    img_split = pd.concat([df[["image_id_a", "split"]].rename(columns={"image_id_a": "i"}),
                           df[["image_id_b", "split"]].rename(columns={"image_id_b": "i"})])
    leaked = int((img_split.groupby("i")["split"].nunique() > 1).sum())
    # image_id_b of an easy negative is in the SAME split as the anchor by construction
    checks["no image spans several splits"] = leaked == 0

    pos = df[df.pair_type == "positive"]
    checks["positives share one image"] = bool((pos.image_id_a == pos.image_id_b).all())
    dr, dc = pos.row_b - pos.row_a, pos.col_b - pos.col_a
    ok = (((pos.label_name == "LEFT") & (dr == 0) & (dc == -1)) | ((pos.label_name == "RIGHT") & (dr == 0) & (dc == 1)) |
          ((pos.label_name == "UP") & (dr == -1) & (dc == 0)) | ((pos.label_name == "DOWN") & (dr == 1) & (dc == 0)))
    checks["positive offsets match labels"] = bool(ok.all())
    hard, easy = df[df.pair_type == "hard"], df[df.pair_type == "easy"]
    checks["hard negatives from the same image"] = bool((hard.image_id_a == hard.image_id_b).all())
    checks["hard negatives are not 4-neighbours"] = bool(
        ((hard.row_a - hard.row_b).abs() + (hard.col_a - hard.col_b).abs() != 1).all())
    checks["easy negatives from other images"] = bool((easy.image_id_a != easy.image_id_b).all())
    checks["no tile paired with itself"] = bool((df.tile_id_a != df.tile_id_b).all())

    lookup = {(a, b): n for a, b, n in zip(pos.tile_id_a, pos.tile_id_b, pos.label_name)}
    checks["reciprocal relations symmetric"] = all(
        lookup.get((b, a)) in (None, OPPOSITE[n]) for (a, b), n in lookup.items())

    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if strict and not all(checks.values()):
        raise AssertionError("pairs.csv failed verification: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def class_weights_balanced(labels: pd.Series, num_classes: int = 5) -> List[float]:
    """total / (num_classes * count_c) -- sklearn 'balanced' weights (what the notebook used)."""
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0)
    return [round(float(len(labels) / (num_classes * c)), 4) if c else 0.0 for c in counts]


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args("Prepare the jigsaw dataset")
    cfg = load_config(args.config, args.overrides)
    root = Path(cfg.root)
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    save_config(cfg, meta / "prepare_config.yaml")

    # 1. validate
    shape = (cfg.image.height, cfg.image.width, cfg.image.channels)
    valid, report = validate_images(cfg.source_dir, shape)
    (meta / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"found {report['found']} | valid {report['valid']} | wrong shape {len(report['invalid_shape'])} "
          f"| unreadable {len(report['unreadable'])}")
    if report["invalid_shape"] or report["unreadable"]:
        print("  (invalid/unreadable files are excluded; see metadata/validation_report.json)")

    # 2. copy (non-destructively renamed) and split
    renamed = {clean_name(n, cfg.get("strip_suffix", "")): n for n in valid}
    splits = make_splits(list(renamed), cfg.seed, cfg.split.train, cfg.split.get("existing_csv"))
    rows = []
    for split, names in splits.items():
        d = root / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(os.path.join(cfg.source_dir, renamed[n]), d / n)
            rows.append({"image_id": os.path.splitext(n)[0], "image_path": str(Path("images") / split / n), "split": split})
    splits_df = pd.DataFrame(rows)
    splits_df.to_csv(meta / "images_splits.csv", index=False)
    print({k: len(v) for k, v in splits.items()})

    # 3. tiles
    g, th, tw = cfg.grid_size, cfg.tile.height, cfg.tile.width
    tiles_df = write_tiles(root, splits, g, th, tw)
    tiles_df.to_csv(meta / "tiles.csv", index=False)
    check_reconstruction(root, tiles_df, splits_df, g, th, tw)

    # 4. pairs
    pairs_df = build_pairs(tiles_df, g, cfg.pairs.hard_neg, cfg.pairs.easy_neg, cfg.seed)
    pairs_df.to_csv(meta / "pairs.csv", index=False)
    print(f"{len(pairs_df)} pairs written")

    # 5. verify + summaries
    verify_pairs(pairs_df, strict=True)
    dist = pd.crosstab(pairs_df.label_name, pairs_df.split, margins=True)
    dist.to_csv(meta / "pair_distribution.csv")
    print(dist.to_string())
    weights = class_weights_balanced(pairs_df.loc[pairs_df.split == "train", "label"])
    (meta / "class_weights.json").write_text(json.dumps({"train_balanced": weights}))
    print("balanced class weights (train):", weights)


if __name__ == "__main__":
    main()
