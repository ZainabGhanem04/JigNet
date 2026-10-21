"""Fast sanity tests (no dataset, no GPU, no trained weights needed)."""
import random

import numpy as np
import pandas as pd
import pytest

from data.prepare import build_pairs, class_weights_balanced, verify_pairs
from data.tiling import make_puzzle, split_image_into_tiles, stitch
from solver import (board_score, build_pairwise_scores, evaluate_board, gt_board, neighbor_accuracy,
                    solve_jigsaw_ilp)
from utils.constants import DOWN_IDX, LEFT_IDX, RIGHT_IDX, UP_IDX


def random_image(size=512, seed=0):
    return np.random.RandomState(seed).randint(0, 255, (size, size, 3), dtype=np.uint8)


def oracle_probs(gt_positions, g, strength=0.95):
    """Probability matrix of a perfect network for a given tile -> (row, col) assignment."""
    n = g * g
    p = np.full((n, n, 5), (1 - strength) / 4, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                p[i, j] = 0
                continue
            (ri, ci), (rj, cj) = gt_positions[i], gt_positions[j]
            cls = {(0, -1): LEFT_IDX, (0, 1): RIGHT_IDX, (-1, 0): UP_IDX, (1, 0): DOWN_IDX}.get((rj - ri, cj - ci), 0)
            p[i, j, cls] = strength
    return p


@pytest.mark.parametrize("mode", ["resize_tiles", "resize_image"])
def test_tiling_roundtrip_g4(mode):
    img = random_image()
    tiles, pos = split_image_into_tiles(img, 4, 128, mode)
    assert len(tiles) == 16 and tiles[0].shape == (128, 128, 3)
    assert pos[5] == (1, 1)
    assert np.array_equal(stitch(tiles, grid_size=4), img)


def test_modes_agree_for_4x4():
    img = random_image(seed=1)
    a, _ = split_image_into_tiles(img, 4, 128, "resize_tiles")
    b, _ = split_image_into_tiles(img, 4, 128, "resize_image")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("g", [3, 5, 6])
def test_tile_shapes_other_grids(g):
    for mode in ("resize_tiles", "resize_image"):
        tiles, pos = split_image_into_tiles(random_image(), g, 128, mode)
        assert len(tiles) == g * g and all(t.shape == (128, 128, 3) for t in tiles)


def test_shuffle_keeps_ground_truth():
    img = random_image(seed=2)
    ref, ref_pos = split_image_into_tiles(img, 4, 128, "resize_tiles")
    tiles, gt = make_puzzle(img, 4, 128, "resize_tiles", True, random.Random(0))
    assert sorted(gt.values()) == sorted(ref_pos)
    for idx, (r, c) in gt.items():
        assert np.array_equal(tiles[idx], ref[r * 4 + c])


@pytest.mark.parametrize("g", [3, 4])
def test_ilp_solves_oracle_scores(g):
    rng = random.Random(0)
    n = g * g
    order = list(range(n))
    rng.shuffle(order)
    gt_pos = {idx: divmod(k, g) for idx, k in enumerate(order)}
    scores = build_pairwise_scores(oracle_probs(gt_pos, g))
    res = solve_jigsaw_ilp(scores, g, time_limit=60)
    exact, correct = evaluate_board(res.board, gt_pos, g)
    assert exact and correct == n
    assert neighbor_accuracy(res.board, gt_pos, g) == 1.0
    assert res.objective == pytest.approx(board_score(gt_board(gt_pos, g), scores))


def _tiles_df(n_images=4, grid=4, splits=("train", "val")):
    rows = []
    for i in range(n_images):
        for k in range(grid * grid):
            rows.append({"tile_id": f"tile_{k}_img_{i}", "img_id": f"img_{i}", "row": k // grid, "col": k % grid,
                         "tile_path": f"tiles/x/tile_{k}_img_{i}.png", "split": splits[i % len(splits)]})
    return pd.DataFrame(rows)


def test_pair_counts_match_notebook_class_weights():
    """3 hard + 3 easy negatives -> 48 positives + 96 negatives per image -> weights [0.3, 2.4, 2.4, 2.4, 2.4]."""
    df = build_pairs(_tiles_df(n_images=4), 4, hard_neg=3, easy_neg=3, seed=0)
    per_image = df.groupby("image_id_a").size()
    assert (per_image == 144).all()
    assert (df.pair_type.value_counts() == 48 * 4).all()
    assert class_weights_balanced(df["label"]) == [0.3, 2.4, 2.4, 2.4, 2.4]
    assert all(verify_pairs(df, strict=False).values())


def test_pairs_are_seeded():
    t = _tiles_df(n_images=4)
    a = build_pairs(t, 4, 3, 3, seed=7)
    b = build_pairs(t, 4, 3, 3, seed=7)
    assert a.equals(b)
