"""Cutting images into tiles and stitching them back (pure numpy / OpenCV, no torch)."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TILING_MODES = ("resize_image", "resize_tiles")


def split_grid(img: np.ndarray, grid_size: int, tile_h: int, tile_w: int) -> List[np.ndarray]:
    """Non-overlapping tile_h x tile_w crops in row-major order (tile index = row * grid_size + col)."""
    return [img[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            for r in range(grid_size) for c in range(grid_size)]


def split_image_into_tiles(img_rgb: np.ndarray, grid_size: int, tile_size: int = 128,
                           mode: str = "resize_tiles") -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Cut an RGB image into grid_size x grid_size tiles of tile_size x tile_size.

    mode:
      "resize_tiles" : crop (h // g) x (w // g) tiles from the image (dropping any remainder pixels), then
                       resize every tile to tile_size.   <- used by the full-pipeline benchmark
      "resize_image" : resize the whole image to (g * tile_size)^2 with INTER_AREA, then cut tiles.
                       <- used by the network-only evaluation
    For grid_size == 4 and a 512x512 image both modes give identical tiles.

    Returns (tiles, positions) with positions[k] = (row, col) of tile k, row-major.
    """
    if mode not in TILING_MODES:
        raise ValueError(f"tiling mode must be one of {TILING_MODES}, got {mode!r}")
    g = grid_size
    if mode == "resize_image":
        side = g * tile_size
        img = cv2.resize(img_rgb, (side, side), interpolation=cv2.INTER_AREA)
        tiles = split_grid(img, g, tile_size, tile_size)
    else:
        h, w = img_rgb.shape[:2]
        th, tw = h // g, w // g
        if th == 0 or tw == 0:
            raise ValueError(f"Image {w}x{h} is too small for a {g}x{g} grid.")
        tiles = [cv2.resize(t, (tile_size, tile_size)) for t in split_grid(img_rgb, g, th, tw)]
    positions = [(r, c) for r in range(g) for c in range(g)]
    return tiles, positions


def make_puzzle(img_rgb: np.ndarray, grid_size: int, tile_size: int = 128, mode: str = "resize_tiles",
                shuffle: bool = True, rng: Optional[random.Random] = None):
    """Tiles (optionally shuffled) + ground truth.

    Returns (tiles, gt_positions) where gt_positions = {tile_index: (row, col) in the original image}.
    """
    tiles, positions = split_image_into_tiles(img_rgb, grid_size, tile_size, mode)
    order = list(range(len(tiles)))
    if shuffle:
        (rng or random).shuffle(order)
    shuffled = [tiles[k] for k in order]
    gt_positions = {idx: positions[k] for idx, k in enumerate(order)}
    return shuffled, gt_positions


def stitch(tiles: Sequence[np.ndarray], order: Optional[Sequence[int]] = None, grid_size: int = 4,
           gap: int = 0, bg: int = 255) -> np.ndarray:
    """Stitch tiles into a grid image. order[k] = index of the tile placed in cell k (row-major)."""
    if order is None:
        order = list(range(len(tiles)))
    th, tw = tiles[0].shape[:2]
    canvas = np.full((grid_size * th + (grid_size - 1) * gap,
                      grid_size * tw + (grid_size - 1) * gap, 3), bg, dtype=np.uint8)
    for k, t in enumerate(order):
        r, c = divmod(k, grid_size)
        y, x = r * (th + gap), c * (tw + gap)
        canvas[y:y + th, x:x + tw] = tiles[t]
    return canvas
