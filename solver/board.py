"""Boards (grid of tile indices), scoring and evaluation metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Board = List[List[Optional[int]]]
Positions = Dict[int, Tuple[int, int]]   # tile index -> (row, col) it belongs to in the original image


def board_score(board: Board, scores) -> float:
    """Sum of edge scores over all horizontal and vertical neighbours of a board."""
    rows, cols = len(board), len(board[0])
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                total += scores[(board[r][c], board[r][c + 1], "RIGHT")]
            if r + 1 < rows:
                total += scores[(board[r][c], board[r + 1][c], "DOWN")]
    return total


def gt_board(gt_positions: Positions, grid_size: int) -> Board:
    """The ground-truth arrangement expressed as a board of tile indices."""
    board: Board = [[None] * grid_size for _ in range(grid_size)]
    for tile, (r, c) in gt_positions.items():
        board[r][c] = tile
    return board


def evaluate_board(board: Optional[Board], gt_positions: Positions, grid_size: int) -> Tuple[bool, int]:
    """(board matches the original exactly, number of tiles in their correct cell)."""
    if board is None:
        return False, 0
    correct = sum(1 for r in range(grid_size) for c in range(grid_size)
                  if gt_positions[board[r][c]] == (r, c))
    return correct == grid_size * grid_size, correct


def neighbor_accuracy(board: Optional[Board], gt_positions: Positions, grid_size: int) -> float:
    """Fraction of the 2*g*(g-1) adjacent pairs in the solved board that are true neighbours in the original.

    Unlike absolute tile accuracy this is not destroyed when a correctly assembled block is shifted.
    """
    if board is None:
        return 0.0
    ok = total = 0
    for r in range(grid_size):
        for c in range(grid_size):
            pr, pc = gt_positions[board[r][c]]
            if c + 1 < grid_size:
                qr, qc = gt_positions[board[r][c + 1]]
                ok += (qr, qc) == (pr, pc + 1)
                total += 1
            if r + 1 < grid_size:
                qr, qc = gt_positions[board[r + 1][c]]
                ok += (qr, qc) == (pr + 1, pc)
                total += 1
    return ok / total
