"""ILP jigsaw solver (PuLP).

Maximise the total RIGHT/DOWN edge score of a g x g arrangement subject to:
  * every cell holds exactly one tile, every tile sits in exactly one cell;
  * an edge variable H[i, j, r, c] (tile i at (r, c) and tile j at (r, c+1)) is on iff the placement
    variables agree. Linking is done with equality "flow" constraints, which have the same integer
    solutions as the pairwise H <= X formulation but a much tighter LP relaxation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pulp

from .board import Board, board_score


def _new_binary(prob, name):
    """PuLP >= 4 creates variables through the problem; PuLP < 4 uses LpVariable."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, 0, 1, "Binary")
    return pulp.LpVariable(name, 0, 1, cat="Binary")


def get_solver(time_limit: float, backend: str = "auto"):
    """backend: auto | cbc | highs.  `auto` = CBC when bundled with PuLP, otherwise HiGHS (needs highspy)."""
    if backend == "cbc" or (backend == "auto" and hasattr(pulp, "PULP_CBC_CMD")):
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    return pulp.HiGHS(msg=False, timeLimit=time_limit)


def _solver_outcome(prob, solve_return, solve_time: float, time_limit: float) -> Tuple[str, bool]:
    """(status string, optimality proven?) for both PuLP APIs. A run stopped by the time limit never
    counts as proven optimal."""
    if hasattr(solve_return, "status") and hasattr(pulp, "LpSolveStatus"):          # PuLP >= 4
        status = getattr(solve_return.status, "name", str(solve_return.status))
        proven = solve_return.status == pulp.LpSolveStatus.Optimal
    else:                                                                           # PuLP < 4
        status = pulp.LpStatus.get(prob.status, "Unknown")
        proven = status == "Optimal"
        sol_status = getattr(prob, "sol_status", None)
        if sol_status is not None and hasattr(pulp, "LpSolutionOptimal"):
            proven = proven and sol_status == pulp.LpSolutionOptimal
    return status, proven and solve_time < time_limit


@dataclass
class SolveResult:
    board: Optional[Board]
    status: str            # solver status string (e.g. "Optimal")
    proven_optimal: bool   # True only if optimality was proven (not stopped by the time limit)
    objective: float       # total RIGHT/DOWN score of the selected arrangement
    runtime: float         # seconds: model building + solving


def solve_jigsaw_ilp(scores, grid_size: int, time_limit: float = 60, backend: str = "auto") -> SolveResult:
    R = C = grid_size
    T = grid_size * grid_size
    t0 = time.perf_counter()

    prob = pulp.LpProblem("Jigsaw_Puzzle_Solver", pulp.LpMaximize)

    def binvar(name, *idx):
        return _new_binary(prob, name + "_" + "_".join(map(str, idx)))

    X = {(i, r, c): binvar("X", i, r, c) for i in range(T) for r in range(R) for c in range(C)}
    H = {(i, j, r, c): binvar("H", i, j, r, c)
         for i in range(T) for j in range(T) if i != j for r in range(R) for c in range(C - 1)}
    V = {(i, j, r, c): binvar("V", i, j, r, c)
         for i in range(T) for j in range(T) if i != j for r in range(R - 1) for c in range(C)}

    # Objective: compatibility of all horizontal + vertical neighbours
    terms = []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            terms += [scores[(i, j, "RIGHT")] * H[i, j, r, c] for r in range(R) for c in range(C - 1)]
            terms += [scores[(i, j, "DOWN")] * V[i, j, r, c] for r in range(R - 1) for c in range(C)]
    prob += pulp.lpSum(terms)

    # One tile per cell, one cell per tile
    for r in range(R):
        for c in range(C):
            prob += pulp.lpSum(X[i, r, c] for i in range(T)) == 1
    for i in range(T):
        prob += pulp.lpSum(X[i, r, c] for r in range(R) for c in range(C)) == 1

    # Edge variables <-> tile placement
    for r in range(R):
        for c in range(C - 1):
            for i in range(T):
                prob += pulp.lpSum(H[i, j, r, c] for j in range(T) if j != i) == X[i, r, c]
                prob += pulp.lpSum(H[j, i, r, c] for j in range(T) if j != i) == X[i, r, c + 1]
    for r in range(R - 1):
        for c in range(C):
            for i in range(T):
                prob += pulp.lpSum(V[i, j, r, c] for j in range(T) if j != i) == X[i, r, c]
                prob += pulp.lpSum(V[j, i, r, c] for j in range(T) if j != i) == X[i, r + 1, c]

    t_solve = time.perf_counter()
    solve_return = prob.solve(get_solver(time_limit, backend))
    solve_time = time.perf_counter() - t_solve
    status, proven = _solver_outcome(prob, solve_return, solve_time, time_limit)

    board: Board = [[None] * C for _ in range(R)]
    for i in range(T):
        for r in range(R):
            for c in range(C):
                v = pulp.value(X[i, r, c])
                if v is not None and v > 0.5:
                    board[r][c] = i
    placed = [t for row in board for t in row]
    valid = None not in placed and len(set(placed)) == T

    return SolveResult(
        board=board if valid else None,
        status=status if valid else f"{status} (no valid board)",
        proven_optimal=proven and valid,
        objective=board_score(board, scores) if valid else float("nan"),
        runtime=time.perf_counter() - t0,
    )
