"""
Cartesian Sort — Monotone Grid Optimization via Directional Sorting
====================================================================

This script arranges N² points (sampled from a geographic or synthetic
dataset) into an N×N grid such that the assignment is "monotone" in all
directions simultaneously:

Base algorithm
    - Rows      : x increases left → right
    - Columns   : y increases top  → bottom

Diagonal extension: Base algorithm +
    - Diagonals : (x + y) increases along each ↗ diagonal
    - Anti-diag : (x - y) decreases along each ↘ diagonal

The algorithm iterates the directional sort passes until the grid is
fully monotone (zero inversions) or a maximum iteration count is reached.

As the total transport energy:

         (x_ij -i)^2 + (y_ij -j)^2 of the grid
         
is a strict monovariant of the algorithm, it is garanted not to cycle, 
and thus to converge in a finit number of iterations because only
a finite number of configurations is possible.
"""
import numpy as np

# ---------------------------------------------------------------------------
# SORT method: with or without (base cartesian grid sort) diagonal swaps 
# ---------------------------------------------------------------------------

WITH_DIAG = True #if True, row/column sort algorithm is enhanced with 
# diagonal procedure (better but slower). Feel free to compare both 
# by setting WITH_DIAG = True/False


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

point_dtype = np.dtype([
    ("x",  np.float64),
    ("y",  np.float64),
    ("id", np.int32),
])


# ---------------------------------------------------------------------------
# Directional sort passes
# ---------------------------------------------------------------------------

def sort_rows(grid: np.ndarray, N: int) -> None:
    """Sort each row by x-coordinate (ascending)."""
    for i in range(N):
        order = np.argsort(grid[i]["x"])
        grid[i] = grid[i][order]


def sort_columns(grid: np.ndarray, N: int) -> None:
    """Sort each column by y-coordinate (ascending)."""
    for j in range(N):
        order = np.argsort(grid[:, j]["y"])
        grid[:, j] = grid[:, j][order]


def sort_diagonals(grid: np.ndarray, N: int) -> None:
    """Sort each ↗ diagonal (i − j = const) by (x + y) ascending."""
    for d in range(-(N - 1), N):
        rows = np.arange(N)
        cols = rows - d
        mask = (cols >= 0) & (cols < N)
        r, c = rows[mask], cols[mask]
        if r.size <= 1:
            continue
        pts = grid[r, c]
        order = np.argsort(pts["x"] + pts["y"])
        grid[r, c] = pts[order]


def sort_antidiagonals(grid: np.ndarray, N: int) -> None:
    """Sort each ↘ anti-diagonal (i + j = const) by (x − y) descending."""
    for s in range(2 * (N - 1) + 1):
        rows = np.arange(N)
        cols = s - rows
        mask = (cols >= 0) & (cols < N)
        r, c = rows[mask], cols[mask]
        if r.size <= 1:
            continue
        pts = grid[r, c]
        order = np.argsort(-(pts["x"] - pts["y"]))   # descending
        grid[r, c] = pts[order]


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def is_monotone(grid: np.ndarray, N: int) -> bool:
    """Return True when all four monotonicity conditions are satisfied."""

    # Rows: x must be non-decreasing left → right
    if np.any(grid[:, 1:]["x"] < grid[:, :-1]["x"]):
        return False

    # Columns: y must be non-decreasing top → bottom
    if np.any(grid[1:, :]["y"] < grid[:-1, :]["y"]):
        return False
    
    if WITH_DIAG:
        # ↗ Diagonals: (x + y) must be non-decreasing
        for d in range(-(N - 1), N):
            rows = np.arange(N)
            cols = rows - d
            mask = (cols >= 0) & (cols < N)
            r, c = rows[mask], cols[mask]
            vals = grid[r, c]["x"] + grid[r, c]["y"]
            if np.any(vals[1:] < vals[:-1]):
                return False

        # ↘ Anti-diagonals: (x − y) must be non-increasing
        for s in range(2 * (N - 1) + 1):
            rows = np.arange(N)
            cols = s - rows
            mask = (cols >= 0) & (cols < N)
            r, c = rows[mask], cols[mask]
            vals = grid[r, c]["x"] - grid[r, c]["y"]
            if np.any(vals[1:] > vals[:-1]):
                return False

    return True


def count_inversions(grid: np.ndarray, N: int) -> int:
    """Count the total number of violated monotonicity constraints."""
    inv = 0

    # Row inversions
    inv += int(np.sum(grid[:, 1:]["x"] < grid[:, :-1]["x"]))

    # Column inversions
    inv += int(np.sum(grid[1:, :]["y"] < grid[:-1, :]["y"]))

    if WITH_DIAG:

        # Diagonal inversions
        for d in range(-(N - 1), N):
            rows = np.arange(N)
            cols = rows - d
            mask = (cols >= 0) & (cols < N)
            r, c = rows[mask], cols[mask]
            vals = grid[r, c]["x"] + grid[r, c]["y"]
            inv += int(np.sum(vals[1:] < vals[:-1]))

        # Anti-diagonal inversions
        for s in range(2 * (N - 1) + 1):
            rows = np.arange(N)
            cols = s - rows
            mask = (cols >= 0) & (cols < N)
            r, c = rows[mask], cols[mask]
            vals = grid[r, c]["x"] - grid[r, c]["y"]
            inv += int(np.sum(vals[1:] > vals[:-1]))

    return inv


def compute_ot_loss(grid: np.ndarray, N: int) -> float:
    """
    Compute the squared L2 transport cost/energy between the grid 
    positions and the uniform [0,1]² lattice.
    """
    col_coords, row_coords = np.meshgrid(
        np.linspace(0, 1, N),
        np.linspace(0, 1, N),
    )
    dx = grid["x"] - col_coords
    dy = grid["y"] - row_coords
    return float(np.sum(dx**2 + dy**2))

def main():
    """
    Run full cartesian grid sort on a random point cloud (65**2 points)
    """
    N = 65
    MAX_ITER = 1000
    pts = np.random.rand(N, N, 2)
    grid = np.empty((N, N), dtype=point_dtype)
    grid["x"] = pts[:, :, 0]
    grid["y"] = pts[:, :, 1]

    # ── Optimisation loop ───────────────────────────────────────────────────
    loss_history       = [compute_ot_loss(grid, N)]
    inversion_history  = [count_inversions(grid, N)]
    iteration = 0

    while iteration < MAX_ITER and not is_monotone(grid, N):
        sort_rows(grid, N)
        sort_columns(grid, N)
        if WITH_DIAG:
            sort_diagonals(grid, N)
            sort_antidiagonals(grid, N)

        iteration += 1
        loss_history.append(compute_ot_loss(grid, N))
        inversion_history.append(count_inversions(grid, N))

    # ── Results ──────────────────────────────────────────────────────────────
    print(f"\n{'Iter':>6}  {'Inversions':>12}  {'OT Loss':>14}")
    print("-" * 38)
    for i, (inv, loss) in enumerate(zip(inversion_history, loss_history)):
        print(f"{i:>6}  {inv:>12}  {loss:>14.3f}")

    if inversion_history[-1] == 0:
        print(f"successfully sorted at iteration {len(inversion_history) - 1}")

if __name__ == "__main__":
    main()