import time
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

GRID = 65


def sort_columns(gridmap: NDArray, xy: NDArray) -> NDArray:
    """Sort each column of `gridmap` by the X coordinate of its points."""
    out = gridmap.copy()
    for j in range(GRID):
        col = out[:, j]
        out[:, j] = col[np.argsort(xy[col, 0])]
    return out


def sort_rows(gridmap: NDArray, xy: NDArray) -> NDArray:
    """Sort each row of `gridmap` by the Y coordinate of its points."""
    out = gridmap.copy()
    for i in range(GRID):
        row = out[i, :]
        out[i, :] = row[np.argsort(xy[row, 1])]
    return out


def disorder(gridmap: NDArray, xy: NDArray) -> int:
    """Count adjacent inversions along both axes (disorder metric)."""
    d = 0
    for i in range(GRID):
        for j in range(GRID):
            if i < GRID - 1:
                d += xy[gridmap[i, j], 0] > xy[gridmap[i + 1, j], 0]
            if j < GRID - 1:
                d += xy[gridmap[i, j], 1] > xy[gridmap[i, j + 1], 1]
    return int(d)


def check_bijective(k: NDArray, IJ: NDArray) -> bool:
    """Sanity check (optional): Check that `IJ` contains each point index exactly once."""
    return np.array_equal(np.sort(IJ.ravel()), k)


def check_monotone(XY: NDArray) -> Tuple[bool, bool]:
    """Sanity check (optional): Check monotonicity of X along rows and Y along columns."""
    X, Y = XY[:, :, 0], XY[:, :, 1]
    monotone_X = np.all(np.diff(X, axis=0) >= 0)
    monotone_Y = np.all(np.diff(Y, axis=1) >= 0)
    return monotone_X, monotone_Y


def cartesian_sort(
    xy: NDArray, max_iter: int = 100
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Gridify a 2D point cloud using the Cartesian Grid Sort algorithm.

    Iteratively sorts columns by X coordinates and rows by Y coordinates
    until convergence (disorder == 0) or until `max_iter` is reached.

    Inputs:
        xy (NDArray): 2D array of shape (N, 2) containing point (x, y) coordinates,
                      where N = GRID * GRID.
        max_iter (int, optional): Safety cap - maximum number of algorithm
                                  iterations before forced termination.
                                  Defaults to 100.

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray]:
            - xy: Original point cloud of shape (N, 2).
            - k: Original flat index array of shape (N,) - natural ordering 0...N-1.
            - XY: Gridified array of shape (GRID, GRID, 2) where XY[I, J]
                  gives (x, y)[k].
            - IJ: Index array of shape (GRID, GRID) mapping grid positions
                  [I, J] back to original point indices k.

    Note:
        The algorithm is guaranteed to converge - and quickly in practice - because the 
        following Transport Metric is a monovariant energy of the system until convergence:
            Sum_I,J { (X[I, J] - I)^2 + (Y[I, J] - J)^2 }
    """
    n = len(xy) #here n should be equal to GRID^2.
    #Else apply padding with dummy +-infinite xy points.
    k = np.arange(n)

    # Initial random grid permutation IJ
    IJ = np.random.permutation(k).reshape(GRID, GRID)

    for _ in range(max_iter):
        IJ = sort_columns(IJ, xy)
        IJ = sort_rows(IJ, xy)
        if disorder(IJ, xy) == 0:
            break

    # Reconstruct 2D spatial grid: shape (GRID, GRID, 2)
    XY = xy[IJ]

    # ====================
    # Algorithmic guarantees:
    # ====================

    # Verify bijection
    assert check_bijective(k, IJ), (
        "Mapping is not bijective: k <-> (I, J)"
    )

    # Verify monotonicity
    monotone_X, monotone_Y = check_monotone(XY)
    assert monotone_X and monotone_Y, (
        "Convergence failed: X / Y coordinates should be monotonic "
        "along rows/columns.\n"
        "Consider increasing max_iter."
    )

    return xy, k, XY, IJ


if __name__ == "__main__":
    np.random.seed(42)

    n = GRID * GRID

    # Generate a random point cloud
    xy = np.random.rand(n, 2)

    # Initial disorder
    k = np.arange(n)
    IJ = np.random.permutation(k).reshape(GRID, GRID)

    print(f"Cloud shape: {xy.shape}")
    print(f"Initial disorder: {disorder(IJ, xy)}")

    # Run Cartesian Sort
    start = time.perf_counter()
    xy, k, XY, IJ = cartesian_sort(xy)
    elapsed = time.perf_counter() - start

    print(f"Grid shape: {XY.shape}")
    print(f"Final disorder: {disorder(IJ, xy)}")
    print(f"Mapping valid: {check_bijective(k, IJ)}")

    print(f"Elapsed time: {elapsed:.4f} s")
