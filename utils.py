"""
utility to check convergence of the cartesian sort algorithm 
"""

import numpy as np
from numpy.typing import NDArray

GRID = 65

def disorder(gridmap: NDArray, xy: NDArray) -> int:
    """Count adjacent inversions along both axes (disorder metric).
    Termination criterion of cartesian sort"""
    d = 0
    for i in range(GRID):
        for j in range(GRID):
            if i < GRID - 1:
                d += xy[gridmap[i, j], 0] > xy[gridmap[i + 1, j], 0]
            if j < GRID - 1:
                d += xy[gridmap[i, j], 1] > xy[gridmap[i, j + 1], 1]
    return int(d)

def check_bijective(k, IJ):
    """
    sanity check (optional, for demonstration purpose)
    """
    return np.array_equal(np.sort(IJ.ravel()), k)