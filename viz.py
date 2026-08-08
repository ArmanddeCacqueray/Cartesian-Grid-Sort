import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from PIL import Image

GRID = 65
BG = "#0a0a14"
PT_SIZE = 5

points = None


def dyadic_mids(n: int, max_depth: int) -> set:
    """Return all dyadic midpoints of [0, n-1] up to `max_depth`."""
    result: set = set()
    stack = [(0, n - 1, 1)]
    while stack:
        lo, hi, d = stack.pop()
        if d > max_depth:
            continue
        mid = (lo + hi) // 2
        if lo < mid < hi:
            result.add(mid)
        if d < max_depth:
            stack.append((lo, mid, d + 1))
            stack.append((mid, hi, d + 1))
    return result


INDEX_DEPTH = np.zeros(GRID, dtype=int)
for _i in range(GRID):
    for _d in range(1, 8):
        if _i in dyadic_mids(GRID, _d):
            INDEX_DEPTH[_i] = _d
            break
        if _i == 0 or _i == GRID - 1:
            #boundarys
            INDEX_DEPTH[_i] = 1

print("Dyadic depth of grid indices:")
for _d in range(1, 7):
    _idxs = sorted(i for i in range(GRID) if INDEX_DEPTH[i] == _d)
    print(f"  depth {_d}: {_idxs}")
print(f"  depth 0 (boundary): {sorted(i for i in range(GRID) if INDEX_DEPTH[i] == 0)}")

DEPTH_STYLE = {
    1: (2.20, 0.90, "#5599ff"),  # median + boundaryes — thick bright blue
    2: (1.40, 0.75, "#4477dd"),  # quartiles
    3: (0.85, 0.60, "#3366bb"),  # eighths
    4: (0.55, 0.45, "#2a5599"),  # sixteenths
    5: (0.38, 0.30, "#1e3d77"),  # 32nds
    6: (0.28, 0.20, "#162c55"),  # 64ths
    0: (0.18, 0.12, "#101e3a"),  # boundary / ordinary
}


def make_target_colors(grid: int) -> np.ndarray:
    R, C = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    H = C.ravel() / (grid - 1)
    S = np.full(grid * grid, 0.85)
    V = 0.35 + 0.65 * (1.0 - R.ravel() / (grid - 1))
    return mcolors.hsv_to_rgb(np.stack([H, S, V], axis=1))  # (N, 3)


TARGET_COLORS = make_target_colors(GRID)


def build_mesh(pos_2d: np.ndarray) -> dict:
    """
    Build line segments grouped by dyadic depth.

    Parameters
    ----------
    pos_2d : ndarray of shape (GRID, GRID, 2)
        Current XY position of every grid cell.

    Returns
    -------
    dict mapping depth -> list of [[x0,y0],[x1,y1]] segments
    """
    layers = {d: [] for d in DEPTH_STYLE}

    # Horizontal edges (i, j) -> (i, j+1)  — depth driven by row index i
    for i in range(GRID):
        d = INDEX_DEPTH[i]
        for j in range(GRID - 1):
            layers[d].append([pos_2d[i, j], pos_2d[i, j + 1]])

    # Vertical edges (i, j) -> (i+1, j)  — depth driven by column index j
    for j in range(GRID):
        d = INDEX_DEPTH[j]
        for i in range(GRID - 1):
            layers[d].append([pos_2d[i, j], pos_2d[i + 1, j]])

    return layers


def interp_positions(gm_a: np.ndarray,
                     gm_b: np.ndarray,
                     t: float) -> np.ndarray:
    """Linearly interpolate point positions between two grid states."""
    xa = points[gm_a.ravel()]
    xb = points[gm_b.ravel()]
    return ((1.0 - t) * xa + t * xb).reshape(GRID, GRID, 2)


def render_frame(pos_2d: np.ndarray,
                 label: str,
                 disorder_val: int,
                 progress: float) -> Image.Image:
    """
    Render one animation frame to a PIL Image.

    Parameters
    ----------
    pos_2d       : (GRID, GRID, 2) current XY positions
    label        : title string shown at the top
    disorder_val : current disorder count
    progress     : 0.0 -> 1.0, drives the progress bar
    """
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- Hierarchical mesh (thin layers first, thick on top) ----------
    layers = build_mesh(pos_2d)
    for depth in [0, 6, 5, 4, 3, 2, 1]:
        segs = layers[depth]
        if not segs:
            continue
        lw, alpha, color = DEPTH_STYLE[depth]
        lc = LineCollection(segs, colors=color, linewidths=lw,
                             alpha=alpha, zorder=1 + (7 - depth) * 0.1,
                             capstyle="round")
        ax.add_collection(lc)

    # -- Points -------------------------------------------------------
    px = pos_2d[:, :, 0].ravel()
    py = pos_2d[:, :, 1].ravel()
    ax.scatter(px, py, c=TARGET_COLORS, s=PT_SIZE, zorder=3, linewidths=0)

    # -- HUD ----------------------------------------------------------
    max_dis = 2 * GRID * (GRID - 1)
    pct = 100.0 * (1.0 - disorder_val / max_dis)

    ax.text(0.5, 1.015, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5, color="#e8e8ff",
            fontweight="bold", fontfamily="monospace")

    # Progress bar
    ax.axhspan(-0.022, -0.008, xmin=0,        xmax=progress,
               transform=ax.transAxes, color="#3a7fd5", zorder=5)
    ax.axhspan(-0.022, -0.008, xmin=progress, xmax=1,
               transform=ax.transAxes, color="#1a1a2e", zorder=5)

    ax.text(0.01, -0.035,
            f"Disorder: {disorder_val:,}    Sorted: {pct:.1f}%",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#8899bb", fontfamily="monospace")

    ax.text(0.99, -0.035, "Cartesian Sort  •  65x65",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#556688", fontfamily="monospace")

    # Mesh legend
    legend_items = [
        ("==", "#5599ff", "median (32)"),
        ("==", "#4477dd", "quartiles (16, 48)"),
        ("- ", "#3366bb", "eighths"),
        ("· ", "#4466aa", "others"),
    ]
    for k, (sym, col, txt) in enumerate(legend_items):
        ax.text(0.01, 0.99 - k * 0.033, f"{sym} {txt}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color=col, fontfamily="monospace", alpha=0.8)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")