"""
Cartesian Sort — Animated 2D Visualization
===========================================

Generates an animated GIF showing the Cartesian Sort algorithm
(from the SquareNet library) gridifying a 65×65 point cloud in real time.

What you see
------------
- 4 225 colored points (65×65) drawn from a non-uniform distribution
  (arc, clusters, noise). Each color encodes the point's *target* grid
  cell — hue = column, brightness = row — so you can track every point
  like a card in a deck being sorted.
- A hierarchical mesh drawn over the grid connections. Line thickness
  and opacity reflect the dyadic depth of each row/column index:
    depth 1  →  median      index 32           (thickest, bright blue)
    depth 2  →  quartiles   indices 16, 48
    depth 3  →  eighths     indices 8, 24, 40, 56
    depth 4  →  sixteenths  indices 4, 12, …, 60
    depth 5  →  32nds       (32 lines)
    depth 6  →  64ths       (32 lines)
    depth 0  →  boundary    indices 0 and 64   (nearly invisible)
  65 = 2^6 + 1, so the grid is perfectly dyadic.
- Smooth interpolation between sorting steps (ease-in/out), like
  watching a deck of cards slide into place one pass at a time.
- A disorder counter and progress bar in the HUD.

Algorithm (Cartesian Sort — "fast" mode from SquareNet)
--------------------------------------------------------
  Initialize gridmap  <-  random permutation of point indices

  repeat until disorder == 0:
      for each column j:
          sort gridmap[:, j] by points[*, 0]   (X coordinate)
      for each row i:
          sort gridmap[i, :] by points[*, 1]   (Y coordinate)

Each 1-D sort is O(N log N). Sorting axis 1 partially undoes axis 0,
so we iterate. Convergence is typically < 10 passes for 65x65.

Requirements
------------
  pip install numpy matplotlib pillow

Usage
-----
  python cartesian_sort_viz.py
  # -> cartesian_sort.gif  (~15 s, ~16 MB)

Tune the constants at the top of the file to adjust the output.
"""

import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from PIL import Image

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

GRID         = 65          # grid is GRID x GRID
N            = GRID * GRID # total number of points
MAX_ITER     = 60          # safety cap on sorting iterations
SEED         = 7           # reproducibility

FPS          = 15
MS           = int(1000 / FPS)   # ms per frame

INTERP_STEPS = 6           # interpolation frames between two sort states
PAUSE_INIT   = 3           # seconds to hold on the initial frame
PAUSE_END    = 4           # seconds to hold on the final frame

BG           = "#0a0a14"   # background color
PT_SIZE      = 5           # scatter point size (pts^2)
OUT          = "cartesian_sort.gif"

np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────
# Dyadic mesh hierarchy
# ──────────────────────────────────────────────────────────────────
# For a grid of size 65 = 2^6 + 1, every index has a natural "depth"
# in the binary subdivision tree:
#   depth 1 -> median      {32}
#   depth 2 -> quartiles   {16, 48}
#   depth 3 -> eighths     {8, 24, 40, 56}
#   ...
# We use this depth to set line width and opacity.

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


# Pre-compute the depth of every index 0 ... GRID-1
INDEX_DEPTH = np.zeros(GRID, dtype=int)
for _i in range(GRID):
    for _d in range(1, 8):
        if _i in dyadic_mids(GRID, _d):
            INDEX_DEPTH[_i] = _d
            break
        if _i == 0 or _i == GRID - 1:
            #boundarys
            INDEX_DEPTH[_i] = 1

# Print the hierarchy for reference
print("Dyadic depth of grid indices:")
for _d in range(1, 7):
    _idxs = sorted(i for i in range(GRID) if INDEX_DEPTH[i] == _d)
    print(f"  depth {_d}: {_idxs}")
print(f"  depth 0 (boundary): {sorted(i for i in range(GRID) if INDEX_DEPTH[i] == 0)}")

# Visual style per depth: (linewidth, alpha, hex_color)
DEPTH_STYLE = {
    1: (2.20, 0.90, "#5599ff"),  # median + boundaryes — thick bright blue
    2: (1.40, 0.75, "#4477dd"),  # quartiles
    3: (0.85, 0.60, "#3366bb"),  # eighths
    4: (0.55, 0.45, "#2a5599"),  # sixteenths
    5: (0.38, 0.30, "#1e3d77"),  # 32nds
    6: (0.28, 0.20, "#162c55"),  # 64ths
    0: (0.18, 0.12, "#101e3a"),  # boundary / ordinary
}

# ──────────────────────────────────────────────────────────────────
# Point cloud
# ──────────────────────────────────────────────────────────────────

def make_cloud(n: int, seed: int) -> np.ndarray:
    """
    Build a non-uniform 2-D point cloud with:
      - an open arc (ring segment)
      - two dense clusters
      - two small satellite clusters
      - uniform background noise
    All coordinates clipped to (0.01, 0.99).
    """
    rng = np.random.default_rng(seed)
    parts = []

    # Open arc
    t = rng.uniform(np.pi * 0.1, np.pi * 1.9, n // 4)
    r = rng.uniform(0.25, 0.38, n // 4)
    parts.append(np.stack([0.5 + r * np.cos(t),
                            0.45 + r * np.sin(t)], axis=1))
    # Dense cluster — top-left
    parts.append(rng.normal([0.20, 0.80], 0.07, (n // 6, 2)))
    # Dense cluster — bottom-right
    parts.append(rng.normal([0.80, 0.20], 0.07, (n // 6, 2)))
    # Small satellite — bottom-left
    parts.append(rng.normal([0.15, 0.25], 0.05, (n // 10, 2)))
    # Small satellite — top-right
    parts.append(rng.normal([0.85, 0.78], 0.05, (n // 10, 2)))
    # Uniform background noise
    remaining = n - sum(len(p) for p in parts)
    parts.append(rng.uniform(0.02, 0.98, (remaining, 2)))

    cloud = np.clip(np.vstack(parts), 0.01, 0.99)
    return cloud[rng.permutation(len(cloud))[:n]]


points = make_cloud(N, SEED)

# ──────────────────────────────────────────────────────────────────
# Target colors  (fixed per grid cell, not per point)
# ──────────────────────────────────────────────────────────────────
# Color encoding: hue = column / (GRID-1), value = 1 - row / (GRID-1)
# -> a 2-D HSV gradient: sorted state = smooth rainbow,
#    unsorted state = colorful confetti.

def make_target_colors(grid: int) -> np.ndarray:
    R, C = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    H = C.ravel() / (grid - 1)
    S = np.full(grid * grid, 0.85)
    V = 0.35 + 0.65 * (1.0 - R.ravel() / (grid - 1))
    return mcolors.hsv_to_rgb(np.stack([H, S, V], axis=1))  # (N, 3)


TARGET_COLORS = make_target_colors(GRID)

# ──────────────────────────────────────────────────────────────────
# Cartesian Sort
# ──────────────────────────────────────────────────────────────────

def sort_columns(gridmap: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Sort each column of `gridmap` by the X coordinate of its points."""
    out = gridmap.copy()
    for j in range(GRID):
        col = out[:, j]
        out[:, j] = col[np.argsort(pts[col, 0])]
    return out


def sort_rows(gridmap: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Sort each row of `gridmap` by the Y coordinate of its points."""
    out = gridmap.copy()
    for i in range(GRID):
        row = out[i, :]
        out[i, :] = row[np.argsort(pts[row, 1])]
    return out


def disorder(gridmap: np.ndarray, pts: np.ndarray) -> int:
    """Count adjacent inversions along both axes (disorder metric)."""
    d  = int(np.sum(pts[gridmap[:-1, :], 0] > pts[gridmap[1:,  :], 0]))
    d += int(np.sum(pts[gridmap[:, :-1], 1] > pts[gridmap[:, 1:  ], 1]))
    return d


# Start from a random permutation of point indices
flat = np.arange(N, dtype=np.int32)
np.random.shuffle(flat)
gridmap = flat.reshape(GRID, GRID)

# Collect key states (snapshots to animate between)
key_states = []

def snap(gm: np.ndarray, label: str) -> None:
    key_states.append((gm.copy(), label))


snap(gridmap, "Initial state — unsorted points")

n_iter = 0
for n_iter in range(MAX_ITER):
    gm1 = sort_columns(gridmap, points)
    snap(gm1, f"Iter {n_iter + 1} · Sorting columns (X axis ->)")
    gm2 = sort_rows(gm1, points)
    snap(gm2, f"Iter {n_iter + 1} · Sorting rows    (Y axis ^)")
    gridmap = gm2
    if disorder(gridmap, points) == 0:
        break

snap(gridmap, f"Perfectly sorted — {n_iter + 1} iterations")
print(f"Final disorder : {disorder(gridmap, points)}")
print(f"Key states     : {len(key_states)}")

# ──────────────────────────────────────────────────────────────────
# Hierarchical mesh builder
# ──────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────

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
            f"Disorder: {disorder_val:,}   Sorted: {pct:.1f}%",
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

# ──────────────────────────────────────────────────────────────────
# Assemble frames
# ──────────────────────────────────────────────────────────────────

pil_frames = []
frame_durs = []
total_steps = len(key_states) - 1


def add_frame(img: Image.Image, ms: int) -> None:
    pil_frames.append(img)
    frame_durs.append(ms)


print("Rendering frames...")

for k, (gm_cur, lbl) in enumerate(key_states):
    dis  = disorder(gm_cur, points)
    prog = k / max(1, total_steps)

    if k == 0:
        pos = points[gm_cur.ravel()].reshape(GRID, GRID, 2)
        img = render_frame(pos, lbl, dis, prog)
        for _ in range(int(PAUSE_INIT * FPS)):
            add_frame(img, MS)
        print(f"  [0/{total_steps}] {lbl}  (initial pause)")
        continue

    gm_prev, _ = key_states[k - 1]
    dis_prev   = disorder(gm_prev, points)

    # Smooth interpolation between previous and current state
    for step in range(INTERP_STEPS + 1):
        t      = step / INTERP_STEPS
        t_ease = t * t * (3.0 - 2.0 * t)   # smoothstep
        pos    = interp_positions(gm_prev, gm_cur, t_ease)
        dis_i  = int(dis_prev + t * (dis - dis_prev))
        prog_i = (k - 1 + t) / total_steps
        add_frame(render_frame(pos, lbl, dis_i, prog_i), MS)

    print(f"  [{k}/{total_steps}] {lbl}")

# Hold on the final sorted state
pos_f = points[key_states[-1][0].ravel()].reshape(GRID, GRID, 2)
img_f = render_frame(pos_f, key_states[-1][1], 0, 1.0)
for _ in range(int(PAUSE_END * FPS)):
    add_frame(img_f, MS)

print(f"\nTotal frames : {len(pil_frames)}")
print(f"Duration     : {sum(frame_durs) / 1000:.1f}s")

# ──────────────────────────────────────────────────────────────────
# Export GIF  (raw -> quantized)
# ──────────────────────────────────────────────────────────────────

print(f"Saving {OUT}...")
pil_frames[0].save(
    OUT,
    save_all=True,
    append_images=pil_frames[1:],
    duration=frame_durs,
    loop=0,
    optimize=False,
)
print(f"  Raw size  : {os.path.getsize(OUT) / 1e6:.1f} MB")

# Re-encode with color quantization to shrink the file
gif      = Image.open(OUT)
q_frames = []
q_durs   = []
try:
    while True:
        q_frames.append(
            gif.copy().convert("RGB").quantize(
                colors=220, method=Image.Quantize.MEDIANCUT
            )
        )
        q_durs.append(gif.info.get("duration", MS))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

q_frames[0].save(
    OUT,
    save_all=True,
    append_images=q_frames[1:],
    duration=q_durs,
    loop=0,
    optimize=True,
)
print(f"  Optimized : {os.path.getsize(OUT) / 1e6:.1f} MB")
print("Done!")
