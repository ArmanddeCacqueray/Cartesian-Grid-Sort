"""
Cartesian Sort — Animated 2D Visualization
===========================================

Generates an animated GIF showing the Cartesian Sort algorithm
(from the SquareNet library) gridifying a 65×65 point cloud in real time.

See https://github.com/ArmanddeCacqueray/Cartesian-Sort/blob/main/sort.py
for the detailed cartesian sort algorithm
"""

import os

import numpy as np
from PIL import Image

import sample
import sort
import viz

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
# Point cloud & Visualization linking
# ──────────────────────────────────────────────────────────────────

points = sample.make_cloud(N, SEED)
viz.points = points

# ──────────────────────────────────────────────────────────────────
# Cartesian Sort execution
# ──────────────────────────────────────────────────────────────────

flat = np.arange(N, dtype=np.int32)
np.random.shuffle(flat)
gridmap = flat.reshape(GRID, GRID)

key_states = []


def snap(gm: np.ndarray, label: str) -> None:
    key_states.append((gm.copy(), label))


snap(gridmap, "Initial state — unsorted points")

n_iter = 0
for n_iter in range(MAX_ITER):
    gm1 = sort.sort_columns(gridmap, points)
    snap(gm1, f"Iter {n_iter + 1} · Sorting columns (X axis ->)")
    gm2 = sort.sort_rows(gm1, points)
    snap(gm2, f"Iter {n_iter + 1} · Sorting rows    (Y axis ^)")
    gridmap = gm2
    if sort.disorder(gridmap, points) == 0:
        break

snap(gridmap, f"Perfectly sorted — {n_iter + 1} iterations")
print(f"Final disorder : {sort.disorder(gridmap, points)}")
print(f"Key states     : {len(key_states)}")

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
    dis  = sort.disorder(gm_cur, points)
    prog = k / max(1, total_steps)

    if k == 0:
        pos = points[gm_cur.ravel()].reshape(GRID, GRID, 2)
        img = viz.render_frame(pos, lbl, dis, prog)
        for _ in range(int(PAUSE_INIT * FPS)):
            add_frame(img, MS)
        print(f"  [0/{total_steps}] {lbl}  (initial pause)")
        continue

    gm_prev, _ = key_states[k - 1]
    dis_prev   = sort.disorder(gm_prev, points)

    # Smooth interpolation between previous and current state
    for step in range(INTERP_STEPS + 1):
        t      = step / INTERP_STEPS
        t_ease = t * t * (3.0 - 2.0 * t)   # smoothstep
        pos    = viz.interp_positions(gm_prev, gm_cur, t_ease)
        dis_i  = int(dis_prev + t * (dis - dis_prev))
        prog_i = (k - 1 + t) / total_steps
        add_frame(viz.render_frame(pos, lbl, dis_i, prog_i), MS)

    print(f"  [{k}/{total_steps}] {lbl}")

# Hold on the final sorted state
pos_f = points[key_states[-1][0].ravel()].reshape(GRID, GRID, 2)
img_f = viz.render_frame(pos_f, key_states[-1][1], 0, 1.0)
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
print(f"  Raw size   : {os.path.getsize(OUT) / 1e6:.1f} MB")

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