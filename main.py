from sort import point_dtype, compute_ot_loss, count_inversions, is_monotone
from sort import sort_rows, sort_columns, sort_diagonals, sort_antidiagonals

import io
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from squarenet import SquareNet


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_frame(grid: np.ndarray) -> Image.Image:
    """Render the current grid state to a PIL Image."""

    sn = SquareNet(gridshape=grid.shape)
    sn.pointsmaped = np.stack([grid["x"], grid["y"]], axis=-1)

    fig, _ = sn.plot(
        style="mesh",
        mesh_long_edge=10,
        save=False,
        show=False,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=120,
    )
    buf.seek(0)

    image = Image.open(buf).convert("RGB").copy()

    plt.close(fig)
    buf.close()

    return image


def show_final(grid: np.ndarray) -> None:
    """Display the final grid state interactively."""

    sn = SquareNet(gridshape=grid.shape)
    sn.pointsmaped = np.stack([grid["x"], grid["y"]], axis=-1)

    sn.neighbormap()
    sn.plot(
        style="mesh",
        mesh_long_edge=10,
        save=False,
        show=True,
    )


def save_gif(frames, filename="cartesian_sort.gif", duration=1000):
    """Save PIL frames directly as an animated GIF.

    Parameters
    ----------
    frames : list[PIL.Image.Image]
        Animation frames.
    filename : str
        Output GIF filename.
    duration : int
        Duration of each frame in milliseconds.
    """

    if not frames:
        return

    frames[0].save(
        filename,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ── Configuration ────────────────────────────────────────────────────────
    DATASET      = "holy" #more datasets available ("france", "germany", "holy", "ball", "spiky"...)
    DYNAMIC_PLOT = True
    N            = 65
    MAX_ITER     = 1000

    # ─────────────────────────────────────────────────────────────────────────

    # Sample N² points and reshape into an N×N structured array
    from squarenet.sampler import samplepoints

    pts = samplepoints(
        method=DATASET,
        size=(N**2, 2),
        plot_points=False,
    )

    pts = pts.reshape(N, N, 2)

    grid = np.empty((N, N), dtype=point_dtype)
    grid["x"] = pts[:, :, 0]
    grid["y"] = pts[:, :, 1]

    # ── Optimisation loop ───────────────────────────────────────────────────

    loss_history = [
        compute_ot_loss(grid, N)
    ]

    inversion_history = [
        count_inversions(grid, N)
    ]

    frames = []

    def update_gif(grid, frames, gif_it):
            # Capture the resulting state.
            if DYNAMIC_PLOT and (gif_it <= 20 or (gif_it <= 80 & gif_it %  4 == 0) or gif_it % 12 == 0):
                frames.append(render_frame(grid))
            gif_it += 1
            return gif_it

    gif_it = update_gif(grid, frames, 0)

    start = time.perf_counter()
    iteration = 0

    while iteration < MAX_ITER and not is_monotone(grid, N):
        # One full pass: four directional sorts
        sort_rows(grid, N)
        gif_it = update_gif(grid, frames, gif_it)
        sort_columns(grid, N)
        gif_it = update_gif(grid, frames, gif_it)
        sort_diagonals(grid, N)
        gif_it = update_gif(grid, frames, gif_it)
        sort_antidiagonals(grid, N)
        gif_it = update_gif(grid, frames, gif_it)

        iteration += 1

        loss_history.append(
            compute_ot_loss(grid, N)
        )

        inversion_history.append(
            count_inversions(grid, N)
        )

    elapsed_ms = (time.perf_counter() - start) * 1_000

    # ── Results ──────────────────────────────────────────────────────────────

    print(f"Iterations    : {iteration}")
    print(f"Runtime (ms)  : {elapsed_ms:.1f}")
    print(
        f"Monotone      : "
        f"{'yes' if is_monotone(grid, N) else 'no'}"
    )

    print(f"\n{'Iter':>6}  {'Inversions':>12}  {'OT Loss':>14}")
    print("-" * 38)

    for i, (inv, loss) in enumerate(
        zip(inversion_history, loss_history)
    ):
        print(
            f"{i:>6}  "
            f"{inv:>12}  "
            f"{loss:>14.3f}"
        )

    # ── Output ───────────────────────────────────────────────────────────────

    if DYNAMIC_PLOT and frames:

        save_gif(
            frames,
            filename="sort.gif",
            duration=500,
        )

        print(
            "\nAnimation saved → cartesian_sort.gif"
        )

    show_final(grid)


if __name__ == "__main__":
    main()