# Cartesian Sort Algorithm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArmanddeCacqueray/SquareNet/blob/main/00_getting_started.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/squarenet.svg)](https://pypi.org/project/squarenet/)
[![Documentation Status](https://readthedocs.org/projects/squarenet/badge/?version=latest)](https://squarenet.readthedocs.io/en/latest/)
[![GitHub](https://img.shields.io/badge/GitHub-Source-6f42c1?logo=github)](https://github.com/ArmanddeCacqueray/SquareNet)
[![HF demo](https://img.shields.io/badge/🤗%20Open%20in%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/adec314/point-to-grid)

This repository illustrates the core Cartesian sort procedure of the `SquareNet` ❒ gridification engine for demonstration purposes.

It showcases the live progress of the multi-key Cartesian sorting algorithm 𝄜 and includes an animated GIF alongside the simple Python script used to generate it.

## Usecase

This repo aim to support the [SquareNet](https://github.com/ArmanddeCacqueray/SquareNet) engine (see link) to illustrate it's core gridification process. Note that to use SquareNet itself, one simply need to
```python
pip install squarenet.
```

Regarding this particular repository:
- Run `main.py` to reproduce the quick visual animated demo
- Feel free to look at what's inside `sort.py` (not optimized, demonstration purpose) to see how the algorithm work.
- See sort_core.cppp for a c++ optimized version, including multi threading, provided for user interested in high performance. It achieve **< 200 ms runtime** (full cartesian sort process) on 1 million 2D points on my old personal laptop (AMD Ryzen 3 3250U / 2 cores, 4 threads).


## Algorithm (2D Version)

*Note: The generalization to higher dimensions is straightforward.*

**Initialization:**
Take $N$ random Euclidean points (in this example, $N = 4225 = 65 \times 65$ points): $(x_k, y_k)_{1 \le k \le N}$.
Randomly split the flat key $k$ into a 2D multi-key to form a grid (purely random initialization): $k \leftrightarrow [i, j]_k$.

**Iterative Sorting Procedure:**
1. Sort the $x$-coordinate along the row key $i$: update $[i, j]_k \leftarrow [i', j]_k$ where $i'$ ensures the $x_k$ coordinates are sorted along the $i$-axis (all columns $j$ are processed in parallel).
2. Sort the $y$-coordinate along the column key $j$: update $[i', j]_k \leftarrow [i', j']_k$.
3. Check if the $x$-sorting was broken by applying the $y$-sorting step (which is highly probable). If so, return to step 1 and repeat until both dimensions are simultaneously satisfied.

## Output

The algorithm produces a **bijective mapping** from the raw points `RP` (shape `[4225, 2]`: $(x_k, y_k)$) to a gridded tensor `GT` (shape `[65, 65, 2]`: $(x_{ij}, y_{ij})$). 

Upon termination, the resulting gridded view `GT` is guaranteed to be monotonic 📈 :
* $x$ strictly increases along $i$ ($\rightarrow$)
* $y$ strictly increases along $j$ ($\uparrow$)

This ensures that the multi-key $[i, j]$ is spatially coherent, meaning local neighborhoods are roughly preserved: the nearest spatial neighbors of a point with multi-key $[i, j]$ will likely have adjacent multi-keys $[i\pm1, j\pm1]$. Though a few outliers will unavoidably land at $[i\pm2, j\pm2]$ or further, this behavior motivates the "robust" and "ultimate" refinements implemented in the full SquareNet gridifier.

By construction, the transformation is a bijection between the raw point key $k$ and the grid multi-key $[i, j]$. This allows for seamless data transfer between the flat point list and the grid using simple fancy indexing operations.

## Proof of Termination & Optimal Transport

A notable aspect of this algorithm is its proof of termination, which is relatively simple and establishes a link to Optimal Transport.

Based on the classical [Rearrangement Inequality](https://en.wikipedia.org/wiki/Rearrangement_inequality), each sorting step strictly decreases the following global structural quantity (average energy) on the `GT`:

$$ \sum_{i,j} \left( (x_{ij} - i)^2 + (y_{ij} - j)^2 \right) $$

This provides a solid monovariant guaranteeing that no cycles will occur, and thus, that the algorithm will mathematically terminate. In practical—and even adversarial—cases, no more than 100 total iterations are typically required.
