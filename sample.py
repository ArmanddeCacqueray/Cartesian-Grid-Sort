import numpy as np

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