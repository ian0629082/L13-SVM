from __future__ import annotations

import numpy as np


MAPPING_METHODS = [
    "z = x1^2 + x2^2",
    "z = exp(-(x1^2 + x2^2))",
]


def simplified_3d_mapping(
    X: np.ndarray,
    method: str,
) -> np.ndarray:
    """Map 2D samples to a simple 3D surface for teaching intuition.

    This is not the true RBF Kernel feature map. It is only a visual metaphor.
    """
    squared_radius = np.sum(X**2, axis=1)
    if method == "z = x1^2 + x2^2":
        # 把離原點較遠的點抬高，常用來說明同心圓資料如何被分開。
        z = squared_radius
    elif method == "z = exp(-(x1^2 + x2^2))":
        # 高斯形狀的示意映射：中心較高、外圈較低。
        z = np.exp(-squared_radius)
    else:
        raise ValueError(f"Unknown mapping method: {method}")
    return np.column_stack([X[:, 0], X[:, 1], z])
