from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler


def generate_dataset(
    n_samples: int,
    noise: float,
    factor: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a nonlinear 2D circle dataset for the SVM demo."""
    # 產生同心圓資料，適合展示「線性分類器在 2D 平面不夠用」的情境。
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    # 標準化讓 SVM 參數 C/gamma 的效果更穩定、也更容易觀察。
    X = StandardScaler().fit_transform(X)
    return X.astype(float), y.astype(int)
