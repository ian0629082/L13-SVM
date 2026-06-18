from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


@dataclass(frozen=True)
class SVMTrainingResult:
    model: SVC
    predictions: np.ndarray
    accuracy: float
    decision_scores: np.ndarray
    support_vectors: np.ndarray


def fit_rbf_svm(
    X: np.ndarray,
    y: np.ndarray,
    c_value: float,
    gamma_value: float,
) -> SVC:
    """Train an RBF Kernel SVM classifier."""
    # RBF kernel 讓 SVM 可在原始 2D 空間形成非線性決策邊界。
    model = SVC(
        kernel="rbf",
        C=c_value,
        gamma=gamma_value,
    )
    model.fit(X, y)
    return model


def train_rbf_svm(
    X: np.ndarray,
    y: np.ndarray,
    c_value: float,
    gamma_value: float,
) -> SVMTrainingResult:
    """Train an RBF SVM and return the model plus teaching outputs."""
    model = fit_rbf_svm(X, y, c_value=c_value, gamma_value=gamma_value)
    predictions = model.predict(X)
    # decision_function 是分類分數，不是 kernel 後的真實高維座標。
    return SVMTrainingResult(
        model=model,
        predictions=predictions,
        accuracy=accuracy_score(y, predictions),
        decision_scores=model.decision_function(X),
        support_vectors=model.support_vectors_,
    )
