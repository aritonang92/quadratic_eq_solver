"""
The black box: a feed-forward Artificial Neural Network (multi-layer perceptron).

FEYNMAN NOTE
------------
Simple idea:  Picture a committee in layers. The first layer of members each looks
              at all 18 facts about a customer, scores them with their own private
              weights and shouts a number if it is positive (ReLU). The next layer
              listens to those shouts and does the same. The last member squashes
              everything into a probability between 0 and 1: "chance this
              customer leaves".
Why opaque:   With 64 + 32 members and thousands of weights, no single weight
              "means" anything. The knowledge is smeared across all of them -
              that is the black box we will open from the outside (perturbation
              methods) and from the inside (gradients).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier

from .data import Dataset


@dataclass
class AttritionANN:
    """Wraps preprocessing + MLP so every explainer can speak in *raw* features."""

    preprocessor: object
    mlp: MLPClassifier
    threshold: float = 0.5

    # ---- the only function perturbation-based explainers need ----------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(customer leaves) for a DataFrame of raw features."""
        return self.mlp.predict_proba(self.preprocessor.transform(X))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ---- white-box access used by Integrated Gradients ------------------------
    def logit_and_grad(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward + backward pass on *encoded* inputs Z (n, d).

        Returns the pre-sigmoid output (logit) and d logit / d Z. We do the
        back-propagation by hand so the framework has no deep-learning dependency
        and the math is visible: chain rule, layer by layer, ReLU passes gradient
        only where the unit was "on".
        """
        acts, masks = [Z], []
        h = Z
        W, B = self.mlp.coefs_, self.mlp.intercepts_
        for w, b in zip(W[:-1], B[:-1]):
            pre = h @ w + b
            masks.append(pre > 0)
            h = np.maximum(pre, 0)
            acts.append(h)
        logit = (h @ W[-1] + B[-1]).ravel()

        grad = np.repeat(W[-1].T, len(Z), axis=0)            # d logit / d h_last
        for w, m in zip(reversed(W[:-1]), reversed(masks)):
            grad = (grad * m) @ w.T                            # through ReLU, then weights
        return logit, grad


def train_ann(data: Dataset, hidden=(64, 32), seed: int = 7) -> AttritionANN:
    Z = data.preprocessor.transform(data.X_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        alpha=1e-3,                 # L2 weight decay: discourages wild weights
        learning_rate_init=1e-3,
        batch_size=128,
        early_stopping=True,        # stop when a held-out slice stops improving
        validation_fraction=0.15,
        n_iter_no_change=15,
        max_iter=400,
        random_state=seed,
    )
    mlp.fit(Z, data.y_train)
    return AttritionANN(data.preprocessor, mlp)


def evaluate(model: AttritionANN, data: Dataset) -> dict:
    p = model.predict_proba(data.X_test)
    yhat = (p >= model.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(data.y_test, yhat).ravel()
    return {
        "roc_auc": roc_auc_score(data.y_test, p),
        "pr_auc": average_precision_score(data.y_test, p),
        "f1": f1_score(data.y_test, yhat),
        "balanced_accuracy": balanced_accuracy_score(data.y_test, yhat),
        "recall_attrition": tp / (tp + fn),
        "precision_attrition": tp / (tp + fp) if (tp + fp) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(data.y_test)),
    }
