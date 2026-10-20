"""Actual fitted ridge and trainable one-hop mean-aggregation GNN regressors.

The graph model learns both the message embedding and prediction head. It is a
small GraphSAGE-like CPU baseline, not a validated constitutive material law.
"""
from __future__ import annotations

import numpy as np


def graph_design(x, edge_index=None, *, edge_weights=None):
    """Concatenate node features and weighted mean neighbour messages, O(N+E)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Node features must be a finite N-by-F matrix")
    neighbours = np.zeros_like(x)
    counts = np.zeros(len(x))
    if edge_index is not None:
        edges = np.asarray(edge_index)
        if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("edge_index must be an integer 2-by-E array")
        if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
            raise ValueError("Graph edge index outside node array")
        weights = np.ones(edges.shape[1]) if edge_weights is None else np.asarray(edge_weights, dtype=float)
        if weights.shape != (edges.shape[1],) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Edge weights must be finite positive values")
        np.add.at(neighbours, edges[1], x[edges[0]] * weights[:, None])
        np.add.at(counts, edges[1], weights)
    # Physical boundary lengths can total less than one micrometre. Only
    # isolated nodes need a fallback divisor; all positive totals normalize.
    neighbours /= np.where(counts > 0, counts, 1)[:, None]
    return np.column_stack([x, neighbours])


class RidgeRegressor:
    def __init__(self, alpha=1e-3):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge regularization must be positive")
        self.alpha = alpha

    def fit(self, x, y, mask):
        design = np.column_stack([x, np.ones(len(x))])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[-1, -1] = 0
        self.coef = np.zeros((design.shape[1], y.shape[1]))
        for column in range(y.shape[1]):
            valid = mask[:, column]
            if not valid.any():
                raise ValueError("Every target needs at least one training label")
            a, b = design[valid], y[valid, column]
            self.coef[:, column] = np.linalg.solve(a.T @ a + penalty, a.T @ b)
        return self

    def predict(self, x):
        return np.column_stack([x, np.ones(len(x))]) @ self.coef

    def state(self):
        return {"coef": self.coef}

    def restore(self, state):
        self.coef = state["coef"]
        return self


class GrainGraphRegressor:
    """A learned mean-neighbour neural regressor with full-batch Adam updates."""
    def __init__(self, input_size, targets, *, hidden=24, seed=0):
        if min(input_size, targets, hidden) < 1:
            raise ValueError("Network dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.params = {"w1": rng.normal(0, 1 / np.sqrt(input_size), (input_size, hidden)),
                       "b1": np.zeros(hidden), "w2": rng.normal(0, .1, (hidden, targets)),
                       "b2": np.zeros(targets)}

    def predict(self, x):
        p = self.params
        return np.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]

    def fit(self, x, y, mask, *, epochs=200, learning_rate=.01,
            validation=None, callback=None):
        p = self.params
        first = {key: np.zeros_like(value) for key, value in p.items()}
        second = {key: np.zeros_like(value) for key, value in p.items()}
        best_loss, best, history = np.inf, None, []
        if not mask.any():
            raise ValueError("Training requires actual nonmasked labels")
        for epoch in range(1, epochs + 1):
            h = np.tanh(x @ p["w1"] + p["b1"])
            residual = np.where(mask, h @ p["w2"] + p["b2"] - y, 0)
            derivative = 2 * residual / mask.sum()
            hidden_derivative = (derivative @ p["w2"].T) * (1 - h * h)
            grads = {"w2": h.T @ derivative, "b2": derivative.sum(axis=0),
                     "w1": x.T @ hidden_derivative, "b1": hidden_derivative.sum(axis=0)}
            for key in p:
                gradient = np.clip(grads[key], -10, 10)
                first[key] = .9 * first[key] + .1 * gradient
                second[key] = .999 * second[key] + .001 * gradient ** 2
                p[key] -= learning_rate * (first[key] / (1 - .9 ** epoch)) / (
                    np.sqrt(second[key] / (1 - .999 ** epoch)) + 1e-8)
            loss = float(np.mean(residual[mask] ** 2))
            if validation is None:
                val_loss = loss
            else:
                vx, vy, vm = validation
                val_loss = float(np.mean((self.predict(vx) - vy)[vm] ** 2))
            if val_loss < best_loss:
                best_loss, best = val_loss, {key: value.copy() for key, value in p.items()}
            history.append({"epoch": epoch, "train_mse_scaled": loss, "validation_mse_scaled": val_loss})
            if callback is not None:
                callback(epoch, history[-1])
        self.params = best
        return history

    def state(self):
        return dict(self.params)

    def restore(self, state):
        self.params = dict(state)
        return self
