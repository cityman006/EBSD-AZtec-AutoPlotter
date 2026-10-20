"""Small NumPy CPU research baselines; no optional ML dependency at import."""
from .baselines import GrainGraphRegressor, RidgeRegressor, graph_design

__all__ = ["GrainGraphRegressor", "RidgeRegressor", "graph_design"]
