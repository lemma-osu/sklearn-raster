from importlib.metadata import version

from .estimator import FeatureArrayEstimator, wrap

__version__ = version("sklearn-raster")


__all__ = [
    "wrap",
    "FeatureArrayEstimator",
]
