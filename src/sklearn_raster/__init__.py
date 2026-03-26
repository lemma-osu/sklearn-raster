from importlib.metadata import PackageNotFoundError, version

from .estimator import FeatureArrayEstimator, wrap

try:
    __version__ = version("sklearn-raster")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = [
    "wrap",
    "FeatureArrayEstimator",
]
