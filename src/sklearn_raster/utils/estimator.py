import warnings
from functools import wraps
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError, check_is_fitted
from typing_extensions import Concatenate

from ..types import RT, P
from .wrapper import GenericWrapper


def is_fitted(estimator: BaseEstimator) -> bool:
    """Return whether an estimator is fitted or not."""
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False


def suppress_feature_name_warnings(func):
    """Suppress warnings related to missing feature names in a wrapped function."""
    msg = "X does not have valid feature names"

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg)
            return func(*args, **kwargs)

    return wrapper


def requires_fitted(
    func: Callable[Concatenate[GenericWrapper, P], RT],
) -> Callable[Concatenate[GenericWrapper, P], RT]:
    """Decorator to check if an estimator is fitted before calling a method."""

    @wraps(func)
    def wrapper(self: GenericWrapper, *args, **kwargs):
        check_is_fitted(self)
        return func(self, *args, **kwargs)

    return wrapper


def generate_sequential_names(n: int, prefix: str) -> list[str]:
    """Generate a list of `n` prefixed sequential names."""
    return [f"{prefix}{i}" for i in range(n)]
