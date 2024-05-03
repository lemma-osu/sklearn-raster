import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sknnr_spatial import wrap
from sknnr_spatial.estimator import is_fitted


def test_unimplemented_methods_raise():
    """Wrapped estimators should raise NotImplementedError for unimplemented methods."""
    estimator = wrap(RandomForestRegressor())
    with pytest.raises(NotImplementedError):
        estimator.kneighbors()


def test_wrapping_fitted_estimators_warns(dummy_model_data):
    """Wrapping fitted estimators should raise a warning and reset the estimator."""
    _, X, y = dummy_model_data

    with pytest.warns(match="has already been fit"):
        estimator = wrap(KNeighborsRegressor().fit(X, y))

    assert not is_fitted(estimator._wrapped)


def test_wrapper_is_fitted(dummy_model_data):
    """A wrapper should appear fitted after fitting the wrapped estimator."""
    _, X, y = dummy_model_data

    estimator = wrap(KNeighborsRegressor())
    assert not is_fitted(estimator._wrapped)

    estimator = estimator.fit(X, y)
    assert is_fitted(estimator._wrapped)
    assert is_fitted(estimator)
