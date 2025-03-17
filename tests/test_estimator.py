"""Tests for wrapped estimators."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.utils.validation import NotFittedError

from sklearn_raster import wrap
from sklearn_raster.estimator import is_fitted

from .image_utils import ModelData, parametrize_model_data, unwrap_image


@parametrize_model_data()
@pytest.mark.parametrize("estimator", [KNeighborsRegressor, RandomForestRegressor])
@pytest.mark.parametrize("single_output", [True, False], ids=["single", "multi"])
@pytest.mark.parametrize("squeeze", [True, False], ids=["squeezed", "unsqueezed"])
def test_predict(model_data: ModelData, estimator, single_output, squeeze):
    """Test that predict works with all image types and a few estimators."""
    X_image, X, y = model_data.set(single_output=single_output, squeeze=squeeze)

    estimator = wrap(estimator()).fit(X, y)

    y_pred = unwrap_image(estimator.predict(X_image))

    assert y_pred.ndim == 3
    expected_shape = (
        1 if single_output else model_data.n_targets,
        model_data.n_rows,
        model_data.n_cols,
    )
    assert_array_equal(y_pred.shape, expected_shape)


@parametrize_model_data()
@pytest.mark.parametrize("estimator", [KMeans, MeanShift, AffinityPropagation])
def test_predict_unsupervised(model_data: ModelData, estimator):
    """Test that predict works with all image types with unsupervised estimators."""
    X_image, X, _ = model_data

    estimator = wrap(estimator()).fit(X)

    y_pred = unwrap_image(estimator.predict(X_image))

    assert y_pred.ndim == 3
    expected_shape = (1, model_data.n_rows, model_data.n_cols)
    assert_array_equal(y_pred.shape, expected_shape)


@parametrize_model_data()
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"k{k}")
def test_kneighbors_with_distance(model_data: ModelData, k):
    """Test kneighbors works with all image types when returning distance."""
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor(n_neighbors=k)).fit(X, y)

    dist, nn = estimator.kneighbors(X_image, return_distance=True)
    dist = unwrap_image(dist)
    nn = unwrap_image(nn)

    assert dist.ndim == 3
    assert nn.ndim == 3

    assert_array_equal(dist.shape, (k, model_data.n_rows, model_data.n_cols))
    assert_array_equal(nn.shape, (k, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"k{k}")
def test_kneighbors_without_distance(model_data: ModelData, k):
    """Test kneighbors works with all image types when NOT returning distance."""
    X_image, X, y = model_data
    estimator = wrap(KNeighborsRegressor(n_neighbors=k)).fit(X, y)

    nn = estimator.kneighbors(X_image, return_distance=False)
    nn = unwrap_image(nn)

    assert nn.ndim == 3

    assert_array_equal(nn.shape, (k, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
@pytest.mark.parametrize("n_neighbors", [1, 5], ids=lambda k: f"n_neighbors={k}")
def test_kneighbors_with_n_neighbors(model_data: ModelData, n_neighbors):
    """Test kneighbors returns n_neighbors when specified."""
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor(n_neighbors=3)).fit(X, y)

    nn = estimator.kneighbors(X_image, n_neighbors=n_neighbors, return_distance=False)
    nn = unwrap_image(nn)

    assert nn.ndim == 3

    assert_array_equal(nn.shape, (n_neighbors, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"k{k}")
def test_kneighbors_unsupervised(model_data: ModelData, k):
    """Test kneighbors works with all image types when unsupervised."""
    X_image, X, y = model_data

    estimator = wrap(NearestNeighbors(n_neighbors=k)).fit(X)

    dist, nn = estimator.kneighbors(X_image, return_distance=True)
    dist = unwrap_image(dist)
    nn = unwrap_image(nn)

    assert dist.ndim == 3
    assert nn.ndim == 3

    assert_array_equal(dist.shape, (k, model_data.n_rows, model_data.n_cols))
    assert_array_equal(nn.shape, (k, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
def test_kneighbors_with_custom_kwarg(model_data: ModelData):
    """Test that kneighbors passes custom kwargs."""

    class CustomEstimator(KNeighborsRegressor):
        def kneighbors(self, X, custom_kwarg, **kwargs):
            assert custom_kwarg is not None
            return super().kneighbors(X, **kwargs)

    X_image, X, y = model_data
    estimator = wrap(CustomEstimator()).fit(X, y)
    unwrap_image(estimator.kneighbors(X_image, return_distance=False, custom_kwarg=1))


@parametrize_model_data(image_types=(xr.DataArray,))
def test_predict_dataarray_with_custom_dim_name(model_data: ModelData):
    """Test that predict works if the band dimension is not named "variable"."""
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor()).fit(X, y)
    X_image = X_image.rename({"variable": "band"})

    y_pred = unwrap_image(estimator.predict(X_image))
    assert y_pred.ndim == 3
    assert_array_equal(
        y_pred.shape, (model_data.n_targets, model_data.n_rows, model_data.n_cols)
    )


@parametrize_model_data(image_types=(xr.DataArray, xr.Dataset))
@pytest.mark.parametrize("crs", ["EPSG:5070", None])
def test_crs_preserved(model_data: ModelData, crs):
    """Test that the original image CRS is preserved."""
    # rioxarray must be imported to register the rio accessor
    import rioxarray  # noqa: F401

    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor()).fit(X, y)

    if crs:
        X_image = X_image.rio.write_crs(crs)

    y_pred = estimator.predict(X_image)
    dist, nn = estimator.kneighbors(X_image, return_distance=True)

    assert y_pred.rio.crs == crs
    assert dist.rio.crs == crs
    assert nn.rio.crs == crs


@parametrize_model_data(image_types=(np.ndarray,))
def test_with_non_image_data(model_data: ModelData):
    """Test that 1D sample data is correctly handled."""
    _, X, y = model_data

    estimator = KNeighborsRegressor().fit(X, y)
    reference_pred = estimator.predict(X)
    ref_dist, ref_nn = estimator.kneighbors(X)

    wrapped = wrap(clone(estimator)).fit(X, y)
    # 1D arrays must be in (features, samples) shape to match the expected
    # input
    check_pred = wrapped.predict(X.T)
    check_dist, check_nn = wrapped.kneighbors(X.T)

    assert_array_equal(reference_pred, check_pred.T)
    assert_array_equal(ref_dist, check_dist.T)
    assert_array_equal(ref_nn, check_nn.T)


@parametrize_model_data(image_types=(xr.DataArray, xr.Dataset))
@pytest.mark.parametrize(
    "fit_with", [np.ndarray, pd.DataFrame, pd.Series], ids=lambda x: x.__name__
)
def test_predicted_var_names(model_data: ModelData, fit_with):
    """Test that variable names are correctly set in a Dataset or DataArray."""
    X_image, X, y = model_data

    # Models fitted without named targets should predict sequential integer names
    if fit_with is np.ndarray:
        expected_var_names = [0, 1, 2]
        y = np.asarray(y)
    # Models fitted with multiple target names should predict those names
    elif fit_with is pd.DataFrame:
        expected_var_names = ["t0", "t1", "t2"]
    # Models fitted with a single named series should predict that name
    elif fit_with is pd.Series:
        expected_var_names = ["t0"]
        y = y["t0"]

    estimator = wrap(KNeighborsRegressor()).fit(X, y)
    y_pred = estimator.predict(X_image)

    if isinstance(X_image, xr.DataArray):
        var_names = y_pred["variable"].values
    else:
        var_names = y_pred.data_vars

    assert list(var_names) == expected_var_names


@parametrize_model_data()
def test_raises_if_not_fitted(model_data: ModelData):
    """Test that wrapped methods raise correctly if the estimator is not fitted."""
    X_image, _, _ = model_data
    estimator = KNeighborsRegressor()
    wrapped = wrap(estimator)

    with pytest.raises(NotFittedError):
        wrapped.predict(X_image)

    with pytest.raises(NotFittedError):
        wrapped.kneighbors(X_image)


@parametrize_model_data(image_types=(np.ndarray,))
def test_predict_warns_missing_feature_names(model_data: ModelData):
    """Test that predict warns when feature names are missing."""
    # Retrieve model data with and without feature names
    X_image_unnamed, X_unnamed, y = model_data
    X_image_named, X_named, _ = model_data.set(image_type=xr.DataArray)

    estimator_fit_with_names = wrap(RandomForestRegressor()).fit(X_named, y)
    estimator_fit_without_names = wrap(RandomForestRegressor()).fit(X_unnamed, y)

    with pytest.warns(match="was fitted with feature names"):
        estimator_fit_with_names.predict(X_image_unnamed)

    with pytest.warns(match="was fitted without feature names"):
        estimator_fit_without_names.predict(X_image_named)


@parametrize_model_data(
    X_image=np.random.random((2, 2, 10)), image_types=(xr.DataArray,)
)
def test_predict_raises_mismatched_feature_names(model_data: ModelData):
    """Test that predict raises when feature names are mismatched."""
    # Retrieve model data with and without feature names
    X_image, X, y = model_data

    # Fit the estimator with different names than the image
    rename_map = {k: k + "_different" for k in X.columns}
    X_renamed = X.rename(columns=rename_map)
    estimator = wrap(RandomForestRegressor()).fit(X_renamed, y)

    with pytest.raises(ValueError, match="Band names unseen at fit time"):
        estimator.predict(X_image)

    # Fit the estimator with same names in a different order than the image
    rename_map = dict(zip(X.columns, X.columns[::-1]))
    X_flipped = X.rename(columns=rename_map)
    estimator = wrap(RandomForestRegressor()).fit(X_flipped, y)

    with pytest.raises(ValueError, match="must be in the same order"):
        estimator.predict(X_image)


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
