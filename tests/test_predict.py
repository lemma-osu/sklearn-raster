import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sknnr_spatial import wrap

from .image_utils import (
    parametrize_image_types,
    parametrize_xarray_image_types,
    unwrap_image,
    wrap_image,
)


@parametrize_image_types
@pytest.mark.parametrize("estimator", [KNeighborsRegressor, RandomForestRegressor])
def test_predict(dummy_model_data, image_type, estimator):
    """Test that predict works with all image types and a few estimators."""
    X_image, X, y = dummy_model_data
    estimator = wrap(estimator()).fit(X, y)

    X_wrapped = wrap_image(X_image, type=image_type.cls)
    y_pred = unwrap_image(estimator.predict(X_wrapped))

    assert y_pred.ndim == 3
    assert_array_equal(y_pred.shape, (X_image.shape[0], X_image.shape[1], y.shape[-1]))


@parametrize_image_types
def test_kneighbors_with_distance(dummy_model_data, image_type):
    """Test kneighbors works with all image types when returning distance."""
    k = 3
    X_image, X, y = dummy_model_data
    estimator = wrap(KNeighborsRegressor(n_neighbors=k)).fit(X, y)

    X_wrapped = wrap_image(X_image, type=image_type.cls)
    dist, nn = estimator.kneighbors(X_wrapped, return_distance=True)
    dist = unwrap_image(dist)
    nn = unwrap_image(nn)

    assert dist.ndim == 3
    assert nn.ndim == 3

    assert_array_equal(dist.shape, (X_image.shape[0], X_image.shape[1], k))
    assert_array_equal(nn.shape, (X_image.shape[0], X_image.shape[1], k))


@parametrize_image_types
def test_kneighbors_without_distance(dummy_model_data, image_type):
    """Test kneighbors works with all image types when NOT returning distance."""
    k = 3
    X_image, X, y = dummy_model_data
    estimator = wrap(KNeighborsRegressor(n_neighbors=k)).fit(X, y)

    X_wrapped = wrap_image(X_image, type=image_type.cls)
    nn = estimator.kneighbors(X_wrapped, return_distance=False)
    nn = unwrap_image(nn)

    assert nn.ndim == 3

    assert_array_equal(nn.shape, (X_image.shape[0], X_image.shape[1], k))


def test_predict_dataarray_with_custom_dim_name(dummy_model_data):
    """Test that predict works if the band dimension is not named "variable"."""
    X_image, X, y = dummy_model_data
    estimator = wrap(KNeighborsRegressor()).fit(X, y)
    X_wrapped = wrap_image(X_image, type=xr.DataArray).rename({"variable": "band"})

    y_pred = unwrap_image(estimator.predict(X_wrapped))
    assert y_pred.ndim == 3
    assert_array_equal(y_pred.shape, (X_image.shape[0], X_image.shape[1], y.shape[-1]))


@parametrize_xarray_image_types
@pytest.mark.parametrize("crs", ["EPSG:5070", None])
def test_crs_preserved(dummy_model_data, image_type, crs):
    """Test that the original image CRS is preserved."""
    X_image, X, y = dummy_model_data
    estimator = wrap(KNeighborsRegressor()).fit(X, y)
    X_wrapped = wrap_image(X_image, type=image_type.cls)

    if crs:
        X_wrapped = X_wrapped.rio.write_crs(crs)

    y_pred = estimator.predict(X_wrapped)
    dist, nn = estimator.kneighbors(X_wrapped, return_distance=True)

    assert y_pred.rio.crs == crs
    assert dist.rio.crs == crs
    assert nn.rio.crs == crs


def test_predict_non_image_data(dummy_model_data):
    """Test that predicting with non-image data falls back to the wrapped estimator."""
    _, X, y = dummy_model_data
    estimator = KNeighborsRegressor().fit(X, y)
    reference_pred = estimator.predict(X)

    check_pred = wrap(clone(estimator)).fit(X, y).predict(X)
    assert_array_equal(reference_pred, check_pred)


def test_kneighbors_non_image_data(dummy_model_data):
    """Test that kneighbors with non-image data falls back to the wrapped estimator."""
    _, X, y = dummy_model_data
    estimator = KNeighborsRegressor().fit(X, y)
    ref_dist, ref_nn = estimator.kneighbors(X)

    check_dist, check_nn = wrap(clone(estimator)).fit(X, y).kneighbors(X)
    assert_array_equal(ref_dist, check_dist)
    assert_array_equal(ref_nn, check_nn)
