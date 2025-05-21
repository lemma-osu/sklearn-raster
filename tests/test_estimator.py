"""Tests for wrapped estimators."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import NotFittedError

from sklearn_raster import wrap
from sklearn_raster.estimator import is_fitted

from .feature_utils import ModelData, parametrize_model_data, unwrap_features


@parametrize_model_data()
@pytest.mark.parametrize("estimator", [KNeighborsRegressor, RandomForestRegressor])
@pytest.mark.parametrize("single_output", [True, False], ids=["single", "multi"])
@pytest.mark.parametrize("squeeze", [True, False], ids=["squeezed", "unsqueezed"])
def test_predict(model_data: ModelData, estimator, single_output, squeeze):
    """Test that predict works with all feature types and a few estimators."""
    X_image, X, y = model_data.set(single_output=single_output, squeeze=squeeze)

    estimator = wrap(estimator()).fit(X, y)

    y_pred = unwrap_features(estimator.predict(X_image))

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
    """Test that predict works with all feature types with unsupervised estimators."""
    X_image, X, _ = model_data

    estimator = wrap(estimator()).fit(X)

    y_pred = unwrap_features(estimator.predict(X_image))

    assert y_pred.ndim == 3
    expected_shape = (1, model_data.n_rows, model_data.n_cols)
    assert_array_equal(y_pred.shape, expected_shape)


@parametrize_model_data(mode="classification")
@pytest.mark.parametrize("estimator", [KNeighborsClassifier, RandomForestClassifier])
@pytest.mark.parametrize("squeeze", [True, False], ids=["squeezed", "unsqueezed"])
def test_predict_proba(model_data: ModelData, estimator, squeeze):
    """Test that predict_proba generates the expected output shape."""
    # Hard-coded in parametrize_model_data
    n_classes = 2

    X_image, X, y = model_data.set(
        single_output=True,
        squeeze=squeeze,
    )

    estimator = wrap(estimator()).fit(X, y)
    y_prob = unwrap_features(estimator.predict_proba(X_image))

    assert y_prob.ndim == 3
    expected_shape = (
        n_classes,
        model_data.n_rows,
        model_data.n_cols,
    )
    assert_array_equal(y_prob.shape, expected_shape)


@parametrize_model_data()
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_predict_with_ndimensions(model_data: ModelData, ndim: int):
    """Test predicting with data of different dimensionality"""
    n_features = model_data.n_features
    # Build an image of shape (n_features, 2, ...) with ndim total dimensions
    img_shape = tuple([n_features] + [2] * (ndim - 1))
    X_image = np.ones(img_shape)
    # Set one NoData pixel to trigger skipping and masking
    X_image[(0,) * ndim] = 0

    # Update the model with the new image
    model_data.set(X_image=X_image)
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor()).fit(X, y)
    y_pred = unwrap_features(estimator.predict(X_image, nodata_input=0))
    assert y_pred.ndim == ndim


@parametrize_model_data()
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"k{k}")
def test_kneighbors_with_distance(model_data: ModelData, k):
    """Test kneighbors works with all feature types when returning distance."""
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor(n_neighbors=k)).fit(X, y)

    dist, nn = estimator.kneighbors(X_image, return_distance=True)
    dist = unwrap_features(dist)
    nn = unwrap_features(nn)

    assert dist.ndim == 3
    assert nn.ndim == 3

    assert_array_equal(dist.shape, (k, model_data.n_rows, model_data.n_cols))
    assert_array_equal(nn.shape, (k, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"k{k}")
def test_kneighbors_without_distance(model_data: ModelData, k):
    """Test kneighbors works with all feature types when NOT returning distance."""
    X_image, X, y = model_data
    estimator = wrap(KNeighborsRegressor(n_neighbors=k)).fit(X, y)

    nn = estimator.kneighbors(X_image, return_distance=False)
    nn = unwrap_features(nn)

    assert nn.ndim == 3

    assert_array_equal(nn.shape, (k, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
@pytest.mark.parametrize("n_neighbors", [1, 5], ids=lambda k: f"n_neighbors={k}")
def test_kneighbors_with_n_neighbors(model_data: ModelData, n_neighbors):
    """Test kneighbors returns n_neighbors when specified."""
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor(n_neighbors=3)).fit(X, y)

    nn = estimator.kneighbors(X_image, n_neighbors=n_neighbors, return_distance=False)
    nn = unwrap_features(nn)

    assert nn.ndim == 3

    assert_array_equal(nn.shape, (n_neighbors, model_data.n_rows, model_data.n_cols))


@parametrize_model_data()
@pytest.mark.parametrize("k", [1, 3], ids=lambda k: f"k{k}")
def test_kneighbors_unsupervised(model_data: ModelData, k):
    """Test kneighbors works with all feature types when unsupervised."""
    X_image, X, y = model_data

    estimator = wrap(NearestNeighbors(n_neighbors=k)).fit(X)

    dist, nn = estimator.kneighbors(X_image, return_distance=True)
    dist = unwrap_features(dist)
    nn = unwrap_features(nn)

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
    unwrap_features(
        estimator.kneighbors(X_image, return_distance=False, custom_kwarg=1)
    )


@parametrize_model_data()
def test_kneighbors_nodata_outputs(model_data: ModelData):
    """Test that kneighbors assigns single or multiple nodata_outputs correctly."""
    shape = model_data.X_image_shape
    # Set the image to be fully masked so that the kneighbors returns arrays filled with
    # the nodata_output values
    model_data.set(X_image=np.full(shape, np.nan))

    X_image, X, y = model_data
    estimator = wrap(KNeighborsRegressor()).fit(X, y)

    # A scalar nodata_output should be assigned to distances and neighbors
    dist, nn = estimator.kneighbors(X_image, return_distance=True, nodata_output=-32768)
    assert np.unique(unwrap_features(dist)) == [-32768]
    assert np.unique(unwrap_features(nn)) == [-32768]

    # Two nodata_outputs should be assigned in order to distances and neighbors
    dist, nn = estimator.kneighbors(
        X_image, return_distance=True, nodata_output=(-32768, 255)
    )
    assert np.unique(unwrap_features(dist)) == [-32768]
    assert np.unique(unwrap_features(nn)) == [255]

    expected_msg = "`nodata_output` must be a scalar when `return_distance` is False"
    with pytest.raises(ValueError, match=expected_msg):
        unwrap_features(
            estimator.kneighbors(
                X_image, return_distance=False, nodata_output=(np.nan, -32768)
            )
        )


@parametrize_model_data(feature_array_types=(xr.DataArray,))
def test_predict_dataarray_with_custom_dim_name(model_data: ModelData):
    """Test that predict works if the feature dimension is not named "variable"."""
    X_image, X, y = model_data

    estimator = wrap(KNeighborsRegressor()).fit(X, y)
    X_image = X_image.rename({"variable": "features"})

    y_pred = unwrap_features(estimator.predict(X_image))
    assert y_pred.ndim == 3
    assert_array_equal(
        y_pred.shape, (model_data.n_targets, model_data.n_rows, model_data.n_cols)
    )


@parametrize_model_data()
def test_roundtrip_transform_preserves_shape(model_data: ModelData):
    """Test forward and inverse PCA transformation produce correct shapes."""
    X_image, X, _ = model_data
    n_components = 2
    transformer = wrap(PCA(n_components=n_components)).fit(X)
    components = transformer.transform(X_image)

    expected_components_shape = (
        n_components,
        model_data.n_rows,
        model_data.n_cols,
    )
    assert_array_equal(unwrap_features(components).shape, expected_components_shape)

    inverted = unwrap_features(transformer.inverse_transform(components))
    assert_array_equal(inverted.shape, unwrap_features(X_image).shape)


# TODO: Remove this test once we have regression tests in place that confirm
# transformations more robustly.
@parametrize_model_data()
def test_roundtrip_transform_values(model_data: ModelData):
    """Test forward and inverse standard scale transformation produce correct values."""
    # Set the features to a constant value so we get exact scaled means
    constant = 42
    model_data.set(X_image=np.full(model_data.X_image_shape, constant))
    model_data.set(X=np.full_like(model_data.X, constant))
    X_image, X, _ = model_data

    transformer = wrap(StandardScaler(with_std=False)).fit(X)
    scaled = transformer.transform(X_image)

    assert unwrap_features(scaled).mean() == 0

    inverted = unwrap_features(transformer.inverse_transform(scaled))
    assert unwrap_features(inverted).mean() == constant


@parametrize_model_data(feature_array_types=(xr.DataArray, xr.Dataset))
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


@parametrize_model_data(feature_array_types=(np.ndarray,))
def test_with_non_1d_data(model_data: ModelData):
    """Test that 1D sample data is correctly handled."""
    _, X, y = model_data

    estimator = KNeighborsRegressor().fit(X, y)
    reference_pred = estimator.predict(X)
    ref_dist, ref_nn = estimator.kneighbors(X)

    wrapped = wrap(clone(estimator)).fit(X, y)
    # Dataframes and derived 1D arrays are in (samples, features) by default, but
    # wrapped estimators require features as the first dimension, hence the need to
    # transpose the input and output.
    check_pred = wrapped.predict(X.T)
    check_dist, check_nn = wrapped.kneighbors(X.T)

    assert_array_equal(reference_pred, check_pred.T)
    assert_array_equal(ref_dist, check_dist.T)
    assert_array_equal(ref_nn, check_nn.T)


@parametrize_model_data(feature_array_types=(xr.DataArray, xr.Dataset))
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


@pytest.mark.filterwarnings("ignore:.*fitted without feature names")
@parametrize_model_data(feature_array_types=(xr.DataArray, xr.Dataset))
@pytest.mark.parametrize(
    "fit_with", [np.ndarray, pd.DataFrame], ids=lambda x: x.__name__
)
def test_transformed_var_names(model_data: ModelData, fit_with):
    """
    Test that variable names are correctly set in a Dataset or DataArray.
    """
    X_image, X, _ = model_data

    # Models fitted without named features should inverse transform to sequential
    # integer names
    if fit_with is np.ndarray:
        expected_inverted_names = list(range(model_data.n_features))
        X = np.asarray(X)
    # Models fitted with feature names should inverse transform back to those names
    elif fit_with is pd.DataFrame:
        expected_inverted_names = list(X.columns)

    estimator = wrap(PCA(n_components=3)).fit(X)
    components = estimator.transform(X_image)
    inverted = estimator.inverse_transform(components)

    if isinstance(X_image, xr.DataArray):
        component_names = components["variable"].values
        inverted_names = inverted["variable"].values
    else:
        component_names = components.data_vars
        inverted_names = inverted.data_vars

    expected_component_names = ["pca0", "pca1", "pca2"]
    assert list(component_names) == expected_component_names
    assert list(inverted_names) == expected_inverted_names


@parametrize_model_data(
    feature_array_types=(xr.DataArray, xr.Dataset), mode="classification"
)
def test_predict_proba_class_names(model_data: ModelData):
    """Test that class names correspond with class values in a Dataset or DataArray."""
    X_image, X, y = model_data.set(single_output=True)
    # Make sure the class values aren't just [0, 1]
    y += 99
    expected_class_names = np.unique(y)

    estimator = wrap(KNeighborsClassifier()).fit(X, y)
    y_prob = estimator.predict_proba(X_image)

    if isinstance(X_image, xr.DataArray):
        var_names = y_prob["class"].values
    else:
        var_names = y_prob.data_vars

    assert list(var_names) == list(expected_class_names)


@parametrize_model_data(
    feature_array_types=(xr.DataArray, xr.Dataset), mode="classification"
)
def test_predict_proba_raises_for_multioutput(model_data: ModelData):
    """Test that an error is raised for multi-output classifiers."""
    X_image, X, y = model_data.set(single_output=False)
    estimator = wrap(KNeighborsClassifier()).fit(X, y)

    expected_msg = "does not currently support multi-output classification"
    with pytest.raises(NotImplementedError, match=expected_msg):
        estimator.predict_proba(X_image)


@pytest.mark.parametrize(
    ("estimator", "method"),
    [
        (RandomForestRegressor, "predict"),
        (KNeighborsRegressor, "kneighbors"),
        (KNeighborsClassifier, "predict_proba"),
        (StandardScaler, "transform"),
        (StandardScaler, "inverse_transform"),
    ],
)
@parametrize_model_data()
def test_raises_if_not_fitted(
    estimator: BaseEstimator, method: str, model_data: ModelData
):
    """Test that wrapped methods raise correctly if the estimator is not fitted."""
    X_image, _, _ = model_data

    with pytest.raises(NotFittedError):
        getattr(wrap(estimator()), method)(X_image)


@parametrize_model_data(feature_array_types=(np.ndarray,))
def test_predict_warns_missing_feature_names(model_data: ModelData):
    """Test that predict warns when feature names are missing."""
    # Retrieve model data with and without feature names
    X_image_unnamed, X_unnamed, y = model_data
    X_image_named, X_named, _ = model_data.set(feature_array_type=xr.DataArray)

    estimator_fit_with_names = wrap(RandomForestRegressor()).fit(X_named, y)
    estimator_fit_without_names = wrap(RandomForestRegressor()).fit(X_unnamed, y)

    with pytest.warns(match="was fitted with feature names"):
        estimator_fit_with_names.predict(X_image_unnamed)

    with pytest.warns(match="was fitted without feature names"):
        estimator_fit_without_names.predict(X_image_named)


@parametrize_model_data(
    X_image=np.random.random((2, 2, 10)), feature_array_types=(xr.DataArray,)
)
def test_predict_raises_mismatched_feature_names(model_data: ModelData):
    """Test that predict raises when feature names are mismatched."""
    # Retrieve model data with and without feature names
    X_image, X, y = model_data

    # Fit the estimator with different names than the features
    rename_map = {k: k + "_different" for k in X.columns}
    X_renamed = X.rename(columns=rename_map)
    estimator = wrap(RandomForestRegressor()).fit(X_renamed, y)

    with pytest.raises(ValueError, match="Feature names unseen at fit time"):
        estimator.predict(X_image)

    # Fit the estimator with same names in a different order than the features
    rename_map = dict(zip(X.columns, X.columns[::-1]))
    X_flipped = X.rename(columns=rename_map)
    estimator = wrap(RandomForestRegressor()).fit(X_flipped, y)

    with pytest.raises(ValueError, match="must be in the same order"):
        estimator.predict(X_image)


@pytest.mark.parametrize(
    ("estimator", "method"),
    [
        (StandardScaler, "predict"),
        (KNeighborsRegressor, "transform"),
        (KNeighborsRegressor, "inverse_transform"),
        (KNeighborsRegressor, "predict_proba"),
        (RandomForestRegressor, "kneighbors"),
    ],
)
def test_unimplemented_methods_raise(estimator: BaseEstimator, method: str):
    """Wrapped estimators should raise for unimplemented methods."""
    expected = f"`{estimator.__name__}` does not implement `{method}`"
    with pytest.raises(NotImplementedError, match=expected):
        getattr(wrap(estimator()), method)()


@pytest.mark.parametrize(
    ("method", "required_attr"),
    [("transform", "get_feature_names_out"), ("predict_proba", "classes_")],
)
def test_missing_required_attrs_raise(method: str, required_attr: str):
    """Wrapped estimators should raise for missing required attrs."""

    class DummyEstimator(BaseEstimator):
        """Dummy estimator that implements methods but is missing attrs."""

        def fit(self, *args, **kwargs):
            self.fitted_ = True
            return self

        def transform(*args, **kwargs): ...
        def predict_proba(*args, **kwargs): ...

    expected = (
        f"`DummyEstimator` is missing a required attribute `{required_attr}` needed to "
        f"implement `{method}`"
    )
    # Fit the estimator to avoid an immediate NotFittedError
    est = wrap(DummyEstimator()).fit(None, None)
    with pytest.raises(NotImplementedError, match=expected):
        getattr(est, method)(None, None)


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
