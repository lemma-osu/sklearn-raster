from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from warnings import warn

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import _get_feature_names
from typing_extensions import Literal, overload

from .features import FeatureArray
from .types import EstimatorType
from .utils.estimator import (
    generate_sequential_names,
    is_fitted,
    requires_fitted,
    suppress_feature_name_warnings,
)
from .utils.wrapper import AttrWrapper, requires_attributes, requires_implementation

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from .types import FeatureArrayType, MaybeTuple, NoDataType

ESTIMATOR_OUTPUT_DTYPES: dict[str, np.dtype] = {
    "classifier": np.int32,
    "clusterer": np.int32,
    "regressor": np.float64,
}


@dataclass
class FittedMetadata:
    """Metadata from a fitted estimator."""

    n_targets: int
    n_features: int
    target_names: list[str]
    feature_names: list[str]


class FeatureArrayEstimator(AttrWrapper[EstimatorType]):
    """
    An estimator wrapper with overriden methods for n-dimensional feature arrays.

    Parameters
    ----------
    wrapped : BaseEstimator
        An sklearn-compatible estimator. Supported methods will be overriden to work
        with n-dimensional feature arrays. If the estimator is already fit, it will be
        reset and a warning will be raised.
    """

    _wrapped: EstimatorType
    _wrapped_meta: FittedMetadata

    def __init__(self, wrapped: EstimatorType):
        super().__init__(self._reset_estimator(wrapped))

    @staticmethod
    def _reset_estimator(estimator: EstimatorType) -> EstimatorType:
        """Take an estimator and reset and warn if it was previously fitted."""
        if is_fitted(estimator):
            warn(
                "Wrapping estimator that has already been fit. The estimator must be "
                "fit again after wrapping.",
                stacklevel=2,
            )
            return clone(estimator)

        return estimator

    def _get_n_targets(self, y: NDArray | pd.DataFrame | pd.Series | None) -> int:
        """Get the number of targets used to fit the estimator."""
        # Unsupervised and single-output estimators should both return a single target
        if y is None or y.ndim == 1:
            return 1

        return y.shape[-1]

    def _get_target_names(self, y: NDArray | pd.DataFrame | pd.Series) -> list[str]:
        """Get the target names used to fit the estimator, if available."""
        # Dataframe
        if hasattr(y, "columns"):
            return list(y.columns)

        # Series
        if hasattr(y, "name"):
            return [y.name]

        # Default to sequential identifiers
        return generate_sequential_names(self._get_n_targets(y), "target")

    @requires_implementation
    def fit(self, X, y=None, **kwargs) -> FeatureArrayEstimator[EstimatorType]:
        """
        Fit an estimator from a training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression). Single-output targets of shape (n_samples, 1) will be squeezed
            to shape (n_samples,) to allow consistent prediction across all estimators.
        **kwargs : dict
            Additional keyword arguments passed to the estimator's `fit` method, e.g.
            `sample_weight`.

        Returns
        -------
        self : FeatureArrayEstimator
            The wrapper around the fitted estimator.
        """
        if y is not None:
            # Squeeze extra y dimensions. This will convert from shape (n_samples, 1)
            # which causes inconsistent output shapes with different sklearn estimators,
            # to (n_samples,), which has a consistent output shape.
            y = y.squeeze()
        self._wrapped = self._wrapped.fit(X, y, **kwargs)
        fitted_feature_names = _get_feature_names(X)

        self._wrapped_meta = FittedMetadata(
            n_targets=self._get_n_targets(y),
            n_features=X.shape[-1],
            target_names=self._get_target_names(y),
            feature_names=list(fitted_feature_names)
            if fitted_feature_names is not None
            else [],
        )

        return self

    @requires_implementation
    @requires_fitted
    def predict(
        self,
        X: FeatureArrayType,
        *,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: float | int = np.nan,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **predict_kwargs,
    ) -> FeatureArrayType:
        """
        Predict target(s) for n-dimensional X features.

        Parameters
        ----------
        X : Numpy or Xarray features
            The n-dimensional input features. Array types should be in the shape
            (features, ...) while xr.Dataset should include features as variables.
            Features should correspond with those used to fit the estimator.
        skip_nodata : bool, default=True
            If True, NoData and NaN values will be skipped during prediction. This
            speeds up processing of partially masked arrays, but may be incompatible if
            estimators expect a consistent number of input samples.
        nodata_input : float or sequence of floats, optional
            NoData values other than NaN to mask in the output array. A single value
            will be broadcast to all features while sequences of values will be assigned
            feature-wise. If None, values will be inferred if possible based on
            available metadata.
        nodata_output : float or int, default np.nan
            NoData samples in the input features will be replaced with this value in the
            output targets. If the value does not fit the array dtype returned by the
            estimator, an error will be raised unless `allow_cast` is True.
        ensure_min_samples : int, default 1
            The minimum number of samples that should be passed to `predict`. If the
            array is fully masked and `skip_nodata=True`, dummy values (0) will be
            inserted to ensure this number of samples. The minimum supported number of
            samples depends on the estimator used. No effect if the array contains
            enough unmasked samples or if `skip_nodata=False`.
        allow_cast : bool, default=False
            If True and the estimator output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype instead
            of raising an error.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not np.nan, a warning will be raised if the
            selected `nodata_output` value is returned by the estimator, as this may
            indicate a valid sample being masked.
        keep_attrs : bool, default=False
            If True and the input is an Xarray object, the output will keep all
            attributes of the input features, unless they're set by the estimator (e.g.
            `_FillValue` or `long_name`). Note that some attributes (e.g.
            `scale_factor`) may become inaccurate, which is why they are dropped by
            default. The `history` attribute will always be kept. No effect if the
            input is a Numpy array.
        **predict_kwargs
            Additional arguments passed to the estimator's `predict` method.

        Returns
        -------
        Numpy or Xarray features
            The predicted values. Array types will be in the shape (targets, ...) while
            xr.Dataset will store targets as variables.
        """
        wrapped_func = self._wrapped.predict
        output_dim_name = "target"
        features = FeatureArray.from_feature_array(X, nodata_input=nodata_input)

        self._check_feature_names(features.feature_names)

        # Any estimator with an undefined type should fall back to floating
        # point for safety.
        estimator_type = getattr(self._wrapped, "_estimator_type", "")
        output_dtype = ESTIMATOR_OUTPUT_DTYPES.get(estimator_type, np.float64)

        return features.apply_ufunc_across_features(
            suppress_feature_name_warnings(wrapped_func),
            output_dims=[[output_dim_name]],
            output_dtypes=[output_dtype],
            output_sizes={output_dim_name: self._wrapped_meta.n_targets},
            output_coords={output_dim_name: list(self._wrapped_meta.target_names)},
            skip_nodata=skip_nodata,
            nodata_output=nodata_output,
            ensure_min_samples=ensure_min_samples,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            nan_fill=0.0,
            keep_attrs=keep_attrs,
            **predict_kwargs,
        )

    @requires_implementation
    @requires_fitted
    @requires_attributes("classes_")
    def predict_proba(
        self,
        X: FeatureArrayType,
        *,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: float | int = np.nan,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **predict_proba_kwargs,
    ) -> FeatureArrayType:
        """
        Predict class probabilities for n-dimensional X features.

        Parameters
        ----------
        X : Numpy or Xarray features
            The n-dimensional input features. Array types should be in the shape
            (features, ...) while xr.Dataset should include features as variables.
            Features should correspond with those used to fit the estimator.
        skip_nodata : bool, default=True
            If True, NoData and NaN values will be skipped during prediction. This
            speeds up processing of partially masked arrays, but may be incompatible if
            estimators expect a consistent number of input samples.
        nodata_input : float or sequence of floats, optional
            NoData values other than NaN to mask in the output array. A single value
            will be broadcast to all features while sequences of values will be assigned
            feature-wise. If None, values will be inferred if possible based on
            available metadata.
        nodata_output : float or int, default np.nan
            NoData samples in the input features will be replaced with this value in the
            output targets. If the value does not fit the array dtype returned by the
            estimator, an error will be raised unless `allow_cast` is True.
        ensure_min_samples : int, default 1
            The minimum number of samples that should be passed to `predict`. If the
            array is fully masked and `skip_nodata=True`, dummy values (0) will be
            inserted to ensure this number of samples. The minimum supported number of
            samples depends on the estimator used. No effect if the array contains
            enough unmasked samples or if `skip_nodata=False`.
        allow_cast : bool, default=False
            If True and the estimator output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype instead
            of raising an error.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not np.nan, a warning will be raised if the
            selected `nodata_output` value is returned by the estimator, as this may
            indicate a valid sample being masked.
        keep_attrs : bool, default=False
            If True and the input is an Xarray object, the output will keep all
            attributes of the input features, unless they're set by the estimator (e.g.
            `_FillValue` or `long_name`). Note that some attributes (e.g.
            `scale_factor`) may become inaccurate, which is why they are dropped by
            default. The `history` attribute will always be kept. No effect if the
            input is a Numpy array.
        **predict_proba_kwargs
            Additional arguments passed to the estimator's `predict_proba` method.

        Returns
        -------
        Numpy or Xarray features
            The predicted class probabilities. Array types will be in the shape
            (classes, ...) while xr.Dataset will store classes as variables.
        """
        wrapped_func = self._wrapped.predict_proba
        output_dim_name = "class"
        features = FeatureArray.from_feature_array(X, nodata_input=nodata_input)

        self._check_feature_names(features.feature_names)

        if self._wrapped_meta.n_targets > 1:
            msg = (
                "`predict_proba` does not currently support multi-output "
                "classification."
            )
            raise NotImplementedError(msg)

        return features.apply_ufunc_across_features(
            suppress_feature_name_warnings(wrapped_func),
            output_dims=[[output_dim_name]],
            output_dtypes=[np.float64],
            output_sizes={output_dim_name: len(self._wrapped.classes_)},
            output_coords={output_dim_name: list(self._wrapped.classes_)},
            skip_nodata=skip_nodata,
            nodata_output=nodata_output,
            ensure_min_samples=ensure_min_samples,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            nan_fill=0.0,
            keep_attrs=keep_attrs,
            **predict_proba_kwargs,
        )

    @requires_implementation
    @requires_fitted
    @overload
    def kneighbors(
        self,
        X: FeatureArrayType,
        *,
        n_neighbors: int | None = None,
        return_distance: Literal[True] = True,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: MaybeTuple[float | int] | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **kneighbors_kwargs,
    ) -> tuple[FeatureArrayType, FeatureArrayType]: ...

    @requires_implementation
    @requires_fitted
    @overload
    def kneighbors(
        self,
        X: FeatureArrayType,
        *,
        n_neighbors: int | None = None,
        return_distance: Literal[False] = False,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: float | int | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **kneighbors_kwargs,
    ) -> FeatureArrayType: ...

    @requires_implementation
    @requires_fitted
    def kneighbors(
        self,
        X: FeatureArrayType,
        *,
        n_neighbors: int | None = None,
        return_distance: bool = True,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: MaybeTuple[float | int] | None = None,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **kneighbors_kwargs,
    ) -> FeatureArrayType | tuple[FeatureArrayType, FeatureArrayType]:
        """
        Find the K-neighbors of each sample in a feature array.

        Returns indices of and distances to the neighbors for each pixel.

        Parameters
        ----------
        X : Numpy or Xarray features
            The n-dimensional input features. Array types should be in the shape
            (features, ...) while xr.Dataset should include features as variables.
            Features should correspond with those used to fit the estimator.
        n_neighbors : int, optional
            Number of neighbors required for each sample. The default is the value
            passed to the wrapped estimator's constructor.
        return_distance : bool, default=True
            If True, return distances to the neighbors of each sample. If False, return
            indices only.
        skip_nodata : bool, default=True
            If True, NoData and NaN values will be skipped during prediction. This
            speeds up processing of partially masked features, but may be incompatible
            if estimators expect a consistent number of input samples.
        nodata_input : float or sequence of floats, optional
            NoData values other than NaN to mask in the output features. A single value
            will be broadcast to all features while sequences of values will be assigned
            feature-wise. If None, values will be inferred if possible based on
            available metadata.
        nodata_output : float or int or tuple, optional
            NoData samples in the input features will be replaced with this value in the
            output targets. If the value does not fit the array dtype returned by the
            estimator, an error will be raised unless `allow_cast` is True. If
            `return_distance` is True, you can provide a tuple of two values to use
            for distances and indexes, respectively. Defaults to np.nan for the distance
            array and -2147483648 for the neighbor array.
        ensure_min_samples : int, default 1
            The minimum number of samples that should be passed to `kneighbors`. If the
            array is fully masked and `skip_nodata=True`, dummy values (0) will be
            inserted to ensure this number of samples. The minimum supported number of
            samples depends on the estimator used. No effect if the array contains
            enough unmasked samples or if `skip_nodata=False`.
        allow_cast : bool, default=False
            If True and the estimator output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype instead
            of raising an error.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not np.nan, a warning will be raised if the
            selected `nodata_output` value is returned by the estimator, as this may
            indicate a valid sample being masked.
        keep_attrs : bool, default=False
            If True and the input is an Xarray object, the output will keep all
            attributes of the input features, unless they're set by the estimator (e.g.
            `_FillValue` or `long_name`). Note that some attributes (e.g.
            `scale_factor`) may become inaccurate, which is why they are dropped by
            default. The `history` attribute will always be kept. No effect if the
            input is a Numpy array.
        **kneighbors_kwargs
            Additional arguments passed to the estimator's `kneighbors` method.

        Returns
        -------
        neigh_dist : Numpy or Xarray features
            Array representing the lengths to neighbors, present if
            return_distance=True. Array types will be in the shape (neighbor, ...) while
            xr.Dataset will store neighbors as variables.
        neigh_ind : Numpy or Xarray features
            Array representing the nearest neighbor indices in the population matrix.
            Array types will be in the shape (neighbor, ...) while xr.Dataset will store
            neighbors as variables.
        """
        wrapped_func = self._wrapped.kneighbors
        output_dim_name = "neighbor"

        if nodata_output is None:
            nodata_output = (np.nan, -2147483648) if return_distance else -2147483648
        elif return_distance is False and isinstance(nodata_output, (tuple, list)):
            msg = "`nodata_output` must be a scalar when `return_distance` is False."
            raise ValueError(msg)

        features = FeatureArray.from_feature_array(X, nodata_input=nodata_input)
        k = n_neighbors or cast(int, getattr(self._wrapped, "n_neighbors", 5))

        self._check_feature_names(features.feature_names)

        return features.apply_ufunc_across_features(
            suppress_feature_name_warnings(wrapped_func),
            output_dims=[[output_dim_name], [output_dim_name]]
            if return_distance
            else [[output_dim_name]],
            output_dtypes=[float, int] if return_distance else [int],
            output_sizes={output_dim_name: k},
            output_coords={
                output_dim_name: generate_sequential_names(k, output_dim_name)
            },
            n_neighbors=k,
            return_distance=return_distance,
            skip_nodata=skip_nodata,
            nodata_output=nodata_output,
            ensure_min_samples=ensure_min_samples,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            nan_fill=0.0,
            keep_attrs=keep_attrs,
            **kneighbors_kwargs,
        )

    @requires_implementation
    @requires_fitted
    @requires_attributes("get_feature_names_out")
    def transform(
        self,
        X: FeatureArrayType,
        *,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: float | int = np.nan,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **transform_kwargs,
    ) -> FeatureArrayType:
        """
        Apply the transformation to n-dimensional X features.

        Parameters
        ----------
        X : Numpy or Xarray features
            The n-dimensional input features. Array types should be in the shape
            (features, ...) while xr.Dataset should include features as variables.
            Features should correspond with those used to fit the estimator.
        skip_nodata : bool, default=True
            If True, NoData and NaN values will be skipped during prediction. This
            speeds up processing of partially masked features, but may be incompatible
            if estimators expect a consistent number of input samples.
        nodata_input : float or sequence of floats, optional
            NoData values other than NaN to mask in the output features. A single value
            will be broadcast to all features while sequences of values will be assigned
            feature-wise. If None, values will be inferred if possible based on
            available metadata.
        nodata_output : float or int or tuple, optional
            NoData samples in the input features will be replaced with this value in the
            output features. If the value does not fit the array dtype returned by the
            estimator, an error will be raised unless `allow_cast` is True. Defaults to
            np.nan.
        ensure_min_samples : int, default 1
            The minimum number of samples that should be passed to `transform`. If the
            array is fully masked and `skip_nodata=True`, dummy values (0) will be
            inserted to ensure this number of samples. The minimum supported number of
            samples depends on the estimator used. No effect if the array contains
            enough unmasked samples or if `skip_nodata=False`.
        allow_cast : bool, default=False
            If True and the estimator output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype instead
            of raising an error.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not np.nan, a warning will be raised if the
            selected `nodata_output` value is returned by the estimator, as this may
            indicate a valid sample being masked.
        keep_attrs : bool, default=False
            If True and the input is an Xarray object, the output will keep all
            attributes of the input features, unless they're set by the estimator (e.g.
            `_FillValue` or `long_name`). Note that some attributes (e.g.
            `scale_factor`) may become inaccurate, which is why they are dropped by
            default. The `history` attribute will always be kept. No effect if the
            input is a Numpy array.
        **transform_kwargs
            Additional arguments passed to the estimator's `transform` method.

        Returns
        -------
        Numpy or Xarray features
            The transformed features. Array types will be in the shape (features, ...)
            while xr.Dataset will store features as variables, with the feature names
            based on the estimator's `get_feature_names_out` method.
        """
        wrapped_func = self._wrapped.transform
        output_dim_name = "feature"
        features = FeatureArray.from_feature_array(X, nodata_input=nodata_input)
        feature_names = self._wrapped.get_feature_names_out()

        self._check_feature_names(features.feature_names)

        return features.apply_ufunc_across_features(
            suppress_feature_name_warnings(wrapped_func),
            output_dims=[[output_dim_name]],
            output_dtypes=[np.float64],
            output_sizes={output_dim_name: len(feature_names)},
            output_coords={output_dim_name: list(feature_names)},
            skip_nodata=skip_nodata,
            nodata_output=nodata_output,
            ensure_min_samples=ensure_min_samples,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            nan_fill=0.0,
            keep_attrs=keep_attrs,
            **transform_kwargs,
        )

    @requires_implementation
    @requires_fitted
    def inverse_transform(
        self,
        X: FeatureArrayType,
        *,
        skip_nodata: bool = True,
        nodata_input: NoDataType = None,
        nodata_output: float | int = np.nan,
        ensure_min_samples: int = 1,
        allow_cast: bool = False,
        check_output_for_nodata: bool = True,
        keep_attrs: bool = False,
        **inverse_transform_kwargs,
    ) -> FeatureArrayType:
        """
        Apply the inverse transformation to n-dimensional X features.

        Parameters
        ----------
        X : Numpy or Xarray features
            The n-dimensional input features. Array types should be in the shape
            (features, ...) while xr.Dataset should include features as variables.
            Features should correspond with those used to fit the estimator.
        skip_nodata : bool, default=True
            If True, NoData and NaN values will be skipped during prediction. This
            speeds up processing of partially masked features, but may be incompatible
            if estimators expect a consistent number of input samples.
        nodata_input : float or sequence of floats, optional
            NoData values other than NaN to mask in the output features. A single value
            will be broadcast to all features while sequences of values will be assigned
            feature-wise. If None, values will be inferred if possible based on
            available metadata.
        nodata_output : float or int or tuple, optional
            NoData samples in the input features will be replaced with this value in the
            output features. If the value does not fit the array dtype returned by the
            estimator, an error will be raised unless `allow_cast` is True. Defaults to
            np.nan.
        ensure_min_samples : int, default 1
            The minimum number of samples that should be passed to `transform`. If the
            array is fully masked and `skip_nodata=True`, dummy values (0) will be
            inserted to ensure this number of samples. The minimum supported number of
            samples depends on the estimator used. No effect if the array contains
            enough unmasked samples or if `skip_nodata=False`.
        allow_cast : bool, default=False
            If True and the estimator output dtype is incompatible with the chosen
            `nodata_output` value, the output will be cast to the correct dtype instead
            of raising an error.
        check_output_for_nodata : bool, default True
            If True and `nodata_output` is not np.nan, a warning will be raised if the
            selected `nodata_output` value is returned by the estimator, as this may
            indicate a valid sample being masked.
        keep_attrs : bool, default=False
            If True and the input is an Xarray object, the output will keep all
            attributes of the input features, unless they're set by the estimator (e.g.
            `_FillValue` or `long_name`). Note that some attributes (e.g.
            `scale_factor`) may become inaccurate, which is why they are dropped by
            default. The `history` attribute will always be kept. No effect if the
            input is a Numpy array.
        **inverse_transform_kwargs
            Additional arguments passed to the estimator's `inverse_transform` method.

        Returns
        -------
        Numpy or Xarray features
            The inverse-transformed features. Array types will be in the shape
            (features, ...) while xr.Dataset will store features as variables.
        """
        wrapped_func = self._wrapped.inverse_transform
        output_dim_name = "feature"
        features = FeatureArray.from_feature_array(X, nodata_input=nodata_input)
        feature_names = self._wrapped_meta.feature_names

        # If the estimator was fitted without feature names, use sequential identifiers
        if not feature_names:
            feature_names = generate_sequential_names(
                self._wrapped_meta.n_features, output_dim_name
            )

        return features.apply_ufunc_across_features(
            suppress_feature_name_warnings(wrapped_func),
            output_dims=[[output_dim_name]],
            output_dtypes=[np.float64],
            output_sizes={output_dim_name: self._wrapped_meta.n_features},
            output_coords={output_dim_name: feature_names},
            skip_nodata=skip_nodata,
            nodata_output=nodata_output,
            ensure_min_samples=ensure_min_samples,
            allow_cast=allow_cast,
            check_output_for_nodata=check_output_for_nodata,
            nan_fill=0.0,
            keep_attrs=keep_attrs,
            **inverse_transform_kwargs,
        )

    def _check_feature_names(self, feature_array_names: NDArray) -> None:
        """Check that feature array names match feature names seen during fitting."""
        fitted_feature_names = self._wrapped_meta.feature_names

        no_fitted_names = len(fitted_feature_names) == 0
        no_feature_names = len(feature_array_names) == 0

        if no_fitted_names and no_feature_names:
            return

        if no_fitted_names:
            warn(
                f"X has feature names, but {self._wrapped.__class__.__name__} was"
                " fitted without feature names",
                stacklevel=2,
            )
            return

        if no_feature_names:
            warn(
                "X does not have feature names, but"
                f" {self._wrapped.__class__.__name__} was fitted with feature names",
                stacklevel=2,
            )
            return

        if len(fitted_feature_names) != len(feature_array_names) or np.any(
            fitted_feature_names != feature_array_names
        ):
            msg = "Feature array names should match those passed during fit.\n"
            fitted_feature_names_set = set(fitted_feature_names)
            feature_array_names_set = set(feature_array_names)

            unexpected_names = sorted(
                feature_array_names_set - fitted_feature_names_set
            )
            missing_names = sorted(fitted_feature_names_set - feature_array_names_set)

            def add_names(names):
                max_n_names = 5
                if len(names) > max_n_names:
                    names = [*names[: max_n_names + 1], "..."]

                return "".join([f"- {name}\n" for name in names])

            if unexpected_names:
                msg += "Feature names unseen at fit time:\n"
                msg += add_names(unexpected_names)

            if missing_names:
                msg += "Feature names seen at fit time, yet now missing:\n"
                msg += add_names(missing_names)

            if not missing_names and not unexpected_names:
                msg += "Feature names must be in the same order as they were in fit.\n"

            raise ValueError(msg)


def wrap(estimator: EstimatorType) -> FeatureArrayEstimator[EstimatorType]:
    """
    Wrap an estimator with overriden methods for n-dimensional feature arrays.

    Parameters
    ----------
    estimator : BaseEstimator
        An sklearn-compatible estimator. Supported methods will be overriden to work
        with n-dimensional feature arrays. If the estimator is already fit, it will be
        reset and a warning will be raised.

    Returns
    -------
    FeatureArrayEstimator
        An estimator with relevant methods overriden to work with n-dimensional feature
        arrays.

    Examples
    --------
    Instantiate an estimator, wrap it, then fit as usual:

    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from sklearn_raster.datasets import load_swo_ecoplot
    >>> X_img, X, y = load_swo_ecoplot(as_dataset=True)
    >>> est = wrap(KNeighborsRegressor(n_neighbors=3)).fit(X, y)

    Use a wrapped estimator to predict from raster data stored in Numpy or Xarray types:

    >>> pred = est.predict(X_img)
    >>> pred.PSME_COV.shape
    (128, 128)
    """
    return FeatureArrayEstimator(estimator)
