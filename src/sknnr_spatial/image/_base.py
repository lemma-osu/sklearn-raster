from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from types import ModuleType
from typing import TYPE_CHECKING, Generic

import numpy as np
from numpy.typing import NDArray

from ..types import ImageType

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.neighbors._base import KNeighborsMixin

    from ..estimator import ImageEstimator
    from ..types import NoDataType


class ImagePreprocessor(ABC):
    """
    A pre-processor for multi-band image data in a machine learning workflow.

    This class handles flattening an image from 3D pixel space (e.g. y, x, band)
    to 2D sample space (sample, band) to allow prediction and other operations
    with scikit-learn estimators designed for sample data, and unflattening to
    convert scikit-learn outputs from sample space back to pixel space.

    Pre-processing can also fill NaN values in sample space to allow prediction with
    estimators that prohibit NaNs, and mask NoData values in sample space outputs
    during unflattening to preserve NoData masks from the input image.

    Parameters
    ----------
    image: ImageType
        An image to be processed.
    nodata_vals : float | tuple[float] | NDArray | None, default None
        Values representing NoData in the image. This can be represented by a
        single value or a sequence of values. If a single value is passed, it will
        be applied to all bands. If a sequence is passed, there must be one element
        for each band in the image. For float images, np.nan will always be treated
        as NoData.
    nan_fill : float | None, default 0.0
        If not None, NaN values in the flattened image will be filled with this
        value. This is important when passing flattened arrays into scikit-learn
        estimators that prohibit NaN values.

    Attributes
    ----------
    image : ImageType
        The original, unmodified image used to construct the preprocessor.
    flat : ImageType
        A flat representation of the image, with the x and y dimensions flattened to a
        sample dimension.
    n_bands : int
        The number of bands present in the image.
    band_dim : int
        The dimension used for bands or features in the unmodified image.
    flat_band_dim : int
        The dimension used for bands or features in the flattened image.
    """

    # The module used for array operations on the image type.
    _backend: ModuleType

    band_dim: int
    flat_band_dim = -1

    def __init__(
        self,
        image: ImageType,
        nodata_vals: NoDataType = None,
        nan_fill: float | None = 0.0,
    ):
        self.image = image
        self.flat = self._flatten(self.image)

        self.nodata_vals = self._validate_nodata_vals(nodata_vals)
        self.nodata_mask = self._get_nodata_mask(self.flat)

        if nan_fill is not None:
            self.flat[self._backend.isnan(self.flat)] = nan_fill

    @property
    def n_bands(self) -> int:
        return self.image.shape[self.band_dim]

    @abstractmethod
    def _flatten(self, image: ImageType) -> ImageType:
        """Ravel the image's x, y dimensions while keeping the band dimension."""

    @abstractmethod
    def unflatten(self, flat_image: ImageType, *, apply_mask: bool = True) -> ImageType:
        """Reconstruct the x, y dims of a flattened image to the original shape."""

    def _get_nodata_mask(self, flat_image: ImageType) -> ImageType | None:
        """
        Get a mask of NoData values in the shape (pixels,) for the flat image.

        NoData values are represented by True in the output array.
        """
        # Skip allocating a mask if the image is float and NoData wasn't given
        if not (is_float := flat_image.dtype.kind == "f") and self.nodata_vals is None:
            return None

        mask = self._backend.zeros(flat_image.shape, dtype=bool)

        # If it's floating point, always mask NaNs
        if is_float:
            mask |= self._backend.isnan(flat_image)

        # If NoData was specified, mask those values
        if self.nodata_vals is not None:
            mask |= flat_image == self.nodata_vals

        # Set the mask where any band contains NoData
        return mask.max(axis=self.flat_band_dim)

    def _validate_nodata_vals(self, nodata_vals: NoDataType) -> NDArray | None:
        """
        Get an array of NoData values in the shape (bands,) based on user input.

        Scalars are broadcast to all bands while sequences are checked against the
        number of bands and cast to ndarrays. There is no need to specify np.nan as a
        NoData value because it will be masked automatically for floating point images.
        """
        if nodata_vals is None:
            return None

        # If it's a numeric scalar, broadcast it to all bands
        if isinstance(nodata_vals, (float, int)) and not isinstance(nodata_vals, bool):
            return np.full((self.n_bands,), nodata_vals)

        # If it's not a scalar, it must be an iterable
        if not isinstance(nodata_vals, Sized) or isinstance(nodata_vals, (str, dict)):
            raise TypeError(
                f"Invalid type `{type(nodata_vals).__name__}` for `nodata_vals`. "
                "Provide a single number to apply to all bands, a sequence of numbers, "
                "or None."
            )

        # If it's an iterable, it must contain one element per band
        if len(nodata_vals) != self.n_bands:
            raise ValueError(
                f"Expected {self.n_bands} NoData values but got {len(nodata_vals)}. "
                f"The length of `nodata_vals` must match the number of bands."
            )

        return np.asarray(nodata_vals, dtype=float)

    def _fill_nodata(self, flat_image: ImageType, nodata_fill_value=0.0) -> ImageType:
        """Fill values in a flat image based on the pre-calculated mask."""
        if self.nodata_mask is None:
            return flat_image

        if isinstance(nodata_fill_value, float):
            flat_image = flat_image.astype(float)

        flat_image[self.nodata_mask, :] = nodata_fill_value

        return flat_image


class ImageWrapper(ABC, Generic[ImageType]):
    """A wrapper around an image that provides sklearn methods."""

    preprocessor_cls: type[ImagePreprocessor]

    def __init__(self, image: ImageType, nodata_vals: NoDataType = None):
        self.image = image
        self.nodata_vals = nodata_vals
        self.preprocessor = self.preprocessor_cls(image, nodata_vals=nodata_vals)

    @abstractmethod
    def predict(
        self,
        *,
        estimator: ImageEstimator[BaseEstimator],
    ) -> ImageType: ...

    @abstractmethod
    def kneighbors(
        self,
        *,
        estimator: ImageEstimator[KNeighborsMixin],
        **kneighbors_kwargs,
    ) -> ImageType | tuple[ImageType, ImageType]: ...
