"""Tests for dataset loading."""

import pickle
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import rasterio
from numpy.testing import assert_array_almost_equal
from typing_extensions import Any

from sklearn_raster.datasets import load_swo_ecoplot
from sklearn_raster.datasets._base import _load_rasters_to_array


@dataclass
class DatasetConfiguration:
    load_function: callable
    load_kwargs: dict[str, Any]
    image_size: tuple[int, int]
    n_samples: int
    n_targets: int
    n_features: int


CONFIGURATIONS = {
    "swo_ecoplot": DatasetConfiguration(
        load_function=load_swo_ecoplot,
        load_kwargs={},
        image_size=(128, 128),
        n_samples=3005,
        n_targets=25,
        n_features=18,
    ),
    "swo_ecoplot_large": DatasetConfiguration(
        load_function=load_swo_ecoplot,
        load_kwargs={"large_rasters": True},
        image_size=(2048, 4096),
        n_samples=3005,
        n_targets=25,
        n_features=18,
    ),
}


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
@pytest.mark.parametrize("as_dataset", [False, True], ids=["as_array", "as_dataset"])
def test_load_dataset(configuration: DatasetConfiguration, as_dataset: bool):
    X_image, X, y = configuration.load_function(
        as_dataset=as_dataset, **configuration.load_kwargs
    )

    assert X.shape == (configuration.n_samples, configuration.n_features)
    assert y.shape == (configuration.n_samples, configuration.n_targets)

    if as_dataset:
        # All datasets should contain global and variable attrs
        assert "title" in X_image.attrs
        assert "comment" in X_image.attrs
        assert "_FillValue" not in X_image.attrs, "_FillValue should not be global"
        for var in X_image.data_vars.values():
            assert "_FillValue" in var.attrs
            assert "long_name" in var.attrs

        # Some Dask schedulers require pickling, so ensure that the loaded dataset is
        # pickleable during compute. We could try computing directly, but that is much
        # slower.
        assert pickle.dumps(X_image)
        assert list(X.columns) == list(X_image.data_vars)
        assert X_image.sizes == {
            "y": configuration.image_size[0],
            "x": configuration.image_size[1],
        }
    else:
        assert X_image.shape == (configuration.n_features, *configuration.image_size)


def test_load_dataset_with_chunks():
    """Test that the chunk size is respected when loading the image as a dataset."""
    chunks = {"x": (128,), "y": (128,)}
    X_image, _, _ = load_swo_ecoplot(as_dataset=True, chunks=chunks)

    assert X_image.chunksizes == chunks


def test_load_dataset_names_match():
    """Test that the X names and order match between the image and dataframe."""
    X_image, X, _ = load_swo_ecoplot(as_dataset=True)

    assert list(X.columns) == list(X_image.data_vars)


@pytest.mark.parametrize("missing_import", ["rioxarray", "rasterio", "sknnr", "pooch"])
def test_load_dataset_missing_imports(missing_import):
    import re

    msg = re.escape("install them with `pip install sklearn-raster[datasets]`")

    with mock.patch.dict(sys.modules):
        sys.modules[missing_import] = None

        # This is pretty brittle, but it currently does the job of "un-importing"
        # the datasets module to force Python to re-import and run the dependency check
        del sys.modules["sklearn_raster.datasets"]
        del sys.modules["sklearn_raster.datasets._base"]

        with pytest.raises(ImportError, match=msg):
            from sklearn_raster.datasets import load_swo_ecoplot  # noqa: F401


def test_load_rasters_promotes_dtype():
    """Test that loading rasters from paths promotes to the largest dtype."""
    int_array = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
    float_array = np.random.rand(10, 10).astype(np.float32)
    expected_array = np.stack([int_array, float_array])

    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
        # For simplicity, just ignore rasterio warnings about missing geotransforms
        warnings.filterwarnings("ignore", message="Dataset has no geotransform")
        int_path = Path(tmpdir) / "int.tif"
        float_path = Path(tmpdir) / "float.tif"
        meta = {"height": 10, "width": 10, "count": 1}

        with rasterio.open(int_path, "w", dtype=np.uint8, **meta) as dst:
            dst.write(int_array, 1)

        with rasterio.open(float_path, "w", dtype=np.float32, **meta) as dst:
            dst.write(float_array, 1)

        array = _load_rasters_to_array([int_path, float_path])

    assert array.dtype == np.float32
    # Allow for small floating point errors during writing/reading
    assert_array_almost_equal(array, expected_array)
