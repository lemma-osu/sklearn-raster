import pickle
from dataclasses import dataclass
from typing import Any

import pytest

from sknnr_spatial.datasets import load_swo_ecoplot


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
        assert X_image.shape == (*configuration.image_size, configuration.n_features)


def test_load_dataset_with_chunks():
    """Test that the chunk size is respected when loading the image as a dataset."""
    chunks = {"x": (128,), "y": (128,)}
    X_image, _, _ = load_swo_ecoplot(as_dataset=True, chunks=chunks)

    assert X_image.chunksizes == chunks


def test_load_dataset_names_match():
    """Test that the X names and order match between the image and dataframe."""
    X_image, X, _ = load_swo_ecoplot(as_dataset=True)

    assert list(X.columns) == list(X_image.data_vars)
