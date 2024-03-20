from dataclasses import dataclass

import pytest

from sknnr_spatial.datasets import load_swo_ecoplot


@dataclass
class DatasetConfiguration:
    load_function: callable
    image_size: tuple[int, int]
    n_samples: int
    n_targets: int
    n_features: int


CONFIGURATIONS = {
    "swo_ecoplot": DatasetConfiguration(
        load_function=load_swo_ecoplot,
        image_size=(128, 128),
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
    X_image, X, y = configuration.load_function(as_dataset=as_dataset)

    assert X.shape == (configuration.n_samples, configuration.n_features)
    assert y.shape == (configuration.n_samples, configuration.n_targets)

    if as_dataset:
        assert list(X.columns) == list(X_image.data_vars)
        assert X_image.sizes == {
            "y": configuration.image_size[0],
            "x": configuration.image_size[1],
        }
    else:
        assert X_image.shape == (*configuration.image_size, configuration.n_features)
