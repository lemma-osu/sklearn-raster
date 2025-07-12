"""
Global test configuration for pytest.

This is placed in the project root so that fixtures are in scope for both the `tests`
directory and doctests that run in the `src` directory.
"""

import os
import warnings

import numpy as np
import pooch
import pytest

from sklearn_raster import __version__
from sklearn_raster.datasets._base import DATA_URL, _data_fetcher, registry


@pytest.fixture
def dummy_model_data():
    n_features = 5
    n_rows = 10

    X_image = np.random.rand(8, 16, n_features)
    X = np.random.rand(n_rows, n_features)
    y = np.random.rand(n_rows, 3)

    return X_image, X, y


@pytest.fixture(scope="session", autouse=True)
def patch_fetcher(session_mocker):
    """
    Fetch Pooch data from main if the current package version has no corresponding tag.

    This is required to avoid failing tests between bumping the package version and
    pushing a corresponding tag, as Pooch will otherwise attempt to fetch from the tag
    before it exists.
    """
    tag_list = os.popen("git tag -l").read().strip().split("\n")
    # Tags are prefixed with "v" but package versions are not
    if f"v{__version__}" in tag_list:
        return

    main_fetcher = pooch.create(
        base_url=DATA_URL.format(version="main"),
        path=pooch.os_cache("sklearn_raster"),
        version=None,
        env="SKLEARNRASTER_DATA_DIR",
        registry=registry,
        retry_if_failed=3,
    )

    warnings.warn(
        f"Current version {__version__} has no corresponding tag. "
        "Fetching data from `main` branch for testing. This warning is expected "
        "when bumping the package version prior to pushing a new tag.",
        UserWarning,
        stacklevel=2,
    )
    session_mocker.patch.object(_data_fetcher, "fetch", main_fetcher.fetch)
