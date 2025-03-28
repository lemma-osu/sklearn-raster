import numpy as np
import pytest


@pytest.fixture
def dummy_model_data():
    n_features = 5
    n_rows = 10

    X_image = np.random.rand(8, 16, n_features)
    X = np.random.rand(n_rows, n_features)
    y = np.random.rand(n_rows, 3)

    return X_image, X, y
