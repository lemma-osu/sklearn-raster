import numpy as np
import xarray as xr
from sknnr import GNNRegressor

from sknnr_spatial import wrap

from .image_utils import parametrize_model_data


@parametrize_model_data(image_types=(xr.DataArray,))
def test_kneighbors_returns_df_index(model_data):
    """Test that sknnr estimators return dataframe indices."""
    # Create dummy plot data
    X = np.random.rand(10, 3) + 10.0
    y = np.random.rand(10, 3)

    # Create an image of zeros and set the first plot to zeros to ensure that the
    # first index is the nearest neighbor to all pixels
    X_image = np.zeros((2, 2, 3))
    X[0] = [0, 0, 0]

    # Convert model data to the correct types
    X_image, X, y = model_data.set(
        X_image=X_image,
        X=X,
        y=y,
    )

    # Offset the dataframe index to differentiate it from the array index
    df_index_offset = 999
    X.index += df_index_offset

    est = wrap(GNNRegressor()).fit(X, y)
    idx = est.kneighbors(X_image, return_distance=False, return_dataframe_index=False)
    df_idx = est.kneighbors(X_image, return_distance=False, return_dataframe_index=True)

    assert idx.shape == df_idx.shape

    # The first neighbor should be the first index for all pixels
    assert (idx.sel(k=1) == 0).all().compute()
    assert (df_idx.sel(k=1) == df_index_offset).all().compute()
