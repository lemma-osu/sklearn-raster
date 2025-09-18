`sklearn-raster` supports applying estimator methods to rasters stored in a variety of formats, referred to collectively as **feature arrays**. Numpy arrays can be used for small rasters that fit easily in memory, while Xarray `Dataset` and `DataArray` are ideal for large datasets that benefit from deferred, parallel computation. Pandas dataframes can be used for extracted tabular data. 

Estimator methods will return data in the same format that it is provided, e.g. predictions generated from an Xarray `xr.Dataset` will be stored in an `xr.Dataset`. Most feature arrays support [arbitrary dimensionality](#dimensionality).

## Raster Formats

### Numpy Array

Any Numpy array in the shape `(band, y, x)`[^bands] can be used with `sklearn-raster` estimators. These can be loaded from a GeoTIFF file using `rasterio`:

```python
import numpy as np
import rasterio

with rasterio.open("./rgb_image.tif") as src:
    array: np.ndarray = src.read()

print(array.shape) # (3, 128, 128)
```

Calling a method like `predict` with a Numpy array will return a Numpy array with the same shape `(band, y, x)`. The spatial dimensions will match the input raster, while the number of bands will match the target data used to fit the estimator. For example, given a wrapped single-output estimator:

```python
pred: np.ndarray = wrapped_estimator.predict(array)
print(pred.shape) # (1, 128, 128) 
```

### Xarray DataArray

An `xr.DataArray` in the shape `(band, y, x)` can also be used with `sklearn-raster` estimators, and offers a number of benefits over a Numpy array:

1. Data can be loaded lazily.
2. Data can be chunked, allowing parallel processing and larger-than-memory rasters.
3. Spatial coordinates, NoData values, and other metadata can be stored in the raster.

Load a `DataArray` from a GeoTIFF file using `rioxarray`:

```python
import rioxarray
import xarray as xr

da: xr.DataArray = rioxarray.open_rasterio("./rgb_image.tif")

print(da.shape) # (3, 128, 128)
```

Calling `predict` on a wrapped estimator will return an `xr.DataArray` with the target band(s) as the first dimension. [Where possible](metadata.md#band-names), band names will be stored as coordinates in the `target` dimension of the output raster.

```python
pred = wrapped_estimator.predict(da)
print(pred.shape) # (1, 128, 128)
print(pred.dims) # ('target', 'y', 'x')
print(pred["target"].values) # ['land_cover']
```

### Xarray Dataset

An `xr.Dataset` in the shape `(y, x)` with bands stored as variables can also be used with `sklearn-raster` estimators. It offers similar benefits to `xr.DataArray`, with the added ability to mix data types[^mixed-types] and NoData values across bands. 

Load a `Dataset` from a GeoTIFF file using `rioxarray`:

```python
import rioxarray
import xarray as xr

ds: xr.Dataset = rioxarray.open_rasterio("./rgb_image.tif", bands_as_variables=True)

print(ds.R.shape) # (128, 128)
```

The output of `predict` will be an `xr.Dataset` with target band(s) as variables.

```python
pred = wrapped_estimator.predict(ds)
print(pred.data_vars) # ['land_cover']
print(pred.land_cover.shape) # (128, 128)
```

### Pandas DataFrame

While dataframes are not a conventional raster format, they can be used for applications like storing extracted pixel values in a tabular format of shape `(samples, band)`. In that context, a `FeatureArrayEstimator` provides some convenient features over an unmodified `sklearn` estimator when predicting or transforming dataframes:

1. Methods return dataframe outputs that preserve the index and target names as columns.
2. Samples with masked values in the input data can be skipped and encoded in the output dataframe.

### Format Summary

| <div style="width: 100px;">Raster format</div> | Arbitrary dimensionality | Parallel operations | Lazy evaluation | Larger-than-memory | Metadata attributes | Mixed variable types |
|:-------------:|--------------------------|---------------------|-----------------|--------------------|---------------------|----------------------|
| `np.ndarray` | ✅ |❌ | ❌ | ❌ | ❌ | ❌ |
| `xr.DataArray` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `xr.Dataset` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅[^mixed-types] |
| `pd.DataFrame` | ❌ |❌ | ❌ | ❌ | ❌ | ✅[^mixed-types] |

## Dimensionality

While the examples above focus on simple spatial rasters with `x` and `y` dimensions, `sklearn-raster` supports arbitrary input and output dimensionality. For example, generating predictions from a time series of climate data at various pressure levels of shape `(variable, time, z, y, x)` would return an output of shape `(target, time, z, y, x)`. Operations are broadcast by implicitly flattening all non-feature dimensions.

[^bands]: `(band, y, x)` is the common shape for 2D geospatial raster data, but any shape is supported as long as the first dimension corresponds with the feature columns of the training dataset.

[^mixed-types]: Data are implicitly converted to `xr.DataArray` when applying estimator methods, which causes mixed data types to be promoted to the maximum data type.