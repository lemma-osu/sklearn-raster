An estimator wrapped by `sklearn-raster` will attempt to utilize metadata like spatial references, band names, and NoData values from input rasters when generating output rasters. The details of how different properties are used and set are given below.

## Spatial References

When the input raster to a wrapped estimator defines a spatial reference and/or spatial coordinates, those will be applied unchanged to output rasters. This is only possible for Xarray [raster formats](raster_formats.md), as Numpy arrays do not store spatial metadata.

!!! tip
    See the [rioxarray CRS guide](https://corteva.github.io/rioxarray/stable/getting_started/crs_management.html) for more details on how coordinate systems are represented in `xarray` objects.

## Band Names

When possible, `sklearn-raster` will extract column names from the `X` features and `y` targets during fitting and use these to validate and set raster band names for compatible [raster formats](raster_formats.md).

For example, if an estimator is fit with targets stored in a Pandas dataframe and used to predict an Xarray dataset, the column names will be used as the variable names in the output raster. Other wrapped methods will attempt to set reasonable variable names, like sequential neighbor indexes for `kneighbors` (`neighbor0`, `neighbor1`, etc.). 

Available column names are also used to validate input band names. For example, attempting to predict from an Xarray dataset with variable names that are mismatched from the input dataframe seen during fitting will raise an error. Fitting with unnamed data (i.e. a Numpy array) and predicting with named data (i.e. Xarray objects), or vice versa, will raise a warning as names cannot be validated.

## Handling NoData

`sklearn-raster` has special handling for pixels that represent masked or null values, a.k.a. NoData. These NoData values include NaN for floating point rasters as well as encoded values like `-32768` or `-9999`. NoData handling includes:

- Filling NaNs in the input raster to facilitate methods that don't support nulls
- Skipping NoData pixels when applying methods to speed up processing of masked rasters
- Masking NoData pixels in output rasters

Details on how NoData is specified and applied are provided below.

### Input NoData

Pixels that contain NoData in the input raster will be skipped when processing with `skip_nodata=True`, and will be replaced with a mask value in the output raster. NoData values in the input raster can be specified in a few different ways:

- By preprocessing to replace encoded values with NaN
- By manually specifying encoded NoData values with the `nodata_input` parameter
- By storing `_FillValue` attributes in an Xarray raster

#### Replacing with NaN

Because NaN is always treated as NoData, the simplest way to identify NoData in an input raster is to replace null pixels with NaN. For example, with a Numpy array:

```python
NODATA = -32768
img = np.where(img == NODATA, np.nan, img)
```

or with an `xr.Dataset`:

```python
NODATA = -32768
img = img.where(img != NODATA)
```

The downside of this approach is that NaN can only be stored in a floating point array. To preserve NoData in integer data types, NoData values can be manually specified or stored in Xarray metadata.

#### Manually Specifying Input NoData

Most methods take a `nodata_input` parameter where you can specify encoded values to treat as NoData. When all bands use the same value, you can provide a scalar value, e.g. `nodata_input=-32768`. When a raster contains multiple bands with different values used to encode NoData, you can provide a sequence with one value per band, e.g. `nodata_input=(-32768, 0, 255)`.

`sklearn-raster` will use the provided values to internally build a NoData mask that it uses to skip null pixels and postprocess output results.

#### Storing NoData Values in Raster Metadata

If `nodata_input` is not provided, `sklearn-raster` will attempt to infer NoData values from the `_FillValue` attribute using the following rules:

- In an `xr.Dataset`, the `_FillValue` set on each variable will be assigned to that feature.
- In an `xr.DataArray`, the `_FillValue` set on the array will be assigned to all features.

If no `_FillValue` attribute is present and no `nodata_input` is provided, all non-NaN values will be treated as valid data.

!!! tip
    `_FillValue` is automatically set when loading GeoTiffs using `rioxarray`, but can also be set manually using [`assign_attrs`](https://docs.xarray.dev/en/latest/generated/xarray.DataArray.assign_attrs.html).

### Skipping NoData

By default, methods like `predict` and `kneighbors` will remove NoData pixels before applying their wrapped methods. This behavior (controlled by the `skip_nodata` parameter) substantially improves performance with heavily-masked rasters, and has little to no overhead otherwise. 

However, some estimator methods have a minimum sample size constraint that may be violated when most or all of a raster is masked. In this case, the `ensure_min_samples` parameter can be adjusted to pass a certain number of dummy values into the method and prevent empty inputs. Alternatively, skipping of NoData can be disabled with `skip_nodata=False`.

When skipping is disabled, `sklearn-raster` will fill NaN values with `0` to facilitate methods that fail on NaN inputs.

### Output NoData

Pixels that contain NoData in the input raster will be masked in the output raster. Masking is applied across bands; if a pixel is encoded as NoData in only one band of the input raster, it will be masked in all bands of the output raster.

Most methods accept a `nodata_output` parameter where you can specify the value to encode for masked pixels. `sklearn-raster` will check this value against the output data type and raise an error when the value is incompatible, e.g. `nodata_output=np.nan` for a `uint8` raster type. Instead of erroring, you can automatically cast output rasters to fit the `nodata_output` by setting `allow_cast=True`. 

If a method returns the selected `nodata_output` value, e.g. an estimator predicts a value that was chosen to represent NoData, like `-32768`, `sklearn-raster` will raise a warning about potentially masking a valid pixel. If this is an expected output, you can disable the warning with `check_output_for_nodata=False`.

For Xarray raster types, `sklearn-raster` will set a `_FillValue` attribute on each variable to the selected non-NaN `nodata_output` value, if given.

## CF Attributes

`sklearn-raster` attempts to maintain [CF compliant attributes](https://cfconventions.org/). This includes:

1. Setting selected NoData output values in variable `_FillValue` attributes.
1. Setting target names in variable `long_name` attributes.
1. Appending time-stamped operations to the `history` attribute to help track provenance. 

By default, wrapped estimator methods discard all other attributes to avoid preserving inaccurate metadata, e.g. a `scale_factor` that no longer applies to a predicted target. To preserve all attributes, pass `keep_attrs=True` to the wrapped estimator.