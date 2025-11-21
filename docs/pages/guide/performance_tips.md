`sklearn-raster` supports using [Dask](https://docs.dask.org/en/stable/) to apply machine learning in parallel. Choosing to [use Dask-backed data](raster_formats.md) instead of Numpy arrays can lead to significant speedups and allows working with large datasets that don't fit into memory, but requires some additional planning to ensure that operations are optimized for your datasets and hardware.

!!! tip
    Optimizing Dask performance is a complicated topic -- the tips in this guide should be considered a supplement to the docs provided by Xarray and Dask, including [Xarray's guide to Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html), [Dask best practices for arrays](https://docs.dask.org/en/stable/array-best-practices.html), and [Dask general best practices](https://docs.dask.org/en/stable/best-practices.html). We recommend familiarizing yourself with those materials before continuing.

## Chunks and GeoTIFFs

One of the most important considerations for Dask performance is how to chunk your data. This topic is covered in detail by the Xarray guide to [chunking and performance](https://docs.xarray.dev/en/stable/user-guide/dask.html#chunking-and-performance) and the Dask guide to [selecting good chunk sizes](https://docs.dask.org/en/stable/array-best-practices.html#select-a-good-chunk-size), so the tips here will focus on performance considerations specific to raster data stored in GeoTIFFs.

### Tiling vs. Striping

GeoTIFFs can be stored using tiling or striping, which determines whether contiguous bytes in the file structure correspond to 2D blocks of pixels or 1D strips, respectively. When generating GeoTIFFs for use in `sklearn-raster` or other Dask workflows, tiling is generally preferable as it allows Dask chunks in the X and Y axes to correspond with contiguous byte ranges in the file, leading to fewer, more efficient file reads. It's also important that Dask chunk sizes are divisible by the GeoTIFF tile size so that each chunk can be read directly without touching unneeded tiles.

!!! tip
    When loading raster datasets with [rioxarray](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.open_rasterio), `chunks="auto"` will attempt to choose a chunk size that minimizes overhead *and* aligns with the GeoTIFF's internal storage.

### Multi-Band Data and Interleaving

Estimator methods can be parallelized across chunks, but always utilize the full set of features -- you can't generate a prediction with only *some* of your model's predictors. This means that the feature dimension (i.e., bands) are always treated as a single chunk. While it's not always practical, you can optimize performance by storing features in a single, pixel-interleaved multi-band GeoTIFF, so that all bands for a given pixel are available in a contiguous block of data.

## Skipping NoData

When your input features contain NoData values that should be masked in the output, `sklearn-raster` optimizes estimator methods by skipping processing for those pixels. This can lead to substantial performance improvements in heavily-masked datasets. 

See the [NoData handling guide](metadata.md#handling-nodata) for details on how to encode and specify NoData values.

## DataArrays vs. Datasets

`sklearn-raster` is compatible with two Xarray data structures: DataArrays and Datasets. While Datasets offer some additional convenience by storing features as distinct variables with independent metadata, they also add additional performance overhead as they must be converted to DataArrays internally prior to applying universal functions (ufuncs).

For performance-critical applications, DataArrays are the recommended data structure.

## Limiting Nested Parallelism

Many of the underlying methods within Scikit-Learn are internally parallelized via Numpy and SciPy. When those methods are further parallelized across Dask chunks, this creates **nested parallelism** and potential **oversubscription**[^oversubscription], where an operation requests more system resources than are available. This can lead to slowdowns, stalled computations, and system crashes.

To avoid nested parallelism, all `FeatureArrayEstimator` methods limit Dask workers to a single thread by default. This typically offers the best performance, parallelizing operations *across* chunks but not *within* chunks. However, there are cases where you may want to allow nested parallelism, e.g. when working with distributed workers that have access to independent thread pools. This can be done by increasing the `inner_thread_limit` parameter to a higher value or disabling it with `inner_thread_limit=None`.

### OpenBLAS Warnings

Even when Dask workers are limited to a single thread, you may still see warnings like the one below under certain unpredictable circumstances:

> OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata. To avoid this warning, please rebuild your copy of OpenBLAS with a larger NUM_THREADS setting or set the environment variable OPENBLAS_NUM_THREADS to 24 or lower.

This warning seems to occur exclusively on machines with high thread counts, independent of the number of threads available at runtime. Users have reported that setting the environment variable `OPENBLAS_NUM_THREADS=1` prior to calling any `sklearn-raster` code is effective at suppressing the warning.

See [this pull request](https://github.com/lemma-osu/sklearn-raster/pull/87) for more details.

[^oversubscription]: For more information on the issue of nested parallelism within the scientific Python ecosystem, see Thomas J. Fan's talk [Can There Be Too Much Parallelism](https://www.youtube.com/watch?v=hy5yDxvLCDA) from SciPy 2023.