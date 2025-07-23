There are a number of other packages that combine Scikit-Learn with Xarray and/or use estimators for mapping. The table below attempts to compare their features (to the best of our understanding) with `sklearn-raster` so you can choose the right tool for your application. 

*Supported features are described in detail [below](#supported-features).*

<style>
/* Highlight the sklearn-raster header */
table tr > th:nth-child(2) {
  border: 2px dotted var(--md-typeset-a-color);
}
</style>

| [Supported features](#supported-features) | [sklearn-raster](https://sklearn-raster.readthedocs.io/en/latest/) | [sklearn-xarray](https://phausamann.github.io/sklearn-xarray/) | [dask-ml](https://ml.dask.org/) | [scikit-map](https://github.com/openlandmap/scikit-map) | [pyimpute](https://github.com/perrygeo/pyimpute) | [pyspatialml](https://stevenpawley.github.io/Pyspatialml/) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 2D outputs | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 3D outputs | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| n-D outputs  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Preserves metadata | ✅ | ❌ | ❌ | ✅[^on-disk] | ✅[^on-disk] | ✅[^on-disk] |
| Parallel computation | ✅ | ✅[^wrapper] | ✅ | ✅ | ❌ | ❌ |
| Lazy computation | ✅ | ✅[^deferred] | ✅ | ❌ | ❌ | ❌ |
| Xarray support | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Parallel fitting | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Custom estimators | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Raster processing tools | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

### Supported Features

- **2D outputs**: Package is capable of generating 2D model outputs from 2D feature inputs, e.g. predicting georeferenced land cover maps or transforming geospatial imagery.
- **3D outputs**: Package is capable of generating 3D model outputs from 3D feature inputs, e.g. predicting a time series of georeferenced maps.
- **n-D outputs**: Package is capable of generating model outputs with arbitrary dimensionality from n-D feature inputs, e.g. predicting a time series of climate variables at various pressure levels.
- **Preserves metadata**: Model outputs retain the metadata of the input data, such as spatial references, band names, and NoData masks.
- **Parallel computation**: Package computes model outputs in parallel.
- **Lazy computation**: Package computes model outputs lazily, deferring computation until necessary.
- **Xarray support**: Package supports Xarray data structures as model inputs and outputs.
- **Parallel fitting**: Package supports fitting estimators in parallel to reduce training time.
- **Custom estimators**: Package implements additional estimators.
- **Raster processing tools**: Package includes additional functionality for processing raster data beyond ML modeling.

[^on-disk]: `scikit-map`, `pyimpute`, and `pyspatialml` only preserve metadata when writing outputs directly to disk.
[^wrapper]: `sklearn-xarray` supports parallel operations by [wrapping `dask-ml` estimators](https://phausamann.github.io/sklearn-xarray/content/wrappers.html#wrapping-dask-ml-estimators).
[^deferred]: `sklearn-xarray` [supports lazy evaluation](https://phausamann.github.io/sklearn-xarray/content/target.html#lazy-evaluation) by deferring computation, but requires that the entire dataset is loaded into memory.