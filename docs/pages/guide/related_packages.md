There are a number of other packages that combine Scikit-Learn with Xarray and/or use estimators for mapping. The table below attempts to compare their features (to the best of our understanding) with `sklearn-raster` so you can choose the right tool for your application.

<style>
/* Highlight the sklearn-raster header */
table tr > th:nth-child(2) {
  border: 2px dotted var(--md-typeset-a-color);
}
</style>

| Supported features | [sklearn-raster](https://sklearn-raster.readthedocs.io/en/latest/) | [sklearn-xarray](https://phausamann.github.io/sklearn-xarray/) | [dask-ml](https://ml.dask.org/) | [scikit-map](https://github.com/openlandmap/scikit-map) | [pyimpute](https://github.com/perrygeo/pyimpute) | [pyspatialml](https://stevenpawley.github.io/Pyspatialml/) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 2D prediction | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 3D prediction | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| n-D prediction  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Preserve NoData and spatial refs. | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Parallel prediction | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Lazy prediction | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Xarray support | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Parallel fitting | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Custom estimators | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Raster processing tools | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
