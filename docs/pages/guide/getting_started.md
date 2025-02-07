`sklearn-raster` extends `scikit-learn` and other compatible estimators to work directly with raster data. This allows you to train models with tabular data and predict raster outputs directly while preserving metadata like spatial references, band names, and NoData masks.

## Raster Estimators

To get started, instantiate a regressor, classifier, or clusterer as normal, wrap it into a raster estimator with `wrap`, then fit[^fit-after-wrap] it with tabular data. The `X` dataset should include predictor features that correspond with your raster bands. For supervised classification, the `y` dataset should include one or more targets that will be predicted as output bands.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn_raster import wrap

est = wrap(RandomForestRegressor(n_estimators=500))
est.fit(X, y)
```

Once fit, methods like `predict`, `predict_proba`, and `kneighbors` can be applied to input rasters.

```python
import rioxarray

da = rioxarray.open_rasterio("rgb_image.tif")
pred = est.predict(da)
```

## Next Steps

### User Guide

The user guide contains more information about specific topics like:

- Which [estimators are compatible](compatible_estimators.md) with `sklearn-raster`
- Supported [raster formats](raster_formats.md) and their pros and cons
- How [metadata](metadata.md) like spatial references, band names, and NoData masks are handled

### Tutorials

Coming soon.

[^fit-after-wrap]: Estimators *must* be wrapped before fitting to allow `sklearn-raster` to access necessary metadata like the names and number of targets. Wrapping a pre-fit estimator will reset the estimator and raise a warning.

