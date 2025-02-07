`sklearn-raster` can wrap estimators from `scikit-learn` and compatible third-party packages to enable raster-based predictions. To be compatible, estimators should implement a `fit` method and one or more output methods like `predict`, `predict_proba`, `transform`, or `kneighbors` which accept an array of features and an optional array of targets (for supervised learning), and return one or more corresponding arrays. The wrapper extends these methods to accept and return rasters, i.e. arrays of pixels with spatial and/or temporal dimensions.

## Supported Features

`sklearn-raster` **does** support:

- Supervised and unsupervised estimators
- Regressors, classifiers, clusterers, and transformers
- Single-output or multi-output predictions
- Output methods that return tuples of arrays, e.g. `kneighbors`
- Third party estimators like [xgboost](https://xgboost.readthedocs.io/en/stable/) and [sknnr](https://sknnr.readthedocs.io/en/latest/)

## Unsupported Features

There are a few unlikely caveats that would make an estimator incompatible. `sklearn-raster` **does not** support:

- Estimators that return an unpredictable number of output targets, such as a `predict` method that returns fewer targets than the estimator was `fit` with. To enable lazy computation, `sklearn-raster` must be able to predict output shape from the input data.
- Estimators that modify the output shape, i.e. that does not return exactly one output sample for every input sample. In this case, `sklearn-raster` isn't able to infer which input coordinate corresponds to which output coordinate, and methods will fail unpredictably.
