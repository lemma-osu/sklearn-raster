# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- Fix potential performance and stability issues from OpenBLAS oversubscription by limiting nested parallelism in [#87](https://github.com/lemma-osu/sklearn-raster/pull/87)

### Added

- All estimator methods now accept an `inner_thread_limit` keyword argument which limits nested parallelism within Dask workers

### Changed

- Replaced `utils.features.reshape_to_samples` with the more generalized `utils.decorators.with_input_dimensions` which supports reshaping multiple arrays to arbitrary dimensionality

## [0.1.0.dev1] - 2025-11-04

### Fixed

- Fix `xarray` being incorrectly listed as an optional dependency
- Fix deprecation warning for `xarray>=2025.08.0` by explicitly setting `compat` mode

### Added

- HTML reprs for `FeatureArrayEstimator` in a Jupyter notebook.
- The estimator wrapped by `FeatureArrayEstimator` is exposed with the `wrapped_estimator` attribute.
- Added support for applying wrapped estimator methods to `pd.DataFrame` feature arrays.
- Expose `sklearn_raster.FeatureArrayEstimator` as a top-level import.
- `FeatureArrayEstimator` exposes `n_features_in_`, `n_targets_in_`, `feature_names_in_`, and `target_names_in_` as public attributes after fitting.

### Changed

- The `wrap` function is deprecated in favor of instantiating `FeatureArrayEstimator` directly.
- Calling `predict`, `predict_proba`, `transform`, and `kneighbors` with `nodata_input=None` now overrides `_FillValue` attributes like any other `nodata_input` value. The default behavior when `nodata_input` is not provided is unchanged, and will infer from `_FillValue` if present.

### Removed

- Dask `diagnostics` and `dataframe` extras are no longer required dependencies
- `FeatureArrayEstimator` no longer supports direct access to the wrapped estimator attributes. These attributes should be accessed via the `wrapped_estimator` attribute instead.

## [0.1.0.dev0] - 2025-07-11

### Added

- Initial developmental release

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[unreleased]: https://github.com/lemma-osu/sklearn-raster/compare/v0.1.0.dev1...HEAD
[0.1.0.dev1]: https://github.com/lemma-osu/sklearn-raster/compare/v0.1.0.dev0...v0.1.0.dev1
[0.1.0.dev0]: https://github.com/lemma-osu/sklearn-raster/releases/tag/v0.1.0.dev0
