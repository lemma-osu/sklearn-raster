# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- HTML reprs for `FeatureArrayEstimator` in a Jupyter notebook.
- The estimator wrapped by `FeatureArrayEstimator` is exposed with the `wrapped_estimator` attribute.

### Changed

- Calling `predict`, `predict_proba`, `transform`, and `kneighbors` with `nodata_input=None` now overrides `_FillValue` attributes like any other `nodata_input` value. The default behavior when `nodata_input` is not provided is unchanged, and will infer from `_FillValue` if present.

### Removed

- Dask `diagnostics` and `dataframe` extras are no longer required dependencies

## [0.1.0.dev0] - 2025-07-11

### Added

- Initial developmental release

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[unreleased]: https://github.com/lemma-osu/sklearn-raster/compare/v0.1.0.dev0...HEAD
[0.1.0.dev0]: https://github.com/lemma-osu/sklearn-raster/releases/tag/v0.1.0.dev0
