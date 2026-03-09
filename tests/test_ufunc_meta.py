from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sklearn_raster.features import FeatureArray
from sklearn_raster.ufunc._meta import Dimension, Output, _UfuncMeta

from .feature_utils import wrap_features


def test_ufuncmeta_attributes_with_multiple_dimensions():
    outputs = [
        Output(
            dims=[
                Dimension(name="x", size=2, coords=["x0", "x1"]),
                Dimension(name="y", size=3, coords=["y0", "y1", "y2"]),
            ],
            dtype=np.float32,
            nodata=-999.0,
        ),
    ]

    meta = _UfuncMeta.from_outputs(outputs)

    assert meta.output_sizes == {"x": 2, "y": 3}
    assert meta.output_dtypes == (np.float32,)
    assert meta.output_core_dims == (["x", "y"],)
    assert meta.output_coords == ({"x": ["x0", "x1"], "y": ["y0", "y1", "y2"]},)
    assert meta.nodata_outputs == (-999.0,)


def test_ufuncmeta_attributes_with_multiple_outputs():
    meta = _UfuncMeta.from_outputs(
        [
            Output(
                dims=[Dimension(name="foo", size=2, coords=["x0", "x1"])],
                dtype=np.float32,
                nodata=-999.0,
            ),
            Output(
                dims=[Dimension(name="bar", size=1, coords=None)],
                dtype=np.uint8,
                nodata=0,
            ),
        ]
    )

    assert meta.output_sizes == {"foo": 2, "bar": 1}
    assert meta.output_dtypes == (np.float32, np.uint8)
    assert meta.output_core_dims == (["foo"], ["bar"])
    assert meta.output_coords == ({"foo": ["x0", "x1"]}, {})
    assert meta.nodata_outputs == (-999.0, 0)


def test_output_from_1d_constructs_dimension():
    meta = Output.from_1d(
        name="variable",
        size=3,
        coords=["a", "b", "c"],
        dtype=np.dtype(np.float32),
        nodata=-999.0,
    )

    assert len(meta.dims) == 1
    assert meta.dims[0] == Dimension(name="variable", size=3, coords=["a", "b", "c"])
    assert meta.dtype == np.dtype(np.float32)
    assert meta.nodata == -999.0


def test_ufuncmeta_attributes_with_none():
    meta = _UfuncMeta.from_outputs(
        [
            Output(
                dims=[
                    Dimension(name="foo", size=None, coords=None),
                    Dimension(name="bar", size=3, coords=["a", "b", "c"]),
                ],
                nodata=0,
            ),
        ]
    )

    assert "foo" in meta.output_core_dims[0]
    # Dimensions with None should be omitted from sizes and coords
    assert meta.output_sizes == {"bar": 3}
    assert meta.output_coords == ({"bar": ["a", "b", "c"]},)
    # Outputs with None should be included in dtypes and nodata_outputs
    assert meta.output_dtypes == (None,)
    assert meta.nodata_outputs == (0,)


def test_ufuncmeta_merges_sizes_across_outputs():
    meta = _UfuncMeta.from_outputs(
        [
            Output(dims=[Dimension(name="k", size=3)], nodata=0),
            Output(dims=[Dimension(name="k", size=3)], nodata=0),
        ]
    )

    assert meta.output_sizes == {"k": 3}


def test_ufuncmeta_raises_on_conflicting_dim_sizes():
    outputs = [
        Output(dims=[Dimension(name="k", size=3)], nodata=0),
        Output(dims=[Dimension(name="k", size=4)], nodata=0),
    ]

    expected_msg = (
        "Different sizes for the same dimension are not supported. "
        "Found dimension 'k' with sizes 3 and 4."
    )

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        _UfuncMeta.from_outputs(outputs)


@pytest.mark.parametrize(
    ("dtype", "expected_nodata"),
    [
        (np.float32, np.float32(np.nan)),
        (np.float64, np.float64(np.nan)),
        (np.int16, -32768),
        (np.uint8, 255),
    ],
)
def test_ufuncmeta_infers_nodata_from_dtype(dtype, expected_nodata):
    meta = Output(dims=[Dimension(name="x", size=2)], dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        assert np.isnan(meta.nodata)
        assert np.dtype(meta.nodata).type == np.dtype(dtype).type
    else:
        assert meta.nodata == expected_nodata


@pytest.mark.parametrize("dtype", [np.dtype(str), None])
def test_ufuncmeta_raises_if_nodata_cannot_be_inferred(dtype):
    expected_msg = "NoData value could not be inferred. Please provide `nodata`."

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        Output(dims=[Dimension(name="x", size=2)], dtype=dtype)


def test_dimension_infers_size_from_coords():
    assert Dimension(name="k", size=None, coords=["a", "b", "c"]).size == 3


def test_dimension_raises_if_size_inconsistent_with_coords():
    expected_msg = "Dimension 'k' has size 2 but 3 coordinates."

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        Dimension(name="k", size=2, coords=["a", "b", "c"])


@pytest.mark.parametrize("coords", [42, "foo", {"x": 1}])
def test_dimension_raises_if_coords_not_tuple_or_list(coords):
    expected_msg = (
        "Dimension coordinates must be a list or tuple of values. Got "
        f"`{coords.__class__.__name__}`."
    )

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        Dimension(name="k", size=None, coords=coords)


def test_with_inputs_constructs_called_meta_with_defaults():
    ufunc_meta = _UfuncMeta.from_outputs(
        [
            Output(
                dims=[Dimension(name="target", size=2)],
                dtype=np.float32,
                nodata=-1.0,
            )
        ]
    )
    features = FeatureArray.from_feature_array(
        wrap_features(np.ones((2, 3, 4)), type=xr.DataArray),
        nodata_input=[-10.0, -20.0],
    )

    called_meta = ufunc_meta.with_inputs(features)

    assert called_meta.num_outputs == 1
    assert called_meta.output_sizes == {"target": 2}
    assert called_meta.output_core_dims == (["target"],)
    assert called_meta.output_dtypes == (np.float32,)
    assert called_meta.nodata_outputs == (-1.0,)
    assert called_meta.feature_dim_name == "variable"
    assert called_meta.input_core_dims == (["variable"],)
    assert called_meta.exclude_dims == {"variable"}
    assert len(called_meta.nodata_inputs) == 1
    assert called_meta.nodata_inputs[0].data.tolist() == [-10.0, -20.0]
    assert called_meta.nodata_inputs[0].mask.tolist() == [False, False]


def test_with_inputs_overrides_nodata_outputs_scalar_and_sequence():
    ufunc_meta = _UfuncMeta.from_outputs(
        [
            Output(dims=[Dimension(name="a", size=1)], dtype=np.float32, nodata=-1.0),
            Output(dims=[Dimension(name="b", size=1)], dtype=np.uint8, nodata=255),
        ]
    )
    features = FeatureArray.from_feature_array(
        wrap_features(np.ones((2, 5)), type=xr.DataArray)
    )

    called_scalar = ufunc_meta.with_inputs(features, nodata_output=-999)
    assert called_scalar.nodata_outputs == (-999, -999)

    called_seq = ufunc_meta.with_inputs(features, nodata_output=(-7, 7))
    assert called_seq.nodata_outputs == (-7, 7)


def test_with_inputs_raises_on_invalid_nodata_output_length():
    ufunc_meta = _UfuncMeta.from_outputs(
        [
            Output(dims=[Dimension(name="a", size=1)], dtype=np.float32, nodata=-1.0),
            Output(dims=[Dimension(name="b", size=1)], dtype=np.uint8, nodata=255),
        ]
    )
    features = FeatureArray.from_feature_array(
        wrap_features(np.ones((2, 5)), type=xr.DataArray)
    )

    expected_msg = (
        "The ufunc defines 2 outputs, but 1 `nodata_output` values were provided."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        ufunc_meta.with_inputs(features, nodata_output=(1,))


def test_with_inputs_raises_on_empty_inputs():
    ufunc_meta = _UfuncMeta.from_outputs(
        [Output(dims=[Dimension(name="a", size=1)], dtype=np.float32, nodata=-1.0)]
    )

    expected_msg = "Ufuncs requires at least one feature array input."
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        ufunc_meta.with_inputs()


def test_with_inputs_raises_on_mismatched_feature_dim_names():
    ufunc_meta = _UfuncMeta.from_outputs(
        [Output(dims=[Dimension(name="a", size=1)], dtype=np.float32, nodata=-1.0)]
    )
    a = FeatureArray.from_feature_array(
        xr.DataArray(np.ones((2, 3, 4)), dims=["variable", "y", "x"])
    )
    b = FeatureArray.from_feature_array(
        xr.DataArray(np.ones((2, 3, 4)), dims=["band", "y", "x"])
    )

    with pytest.raises(
        ValueError,
        match="All input feature arrays must share the same feature dimension name",
    ):
        ufunc_meta.with_inputs(a, b)


def test_with_inputs_uses_highest_priority_postprocessor():
    ufunc_meta = _UfuncMeta.from_outputs(
        [Output(dims=[Dimension(name="a", size=1)], dtype=np.float32, nodata=-1.0)]
    )
    df_features = FeatureArray.from_feature_array(
        wrap_features(np.ones((2, 5)), type=pd.DataFrame)
    )
    da_features = FeatureArray.from_feature_array(
        wrap_features(np.ones((2, 5)), type=xr.DataArray)
    )
    ds_features = FeatureArray.from_feature_array(
        wrap_features(np.ones((2, 5)), type=xr.Dataset)
    )

    called_meta = ufunc_meta.with_inputs(df_features, da_features, ds_features)

    assert called_meta.postprocessor.__self__ is ds_features
