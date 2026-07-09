# tests/test_utils_methods.py
"""Unit tests for s1swotcolocs.utils."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from s1swotcolocs import utils

# ----------------------------------------------------------------------
# Tests for get_conf_content
# ----------------------------------------------------------------------


@patch("builtins.open", new_callable=mock_open, read_data="key: value")
def test_get_conf_content(mock_file):
    conf = utils.get_conf_content("/fake/path.yml")
    assert conf == {"key": "value"}
    mock_file.assert_called_once_with("/fake/path.yml", "r")


# ----------------------------------------------------------------------
# Tests for parse_swot_filename_times
# ----------------------------------------------------------------------


def test_parse_swot_filename_times():
    fname = "SWOT_L2_LR_SSH_WindWave_043_518_20260101T002513_20260101T011641_PID0_01.nc"
    t0, t1 = utils.parse_swot_filename_times(fname)
    assert t0 == datetime(2026, 1, 1, 0, 25, 13, tzinfo=timezone.utc)
    assert t1 == datetime(2026, 1, 1, 1, 16, 41, tzinfo=timezone.utc)

    # Non-matching
    t0, t1 = utils.parse_swot_filename_times("random.nc")
    assert t0 is None and t1 is None


# ----------------------------------------------------------------------
# Tests for s1_unwrap_longitudes
# ----------------------------------------------------------------------


def test_s1_unwrap_longitudes():
    lons = np.array([179, 180, -179, -178])
    unwrapped = utils.s1_unwrap_longitudes(lons)
    expected = np.array([179, 180, 181, 182])
    np.testing.assert_allclose(unwrapped, expected)


# ----------------------------------------------------------------------
# Tests for _make_sub_polygon
# ----------------------------------------------------------------------


def test_make_sub_polygon():
    # Valid polygon
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = utils._make_sub_polygon(coords)
    assert poly is not None
    assert poly.geom_type == "Polygon"
    assert poly.is_valid

    # Too few points
    coords = [(0, 0), (1, 1)]
    poly = utils._make_sub_polygon(coords)
    assert poly is None

    # With NaN values
    coords = [(0, 0), (np.nan, np.nan), (1, 1), (0, 1)]
    poly = utils._make_sub_polygon(coords)
    assert poly is not None  # NaN dropped


# ----------------------------------------------------------------------
# Tests for _bbox_overlaps
# ----------------------------------------------------------------------


def test_bbox_overlaps():
    # Overlapping
    assert utils._bbox_overlaps(-10, 10, -10, 10, -5, 5, -5, 5) is True
    # Non-overlapping
    assert utils._bbox_overlaps(-10, 10, -10, 10, 20, 30, 20, 30) is False
    # Cross antimeridian
    assert utils._bbox_overlaps(170, -170, -10, 10, 175, -175, -5, 5) is True


# ----------------------------------------------------------------------
# Tests for robust_swot_time_from_attrs
# ----------------------------------------------------------------------


def test_robust_swot_time_from_attrs():
    attrs = {
        "time_coverage_start": "2026-01-01T00:25:13.000000",
        "time_coverage_end": "2026-01-01T01:16:41.000000",
    }
    t0, t1 = utils.robust_swot_time_from_attrs(attrs)
    assert t0 == datetime(2026, 1, 1, 0, 25, 13, 0, tzinfo=timezone.utc)
    assert t1 == datetime(2026, 1, 1, 1, 16, 41, 0, tzinfo=timezone.utc)

    # With Z suffix
    attrs_z = attrs.copy()
    attrs_z["time_coverage_start"] = "2026-01-01T00:25:13.000000Z"
    attrs_z["time_coverage_end"] = "2026-01-01T01:16:41.000000Z"
    t0, t1 = utils.robust_swot_time_from_attrs(attrs_z)
    assert t0 == datetime(2026, 1, 1, 0, 25, 13, 0, tzinfo=timezone.utc)

    # Missing key
    with pytest.raises(KeyError):
        utils.robust_swot_time_from_attrs({})


# ----------------------------------------------------------------------
# Tests for overlap_pct
# ----------------------------------------------------------------------


def test_overlap_pct():
    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    small = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    intersection = big.intersection(small)
    pct = utils.overlap_pct(intersection, big)
    # small area = 36, big area = 100 -> 36%
    assert round(pct, 1) == 36.0

    # No overlap
    other = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    inter = big.intersection(other)
    pct = utils.overlap_pct(inter, big)
    assert pct == 0.0

    # Zero area polygon
    pct = utils.overlap_pct(Polygon(), Polygon())
    assert pct == 0.0


# ----------------------------------------------------------------------
# Tests for extract_swot_edges
# ----------------------------------------------------------------------


def test_extract_swot_edges():
    # Create a mock xarray dataset
    mock_ds = MagicMock(spec=xr.Dataset)
    n_lines = 10
    n_pix = 69
    # Longitudes in [0, 360]
    lons = np.linspace(0, 360, n_lines).reshape(-1, 1) + np.linspace(
        0, 10, n_pix
    ).reshape(1, -1)
    lats = (
        np.linspace(80, -80, n_lines).reshape(-1, 1)
        + np.linspace(0, 2, n_pix).reshape(1, -1) * 0.3
    )
    times = pd.date_range("2026-01-01T00:25:13", periods=n_lines, freq="30s").values

    # Set up mocks for isel
    def isel_side_effect(**kwargs):
        if "num_pixels" in kwargs:
            idx = kwargs["num_pixels"]
            mock_arr = MagicMock()
            if "num_lines" in kwargs:
                line = kwargs["num_lines"]
                if line < 0:
                    line = n_lines + line
                vals = (
                    lons[line, idx] if "longitude" in str(kwargs) else lats[line, idx]
                )
            else:
                vals = lons[:, idx] if "longitude" in str(kwargs) else lats[:, idx]
            mock_arr.values = vals
            return mock_arr
        return MagicMock()

    # We need separate mocks for longitude and latitude
    mock_ds.longitude = MagicMock()
    mock_ds.longitude.isel.side_effect = lambda **kw: isel_side_effect(
        **{**kw, "longitude": True}
    )
    mock_ds.latitude = MagicMock()
    mock_ds.latitude.isel.side_effect = lambda **kw: isel_side_effect(
        **{**kw, "latitude": True}
    )
    mock_ds.time = MagicMock()
    mock_ds.time.values = times
    mock_ds.sizes = {"num_lines": n_lines, "num_pixels": n_pix}

    edges = utils.extract_swot_edges(mock_ds)
    assert "lons" in edges
    assert "lats" in edges
    assert "valid_rows" in edges
    assert "times" in edges
    assert edges["lons"].shape == (n_lines, 4)  # 4 edge columns
    assert edges["lats"].shape == (n_lines, 4)
    assert len(edges["valid_rows"]) == n_lines
    assert len(edges["times"]) == n_lines


# ----------------------------------------------------------------------
# Tests for swot_footprint_polygon
# ----------------------------------------------------------------------


def test_swot_footprint_polygon():
    # Build a simple edge dict with 4 columns
    n_lines = 150
    lons = (
        np.linspace(-170, 170, n_lines).reshape(-1, 1) + np.array([0, 0.5, 0.5, 1]) * 5
    )
    lats = np.linspace(80, -80, n_lines).reshape(-1, 1) + np.array([0, 0.5, 0.5, 1]) * 2
    edges = {
        "lons": lons,
        "lats": lats,
        "valid_rows": np.ones(n_lines, dtype=bool),
        "times": np.arange(n_lines),
        "lon_min": -175,
        "lon_max": 175,
        "lat_min": -85,
        "lat_max": 85,
    }
    poly = utils.swot_footprint_polygon(edges)
    assert poly is not None
    assert poly.geom_type in ("Polygon", "MultiPolygon")
    assert poly.is_valid

    # Too few valid rows -> ValueError
    edges_bad = edges.copy()
    edges_bad["valid_rows"] = np.zeros(n_lines, dtype=bool)
    with pytest.raises(ValueError, match="Too few valid edge rows"):
        utils.swot_footprint_polygon(edges_bad)


# ----------------------------------------------------------------------
# Tests for NumpyEncoder
# ----------------------------------------------------------------------


def test_numpy_encoder():
    encoder = utils.NumpyEncoder()
    # numpy integer
    assert encoder.default(np.int64(5)) == 5
    # numpy float - use approx due to precision
    assert encoder.default(np.float32(3.14)) == pytest.approx(3.14, rel=1e-6)
    # numpy array
    arr = np.array([1, 2, 3])
    assert encoder.default(arr) == [1, 2, 3]

    # Full serialization with mixed types including string
    data = {
        "a": np.int64(10),
        "b": np.float32(1.5),
        "c": np.array([4, 5, 6]),
        "d": "hello",
    }
    json_str = json.dumps(data, cls=utils.NumpyEncoder)
    assert json.loads(json_str) == {"a": 10, "b": 1.5, "c": [4, 5, 6], "d": "hello"}
