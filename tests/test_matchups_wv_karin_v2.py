# tests/test_matchups_wv_karin_v2.py
"""Unit tests for matchups_WV_KaRIn_v2.py using pytest and mocks."""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

# Import the module under test
from s1swotcolocs import matchups_WV_KaRIn_v2 as mwv
from s1swotcolocs.utils import s1_unwrap_longitudes, _make_sub_polygon

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def mock_swot_filename():
    """Return a SWOT filename with parseable timestamps."""
    return "SWOT_L2_LR_SSH_WindWave_043_518_20260101T002513_20260101T011641_PID0_01.nc"


@pytest.fixture
def mock_swot_attrs():
    """Return a dictionary mimicking SWOT NetCDF global attributes."""
    return {
        "geospatial_lon_min": -180.0,
        "geospatial_lon_max": 180.0,
        "geospatial_lat_min": -90.0,
        "geospatial_lat_max": 90.0,
        "time_coverage_start": "2026-01-01T00:25:13.000000",
        "time_coverage_end": "2026-01-01T01:16:41.000000",
        "cycle_number": 43,
        "pass_number": 518,
        "crid": "01",
    }


# ----------------------------------------------------------------------
# Tests for parsing / helpers
# ----------------------------------------------------------------------


def test_parse_swot_filename_times(mock_swot_filename):
    t0, t1 = mwv.parse_swot_filename_times(mock_swot_filename)
    assert t0 == datetime(2026, 1, 1, 0, 25, 13, tzinfo=timezone.utc)
    assert t1 == datetime(2026, 1, 1, 1, 16, 41, tzinfo=timezone.utc)

    # Non-matching filename
    t0, t1 = mwv.parse_swot_filename_times("random.nc")
    assert t0 is None and t1 is None


def test__doy():
    d = date(2026, 1, 1)
    assert mwv._doy(d) == 1
    d = date(2026, 12, 31)
    assert mwv._doy(d) == 365


def test__swot_bbox_from_attrs(mock_swot_attrs):
    lon_min, lon_max, lat_min, lat_max = mwv._swot_bbox_from_attrs(mock_swot_attrs)
    assert lon_min == -180.0
    assert lon_max == 180.0
    assert lat_min == -90.0
    assert lat_max == 90.0

    # Missing keys -> defaults
    empty_attrs = {}
    lon_min, lon_max, lat_min, lat_max = mwv._swot_bbox_from_attrs(empty_attrs)
    assert lon_min == -180.0
    assert lon_max == 180.0
    assert lat_min == -90.0
    assert lat_max == 90.0


def test_robust_swot_time_from_attrs(mock_swot_attrs):
    t0, t1 = mwv.robust_swot_time_from_attrs(mock_swot_attrs)
    assert t0 == datetime(2026, 1, 1, 0, 25, 13, 0, tzinfo=timezone.utc)
    assert t1 == datetime(2026, 1, 1, 1, 16, 41, 0, tzinfo=timezone.utc)

    # With Z suffix
    attrs_z = mock_swot_attrs.copy()
    attrs_z["time_coverage_start"] = "2026-01-01T00:25:13.000000Z"
    attrs_z["time_coverage_end"] = "2026-01-01T01:16:41.000000Z"
    t0, t1 = mwv.robust_swot_time_from_attrs(attrs_z)
    assert t0 == datetime(2026, 1, 1, 0, 25, 13, 0, tzinfo=timezone.utc)


def test__bbox_overlaps():
    # Fully overlapping
    assert mwv._bbox_overlaps(-10, 10, -10, 10, -5, 5, -5, 5) is True
    # Non-overlapping
    assert mwv._bbox_overlaps(-10, 10, -10, 10, 20, 30, 20, 30) is False
    # Cross antimeridian (should return True)
    assert mwv._bbox_overlaps(170, -170, -10, 10, 175, -175, -5, 5) is True


def test__unwrap_longitudes():
    lons = np.array([179, 180, -179, -178])
    unwrapped = s1_unwrap_longitudes(lons)
    expected = np.array([179, 180, 181, 182])  # unwrapped across antimeridian
    np.testing.assert_allclose(unwrapped, expected)


def test__make_sub_polygon():
    # Valid polygon
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = _make_sub_polygon(coords)
    assert poly is not None
    assert poly.geom_type == "Polygon"
    assert poly.is_valid

    # Too few points
    coords = [(0, 0), (1, 1)]
    poly = _make_sub_polygon(coords)
    assert poly is None

    # With NaN values
    coords = [(0, 0), (np.nan, np.nan), (1, 1), (0, 1)]
    poly = _make_sub_polygon(coords)
    assert poly is not None  # should drop the NaN


# ----------------------------------------------------------------------
# Tests for S1-WV helpers
# ----------------------------------------------------------------------


@patch("xarray.open_dataset")
def test_load_s1_wv(mock_open_dataset):
    mock_ds = MagicMock()
    # Create a mock for fdatedt that has both .values and .size
    mock_fdatedt = MagicMock()
    mock_fdatedt.values = np.array([pd.Timestamp("2026-01-01T00:00:00")])
    mock_fdatedt.size = 1

    # Set the attribute directly (for ds.fdatedt.size and ds.fdatedt.values)
    mock_ds.fdatedt = mock_fdatedt

    # Mock __getitem__ to return the same mock for 'fdatedt' (for ds["fdatedt"])
    def getitem_side_effect(key):
        if key == "fdatedt":
            return mock_fdatedt
        elif key == "lon":
            mock_lon = MagicMock()
            mock_lon.values = np.array([10.0])
            return mock_lon
        elif key == "lat":
            mock_lat = MagicMock()
            mock_lat.values = np.array([20.0])
            return mock_lat
        elif key == "wv_mode":
            mock_wv = MagicMock()
            mock_wv.values = np.array(["WV1"])
            return mock_wv
        elif key == "sensor":
            mock_sensor = MagicMock()
            mock_sensor.values = np.array(["s1a"])
            return mock_sensor
        elif key == "subpath":
            mock_subpath = MagicMock()
            mock_subpath.values = np.array(["subpath1"])
            return mock_subpath
        elif key == "polygon":
            mock_poly = MagicMock()
            poly = np.array(
                [[[10.0, 10.1, 10.2, 10.1, 10.0], [20.0, 20.1, 20.2, 20.1, 20.0]]]
            )
            mock_poly.values = poly
            return mock_poly
        else:
            return MagicMock()

    mock_ds.__getitem__.side_effect = getitem_side_effect
    mock_open_dataset.return_value = mock_ds

    path = Path("/fake/path.nc")
    df = mwv.load_s1_wv(path)
    assert len(df) == 1
    assert df.iloc[0]["time"] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    assert df.iloc[0]["lon"] == 10.0
    assert df.iloc[0]["lat"] == 20.0
    assert df.iloc[0]["sensor"] == "s1a"
    assert df.iloc[0]["path"] == path
    assert "polygon_lons" in df.columns
    mock_open_dataset.assert_called_once_with(path)


def test_s1_scene_polygon():
    # Create a sample row with a square polygon
    row = pd.Series(
        {
            "polygon_lons": [10, 10.1, 10.2, 10.1, 10],
            "polygon_lats": [20, 20.1, 20.2, 20.1, 20],
        }
    )
    poly = mwv.s1_scene_polygon(row, swot_lon_max=180)
    assert poly.geom_type == "Polygon"
    assert poly.is_valid

    # Test antimeridian handling: swot_lon_max >= 180
    # Create a polygon crossing Greenwich (lon values around 179 and -179)
    row = pd.Series(
        {
            "polygon_lons": [179.9, -179.9, -179.8, 179.8, 179.9],
            "polygon_lats": [0, 0.1, 0.2, 0.1, 0],
        }
    )
    poly = mwv.s1_scene_polygon(row, swot_lon_max=200)
    # Should not raise, and the polygon should be valid (we can't easily assert correct wrap)
    assert poly.is_valid


def test_overlap_pct():
    # Create a large square and a smaller one inside
    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    small = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    intersection = big.intersection(small)
    pct = mwv.overlap_pct(intersection, big)
    # small area = 36, big area = 100 -> 36%
    assert round(pct, 1) == 36.0

    # No overlap
    other = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    inter = big.intersection(other)
    pct = mwv.overlap_pct(inter, big)
    assert pct == 0.0


# ----------------------------------------------------------------------
# Tests for file discovery
# ----------------------------------------------------------------------


@patch("pathlib.Path.glob")
def test_find_s1_wv_files_for_swot(mock_glob):
    # Mock glob to return some files
    mock_glob.return_value = [
        Path("/dummy/S1A_WV_L2D_enriched_LOPS_20260101_daily_IPF_004.02.nc")
    ]
    s1_root = Path("/fake/s1")
    swot_t0 = datetime(2026, 1, 1, 0, 25, 13, tzinfo=timezone.utc)
    swot_t1 = datetime(2026, 1, 1, 1, 16, 41, tzinfo=timezone.utc)
    found = mwv.find_s1_wv_files_for_swot(swot_t0, swot_t1, s1_root)
    # Should return one file (the glob returns one) because we pass it directly
    assert len(found) == 1
    assert found[0].name == "S1A_WV_L2D_enriched_LOPS_20260101_daily_IPF_004.02.nc"

    # If glob returns empty list, should return []
    mock_glob.return_value = []
    found = mwv.find_s1_wv_files_for_swot(swot_t0, swot_t1, s1_root)
    assert found == []


@patch("pathlib.Path.rglob")
def test_find_swot_files(mock_rglob):
    mock_rglob.return_value = [
        Path("/fake/SWOT_L2_LR_SSH_001.nc"),
        Path("/fake/SWOT_L2_LR_SSH_002.nc"),
    ]
    root = Path("/fake")
    files = mwv.find_swot_files(root)
    assert len(files) == 2
    mock_rglob.assert_called_once_with("SWOT_L2_LR_SSH_*.nc")


# ----------------------------------------------------------------------
# Integration-like test for collocate_swot_file with mocks
# ----------------------------------------------------------------------


@patch("s1swotcolocs.matchups_WV_KaRIn_v2.s1_scene_polygon")
@patch("s1swotcolocs.matchups_WV_KaRIn_v2.swot_footprint_polygon")
@patch("s1swotcolocs.matchups_WV_KaRIn_v2.extract_swot_edges")
@patch("s1swotcolocs.matchups_WV_KaRIn_v2.load_s1_wv")
@patch("s1swotcolocs.matchups_WV_KaRIn_v2.find_s1_wv_files_for_swot")
@patch("xarray.open_dataset")
def test_collocate_swot_file(
    mock_open_dataset,
    mock_find_s1_wv_files,
    mock_load_s1_wv,
    mock_extract_swot_edges,
    mock_swot_footprint_polygon,
    mock_s1_scene_polygon,
    mock_swot_filename,
    mock_swot_attrs,
    tmp_path,
):
    # Mock S1-WV file finding
    mock_find_s1_wv_files.return_value = [Path("/fake/s1.nc")]

    # Mock load_s1_wv to return a DataFrame with two scenes
    df_wv = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2026-01-01T00:45:00", tz="UTC"),
                pd.Timestamp("2026-01-01T01:10:00", tz="UTC"),
            ],
            "lon": [10.0, 20.0],
            "lat": [30.0, 40.0],
            "wv_mode": ["WV1", "WV2"],
            "sensor": ["s1a", "s1c"],
            "subpath": ["sub1", "sub2"],
            "path": [Path("/fake/s1.nc"), Path("/fake/s1.nc")],
            "polygon_lons": [[10, 10.1, 10.2, 10.1, 10], [20, 20.1, 20.2, 20.1, 20]],
            "polygon_lats": [[30, 30.1, 30.2, 30.1, 30], [40, 40.1, 40.2, 40.1, 40]],
        }
    )
    mock_load_s1_wv.return_value = df_wv

    # Mock SWOT dataset (only attributes needed)
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.attrs = mock_swot_attrs.copy()
    mock_open_dataset.return_value = mock_ds

    # Mock extract_swot_edges to return a predefined edges dict
    # Generate enough lines to cover both S1 times
    n_lines = 200
    start_time = pd.Timestamp("2026-01-01T00:25:13", tz="UTC")
    times = pd.date_range(start=start_time, periods=n_lines, freq="30s").values
    lons = (
        np.linspace(-170, 170, n_lines).reshape(-1, 1) + np.array([0, 0.5, 0.5, 1]) * 5
    )
    lats = np.linspace(80, -80, n_lines).reshape(-1, 1) + np.array([0, 0.5, 0.5, 1]) * 2
    edges = {
        "lons": lons,
        "lats": lats,
        "valid_rows": np.ones(n_lines, dtype=bool),
        "times": times,
        "lon_min": -175.0,
        "lon_max": 175.0,
        "lat_min": -85.0,
        "lat_max": 85.0,
    }
    mock_extract_swot_edges.return_value = edges

    # Mock swot_footprint_polygon to return a huge polygon
    mock_swot_poly = Polygon([(-180, -90), (180, -90), (180, 90), (-180, 90)])
    mock_swot_footprint_polygon.return_value = mock_swot_poly

    # Mock s1_scene_polygon to return a polygon that intersects the SWOT polygon
    def mock_s1_polygon(row, swot_lon_max):
        # Create a small square around the S1 point
        lon = row["lon"]
        lat = row["lat"]
        return Polygon(
            [
                (lon - 0.5, lat - 0.5),
                (lon + 0.5, lat - 0.5),
                (lon + 0.5, lat + 0.5),
                (lon - 0.5, lat + 0.5),
            ]
        )

    mock_s1_scene_polygon.side_effect = mock_s1_polygon

    # Run collocation with a large time margin to guarantee time filter passes
    swot_path = Path("/fake") / mock_swot_filename
    s1_root = Path("/fake/s1")
    output_dir = tmp_path / "output"
    matchups, suffix = mwv.collocate_swot_file(
        swot_path,
        s1_root,
        output_dir,
        max_time_diff_min=1000,  # large enough
        debug_image=False,
    )

    # Expect two matchups
    assert len(matchups) == 2
    # Check JSON files are created
    json_files = list(output_dir.rglob("*.json"))
    assert len(json_files) == 2
    # Check matchup structure
    for m in matchups:
        assert "id" in m
        assert "properties" in m
        assert "assets" in m
        assert "s1_wv" in m["assets"]
        assert "swot_karin" in m["assets"]

    # Verify mocks were called
    mock_swot_footprint_polygon.assert_called_once()
    mock_extract_swot_edges.assert_called_once()
    assert mock_s1_scene_polygon.call_count == 2


# ----------------------------------------------------------------------
# Test for matchups_to_dataframe
# ----------------------------------------------------------------------


def test_matchups_to_dataframe():
    # Create a minimal matchup item
    item = {
        "id": "test_id",
        "properties": {
            "s1_time": "2026-01-01T00:45:00+00:00",
            "swot_time_at_coloc": "2026-01-01T00:50:00+00:00",
            "time_diff_seconds": 300.0,
            "overlap_pct": 50.0,
            "s1_lon": 10.0,
            "s1_lat": 20.0,
            "s1_wv_mode": "WV1",
            "s1_sensor": "s1a",
            "swot_cycle": 43,
            "swot_pass": 518,
            "swot_crid": "01",
            "s1_subpath": "sub1",
            "s1_path": "/fake/s1.nc",
        },
        "assets": {"swot_karin": {"href": "/fake/swot.nc"}},
    }
    df = mwv.matchups_to_dataframe([item])
    assert len(df) == 1
    assert df.iloc[0]["id"] == "test_id"
    assert df.iloc[0]["s1_lon"] == 10.0
    assert df.iloc[0]["swot_path"] == "/fake/swot.nc"
