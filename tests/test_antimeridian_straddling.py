"""
Regression tests for GH issue #20: SWOT polygons corrupted.

When a SWOT orbit crosses the antimeridian, `slice_swot` used to shape the
whole point set (alphashape / convex hull) while points existed on both
sides of 180E/W. The resulting "banana" polygon spans ~360 degrees of
longitude; after the land mask, fragments far from the real swath were
saved as SWOT polygons and produced bogus colocations (e.g. a 127 deg²
polygon over the Indian Ocean from a swath at the antimeridian, see
https://github.com/umr-lops/s1swotcolocs/issues/20).
"""

import collections
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import (
    slice_swot,
    split_points_at_antimeridian,
)


class TestSplitPointsAtAntimeridian(unittest.TestCase):
    def test_no_straddling_returns_single_set(self):
        points = np.array(
            [
                [100.0, 10.0],
                [100.5, 10.0],
                [100.0, 11.0],
                [100.5, 11.0],
            ]
        )
        result = split_points_at_antimeridian(points)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], points)

    def test_straddling_returns_two_halves(self):
        eastern = np.array([[178.0, 10.0], [179.0, 10.0], [178.5, 11.0], [179.5, 11.0]])
        western = np.array(
            [[-179.0, 10.0], [-178.0, 10.0], [-179.5, 11.0], [-178.5, 11.0]]
        )
        points = np.concatenate([eastern, western])
        result = split_points_at_antimeridian(points)
        self.assertEqual(len(result), 2)
        for part in result:
            # each half must be confined to one side of the antimeridian
            self.assertLess(part[:, 0].max() - part[:, 0].min(), 180.0)
        all_lons = np.concatenate([part[:, 0] for part in result])
        np.testing.assert_allclose(sorted(all_lons), sorted(points[:, 0]))

    def test_all_points_same_side(self):
        points = np.array(
            [[-179.0, 10.0], [-178.0, 10.0], [-179.5, 11.0], [-178.5, 11.0]]
        )
        result = split_points_at_antimeridian(points)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], points)

    def test_one_shapable_half_kept(self):
        # eastern side has too few points to be shaped, western side is fine
        points = np.array(
            [
                [179.9, 0.0],
                [179.95, 1.0],
                [179.97, 2.0],
                [-179.0, 10.0],
                [-178.0, 10.0],
                [-179.5, 11.0],
                [-178.5, 11.0],
                [-178.2, 12.0],
            ]
        )
        result = split_points_at_antimeridian(points)
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0][:, 0].max(), 0.0)  # only the western half

    def test_no_shapable_half_falls_back(self):
        # both halves too small -> keep the original set (legacy behaviour)
        points = np.array(
            [
                [179.9, 0.0],
                [179.95, 1.0],
                [-179.95, 10.0],
                [-179.9, 11.0],
            ]
        )
        result = split_points_at_antimeridian(points)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], points)


def _make_swot_ds(straddling: bool) -> xr.Dataset:
    """
    Build a synthetic SWOT L2-like dataset: a thin N-S swath, 100 lines,
    5 cross-track pixels per line.

    - straddling=True:  the swath runs from 178E to 183.75E (0-360
      representation), i.e. it crosses the antimeridian;
    - straddling=False: the same swath placed at 100-105E (no crossing).
    """
    num_lines = 100
    num_pixels = 5
    lats = (20.0 + 0.15 * np.arange(num_lines))[:, None] * np.ones((1, num_pixels))
    if straddling:
        base_lon = 178.0 + 0.05 * np.arange(num_lines)
    else:
        base_lon = 100.0 + 0.02 * np.arange(num_lines)
    lons = base_lon[:, None] + 0.2 * np.arange(num_pixels)[None, :]

    times = np.datetime64("2026-02-18T02:32:55", "ns") + np.arange(
        num_lines
    ) * np.timedelta64(1, "s")
    ds = xr.Dataset(
        {
            "longitude": (("num_lines", "num_pixels"), lons),
            "latitude": (("num_lines", "num_pixels"), lats),
            "time": (("num_lines",), times),
        }
    )
    ds.encoding = {"source": "SWOT_L2_LR_SSH_WindWave_999_999_TEST.nc"}
    return ds


# a small "island" far from the test swath: makes the land mask non-empty
# without eating any part of the swath
_TEST_LAND = gpd.GeoDataFrame(
    geometry=[Polygon([(10.0, 45.0), (11.0, 45.0), (11.0, 46.0), (10.0, 46.0)])]
)


class TestSliceSwotAntimeridian(unittest.TestCase):
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.geodatasets.get_path")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.gpd.read_file")
    def test_straddling_swath_shaped_per_half(self, mock_read_file, mock_get_path):
        mock_get_path.return_value = "unused"
        mock_read_file.return_value = _TEST_LAND

        ds = _make_swot_ds(straddling=True)
        cpt = collections.defaultdict(int)
        gdfs, cpt = slice_swot(
            ds, idxstart=0, idxstop=100, cpt=cpt, max_area_size=200, delta_hours=1
        )

        # the segment must have been detected as crossing the antimeridian
        self.assertEqual(cpt["segment_split_at_antimeridian"], 1)
        # one piece per contiguous half, near the real swath
        self.assertEqual(len(gdfs), 2)

        east_seen = west_seen = False
        for gdf in gdfs:
            geom = gdf.geometry.iloc[0]
            self.assertTrue(geom.is_valid)
            lons = np.array(geom.exterior.coords.xy[0])
            # no polygon may span a big chunk of the globe (the buggy
            # "banana" spanned ~356 degrees of longitude)
            self.assertLess(lons.max() - lons.min(), 30.0)
            # every vertex must stay in the zone of the real swath
            self.assertTrue(np.all(np.abs(lons) > 170.0))
            east_seen |= lons.max() > 0
            west_seen |= lons.min() < 0
            # the time window must come from this segment only
            self.assertGreaterEqual(
                gdf["start_datetime"].iloc[0], pd.Timestamp("2026-02-17")
            )
            self.assertLessEqual(
                gdf["end_datetime"].iloc[0], pd.Timestamp("2026-02-19")
            )

        self.assertTrue(east_seen, "expected a piece on the eastern side")
        self.assertTrue(west_seen, "expected a piece on the western side")

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.geodatasets.get_path")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.gpd.read_file")
    def test_non_straddling_swath_unchanged(self, mock_read_file, mock_get_path):
        mock_get_path.return_value = "unused"
        mock_read_file.return_value = _TEST_LAND

        ds = _make_swot_ds(straddling=False)
        cpt = collections.defaultdict(int)
        gdfs, cpt = slice_swot(
            ds, idxstart=0, idxstop=100, cpt=cpt, max_area_size=200, delta_hours=1
        )

        self.assertEqual(cpt["segment_split_at_antimeridian"], 0)
        self.assertEqual(len(gdfs), 1)
        geom = gdfs[0].geometry.iloc[0]
        self.assertTrue(geom.is_valid)
        lons = np.array(geom.exterior.coords.xy[0])
        self.assertGreater(lons.min(), 95.0)
        self.assertLess(lons.max(), 110.0)


if __name__ == "__main__":
    unittest.main()
