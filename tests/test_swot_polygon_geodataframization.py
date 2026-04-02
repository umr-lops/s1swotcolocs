import unittest
import numpy as np
import xarray as xr
from shapely.geometry import Polygon
import logging
from collections import defaultdict
from unittest.mock import patch

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import (
    treat_a_clean_piece_of_swot_orbit,
    is_nearly_collinear,
    is_degenerate_swath,
    _safe_fix_polygon,
)

app_logger = logging.getLogger("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS")
app_logger.addHandler(logging.NullHandler())
app_logger.propagate = False


class TestSwotPolygonHelpers(unittest.TestCase):
    def test_collinear_detection(self):
        """Test the svd-based collinearity check."""
        # Perfectly collinear points
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.assertTrue(is_nearly_collinear(points))
        # Non-collinear points
        points_box = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.assertFalse(is_nearly_collinear(points_box))

    def test_degenerate_swath(self):
        """Test the aspect ratio check."""
        # Very narrow lon range vs lat range
        points = np.array([[0.001, 0], [0.001, 1], [0.002, 2], [0.002, 3]])
        self.assertTrue(is_degenerate_swath(points, bbox_ratio_threshold=0.5))

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.fix_polygon")
    def test_safe_fix_polygon_value_error(self, mock_fix):
        """Test that _safe_fix_polygon catches the specific LinearRing ValueError."""
        mock_fix.side_effect = ValueError(
            "A linearring requires at least 4 coordinates"
        )
        cpt = defaultdict(int)
        poly = Polygon([(0, 0), (1, 1), (1, 0)])
        res, ok = _safe_fix_polygon(poly, cpt, context="unit test")
        self.assertFalse(ok)
        self.assertEqual(cpt["linearring_error_at_fix_polygon"], 1)


class TestTreatACleanPieceOfSwotOrbit(unittest.TestCase):
    def _create_dummy_onedsswot(self, times_data, lons_data, lats_data):
        onedsswot = xr.Dataset(
            {
                "longitude": (("num_lines", "num_pixels"), lons_data),
                "latitude": (("num_lines", "num_pixels"), lats_data),
                "time": (("num_lines",), times_data),
            }
        )
        onedsswot.encoding["source"] = "/fake/swot.nc"
        return onedsswot

    def test_nat_handling(self):
        """Test that NaT values in SWOT time result in skipped pieces."""
        swotpiece = Polygon([(10, 20), (11, 20), (11, 21), (10, 21)])
        lons = np.zeros((2, 3))
        lats = np.zeros((2, 3))
        # Set one time to NaT
        times_np = np.array([np.datetime64("NaT"), np.datetime64("2023-01-01")])

        onedsswot = self._create_dummy_onedsswot(times_np, lons, lats)
        points = np.column_stack((lons.ravel(), lats.ravel()))
        cpt = defaultdict(int)

        result_gdf, cpt = treat_a_clean_piece_of_swot_orbit(
            swotpiece, points, onedsswot, "IW", "SLC", np.timedelta64(1, "h"), cpt
        )

        self.assertTrue(result_gdf.empty)
        self.assertEqual(cpt["NaT_time_skipped"], 1)

    def test_basic_scenario(self):
        """Standard valid matchup scenario."""
        swotpiece = Polygon([(10, 20), (11, 21), (10, 21)])
        lons_data = np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])
        lats_data = np.array([[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]])
        times_np = np.array(
            [np.datetime64("2023-01-01T12:00:00"), np.datetime64("2023-01-01T12:10:00")]
        )

        onedsswot = self._create_dummy_onedsswot(times_np, lons_data, lats_data)
        points = np.column_stack((lons_data.ravel(), lats_data.ravel()))
        cpt = defaultdict(int)

        result_gdf, cpt = treat_a_clean_piece_of_swot_orbit(
            swotpiece, points, onedsswot, "IW", "SLC", np.timedelta64(1, "h"), cpt
        )

        self.assertFalse(result_gdf.empty)
        self.assertEqual(result_gdf["sensormode"].iloc[0], "IW")
        self.assertEqual(cpt["Ok_SWOT_time_values"], 1)


if __name__ == "__main__":
    unittest.main()
