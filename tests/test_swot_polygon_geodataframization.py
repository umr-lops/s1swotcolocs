import unittest
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
import geopandas as gpd
import logging
from collections import defaultdict

# Assuming the function is in this path, adjust if necessary
from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import (
    treat_a_clean_piece_of_swot_orbit,
)

# Suppress logger output during tests for the specific logger used in the function
app_logger = logging.getLogger("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS")
if (
    not app_logger.handlers
):  # Avoid adding multiple NullHandlers if tests are run multiple times
    app_logger.addHandler(logging.NullHandler())
app_logger.propagate = False


class TestTreatACleanPieceOfSwotOrbit(unittest.TestCase):

    def _create_dummy_onedsswot(
        self, times_data, lons_data, lats_data, source_filename
    ):
        """Helper to create a dummy xarray.Dataset for SWOT data."""
        onedsswot = xr.Dataset(
            {
                "longitude": (("num_lines", "num_pixels"), lons_data),
                "latitude": (("num_lines", "num_pixels"), lats_data),
                "time": (("num_lines",), times_data),
            },
            coords={
                "num_lines": np.arange(len(times_data)),
                "num_pixels": np.arange(lons_data.shape[1]),
            },
        )
        onedsswot.encoding["source"] = source_filename
        return onedsswot

    def test_basic_scenario(self):
        """Test with time_north > time_south."""
        swotpiece = Polygon([(10, 20), (11, 20), (11, 21), (10, 21)])

        # num_lines=2, num_pixels=3.
        # Point (11, 21) (lonmax, latmax) should map to num_lines=1
        # Point (10, 20) (lonmin, latmin) should map to num_lines=0
        lons_data = np.array([[10.0, 10.5, 10.8], [10.2, 10.7, 11.0]])
        lats_data = np.array([[20.0, 20.0, 20.2], [20.8, 20.9, 21.0]])
        times_np = np.array(
            [
                np.datetime64("2023-01-01T12:00:00"),  # time_south (num_lines=0)
                np.datetime64("2023-01-01T12:05:00"),  # time_north (num_lines=1)
            ]
        )

        onedsswot = self._create_dummy_onedsswot(
            times_np, lons_data, lats_data, "/fake/swot_file1.nc"
        )
        points_for_kdtree = np.column_stack((lons_data.ravel(), lats_data.ravel()))

        mode = "IW"
        producttype = "SLC"
        delta_t_max = np.timedelta64(1, "h")
        cpt = defaultdict(int)
        result_gdf,cpt = treat_a_clean_piece_of_swot_orbit(
            swotpiece, points_for_kdtree, onedsswot, mode, producttype, delta_t_max,
            cpt=cpt
        )

        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        self.assertEqual(len(result_gdf), 1)

        expected_start_swot_dt = pd.to_datetime(times_np[0])
        expected_stop_swot_dt = pd.to_datetime(times_np[1])

        expected_sta = (
            (expected_start_swot_dt - pd.to_timedelta(delta_t_max))
            .round("us")
            .to_pydatetime()
        )
        expected_sto = (
            (expected_stop_swot_dt + pd.to_timedelta(delta_t_max))
            .round("us")
            .to_pydatetime()
        )

        self.assertEqual(result_gdf["start_datetime"].iloc[0], expected_sta)
        self.assertEqual(result_gdf["end_datetime"].iloc[0], expected_sto)
        self.assertTrue(result_gdf["geometry"].iloc[0].equals(swotpiece))
        self.assertEqual(result_gdf["collection"].iloc[0], "SENTINEL-1")
        self.assertEqual(result_gdf["sensormode"].iloc[0], mode)
        self.assertEqual(result_gdf["producttype"].iloc[0], producttype)
        expected_id = f"SWOT swot_file1.nc {times_np[0]} {times_np[1]}"
        self.assertEqual(result_gdf["id_query"].iloc[0], expected_id)

    def test_time_reversed_scenario(self):
        """Test with time_north < time_south (forcing internal sort)."""
        swotpiece = Polygon([(10, 20), (11, 20), (11, 21), (10, 21)])

        lons_data = np.array(
            [[10.0, 10.5, 10.8], [10.2, 10.7, 11.0]]
        )  # lonmax=11, latmax=21 -> num_lines=1
        lats_data = np.array(
            [[20.0, 20.0, 20.2], [20.8, 20.9, 21.0]]
        )  # lonmin=10, latmin=20 -> num_lines=0

        # time_north (num_lines=1) is earlier than time_south (num_lines=0)
        times_np = np.array(
            [
                np.datetime64("2023-01-01T12:05:00"),  # time_south_actual (num_lines=0)
                np.datetime64("2023-01-01T12:00:00"),  # time_north_actual (num_lines=1)
            ]
        )

        onedsswot = self._create_dummy_onedsswot(
            times_np, lons_data, lats_data, "another_swot.nc"
        )  # No path for basename test
        points_for_kdtree = np.column_stack((lons_data.ravel(), lats_data.ravel()))

        mode = "EW"
        producttype = "GRD"
        delta_t_max = np.timedelta64(30, "m")  # 30 minutes
        cpt = defaultdict(int)
        result_gdf,cpt = treat_a_clean_piece_of_swot_orbit(
            swotpiece, points_for_kdtree, onedsswot, mode, producttype, delta_t_max,
            cpt=cpt
        )

        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        self.assertEqual(len(result_gdf), 1)

        # The function sorts these: startswot will be times_np[1], stopswot will be times_np[0]
        expected_start_swot_dt = pd.to_datetime(times_np[1])  # The earlier time
        expected_stop_swot_dt = pd.to_datetime(times_np[0])  # The later time

        expected_sta = (
            (expected_start_swot_dt - pd.to_timedelta(delta_t_max))
            .round("us")
            .to_pydatetime()
        )
        expected_sto = (
            (expected_stop_swot_dt + pd.to_timedelta(delta_t_max))
            .round("us")
            .to_pydatetime()
        )

        self.assertEqual(result_gdf["start_datetime"].iloc[0], expected_sta)
        self.assertEqual(result_gdf["end_datetime"].iloc[0], expected_sto)
        self.assertTrue(result_gdf["geometry"].iloc[0].equals(swotpiece))

        # id_query uses the sorted startswot and stopswot values
        expected_id = f"SWOT another_swot.nc {times_np[1]} {times_np[0]}"
        self.assertEqual(result_gdf["id_query"].iloc[0], expected_id)


if __name__ == "__main__":
    unittest.main()
