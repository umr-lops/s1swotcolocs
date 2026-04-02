import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from unittest.mock import patch, MagicMock
from collections import defaultdict
from shapely.geometry import Polygon

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import (
    get_swot_date_info,
    save_netcdf_file_per_swot_piece_orbit_core,
    do_cdse_query,
    compute_alphashape_safe,
    treat_one_day_wrapper,
)


class TestColocPersistenceAndWrapper(unittest.TestCase):

    def test_get_swot_date_info(self):
        """Test date string and components extraction."""
        dt = np.datetime64("2025-10-17T15:12:10")
        fmt, y, m, d = get_swot_date_info(dt)
        self.assertEqual(fmt, "20251017T151210")
        self.assertEqual(y, 2025)
        self.assertEqual(m, "10")
        self.assertEqual(d, "17")

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.xr.Dataset.to_netcdf")
    @patch("os.chmod")
    @patch("os.makedirs")
    def test_save_netcdf_core(self, mock_mkdir, mock_chmod, mock_to_nc):
        """Test the NetCDF creation from CDSE results."""
        # Mock CDSE query output
        cdse_output = pd.DataFrame(
            {
                "Name": ["S1A_IW_SLC__1", "S1A_IW_SLC__2"],
                "geometry": [
                    Polygon([(0, 0), (1, 1), (1, 0)]),
                    Polygon([(0, 0), (1, 1), (1, 0)]),
                ],
                "ContentDate": [
                    {"Start": "2025-10-17T15:00:00Z"},
                    {"Start": "2025-10-17T15:10:00Z"},
                ],
                "id_original_query": [
                    "SWOT file.nc 15:00 15:10",
                    "SWOT file.nc 15:00 15:10",
                ],
            }
        )

        # Mock SWOT piece info
        swot_gdf = pd.DataFrame(
            {
                "id_query": ["SWOT file.nc 2025-10-17T15:12:10"],
                "geometry": [Polygon([(0, 0), (5, 5), (5, 0)])],
            }
        )

        cpt = defaultdict(int)
        updated_cpt = save_netcdf_file_per_swot_piece_orbit_core(
            cdse_output, swot_gdf, "test_out.nc", 6, cpt
        )

        self.assertTrue(mock_to_nc.called)
        self.assertEqual(updated_cpt["new_file"], 1)

    @patch("cdsodatacli.query.fetch_data")
    def test_do_cdse_query_nat_guard(self, mock_fetch):
        """Ensure query returns None if dates are NaT."""
        gdf_nat = gpd.GeoDataFrame(
            {"start_datetime": [pd.NaT], "end_datetime": [pd.NaT], "id_query": ["test"]}
        )
        res = do_cdse_query(gdf_nat)
        self.assertIsNone(res)
        self.assertFalse(mock_fetch.called)

    def test_compute_alphashape_fallback(self):
        """Test that alphashape falls back to convex_hull on error."""
        # Create a tiny point set that usually causes issues
        points = np.array([[0, 0], [0.0001, 0.0001]])
        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])  # Dummy

        cpt = defaultdict(int)
        # We mock alphashape to raise an error
        with patch("alphashape.alphashape", side_effect=RuntimeError("Singular")):
            res, updated_cpt = compute_alphashape_safe(gdf, points, 0.1, cpt)
            self.assertEqual(updated_cpt["alphashape_fallback_to_convex_hull"], 1)
            # Result should be the convex hull of the points (a LineString in this case)
            self.assertFalse(res.is_empty)

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.get_conf_content")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.glob.glob")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.get_swot_geoloc")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.do_cdse_query")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.save_meta_coloc_output")
    def test_treat_one_day_wrapper_robustness(
        self, mock_save, mock_query, mock_geoloc, mock_glob, mock_conf
    ):
        """Test that wrapper continues even if one CDSE query fails with ValueError."""
        # 1. Setup Configuration mock
        mock_conf.return_value = {
            "SWOT_L2_AVISO_DIR": "/tmp",
            "CACHE_CDSE": "/tmp",
            "DELTA_HOURS": 6,
            "MAX_AREA_SIZE": 100,
            "TOLERANCE_SIMPLIFICATION": 0.1,
        }
        mock_glob.return_value = ["file1.nc"]

        # 2. Simulate 2 pieces of SWOT orbit found in the file
        mock_geoloc.return_value = ([MagicMock(), MagicMock()], defaultdict(int))

        # 3. Simulate CDSE query behavior: first fails, second succeeds
        mock_query.side_effect = [
            ValueError("CDSE error"),
            pd.DataFrame({"Name": ["S1"]}),
        ]

        # 4. FIX: Handle both positional and keyword arguments for 'cpt'
        # In treat_one_day_wrapper, it is called as a keyword argument 'cpt=cpt'
        def side_effect_action(*args, **kwargs):
            if "cpt" in kwargs:
                return kwargs["cpt"]
            return args[4]  # Fallback to positional if needed

        mock_save.side_effect = side_effect_action

        # 5. Execute
        cpt_result = treat_one_day_wrapper(
            "20251017", "/tmp/out", "IW", "fake_conf.yml"
        )

        # 6. Assertions
        # Check that the error in the loop was caught and counted
        self.assertEqual(cpt_result["problematic_gdf"], 1)
        # Check that the successful query was counted
        self.assertEqual(cpt_result["sentinel1_product_matching"], 1)
        # Verify that save_meta_coloc_output was actually called
        self.assertTrue(mock_save.called)


if __name__ == "__main__":
    unittest.main()
