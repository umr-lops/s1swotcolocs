import collections
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import get_swot_geoloc


class TestGetSwotGeoloc(unittest.TestCase):

    def create_mock_swot_ds(self, num_lines=1500, has_nat=False):
        """Creates a fake SWOT dataset."""
        lons = np.linspace(0, 10, num_lines * 5).reshape(num_lines, 5)
        lats = np.linspace(0, 10, num_lines * 5).reshape(num_lines, 5)

        # Explicitly use nanosecond resolution [ns]
        base_time = np.datetime64("2023-01-01T12:00:00", "ns")
        times = base_time + np.arange(num_lines) * np.timedelta64(1, "s")
        times = times.astype("datetime64[ns]")

        if has_nat:
            # Explicitly specify the unit for NaT
            times[0] = np.datetime64("NaT", "ns")
            times[-1] = np.datetime64("NaT", "ns")

        ds = xr.Dataset(
            {
                "longitude": (("num_lines", "num_pixels"), lons),
                "latitude": (("num_lines", "num_pixels"), lats),
                "time": (("num_lines",), times),
            }
        )
        return ds

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.xr.open_dataset")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.slice_swot")
    def test_get_swot_geoloc_segmentation(self, mock_slice, mock_open):
        """Verify that segments of 1000 lines are processed."""
        mock_open.return_value = self.create_mock_swot_ds(num_lines=2500)
        mock_slice.return_value = ([], collections.defaultdict(int))

        cpt = collections.defaultdict(int)
        get_swot_geoloc("fake.nc", max_area_size=100, cpt=cpt)

        self.assertEqual(mock_slice.call_count, 3)

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.xr.open_dataset")
    def test_nat_trimming_logic(self, mock_open):
        """Test the trimming of NaT values at file start/end."""
        mock_ds = MagicMock()
        # Explicitly use [ns] unit
        times = np.array(
            [
                np.datetime64("NaT", "ns"),
                np.datetime64("2023-01-01", "ns"),
                np.datetime64("NaT", "ns"),
            ]
        )

        mock_ds.__getitem__.return_value.values = times
        mock_ds.isel.return_value = mock_ds
        mock_open.return_value = mock_ds

        valid_mask = ~np.isnat(times)
        valid_indices = np.where(valid_mask)[0]

        self.assertEqual(len(valid_indices), 1)
        self.assertEqual(valid_indices[0], 1)

    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.slice_swot")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.xr.open_dataset")
    def test_empty_dataset(self, mock_open, mock_slice):
        """Test behavior when SWOT file is entirely NaT."""
        # FIX: Create a numpy array with an explicit [ns] dtype instead of a Python list
        times = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        ds = xr.Dataset({"time": (("num_lines",), times)})
        mock_open.return_value = ds

        cpt = collections.defaultdict(int)
        res, updated_cpt = get_swot_geoloc("empty.nc", max_area_size=100, cpt=cpt)

        self.assertEqual(len(res), 0)
        self.assertEqual(mock_slice.call_count, 0)


if __name__ == "__main__":
    unittest.main()
