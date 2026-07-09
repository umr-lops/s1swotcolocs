# File: test_swot_versioning.py

import unittest

from s1swotcolocs.pickup_best_swot_file import (
    _get_acquisition_id,
    _get_crid_sort_key,
    check_if_latest_version,
    select_latest_version,
)


class TestSwotVersioning(unittest.TestCase):
    """Test suite for the SWOT product versioning logic with new filename pattern."""

    def setUp(self):
        """Set up a list of mock filenames for testing."""
        self.all_files = [
            # Acquisition 018_555: Has 'I' and 'G' versions + file versions
            "SWOT_L2_LR_SSH_018_555_..._PIB0_01.nc",
            "SWOT_L2_LR_SSH_018_555_..._PIC0_01.nc",  # Newer letter
            "SWOT_L2_LR_SSH_018_555_..._PIC0_02.nc",  # Newer file version
            "SWOT_L2_LR_SSH_018_555_..._PGA0_01.nc",  # THE LATEST for this acquisition ('G')
            # Acquisition 020_111: Only has 'I' versions
            "SWOT_L2_LR_SSH_020_111_..._PIA1_01.nc",
            "SWOT_L2_LR_SSH_020_111_..._PIA1_02.nc",  # THE LATEST for this acquisition
            # File without a file version number (should default to 0)
            "SWOT_L2_LR_SSH_020_111_..._PIA0.nc",
            # Invalid or malformed filenames
            "invalid_filename_without_version_info.nc",
            "SWOT_L2_LR_SSH_malformed_id_..._PIC0_01.nc",
        ]

    def test_get_crid_sort_key(self):
        """Test the CRID to sort key conversion with file version."""
        # Standard case with file version
        self.assertEqual(_get_crid_sort_key("..._PIC0_01.nc"), (0, "C", 0, 1))
        # 'G' processing type
        self.assertEqual(_get_crid_sort_key("..._PGA0_10.nc"), (1, "A", 0, 10))
        # Case without a file version, should default to 0
        self.assertEqual(_get_crid_sort_key("..._PIB2.nc"), (0, "B", 2, 0))
        # No CRID found should return the lowest priority key
        self.assertEqual(_get_crid_sort_key("no_crid_here.nc"), (-1, "", -1, -1))

    def test_get_acquisition_id(self):
        """Test the extraction of the acquisition ID."""
        self.assertEqual(
            _get_acquisition_id("SWOT_L2_LR_SSH_018_555_..._PIC0_01.nc"), "018_555"
        )
        self.assertIsNone(_get_acquisition_id("some_other_file.nc"))

    def test_select_latest_version(self):
        """Test the selection of the latest file from a list."""
        # Test Case 1: File version is the deciding factor
        files1 = [
            "SWOT_L2_LR_SSH_018_555_..._PIC0_01.nc",
            "SWOT_L2_LR_SSH_018_555_..._PIC0_03.nc",
            "SWOT_L2_LR_SSH_018_555_..._PIC0_02.nc",
        ]
        expected1 = "SWOT_L2_LR_SSH_018_555_..._PIC0_03.nc"
        self.assertEqual(select_latest_version(files1), expected1)

        # Test Case 2: 'G' version wins over everything, even a higher 'I' file version
        files2 = [
            "SWOT_L2_LR_SSH_018_555_..._PIC9_05.nc",  # Highest 'I' version
            "SWOT_L2_LR_SSH_018_555_..._PGA0_01.nc",  # 'G' version
        ]
        expected2 = "SWOT_L2_LR_SSH_018_555_..._PGA0_01.nc"
        self.assertEqual(select_latest_version(files2), expected2)

        # Test Case 3: An empty list should return None
        self.assertIsNone(select_latest_version([]))

    def test_check_if_latest_version(self):
        """Test the validation logic for a specific file."""
        # Test Case 1: File is NOT the latest (a newer file version exists)
        file_to_check1 = "SWOT_L2_LR_SSH_018_555_..._PIC0_01.nc"
        is_latest, real_latest = check_if_latest_version(file_to_check1, self.all_files)

        self.assertFalse(is_latest)
        self.assertEqual(real_latest, "SWOT_L2_LR_SSH_018_555_..._PGA0_01.nc")

        # Test Case 2: File is NOT the latest (a 'G' version exists)
        file_to_check2 = "SWOT_L2_LR_SSH_018_555_..._PIC0_02.nc"
        is_latest, real_latest = check_if_latest_version(file_to_check2, self.all_files)

        self.assertFalse(is_latest)
        self.assertEqual(real_latest, "SWOT_L2_LR_SSH_018_555_..._PGA0_01.nc")

        # Test Case 3: File IS the latest for its acquisition
        file_to_check3 = "SWOT_L2_LR_SSH_018_555_..._PGA0_01.nc"
        is_latest, real_latest = check_if_latest_version(file_to_check3, self.all_files)

        self.assertTrue(is_latest)
        self.assertIsNone(real_latest)

        # Test Case 4: File IS the latest for a DIFFERENT acquisition
        file_to_check4 = "SWOT_L2_LR_SSH_020_111_..._PIA1_02.nc"
        is_latest, real_latest = check_if_latest_version(file_to_check4, self.all_files)

        self.assertTrue(is_latest)
        self.assertIsNone(real_latest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
