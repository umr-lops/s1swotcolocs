import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime
from collections import defaultdict

# Import the module under test
import s1swotcolocs.convert_matchups_nc_to_stac as converter


@pytest.fixture
def mock_config():
    return {"SWOT_L2_AVISO_DIR": "/fake/aviso/l2"}


@pytest.fixture
def mock_nc_dataset():
    """Creates a mock xarray dataset to simulate the matchup NetCDF."""
    ds = MagicMock()

    # Global Attributes
    ds.attrs = {
        "s1swotcolocs_python_lib_version": "0.1.test",
        "searching_windows_width_in_hours": 1.0,
    }

    # 1. Mock the length for the loop
    # We must ensure len(ds.sar_start_time_slice) returns 1
    ds.sar_start_time_slice.__len__.return_value = 1

    # 2. Use real numpy/bytes data so logic like .decode() or str() works
    ds.sar_start_time_slice.values = np.array(
        ["2025-10-19T22:35:21"], dtype="datetime64[ns]"
    )
    ds.SWOT_start_time_slice.values = np.array([0])

    # These need to be indexed like var.values[i]
    ds.sar_safe_name.values = [b"S1C_IW_SLC__1SDV_20251019T223521_004637.SAFE"]
    ds.filepath_swot.values = [b"SWOT_L3_LR_SSH_Expert_040_224_20251019.nc"]
    ds.sar_polygon.values = [b"POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"]

    # Mock numeric variable
    delta_val = MagicMock()
    delta_val.values = np.array([2.5])
    ds.delta_diff_time = delta_val

    # 3. FIX: Mock swot_polygon.values as a real bytes object
    # This prevents the shapely ParseException: Unknown type: '<MAGICMOCK'
    ds.swot_polygon.values = b"POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"

    return ds


@patch("s1swotcolocs.convert_matchups_nc_to_stac.xr.open_dataset")
@patch("s1swotcolocs.convert_matchups_nc_to_stac.get_swot_date_info")
@patch("s1swotcolocs.convert_matchups_nc_to_stac.get_s1_full_path")
@patch("s1swotcolocs.convert_matchups_nc_to_stac.get_nasa_stac_info")
@patch("s1swotcolocs.convert_matchups_nc_to_stac.pystac.Item.save_object")
@patch("os.path.exists")
@patch("os.makedirs")
def test_process_nc_to_stac_success(
    mock_makedirs,
    mock_exists,
    mock_save_object,
    mock_nasa_stac,
    mock_s1_path,
    mock_swot_date,
    mock_xr_open,
    mock_nc_dataset,
    mock_config,
):
    """Test a successful conversion from NC to STAC."""
    mock_xr_open.return_value = mock_nc_dataset
    mock_swot_date.return_value = ("20251019T220435", "2025", "10", "19")
    mock_s1_path.return_value = "/archive/S1C_IW_SLC.SAFE"
    mock_nasa_stac.return_value = ("NASA_ID_123", "https://nasa.gov/stac/item")

    # First call to exists (check output dir) = False,
    # Second call (check if json exists) = False
    mock_exists.return_value = False

    counters = defaultdict(int)

    files_gen, updated_counters = converter.process_nc_to_stac(
        nc_path="dummy.nc",
        output_dir="/tmp/out",
        config=mock_config,
        extension_url="http://schema.json",
        counters=counters,
        overwrite=True,
    )

    assert len(files_gen) == 1
    assert updated_counters["matchups_with_STAC_NASA"] == 1
    assert updated_counters["matchups_with_fullpath_s1_ifremer"] == 1
    assert mock_save_object.called


@patch("s1swotcolocs.convert_matchups_nc_to_stac.xr.open_dataset")
@patch("s1swotcolocs.convert_matchups_nc_to_stac.get_swot_date_info")
@patch("os.path.exists")
def test_process_nc_to_stac_overwrite_false(
    mock_exists, mock_swot_date, mock_xr_open, mock_nc_dataset, mock_config
):
    """Test that existing files are skipped when overwrite=False."""
    mock_xr_open.return_value = mock_nc_dataset
    mock_swot_date.return_value = ("20251019T220435", "2025", "10", "19")

    # We need to simulate that the output JSON exists.
    # The script checks if output dir exists (True), then if JSON exists (True)
    mock_exists.return_value = True

    counters = defaultdict(int)

    files_gen, updated_counters = converter.process_nc_to_stac(
        nc_path="dummy.nc",
        output_dir="/tmp/out",
        config=mock_config,
        extension_url="url",
        counters=counters,
        overwrite=False,
    )

    assert len(files_gen) == 0
    assert updated_counters["matchups_already_available_in_STAC"] == 1


def test_calculate_overlap_percentage():
    """Test geometry projection and overlap."""
    from shapely.wkt import loads

    sar = loads("POLYGON ((0 0, 0 0.1, 0.1 0.1, 0.1 0, 0 0))")
    swot = loads("POLYGON ((0 0, 0 0.1, 0.05 0.1, 0.05 0, 0 0))")
    res = converter.calculate_overlap_percentage(sar, swot)
    assert 48 < res < 52


@patch("s1swotcolocs.convert_matchups_nc_to_stac.Client.open")
def test_get_nasa_stac_info_failure(mock_client_open):
    """Test failure handling in NASA STAC query."""
    mock_client_open.side_effect = Exception("API Down")
    res_id, res_url = converter.get_nasa_stac_info(
        datetime.now(), MagicMock(), "040", "224"
    )
    assert res_id is None
