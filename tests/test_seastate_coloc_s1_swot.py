import collections
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import s1swotcolocs

# Import the script we want to test
from s1swotcolocs import seastate_colocs_s1_swot as s1_coloc
from s1swotcolocs.utils import get_conf_content

potential_config_file_1 = os.path.join(
    os.path.dirname(s1swotcolocs.__file__), "localconfig.yml"
)
potential_config_file_2 = os.path.join(
    os.path.dirname(s1swotcolocs.__file__), "config.yml"
)
if not os.path.exists(potential_config_file_1):
    confpath = potential_config_file_2
else:
    confpath = potential_config_file_1

conf = get_conf_content(confpath)

assets_dir = os.path.join(os.path.dirname(s1swotcolocs.__file__), "assets")

# --- Pytest Fixtures: Reusable setup for tests ---


@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary."""
    return {
        "HOST_SEASTATE_COLOC_OUTPUT_DIR": conf["HOST_SEASTATE_COLOC_OUTPUT_DIR"],
        "SWOT_L3_AVISO_FILE": os.path.join(
            assets_dir,
            "SWOT_L3_LR_SSH_Expert_026_165_20241229T165806_20241229T174933_v2.0.1.nc",
        ),
        "SWOT_L2_AVISO_FILE": os.path.join(
            assets_dir,
            "SWOT_L2_LR_SSH_WindWave_018_555_20240729T172147_20240729T181315_PIC0_01.nc",
        ),
        "S1_L2_WAV_FILE": os.path.join(
            assets_dir,
            "l2-s1a-iw3-wav-dv-20240729t172529-20240729t172557-054978-06b28f-e12.nc",
        ),
        "metacoloc": os.path.join(
            assets_dir, "coloc_SWOT_L3_Sentinel-1_IW_20240729T172147.nc"
        ),
        "RADIUS_COLOC": 0.07,
    }


@pytest.fixture
def sample_metacoloc_ds(tmp_path):
    """Creates a sample meta-colocation xarray.Dataset for testing."""
    # Create a dummy NetCDF file because some functions expect a path
    filepath = (
        tmp_path
        / "seastate_coloc_S1A_IW_SLC__1SDV_20240729T172527_20240729T172557_054978_06B28F_CEF7-iw3_SWOT_L3_Sentinel-1_IW_20240729T172147.nc"
    )
    ds = xr.Dataset(
        {
            "sar_safe_name": (("coloc",), ["S1A_IW_SLC__1SDV_..."]),
            "filepath_swot": (("coloc",), ["SWOT_L3_Expert_..._20230414T035201_..."]),
        },
        attrs={"description": "A test meta-colocation file."},
    )
    ds.to_netcdf(filepath)
    return ds, filepath


@pytest.fixture
def sample_sar_l2_ds():
    """Creates a sample SAR L2 xarray.Dataset for testing."""
    return xr.Dataset(
        {
            "longitude": (
                ("tile_line", "tile_sample"),
                np.array([[10.0, 10.1], [10.0, 10.1]]),
            ),
            "latitude": (
                ("tile_line", "tile_sample"),
                np.array([[50.0, 50.0], [50.1, 50.1]]),
            ),
        },
        attrs={"footprint": "POLYGON ((...))"},
    )


@pytest.fixture
def sample_swot_l2_ds():
    """Creates a sample SWOT L2 xarray.Dataset for testing."""
    return xr.Dataset(
        {
            "longitude": (
                ("num_lines", "num_pixels"),
                np.array([[10.05, 10.15], [10.05, 10.15]]),
            ),
            "latitude": (
                ("num_lines", "num_pixels"),
                np.array([[50.02, 50.02], [50.12, 50.12]]),
            ),
            "swh_karin": (("num_lines", "num_pixels"), np.ones((2, 2)) * 2.5),
            "wind_speed_karin": (("num_lines", "num_pixels"), np.ones((2, 2)) * 2.5),
            "wind_speed_karin_2": (("num_lines", "num_pixels"), np.ones((2, 2)) * 2.5),
            "wind_speed_model_u": (("num_lines", "num_pixels"), np.ones((2, 2)) * 2.5),
            "wind_speed_model_v": (("num_lines", "num_pixels"), np.ones((2, 2)) * 2.5),
            "swh_nadir_altimeter": (("num_lines", "num_pixels"), np.ones((2, 2)) * 2.5),
        }
    )


# --- Unit Tests for each function ---


def test_get_original_sar_filepath_from_metacoloc(sample_metacoloc_ds):
    """
    Tests finding the original SAR SAFE path.
    We mock the s1ifr function to avoid depending on a real filesystem.
    """
    metacoloc_ds, _ = sample_metacoloc_ds
    cpt = collections.defaultdict(int)

    # Mock the external dependency
    with patch(
        "s1ifr.get_path_from_base_safe.get_path_from_base_safe"
    ) as mock_get_path:
        # Simulate that the file is found in the 'datawork' archive
        mock_get_path.return_value = "/path/to/found/S1A_IW_SLC__1SDV_....SAFE"

        # Mock os.path.exists
        with patch("os.path.exists", return_value=True):
            paths, _ = s1_coloc.get_original_sar_filepath_from_metacoloc(
                metacoloc_ds, cpt
            )

            assert len(paths) == 1
            assert paths[0] == "/path/to/found/S1A_IW_SLC__1SDV_....SAFE"


def test_get_L2WAV_S1_IW_path():
    """
    Tests getting the L2 WAV path from a given L1 SLC path.
    We mock the `s1ifr.paths_safe_product_family` library call.
    """
    mock_l1_path = "/path/to/S1A_IW_SLC.SAFE"

    # Mock the return value of the external library
    mock_dataframe = pd.DataFrame(
        {"L2_WAV_E12": ["/path/to/L2_WAV.SAFE"], "L2_WAV_E11": [None]}
    )

    with patch(
        "s1ifr.paths_safe_product_family.get_products_family",
        return_value=mock_dataframe,
    ) as mock_get_family:
        result_path = s1_coloc.get_L2WAV_S1_IW_path(mock_l1_path)

        mock_get_family.assert_called_once()
        assert result_path == "/path/to/L2_WAV.SAFE"


def test_create_empty_coloc_res():
    """Tests the creation of an empty result dataset."""
    indexes = {"tile_line": 0, "tile_sample": 0}
    ds = s1_coloc.create_empty_coloc_res(indexes)

    assert "latitude_mean" in ds
    assert "swh_karin_std" in ds
    assert ds["swh_karin_std"].dims == ("tile_line", "tile_sample")
    assert np.isnan(ds["latitude_mean"].values)


def test_s1swot_core_tile_coloc(sample_swot_l2_ds):
    """Tests the core co-location logic for a single SAR tile."""
    dsswot = sample_swot_l2_ds
    cpt = collections.defaultdict(int)

    # Create a KDTree from the SWOT data
    points = np.c_[
        dsswot["longitude"].values.ravel(), dsswot["latitude"].values.ravel()
    ]
    treeswot = s1_coloc.spatial.KDTree(points)

    # Test a point that should have neighbors
    lontile, lattile = 10.0, 50.0
    radius_coloc = 0.1
    indexes_sar = {"tile_line": 0, "tile_sample": 0}

    condensated_ds, _ = s1_coloc.s1swot_core_tile_coloc(
        lontile, lattile, treeswot, radius_coloc, dsswot, indexes_sar, cpt
    )

    assert "nb_SWOT_points" in condensated_ds
    assert condensated_ds["nb_SWOT_points"].item() > 0
    assert np.isclose(condensated_ds["swh_karin_mean"].item(), 2.5)

    # Test a point that should NOT have neighbors
    lontile_far, lattile_far = 0.0, 0.0
    condensated_ds_empty, _ = s1_coloc.s1swot_core_tile_coloc(
        lontile_far, lattile_far, treeswot, radius_coloc, dsswot, indexes_sar, cpt
    )

    assert (
        "nb_SWOT_points" not in condensated_ds_empty
    )  # It should be an empty dataset from create_empty_coloc_res
    assert np.isnan(condensated_ds_empty["swh_karin_mean"].values)


def test_save_sea_state_coloc_file(tmp_path):
    """Tests saving the final NetCDF file."""
    # tmp_path is a pytest fixture for a temporary directory
    output_path = tmp_path / "test_output.nc"
    cpt = collections.defaultdict(int)

    # FIX: Add coordinates to the dimension 'dim'
    # This prevents the H5DSis_scale error in h5netcdf
    ds_to_save = xr.Dataset({"data": (("dim",), [1, 2, 3])}, coords={"dim": [0, 1, 2]})

    # Run the function
    s1_coloc.save_sea_state_coloc_file(ds_to_save, str(output_path), cpt)

    # Assertions
    assert os.path.exists(output_path)
    assert cpt["new_file"] == 1


def test_associate_sar_and_swot_seastate_params_missing_field(
    sample_metacoloc_ds, mock_config
):
    """
    Tests that the main function raises a ValueError if 'filepath_swot' is missing.
    """
    metacoloc_ds, metacoloc_path = sample_metacoloc_ds

    # Create a version of the dataset without the required field
    del metacoloc_ds["filepath_swot"]

    # Use a mock to prevent the function from actually running, we just want to test the check
    with patch("s1swotcolocs.utils.get_conf_content", return_value=mock_config):
        with patch("xarray.open_dataset", return_value=metacoloc_ds):
            # We expect a KeyError (or your custom MissingFieldError) to be raised
            with pytest.raises(KeyError):
                s1_coloc.associate_sar_and_swot_seastate_params(
                    metacolocpath=metacoloc_path, confpath=confpath
                )
