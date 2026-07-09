import datetime
from os.path import exists as real_exists  # Import the real function for fallback
from stat import S_IFDIR, S_IFREG
from unittest.mock import MagicMock, patch

# Import the specific script
# Note: Ensure the .py file is in your python path or the same directory
import s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes as swot_script

# --- 1. Unit Tests for Utility Functions ---


def test_extract_date_from_filename():
    # Valid case
    fn = "SWOT_L2_LR_SSH_WindWave_032_268_20250507T134734_v1.nc"
    assert swot_script.extract_date_from_filename(fn) == datetime.date(2025, 5, 7)

    # Invalid cases
    assert swot_script.extract_date_from_filename("random.nc") is None
    assert swot_script.extract_date_from_filename("SWOT_2025T.nc") is None


def test_is_dir():
    mock_sftp = MagicMock()
    # Mocking directory mode
    mock_sftp.stat.return_value.st_mode = S_IFDIR
    assert swot_script.is_dir(mock_sftp, "/remote/dir") is True

    # Mocking file mode
    mock_sftp.stat.return_value.st_mode = S_IFREG
    assert swot_script.is_dir(mock_sftp, "/remote/file.nc") is False


@patch("s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.tqdm")
def test_download_with_progress(mock_tqdm):
    mock_sftp = MagicMock()
    mock_sftp.stat.return_value.st_size = 100

    swot_script.download_with_progress(mock_sftp, "rem", "loc", "file.nc")

    mock_sftp.get.assert_called_once()
    assert mock_sftp.get.call_args[0][0] == "rem"
    assert mock_sftp.get.call_args[0][1] == "loc"


# --- 2. Integration Test for main() ---


@patch("s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.paramiko.SSHClient")
@patch("s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.os.makedirs")
@patch("s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.os.path.exists")
@patch(
    "s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.argparse.ArgumentParser.parse_args"
)
@patch(
    "s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.download_with_progress"
)
def test_main_execution_logic(
    mock_download, mock_args, mock_exists_patch, mock_makedirs, mock_ssh_class
):
    # 1. Setup Mock Arguments
    mock_args.return_value = MagicMock(
        user="myuser",
        password="mypassword",
        dest="/local/path",
        start="2025-05-01",
        end="2025-05-10",
        host="host",
        port=2221,
        productID="PID0",
        verbose=True,
        dry_run=False,
    )

    # 2. Define a selective side_effect function
    # This prevents argparse/gettext from crashing
    def selective_exists(path):
        # If the path looks like one of our target files, return our controlled logic
        if "20250505T" in str(path):
            return False  # File 1: In range, does not exist -> will download
        if "20250506T" in str(path):
            return True  # File 2: In range, exists -> will skip
        # For everything else (argparse, locale files, etc.), use the real OS check
        return real_exists(path)

    mock_exists_patch.side_effect = selective_exists

    # 3. Setup SSH/SFTP Mocks
    mock_ssh = mock_ssh_class.return_value
    mock_sftp = mock_ssh.open_sftp.return_value

    mock_sftp.listdir.side_effect = [
        ["cycle_01"],
        [
            "SWOT_L2_LR_SSH_WindWave_01_01_20250505T120000_v1.nc",  # In range (selective_exists -> False)
            "SWOT_L2_LR_SSH_WindWave_01_01_20250101T120000_v1.nc",  # Out of range (date check skips it)
            "SWOT_L2_LR_SSH_WindWave_01_01_20250506T120000_v1.nc",  # In range (selective_exists -> True)
        ],
    ]

    # 4. Execute
    from s1swotcolocs import download_l2_swot_karin_lr_ssh_windwave_cnes as swot_script

    swot_script.main()

    # 5. Assertions
    # Connection logic
    mock_ssh.connect.assert_called_with(
        "host", port=2221, username="myuser", password="mypassword"
    )

    # download_with_progress should only be called for the 05-05 file
    assert mock_download.call_count == 1
    # Check that it tried to download the correct remote path
    assert "20250505T" in mock_download.call_args[0][1]


@patch(
    "s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.argparse.ArgumentParser.parse_args"
)
@patch("s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.paramiko.SSHClient")
def test_main_dry_run_logic(mock_ssh_class, mock_args):
    """Verifies that in dry-run mode, no downloads happen but stats are logged."""
    mock_args.return_value = MagicMock(
        user="u",
        password="p",
        dest="d",
        start="2025-05-01",
        end="2025-05-10",
        host="h",
        port=22,
        productID="PID0",
        verbose=False,
        dry_run=True,
    )

    mock_ssh = mock_ssh_class.return_value
    mock_sftp = mock_ssh.open_sftp.return_value
    mock_sftp.listdir.side_effect = [
        ["cycle_01"],
        ["SWOT_L2_LR_SSH_WindWave_01_01_20250505T120000_v1.nc"],
    ]

    with patch(
        "s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.os.path.exists",
        return_value=False,
    ):
        with patch(
            "s1swotcolocs.download_l2_swot_karin_lr_ssh_windwave_cnes.download_with_progress"
        ) as mock_dl:
            swot_script.main()
            # In dry run, download function is never called
            assert mock_dl.call_count == 0
