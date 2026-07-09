# File: swot_versioning.py

import re
from typing import Optional


def _get_acquisition_id(filename: str) -> Optional[str]:
    """
    Extracts the acquisition identifier (cycle_pass) from a SWOT filename.
    Assumes the format _<3-digit-cycle>_<3-digit-pass>_

    Args:
        filename (str): The filename to parse.

    Returns:
        The acquisition ID string (e.g., "018_555") or None if not found.
    """
    match = re.search(r"_(\d{3}_\d{3})_", filename)
    if match:
        return match.group(1)
    return None


def _get_crid_sort_key(filename: str) -> tuple:
    """
    Generates a comparable sort key from a filename based on its CRID and file version.

    The key is a 4-element tuple representing the version hierarchy:
    1. Processing Type ('G' > 'I')
    2. Algorithm Letter ('C' > 'B' > 'A')
    3. Algorithm Number (2 > 1 > 0)
    4. File Version (e.g., 02 > 01)

    Args:
        filename (str): The filename containing a KaRIn CRID.

    Returns:
        A tuple for comparison, e.g., for "..._PIC0_01.nc" -> (0, 'C', 0, 1).
        Returns a "lowest possible" key if no version is found.
    """
    # Regex to find the CRID and the optional file version number following it.
    # (P[IG][A-Z]\d) -> Captures the CRID, e.g., "PIC0"
    # (?:_(\d+))?   -> Optionally captures the file version number, e.g., "01"
    # The ?: makes the group non-capturing, but the inner (\d+) is capturing.
    # The final ? makes the whole group optional.
    version_pattern = r"(P[IG][A-Z]\d)(?:_(\d+))?"
    match = re.search(version_pattern, filename)

    if not match:
        # Return a key that will be sorted first (lowest priority)
        return (-1, "", -1, -1)

    crid = match.group(1)  # e.g., "PIC0"
    file_version_str = match.group(2)  # e.g., "01" or None if not present

    # Deconstruct the CRID
    processing_type = crid[1]
    alg_letter = crid[2]
    alg_number = int(crid[3])

    # 1. Convert processing type to a number for comparison (G=1, I=0)
    processing_val = 1 if processing_type == "G" else 0

    # 4. Convert file version to an int, defaulting to 0 if not present
    file_version_num = int(file_version_str) if file_version_str else 0

    # Return the 4-element sort key tuple
    return (processing_val, alg_letter, alg_number, file_version_num)


def select_latest_version(filenames: list[str]) -> Optional[str]:
    """
    Selects the filename with the latest KaRIn product version from a list.
    """
    if not filenames:
        return None
    return max(filenames, key=_get_crid_sort_key)


def check_if_latest_version(
    file_to_check: str, all_available_files: list[str]
) -> tuple[bool, Optional[str]]:
    """Validates if a given file is the definitive latest version for its acquisition.

    This function determines if a specific SWOT data file (`file_to_check`)
    represents the most recent version available for its unique acquisition
    (identified by its cycle and pass number, e.g., '018_555').

    The process is as follows:
    1. It extracts the acquisition ID from `file_to_check`.
    2. It filters the `all_available_files` list to create a subgroup of
       files that share the same acquisition ID.
    3. It identifies the single "latest" file from this subgroup using the
       established versioning rules (Processing Type > Algorithm Letter >
       Algorithm Number > File Version).
    4. It compares `file_to_check` with this definitive latest version.

    Args:
        file_to_check (str): The full filename or path of the SWOT product
            to validate.
        all_available_files (list[str]): A comprehensive list of all candidate
            filenames to search within (e.g., from a directory listing).

    Returns:
        A tuple containing two elements:
        - is_the_latest_version bool: `True` if `file_to_check` is the latest version for its
          acquisition; `False` otherwise.
        - Optional[str]:
            - If the boolean is `False`, this holds the filename of the
              actual latest version.
            - If the boolean is `True`, this is `None`.
            - It is also `None` if the acquisition ID cannot be determined
              from `file_to_check`, as a comparison is not possible.

    Example:
        >>> all_files = [
        ...     "SWOT_L2_LR_..._018_555_..._PIC0_01.nc",
        ...     "SWOT_L2_LR_..._018_555_..._PGA0_01.nc",  # The true latest
        ... ]
        >>> old_file = "SWOT_L2_LR_..._018_555_..._PIC0_01.nc"
        >>> is_latest, actual_latest = check_if_latest_version(old_file, all_files)
        >>> print(is_latest)
        False
        >>> print(actual_latest)
        SWOT_L2_LR_..._018_555_..._PGA0_01.nc

        >>> latest_file = "SWOT_L2_LR_..._018_555_..._PGA0_01.nc"
        >>> is_latest, actual_latest = check_if_latest_version(latest_file, all_files)
        >>> print(is_latest)
        True
        >>> print(actual_latest)
        None
    """
    acq_id = _get_acquisition_id(file_to_check)
    is_the_latest_version = False
    if not acq_id:
        print(f"Warning: Could not determine acquisition ID for '{file_to_check}'.")
        return (is_the_latest_version, None)

    related_files = [f for f in all_available_files if _get_acquisition_id(f) == acq_id]

    if not related_files:
        return (is_the_latest_version, None)

    true_latest_file = select_latest_version(related_files)

    if file_to_check == true_latest_file:
        is_the_latest_version = True
        return (is_the_latest_version, None)
    else:
        return (is_the_latest_version, true_latest_file)
