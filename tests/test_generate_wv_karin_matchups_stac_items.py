import pytest
from s1swotcolocs.generate_wv_karin_matchups_stac_items import is_match

def test_is_match_temporal_success():
    s1 = {"timestamp": 100, "footprint": None} # Mock timestamps in seconds since epoch
    swot = {"timestamp": 105, "footprint": None}
    assert is_match(s1, swot, time_threshold_min=10)

def test_is_match_temporal_fail():
    s1 = {"timestamp": 100, "footprint": None}
    swot = {"timestamp": 700, "footprint": None}
    assert not is_match(s1, swot, time_threshold_min=10)

def test_is_match_spatial_success():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"timestamp": 105, "footprint": box(0.5, 0.5, 1.5, 1.5)}
    assert is_match(s1, swot)

def test_is_match_spatial_fail():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"timestamp": 105, "footprint": box(2, 2, 3, 3)}
    assert not is_match(s1, swot)

import pytest
from s1swotcolocs.generate_wv_karin_matchups_stac_items import is_match

def test_is_match_temporal_success():
    s1 = {"timestamp": 100, "footprint": None} # Mock timestamps in seconds since epoch
    swot = {"timestamp": 105, "footprint": None}
    assert is_match(s1, swot, time_threshold_min=10)

def test_is_match_temporal_fail():
    s1 = {"timestamp": 100, "footprint": None}
    swot = {"timestamp": 700, "footprint": None}
    assert not is_match(s1, swot, time_threshold_min=10)

def test_is_match_spatial_success():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"timestamp": 105, "footprint": box(0.5, 0.5, 1.5, 1.5)}
    assert is_match(s1, swot)

def test_is_match_spatial_fail():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"timestamp": 105, "footprint": box(2, 2, 3, 3)}
    assert not is_match(s1, swot)

def test_wv_swot_specific_matchup():
    """
    Test the specific pair mentioned by user:
    S1: s1a-wv2-ocn-vv-20250531t113616-20250531t113619...
    SWOT: SWOT_L2_LR_SSH_WindWave_033_353_20250531T112541_20250531T121709...
    """
    # This test will be expanded once we have real-world mocks or data access
    pass

