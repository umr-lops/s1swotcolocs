import pytest
from s1swotcolocs.generate_wv_karin_matchups_stac_items import is_match

def test_is_match_temporal_success():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)} # Mock timestamps in seconds since epoch
    swot = {"start_time": 90, "end_time": 110, "footprint": box(0.5, 0.5, 1.5, 1.5)}
    assert is_match(s1, swot, time_threshold_min=10)

def test_is_match_temporal_fail():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"start_time": 700, "end_time": 800, "footprint": box(2, 2, 3, 3)}
    assert not is_match(s1, swot, time_threshold_min=10)

def test_is_match_spatial_success():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"start_time": 90, "end_time": 110, "footprint": box(0.5, 0.5, 1.5, 1.5)}
    assert is_match(s1, swot)

def test_is_match_spatial_fail():
    from shapely.geometry import box
    s1 = {"timestamp": 100, "footprint": box(0, 0, 1, 1)}
    swot = {"start_time": 90, "end_time": 110, "footprint": box(2, 2, 3, 3)}
    assert not is_match(s1, swot)

def test_wv_swot_specific_matchup():
    """
    Test the specific pair mentioned by user:
    S1: s1a-wv2-ocn-vv-20250531t113616-20250531t113619...
    SWOT: SWOT_L2_LR_SSH_WindWave_033_353_20250531T112541_20250531T121709...
    """
    from shapely.geometry import box
    # S1: 11:36:16 -> 1748711776 (approx)
    s1 = {
        "timestamp": 1748711776, 
        "footprint": box(0, 0, 1, 1), 
        "filename": "S1_test.nc"
    }
    # SWOT: [11:25:41, 12:17:09] -> [1748711141, 1748712429]
    swot = {
        "start_time": 1748711141, 
        "end_time": 1748712429, 
        "footprint": box(0.5, 0.5, 1.5, 1.5), 
        "filename": "SWOT_test.nc"
    }
    assert is_match(s1, swot)
