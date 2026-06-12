import os
import pytest
from s1swotcolocs.generate_wv_karin_matchups_stac_items import find_matchups

def test_e2e_wv_swot_matchup():
    """
    End-to-end test with real assets from src/s1swotcolocs/assets.
    """
    # Use absolute paths to avoid issues during pytest run
    cwd = os.getcwd()
    wv_dir = os.path.join(cwd, "src", "s1swotcolocs", "assets")
    swot_dir = os.path.join(cwd, "src", "s1swotcolocs", "assets")
    
    config = {"TIME_THRESHOLD_MIN": 10}
    results = find_matchups(wv_dir, swot_dir, config)
    
    # Based on filenames:
    # S1: l2-s1a-iw3-wav-dv-20240729t172529... (T=172529)
    # SWOT: SWOT_L2_LR_SSH_WindWave_018_555_20240729T172147... (Start=172147, End=181315)
    # 17:25:29 is within [17:21:47, 18:13:15]
    # Spatially they should overlap if it's a real pair.
    
    assert len(results) >= 1
    item = results[0]
    assert "sarmatchup:sar_platform" in item.properties
    assert "sarmatchup:other_instrument" in item.properties
    assert item.assets["s1_wv"].href.endswith(".nc")
    assert item.assets["swot_l2"].href.endswith(".nc")
