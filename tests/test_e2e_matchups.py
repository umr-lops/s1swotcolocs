import os
import pytest
from s1swotcolocs.generate_wv_karin_matchups_stac_items import find_matchups

def test_e2e_wv_swot_matchup():
    """
    End-to-end test with minimal assets from src/s1swotcolocs/assets.
    """
    # Use absolute paths to avoid issues during pytest run
    cwd = os.getcwd()
    wv_dir = os.path.join(cwd, "src", "s1swotcolocs", "assets")
    swot_dir = os.path.join(cwd, "src", "s1swotcolocs", "assets")
    
    config = {"TIME_THRESHOLD_MIN": 10}
    results = find_matchups(wv_dir, swot_dir, config)
    
    # We expect at least the minimal pair to be found
    assert len(results) >= 1
    item = results[0]
    assert "sarmatchup:sar_platform" in item.properties
    assert "sarmatchup:other_instrument" in item.properties
    assert "test_s1_wv_minimal.nc" in item.assets["s1_wv"].href
    assert "test_swot_l2_minimal.nc" in item.assets["swot_l2"].href
