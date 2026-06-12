import os
import datetime
from typing import List, Tuple, Optional, Dict, Any
import xarray as xr
from shapely.geometry import shape, box, Polygon
import geopandas as gpd
from pystac import Item, Asset
import numpy as np
import re

def extract_time_from_wv_filename(filename: str) -> Optional[int]:
    \"\"\"
    Extracts timestamp from S1 WV filename format: 
    e.g., s1a-wv2-ocn-vv-20250531t113616-20250531t113619...
    \"\"\"
    match = re.search(r'(\d{8}t\d{6})', filename)
    if match:
        time_str = match.group(1)
        dt = datetime.datetime.strptime(time_str, "%Y%m%dT%H%M%S")
        return int(dt.timestamp())
    return None

def extract_times_from_swot_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    \"\"\"
    Extracts start and end timestamps from SWOT L2 filename format:
    e.g., SWOT_L2_LR_SSH_WindWave_033_353_20250531T112541_20250531T121709...
    \"\"\"
    matches = re.findall(r'(\d{8}T\d{6})', filename)
    if len(matches) >= 2:
        start = datetime.datetime.strptime(matches[0], "%Y%m%dT%H%M%S")
        end = datetime.datetime.strptime(matches[1], "%Y%m%dT%H%M%S")
        return int(start.timestamp()), int(end.timestamp())
    elif len(matches) == 1:
        t = datetime.datetime.strptime(matches[0], "%Y%m%dT%H%M%S")
        val = int(t.timestamp())
        return val, val
    return None, None

def parse_wv_metadata(filepath: str) -> Dict[str, Any]:
    \"\"\"
    Extract timestamp and spatial footprint from S1 WV-L2F netCDF file.
    \"\"\"
    filename = os.path.basename(filepath)
    timestamp = None
    footprint = None

    try:
        with xr.open_dataset(filepath) as ds:
            if 'time' in ds.coords:
                t_val = ds['time'].values[0]
                if hasattr(t_val, 'astype'):
                    timestamp = np.datetime64(t_val).astype('datetime64[s]').astype(int)
                else:
                    timestamp = int(t_val)

            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                footprint = box(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    except Exception:
        pass

    if timestamp is None:
        timestamp = extract_time_from_wv_filename(filename)

    return {
        "filepath": filepath,
        "timestamp": timestamp,
        "footprint": footprint,
        "filename": filename
    }

def parse_swot_metadata(filepath: str) -> Dict[str, Any]:
    \"\"\"
    Extract timestamps and spatial footprint from SWOT L2 WindWave netCDF file.
    \"\"\"
    filename = os.path.basename(filepath)
    start_time = None
    end_time = None
    footprint = None

    try:
        with xr.open_dataset(filepath) as ds:
            if 'time' in ds.coords:
                t_vals = ds['time'].values
                if len(t_vals) >= 1:
                    s_val = t_vals[0]
                    if hasattr(s_val, 'astype'):
                        start_time = np.datetime64(s_val).astype('datetime64[s]').astype(int)
                    else:
                        start_time = int(s_val)
                if len(t_vals) > 1:
                    e_val = t_vals[-1]
                    if hasattr(e_val, 'astype'):
                        end_time = np.datetime64(e_val).astype('datetime64[s]').astype(int)
                    else:
                        end_time = int(e_val)

            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                lon_corr = np.where(lon >= 180, lon - 360, lon)
                footprint = box(float(lon_corr.min()), float(lat.min()), float(lon_corr.max()), float(lat.max()))
    except Exception:
        pass

    if start_time is None or end_time is None:
        s, e = extract_times_from_swot_filename(filename)
        if start_time is None: start_time = s
        if end_time is None: end_time = e

    return {
        "filepath": filepath,
        "start_time": start_time,
        "end_time": end_time,
        "footprint": footprint,
        "filename": filename
    }

def is_match(s1_meta: Dict[str, Any], swot_meta: Dict[str, Any], time_threshold_min: int = 10) -> bool:
    \"\"\"
    Check if S1 and SWOT records match temporally (S1 within SWOT window +/- threshold) 
    and spatially (intersect).
    \"\"\"
    t_s1 = s1_meta["timestamp"]
    t_start = swot_meta["start_time"]
    t_end = swot_meta["end_time"]

    if t_s1 is None or t_start is None or t_end is None:
        return False

    threshold_sec = time_threshold_min * 60
    if not (t_start - threshold_sec <= t_s1 <= t_end + threshold_sec):
        return False

    if s1_meta["footprint"] is None or swot_meta["footprint"] is None:
        return False
    
    if not s1_meta["footprint"].intersects(swot_meta["footprint"]):
        return False

    return True

def generate_stac_item(s1_meta: Dict[str, Any], swot_meta: Dict[str, Any]) -> Item:
    \"\"\"
    Create a STAC item representing the matchup.
    \"\"\"
    geometry = s1_meta["footprint"].__geo_interface__
    bbox = list(s1_meta["footprint"].bounds)
    stac_time = datetime.datetime.fromtimestamp(s1_meta["timestamp"])

    item = Item(id=f"matchup-{s1_meta['filename']}-{swot_meta['filename']}", 
                geometry=geometry,
                bbox=bbox,
                datetime=stac_time)
    
    item.assets["s1_wv"] = Asset(href=s1_meta["filepath"])
    item.assets["swot_l2"] = Asset(href=swot_meta["filepath"])
    
    return item

def find_matchups(wv_dir: str, swot_dir: str, config: Dict[str, Any]) -> List[Item]:
    \"\"\"
    Main entry point to scan directories and return list of matched STAC items.
    \"\"\"
    import glob
    matches = []
    time_threshold = config.get("TIME_THRESHOLD_MIN", 10)

    wv_files = glob.glob(os.path.join(wv_dir, "**/*.nc"), recursive=True)
    swot_files = glob.glob(os.path.join(swot_dir, "**/*.nc"), recursive=True)

    wv_metas = [parse_wv_metadata(f) for f in wv_files]
    swot_metas = [parse_swot_metadata(f) for f in swot_files]

    for s1 in wv_metas:
        for swot in swot_metas:
            if is_match(s1, swot, time_threshold):
                matches.append(generate_stac_item(s1, swot))
                
    return matches

if __name__ == \"__main__\":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wv_dir", required=True)
    parser.add_argument("--swot_dir", required=True)
    parser.add_argument("--output", required=True, help="Path to save STAC catalog/JSON")
    args = parser.parse_args()

    config = {"TIME_THRESHOLD_MIN": 10}
    results = find_matchups(args.wv_dir, args.swot_dir, config)
    
    import json
    with open(args.output, "w") as f:
        json.dump([item.to_dict() for item in results], f, indent=2)

