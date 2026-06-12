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

def extract_time_from_swot_filename(filename: str) -> Optional[int]:
    \"\"\"
    Extracts start timestamp from SWOT L2 filename format:
    e.g., SWOT_L2_LR_SSH_WindWave_033_353_20250531T112541_...
    \"\"\"
    match = re.search(r'(\d{8}T\d{6})', filename)
    if match:
        time_str = match.group(1)
        dt = datetime.datetime.strptime(time_str, "%Y%m%dT%H%M%S")
        return int(dt.timestamp())
    return None

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
                # Handle numpy datetime64
                if hasattr(t_val, 'astype'):
                    timestamp = np.datetime64(t_val).astype('datetime64[s]').astype(int)
                else:
                    timestamp = t_val

            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                footprint = box(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    except Exception as e:
        # Log error or handle it; fallback to filename
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
    Extract timestamp and spatial footprint from SWOT L2 WindWave netCDF file.
    \"\"\"
    filename = os.path.basename(filepath)
    timestamp = None
    footprint = None

    try:
        with xr.open_dataset(filepath) as ds:
            # Handle longitude shift (Standard in this repo for SWOT)
            if 'longitude' in ds.coords:
                lon = ds['longitude'].values
                # Adjust 0-360 to -180 to 180 if necessary
                lon = np.where(lon >= 180, lon - 360, lon)
                
            if 'time' in ds.coords:
                t_val = ds['time'].values[0]
                if hasattr(t_val, 'astype'):
                    timestamp = np.datetime64(t_val).astype('datetime64[s]').astype(int)
                else:
                    timestamp = t_val

            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lat = ds['latitude'].values
                # use the corrected lon from above
                lon_corr = np.where(ds['longitude'].values >= 180, ds['longitude'].values - 360, ds['longitude'].values)
                footprint = box(float(lon_corr.min()), float(lat.min()), float(lon_corr.max()), float(lat.max()))
    except Exception as e:
        pass

    if timestamp is None:
        timestamp = extract_time_from_swot_filename(filename)

    return {
        "filepath": filepath,
        "timestamp": timestamp,
        "footprint": footprint,
        "filename": filename
    }

def is_match(s1_meta: Dict[str, Any], swot_meta: Dict[str, Any], time_threshold_min: int = 10) -> bool:
    \"\"\"
    Check if S1 and SWOT records match temporally (< threshold) and spatially (intersect).
    \"\"\"
    if s1_meta["timestamp"] is None or swot_meta["timestamp"] is None:
        return False

    # Temporal check
    time_diff = abs(s1_meta["timestamp"] - swot_meta["timestamp"])
    if time_diff > (time_threshold_min * 60):
        return False

    # Spatial check
    if s1_meta["footprint"] is None or swot_meta["footprint"] is None:
        return False
    
    if not s1_meta["footprint"].intersects(swot_meta["footprint"]):
        return False

    return True

def generate_stac_item(s1_meta: Dict[str, Any], swot_meta: Dict[str, Any]) -> Item:
    \"\"\"
    Create a STAC item representing the matchup.
    \"\"\"
    # Use s1 footprint as primary geometry for the matchup record
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

    # Pre-parse metadata to avoid repeated file reads
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
