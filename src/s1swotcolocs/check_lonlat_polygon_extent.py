"""
set of methods to spot strange polygons in SWOT data
"""

import numpy as np
from shapely.geometry import Polygon


def angular_distance(lon1, lon2):
    """Compute shortest angular distance on a circle between two longitudes."""
    return min((lon2 - lon1) % 360, (lon1 - lon2) % 360)


def longitude_extent(lons):
    """Compute maximal longitudinal extent, accounting for wrap-around."""
    lons_360 = np.mod(lons, 360)
    lons_sorted = np.sort(lons_360)
    gaps = [
        (lons_sorted[(i + 1) % len(lons_sorted)] - lons_sorted[i]) % 360
        for i in range(len(lons_sorted))
    ]
    max_gap = max(gaps)
    return 360 - max_gap  # The minimal enclosing arc


def latitude_extent(lats):
    return np.max(lats) - np.min(lats)


def check_longitude_smaller_than_latitude_extent(polygon: Polygon) -> bool:
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Invalid or empty polygon.")

    lons, lats = np.array(polygon.exterior.coords.xy)
    lon_extent = longitude_extent(lons)
    lat_extent = latitude_extent(lats)

    print(f"Longitude extent: {lon_extent:.2f}°")
    print(f"Latitude extent:  {lat_extent:.2f}°")

    return lon_extent < lat_extent
