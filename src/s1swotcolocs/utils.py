import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from yaml import CLoader as Loader
from yaml import load

logger = logging.getLogger("s1swotcolocs.get_config_info")
logger.addHandler(logging.NullHandler())


# SWOT swath edge pixel indices (from plot_swot_footprint)
# col 0  = outer-left edge
# col 29 = inner-left edge (gap boundary)
# col 39 = inner-right edge (gap boundary)
# col -1 = outer-right edge
SWOT_EDGE_INDICES = [0, 29, 39, -1]

# def get_config_file_path():
#     # The configuration path is determined in the following order:
#     # 1. First, check the XSARSLC_CONFIG_PATH environment variable if it's set.
#     # 2. If not set, fall back to localconfig.yaml.
#     # 3. If neither is found, default to config.yaml.
#
#     default_local_config_path = os.path.join(
#         os.path.dirname(s1swotcolocs.__file__), "localconfig.yml"
#     )
#     default_config_path = os.path.join(os.path.dirname(s1swotcolocs.__file__), "config.yml")
#     potential_local_config_path = os.environ.get(
#         "XSARSLC_CONFIG_PATH", default_local_config_path
#     )
#
#     if os.path.exists(potential_local_config_path):
#         config_path = potential_local_config_path
#     else:
#         if os.path.exists(default_local_config_path):
#             config_path = default_local_config_path
#         else:
#             config_path = default_config_path
#
#     logger.info("Config path: %s", config_path)
#     return config_path

_SWOT_FN_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")
_SWOT_FN_FMT = "%Y%m%dT%H%M%S"


# ═════════════════════════════════════════════════════════════════════════════
# Overlap percentage
# ═════════════════════════════════════════════════════════════════════════════


def overlap_pct(intersection, wv_poly: Polygon) -> float:
    """Compute the fraction of the S1-WV scene covered by the SWOT swath.

    Uses geographic degree areas as an approximation (acceptable for small
    ~20 km × 20 km scenes). For rigorous results, project to an equal-area
    CRS.

    Args:
        intersection: Shapely geometry of the intersection.
        wv_poly (Polygon): S1-WV scene polygon.

    Returns:
        float: Overlap percentage (0‑100).
    """
    if wv_poly.area == 0:
        return 0.0
    return min(100.0, intersection.area / wv_poly.area * 100.0)


def get_conf_content(conf_path):
    # stream = open(get_config_file_path(), "r")
    stream = open(conf_path)
    conf = load(stream, Loader=Loader)
    return conf


def parse_swot_filename_times(fname: str) -> tuple:
    """Extract start and end times from a SWOT filename.

    Args:
        fname (str): SWOT filename.

    Returns:
        tuple: (t_start, t_end) as datetime objects in UTC, or (None, None)
            if the pattern is not found.
    """
    m = _SWOT_FN_RE.search(fname)
    if not m:
        return None, None
    t0 = datetime.strptime(m.group(1), _SWOT_FN_FMT).replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(m.group(2), _SWOT_FN_FMT).replace(tzinfo=timezone.utc)
    return t0, t1


def s1_unwrap_longitudes(lons):
    """Unwrap longitudes (wrapper around numpy.unwrap).

    Args:
        lons (np.ndarray): Array of longitudes.

    Returns:
        np.ndarray: Unwrapped longitudes in degrees.
    """
    lons = np.asarray(lons, dtype=float)
    return np.rad2deg(np.unwrap(np.deg2rad(lons)))


def _bbox_overlaps(
    lon_min_a,
    lon_max_a,
    lat_min_a,
    lat_max_a,
    lon_min_b,
    lon_max_b,
    lat_min_b,
    lat_max_b,
) -> bool:
    """Check if two bounding boxes overlap.

    Boxes that cross the antimeridian are expanded to the full (-180, 180)
    range to be conservative (avoid false negatives).

    Args:
        lon_min_a, lon_max_a, lat_min_a, lat_max_a: Bounding box A.
        lon_min_b, lon_max_b, lat_min_b, lat_max_b: Bounding box B.

    Returns:
        bool: True if the boxes overlap (or could overlap across the antimeridian).
    """

    def _norm(lmin, lmax):
        return (-180.0, 180.0) if lmin > lmax else (lmin, lmax)

    lon_min_a, lon_max_a = _norm(lon_min_a, lon_max_a)
    lon_min_b, lon_max_b = _norm(lon_min_b, lon_max_b)

    return (
        lat_min_a <= lat_max_b
        and lat_max_a >= lat_min_b
        and lon_min_a <= lon_max_b
        and lon_max_a >= lon_min_b
    )


def extract_swot_edges(ds_swot: xr.Dataset) -> dict:
    """Extract only the SWOT swath-edge pixel coordinates and times.

    Longitudes are normalised to [-180, 180]. Only the four edge columns
    (outer-left, inner-left, inner-right, outer-right) are kept. The cap
    rows (first and last line) are also included for polygon closure.

    Args:
        ds_swot (xr.Dataset): SWOT dataset (lazily loaded).

    Returns:
        dict: Contains:
            - 'lons': (n_lines, 4) array of longitudes at edges.
            - 'lats': (n_lines, 4) array of latitudes at edges.
            - 'valid_rows': (n_lines,) boolean array where all 4 edge values are finite.
            - 'times': (n_lines,) datetime64 array of SWOT time per line.
            - 'lon_min', 'lon_max', 'lat_min', 'lat_max': tight bbox of all valid edge pixels.
    """
    lons = ds_swot.longitude.isel(num_pixels=SWOT_EDGE_INDICES).values  # (lines, 4)
    lons = (lons + 180) % 360 - 180  # normalise to [-180, 180]  (fix 2.2)
    lats = ds_swot.latitude.isel(num_pixels=SWOT_EDGE_INDICES).values  # (lines, 4)
    times = ds_swot.time.values  # (lines,)

    # Cap rows (top/bottom) connecting the two sub-swaths, mirroring
    # plot_swot_footprint: columns 0..28 and 39..end for first and last line.
    n_pix = ds_swot.sizes["num_pixels"]
    left_cols = list(range(SWOT_EDGE_INDICES[0], SWOT_EDGE_INDICES[1]))
    right_cols = list(range(SWOT_EDGE_INDICES[2], n_pix))

    cap_lons = np.concatenate(
        [
            ds_swot.longitude.isel(num_lines=0, num_pixels=left_cols).values,
            ds_swot.longitude.isel(num_lines=0, num_pixels=right_cols).values,
            ds_swot.longitude.isel(num_lines=-1, num_pixels=left_cols).values,
            ds_swot.longitude.isel(num_lines=-1, num_pixels=right_cols).values,
        ]
    )
    cap_lons = (cap_lons + 180) % 360 - 180  # normalise

    cap_lats = np.concatenate(
        [
            ds_swot.latitude.isel(num_lines=0, num_pixels=left_cols).values,
            ds_swot.latitude.isel(num_lines=0, num_pixels=right_cols).values,
            ds_swot.latitude.isel(num_lines=-1, num_pixels=left_cols).values,
            ds_swot.latitude.isel(num_lines=-1, num_pixels=right_cols).values,
        ]
    )

    all_lons = np.concatenate([lons.ravel(), cap_lons])
    all_lats = np.concatenate([lats.ravel(), cap_lats])
    valid = np.isfinite(all_lons) & np.isfinite(all_lats)

    # Row-level validity mask: a line is usable only if ALL 4 edge cols are finite.
    valid_rows = np.all(
        np.isfinite(lons) & np.isfinite(lats), axis=1
    )  # axis = 1: on regarde les 4 edges

    return {
        "lons": lons,
        "lats": lats,
        "valid_rows": valid_rows,
        "times": times,
        "lon_min": float(np.nanmin(all_lons[valid])),
        "lon_max": float(np.nanmax(all_lons[valid])),
        "lat_min": float(np.nanmin(all_lats[valid])),
        "lat_max": float(np.nanmax(all_lats[valid])),
    }


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy integer and float types.

    This encoder converts NumPy integers and floating-point numbers to
    native Python types, and converts NumPy arrays to lists.

    Attributes:
        Inherits from json.JSONEncoder.
    """

    def default(self, obj):
        """Convert NumPy objects to JSON-serializable types.

        Args:
            obj: Object to encode.

        Returns:
            JSON-serializable representation of `obj`.

        Raises:
            TypeError: If `obj` is not supported by the parent encoder.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def swot_footprint_polygon(edges: dict):
    """Build the complete SWOT footprint polygon from edge points.

    The edge dictionary must contain 'lons' and 'lats' arrays of shape
    (n_lines, 4) with columns: outer-left, inner-left, inner-right, outer-right.
    Rows with NaN in any edge are dropped before constructing polygons,
    and a subsampling step of 50 lines is applied to reduce complexity.

    Args:
        edges (dict): Output from extract_swot_edges().

    Returns:
        Polygon or MultiPolygon: The combined footprint geometry.

    Raises:
        ValueError: If too few valid edge rows remain, or both sub-swaths
            are empty after filtering.
    """
    lons = edges["lons"]  # (n_lines, 4)
    lats = edges["lats"]
    ok = edges["valid_rows"]  # (n_lines,) bool — True where all 4 cols finite

    lons_ok = s1_unwrap_longitudes(lons[ok])
    lats_ok = lats[ok]

    if lons_ok.shape[0] < 3:
        raise ValueError(
            f"Too few valid edge rows after NaN filtering ({lons_ok.shape[0]}); "
            "SWOT track may be entirely masked."
        )

    step = 50  # 1 point tous les 100 km

    left_outer = list(zip(lons_ok[::step, 0], lats_ok[::step, 0]))
    left_inner = list(zip(lons_ok[::step, 1], lats_ok[::step, 1]))
    right_inner = list(zip(lons_ok[::step, 2], lats_ok[::step, 2]))
    right_outer = list(zip(lons_ok[::step, 3], lats_ok[::step, 3]))

    poly_left = _make_sub_polygon(left_outer + list(reversed(left_inner)))
    poly_right = _make_sub_polygon(right_inner + list(reversed(right_outer)))

    candidates = [p for p in (poly_left, poly_right) if p is not None]
    if not candidates:
        raise ValueError("Both sub-swath polygons are empty after NaN filtering.")

    combined = unary_union(candidates)
    if not combined.is_valid:
        combined = make_valid(combined)
    return combined


def _make_sub_polygon(coords: list) -> Polygon | None:
    """Build a valid polygon from a list of (lon, lat) coordinates.

    Applies make_valid() to repair geometry issues. If the result is a
    GeometryCollection, it extracts only Polygons and unions them.

    Args:
        coords (list): List of (lon, lat) coordinate pairs.

    Returns:
        Polygon | None: A valid Shapely Polygon, or None if fewer than 3
            valid points remain.
    """
    # Drop any pairs that slipped through with non-finite values
    clean = [(x, y) for x, y in coords if np.isfinite(x) and np.isfinite(y)]
    if len(clean) < 3:
        return None
    poly = Polygon(clean)
    if not poly.is_valid:
        poly = make_valid(poly)
    # make_valid may return a GeometryCollection; extract only Polygons
    if poly.geom_type == "GeometryCollection":
        polys = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        poly = unary_union(polys)
    return poly


def robust_swot_time_from_attrs(attrs: dict) -> tuple:
    """Parse time_coverage_start and time_coverage_end from global attributes.

    Supports both formats: "%Y-%m-%dT%H:%M:%S.%f" and with trailing 'Z'.

    Args:
        attrs (dict): Global attributes of the SWOT dataset.

    Returns:
        tuple: (t_start, t_end) as timezone-aware datetime objects (UTC).

    Raises:
        KeyError: If 'time_coverage_start' or 'time_coverage_end' is missing.
    """
    fmt_without_z = "%Y-%m-%dT%H:%M:%S.%f"
    fmt_with_z = "%Y-%m-%dT%H:%M:%S.%fZ"

    def parse_time(time_str: str) -> datetime:
        try:
            return datetime.strptime(time_str, fmt_without_z).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return datetime.strptime(time_str, fmt_with_z).replace(tzinfo=timezone.utc)

    t0 = parse_time(attrs["time_coverage_start"])
    t1 = parse_time(attrs["time_coverage_end"])
    return t0, t1
