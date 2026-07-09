# -*- coding: utf-8 -*-
"""
matchups_WV_KaRIn_v2.py
====================
Collocate Sentinel-1 WaveMode (WV) L2 products with SWOT KaRIn L2_LR_SSH products.

Strategy (memory-efficient):
  - For each SWOT file, parse its date(s) from the filename → derive the
    day-of-year → look up the S1-WV daily files at their known path.
    No directory scanning of S1 data at all.
  - SWOT KaRIn files are large (~129 MB in memory) → read only the
    spatiotemporal metadata we need (lat/lon edges + time) WITHOUT loading
    the full dataset: lazy xarray + bbox pre-filter from global NetCDF
    attributes, then only swath-edge pixel columns are loaded.

Usage:
  python matchups_WV_KaRIn_v2.py \\
      --conf-file config.yml \\
      --start-date 2026-01-01 --stop-date 2026-01-10 --dev \\
      --output /path/to/output \\
      --log-level DEBUG
"""

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from s1swotcolocs.utils import get_conf_content

# ##################
# font properties #
# ##################

import matplotlib.font_manager as fm

# Regular and bold font paths

# font_regular_path = "/home1/datahome/msimonne/.fonts/times/times.ttf"
# font_bold_path = "/home1/datahome/msimonne/.fonts/times/timesbd.ttf"  # Bold
# fm.fontManager.addfont(font_bold_path)
# fm.fontManager.addfont(font_regular_path)
# # Create font properties
# font_regular = fm.FontProperties(fname=font_regular_path)
# font_bold = fm.FontProperties(fname=font_bold_path)
# # Set default font to regular Times New Roman
# plt.rcParams['font.family'] = font_regular.get_name()



# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  –  edit these paths / thresholds
# ═════════════════════════════════════════════════════════════════════════════

# Root directory containing SWOT KaRIn files (searched recursively for *.nc)
# Root directory containing SWOT KaRIn files (searched recursively for *.nc).
# rglob() recurses into all subdirectories (PID0, PGC0, ...) automatically.
# Do NOT add a trailing "/*": Path does not expand shell globs.
SWOT_ROOT = Path("/home/datawork-cersat-public/project/mpc-sentinel1/data/ancillary/SWOT_L2_KARIN_LR_WindWave_AVISO")

# Root of the S1-WV daily enriched file tree:
#   <S1_WV_ROOT>/<yyyy>/<doy>/S1A_WV_L2D_enriched_LOPS_<yyyymmdd>_daily_IPF_*.nc
S1_WV_ROOT = Path("/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/L2_daily/0.12")

# Output directory for STAC-like JSON matchup items
OUTPUT_DIR = Path("/tmp/matchups_S1wv_karin")

# Maximum time difference allowed between S1 acquisition and SWOT pass [minutes]
MAX_TIME_DIFF_MIN = 360 # 120 #30

# SWOT swath edge pixel indices (from plot_swot_footprint)
# col 0  = outer-left edge
# col 29 = inner-left edge (gap boundary)
# col 39 = inner-right edge (gap boundary)
# col -1 = outer-right edge
SWOT_EDGE_INDICES = [0, 29, 39, -1]


# ===================================
#
# ===================================

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


# ═════════════════════════════════════════════════════════════════════════════
# S1-WV file lookup  (direct path build — no rglob)
# ═════════════════════════════════════════════════════════════════════════════

def _doy(d: date) -> int:
    """Return the day-of-year (1-based) for a given date.

    Args:
        d (date): Input date.

    Returns:
        int: Day of year (1 … 366).
    """
    return d.timetuple().tm_yday


def find_s1_wv_files_for_swot(
    swot_t0: datetime,
    swot_t1: datetime,
    s1_root: Path,
) -> list[Path]:
    """Find S1-WV daily files that overlap the SWOT pass time window.

    Given the SWOT pass time window [swot_t0, swot_t1], return ALL S1-WV
    daily files that could contain concurrent acquisitions. For a given day
    there can be one file per active Sentinel-1 satellite (S1A, S1C, …).
    All are returned so that scenes from every satellite are tested.

    The file tree layout is:
        <s1_root>/<yyyy>/<ddd>/S1{A,C,D}_WV_L2D_enriched_LOPS_<yyyymmdd>_daily_IPF_*.nc

    We glob only within the (at most 2) known day subdirectories — no tree walk.

    Args:
        swot_t0 (datetime): Start time of the SWOT pass.
        swot_t1 (datetime): End time of the SWOT pass.
        s1_root (Path): Root directory of the S1-WV daily files.

    Returns:
        list[Path]: List of matching S1-WV file paths.
    """
    d0 = swot_t0.date()
    d1 = swot_t1.date()

    candidate_dates = sorted({d0, d1})   # set deduplicates same-day passes

    found = []
    for d in candidate_dates:
        subdir = s1_root / f"{d.year}" / f"{_doy(d):03d}"
        matches = sorted(subdir.glob(
            f"S1*_WV_L2D_enriched_LOPS_{d.strftime('%Y%m%d')}_daily_IPF_*.nc"
        ))
        if matches:
            # Keep ALL satellites (S1A, S1C, …) for this day
            found.extend(matches)
            log.debug("Found %d S1-WV file(s) for %s: %s", len(matches), d,
                      [p.name for p in matches])
        else:
            log.debug("No S1-WV file for %s at %s", d, subdir)

    return found


# ═════════════════════════════════════════════════════════════════════════════
# SWOT helpers
# ═════════════════════════════════════════════════════════════════════════════

_SWOT_FN_RE  = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")
_SWOT_FN_FMT = "%Y%m%dT%H%M%S"


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


def _swot_bbox_from_attrs(attrs: dict) -> tuple:
    """Read the global bounding box from SWOT NetCDF attributes.

    Args:
        attrs (dict): Global attributes of the SWOT dataset.

    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) as floats.
            Defaults to (-180, 180, -90, 90) if keys are missing.
    """
    return (
        float(attrs.get("geospatial_lon_min", -180)),
        float(attrs.get("geospatial_lon_max",  180)),
        float(attrs.get("geospatial_lat_min",  -90)),
        float(attrs.get("geospatial_lat_max",   90)),
    )


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
            return datetime.strptime(time_str, fmt_without_z).replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.strptime(time_str, fmt_with_z).replace(tzinfo=timezone.utc)

    t0 = parse_time(attrs["time_coverage_start"])
    t1 = parse_time(attrs["time_coverage_end"])
    return t0, t1


def _bbox_overlaps(
    lon_min_a, lon_max_a, lat_min_a, lat_max_a,
    lon_min_b, lon_max_b, lat_min_b, lat_max_b,
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
        lat_min_a <= lat_max_b and lat_max_a >= lat_min_b
        and lon_min_a <= lon_max_b and lon_max_a >= lon_min_b
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
    lons  = ds_swot.longitude.isel(num_pixels=SWOT_EDGE_INDICES).values  # (lines, 4)
    lons  = (lons + 180) % 360 - 180   # normalise to [-180, 180]  (fix 2.2)
    lats  = ds_swot.latitude.isel(num_pixels=SWOT_EDGE_INDICES).values   # (lines, 4)
    times = ds_swot.time.values                                            # (lines,)

    # Cap rows (top/bottom) connecting the two sub-swaths, mirroring
    # plot_swot_footprint: columns 0..28 and 39..end for first and last line.
    n_pix = ds_swot.sizes["num_pixels"]
    left_cols  = list(range(SWOT_EDGE_INDICES[0], SWOT_EDGE_INDICES[1]))
    right_cols = list(range(SWOT_EDGE_INDICES[2], n_pix))

    cap_lons = np.concatenate([
        ds_swot.longitude.isel(num_lines=0,  num_pixels=left_cols).values,
        ds_swot.longitude.isel(num_lines=0,  num_pixels=right_cols).values,
        ds_swot.longitude.isel(num_lines=-1, num_pixels=left_cols).values,
        ds_swot.longitude.isel(num_lines=-1, num_pixels=right_cols).values,
    ])
    cap_lons = (cap_lons + 180) % 360 - 180   # normalise

    cap_lats = np.concatenate([
        ds_swot.latitude.isel(num_lines=0,  num_pixels=left_cols).values,
        ds_swot.latitude.isel(num_lines=0,  num_pixels=right_cols).values,
        ds_swot.latitude.isel(num_lines=-1, num_pixels=left_cols).values,
        ds_swot.latitude.isel(num_lines=-1, num_pixels=right_cols).values,
    ])

    all_lons = np.concatenate([lons.ravel(), cap_lons])
    all_lats = np.concatenate([lats.ravel(), cap_lats])
    valid    = np.isfinite(all_lons) & np.isfinite(all_lats)

    # Row-level validity mask: a line is usable only if ALL 4 edge cols are finite.
    valid_rows = np.all(np.isfinite(lons) & np.isfinite(lats), axis=1)  # axis = 1: on regarde les 4 edges

    return {
        "lons":       lons,
        "lats":       lats,
        "valid_rows": valid_rows,
        "times":      times,
        "lon_min":    float(np.nanmin(all_lons[valid])),
        "lon_max":    float(np.nanmax(all_lons[valid])),
        "lat_min":    float(np.nanmin(all_lats[valid])),
        "lat_max":    float(np.nanmax(all_lats[valid])),
    }


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
    lons = edges["lons"]          # (n_lines, 4)
    lats = edges["lats"]
    ok   = edges["valid_rows"]    # (n_lines,) bool — True where all 4 cols finite

    lons_ok = _unwrap_longitudes(lons[ok])
    lats_ok = lats[ok]

    if lons_ok.shape[0] < 3:
        raise ValueError(
            f"Too few valid edge rows after NaN filtering ({lons_ok.shape[0]}); "
            "SWOT track may be entirely masked."
        )

    step = 50  # 1 point tous les 100 km

    left_outer  = list(zip(lons_ok[::step, 0], lats_ok[::step, 0]))
    left_inner  = list(zip(lons_ok[::step, 1], lats_ok[::step, 1]))
    right_inner = list(zip(lons_ok[::step, 2], lats_ok[::step, 2]))
    right_outer = list(zip(lons_ok[::step, 3], lats_ok[::step, 3]))

    poly_left  = _make_sub_polygon(left_outer  + list(reversed(left_inner)))
    poly_right = _make_sub_polygon(right_inner + list(reversed(right_outer)))

    candidates = [p for p in (poly_left, poly_right) if p is not None]
    if not candidates:
        raise ValueError("Both sub-swath polygons are empty after NaN filtering.")

    combined = unary_union(candidates)
    if not combined.is_valid:
        combined = make_valid(combined)
    return combined


def plot_multipolygon(mp, ax, label=None, color="blue"):
    """Plot a MultiPolygon on a Cartopy axes.

    Args:
        mp: MultiPolygon geometry.
        ax: Cartopy axes.
        label (str, optional): Label for the first polygon (for legend).
        color (str): Edge color.
    """
    for i, poly in enumerate(mp.geoms):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, label=label if i == 0 else None, transform=ccrs.PlateCarree())

        # Optional: show holes
        for interior in poly.interiors:
            hx, hy = interior.xy
            ax.plot(hx, hy, color=color, transform=ccrs.PlateCarree())


def _add_features(ax):
    """Add common Cartopy features (land, coastline, borders, gridlines).

    Args:
        ax: Cartopy axes.
    """
    ax.add_feature(cfeature.LAND, color="lightgrey", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False


def _unwrap_longitudes(lons):
    """Remove antimeridian jumps by unwrapping longitudes.

    This applies numpy.unwrap in radians along the first axis.

    Args:
        lons (np.ndarray): Array of longitudes in degrees.

    Returns:
        np.ndarray: Unwrapped longitudes in degrees.
    """
    lons = np.asarray(lons, dtype=float)
    rad = np.deg2rad(lons)
    unwrapped = np.rad2deg(np.unwrap(rad, axis=0))
    return unwrapped


def s1_unwrap_longitudes(lons):
    """Unwrap longitudes (wrapper around numpy.unwrap).

    Args:
        lons (np.ndarray): Array of longitudes.

    Returns:
        np.ndarray: Unwrapped longitudes in degrees.
    """
    lons = np.asarray(lons, dtype=float)
    return np.rad2deg(np.unwrap(np.deg2rad(lons)))


# ═════════════════════════════════════════════════════════════════════════════
# S1-WV helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_s1_wv(path: Path) -> pd.DataFrame:
    """Load a S1-WV daily enriched NetCDF into a tidy DataFrame.

    Only the fields needed for collocation are kept:
        time, lon, lat, wv_mode, sensor, subpath, path, and polygon vertices.

    Args:
        path (Path): Path to the NetCDF file.

    Returns:
        pd.DataFrame: DataFrame with one row per WV scene.
    """
    ds = xr.open_dataset(path)

    df = pd.DataFrame({
        "time":    pd.to_datetime(ds["fdatedt"].values, utc=True),
        "lon":     ds["lon"].values,
        "lat":     ds["lat"].values,
        "wv_mode": ds["wv_mode"].values,
        "sensor":  ds["sensor"].values,
        "subpath": ds["subpath"].values,
        "path": [path for k in range(ds.fdatedt.size)]
    })

    # polygon: (fdatedt, lonlat=2, polygonsize=5)  (5 because closed squares)
    poly = ds["polygon"].values    # (n, 2, 5)
    df["polygon_lons"] = list(poly[:, 0, :])
    df["polygon_lats"] = list(poly[:, 1, :])

    ds.close()
    return df


def s1_scene_polygon(row: pd.Series, swot_lon_max=179):
    """Build a Shapely Polygon for a single S1-WV scene.

    Handles antimeridian crossing by shifting coordinates based on the
    maximum longitude of the SWOT footprint.

    Args:
        row (pd.Series): Row from the S1-WV DataFrame containing
            'polygon_lons' and 'polygon_lats'.
        swot_lon_max (float): Maximum longitude of the SWOT footprint
            in [0, 360] space. If >= 180, work in [0, 360], otherwise in [-180, 180].

    Returns:
        Polygon: Valid Shapely polygon.
    """
    lons = np.asarray(row["polygon_lons"], dtype=float).copy()

    if swot_lon_max >= 180:
        # Work in [0,360]
        lons = (lons + 360) % 360
        # Crosses Greenwich?
        if np.ptp(lons) > 180:
            lons[lons < 180] += 360
    else:
        # Work in [-180,180]
        # Crosses the dateline?
        if np.ptp(lons) > 180:
            lons[lons > 0] -= 360

    poly = Polygon(zip(lons, row["polygon_lats"]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


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


# ═════════════════════════════════════════════════════════════════════════════
# Collocation core
# ═════════════════════════════════════════════════════════════════════════════



def collocate_swot_file(
    swot_path: Path,
    s1_root: Path,
    output_dir: Path,
    max_time_diff_min: float = MAX_TIME_DIFF_MIN,
    debug_image=False,
    debug_image_csv_suffix=""
) -> list[dict]:
    """Collocate one SWOT KaRIn file against the relevant S1-WV daily file(s).

    The S1-WV files are resolved directly from their known path structure.
    Returns a list of matchup dicts (also written as JSON to output_dir).

    Args:
        swot_path (Path): Path to the SWOT NetCDF file.
        s1_root (Path): Root directory of the S1-WV daily files.
        output_dir (Path): Where to write JSON matchups and debug images.
        max_time_diff_min (float): Maximum allowed time difference [minutes].
        debug_image (bool): If True, generate debug plots.
        debug_image_csv_suffix (str): Suffix for debug CSV files (unused).

    Returns:
        list[dict]: List of matchup item dictionaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matchups = []
    debug_image_csv_suffix = ""

    # ── 1. Parse SWOT time window from filename (zero I/O) ────────────────
    swot_t0_fn, swot_t1_fn = parse_swot_filename_times(swot_path.name)
    if swot_t0_fn is None:
        log.warning("Cannot parse time from SWOT filename: %s — skipping", swot_path.name)
        return matchups, debug_image_csv_suffix

    if np.abs(swot_t1_fn - swot_t0_fn) < timedelta(seconds=0.05):
        log.warning(f"SWOT t_start ({swot_t0_fn}) equals SWOT t_end ({swot_t1_fn}). Beginning of SCIENCE phase ?"
                    "\n %s - skipping", swot_path.name)
        return matchups, debug_image_csv_suffix

    log.info("Processing SWOT: %s", swot_path.name)

    # ── 2. Find the 1 or 2 relevant S1-WV daily files ─────────────────────
    s1_files = find_s1_wv_files_for_swot(swot_t0_fn, swot_t1_fn, s1_root)
    if not s1_files:
        log.info("  No S1-WV files found for this SWOT pass — skipping")
        return matchups, debug_image_csv_suffix

    # ── 3. Load S1-WV data (small, ~3 MB each) ────────────────────────────
    df_wv_parts = []
    for s1_path in s1_files:
        log.info("  Loading S1-WV: %s", s1_path.name)
        df_wv_parts.append(load_s1_wv(s1_path))
    df_wv = pd.concat(df_wv_parts, ignore_index=True)
    log.info("  %d S1-WV scenes to test", len(df_wv))

    if df_wv.empty:
        return matchups, debug_image_csv_suffix

    # ── 4. Open SWOT lazily; read only global attrs (header only) ─────────
    ds_swot = xr.open_dataset(swot_path)
    attrs   = ds_swot.attrs

    swot_lon_min, swot_lon_max, swot_lat_min, swot_lat_max = _swot_bbox_from_attrs(attrs)
    swot_t0, swot_t1 = robust_swot_time_from_attrs(attrs)
    time_margin = pd.Timedelta(minutes=max_time_diff_min)

    # ── 5. Attribute-level bbox pre-filter ────────────────────────────────

    any_bbox_ok = _bbox_overlaps(
        df_wv["lon"].min(), df_wv["lon"].max(),
        df_wv["lat"].min(), df_wv["lat"].max(),
        swot_lon_min, swot_lon_max, swot_lat_min, swot_lat_max,
    )
    if not any_bbox_ok:
        ds_swot.close()
        log.info("  Skipped: no S1-WV scenes overlap SWOT bbox")
        return matchups, debug_image_csv_suffix

    # ── 6. Load only swath-edge pixels ────────────────────────────────────
    log.info("  Loading SWOT edge pixels …")
    edges = extract_swot_edges(ds_swot)
    ds_swot.close()

    try:
        swot_poly = swot_footprint_polygon(edges)
    except Exception as exc:
        log.warning("  Could not build SWOT polygon: %s", exc)
        return matchups, debug_image_csv_suffix

    swot_times = pd.to_datetime(edges["times"], utc=True)

    # ── 7. Per-scene collocation ───────────────────────────────────────────
    for _, wv_row in df_wv.iterrows():

        # a) Time filter
        dt_series = np.abs(swot_times - wv_row["time"])
        min_dt    = dt_series.min()
        if min_dt > time_margin:
            continue

        # b) Bbox filter
        if not _bbox_overlaps(
            wv_row["lon"], wv_row["lon"],
            wv_row["lat"], wv_row["lat"],
            edges["lon_min"], edges["lon_max"],
            edges["lat_min"], edges["lat_max"],
        ):
            continue

        # c) Exact polygon intersection
        swot_lon_max_0_360 = np.max(_unwrap_longitudes(edges["lons"]))
        wv_poly = s1_scene_polygon(wv_row, swot_lon_max_0_360)
        if not swot_poly.intersects(wv_poly):
            continue

        intersection = swot_poly.intersection(wv_poly)
        if intersection.is_empty:
            continue

        # d) Overlap percentage
        pct = overlap_pct(intersection, wv_poly)

        # ── 8. Build STAC-like matchup Item ───────────────────────────────
        closest_line_idx = int(np.argmin(
            np.sum((wv_row["lon"] - edges["lons"])**2, axis=1) +
            np.sum((wv_row["lat"] - edges["lats"])**2, axis=1)
        ))
        swot_time_at_coloc = swot_times[closest_line_idx].isoformat()
        dt_at_coloc = dt_series[closest_line_idx]

        matchup_id = (
            f"S1WV_{wv_row['sensor']}_{wv_row['time'].strftime('%Y%m%dT%H%M%S')}"
            f"_SWOT_{swot_t0.strftime('%Y%m%dT%H%M%S')}"
        )

        if debug_image:
            debug_image_csv_suffix = "_debug_image"
            fig = plt.figure(figsize=(10,8))
            ax = plt.axes(projection=ccrs.Mercator())
            _add_features(ax)

            for geom, color, label in [(swot_poly, "blue", "KaRIn footprint"),
                                       (wv_poly, "red", "S1 WV mode footprint"),
                                       (intersection, "green", "Intersection")]:
                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=color, transform=ccrs.PlateCarree(), label=label)
                elif geom.geom_type == "MultiPolygon":
                    plot_multipolygon(geom, ax, label, color)

            plt.legend(loc="upper right")
            plt.title("SWOT KaRIn - Sentinel-1 WV mode collocation \n")
            fig.text(s=f"SWOT cycle: {attrs.get('cycle_number', '')}  --- SWOT pass: {attrs.get('pass_number', '')} --- SWOT crid: {attrs.get('crid', '')}", x=0.5, y=0.9)
            fig.text(s=f"SWOT KaRIn datetime: {swot_time_at_coloc} \n"
                       f"Sentinel-1 datetime: {wv_row['time'].isoformat()} \n"
                       f"Timedelta (minutes): {float(dt_at_coloc.total_seconds())//60} \n"
                       f"KaRIn/WV overlap(%): {round(pct, 2)} \n"
                       f"Sentinel-1X version: {str(wv_row['sensor'])}", x=0.1, y=0.7)

            ax.set_extent([wv_row["lon"]-10, wv_row["lon"]+10, wv_row["lat"]-10, wv_row["lat"]+10])
            plt.savefig(output_dir / "output_images" / f"debug_map_{matchup_id}.png", dpi=200)
            plt.close()

        bbox = list(intersection.bounds)

        item = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": matchup_id,
            "geometry": {
                "intersection": mapping(intersection),
                "wv_poly": mapping(wv_poly),
                "swot_poly": mapping(swot_poly),
            },
            "bbox": bbox,
            "properties": {
                "time_diff_seconds": float(dt_at_coloc.total_seconds()),
                "overlap_pct":       round(pct, 2),
                "s1_time":    wv_row["time"].isoformat(),
                "s1_lon":     float(wv_row["lon"]),
                "s1_lat":     float(wv_row["lat"]),
                "s1_wv_mode": str(wv_row["wv_mode"]),
                "s1_sensor":  str(wv_row["sensor"]),
                "s1_path": str(wv_row["path"]),
                "s1_subpath": str(wv_row["subpath"]),
                "swot_time_at_coloc": swot_time_at_coloc,
                "swot_time_start":    swot_t0.isoformat(),
                "swot_time_end":      swot_t1.isoformat(),
                "swot_cycle": attrs.get("cycle_number", ""),
                "swot_pass":  attrs.get("pass_number", ""),
                "swot_crid":  attrs.get("crid", ""),
            },
            "links": [],
            "assets": {
                "s1_wv": {
                    "href":  str(wv_row["path"]),
                    "type":  "application/x-netcdf",
                    "title": "Sentinel-1 WV L2 daily file(s)",
                },
                "swot_karin": {
                    "href":  str(swot_path),
                    "type":  "application/x-netcdf",
                    "title": "SWOT KaRIn L2_LR_SSH file",
                },
            },
        }

        yyyy = wv_row["time"].strftime("%Y")
        doy = wv_row["time"].strftime("%j")
        out_path = output_dir / "json_matchups" / yyyy / doy / f"{matchup_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as fh:
            json.dump(item, fh, indent=2, cls=NumpyEncoder)

        matchups.append(item)
        log.info(
            "  MATCHUP %-55s  Δt=%5.1f min  overlap=%5.1f%%",
            matchup_id, float(dt_at_coloc.total_seconds()) / 60, pct,
        )

    return matchups, debug_image_csv_suffix


def matchups_to_dataframe(matchups) -> pd.DataFrame:
    """Convert a list of matchup JSON items into a flat DataFrame.

    Args:
        matchups (list[dict]): List of matchup items (as output by collocate_swot_file).

    Returns:
        pd.DataFrame: DataFrame with one row per matchup and selected columns.
    """
    rows = []
    for m in matchups:
        p = m["properties"]
        rows.append({
            "id": m["id"],
            "s1_time": p["s1_time"],
            "swot_time_at_coloc": p["swot_time_at_coloc"],
            "time_diff_seconds": p["time_diff_seconds"],
            "overlap_pct": p["overlap_pct"],
            "s1_lon": p["s1_lon"],
            "s1_lat": p["s1_lat"],
            "s1_wv_mode": p["s1_wv_mode"],
            "s1_sensor": p["s1_sensor"],
            "swot_cycle": p["swot_cycle"],
            "swot_pass": p["swot_pass"],
            "swot_crid": p["swot_crid"],
            "s1_subpath": p["s1_subpath"],
            "s1_path": p["s1_path"],
            "swot_path": m["assets"]["swot_karin"]["href"],
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def find_swot_files(root: Path) -> list[Path]:
    """Recursively find all SWOT KaRIn NetCDF files under a root directory.

    Args:
        root (Path): Root directory to search.

    Returns:
        list[Path]: Sorted list of found file paths.
    """
    log.info("Searching for SWOT KaRIn L2_LR_SSH files in %s …", root)
    return sorted(root.rglob("SWOT_L2_LR_SSH_*.nc"))


def run(
    conf_file: Path,
    output_dir: Path = OUTPUT_DIR,
    max_time_diff_min: float = MAX_TIME_DIFF_MIN,
    start_date: str = None,
    stop_date: str = None,
    save_every: int = 100,
    dev: bool = False,
) -> int:
    """Main orchestration: find SWOT files, filter by date, and run collocation.

    Args:
        conf_file (Path): Path to a config YAML file (overrides defaults).
        output_dir (Path): Output directory for matchup JSON files and CSV summary.
        max_time_diff_min (float): Maximum allowed time difference [minutes].
        start_date (str): Start date (YYYY-MM-DD or YYYYMMDD).
        stop_date (str): Stop date (YYYY-MM-DD or YYYYMMDD).
        save_every (int): Save matchups to CSV every N SWOT files.
        dev (bool): If True, process only the first 2 SWOT files.

    Returns:
        int: Total number of matchups found.
    """
    conf = get_conf_content(conf_file) if conf_file else None
    swot_root = Path(conf.get("SWOT_L2_AVISO_DIR", SWOT_ROOT))
    s1_root   = Path(conf.get("S1_WV_ROOT", S1_WV_ROOT))
    swot_files = find_swot_files(swot_root)
    output_dir = Path(conf.get("WV_MATCHUP_OUTPUT_DIR", OUTPUT_DIR)) if output_dir is None else output_dir
    log.info("Found %d SWOT files before date filtering", len(swot_files))

    # ── Filter by date range ──────────────────────────────────────────────
    if start_date is not None or stop_date is not None:
        def parse_date_arg(date_str):
            for fmt in ("%Y-%m-%d", "%Y%m%d"):
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            raise ValueError(f"Date must be YYYY-MM-DD or YYYYMMDD, got {date_str}")

        start_dt = parse_date_arg(start_date) if start_date else datetime(1900, 1, 1, tzinfo=timezone.utc)
        stop_dt  = parse_date_arg(stop_date)  if stop_date  else datetime(2100, 1, 1, tzinfo=timezone.utc)

        if start_dt > stop_dt:
            raise ValueError("start-date cannot be after stop-date")

        filtered = []
        for f in swot_files:
            t0, _ = parse_swot_filename_times(f.name)
            if t0 is None:
                log.warning("Cannot parse date from %s — skipping", f.name)
                continue
            if start_dt <= t0 <= stop_dt:
                filtered.append(f)
        swot_files = filtered
        log.info("Filtered to %d SWOT files between %s and %s",
                 len(swot_files), start_dt.isoformat(), stop_dt.isoformat())

    if not swot_files:
        log.warning("No SWOT files to process")
        return 0

    # ── Development mode: restrict to 2 files ──────────────────────────────
    if dev:
        swot_files = swot_files[:2]
        log.info("Development mode: processing only the first 2 SWOT files")
    log.info("Processing %d SWOT files", len(swot_files))

    all_matchups = []
    corrupted_files = []
    total_matchups = 0

    csv_path = output_dir / "test_json_matchups.csv"
    csv_exists = csv_path.exists()
    debug_image_csv_suffix = ""

    for i, swot_path in enumerate(swot_files, start=1):
        try:
            matchups, debug_image_csv_suffix = collocate_swot_file(
                swot_path,
                s1_root=s1_root,
                output_dir=output_dir,
                max_time_diff_min=max_time_diff_min,
            )
            all_matchups.extend(matchups)
        except Exception as exc:
            log.exception("Fatal Bazooka error while processing %s", swot_path.name)
            corrupted_files.append({
                "filename": swot_path.name,
                "full_path": str(swot_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "timestamp_utc": datetime.utcnow().isoformat(),
            })

        # Save every N SWOT files (or after the last one)
        if i % save_every == 0 or i == len(swot_files):
            if all_matchups:
                df_summary = matchups_to_dataframe(all_matchups)
                df_summary.to_csv(
                    csv_path,
                    mode="a",
                    header=not csv_exists,
                    index=False,
                )
                csv_exists = True
                total_matchups += len(df_summary)
                log.info(
                    "Saved %d matchups after processing %d/%d SWOT files.",
                    len(df_summary),
                    i,
                    len(swot_files),
                )
                all_matchups.clear()

    log.info("Total matchups found: %d", total_matchups)

    # Save corrupted file list
    if corrupted_files:
        df_bad = pd.DataFrame(corrupted_files)
        bad_csv = output_dir / f"corrupted_swot_files{debug_image_csv_suffix}.csv"
        df_bad.to_csv(bad_csv, index=False)
        log.warning("Saved %d corrupted SWOT files to %s", len(df_bad), bad_csv)

    # Remove potential duplicates (due to relaunching after checkpoints)
    try:
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset="id")
        df.to_csv(csv_path, index=False)
    except FileNotFoundError:
        pass

    return total_matchups


def entrypoint():
    """Command-line entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collocate S1-WV L2 and SWOT KaRIn L2_LR_SSH products.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Output directory for matchup JSON files and summary CSV",
    )
    parser.add_argument(
        "--max-time-diff", type=float, default=MAX_TIME_DIFF_MIN,
        help="Maximum time difference between S1 and SWOT acquisitions [minutes]",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date (YYYY-MM-DD or YYYYMMDD) — only process SWOT files on/after this date"
    )
    parser.add_argument(
        "--stop-date", type=str, default=None,
        help="Stop date (YYYY-MM-DD or YYYYMMDD) — only process SWOT files on/before this date"
    )
    parser.add_argument(
        "--dev", action="store_true", default=False,
        help="Development mode: process only 2 SWOT files (useful for testing)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument("--conf-file", type=Path, help="Config YAML file (overrides defaults)")

    args = parser.parse_args()

    # ── Set logging level ─────────────────────────────────────────────
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    run(
        output_dir=args.output,
        max_time_diff_min=args.max_time_diff,
        start_date=args.start_date,
        stop_date=args.stop_date,
        dev=args.dev,
        conf_file=args.conf_file
    )


if __name__ == "__main__":
    entrypoint()