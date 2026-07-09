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
      --swot-root /path/to/SWOT_L2_KARIN_LR_WindWave_AVISO \\
      --s1-root   /path/to/s1/L2_daily/0.12 \\
      --output    ./matchups \\
      --max-time-diff 30
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
    def default(self, obj):
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
    """Return the day-of-year (1-based) for a date object."""
    return d.timetuple().tm_yday


def find_s1_wv_files_for_swot(
    swot_t0: datetime,
    swot_t1: datetime,
    s1_root: Path,
) -> list[Path]:
    """
    Given the SWOT pass time window [swot_t0, swot_t1], return ALL S1-WV
    daily files that could contain concurrent acquisitions.

    For a given day there can be one file per active Sentinel-1 satellite
    (S1A, S1C, and eventually S1D).  All of them are returned so that scenes
    from every satellite are tested for collocation.

    File tree layout:
        <s1_root>/<yyyy>/<ddd>/S1{A,C,D}_WV_L2D_enriched_LOPS_<yyyymmdd>_daily_IPF_*.nc

    We glob only within the (at most 2) known day subdirectories — no tree walk.
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
    """Extract (t_start, t_end) from a SWOT filename, or (None, None)."""
    m = _SWOT_FN_RE.search(fname)
    if not m:
        return None, None
    t0 = datetime.strptime(m.group(1), _SWOT_FN_FMT).replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(m.group(2), _SWOT_FN_FMT).replace(tzinfo=timezone.utc)
    return t0, t1


def _swot_bbox_from_attrs(attrs: dict) -> tuple:
    """Read the global bbox from SWOT NetCDF header. Returns (lon_min, lon_max, lat_min, lat_max)."""
    return (
        float(attrs.get("geospatial_lon_min", -180)),  # dict.get("key", default_value if "key" not in dict)
        float(attrs.get("geospatial_lon_max",  180)),
        float(attrs.get("geospatial_lat_min",  -90)),
        float(attrs.get("geospatial_lat_max",   90)),
    )


def robust_swot_time_from_attrs(attrs: dict) -> tuple:
    """Parse time_coverage_start / _end from global attributes."""
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
    """
    Bounding-box overlap check. Antimeridian-crossing boxes are expanded to
    (-180, 180) — conservative but avoids false negatives.
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
    """
    Load only the SWOT swath-edge pixels into memory (tiny fraction of the
    full 129 MB dataset).

    Longitude normalisation (2.2): the formula (lon + 180) % 360 - 180 maps
    any value in [0, 360] to [-180, 180] and is a strict no-op for values
    already in [-180, 180].  It is always applied as a safety measure.

    Returns a dict:
        lons, lats  : (n_lines, 4) arrays  — the 4 edge columns, NaN where masked
        valid_mask  : (n_lines,) bool — True where both edge cols are finite
        times       : (n_lines,)  datetime64
        lon_min/max, lat_min/max : tight bbox of all valid edge pixels
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
    cap_lons = (cap_lons + 180) % 360 - 180   # normalise  (fix 2.2)

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
    # This is used in swot_footprint_polygon to drop masked/cut lines (fix 2.1).
    valid_rows = np.all(np.isfinite(lons) & np.isfinite(lats), axis=1)  # axis = 1: on regarde les 4 edges # (lines,) 

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
    """
    Build a single sub-swath polygon from a list of (lon, lat) coordinate pairs,
    applying make_valid() to repair any geometry issues caused by cut/partial tracks.
    Returns None if fewer than 3 valid points remain after NaN filtering.
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
    """
    Build a Shapely geometry from SWOT swath edges.

    Column layout of edges["lons"] / edges["lats"] (4 cols):
        0 → outer-left   1 → inner-left   2 → inner-right   3 → outer-right

    Fix 2.1: rows where any of the 4 edge columns is NaN (masked pixels on
    cut/partial SWOT tracks, over land, or at the polar boundary) are dropped
    before constructing the polygon rings.  make_valid() is applied to each
    sub-polygon before the union to avoid TopologyException.
    """
    lons = edges["lons"]          # (n_lines, 4)
    lats = edges["lats"]
    ok   = edges["valid_rows"]    # (n_lines,) bool — True where all 4 cols finite

    #lons_ok = lons[ok]
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
    for i, poly in enumerate(mp.geoms):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, label=label if i == 0 else None, transform=ccrs.PlateCarree())

        # Optional: show holes
        for interior in poly.interiors:
            hx, hy = interior.xy
            ax.plot(hx, hy, color=color, transform=ccrs.PlateCarree())

def _add_features(ax):
    ax.add_feature(cfeature.LAND, color="lightgrey", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False


def _unwrap_longitudes(lons):
    
    #Remove antimeridian jumps by working in a continuous longitude space.
    
    lons = np.asarray(lons, dtype=float)

    rad = np.deg2rad(lons)
    unwrapped = np.rad2deg(np.unwrap(rad, axis=0))

    return unwrapped


def s1_unwrap_longitudes(lons):
    lons = np.asarray(lons, dtype=float)
    return np.rad2deg(np.unwrap(np.deg2rad(lons)))



# ═════════════════════════════════════════════════════════════════════════════
# S1-WV helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_s1_wv(path: Path) -> pd.DataFrame:
    """
    Load a S1-WV daily enriched NetCDF into a tidy DataFrame.
    Only the fields needed for collocation are kept.
    """
    ds = xr.open_dataset(path)

    df = pd.DataFrame({
        "time":    pd.to_datetime(ds["fdatedt"].values, utc=True),
        #"lon":     _unwrap_longitudes(ds["lon"].values),
        "lon":     ds["lon"].values,
        "lat":     ds["lat"].values,
        #"lon_min": ds["lonmin"].values,   #  inutile
        #"lon_max": ds["lonmax"].values,
        #"lat_min": ds["latmin"].values,
        # "lat_max": ds["latmax"].values,
        "wv_mode": ds["wv_mode"].values,
        "sensor":  ds["sensor"].values,
        "subpath": ds["subpath"].values,   # ESA's native storage ? (pas sûr de moi mais en tous cas c'est sous forme de .SAFE)
        "path": [path for k in range(ds.fdatedt.size)]    # le chemin sur Datarmor

    })

    # polygon: (fdatedt, lonlat=2, polygonsize=5)  (5 because closed squares ==> 4 vertices + 1 for closing)
    poly = ds["polygon"].values    # (n, 2, 5)
    #df["polygon_lons"] = list(_unwrap_longitudes(poly[:, 0, :]))
    df["polygon_lons"] = list(poly[:, 0, :])
    df["polygon_lats"] = list(poly[:, 1, :])

    ds.close()
    return df


def s1_scene_polygon(row: pd.Series, swot_lon_max=179):

    lons = np.asarray(row["polygon_lons"], dtype=float).copy()  # -0.1, 0.1 ou 179.9, -179.9 

    if swot_lon_max >= 180:
        # Work in [0,360]
        lons = (lons + 360) % 360  # 359.9, 0.1  ou 179.9, 180.1

        # Crosses Greenwich?
        if np.ptp(lons) > 180:   
            lons[lons < 180] += 360  # 359.9, 360.1

    else:
        # Work in [-180,180]  ==>  # -0.1, 0.1 ou 179.9, -179.9 

        # Crosses the dateline?
        if np.ptp(lons) > 180:  
            lons[lons > 0] -= 360    # -180.1, 179.9
 
    poly = Polygon(zip(lons, row["polygon_lats"]))

    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly

def view_s1_scene_polygon(ds_wv, swot_lon_max=179):
    poly = ds_wv["polygon"].values
    lons, lats = poly[0, :], poly[1, :]
    #lons = np.asarray(row["polygon_lons"], dtype=float).copy()  # -0.1, 0.1 ou 179.9, -179.9 
    if swot_lon_max >= 180:
        # Work in [0,360]
        lons = (lons + 360) % 360  # 359.9, 0.1  ou 179.9, 180.1

        # Crosses Greenwich?
        if np.ptp(lons) > 180:   
            lons[lons < 180] += 360  # 359.9, 360.1

    else:
        # Work in [-180,180]  ==>  # -0.1, 0.1 ou 179.9, -179.9 

        # Crosses the dateline?
        if np.ptp(lons) > 180:  
            lons[lons > 0] -= 360    # -180.1, 179.9
 
    wv_poly = Polygon(zip(lons, lats))

    if not wv_poly.is_valid:
        wv_poly = wv_poly.buffer(0)

    return wv_poly


# ═════════════════════════════════════════════════════════════════════════════
# Overlap percentage
# ═════════════════════════════════════════════════════════════════════════════

def overlap_pct(intersection, wv_poly: Polygon) -> float:
    """
    Fraction of the S1-WV scene covered by the SWOT swath, in percent.

        overlap_pct = intersection.area / wv_poly.area * 100

    Areas are in geographic degrees (not equal-area), which is an acceptable
    approximation for the small WV scene (~20 km × 20 km).  For a rigorous
    result, project to a local equal-area CRS first.
    """
    if wv_poly.area == 0:
        return 0.0
    return min(100.0, intersection.area / wv_poly.area * 100.0)


# ═════════════════════════════════════════════════════════════════════════════
# Collocation core
# ═════════════════════════════════════════════════════════════════════════════

Matu_thoughts = True

def collocate_swot_file(
    swot_path: Path,
    s1_root: Path,
    output_dir: Path,
    max_time_diff_min: float = MAX_TIME_DIFF_MIN,
    debug_image=False,
    debug_image_csv_suffix = ""
) -> list[dict]:
    """
    Collocate one SWOT KaRIn file against the relevant S1-WV daily file(s).

    The S1-WV files are resolved directly from their known path structure —
    no directory scanning required.

    Returns a list of matchup dicts (also written as JSON to output_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True) # obsolète (?) 
    matchups = []
    debug_image_csv_suffix = ""

    # ── 1. Parse SWOT time window from filename (zero I/O) ────────────────
    swot_t0_fn, swot_t1_fn = parse_swot_filename_times(swot_path.name)
    if swot_t0_fn is None:
        log.warning("Cannot parse time from SWOT filename: %s — skipping", swot_path.name)
        return matchups
    
    if np.abs(swot_t1_fn - swot_t0_fn) < timedelta(seconds=0.05):
        log.warning(f"SWOT t_start ({swot_t0_fn}) equals SWOT t_end ({swot_t1_fn}). Beginning of SCIENCE phase ?" 
                    "\n %s - skipping", swot_path.name) 
        return matchups
    
    log.info("Processing SWOT: %s", swot_path.name)

    # ── 2. Find the 1 or 2 relevant S1-WV daily files ─────────────────────
    s1_files = find_s1_wv_files_for_swot(swot_t0_fn, swot_t1_fn, s1_root)
    if not s1_files:
        log.info("  No S1-WV files found for this SWOT pass — skipping")
        return matchups

    # ── 3. Load S1-WV data (small, ~3 MB each) ────────────────────────────
    df_wv_parts = []
    for s1_path in s1_files:
        log.info("  Loading S1-WV: %s", s1_path.name)
        df_wv_parts.append(load_s1_wv(s1_path))
    df_wv = pd.concat(df_wv_parts, ignore_index=True)
    log.info("  %d S1-WV scenes to test", len(df_wv))

    if df_wv.empty:
        return matchups

    # ── 4. Open SWOT lazily; read only global attrs (header only) ─────────
    ds_swot = xr.open_dataset(swot_path)
    attrs   = ds_swot.attrs

    swot_lon_min, swot_lon_max, swot_lat_min, swot_lat_max = _swot_bbox_from_attrs(attrs)
    swot_t0, swot_t1 = robust_swot_time_from_attrs(attrs)
    time_margin = pd.Timedelta(minutes=max_time_diff_min)

    # ── 5. Attribute-level bbox pre-filter ────────────────────────────────
    
    ### Matu comment:
    ### Useless because each S1 file is a daily file: in one day, the Earth rotates completely. S1 nodal period = 1h38, so if there is continuously acquisition, lon_min, lon_max
    ### will always be very close to [-180, 180], which will, obviously, intersects with SWOT's huge half orbit [lon_min, lon_max] footprint. 
    
    if Matu_thoughts:
        pass
    else:
        any_bbox_ok = _bbox_overlaps(
            df_wv["lon"].min(), df_wv["lon"].max(),
            df_wv["lat"].min(), df_wv["lat"].max(),
            swot_lon_min, swot_lon_max, swot_lat_min, swot_lat_max,
        )
        if not any_bbox_ok:
            ds_swot.close()
            log.info("  Skipped: no S1-WV scenes overlap SWOT bbox")
            return matchups


    # ── 6. Load only swath-edge pixels ────────────────────────────────────
    log.info("  Loading SWOT edge pixels …")
    edges = extract_swot_edges(ds_swot)
    ds_swot.close()

    try:
        swot_poly = swot_footprint_polygon(edges)
    except Exception as exc:
        log.warning("  Could not build SWOT polygon: %s", exc)
        return matchups

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
            wv_row["lon"], wv_row["lon"],  # dans [-180, 180]
            wv_row["lat"], wv_row["lat"],
            edges["lon_min"],  edges["lon_max"],  # dans [-180, 180]
            edges["lat_min"],  edges["lat_max"],
        ):
            continue

        # c) Exact polygon intersection
        swot_lon_max_0_360 = np.max(_unwrap_longitudes(edges["lons"]))   # on regarde le max dans [0,360]
        wv_poly = s1_scene_polygon(wv_row, swot_lon_max_0_360)   # disjonction de cas selon swot_lon_max_0_360 > 180 ° 
        if not swot_poly.intersects(wv_poly):
            continue

        intersection = swot_poly.intersection(wv_poly)
        if intersection.is_empty:
            continue
        #print("#### swot_poly ####", swot_poly)
        #print("-x-x-x- intersection -x-x-x-", intersection)
        # d) Overlap percentage: what fraction of the WV scene is covered?
        pct = overlap_pct(intersection, wv_poly)

        # ── 8. Build STAC-like matchup Item ───────────────────────────────
        # Normalement, ça revient à calculer la somme de la distance du centre de la WV aux 4 SWOT edges de la line i (axis=1). On a donc une shape (n_lines,)
        # Puis de prendre l'argmin de ce truc, qui, a priori, donne l'endroit où la coloc a lieu spatialement. 
        closest_line_idx   = int(np.argmin(np.sum((wv_row["lon"] - edges["lons"])**2,axis=1) + np.sum((wv_row["lat"] - edges["lats"])**2,axis=1))) # int(dt_series.argmin())
        swot_time_at_coloc = swot_times[closest_line_idx].isoformat()
        dt_at_coloc = dt_series[closest_line_idx]#.isoformat()
        print(type(dt_at_coloc))
        print('dt_at_coloc_isoformat', dt_series[closest_line_idx].isoformat())

        matchup_id = (
            f"S1WV_{wv_row['sensor']}_{wv_row['time'].strftime('%Y%m%dT%H%M%S')}"
            f"_SWOT_{swot_t0.strftime('%Y%m%dT%H%M%S')}"
        )

        
        debug_image = False

        if debug_image:
            debug_image_csv_suffix = "_debug_image"
            fig = plt.figure(figsize=(10,8))
            ax = plt.axes(projection=ccrs.Mercator())
            
            #ax.coastlines()
            _add_features(ax)
            
            for geom, color, label in [(swot_poly, "blue", "KaRIn footprint"), (wv_poly, "red", "S1 WV mode footprint"), (intersection, "green", "Intersection")]:
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


        """
        plot_multipolygon(swot_poly, ax, "SWOT")
        
        if not intersection.is_empty:
            if intersection.geom_type == "Polygon":
                x, y = intersection.exterior.xy
                ax.plot(x, y, linewidth=2, label="Intersection")
            elif intersection.geom_type == "MultiPolygon":
                plot_multipolygon(intersection, ax, "Intersection")
        
        ax.set_aspect("equal")
        plt.savefig(output_dir / "output_images" / "debug_swot.png", dpi=200)
        plt.close()
        """

        
        bbox = list(intersection.bounds)  # [lon_min, lat_min, lon_max, lat_max]

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
                #"datetime":          wv_row["time"].isoformat(),  # inutile, c'est la même chose que "s1_time"
                "time_diff_seconds": float(dt_at_coloc.total_seconds()),
                "overlap_pct":       round(pct, 2),
                # S1-WV
                "s1_time":    wv_row["time"].isoformat(),
                "s1_lon":     float(wv_row["lon"]),
                "s1_lat":     float(wv_row["lat"]),
                #"s1_lon_min": float(wv_row["lon_min"]),  # inutile
                #"s1_lon_max": float(wv_row["lon_max"]),
                #"s1_lat_min": float(wv_row["lat_min"]),
                #"s1_lat_max": float(wv_row["lat_max"]),
                "s1_wv_mode": str(wv_row["wv_mode"]),
                "s1_sensor":  str(wv_row["sensor"]),
                "s1_path": str(wv_row["path"]),
                "s1_subpath": str(wv_row["subpath"]),
                # SWOT
                "swot_time_at_coloc": swot_time_at_coloc,
                "swot_time_start":    swot_t0.isoformat(),
                "swot_time_end":      swot_t1.isoformat(),
                #"swot_lon_min": float(edges["lon_min"]),   # inutile
                #"swot_lon_max": float(edges["lon_max"]),
                #"swot_lat_min": float(edges["lat_min"]),
                #"swot_lat_max": float(edges["lat_max"]),
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

        # Calcul de yyyy et doy
        yyyy = wv_row["time"].strftime("%Y")
        doy = wv_row["time"].strftime("%j")  # %j = jour de l'année (001-366)

        out_path = output_dir / "json_matchups" / yyyy / doy / f"{matchup_id}.json"
        
        # Création des dossiers parents (s'ils n'existent pas)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as fh:
            #json.dump(item, fh, indent=2)
            json.dump(item, fh, indent=2, cls=NumpyEncoder)

        matchups.append(item)
        log.info(
            "  MATCHUP %-55s  Δt=%5.1f min  overlap=%5.1f%%",
            matchup_id, float(dt_at_coloc.total_seconds()) / 60, pct,
        )

    return matchups, debug_image_csv_suffix


def matchups_to_dataframe(matchups):
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
    
):
    
    conf = get_conf_content(conf_file) if conf_file else None
    swot_root = Path(conf.get("SWOT_L2_AVISO_DIR", SWOT_ROOT))
    s1_root   = Path(conf.get("S1_WV_ROOT", S1_WV_ROOT))
    swot_files = find_swot_files(swot_root)
    output_dir = Path(conf.get("WV_MATCHUP_OUTPUT_DIR", OUTPUT_DIR))
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

    #csv_path = output_dir / "final_accumulated_matchups_summary_360_min.csv"
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

        # ----------------------------------------------------------
        # Save every N SWOT files (or after the last one)
        # ----------------------------------------------------------
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

                # Free memory
                all_matchups.clear()

    log.info("Total matchups found: %d", total_matchups)

    # ------------------------------------------------------------------
    # Save corrupted file list
    # ------------------------------------------------------------------

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Collocate S1-WV L2 and SWOT KaRIn L2_LR_SSH products.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "--swot-root", type=Path, default=SWOT_ROOT,
    #     help="Root directory containing SWOT KaRIn *.nc files (searched recursively)",
    # )
    # parser.add_argument(
    #     "--s1-root", type=Path, default=S1_WV_ROOT,
    #     help="Root of the S1-WV daily file tree  (<root>/<yyyy>/<doy>/S1*_WV_L2D_*.nc)",
    # )
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