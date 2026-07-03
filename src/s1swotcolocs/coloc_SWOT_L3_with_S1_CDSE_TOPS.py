"""
coloc_SWOT_L3_with_S1_CDSE_TOPS.py  — robustified version

Changes vs original
-------------------
slice_swot:
  - Both fix_polygon() calls now catch ValueError (the "linearring requires
    at least 4 coordinates" crash from antimeridian) in addition to the
    existing AssertionError catch. The new counter is
    "linearring_error_at_fix_polygon".
  - The guard block for degenerate / too-few-point geometries is tightened:
    is_nearly_collinear() is also called before alphashape so that a collinear
    set never reaches the triangulation.

treat_a_clean_piece_of_swot_orbit:
  - The stray breakpoint() that was left in production code after the NaT
    warning has been removed.

treat_one_day_wrapper:
  - The bare "raise ValueError" inside the do_cdse_query loop has been
    removed. It was crashing the entire day when a single GDF query failed.
    The error is now logged and counted, the loop continues.

No other logic has been changed.
"""
import pyogrio
import fiona  
import collections
import datetime
import glob
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

from shapely import wkt
import numpy as np
import pandas as pd
import xarray as xr
from antimeridian import fix_polygon
from antimeridian._implementation import FixWindingWarning
from shapely.geometry import MultiPoint, MultiPolygon
from tqdm import tqdm
import geopandas as gpd
from scipy import spatial
import alphashape
import geodatasets
import argparse

import cdsodatacli
import cdsodatacli.query
from eodms_rapi import EODMSRAPI
import s1swotcolocs
from s1swotcolocs.utils import (
    get_conf_content,
    get_netcdf_attribute,
    normalize_mission,
    parse_safe_name,
)
from s1swotcolocs.check_lonlat_polygon_extent import (
    check_longitude_smaller_than_latitude_extent,
)

warnings.filterwarnings("ignore", category=FixWindingWarning)
warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS",
    category=UserWarning,
    module="cdsodatacli",
)
warnings.filterwarnings("error", message="Singular matrix")

app_logger = logging.getLogger(__name__)


class _RaiseOnSingularMatrix(logging.Filter):
    def filter(self, record):
        if "Singular matrix" in record.getMessage():
            raise RuntimeError("Singular matrix: " + record.getMessage())
        return True


_singular_filter = _RaiseOnSingularMatrix()

LOGGERS_TO_SILENCE = ["cdsodatacli", "cdsodatacli.query"]
for logger_name in LOGGERS_TO_SILENCE:
    lib_logger = logging.getLogger(logger_name)
    lib_logger.handlers.clear()
    lib_logger.addHandler(logging.NullHandler())
    lib_logger.propagate = False
    lib_logger.setLevel(logging.CRITICAL + 1)


class CDSODATACLIQueryFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("cdsodatacli.query")


class SuppressCDSODATACLIQuery(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("cdsodatacli.query")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def is_degenerate_swath(points: np.ndarray, bbox_ratio_threshold: float = 0.5) -> bool:
    """
    Returns True if the point set forms a swath too narrow for alphashape.
    Uses the lon/lat bounding box aspect ratio.
    """
    if len(points) < 4:
        return True
    lon_range = points[:, 0].max() - points[:, 0].min()
    lat_range = points[:, 1].max() - points[:, 1].min()
    if lat_range == 0:
        return True
    return (lon_range / lat_range) < bbox_ratio_threshold


def is_nearly_collinear(points: np.ndarray, threshold: float = 0.05) -> bool:
    """
    Returns True if points are nearly collinear (causes singular matrix in alphashape).
    """
    if len(points) < 4:
        return True
    centered = points - points.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    if sv[0] == 0:
        return True
    return (sv[1] / sv[0]) < threshold


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------
COLLECTIONS = {
    'S1'  : 'SENTINEL-1',
    'RCM' : 'RCMImageProducts',
    'RS2' : 'Radarsat2RawProducts'
}

def _get_collection(m):
    return COLLECTIONS.get(m, None)


def treat_a_clean_piece_of_swot_orbit(
    swotpiece, points, swotsub, mission, mode, producttype, delta_t_max, cpt
):
    """
    :param swotpiece: shapely.geometry.Polygon simplified, not crossing antimeridian
    :param points: 2D matrix with lon and lat from SWOT
    :param swotsub: sub part of a SWOT xarray.Dataset 
    :parap mission: str "S1" or "RCM" or "RS2"
    :return: (GeoDataFrame, cpt)
    """
    app_logger.debug("swotpiece : %s", swotpiece)
    original_filename = os.path.basename(swotsub.encoding["source"])
    lonmin = np.amin(swotpiece.exterior.xy[0])
    lonmax = np.amax(swotpiece.exterior.xy[0])
    latmin = np.amin(swotpiece.exterior.xy[1])
    latmax = np.amax(swotpiece.exterior.xy[1])

    tree = spatial.KDTree(points)
    app_logger.debug("coords North point: %s %s", lonmax, latmax)
    dd, idx_north = tree.query([lonmax, latmax], k=1)
    dd, idx_south = tree.query([lonmin, latmin], k=1)
    app_logger.debug("idx_north : %s", idx_north)
    app_logger.debug("idx_south : %s", idx_south)
    num_line_idx_north, _ = np.unravel_index(idx_north, swotsub["longitude"].shape)
    num_line_idx_south, _ = np.unravel_index(idx_south, swotsub["longitude"].shape)
    app_logger.debug("num_line_idx_north : %s", num_line_idx_north)
    app_logger.debug("num_line_idx_south : %s", num_line_idx_south)
    time_north = swotsub["time"].isel(num_lines=num_line_idx_north).values
    time_south = swotsub["time"].isel(num_lines=num_line_idx_south).values
    app_logger.debug("time_north %s", time_north)
    app_logger.debug("time_south %s", time_south)

    if pd.isnull(time_north) or pd.isnull(time_south):
        app_logger.warning(
            "NaT time values in SWOT file %s at lines %s/%s — skipping piece",
            original_filename,
            num_line_idx_north,
            num_line_idx_south,
        )
        # ── FIX: removed stray breakpoint() that was left in production code ──
        cpt["NaT_time_skipped"] += 1
        return gpd.GeoDataFrame(), cpt
    else:
        cpt["Ok_SWOT_time_values"] += 1

    if time_north > time_south:
        startswot = time_south
        stopswot = time_north
    else:
        startswot = time_north
        stopswot = time_south

    sta = pd.to_datetime((startswot - delta_t_max)).round("us").to_pydatetime()
    sto = pd.to_datetime((stopswot + delta_t_max)).round("us").to_pydatetime()
    app_logger.debug("sta : %s sto : %s", sta, sto)

    gdf = gpd.GeoDataFrame(
        {
            "start_datetime": [sta],
            "end_datetime": [sto],
            "geometry": [swotpiece],
            "collection": [_get_collection(mission)],
            "name": [None],
            "sensormode": [mode],
            "producttype": [producttype],
            "Attributes": [None],
            "id_query": ["SWOT %s %s %s" % (original_filename, startswot, stopswot)],
        }
    )
    return gdf, cpt


def compute_alphashape_safe(gdfswot, points, alpha, cpt):
    """
    Compute alphashape with a convex_hull fallback on degenerate point sets.
    """
    logging.getLogger().addFilter(_singular_filter)
    try:
        result = alphashape.alphashape(gdfswot, alpha=alpha)
    except (RuntimeError, Exception) as e:
        app_logger.debug("alphashape failed (%s) — falling back to convex_hull.", e)
        cpt["alphashape_fallback_to_convex_hull"] += 1
        result = MultiPoint(points).convex_hull
    finally:
        logging.getLogger().removeFilter(_singular_filter)
    return result, cpt


def _safe_fix_polygon(
    polygon, cpt: collections.defaultdict, context: str = "", **kwargs
):
    """
    Wrapper around antimeridian.fix_polygon that catches both AssertionError
    and ValueError.

    The ValueError "A linearring requires at least 4 coordinates" is raised
    when antimeridian splits a polygon at the antimeridian and the resulting
    sub-polygon has too few vertices to form a valid LinearRing. This happens
    on degenerate SWOT swath geometries near the poles.

    Returns (result, ok) where ok is False when the call failed.
    """
    try:
        return fix_polygon(polygon, **kwargs), True

    except AssertionError:
        cpt["impossible_to_fix_polygon"] += 1
        app_logger.debug(
            "AssertionError in fix_polygon%s — skipping piece.",
            f" ({context})" if context else "",
        )
        return None, False

    except ValueError as exc:
        exc_str = str(exc).lower()
        if "linearring" in exc_str or "at least 4 coordinates" in exc_str:
            cpt["linearring_error_at_fix_polygon"] += 1
            app_logger.warning(
                "Linearring error in fix_polygon%s — skipping piece. "
                "This is a known antimeridian/shapely issue on degenerate "
                "SWOT polygons near poles. Detail: %s",
                f" ({context})" if context else "",
                exc,
            )
        else:
            cpt["valueerror_at_fix_polygon"] += 1
            app_logger.warning(
                "ValueError in fix_polygon%s — skipping piece. Detail: %s",
                f" ({context})" if context else "",
                exc,
            )
        return None, False


def slice_swot(
    onedsswot,
    idxstart,
    idxstop,
    cpt,
    max_area_size,
    delta_hours=6,
    mission="S1",
    mode="IW",
    producttype="SLC",
    tolerance_simplification=0.1,
):
    """
    Treat the SWOT swath by pieces to avoid oversized polygons.

    Key robustness changes vs original:
      - Both fix_polygon() calls are now routed through _safe_fix_polygon()
        which catches ValueError (linearring) in addition to AssertionError.
      - is_nearly_collinear() guard added alongside is_degenerate_swath().
    """
    sub_gdf = []
    delta_t_max = np.timedelta64(delta_hours, "h")
    steps_in_swot = 20
    swotsub = onedsswot.isel({"num_lines": slice(idxstart, idxstop, steps_in_swot)})
    lonswot = swotsub["longitude"].values.ravel()
    lonswot[lonswot > 180] += -360.0
    points = np.column_stack((lonswot, swotsub["latitude"].values.ravel()))

    if len(points) < 4:
        app_logger.debug(
            "Skipping segment: too few points after subsampling (%i)", len(points)
        )
        cpt["too_few_points_skipped"] += 1
        return sub_gdf, cpt

    multi_point = MultiPoint(points)
    gdfswot = gpd.GeoDataFrame(geometry=list(multi_point.geoms))

    # ── FIX: also check collinearity before alphashape, not just aspect ratio ──
    if is_degenerate_swath(points, bbox_ratio_threshold=0.5) or is_nearly_collinear(
        points
    ):
        app_logger.debug(
            "Degenerate swath shape detected (narrow or collinear) "
            "— falling back to convex_hull."
        )
        cpt["degenerate_collinear_fallback"] += 1
        alpha_shape_swot = MultiPoint(points).convex_hull
    else:
        alpha_shape_swot, cpt = compute_alphashape_safe(
            gdfswot, points, tolerance_simplification, cpt
        )

    land_path = geodatasets.get_path("naturalearth.land")
    land = gpd.read_file(land_path)
    land_union = land.union_all()

    ocean_part = alpha_shape_swot.difference(land_union)
    tolerance = tolerance_simplification
    simplified_polygon = ocean_part.simplify(tolerance)
    simplified_polygon = simplified_polygon.buffer(0)

    if simplified_polygon.is_empty:
        app_logger.debug("one empty polygon")
        cpt["empty_polygon"] += 1
        return sub_gdf, cpt

    if isinstance(simplified_polygon, MultiPolygon):
        app_logger.debug("Nb parts: %i", len(simplified_polygon.geoms))
        for iip, partswot in enumerate(simplified_polygon.geoms):

            # ── FIX: _safe_fix_polygon catches ValueError + AssertionError ──
            subpartswot, ok = _safe_fix_polygon(
                partswot, cpt, context=f"MultiPolygon part {iip}"
            )
            if not ok:
                continue

            if isinstance(subpartswot, MultiPolygon):
                cpt["segment_interupted_by_land_and_antimeridian"] += 1
                for yyp, subsubpartswot in enumerate(subpartswot.geoms):
                    is_ok_extents = check_longitude_smaller_than_latitude_extent(
                        subsubpartswot
                    )
                    if subsubpartswot.area < max_area_size and is_ok_extents:
                        gdf, cpt = treat_a_clean_piece_of_swot_orbit(
                            subsubpartswot,
                            points,
                            swotsub,
                            mission,
                            mode,
                            producttype,
                            delta_t_max,
                            cpt=cpt,
                        )
                        sub_gdf.append(gdf)
                    else:
                        cpt["segment_with_area_too_large"] += 1
            else:
                cpt["segment_interupted_by_land_only"] += 1
                if subpartswot.area < max_area_size:
                    gdf, cpt = treat_a_clean_piece_of_swot_orbit(
                        subpartswot,
                        points,
                        swotsub,
                        mission,
                        mode,
                        producttype,
                        delta_t_max,
                        cpt=cpt,
                    )
                    sub_gdf.append(gdf)
                else:
                    cpt["segment_with_area_too_large"] += 1

    else:
        # Single contiguous polygon over ocean
        # ── FIX: _safe_fix_polygon catches ValueError + AssertionError ──
        subpartswot, ok = _safe_fix_polygon(
            simplified_polygon,
            cpt,
            context="single polygon",
            fix_winding=True,
        )
        if not ok:
            return sub_gdf, cpt

        if isinstance(subpartswot, MultiPolygon):
            cpt["segment_continuous_with_antimeridian"] += 1
            for yyp, subsubpartswot in enumerate(subpartswot.geoms):
                if subsubpartswot.area < max_area_size:
                    gdf, cpt = treat_a_clean_piece_of_swot_orbit(
                        subsubpartswot,
                        points,
                        swotsub,
                        mission,
                        mode,
                        producttype,
                        delta_t_max,
                        cpt=cpt,
                    )
                    sub_gdf.append(gdf)
                else:
                    cpt["segment_with_area_too_large"] += 1
        else:
            cpt["segment_continuous_without_antimeridian"] += 1
            if subpartswot.area < max_area_size:
                gdf, cpt = treat_a_clean_piece_of_swot_orbit(
                    subpartswot,
                    points,
                    swotsub,
                    mission,
                    mode,
                    producttype,
                    delta_t_max,
                    cpt=cpt,
                )
                sub_gdf.append(gdf)
            else:
                cpt["segment_with_area_too_large"] += 1

    return sub_gdf, cpt


def get_swot_geoloc(
    one_swot_file,
    max_area_size,
    delta_hours=6,
    mission="S1",
    mode="IW",
    producttype="SLC",
    cpt=None,
    tolerance_simplification=0.1,
) -> tuple[list, collections.defaultdict]:
    """
    :param one_swot_file: str
    :param max_area_size: float
    :param delta_hours: int
    :param mode: str IW or EW
    :param producttype: str SLC or GRD
    :param cpt: collections.defaultdict(int)
    :param tolerance_simplification: float
    :return: (allgdfs_swot, cpt)
    """
    allgdfs_swot = []
    app_logger.debug("%s", one_swot_file)
    onedsswot = xr.open_dataset(one_swot_file)

    time_vals = onedsswot["time"].values
    valid_mask = ~np.isnat(time_vals)
    if not valid_mask.all():
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            app_logger.warning(
                "SWOT file %s has no valid time values — skipping", one_swot_file
            )
            return [], cpt
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        n_trimmed = (~valid_mask).sum()
        app_logger.info(
            "Trimming %i NaT lines from SWOT file (lines %i to %i kept out of %i)",
            n_trimmed,
            first_valid,
            last_valid,
            len(time_vals),
        )
        onedsswot = onedsswot.isel(num_lines=slice(first_valid, last_valid + 1))

    app_logger.debug("full size time %s", onedsswot["time"].sizes)
    segment = 1000
    if cpt is None:
        cpt = collections.defaultdict(int)

    for oo in np.arange(0, onedsswot["time"].sizes["num_lines"], segment):
        tmplistgdf, cpt = slice_swot(
            onedsswot,
            idxstart=oo,
            idxstop=oo + segment,
            cpt=cpt,
            max_area_size=max_area_size,
            delta_hours=delta_hours,
            mission=mission,
            mode=mode,
            producttype=producttype,
            tolerance_simplification=tolerance_simplification,
        )
        allgdfs_swot += tmplistgdf

    return allgdfs_swot, cpt


def do_cdse_query(gdf, mini_ocean=10, cache_dir=None):
    if pd.isnull(gdf["start_datetime"].iloc[0]) or pd.isnull(
        gdf["end_datetime"].iloc[0]
    ):
        app_logger.warning(
            "Skipping CDSE query for %s: NaT start or end datetime",
            gdf["id_query"].iloc[0],
        )
        return None

    collected_data_norm = cdsodatacli.query.fetch_data(
        gdf,
        min_sea_percent=mini_ocean,
        timedelta_slice=datetime.timedelta(days=4),
        cache_dir=cache_dir,
    )
    return collected_data_norm


def do_eodms_query(gdf, eodmsrapi, feat_op='overlaps'):
    if pd.isnull(gdf["start_datetime"].iloc[0]) or pd.isnull(
        gdf["end_datetime"].iloc[0]
    ):
        app_logger.warning(
            "Skipping EODMS query for %s: NaT start or end datetime",
            gdf["id_query"].iloc[0],
        )
        return None
    
    eodmsrapi.search(
        gdf['collection'].iloc[0],
        dates=[{
            "start": gdf["start_datetime"].iloc[0],
            "end": gdf["end_datetime"].iloc[0]
            }],
        features=[(feat_op, gdf['geometry'].iloc[0])],
        max_results=100,
    )

    df = pd.DataFrame(eodmsrapi.get_results("full"))
    if len(df)==0:
        return None
    
    df["id_original_query"] = gdf["id_query"].iloc[0]
    return df


def save_netcdf_file_per_swot_piece_orbit_core_s1(
    query_output, swot_gdf, fpath_out, delta_t_max, cpt
):
    """
    Save the result for one SWOT query matching one or more S1 product(s).
    """
    SWOT_start_piece = pd.to_datetime(swot_gdf["id_query"][0].split(" ")[2])
    SWOT_start_piece = SWOT_start_piece.tz_localize("UTC")
    swot_polygon = "%s" % swot_gdf["geometry"][0]

    all_SAR_polygones = []
    all_SWOT_fpath = []
    all_start_SAR = []
    all_delta_times = []

    start_time_strings = query_output["ContentDate"].str["Start"]
    query_output["Start_dt"] = pd.to_datetime(start_time_strings, utc=True)

    for sasa in range(len(query_output["geometry"])):
        all_SAR_polygones.append("%s" % query_output["geometry"].iloc[sasa])
        SAR_start_slice = query_output["Start_dt"].iloc[sasa]
        delta_diff_time = SWOT_start_piece - SAR_start_slice
        delta_diff_time_minutes = delta_diff_time / np.timedelta64(1, "m")
        all_start_SAR.append(SAR_start_slice.tz_localize(None))
        all_delta_times.append(delta_diff_time_minutes)
        all_SWOT_fpath.append(query_output["id_original_query"].iloc[sasa].split(" ")[1])

    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    SWOT_start_piece = np.array(SWOT_start_piece.tz_localize(None)).astype(
        "datetime64[s]"
    )
    sar_names = np.array(query_output["Name"].values, dtype=object)
    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    all_delta_times = np.array(all_delta_times, dtype=np.float64)
    SWOT_start_piece = np.array(SWOT_start_piece).astype("datetime64[s]")

    colocds = xr.Dataset()
    colocds["sar_safe_name"] = xr.DataArray(
        query_output["Name"].values,
        dims="sar_start_time_slice",
        coords={"sar_start_time_slice": all_start_SAR},
        attrs={"description": "name of the SAFE Sentinel-1 products colocated"},
    )
    colocds["filepath_swot"] = xr.DataArray(
        all_SWOT_fpath,
        dims="sar_start_time_slice",
        attrs={"description": "file paths of SWOT products colocated"},
    )
    colocds["delta_diff_time"] = xr.DataArray(
        all_delta_times,
        dims="sar_start_time_slice",
        attrs={"description": "delta time SWOT - SAR in minutes"},
    )
    colocds["SWOT_start_time_slice"] = xr.DataArray(
        SWOT_start_piece, attrs={"description": "SWOT slice start date"}
    )
    colocds["sar_safe_name"] = xr.DataArray(
        sar_names,
        dims="sar_start_time_slice",
        attrs={"description": "name of the SAFE Sentinel-1 products colocated"},
    )
    colocds["swot_polygon"] = xr.DataArray(
        swot_polygon, attrs={"description": "polygon of SWOT piece of orbit"}
    )
    colocds["sar_polygon"] = xr.DataArray(
        all_SAR_polygones,
        dims="sar_start_time_slice",
        attrs={"description": "polygons of SAR products"},
    )
    colocds.attrs["s1swotcolocs_python_lib_version"] = s1swotcolocs.__version__
    colocds.attrs["searching_windows_width_in_hours"] = delta_t_max

    if os.path.exists(fpath_out):
        logging.info("remove the existing file")
        os.remove(fpath_out)
        cpt["file_replaced"] += 1
    else:
        logging.debug("file does not exist -> brand-new file on disk")
        cpt["new_file"] += 1

    if not os.path.exists(os.path.dirname(fpath_out)):
        os.makedirs(os.path.dirname(fpath_out), mode=0o775)

    colocds.to_netcdf(fpath_out, engine="h5netcdf")
    os.chmod(fpath_out, 0o664)
    app_logger.info("coloc file created : %s", fpath_out)
    return cpt


def save_netcdf_file_per_swot_piece_orbit_core_rsat(
    query_output, swot_gdf, fpath_out, delta_t_max, cpt
):
    """
    Save the result for one SWOT query matching one or more RADARSAT product(s).
    """
    SWOT_start_piece = pd.to_datetime(swot_gdf["id_query"][0].split(" ")[2])
    SWOT_start_piece = SWOT_start_piece.tz_localize("UTC")
    swot_polygon = "%s" % swot_gdf["geometry"][0]

    all_SAR_polygones = []
    all_SWOT_fpath = []
    all_start_SAR = []
    all_delta_times = []

    query_output["Start_dt"] = pd.to_datetime(query_output["acquisitionStartDate"], utc=True)

    for sasa in range(len(query_output["geometry"])):
        all_SAR_polygones.append("%s" % query_output["wktGeometry"].iloc[sasa])
        SAR_start_slice = query_output["Start_dt"].iloc[sasa]
        delta_diff_time = SWOT_start_piece - SAR_start_slice
        delta_diff_time_minutes = delta_diff_time / np.timedelta64(1, "m")
        all_start_SAR.append(SAR_start_slice.tz_localize(None))
        all_delta_times.append(delta_diff_time_minutes)
        all_SWOT_fpath.append(query_output["id_original_query"].iloc[sasa].split(" ")[1])

    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    SWOT_start_piece = np.array(SWOT_start_piece.tz_localize(None)).astype(
        "datetime64[s]"
    )
    sar_names = np.array(query_output["title"].values, dtype=object)
    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    all_delta_times = np.array(all_delta_times, dtype=np.float64)
    SWOT_start_piece = np.array(SWOT_start_piece).astype("datetime64[s]")

    colocds = xr.Dataset()
    colocds["sar_safe_name"] = xr.DataArray(
        sar_names,
        dims="sar_start_time_slice",
        coords={"sar_start_time_slice": all_start_SAR},
        attrs={"description": "name of the RADARSAT products colocated"},
    )
    colocds["filepath_swot"] = xr.DataArray(
        all_SWOT_fpath,
        dims="sar_start_time_slice",
        attrs={"description": "file paths of SWOT products colocated"},
    )
    colocds["delta_diff_time"] = xr.DataArray(
        all_delta_times,
        dims="sar_start_time_slice",
        attrs={"description": "delta time SWOT - SAR in minutes"},
    )
    colocds["SWOT_start_time_slice"] = xr.DataArray(
        SWOT_start_piece, attrs={"description": "SWOT slice start date"}
    )
    colocds["swot_polygon"] = xr.DataArray(
        swot_polygon, attrs={"description": "polygon of SWOT piece of orbit"}
    )
    colocds["sar_polygon"] = xr.DataArray(
        all_SAR_polygones,
        dims="sar_start_time_slice",
        attrs={"description": "polygons of SAR products"},
    )
    # Extra metadata specific to this API
    colocds["sar_product_id"] = xr.DataArray(
        query_output["productId"].values,
        dims="sar_start_time_slice",
        attrs={"description": "product ID from RADARSAT API"},
    )
    colocds["sar_satellite_id"] = xr.DataArray(
        query_output["satelliteId"].values,
        dims="sar_start_time_slice",
        attrs={"description": "satellite ID (e.g. RCM-1, RCM-2, RCM-3)"},
    )
    colocds["sar_beam_mnemonic"] = xr.DataArray(
        query_output["beamMnemonic"].values,
        dims="sar_start_time_slice",
        attrs={"description": "beam mnemonic of the SAR acquisition"},
    )
    colocds["sar_polarization"] = xr.DataArray(
        query_output["polarization"].values,
        dims="sar_start_time_slice",
        attrs={"description": "polarization mode of the SAR acquisition"},
    )

    colocds.attrs["swotcolocs_python_lib_version"] = s1swotcolocs.__version__
    colocds.attrs["searching_windows_width_in_hours"] = delta_t_max

    if os.path.exists(fpath_out):
        logging.info("remove the existing file")
        os.remove(fpath_out)
        cpt["file_replaced"] += 1
    else:
        logging.debug("file does not exist -> brand-new file on disk")
        cpt["new_file"] += 1

    if not os.path.exists(os.path.dirname(fpath_out)):
        os.makedirs(os.path.dirname(fpath_out), mode=0o775)

    colocds.to_netcdf(fpath_out, engine="h5netcdf")
    os.chmod(fpath_out, 0o664)
    app_logger.info("coloc file created : %s", fpath_out)
    return cpt


def get_swot_date_info(SWOT_start_piece):
    """
    Arguments:
        SWOT_start_piece (np.datetime64):
    Returns:
        swot_formated_date (str): e.g. 20251017T151210
        year (int), month (str MM), day (str DD)
    """
    dt_py = SWOT_start_piece.astype("M8[D]").astype(object)
    year = dt_py.year
    month = f"{dt_py.month:02d}"
    day = f"{dt_py.day:02d}"
    swot_formated_date = (
        ("%s" % SWOT_start_piece).replace("-", "").replace(":", "").split(".")[0]
    )
    return swot_formated_date, year, month, day


def save_meta_coloc_output(
    query_outputs, SWOTgdfs, dir_output, mission, mode, delta_t_max, cpt, disable_tqdm=False
):
    save_func = {
        "S1":  save_netcdf_file_per_swot_piece_orbit_core_s1,
        "RCM": save_netcdf_file_per_swot_piece_orbit_core_rsat,
        "RS2": save_netcdf_file_per_swot_piece_orbit_core_rsat,
    }.get(mission)

    if save_func is None:
        raise ValueError(f"Mission '{mission}' not supported. Expected one of: S1, RCM, RS2")

    if mission in ["RCM", "RS2"]:
        mode = "all"
    assert len(query_outputs) == len(SWOTgdfs)
    for xxi in tqdm(range(len(query_outputs)), disable=disable_tqdm):
        one_output = query_outputs[xxi]
        swot_gdf = SWOTgdfs[xxi]
        if one_output is not None:
            SWOT_start_piece = np.datetime64(swot_gdf["id_query"][0].split(" ")[2])
            swot_formated_date, year, month, day = get_swot_date_info(SWOT_start_piece)
            fpath_out = os.path.join(
                dir_output,
                "%s" % year,
                "%s" % month,
                "%s" % day,
                f"coloc_SWOT_L3_{mission}_{mode}_{swot_formated_date}.nc",
            )
            app_logger.info("fpath_out: %s", fpath_out)
            cpt = save_func(
                query_output=one_output,
                swot_gdf=swot_gdf,
                fpath_out=fpath_out,
                delta_t_max=delta_t_max,
                cpt=cpt,
            )
            cpt["written"] += 1
        else:
            cpt["no_coloc"] += 1

    app_logger.info(
        "number of coloc files written : %i/%i", cpt["written"], len(query_outputs)
    )
    app_logger.info(
        "number of SWOT piece of orbit without S1 coloc : %i/%i",
        cpt["no_coloc"],
        len(query_outputs),
    )
    return cpt


def parse_args():
    parser = argparse.ArgumentParser(description="S1-SWOT meta coloc")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--day2treat", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--mission",
        required=False,
        choices=["S1", "RCM", "RS2"],
        default="S1",
        help="S1, RCM or RS2 [default=S1]",
    )
    parser.add_argument(
        "--mode",
        required=False,
        choices=["IW", "EW"],
        default="IW",
        help="IW or EW [default=IW]",
    )
    parser.add_argument(
        "--producttype",
        required=False,
        choices=["SLC", "GRD"],
        default="SLC",
        help="SLC or GRD [default=SLC]",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="directory where to store output netCDF files",
    )
    parser.add_argument("--conf", required=True, help="config file to use")
    return parser.parse_args()

def _get_swot_intersecting_sar(swot_gdf, sar_footprint):
    """
    Return SWOT polygons that intersect the SAR footprint.

    Parameters
    ----------
    swot_gdf : gpd.GeoDataFrame
        GeoDataFrame of SWOT polygons.
    sar_footprint : str or shapely.geometry
        SAR footprint as WKT string or shapely geometry.

    Returns
    -------
    gpd.GeoDataFrame
        Subset of SWOT polygons intersecting the SAR footprint, or None if no intersection.
    """
    if isinstance(sar_footprint, str):
        from shapely import wkt
        sar_footprint = wkt.loads(sar_footprint)

    sar_gdf = gpd.GeoDataFrame(geometry=[sar_footprint], crs="EPSG:4326")

    intersecting = gpd.sjoin(
        swot_gdf,
        sar_gdf,
        how="inner",
        predicate="intersects",
    )

    if len(intersecting) == 0:
        return None
    
    return intersecting


def save_matchup(
        swot_gdf_intersect_sar,
        swot_dir,
        sar_safe, 
        sar_fp, 
        sar_start,
        delta_t_hours, 
        sar_end=None, 
        out_dir=None
    ):
    """
    Save SWOT/SAR matchup results to a CSV file.

    Parameters
    ----------
    swot_gdf_intersect_sar : gpd.GeoDataFrame
        SWOT polygons intersecting the SAR footprint.
    sar_safe : str
        SAR SAFE filename (used as output filename).
    sar_fp : shapely.geometry
        SAR footprint geometry.
    sar_start : datetime
        SAR acquisition start time.
    sar_end : datetime, optional
        SAR acquisition end time.
    out_dir : str or Path, optional
        Output directory. If None, saves in the current working directory.
    """
    if out_dir is None:
        out_dir = Path.cwd()
        app_logger.warning("No output directory provided, saving in current working directory: %s", out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            app_logger.info("Output directory does not exist, creating it: %s", out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        elif not out_dir.is_dir():
            raise ValueError(f"out_dir exists but is not a directory: {out_dir}")

    rows = []
    for _, row in swot_gdf_intersect_sar.iterrows():
        swot_filename = Path(swot_dir) / row["id_query"].split(" ")[1]
        rows.append({
            "swot_filename": swot_filename,
            "swot_start":    row["start_datetime"],
            "swot_end":      row["end_datetime"],
            "swot_geometry": row["geometry"].wkt,
            "sar_safe":      sar_safe,
            "sar_start":     sar_start,
            "sar_end":       sar_end,
            "sar_geometry":  sar_fp.wkt,
            "delta_t_hours": delta_t_hours,
        })

    df_out = pd.DataFrame(rows)

    safe_name = Path(sar_safe).stem
    fpath_out = out_dir / f"matchup_SAR-SWOT_{safe_name}_dt{delta_t_hours}h.csv"

    df_out.to_csv(fpath_out, index=False)
    app_logger.info("Matchup saved : %s", fpath_out)

    return fpath_out


def treat_one_safe(safe, confpath, disable_tqdm=False, dev=False):

    # --- Load config ---
    conf = get_conf_content(confpath)
    SWOTDIR = Path(conf["SWOT_L2_AVISO_DIR"])
    DT = conf["DELTA_HOURS"]

    # --- Parse SAR SAFE metadata ---
    safe_info = parse_safe_name(safe)
    sar_start = safe_info['startdate'] + safe_info['starttime']
    sar_start = datetime.datetime.strptime(sar_start, "%Y%m%d%H%M%S")
    mission = normalize_mission(safe_info['mission_id'])
    mode = safe_info['mode']
    producttype = safe_info['type'] if mission == 'S1' else 'GRD'

    app_logger.info("Processing SAFE : %s | start=%s", Path(safe).stem, sar_start)

    # --- Define time window around SAR acquisition ---
    time_delta = datetime.timedelta(hours=DT)
    t1 = sar_start - time_delta
    t2 = sar_start + time_delta
    app_logger.info("Time window : %s  -->  %s  (+/- %sh)", t1, t2, DT)

    # --- Find SWOT files matching the acquisition date ---
    pattern = f"SWOT_L2_LR_SSH_WindWave_*{sar_start.strftime('_%Y%m%dT')}*.nc"
    ncfiles = list(SWOTDIR.glob(pattern))
    app_logger.info("SWOT files found before time filtering : %i", len(ncfiles))

    # --- Filter SWOT files whose time range overlaps the SAR time window ---
    lstswotfiles = []
    for n in ncfiles:
        parts = n.stem.split('_')
        start = datetime.datetime.strptime(parts[7], "%Y%m%dT%H%M%S")
        end   = datetime.datetime.strptime(parts[8], "%Y%m%dT%H%M%S")
        if (start < t2) and (end > t1):
            lstswotfiles.append(n)

    app_logger.info("SWOT files retained after time filtering : %i", len(lstswotfiles))

    if len(lstswotfiles) == 0:
        app_logger.info("No SWOT files in time window, skipping SAFE : %s", os.path.basename(safe))
        return

    if dev:
        app_logger.info("Development mode: restricting to first 2 SWOT files")
        lstswotfiles = lstswotfiles[:2]

    # --- Build SWOT GeoDataFrames from each SWOT file ---
    app_logger.info("Building SWOT GeoDataFrames (+/- %sh)", DT)
    SWOTgdfs = []
    cpt = collections.defaultdict(int)
    cpt["nbSWOTfiles"] = len(lstswotfiles)

    for oneswotfile in tqdm(lstswotfiles, disable=disable_tqdm, desc="Processing SWOT files", unit="file"):
        swot_distinct_gdfs_list, cpt = get_swot_geoloc(
            oneswotfile,
            delta_hours=0,  # time filtering is applied on the GDF afterwards
            max_area_size=conf["MAX_AREA_SIZE"],
            mission=mission,
            mode=mode,
            producttype=producttype,
            cpt=cpt,
            tolerance_simplification=conf["TOLERANCE_SIMPLIFICATION"],
        )
        SWOTgdfs += swot_distinct_gdfs_list

    # --- Concatenate all GDFs and enforce WGS84 CRS ---
    tmp = [gdf.set_crs("EPSG:4326", allow_override=True) for gdf in SWOTgdfs]
    swot_gdf = gpd.GeoDataFrame(pd.concat(tmp, ignore_index=True), crs="EPSG:4326")
    app_logger.info("Total SWOT polygons before time filtering : %i", len(swot_gdf))

    # --- Filter SWOT polygons to the SAR time window ---
    swot_gdf_filtered = swot_gdf[
        (swot_gdf["start_datetime"] <= t2) & (swot_gdf["end_datetime"] >= t1)
    ]
    app_logger.info("SWOT polygons after time filtering : %i", len(swot_gdf_filtered))

    if len(swot_gdf_filtered) == 0:
        app_logger.info("No SWOT polygons in time window, skipping SAFE : %s", os.path.basename(safe))
        return

    # --- Extract SAR footprint ---
    sar_ncfile = list(Path(safe).rglob('*.nc'))[0]
    sar_fp = wkt.loads(get_netcdf_attribute(sar_ncfile, "main_footprint"))
    app_logger.info("SAR footprint loaded from : %s", sar_ncfile.name)

    # --- Fix antimeridian crossing if needed ---
    sar_fp_fixed, ok = _safe_fix_polygon(sar_fp, cpt, context=os.path.basename(safe))
    if not ok:
        app_logger.warning("Could not fix SAR footprint antimeridian crossing, using original polygon.")
        sar_fp_fixed = sar_fp

    # --- Handle MultiPolygon case (SAR footprint split at antimeridian) ---
    if sar_fp_fixed.geom_type == "MultiPolygon":
        app_logger.info("SAR footprint is a MultiPolygon after antimeridian fix — processing each sub-polygon separately.")
        sar_subpolygons = list(sar_fp_fixed.geoms)
    else:
        sar_subpolygons = [sar_fp_fixed]

    # --- Find SWOT polygons spatially intersecting the SAR footprint ---
    swot_gdf_intersect_list = []
    for sub_fp in sar_subpolygons:
        result = _get_swot_intersecting_sar(swot_gdf_filtered, sub_fp)
        if result is not None:
            swot_gdf_intersect_list.append(result)

    if len(swot_gdf_intersect_list) == 0:
        app_logger.info("No SWOT polygons intersect the SAR footprint, skipping SAFE : %s", os.path.basename(safe))
        return

    swot_gdf_intersect_sar = gpd.GeoDataFrame(
        pd.concat(swot_gdf_intersect_list, ignore_index=True), crs="EPSG:4326"
    ).drop_duplicates()  # évite les doublons si un polygone SWOT intersecte plusieurs sous-polygones

    app_logger.info("SWOT polygons intersecting SAR footprint : %i", len(swot_gdf_intersect_sar))

    # --- Save matchups to CSV ---
    save_matchup(
        swot_gdf_intersect_sar=swot_gdf_intersect_sar,
        swot_dir=SWOTDIR,
        sar_safe=safe,
        sar_fp=sar_fp,
        sar_start=sar_start,
        delta_t_hours=conf["DELTA_HOURS"],
        sar_end=None,
        out_dir=conf["HOST_META_COLOC_OUTPUT_DIR"],
    )

    return
    

def treat_one_day_wrapper(
    day2treat, outputdir, mission, mode, producttype, confpath, disable_tqdm=False, dev=False
):
    """
    :param day2treat: str YYYYMMDD
    :param outputdir: str
    :parap mission: str "S1" or "RCM" or "RS2"
    :param mode: str "IW" or "EW"
    :param producttype: str "SLC" or "GRD"
    :param confpath: str full path of the config.yml
    :param disable_tqdm: bool
    :param dev: bool, True -> use a smaller subset of SWOT files for faster dev iterations

    :return: cpt (collections.defaultdict)
    """
    t0 = time.time()
    conf = get_conf_content(confpath)
    dswot = conf["SWOT_L2_AVISO_DIR"]
    CACHE_CDSE = conf["CACHE_CDSE"]

    dd = datetime.datetime.strptime(day2treat, "%Y%m%d")
    app_logger.info("treat day : %s", dd)
    pattern = os.path.join(
        dswot,
        f"SWOT_L2_LR_SSH_WindWave_*{dd.strftime('_%Y%m%dT')}*.nc",
    )
    app_logger.info("pattern : %s", pattern)
    print("pattern SWOT", pattern)
    lstswotfiles = glob.glob(pattern)
    app_logger.info("Nb files SWOT found : %i", len(lstswotfiles))
    app_logger.info(
        "first step: creation of SWOT geodataframes with +/-%i hours shift vs S-1",
        conf["DELTA_HOURS"],
    )

    SWOTgdfs = []
    cpt = collections.defaultdict(int)
    cpt["nbSWOTfiles"] = len(lstswotfiles)

    if dev:
        app_logger.info(
            "Development mode: using only a subset of SWOT files for faster iterations."
        )
        lstswotfiles = lstswotfiles[
            :2
        ]  # Use only the first 2 SWOT files for development iterations

    for ii in tqdm(range(len(lstswotfiles)), disable=disable_tqdm):
        oneswotfile = lstswotfiles[ii]
        swot_distinct_gdfs_list, cpt = get_swot_geoloc(
            oneswotfile,
            delta_hours=conf["DELTA_HOURS"],
            max_area_size=conf["MAX_AREA_SIZE"],
            mission=mission,
            mode=mode,
            producttype=producttype,
            cpt=cpt,
            tolerance_simplification=conf["TOLERANCE_SIMPLIFICATION"],
        )
        SWOTgdfs += swot_distinct_gdfs_list

    query_outputs = []

    if mission=='S1':
        app_logger.info("GeoDataFrames prepared for CDSE queries: %s", cpt)
        app_logger.info("nb GeoDataFrames: %i", len(SWOTgdfs))  
        for ii in tqdm(range(len(SWOTgdfs)), disable=disable_tqdm):
            gdf = SWOTgdfs[ii]
            try:
                res = do_cdse_query(gdf, mini_ocean=10, cache_dir=CACHE_CDSE)
                if res is not None:
                    cpt["sentinel1_product_matching"] += len(res)
            except ValueError:
                # ── FIX: removed bare "raise ValueError" that crashed the whole day ──
                # Log the problematic GDF and continue with the next one.
                app_logger.error("problematic gdf: %s", gdf)
                app_logger.error("traceback: %s", traceback.format_exc())
                res = None
                cpt["problematic_gdf"] += 1

            query_outputs.append(res)

        app_logger.info("CDSE queries performed.")
    
    elif mission in ['RCM', 'RS2']:
        app_logger.info("GeoDataFrames prepared for EODMS queries: %s", cpt)
        app_logger.info("nb GeoDataFrames: %i", len(SWOTgdfs))
        eodmsrapi = EODMSRAPI(
            conf["EODMS"]["user_name"],
            conf["EODMS"]["pswd"],
        )
        for ii in tqdm(range(len(SWOTgdfs)), disable=disable_tqdm):
            gdf = SWOTgdfs[ii]
            try:
                res = do_eodms_query(gdf.to_wkt(), eodmsrapi, mission)
                if res is not None:
                    cpt[f"{mission}_product_matching"] += len(res)
            except ValueError:
                # ── FIX: removed bare "raise ValueError" that crashed the whole day ──
                # Log the problematic GDF and continue with the next one.
                app_logger.error("problematic gdf: %s", gdf)
                app_logger.error("traceback: %s", traceback.format_exc())
                res = None
                cpt["problematic_gdf"] += 1

            query_outputs.append(res)

        app_logger.info("EODMS queries performed.")


    if len(SWOTgdfs) > 0:
        cpt = save_meta_coloc_output(
            query_outputs,
            SWOTgdfs,
            dir_output=outputdir,
            mission=mission,
            mode=mode,
            delta_t_max=conf["DELTA_HOURS"],
            cpt=cpt,
            disable_tqdm=disable_tqdm,
        )

    elapsed = time.time() - t0
    app_logger.info("end of analysis in %1.1f seconds", elapsed)
    return cpt


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    nouvelle_date_format = "%d-%m-%Y %H:%M:%S"
    nouveau_formatter = logging.Formatter(log_format, datefmt=nouvelle_date_format)
    console_handler_app = logging.StreamHandler(sys.stdout)
    console_handler_app.setFormatter(nouveau_formatter)
    app_logger.addHandler(console_handler_app)
    app_logger.setLevel(log_level)

    cpt = treat_one_day_wrapper(
        day2treat=args.day2treat,
        outputdir=args.outputdir,
        mission=args.mission,
        mode=args.mode,
        producttype=args.producttype,
        confpath=args.conf,
        disable_tqdm=False,
    )
    for uu in cpt:
        logging.info(
            "\ncounters for day %s , key %s = %s\n", args.day2treat, uu, cpt[uu]
        )


if __name__ == "__main__":
    main()
