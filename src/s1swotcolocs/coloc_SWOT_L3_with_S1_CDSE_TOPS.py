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

import collections
import datetime
import glob
import logging
import os
import sys
import time
import traceback
import warnings

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
import s1swotcolocs
from s1swotcolocs.utils import get_conf_content
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


def treat_a_clean_piece_of_swot_orbit(
    swotpiece, points, swotsub, mode, producttype, delta_t_max, cpt
):
    """
    :param swotpiece: shapely.geometry.Polygon simplified, not crossing antimeridian
    :param points: 2D matrix with lon and lat from SWOT
    :param swotsub: sub part of a SWOT xarray.Dataset 
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
            "collection": ["SENTINEL-1"],
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
        cache_dir=None,
    )
    return collected_data_norm


def save_netcdf_file_per_swot_piece_orbit_core(
    cdse_output, swot_gdf, fpath_out, delta_t_max, cpt
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

    start_time_strings = cdse_output["ContentDate"].str["Start"]
    cdse_output["Start_dt"] = pd.to_datetime(start_time_strings, utc=True)

    for sasa in range(len(cdse_output["geometry"])):
        all_SAR_polygones.append("%s" % cdse_output["geometry"].iloc[sasa])
        SAR_start_slice = cdse_output["Start_dt"].iloc[sasa]
        delta_diff_time = SWOT_start_piece - SAR_start_slice
        delta_diff_time_minutes = delta_diff_time / np.timedelta64(1, "m")
        all_start_SAR.append(SAR_start_slice.tz_localize(None))
        all_delta_times.append(delta_diff_time_minutes)
        all_SWOT_fpath.append(cdse_output["id_original_query"].iloc[sasa].split(" ")[1])

    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    SWOT_start_piece = np.array(SWOT_start_piece.tz_localize(None)).astype(
        "datetime64[s]"
    )
    sar_names = np.array(cdse_output["Name"].values, dtype=object)
    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    all_delta_times = np.array(all_delta_times, dtype=np.float64)
    SWOT_start_piece = np.array(SWOT_start_piece).astype("datetime64[s]")

    colocds = xr.Dataset()
    colocds["sar_safe_name"] = xr.DataArray(
        cdse_output["Name"].values,
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
    cddesS1outputs, SWOTgdfs, dir_output, delta_t_max, cpt, disable_tqdm=False
):
    assert len(cddesS1outputs) == len(SWOTgdfs)
    for xxi in tqdm(range(len(cddesS1outputs)), disable=disable_tqdm):
        one_cds_output = cddesS1outputs[xxi]
        swot_gdf = SWOTgdfs[xxi]
        if one_cds_output is not None:
            SWOT_start_piece = np.datetime64(swot_gdf["id_query"][0].split(" ")[2])
            swot_formated_date, year, month, day = get_swot_date_info(SWOT_start_piece)
            fpath_out = os.path.join(
                dir_output,
                "%s" % year,
                "%s" % month,
                "%s" % day,
                "coloc_SWOT_L3_Sentinel-1_IW_%s.nc" % swot_formated_date,
            )
            app_logger.info("fpath_out: %s", fpath_out)
            cpt = save_netcdf_file_per_swot_piece_orbit_core(
                cdse_output=one_cds_output,
                swot_gdf=swot_gdf,
                fpath_out=fpath_out,
                delta_t_max=delta_t_max,
                cpt=cpt,
            )
            cpt["written"] += 1
        else:
            cpt["no_coloc"] += 1

    app_logger.info(
        "number of coloc files written : %i/%i", cpt["written"], len(cddesS1outputs)
    )
    app_logger.info(
        "number of SWOT piece of orbit without S1 coloc : %i/%i",
        cpt["no_coloc"],
        len(cddesS1outputs),
    )
    return cpt


def parse_args():
    parser = argparse.ArgumentParser(description="S1-SWOT meta coloc")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--day2treat", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--mode",
        required=False,
        choices=["IW", "EW"],
        default="IW",
        help="IW or EW [default=IW]",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="directory where to store output netCDF files",
    )
    parser.add_argument("--conf", required=True, help="config file to use")
    return parser.parse_args()


def treat_one_day_wrapper(
    day2treat, outputdir, mode, confpath, disable_tqdm=False, dev=False
):
    """
    :param day2treat: str YYYYMMDD
    :param outputdir: str
    :param mode: str "IW" or "EW"
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
            mode=mode,
            cpt=cpt,
            tolerance_simplification=conf["TOLERANCE_SIMPLIFICATION"],
        )
        SWOTgdfs += swot_distinct_gdfs_list

    app_logger.info("GeoDataFrames prepared for CDSE queries: %s", cpt)
    app_logger.info("nb GeoDataFrames: %i", len(SWOTgdfs))

    cddesS1outputs = []
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

        cddesS1outputs.append(res)

    app_logger.info("CDSE queries performed.")

    if len(SWOTgdfs) > 0:
        cpt = save_meta_coloc_output(
            cddesS1outputs,
            SWOTgdfs,
            dir_output=outputdir,
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
        mode=args.mode,
        confpath=args.conf,
        disable_tqdm=False,
    )
    for uu in cpt:
        logging.info(
            "\ncounters for day %s , key %s = %s\n", args.day2treat, uu, cpt[uu]
        )


if __name__ == "__main__":
    main()
