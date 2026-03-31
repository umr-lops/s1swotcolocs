import argparse
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
import cdsodatacli
import cdsodatacli.query
import s1swotcolocs
from s1swotcolocs.utils import get_conf_content
from s1swotcolocs.check_lonlat_polygon_extent import (
    check_longitude_smaller_than_latitude_extent,
)

# Ignorer uniquement FixWindingWarning
warnings.filterwarnings("ignore", category=FixWindingWarning)
warnings.filterwarnings("ignore", module="cdsodatacli")


app_logger = logging.getLogger(__file__)
LOGGERS_TO_SILENCE = ["cdsodatacli", "cdsodatacli.query"]

for logger_name in LOGGERS_TO_SILENCE:
    lib_logger = logging.getLogger(logger_name)
    lib_logger.handlers.clear()  # Supprimer tous les handlers existants
    lib_logger.addHandler(logging.NullHandler())  # Envoyer les logs nulle part
    lib_logger.propagate = False  # Ne pas transmettre aux parents
    lib_logger.setLevel(
        logging.CRITICAL + 1
    )  # Mettre un niveau très élevé pour être sûr


class CDSODATACLIQueryFilter(logging.Filter):
    def filter(self, record):
        # Allow only messages that are not from cdsodatacli.query
        return not record.name.startswith("cdsodatacli.query")


class SuppressCDSODATACLIQuery(logging.Filter):
    def filter(self, record):
        # Bloque tous les messages venant du module cdsodatacli.query ou sous-modules
        return not record.name.startswith("cdsodatacli.query")


def treat_a_clean_piece_of_swot_orbit(
    swotpiece, points, onedsswot, mode, producttype, delta_t_max
):
    """

    :param swotpiece: shapely.geometry.Polygon that is simplified and that do no cross antimeridian
    :param points: 2D matrix with lon and lat from SWOT
    :param onedsswot: dataset xarray SWOT L3
    :return:
    """
    # app_logger.info('partswot: %i size %i', iip, len(swotpiece.exterior.xy[0]))
    app_logger.debug("swotpiece : %s", swotpiece)
    original_filename = os.path.basename(onedsswot.encoding["source"])
    lonmin = np.amin(swotpiece.exterior.xy[0])
    lonmax = np.amax(swotpiece.exterior.xy[0])
    latmin = np.amin(swotpiece.exterior.xy[1])
    latmax = np.amax(swotpiece.exterior.xy[1])

    # do the link between the closest point in the SWOT file and the dimension num_lines to get the associated time
    # SWOT always go to the EAST contrarily to Heliosynchronous mission going to West
    # points = np.c_[.ravel(), y.ravel()]
    tree = spatial.KDTree(points)
    app_logger.debug("coords North point: %s %s", lonmax, latmax)
    # idx_north = sorted(tree.query_ball_point([lonmax, latmax], 1))
    # idx_south = sorted(tree.query_ball_point([lonmin, latmin], 1))
    dd, idx_north = tree.query([lonmax, latmax], k=1)
    dd, idx_south = tree.query([lonmin, latmin], k=1)
    app_logger.debug("idx_north : %s", idx_north)
    app_logger.debug("idx_south : %s", idx_south)
    num_line_idx_north, _ = np.unravel_index(idx_north, onedsswot["longitude"].shape)
    num_line_idx_south, _ = np.unravel_index(idx_south, onedsswot["longitude"].shape)
    app_logger.debug("num_line_idx_north : %s", num_line_idx_north)
    app_logger.debug("num_line_idx_south : %s", num_line_idx_south)
    time_north = onedsswot["time"].isel(num_lines=num_line_idx_north).values
    time_south = onedsswot["time"].isel(num_lines=num_line_idx_south).values
    app_logger.debug("time_north %s", time_north)
    app_logger.debug("time_south %s", time_south)
    if time_north > time_south:
        startswot = time_south
        stopswot = time_north
        sta = pd.to_datetime((startswot - delta_t_max)).round("us").to_pydatetime()
        sto = pd.to_datetime((stopswot + delta_t_max)).round("us").to_pydatetime()
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

    return gdf


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
    treat the SWOT swath by pieces to avoid too large polygon when computing the convex_hull of the piece

    :param onedsswot: xarray.Dataset SWOT L3 dataset
    :param idxstart: int beginning of the part of SWOT orbit to be treated
    :param idxstop: int end of the part of SWOT orbit to be treated
    :param cpt: collections.defaultdict(int)
    :param max_area_size: float maximum size of the SWOT polygon area (security to discard weird polygons)
    :param delta_hours: int maximum time window (in hours) to search for S1 product around the SWOT date
    :param mode: str IW or EW
    :param producttype: str SLC or GRD
    :param tolerance_simplification: float to simplify SWOT swath polygon (used as alphashape alpha)
    :return:
        sub_gdf (list): list of the geodataframes computed on each piece of SWOT orbit
         (we can have several pieces per segment if there is land interrupting the swath)
    """
    sub_gdf = []

    delta_t_max = np.timedelta64(delta_hours, "h")
    steps_in_swot = 20
    swotsub = onedsswot.isel({"num_lines": slice(idxstart, idxstop, steps_in_swot)})
    lonswot = swotsub["longitude"].values.ravel()
    lonswot[lonswot > 180] += -360.0
    points = np.column_stack((lonswot, swotsub["latitude"].values.ravel()))
    # Create a MultiPoint object
    multi_point = MultiPoint(points)

    # Get the convex hull (smallest polygon enclosing all points)
    # polygon = multi_point.convex_hull
    gdfswot = gpd.GeoDataFrame(geometry=list(multi_point.geoms))
    alpha_shape_swot = alphashape.alphashape(gdfswot, alpha=tolerance_simplification)
    land_path = geodatasets.get_path("naturalearth.land")
    land = gpd.read_file(land_path)  # .to_crs(epsg=3857)
    # land_union = land.unary_union  # a single MultiPolygon of all land
    land_union = land.union_all()

    # --- Step 3: Subtract land from your polygon ---
    ocean_part = alpha_shape_swot.difference(land_union)
    # simplify the swot polygon on ocean
    tolerance = 0.5  # Adjust the tolerance value to control the level of simplification
    # tolerance = 0.1
    tolerance = tolerance_simplification
    simplified_polygon = ocean_part.simplify(tolerance)
    # simplified_polygon = simplified_polygon.make_valid()
    simplified_polygon = simplified_polygon.buffer(0)
    # print(simplified_polygon.area) # usually not exceeding 200km²
    if simplified_polygon.is_empty is False:
        app_logger.debug("ocean_part : %s", ocean_part)
        # simplified_polygon = ocean_part
        # simplified_polygon = polygon
        if isinstance(simplified_polygon, MultiPolygon):
            app_logger.debug("Nb parts: %i", len(simplified_polygon.geoms))
            for iip, partswot in enumerate(simplified_polygon.geoms):
                try:
                    subpartswot = fix_polygon(
                        partswot
                    )  # fix antimeridian crossing if needed
                except AssertionError:
                    cpt["impossible_to_fix_polygon"] += 1
                    continue
                if isinstance(subpartswot, MultiPolygon):
                    cpt["segment_interupted_by_land_and_antimeridian"] += 1
                    for yyp, subsubpartswot in enumerate(subpartswot.geoms):
                        is_ok_extents = check_longitude_smaller_than_latitude_extent(
                            subsubpartswot
                        )
                        if subsubpartswot.area < max_area_size and is_ok_extents:
                            gdf = treat_a_clean_piece_of_swot_orbit(
                                subsubpartswot,
                                points,
                                onedsswot,
                                mode,
                                producttype,
                                delta_t_max,
                            )
                            sub_gdf.append(gdf)
                        else:
                            cpt["segment_with_area_too_large"] += 1
                else:
                    cpt["segment_interupted_by_land_only"] += 1
                    if subpartswot.area < max_area_size:
                        gdf = treat_a_clean_piece_of_swot_orbit(
                            subpartswot,
                            points,
                            onedsswot,
                            mode,
                            producttype,
                            delta_t_max,
                        )
                        sub_gdf.append(gdf)
                    else:
                        cpt["segment_with_area_too_large"] += 1
        else:  # easy case with only one polygon contigous over ocean for the SWOT file
            # cpt['continuous_polygon_over_segment'] += 1
            try:
                subpartswot = fix_polygon(
                    simplified_polygon, fix_winding=True
                )  # fix antimeridian crossing if needed
            except AssertionError:
                cpt["error_at_fix_antimeridian"] += 1
                app_logger.error("%s", traceback.format_exc())
                subpartswot = None
                pass
            if subpartswot is not None:
                if isinstance(subpartswot, MultiPolygon):
                    cpt["segment_continuous_with_antimeridian"] += 1
                    for yyp, subsubpartswot in enumerate(subpartswot.geoms):
                        if subsubpartswot.area < max_area_size:
                            gdf = treat_a_clean_piece_of_swot_orbit(
                                subsubpartswot,
                                points,
                                onedsswot,
                                mode,
                                producttype,
                                delta_t_max,
                            )
                            sub_gdf.append(gdf)
                        else:
                            cpt["segment_with_area_too_large"] += 1
                else:
                    cpt["segment_continuous_without_antimeridian"] += 1
                    if subpartswot.area < max_area_size:
                        gdf = treat_a_clean_piece_of_swot_orbit(
                            subpartswot,
                            points,
                            onedsswot,
                            mode,
                            producttype,
                            delta_t_max,
                        )
                        sub_gdf.append(gdf)
                    else:
                        cpt["segment_with_area_too_large"] += 1

    else:
        app_logger.debug("one empty polygon")
        cpt["empty_polygon"] += 1
    # app_logger.info('counter: %s',cpt)
    return sub_gdf, cpt


def get_swot_geoloc(
    one_swot_l3_file,
    max_area_size,
    delta_hours=6,
    mode="IW",
    producttype="SLC",
    cpt=None,
    tolerance_simplification=0.1,
) -> (list, collections.defaultdict):
    """

    :param one_swot_l3_file: str
    :param max_area_size: float
    :param delta_hours: int
    :param mode: str IW or EW
    :param producttype: str SLC or GRD
    :param cpt: collections.defaultdict(int)
    :param tolerance_simplification:
    :return:
        allgdfs_swot: list
        cpt: collections.defaultdict
    """
    allgdfs_swot = []
    app_logger.debug("%s", one_swot_l3_file)
    onedsswot = xr.open_dataset(one_swot_l3_file)
    app_logger.debug("full size time %s", onedsswot["time"].sizes)
    segment = 1000  # number of points in the azimuth direction
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
    collected_data_norm = cdsodatacli.query.fetch_data(
        gdf,
        min_sea_percent=mini_ocean,
        timedelta_slice=datetime.timedelta(days=4),
        cache_dir=None,
    )
    # print(collected_data_norm)
    return collected_data_norm


def save_netcdf_file_per_swot_piece_orbit_core(
    cdse_output, swot_gdf, fpath_out, delta_t_max, cpt
):
    """

    save the result for one SWOT query matching one or more S1 product(s)

    :param cdse_output: pd.GeoDataFrame (containing Sentinel-1 meta data)
    :param swot_gdf: pd.GeoDataFrame (original SWOT geodataframe)
    :param fpath_out: str full path of the output file
    :param delta_t_max: int number of hours plus or minus considered for coloc
    :param cpt: collections.defaultdict(int)
    :return:
    """
    # merged_gdf = pd.merge([cdse_output,swot_gdf])

    # SWOT_start_piece = np.datetime64(swot_gdf['id_query'][0].split(' ')[2])
    SWOT_start_piece = pd.to_datetime(swot_gdf["id_query"][0].split(" ")[2])
    SWOT_start_piece = SWOT_start_piece.tz_localize("UTC")
    # filepath_swot = swot_gdf["id_query"][0].split(" ")[1]

    swot_polygon = "%s" % swot_gdf["geometry"][0]
    # all_matches = []
    all_SAR_polygones = []
    all_SWOT_fpath = []
    all_start_SAR = []
    all_delta_times = []
    # 1. Extract the 'Start' time strings into a new Series.
    start_time_strings = cdse_output["ContentDate"].str["Start"]
    # 2. Convert the entire Series to timezone-aware datetime objects, standardized to UTC.
    #    This is the key step that resolves the warning explicitly and efficiently.
    cdse_output["Start_dt"] = pd.to_datetime(start_time_strings, utc=True)
    for sasa in range(len(cdse_output["geometry"])):
        all_SAR_polygones.append("%s" % cdse_output["geometry"].iloc[sasa])
        # SAR_start_slice = np.datetime64(cdse_output['ContentDate'].iloc[sasa]['Start'])  # .astype('<M8[ns]')
        # The value is a Pandas Timestamp, which works just like np.datetime64.
        SAR_start_slice = cdse_output["Start_dt"].iloc[sasa]
        delta_diff_time = SWOT_start_piece - SAR_start_slice
        delta_diff_time_minutes = delta_diff_time / np.timedelta64(1, "m")
        # assert isinstance(SAR_start_slice, np.datetime64)
        all_start_SAR.append(SAR_start_slice.tz_localize(None))
        all_delta_times.append(delta_diff_time_minutes)
        all_SWOT_fpath.append(cdse_output["id_original_query"].iloc[sasa].split(" ")[1])
    # for xx in cdse_output['Name'].values:
    #     # print(xx)
    #     # all_matches += uu['Name'].values
    #     if xx not in all_matches:
    #         all_matches.append(xx)
    all_start_SAR = np.array(all_start_SAR).astype("datetime64[s]")
    SWOT_start_piece = np.array(SWOT_start_piece.tz_localize(None)).astype(
        "datetime64[s]"
    )
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
    # colocds["sar_start_time_slice"] = xr.DataArray(
    #     all_start_SAR,
    #     dims="coloc",
    #     attrs={
    #         "description": "SAR product start of slice date",
    #     },
    # )
    # 'units' : "seconds since 1970-01-01 00:00:00",
    # 'calendar' : "standard",})
    colocds["SWOT_start_time_slice"] = xr.DataArray(
        SWOT_start_piece, attrs={"description": "SWOT slice start date"}
    )
    colocds["sar_safe_name"] = xr.DataArray(
        cdse_output["Name"].values,
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
    # colocds.attrs["filepath_swot"] = filepath_swot
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
        swot_formated_date (str): eg 20251017T151210
        year (int): 
        month (str): MM
        day (str): DD
    """
    dt_py = SWOT_start_piece.astype("M8[D]").astype(object)

    # Extract year, month, day
    year = dt_py.year
    month = f"{dt_py.month:02d}"
    day = f"{dt_py.day:02d}"
    # year = SWOT_start_piece.astype('datetime64[Y]').astype(int) + 1970
    # month = SWOT_start_piece.astype('datetime64[M]').astype(int) % 12 + 1
    # day = SWOT_start_piece - SWOT_start_piece.astype('datetime64[M]') + 1
    swot_formated_date = (
        ("%s" % SWOT_start_piece)
        .replace("-", "")
        .replace(":", "")
        .split(".")[0]
    )
    return swot_formated_date,year,month,day

def save_meta_coloc_output(
    cddesS1outputs, SWOTgdfs, dir_output, delta_t_max, cpt, disable_tqdm=False
):
    """


    :param cddesS1outputs:
    :param SWOTgdfs: list of geodataframes
    :param dir_output: str
    :param delta_t_max: int
    :param cpt: collections.defaultdict(int)
    :param disable_tqdm: bool True-> hide progress bar in stdout
    :return:
    """
    assert len(cddesS1outputs) == len(
        SWOTgdfs
    )  # there is as much CDSE returns than SWOT gdf
    for xxi in tqdm(range(len(cddesS1outputs)), disable=disable_tqdm):

        one_cds_output = cddesS1outputs[xxi]
        swot_gdf = SWOTgdfs[xxi]
        # if not one_cds_output.empty:

        if one_cds_output is not None:
            SWOT_start_piece = np.datetime64(swot_gdf["id_query"][0].split(" ")[2])
            swot_formated_date,year,month,day = get_swot_date_info(SWOT_start_piece)
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
    parser.add_argument("--day2treat", required=True, help="for instance YYYYMMDD")
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
    args = parser.parse_args()
    return args


def treat_one_day_wrapper(day2treat, outputdir, mode, confpath, disable_tqdm=False):
    """

    :param day2treat: datetime.datetime
    :param outputdir: str
    :param mode: str "IW" or "EW"
    :param confpath: str full path of the config.yml to be used
    :return:
    """
    t0 = time.time()
    conf = get_conf_content(confpath)
    # MAX_AREA_SIZE: 200
    # DELTA_HOURS: 1
    dswot = conf["SWOT_L2_AVISO_DIR"]
    CACHE_CDSE = conf["CACHE_CDSE"]
    lstswotfiles = []
    dd = datetime.datetime.strptime(day2treat, "%Y%m%d")
    app_logger.info("treat day : %s", dd)
    pattern = os.path.join(dswot, "*nc")
    app_logger.info("pattern : %s", pattern)
    lstswotfiles += glob.glob(pattern)
    app_logger.info("Nb files SWOT found : %i", len(lstswotfiles))
    app_logger.info(
        "first step: creation of SWOT geodataframes with +/-%i hours shift vs S-1",
        conf["DELTA_HOURS"],
    )
    SWOTgdfs = []
    cpt = collections.defaultdict(int)
    cpt["nbSWOTfiles"] = len(lstswotfiles)
    for ii in tqdm(range(len(lstswotfiles)), disable=disable_tqdm):
        oneswotfile = lstswotfiles[ii]
        gdf, cpt = get_swot_geoloc(
            oneswotfile,
            delta_hours=conf["DELTA_HOURS"],
            max_area_size=conf["MAX_AREA_SIZE"],
            mode=mode,
            cpt=cpt,
            tolerance_simplification=conf["TOLERANCE_SIMPLIFICATION"],
        )
        # all_gdf.append(gdf)
        SWOTgdfs += gdf
        # if ii==5:
        #     break
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
            app_logger.error("problematic gdf: %s", gdf)
            app_logger.error("traceback: %s", traceback.format_exc())
            res = None
            cpt["problematic_gdf"] += 1
            raise ValueError

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
    """

    treat a day of SWOT data to get match-ups with Sentinel-1

    :param mode: str IW or EW
    :return:
    """
    # root = logging.getLogger()
    # if root.handlers:
    #     for handler in root.handlers:
    #         root.removeHandler(handler)

    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # log_format = '%(asctime)s %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s'
    log_format = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    nouvelle_date_format = "%d-%m-%Y %H:%M:%S"
    nouveau_formatter = logging.Formatter(log_format, datefmt=nouvelle_date_format)
    console_handler_app = logging.StreamHandler(
        sys.stdout
    )  # Écrit sur la sortie standard
    console_handler_app.setFormatter(nouveau_formatter)
    app_logger.addHandler(console_handler_app)
    app_logger.setLevel(log_level)
    # if app_logger.hasHandlers():  # Vérifie s'il y a des handlers attachés directement
    #     for handler in app_logger.handlers:
    #         handler.setFormatter(nouveau_formatter)
    # else:
    #     print('app_logger na pas de handler')
    # Utiliser force=True pour s'assurer que cette configuration s'applique,
    # écrasant toute configuration de basicConfig faite par une bibliothèque sans force=True.
    # logging.basicConfig(level=log_level, format=log_format,
    #                     datefmt='%Y-%m-%d %H:%M:%S', force=True)
    # Apply the filter to the root logger (or any specific handler)
    # root_logger = logging.getLogger()
    # for handler in root_logger.handlers:
    #     handler.addFilter(SuppressCDSODATACLIQuery())

    # # Cibler les loggers problématiques
    # for logger_name in ["cdsodatacli", "cdsodatacli.query"]:
    #     logger = logging.getLogger(logger_name)
    #     logger.setLevel(logging.CRITICAL)  # Bloquer tout sauf les erreurs critiques
    #     logger.propagate = False  # Empêche les logs de remonter au logger parent
    #
    #     # Supprime tous les handlers qui pourraient afficher les logs
    #     logger.handlers.clear()
    cpt = treat_one_day_wrapper(
        day2treat=args.day2treat,
        outputdir=args.outputdir,
        mode=args.mode,
        confpath=args.conf,
    )
    for uu in cpt:
        logging.info(
            "\ncounters for day %s , key %s = %s\n", args.day2treat, uu, cpt[uu]
        )


if __name__ == "__main__":
    main()
