import collections

import s1ifr
import os
import argparse
import pandas as pd
import sys
import glob
import logging
import xarray as xr
from tqdm import tqdm
from scipy import spatial
import numpy as np
import alphashape
import geopandas as gpd
from shapely.geometry import MultiPoint
import datetime
from itertools import repeat
from shapely import wkt
from s1ifr import paths_safe_product_family
from s1swotcolocs.utils import get_conf_content
from s1swotcolocs.pickup_best_swot_file import check_if_latest_version

DEFAULT_IFREMER_S1_VERSION_L1B = ["A17", "A18", "A21", "A23", "A16", "A15"]
SWOT_SWATH_LIMITS = {"left": (0, 34), "right": (35, 69)}
DEFAULT_SWOT_VARIABLES = [
    "latitude",
    "longitude",
    "swh_karin",
    "wind_speed_karin",
    "wind_speed_karin_2",
    "wind_speed_model_u",
    "wind_speed_model_v",
    "swh_nadir_altimeter",
]  # Level2 # ,'swh_karin_qual','sig0_karin_uncert'
UNTRUSTABLE_SWH = 30  # m, threshold above which the SWH is considered untrustable
lines_to_keep_in_swot_swath = [28, 29, 30, 39, 40, 41]
app_logger = logging.getLogger(__file__)
console_handler_app = logging.StreamHandler(sys.stdout)


def xndindex(sizes):
    """
    xarray equivalent of np.ndindex iterator with defined dimension names

    Args:
        sizes (dict): dict of form {dimension_name (str): size(int)}
    Return:
        iterator over dict
    """

    for d, k in zip(
        repeat(tuple(sizes.keys())), zip(np.ndindex(tuple(sizes.values())))
    ):
        yield {k: lll for k, lll in zip(d, k[0])}


def get_original_sar_filepath_from_metacoloc(
    metacolocds, cpt, pola_selected="SDV"
) -> (list, dict):
    """

    :param metacolocds:
    :param cpt: defaultdict
    :return:
        fullpath_iw_slc_safes: str
    """
    fullpath_iw_slc_safe = None
    basesafes = metacolocds["sar_safe_name"].values
    fullpath_iw_slc_safes = []
    cpt["safe_listed_in_metacoloc"] += len(basesafes)
    for basesafe in basesafes:
        if pola_selected in basesafe:
            app_logger.info("basesafe : %s", basesafe)
            orignal_s1 = s1ifr.get_path_from_base_safe.get_path_from_base_safe(
                basesafe, archive_name="datawork"
            )
            orignal_s1_bis = s1ifr.get_path_from_base_safe.get_path_from_base_safe(
                basesafe, archive_name="scale"
            )
            if (
                os.path.exists(orignal_s1) is True
                or os.path.exists(orignal_s1_bis) is True
            ):
                if os.path.exists(orignal_s1) is True:
                    fullpath_iw_slc_safe = orignal_s1
                else:
                    fullpath_iw_slc_safe = orignal_s1_bis
                if os.path.exists(fullpath_iw_slc_safe):
                    fullpath_iw_slc_safes.append(fullpath_iw_slc_safe)
                else:
                    cpt["safe_iw_slc_not_found"] += 1
                    # fullpath_iw_slc_safes.append(None)
                    app_logger.debug("dev,safe missing %s", fullpath_iw_slc_safe)
            else:
                cpt["safe_iw_slc_not_found"] += 1
                app_logger.debug("dev,safe missing %s", orignal_s1)
        else:
            cpt["IW_SAFE_BAD_POLA"] += 1

    return fullpath_iw_slc_safes, cpt


def get_L2WAV_S1_IW_path(fullpath_s1_iw_slc, version_L1B=None):
    """

    :param fullpath_s1_iw_slc: str
    :param version_L1B: list of str
    :return:
    """
    if version_L1B is None:
        version_L1B = DEFAULT_IFREMER_S1_VERSION_L1B
    path_l2wav_sar = None
    df = pd.DataFrame({"L1_SLC": [fullpath_s1_iw_slc]})
    app_logger.info("df : %s", df)
    newdf = paths_safe_product_family.get_products_family(
        df, l1bversions=version_L1B, disable_tqdm=True
    )
    if len(newdf["L2_WAV_E12"]) > 0:
        path_l2wav_sar = newdf["L2_WAV_E12"].values[0]
    elif len(newdf["L2_WAV_E11"]) > 0:
        path_l2wav_sar = newdf["L2_WAV_E11"].values[0]

    return path_l2wav_sar


def read_swot_windwave_l2_file(metacolocds, confpath, cpt=None) -> (xr.Dataset, str):
    """

    :param metacolocds: xr.Dataset
    :param confpath: str
    :param cpt: collections.defaultdict
    :return:
        dsswotl2: xr.Dataset or None SWOT Level-2 WindWave AVISO product
    """
    dsswotl2 = None
    pathswotl2final = None
    baseswotl3 = metacolocds["filepath_swot"].values[0].item()
    app_logger.debug("baseswotl3 %s", baseswotl3)
    conf = get_conf_content(confpath)
    dir_swot_l3 = conf["SWOT_L3_AVISO_DIR"]
    assert os.path.exists(dir_swot_l3)

    date_swot_dt = datetime.datetime.strptime(baseswotl3.split("_")[7], "%Y%m%dT%H%M%S")
    full_path_swot = os.path.join(
        dir_swot_l3,
        date_swot_dt.strftime("%Y"),
        date_swot_dt.strftime("%j"),
        baseswotl3,
    )
    if os.path.exists(full_path_swot):
        dir_swot_l2 = conf["SWOT_L2_AVISO_DIR"]
        pattern_l2 = os.path.join(
            dir_swot_l2,
            baseswotl3.replace("Expert", "WindWave")
            .replace("L3", "L2")
            .split("_v")[0][0:-17]  # remove the dates up to last second
            + "*.nc",
        )
        app_logger.debug("pattern_l2 %s", pattern_l2)
        lst_nc = glob.glob(pattern_l2)
        if len(lst_nc) > 0:
            pathswotl2final = lst_nc[0]
            is_the_latest_version, true_latest_file = check_if_latest_version(
                file_to_check=pathswotl2final, all_available_files=lst_nc
            )
            if true_latest_file is not None:
                cpt["swot_file_change_for_latest"] += 1
                pathswotl2final = true_latest_file
            else:
                cpt["swot_file_direct_pickup_latest"] += 1
            dsswotl2 = xr.open_dataset(pathswotl2final).load()
            dsswotl2["longitude"] = dsswotl2["longitude"].where(
                dsswotl2["longitude"] < 180, dsswotl2["longitude"] - 360
            )

        else:
            app_logger.debug("dev  L2 SWOT pattern missing %s", pattern_l2)
    else:
        cpt["swot_L3_file_absent"] += 1
        app_logger.debug("dev L3 SWOT missing %s", full_path_swot)
    return dsswotl2, pathswotl2final, cpt


def create_empty_coloc_res(indexes_sar_grid) -> xr.Dataset:
    """

    :param indexes_sar_grid: dict with tile_line and tile_sample keys
    :return:
        empty_dummy_condensated_swot_coloc: xr.Dataset filled with NaN variables
    """
    empty_dummy_condensated_swot_coloc = xr.Dataset()
    # fcts = {'mean': np.nanmean,
    #             'med': np.nanmedian,
    #             'std': np.nanstd}
    fcts = {"mean": np.mean, "med": np.median, "std": np.std}
    for vv in DEFAULT_SWOT_VARIABLES:
        for fct in fcts:
            empty_dummy_condensated_swot_coloc["%s_%s" % (vv, fct)] = xr.DataArray(
                np.nan,
                attrs={
                    "description": "%s of %s variable from SWOT product" % (fct, vv)
                },
            )
    empty_dummy_condensated_swot_coloc = (
        empty_dummy_condensated_swot_coloc.assign_coords(indexes_sar_grid)
    )
    empty_dummy_condensated_swot_coloc = empty_dummy_condensated_swot_coloc.expand_dims(
        ["tile_line", "tile_sample"]
    )
    return empty_dummy_condensated_swot_coloc


def filter_out_swot_swath_edges(dsswotl2):
    """

    This method would allow to filter out the edges of the SWOT swath where the SWH and wind speed values are often unphysical.
    it could be used for data in PIC0 processing but it is interesting to keep the product as it is only using the quality flags
    this allows to monitore product improvements over time.

    Args:
        dsswotl2: xr.Dataset SWOT Level-2 WindWave AVISO product

    Returns:
        dsswotl2: xr.Dataset SWOT Level-2 WindWave AVISO product with edges set to np.nan


    """
    # definition of index num_pixels to set to np.nan
    slices = [slice(0, 4), slice(29, 39), slice(65, 69)]

    # Mettre à np.nan les valeurs pour les bandes spécifiées
    dsswotl2 = dsswotl2.copy()  # Pour éviter de modifier l'original
    for s in slices:
        dsswotl2["swh_karin"][:, s] = np.nan
        dsswotl2["wind_speed_karin"][:, s] = np.nan
        dsswotl2["wind_speed_karin_2"][:, s] = np.nan

    return dsswotl2


def compute_gradient_swot_swh(dsswotl2):
    """

    use gradient of SWH to spot unphysical values

    Args:
        dsswotl2: xr.Dataset SWOT Level-2 WindWave AVISO product

    Returns:
        dsswotl2: xr.Dataset SWOT Level-2 WindWave AVISO product with dHsdx and dHsdy variables added (gradient of SWH in m/km)

    """
    #################################################################
    # FILTERING UNFILTERED RAIN OR BUMPS IN HS CLEARLY NOT PHYSICAL #
    #################################################################

    # columns_not_fully_nan = np.argwhere(np.nansum(dsswotl2.swh_karin.values,axis=0) !=0)
    # summed_columns_zero_or_non_zero = np.nansum(dsswotl2.swh_karin.values,axis=0) != 0
    # transitions = np.diff(summed_columns_zero_or_non_zero.astype(int))

    # start_idxs = np.where(transitions == 1)[0] + 1  # add 1 because diff shifts by one
    # end_idxs = np.where(transitions == -1)[0]       # these are already at the right place
    # print('start_idxs',start_idxs)
    # print('end_idxs',end_idxs)
    # import pdb
    # pdb.set_trace()
    swh_karin_raw = dsswotl2["swh_karin"].values
    # On calcule les diff selon x ou selon y, avec algo de différence centrée - comme on a 2km de pixel spacing dans le produit WindWave (et expert), et qu'on prend -1 , +1, ça fait du 2dx = 2dy = 4 km)
    swh_karin_diff_x = np.nan * np.ones(swh_karin_raw.shape)
    swh_karin_diff_y = np.nan * np.ones(swh_karin_raw.shape)
    # for i in range(2):
    #     swh_karin_diff_x[:, start_idxs[i] + 1 : end_idxs[i] - 1] = swh_karin_raw[:,start_idxs[i] + 2 : end_idxs[i]] - swh_karin_raw[:,start_idxs[i] : end_idxs[i] - 2]
    swh_karin_diff_x[:, 1:-1] = (
        swh_karin_raw[:, 2:] - swh_karin_raw[:, :-2]
    )  # simplification compared to orginal code
    swh_karin_diff_y[1:-1, :] = swh_karin_raw[2:, :] - swh_karin_raw[:-2, :]
    dsswotl2["dHsdx"] = (
        ("num_lines", "num_pixels"),
        swh_karin_diff_x / 4,
    )  # gradient en m/km
    dsswotl2["dHsdy"] = (
        ("num_lines", "num_pixels"),
        swh_karin_diff_y / 4,
    )  # gradient en m/km
    return dsswotl2


def s1swot_core_tile_coloc(
    lontile, lattile, treeswot, radius_coloc, dsswot, indexes_sar, cpt
) -> xr.Dataset:
    """

    for a given SAR tile, look for SWOT neighbors points

    Args:
        lontile: float longitude of a SAR point
        lattile: float latitude of a SAR point
        treeswot: scipy.spatial SWOT
        radius_coloc: float
        dsswot: xr.Dataset SWOT data
        indexes_sar: tile_line and tile_sample indexes to be able to reconstruct the SAR swath

    Returns:
        condensated_swot: xr.Dataset
    """
    neighbors = treeswot.query_ball_point([lontile, lattile], r=radius_coloc)
    indices = []
    condensated_swot = xr.Dataset()
    for oneneighbor in neighbors:
        index_original_shape_swot_num_lines, index_original_shape_swot_num_pixels = (
            np.unravel_index(oneneighbor, dsswot["longitude"].shape)
        )
        indices.append(
            (index_original_shape_swot_num_lines, index_original_shape_swot_num_pixels)
        )
    subset = [dsswot.isel(num_lines=i, num_pixels=j) for i, j in indices]

    if len(subset) > 0:
        swotclosest = xr.concat(subset, dim="points")

        # number of points in the radius
        condensated_swot["nb_SWOT_points"] = xr.DataArray(len(swotclosest["points"]))
        # variables wind/ std / mean /median
        fcts = {"mean": np.nanmean, "med": np.nanmedian, "std": np.nanstd}

        for vv in DEFAULT_SWOT_VARIABLES:
            for fct in fcts:
                if np.isfinite(swotclosest[vv].values).any():
                    if np.isfinite(swotclosest[vv].values).sum() == 1:
                        if fct == "std":
                            valval = np.nan
                        else:
                            valval = swotclosest[vv].values
                    else:
                        # many pts I use quality flag SWOT
                        if vv == "swk_karin":
                            maskswotswhqual = (
                                (swotclosest["swh_karin_qual"].values == 0)
                                & (swotclosest["rain_flag"].values == 0)
                                & (swotclosest["swh_karin"].values > 0)
                                & (swotclosest["swh_karin"].values < UNTRUSTABLE_SWH)
                            )
                        else:
                            maskswotswhqual = True
                        valval = fcts[fct](swotclosest[vv].values[maskswotswhqual])

                else:
                    valval = (
                        np.nan
                    )  # to avoid RuntimeWarning Degrees of freedom <= 0 for slice
                condensated_swot["%s_%s" % (vv, fct)] = xr.DataArray(
                    valval,
                    attrs={
                        "description": "%s of %s variable from SWOT points within a %f deg radius after swh_karin_qual=0 and rain_flag=0 filtering"
                        % (fct, vv, radius_coloc)
                    },
                )
                condensated_swot["%s_%s" % (vv, fct)].attrs.update(
                    swotclosest[vv].attrs
                )
        condensated_swot = condensated_swot.assign_coords(indexes_sar)
        condensated_swot = condensated_swot.expand_dims(["tile_line", "tile_sample"])
        cpt["tile_with_SWOT_neighbors"] += 1
    else:
        app_logger.debug("one tile without SWOT neighbors")
        cpt["tile_without_SWOT_neighbors"] += 1
        condensated_swot = create_empty_coloc_res(indexes_sar_grid=indexes_sar)
    return condensated_swot, cpt


def get_swot_tree(dsswot):
    """

    Arguments:
        dsswot (xr.Dataset)
    """
    lonswot = dsswot["longitude"].values.ravel()
    latswot = dsswot["latitude"].values.ravel()
    maskswot = np.isfinite(lonswot) & np.isfinite(latswot)
    points = np.c_[lonswot[maskswot], latswot[maskswot]]
    treeswot = spatial.KDTree(points)
    return treeswot


def loop_on_each_sar_tiles(
    dssar, dsswotl2, radius_coloc, full_path_swot, cpt
) -> (xr.Dataset, collections.defaultdict):
    """

    :param dssar: xr.Dataset Sentinel-1 IW Ifremer Level-2 WAV product
    :param dsswotl2: xr.Dataset SWOT Level-2 WindWave AVISO product
    :param radius_coloc: float
    :param full_path_swot: str
    :param cpt: defaultdict
    :return:
        ds_l2c (xr.Dataset)
        ds_l2c_nadir (xr.Dataset)
        cpt (collections.defaultdict)
    """

    dsswotl2_closenadir = dsswotl2.isel(
        {"num_pixels": lines_to_keep_in_swot_swath}
    )  # selection validated in notebook
    treeswot_nadir = get_swot_tree(dsswot=dsswotl2_closenadir)
    treeswot = get_swot_tree(dsswot=dsswotl2)
    # lonswot_nadir = dsswotl2_closenadir["longitude"].values.ravel()
    # latswot_nadir = dsswotl2_closenadir["latitude"].values.ravel()
    # maskswot_nadir = np.isfinite(lonswot_nadir) & np.isfinite(latswot_nadir)
    # points = np.c_[lonswot_nadir[maskswot_nadir], latswot_nadir[maskswot_nadir]]
    # treeswot_nadir = spatial.KDTree(points)

    # lonswot = dsswotl2["longitude"].values.ravel()

    # # lonswot = (lonswot + 180) % 360 - 180 # already done in read_swot_windwave_l2_file()
    # latswot = dsswotl2["latitude"].values.ravel()
    # maskswot = np.isfinite(lonswot) & np.isfinite(latswot)
    # app_logger.info(
    #     "nb NaN in the 2km SWOT grid: %i/%i",
    #     len(latswot) - maskswot.sum(),
    #     len(latswot),
    # )
    # points = np.c_[lonswot[maskswot], latswot[maskswot]]
    # treeswot = spatial.KDTree(points)

    all_tiles_colocs = []
    all_tiles_colocs_nadir = []
    gridsarL2 = {
        d: k
        for d, k in dssar.sizes.items()
        # if d in ["burst", "tile_sample", "tile_line"]
        if d in ["tile_sample", "tile_line"]
    }
    all_tile_cases = [i for i in xndindex(gridsarL2)]
    app_logger.info("enter the loop over SAR tiles")
    for x in tqdm(range(len(all_tile_cases)), desc="tile-loop", disable=True):
        # for ii in tqdm(range(len(sardf['lat_centroid_sar']))):
        # lontile = sardf['lon_centroid_sar'].iloc[ii]
        # lattile = sardf['lat_centroid_sar'].iloc[ii]

        i = all_tile_cases[x]
        lontile = dssar["longitude"][i].values
        lattile = dssar["latitude"][i].values

        if np.isfinite(lontile) and np.isfinite(lattile):
            tile_swot_condensated_at_SAR_point, cpt = s1swot_core_tile_coloc(
                lontile,
                lattile,
                treeswot,
                radius_coloc,
                dsswot=dsswotl2,
                indexes_sar=i,
                cpt=cpt,
            )
            tile_swot_nadir_condensated_at_SAR_point, cpt = s1swot_core_tile_coloc(
                lontile,
                lattile,
                treeswot_nadir,
                radius_coloc,
                dsswot=dsswotl2_closenadir,
                indexes_sar=i,
                cpt=cpt,
            )
        else:
            cpt["tile_sar_with_corrupted_geolocation"] += 1
            tile_swot_condensated_at_SAR_point = create_empty_coloc_res(
                indexes_sar_grid=i
            )
            tile_swot_nadir_condensated_at_SAR_point = create_empty_coloc_res(
                indexes_sar_grid=i
            )

        all_tiles_colocs.append(tile_swot_condensated_at_SAR_point)
        all_tiles_colocs_nadir.append(tile_swot_nadir_condensated_at_SAR_point)
    consolidated_all_tiles_colocs = []
    consolidated_all_tiles_colocs_nadir = []
    for uu in all_tiles_colocs_nadir:
        if "dim_0" in uu.dims:
            app_logger.debug("variables with dim_0 : %s", uu)
        else:
            consolidated_all_tiles_colocs_nadir.append(uu)
    for uu in all_tiles_colocs:
        if "dim_0" in uu.dims:
            app_logger.debug("variables with dim_0 : %s", uu)
        else:
            consolidated_all_tiles_colocs.append(uu)
    app_logger.info("merge the colocated tiles.")
    ds_colocation_swot_nadir = xr.merge(consolidated_all_tiles_colocs_nadir)
    ds_colocation_swot = xr.merge(
        consolidated_all_tiles_colocs,
    )
    # xr.combine_by_coords(all_tiles_colocs)
    app_logger.info("merge SAR and SWOT data.")
    ds_l2c = xr.merge([dssar, ds_colocation_swot])
    ds_l2c.attrs["SWOT_L3_data"] = full_path_swot
    ds_l2c_nadir = xr.merge([dssar, ds_colocation_swot_nadir])
    ds_l2c_nadir.attrs["SWOT_L3_data"] = full_path_swot
    return ds_l2c, ds_l2c_nadir, cpt


def save_sea_state_coloc_file(colocds, fpath_out, cpt):
    """
    Saves the dataset using the netcdf4 engine to avoid H5DSis_scale errors.
    """
    if os.path.exists(fpath_out):
        app_logger.info("remove the existing file")
        os.remove(fpath_out)
        cpt["file_replaced"] += 1
    else:
        app_logger.debug("file does not exist -> brand-new file on disk")
        cpt["new_file"] += 1

    if not os.path.exists(os.path.dirname(fpath_out)):
        os.makedirs(os.path.dirname(fpath_out), mode=0o775, exist_ok=True)

    # CHANGE: Use engine="netcdf4" instead of "h5netcdf"
    colocds.to_netcdf(fpath_out, engine="netcdf4")

    os.chmod(fpath_out, 0o664)
    app_logger.info("coloc file created : %s", fpath_out)
    return cpt


def associate_sar_and_swot_seastate_params(
    metacolocpath, confpath, groupsar="intraburst", overwrite=True, outputdir=None
):
    """

    :param metacolocpath: str (e.g. .../coloc_SWOT_L3_Sentinel-1_IW_20240729T172147.nc)
    :param confpath: str
    :param groupsar: str intraburst or interburst
    :param outputdir: path where to store the output sea state coloc files (.nc), will superseed the config file
    :return:
    """
    app_logger.info("SAR grid : %s", groupsar)
    cpt = collections.defaultdict(int)
    conf = get_conf_content(confpath)
    if outputdir is None:
        app_logger.info("outputdir from config file")
        outputdir = conf["HOST_SEASTATE_COLOC_OUTPUT_DIR"]
    mode = os.path.basename(metacolocpath).split("_")[4]  # IW or EW"
    metacolocds = xr.open_dataset(metacolocpath, engine="h5netcdf")
    if "filepath_swot" not in metacolocds:
        if "filepath_swot" in metacolocds.attrs:
            metacolocds["filepath_swot"] = xr.DataArray(
                np.tile(metacolocds.attrs["filepath_swot"], len(metacolocds["coloc"])),
                dims="coloc",
            )
        else:
            raise KeyError("filepath_swot not present in metacolocds")

    SWOT_start_piece = datetime.datetime.strptime(
        os.path.basename(metacolocpath).split("_")[5].replace(".nc", ""),
        "%Y%m%dT%H%M%S",
    )
    year = SWOT_start_piece.strftime("%Y")
    month = SWOT_start_piece.strftime("%m")
    day = SWOT_start_piece.strftime("%d")
    fullpath_iw_slc_safes, cpt = get_original_sar_filepath_from_metacoloc(
        metacolocds, cpt=cpt
    )
    date_swot_str = SWOT_start_piece.strftime("%Y%m%dT%H%M%S")
    app_logger.info("nb IW SAFE found at Ifremer: %i", len(fullpath_iw_slc_safes))
    new_files = []
    for iixx in tqdm(range(len(fullpath_iw_slc_safes)), disable=True):
        iw_slc_safe = fullpath_iw_slc_safes[iixx]
        cpt["total_safe_SAR_tested"] += 1
        app_logger.info("treat : %s", iw_slc_safe)
        dsswotl2, pathswotl2final, cpt = read_swot_windwave_l2_file(
            metacolocds, confpath=confpath, cpt=cpt
        )
        if dsswotl2 is not None:
            version_swot_processing = dsswotl2.attrs.get("crid", "unknown")
            part_swot_basename = (
                "SWOT_L2_WindWave_" + date_swot_str + "_" + version_swot_processing
            )
            path_l2wav_sar_safe = get_L2WAV_S1_IW_path(
                iw_slc_safe, version_L1B=DEFAULT_IFREMER_S1_VERSION_L1B
            )
            for subswath_sar in ["iw1", "iw2", "iw3"]:
                app_logger.info("subswath SAR : %s", subswath_sar)
                sar_basename_part = (
                    os.path.basename(iw_slc_safe).replace(".SAFE", "")
                    + "-"
                    + subswath_sar
                )
                cpt["total_suswath_sar_tested"] += 1
                fpath_out = os.path.join(
                    outputdir,
                    mode,
                    "%s" % year,
                    "%s" % month,
                    "%s" % day,
                    # os.path.basename(metacolocpath).replace(
                    #     "coloc_",
                    #     "seastate_coloc_%s_"
                    #     % (
                    #         os.path.basename(iw_slc_safe).replace(
                    #             ".SAFE", "-" + subswath_sar
                    #         )
                    #     ),
                    # ),
                    "seastate_coloc_%s_%s.nc" % (sar_basename_part, part_swot_basename),
                )
                fpath_out_nadir = os.path.join(
                    outputdir,
                    mode,
                    "%s" % year,
                    "%s" % month,
                    "%s" % day,
                    # os.path.basename(metacolocpath).replace(
                    #     "coloc_",
                    #     "seastate_coloc_%s_"
                    #     % (
                    #         os.path.basename(iw_slc_safe).replace(
                    #             ".SAFE", "-" + subswath_sar
                    #         )
                    #     ),
                    # ),
                    "seastate_nadirlike_coloc_%s_%s.nc"
                    % (sar_basename_part, part_swot_basename),
                )
                if os.path.exists(fpath_out) and overwrite is False:
                    cpt["output_file_already_existing"] += 1
                    app_logger.info("coloc file already exists: %s", fpath_out)
                else:
                    pattern_sar = os.path.join(
                        path_l2wav_sar_safe, "l2*" + subswath_sar + "*.nc"
                    )
                    lst_nc_sar = glob.glob(pattern_sar)
                    if len(lst_nc_sar) > 1:
                        app_logger.warning(
                            "\n warning : nb of SAR matching pattern %s is : %i"
                            % (pattern_sar, len(lst_nc_sar))
                        )
                    if len(lst_nc_sar) > 0:
                        fsar = lst_nc_sar[0]
                        dssar = xr.open_dataset(fsar, group=groupsar).load()
                        polygon_sar_swubswath = wkt.loads(dssar.attrs["footprint"])
                        delta_bound = 1.9  # deg
                        ymin = dssar["latitude"].values.ravel().min() - delta_bound
                        ymax = dssar["latitude"].values.ravel().max() + delta_bound
                        xmin = dssar["longitude"].values.ravel().min() - delta_bound
                        xmax = dssar["longitude"].values.ravel().max() + delta_bound
                        # subset SWOT dataset
                        subswot = dsswotl2.where(
                            (dsswotl2["latitude"] >= ymin)
                            & (dsswotl2["latitude"] <= ymax)
                            & (dsswotl2["longitude"] >= xmin)
                            & (dsswotl2["longitude"] <= xmax),
                            drop=True,
                        )
                        subswot = compute_gradient_swot_swh(dsswotl2=subswot)
                        points = np.column_stack(
                            (
                                subswot["longitude"].values.ravel(),
                                subswot["latitude"].values.ravel(),
                            )
                        )
                        # Create a MultiPoint object
                        multi_point = MultiPoint(points)

                        # Get the convex hull (smallest polygon enclosing all points)
                        # polygon = multi_point.convex_hull
                        gdfswot = gpd.GeoDataFrame(geometry=list(multi_point.geoms))
                        tolerance_simplification = 0.1
                        alpha_shape_swot = alphashape.alphashape(
                            gdfswot, alpha=tolerance_simplification
                        )
                        if alpha_shape_swot.intersects(polygon_sar_swubswath):
                            l2c_ds, ds_l2c_nadir, cpt = loop_on_each_sar_tiles(
                                dssar,
                                dsswotl2=subswot,
                                radius_coloc=conf["RADIUS_COLOC"],
                                full_path_swot=pathswotl2final,
                                cpt=cpt,
                            )

                            cpt = save_sea_state_coloc_file(
                                colocds=l2c_ds, fpath_out=fpath_out, cpt=cpt
                            )
                            app_logger.info("redo association for close nadir data")
                            cpt = save_sea_state_coloc_file(
                                colocds=ds_l2c_nadir, fpath_out=fpath_out_nadir, cpt=cpt
                            )
                            new_files.append(fpath_out)
                        else:
                            app_logger.info(
                                "subswath %s SAR not intersecting this SWOT swath",
                                subswath_sar,
                            )
                            cpt["subswath_not_intersecting_swot"] += 1
                    else:
                        app_logger.info(
                            "SAR IW Level-2 WAV product %s is not available : %s",
                            pattern_sar,
                        )
                        cpt["SAR_IW_L2WAV_not_available"] += 1
        else:
            app_logger.debug("SWOT Level-2 is not available.")
            cpt["SWOT_L2_not_available"] += 1
            app_logger.debug("dev SWOT missing", pathswotl2final)
    for kk in cpt.keys():
        app_logger.debug("cpt[%s] = %s", kk, cpt[kk])
    # app_logger.info("counter : %s", cpt)
    if len(new_files) > 0:
        app_logger.debug("example of new coloc files created : %s", new_files[0])
    return cpt, new_files


def parse_args():
    parser = argparse.ArgumentParser(description="S1SWOTswhcoloc")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        required=False,
        help="overwrite existing coloc file [default: False]",
    )
    parser.add_argument(
        "--metacolocfile", required=True, help="full path of meta coloc file"
    )
    parser.add_argument("--confpath", required=True, help="full path of config file")
    parser.add_argument(
        "--outputdir",
        required=True,
        help="directory where to store output netCDF files, path will be completed by mypath/IW/YYYY/MM/DD/filename.nc",
    )
    parser.add_argument(
        "--groupsar",
        required=False,
        choices=["intraburst", "interburst"],
        default="intraburst",
        help="intraburst or interburst [default=intraburst]",
    )
    args = parser.parse_args()
    return args


def main():
    """

    treat a meta coloc file SWOT-s1 data to generate a sea state coloc file

    :return:
    """
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    date_format = "%d-%m-%Y %H:%M:%S"
    nouveau_formatter = logging.Formatter(log_format, datefmt=date_format)
    # console_handler_app = logging.StreamHandler(sys.stdout)
    console_handler_app.setFormatter(nouveau_formatter)

    # It's good practice to remove existing handlers, especially if main() might be called multiple times
    # Iterate over a slice [:] to avoid issues when modifying the list during iteration
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)

    app_logger.addHandler(console_handler_app)
    app_logger.setLevel(log_level)
    app_logger.propagate = False  # <--- THIS IS THE KEY CHANGE

    associate_sar_and_swot_seastate_params(
        metacolocpath=args.metacolocfile,
        confpath=args.confpath,
        groupsar=args.groupsar,
        overwrite=args.overwrite,
        outputdir=args.outputdir,
    )


if __name__ == "__main__":
    main()
