import argparse
import glob
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pyproj
import pystac
import shapely.wkt
import xarray as xr
from pystac_client import Client

# Internal Library Imports
from s1ifr.get_path_from_base_safe import get_path_from_base_safe
from shapely.geometry import mapping
from shapely.ops import transform
from tqdm import tqdm

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import get_swot_date_info
from s1swotcolocs.utils import get_conf_content

# Official NASA SWOT STAC Configuration
NASA_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/POCLOUD"
SWOT_L2_COLLECTION = "SWOT_L2_LR_SSH_WINDWAVE_D_D"

CDSE_STAC_BASE = "https://stac.dataspace.copernicus.eu/v1/collections"
DEFAULT_EXTENSION_SCHEMA = "/home1/datahome/agrouaze/sources/git/app_metacoloc_visu/docs/examples/stac_sar_matchup_schema.json"


def get_memory_usage():
    try:
        import resource

        memory_used_go = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 / 1000.0
        )
    except ImportError:  # on windows resource is not usable
        import psutil

        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.0
    str_mem = f"RAM usage: {memory_used_go:1.1f} Go"
    return str_mem


def get_nasa_stac_info(target_dt, sar_poly, cycle, pass_id):
    """
    Queries NASA POCLOUD STAC to find the exact Granule ID and Link.
    Matches by spatial intersection and Cycle/Pass ID prefix.
    """
    try:
        client = Client.open(NASA_STAC_URL)

        # Search window +/- 15 mins around SAR acquisition
        search = client.search(
            collections=[SWOT_L2_COLLECTION],
            intersects=mapping(sar_poly),
            datetime=[
                target_dt - timedelta(minutes=15),
                target_dt + timedelta(minutes=15),
            ],
            max_items=10,
        )

        # Expected ID prefix based on NASA/AVISO convention
        expected_prefix = f"SWOT_L2_LR_SSH_WindWave_{int(cycle):03d}_{int(pass_id):03d}"

        # Fixed: use items() instead of deprecated get_items()
        items = list(search.items())
        for item in items:
            if item.id.startswith(expected_prefix):
                logging.debug(f"  [STAC] NASA Match Found: {item.id}")
                return item.id, item.get_self_href()

    except Exception as e:
        logging.warning(f"NASA STAC query failed: {e}")
    return None, None


def calculate_overlap_percentage(sar_poly, swot_poly):
    """Accurate area overlap using local UTM projection."""
    try:
        if not sar_poly.intersects(swot_poly):
            return 0.0
        lon, lat = sar_poly.centroid.x, sar_poly.centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        project = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        ).transform
        s_prj = transform(project, sar_poly)
        w_prj = transform(project, swot_poly)
        inter = s_prj.intersection(w_prj)
        return round((inter.area / s_prj.area) * 100, 2)
    except Exception:
        return None


def get_s1_full_path(safe_name):
    """Uses s1ifr dependency to find full path on Ifremer archive."""
    safe = safe_name.replace(".zip", "").replace(".SAFE", "")
    for archive in ["scale", "datawork"]:
        fp = get_path_from_base_safe(
            safe_basename=safe, archive_name=archive, check_existence=True
        )
        if fp:
            return fp
    return None


def process_nc_to_stac(nc_path, output_dir, config, extension_url, counters, overwrite):
    all_output_files = []
    swot_l2_dir = config.get("SWOT_L2_AVISO_DIR")

    try:
        ds = xr.open_dataset(nc_path)
        lib_v = str(ds.attrs.get("s1swotcolocs_python_lib_version", "unknown"))
        window = float(ds.attrs.get("searching_windows_width_in_hours", 1.0))

        # SWOT Polygon (Global swath piece from NetCDF)
        swot_wkt = (
            ds.swot_polygon.values.decode()
            if isinstance(ds.swot_polygon.values, bytes)
            else str(ds.swot_polygon.values)
        )
        swot_poly = shapely.wkt.loads(swot_wkt)

        num_matches = len(ds.sar_start_time_slice)
        logging.debug(f"File: {os.path.basename(nc_path)} | Matches: {num_matches}")

        for i in range(num_matches):

            def get_val(var, idx):
                v = var.values[idx]
                if isinstance(v, bytes):
                    return v.decode()
                return v.item() if hasattr(v, "item") else v

            swot_formated_date, year, month, day = get_swot_date_info(
                SWOT_start_piece=ds.SWOT_start_time_slice.values
            )
            sar_safe = str(get_val(ds.sar_safe_name, i))
            item_id = f"matchup_{sar_safe.replace('.SAFE', '')}_SWOT_KaRin_{swot_formated_date}"
            # out_path = os.path.join(output_dir, f"{item_id}.json")
            out_path = os.path.join(
                output_dir, f"{year}", f"{month}", f"{day}", f"{item_id}.json"
            )
            if os.path.exists(out_path) and overwrite is False:
                counters["matchups_already_available_in_STAC"] += 1
            else:
                logging.debug("file STAC creation.")
                l3_path = str(get_val(ds.filepath_swot, i))
                sar_poly_wkt = str(get_val(ds.sar_polygon, i))
                sar_poly = shapely.wkt.loads(sar_poly_wkt)

                # Time conversion
                dt = datetime.fromtimestamp(
                    ds.sar_start_time_slice.values[i].astype("O") / 1e9, tz=timezone.utc
                )

                # Extract Cycle/Pass from L3 name
                parts_l3 = os.path.basename(l3_path).split("_")
                cycle, pass_id = parts_l3[5], parts_l3[6]

                # Resolve S1 Local Path
                s1_full_path = get_s1_full_path(sar_safe)
                if s1_full_path is not None:
                    counters["matchups_with_fullpath_s1_ifremer"] += 1
                else:
                    counters["matchups_absent_fullpath_s1_ifremer"] += 1
                # Resolve SWOT L2 Local Path
                l2_local_path = None
                pattern = os.path.join(swot_l2_dir, f"*{cycle}_{pass_id}*.nc")
                matches = glob.glob(pattern)
                if matches:
                    l2_local_path = matches[0]
                    counters["matchups_with_fullpath_swot_l2"] += 1
                else:
                    counters["matchups_absent_fullpath_swot_l2"] += 1

                # Query NASA for official STAC Granule Info
                official_id, official_url = get_nasa_stac_info(
                    dt, sar_poly, cycle, pass_id
                )
                if official_id is not None:
                    counters["matchups_with_STAC_NASA"] += 1
                else:
                    counters["matchups_absent_STAC_NASA"] += 1

                # Fallback
                final_id = (
                    official_id
                    if official_id
                    else os.path.basename(
                        l2_local_path if l2_local_path else l3_path
                    ).replace(".nc", "")
                )
                final_url = (
                    official_url
                    if official_url
                    else f"{NASA_STAC_URL}/collections/{SWOT_L2_COLLECTION}/items/{final_id}"
                )

                # Calculations
                overlap = calculate_overlap_percentage(sar_poly, swot_poly)
                s1_parts = sar_safe.split("_")
                ptype = (
                    "GRD"
                    if "GRD" in s1_parts[2]
                    else ("SLC" if "SLC" in s1_parts[2] else "OCN")
                )

                # Build STAC Item

                item = pystac.Item(
                    id=item_id,
                    geometry=mapping(sar_poly),
                    bbox=list(sar_poly.bounds),
                    datetime=dt,
                    properties={},
                )

                item.properties.update(
                    {
                        "sarmatchup:sar_platform": s1_parts[0]
                        .lower()
                        .replace("s1", "sentinel-1"),
                        "sarmatchup:sar_instrument_mode": s1_parts[1],
                        "sarmatchup:sar_product_type": ptype,
                        "sarmatchup:sar_safe": sar_safe,
                        "sarmatchup:sar_local_path": s1_full_path,
                        "sarmatchup:other_instrument": "swot KaRin",
                        "sarmatchup:other_type": "swath_altimeter",
                        "sarmatchup:other_id": final_id,
                        "sarmatchup:other_local_path": (
                            l2_local_path if l2_local_path else l3_path
                        ),
                        "sarmatchup:other_polygon": swot_wkt,
                        "sarmatchup:delta_diff_time_minutes": float(
                            ds.delta_diff_time.values[i]
                        ),
                        "sarmatchup:searching_window_hours": window,
                        "sarmatchup:overlap_percentage": overlap,
                        "sarmatchup:lib_version": lib_v,
                        "sarmatchup:sar_cdse_item_url": f"{CDSE_STAC_BASE}/sentinel-1-{ptype.lower()}/items/{sar_safe.replace('.SAFE','')}",
                        "sarmatchup:other_catalog_url": final_url,
                    }
                )

                item.stac_extensions.append(extension_url)
                item.add_asset(
                    "source_nc",
                    pystac.Asset(
                        href=os.path.abspath(nc_path),
                        media_type="application/x-netcdf",
                        roles=["metadata"],
                    ),
                )

                item.save_object(dest_href=out_path)
                logging.debug(f"output STAC file: {out_path}")
                all_output_files.append(out_path)

        ds.close()
    except Exception:
        logging.exception(f"Error processing {nc_path}")
    return all_output_files, counters


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-nc", type=str, help="individual full path of SWOT KaRin matchups .nc"
    )
    group.add_argument(
        "--input-list", type=str, help="listing of path of SWOT KaRin matchups .nc"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to localconfig.yml"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        required=False,
        help="where to store STAC files [optional, default is .]",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        default=False,
        help="True -> overwrite existing STAC .json files, False-> keep existing files. [default=False]",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    t0 = datetime.today()
    config = get_conf_content(args.config)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    files = [args.input_nc] if args.input_nc else []
    if args.input_list:
        with open(args.input_list) as f:
            files = [lline.strip() for lline in f if lline.strip()]
    final_listing_stac_generated = []
    counters = defaultdict(int)
    for f in tqdm(files, desc="Converting", disable=args.verbose):
        counters["nc_matchup_input_file_treated"] += 1
        tmp_list_outstac, counters = process_nc_to_stac(
            f,
            args.out_dir,
            config,
            DEFAULT_EXTENSION_SCHEMA,
            counters=counters,
            overwrite=args.overwrite,
        )
        final_listing_stac_generated += tmp_list_outstac
    if len(final_listing_stac_generated) > 0:
        logging.info(
            "example of output file STAC generated : %s",
            final_listing_stac_generated[-1],
        )
    logging.info("counters: %s", counters)
    logging.info(
        "Number of STAC files generated: %s", len(final_listing_stac_generated)
    )
    logging.info("elapsed time: %s", (datetime.today() - t0))
    logging.info("memory: %s", get_memory_usage())


if __name__ == "__main__":
    main()
