import logging
import pyproj
from yaml import CLoader as Loader
from yaml import load
from shapely.ops import transform

logger = logging.getLogger("s1swotcolocs.get_config_info")
logger.addHandler(logging.NullHandler())


def get_conf_content(conf_path):
    stream = open(conf_path, "r")
    conf = load(stream, Loader=Loader)
    return conf

def calculate_overlap_percentage(sar_poly, swot_poly):
    """ Accurate area overlap using local UTM projection. """
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

def get_swot_date_info(SWOT_start_piece):
    """
    Arguments:
        SWOT_start_piece (np.datetime64):
    Returns:
        swot_formated_date (str), year (int), month (str MM), day (str DD)
    """
    import numpy as np
    dt_py = SWOT_start_piece.astype("M8[D]").astype(object)
    year = dt_py.year
    month = f"{dt_py.month:02d}"
    day = f"{dt_py.day:02d}"
    swot_formated_date = (
        ("%s" % SWOT_start_piece).replace("-", "").replace(":", "").split(".")[0]
    )
    return swot_formated_date, year, month, day
