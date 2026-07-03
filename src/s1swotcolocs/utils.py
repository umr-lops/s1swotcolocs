import logging
from yaml import CLoader as Loader
from yaml import load
from functools import partial
from pathlib import Path
import re
import h5py

logger = logging.getLogger("s1swotcolocs.get_config_info")
logger.addHandler(logging.NullHandler())


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


def get_conf_content(conf_path):
    # stream = open(get_config_file_path(), "r")
    stream = open(conf_path, "r")
    conf = load(stream, Loader=Loader)
    return conf


SAFE_PATTERN_S1 = (
    r"^(?P<mission_id>\w{3})_"
    + r"(?P<mode>\w{2})_"
    + r"(?P<type>\w{3})(?P<res>\w|_)_"
    + r"(?P<level>\w{1})(?P<class>\w{1})(?P<pol>\w{2})_"
    + r"(?P<startdate>\w{8})T"
    + r"(?P<starttime>\w{6})_"
    + r"(?P<enddate>\w{8})T"
    + r"(?P<endtime>\w{6})_"
    + r"(?P<orbit_no>\w{6})_"
    + r"(?P<datatake_id>\w{6})_"
    + r"(?P<id>\w{4})"
)

SAFE_PATTERN_RCM = (
    r"^(?P<mission_id>\w{4})_"
    + r"(?P<order_id>\w{8,9})_"
    + r"(?P<product_id>\w{9})_"
    + r"(?P<level>\w{1})_"
    + r"(?P<mode>\w[^_]+)_"
    + r"(?P<startdate>\w{8})_"
    + r"(?P<starttime>\w{6})_"
    + r"(?P<pol>\w{2,5})_"
    + r"(?P<product_type>\w{3})"
)

SAFE_PATTERN_RS2 = (
    r"^(?P<mission_id>\w{3})_"
    + r"(?P<order_id>\w[^_]+)_"
    + r"(?P<product_id>\w[^_]+)_"
    + r"(?P<deliver_id>\w[^_]+)_"
    + r"(?P<mode>\w{4})_"
    + r"(?P<startdate>\w{8})_"
    + r"(?P<starttime>\w{6})_"
    + r"(?P<pol>\w{2,5})_"
    + r"(?P<processing_level>\w{3})"
)

VERS_SAFE_PATTERN_S1 = SAFE_PATTERN_S1 + r"_(?P<version>\w{3})"

def parse(name, patterns):
    name = Path(name).name
    m = None
    for pattern in patterns:
        m = re.match(pattern, name)
        if m is not None:
            return m.groupdict()
    if m is None:
        return None


def map_mission_mode(safe_info):
    mission = safe_info["mission_id"]
    mode = safe_info["mode"]

    # --- S1 ---
    if mission.startswith("S1"):
        if mode in ["IW", "EW"]:
            return ('S1', mode)
        else:
            return ("S1", "Unknown")

    # --- RCM ---
    elif mission.startswith("RCM"):
        if mode.startswith("SCLN"):
            return ("RCM", "SCLN")
        elif mode.startswith("SC"):
            return ("RCM", "SCM")
        else:
            return ("RCM", "Unknown")

    # --- RS2 ---
    elif mission == "RS2":
        if mode.startswith("SCN"):
            return ("RS2", "SCN")
        elif mode.startswith("SCW"):
            return ("RS2", "SCW")
        elif mode.startswith("XXXX"):
            return ("RS2", "SCW")
        else:
            return ("RS2", "Unknown")


parse_safe_name = partial(parse, patterns=(VERS_SAFE_PATTERN_S1, SAFE_PATTERN_S1, SAFE_PATTERN_RCM, SAFE_PATTERN_RS2))


def normalize_mission(mission):
    if mission.startswith("S1"):
        return "S1"
    elif mission.startswith("RCM"):
        return "RCM"
    elif mission.startswith("RS2"):
        return "RS2"
    else:
        raise ValueError(f"Unkown mission : {mission}")


def get_netcdf_attribute(fpath, attr_name):
    with h5py.File(fpath, "r") as f:
        return f.attrs.get(attr_name, None).decode("utf-8")