import argparse
import collections
import datetime
import glob
import logging
import os
import sys

from dateutil import rrule
from tqdm import tqdm

from s1swotcolocs.seastate_colocs_s1_swot import associate_sar_and_swot_seastate_params
from s1swotcolocs.utils import get_conf_content

app_logger = logging.getLogger(__file__)


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
        "--startdate",
        required=True,
        help="starting date YYYYMMDD for searching for meta colocs files and producing SWH colocs",
    )
    parser.add_argument(
        "--stopdate",
        required=True,
        help="stop date YYYYMMDD for searching for meta colocs files and producing SWH colocs",
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
    console_handler_app = logging.StreamHandler(sys.stdout)
    console_handler_app.setFormatter(nouveau_formatter)

    # It's good practice to remove existing handlers, especially if main() might be called multiple times
    # Iterate over a slice [:] to avoid issues when modifying the list during iteration
    # for handler in app_logger.handlers[:]:
    #    app_logger.removeHandler(handler)
    for handler in app_logger.handlers[:]:
        # handler.setFormatter(nouveau_formatter)
        app_logger.removeHandler(handler)
    app_logger.addHandler(console_handler_app)
    # app_logger.addHandler(lowerhandler)
    app_logger.setLevel(log_level)
    app_logger.propagate = False  # <--- THIS IS THE KEY CHANGE
    conf = get_conf_content(args.confpath)
    metadir = conf["HOST_META_COLOC_OUTPUT_DIR"]
    app_logger.info("dir to search for meta colocs : %s", metadir)
    sta = datetime.datetime.strptime(args.startdate, "%Y%m%d")
    sto = datetime.datetime.strptime(args.stopdate, "%Y%m%d")
    lst_files = []
    for dd in rrule.rrule(rrule.DAILY, dtstart=sta, until=sto):
        pat = os.path.join(
            metadir,
            "IW",
            dd.strftime("%Y"),
            dd.strftime("%m"),
            dd.strftime("%d"),
            "coloc_SWOT*.nc",
        )
        lst_files += glob.glob(pat)
    lst_files = sorted(lst_files)
    app_logger.info("nb files meta data colocs found; %i", len(lst_files))
    app_logger.info("outputdir : %s", args.outputdir)
    bigcpt = collections.defaultdict(int)
    for uu in tqdm(range(len(lst_files)), desc="overall progress meta-coloc"):
        ffmeta = lst_files[uu]
        app_logger.debug("ffmeta : %s", ffmeta)
        cpt, new_files = associate_sar_and_swot_seastate_params(
            metacolocpath=ffmeta,
            confpath=args.confpath,
            groupsar=args.groupsar,
            overwrite=args.overwrite,
            outputdir=args.outputdir,
        )
        # app_logger.info('%i counter : %s',uu,cpt)

        # bigcpt.update(cpt)
        for key in cpt:
            bigcpt[key] += cpt[key]

    # app_logger.info("done : %s", bigcpt)
    for kk in bigcpt.keys():
        app_logger.info("bigcpt[%s] = %s", kk, bigcpt[kk])
    if len(new_files) > 0:
        app_logger.info("example of new coloc files created : %s", new_files[0])


if __name__ == "__main__":
    main()
