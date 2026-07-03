#!/usr/bin/env python3
"""
Script to process one SAR SAFE file and find SWOT colocalizations.
"""
import argparse
import logging

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import treat_one_safe

DEFAULT_CONFPATH = "./config.yml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find SWOT colocalizations for a given SAR SAFE file."
    )
    parser.add_argument(
        "safe",
        type=str,
        help="Path to the SAR SAFE file to process.",
    )
    parser.add_argument(
        "--confpath",
        type=str,
        default=DEFAULT_CONFPATH,
        help=f"Path to the config file (default: {DEFAULT_CONFPATH}).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Development mode: restrict to first 2 SWOT files for faster iterations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("SAFE      : %s", args.safe)
    logger.info("Config    : %s", args.confpath)
    logger.info("Dev mode  : %s", args.dev)

    treat_one_safe(
        safe=args.safe,
        confpath=args.confpath,
        dev=args.dev,
    )


if __name__ == "__main__":
    main()