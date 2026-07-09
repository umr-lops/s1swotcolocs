"""
Robustified version of coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel.py
May 2025 - A Grouazel

Changes vs original:
  - treat_one_day_wrapper call wrapped with specific ValueError catch
    for the antimeridian/shapely "linearring requires at least 4 coordinates"
    crash, plus a general Exception catch for any other library-level error
  - Per-day counters are accumulated into a global summary dict
  - Global summary is printed at the end of the run
  - A dedicated counter tracks how many days were skipped per error type
"""

import argparse
import datetime
import logging
import os
import sys
from collections import defaultdict

from dateutil import rrule

from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import treat_one_day_wrapper

# ---------------------------------------------------------------------------
# Global counter keys
# ---------------------------------------------------------------------------

CTR_DAYS_OK = "days_processed_ok"
CTR_DAYS_LINEARRING = "days_skipped_linearring_error"
CTR_DAYS_ERROR = "days_skipped_other_error"
CTR_TOTAL_MATCHUPS = "total_matchups_found"  # sum of cpt values across days


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    return logging.getLogger(__name__)


def silence_verbose_libs():
    """Silence libraries that are too verbose at root level."""
    for logger_name in ["cdsodatacli", "cdsodatacli.query"]:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers.clear()
        lib_logger.addHandler(logging.NullHandler())
        lib_logger.propagate = False
        lib_logger.setLevel(logging.CRITICAL + 1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_yyyymmdd(s):
    try:
        return datetime.datetime.strptime(s, "%Y%m%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{s}'. Expected format is YYYYMMDD."
        )


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def _print_global_summary(global_counters: defaultdict, logger) -> None:
    """Print a formatted summary of all accumulated counters."""
    width = 40
    logger.info("=" * (width + 10))
    logger.info("  GLOBAL RUN SUMMARY")
    logger.info("=" * (width + 10))
    labels = {
        CTR_DAYS_OK: "Days processed successfully",
        CTR_DAYS_LINEARRING: "Days skipped (linearring error)",
        CTR_DAYS_ERROR: "Days skipped (other error)",
        CTR_TOTAL_MATCHUPS: "Total matchups found across all days",
    }
    for key, label in labels.items():
        logger.info("  %-*s %d", width, label, global_counters[key])
    logger.info("=" * (width + 10))


# ---------------------------------------------------------------------------
# Safe wrapper around treat_one_day_wrapper
# ---------------------------------------------------------------------------


def safe_treat_one_day(
    day_str: str,
    outd: str,
    mode: str,
    disable_tqdm: bool,
    confpath: str,
    global_counters: defaultdict,
    logger,
    dev: bool = False,
) -> dict:
    """
    Call treat_one_day_wrapper with targeted exception handling.

    The antimeridian library raises:
        ValueError: A linearring requires at least 4 coordinates.
    from deep inside fix_polygon → fix_polygon_to_list → Polygon().

    This is a known data-quality issue on certain SWOT orbit geometries
    near the poles or antimeridian. We catch it specifically, log a warning
    with the date and mode, increment a counter, and continue the loop
    instead of crashing the entire run.

    Any other exception is also caught, logged at ERROR level with the full
    traceback, and counted separately so it does not silently mask bugs.

    Returns the cpt dict from treat_one_day_wrapper, or an empty dict on error.

    Args:
        day_str: Date string in YYYYMMDD format for the day to process.
        outd: Output directory for the day's results.
        mode: Collocation mode ('IW' or 'EW').
        disable_tqdm: Whether to disable tqdm progress bars.
        confpath: Path to the config.yml file.
        global_counters: A defaultdict to accumulate global counts across days.
        logger: Logger instance for logging messages.
        dev: Whether to run in development mode (not used here but passed through).

    Returns:
        dict: The cpt dictionary returned by treat_one_day_wrapper, or empty dict on error

    Raises:
        None. All exceptions are caught and logged, and the function returns an empty dict on error.
    """
    try:
        cpt = treat_one_day_wrapper(
            day2treat=day_str,
            outputdir=outd,
            mode=mode,
            disable_tqdm=disable_tqdm,
            confpath=confpath,
            dev=dev,
        )
        global_counters[CTR_DAYS_OK] += 1

        # Accumulate per-day matchup counts into global total
        if isinstance(cpt, dict):
            for v in cpt.values():
                if isinstance(v, (int, float)):
                    global_counters[CTR_TOTAL_MATCHUPS] += int(v)

        return cpt

    except ValueError as exc:
        exc_str = str(exc).lower()
        if "linearring" in exc_str or "at least 4 coordinates" in exc_str:
            global_counters[CTR_DAYS_LINEARRING] += 1
            logger.warning(
                "Linearring error on %s mode=%s — skipping day. "
                "This is a known antimeridian/shapely issue on degenerate "
                "SWOT polygons near poles. Detail: %s",
                day_str,
                mode,
                exc,
            )
        else:
            global_counters[CTR_DAYS_ERROR] += 1
            logger.error(
                "ValueError (non-linearring) on %s mode=%s — skipping day.",
                day_str,
                mode,
                exc_info=True,
            )
        return {}

    except Exception:
        global_counters[CTR_DAYS_ERROR] += 1
        logger.error(
            "Unexpected error on %s mode=%s — skipping day.",
            day_str,
            mode,
            exc_info=True,
        )
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sequential colocation of SWOT L3 with Sentinel-1 (replaces prun)"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--startdate",
        help="YYYYMMDD — SWOT L3 Ifremer collection starts 20230328",
        required=True,
        type=parse_yyyymmdd,
    )
    parser.add_argument(
        "--stopdate",
        help="YYYYMMDD stop (inclusive)",
        required=True,
        type=parse_yyyymmdd,
    )
    parser.add_argument(
        "--outputdir",
        help="Path where the metadata coloc files (.nc) will be saved.",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--confpath",
        help="Path of the config.yml to use",
        required=True,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Development mode: use a smaller subset of SWOT files for faster iterations",
    )
    parser.add_argument(
        "--mode",
        help="Collocation mode: 'IW' or 'EW'",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    silence_verbose_libs()

    global_counters: defaultdict = defaultdict(int)

    logger.info(
        "Start loops — startdate=%s stopdate=%s",
        args.startdate.strftime("%Y%m%d"),
        args.stopdate.strftime("%Y%m%d"),
    )
    logger.info("Output directory: %s", args.outputdir)
    logger.info("Config path: %s", args.confpath)
    logger.info("Development mode: %s", args.dev)
    if args.mode:
        modes_to_process = [args.mode]
    else:
        modes_to_process = ["IW", "EW"]
    logger.info("Collocation modes to process: %s", ", ".join(modes_to_process))
    for mode in modes_to_process:
        logger.info("Processing mode: %s", mode)

        for dd in rrule.rrule(rrule.DAILY, dtstart=args.startdate, until=args.stopdate):
            day_str = dd.strftime("%Y%m%d")
            outd = os.path.join(args.outputdir, mode)

            cpt = safe_treat_one_day(
                day_str=day_str,
                outd=outd,
                mode=mode,
                disable_tqdm=not args.verbose,
                confpath=args.confpath,
                global_counters=global_counters,
                logger=logger,
                dev=args.dev,
            )

            # Per-day detail log (same as original)
            if cpt:
                logger.info("Day: %s mode: %s counters:", day_str, mode)
                for key, val in cpt.items():
                    logger.info("\t %s: %s", key, val)
            else:
                logger.info(
                    "Day: %s mode: %s — no output (skipped or empty)", day_str, mode
                )

    _print_global_summary(global_counters, logger)
    logger.info("Done — %s", os.path.basename(__file__))


if __name__ == "__main__":
    main()
