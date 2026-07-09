#!/usr/bin/env python3

import argparse
import logging
import os  # <--- NEW: Import the os module
import subprocess
import sys
from datetime import date, datetime, timedelta

from yaml import CLoader as Loader
from yaml import load


def setup_logging():
    """Configures the logging format for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def is_valid_date_format(date_string):
    """Checks if a string is in the YYYYMMDD format."""
    try:
        datetime.strptime(date_string, "%Y%m%d")
        return True
    except ValueError:
        return False


def calculate_default_date_range():
    """Calculates the default start (D-15) and stop (D) dates."""
    logging.info("Using default date range (D-15 to D).")
    today = date.today()
    start_day = today - timedelta(days=15)
    stop_date_str = today.strftime("%Y%m%d")
    start_date_str = start_day.strftime("%Y%m%d")
    logging.info(f"Start Date (D-15): {start_date_str}")
    logging.info(f"Stop Date  (D)  : {stop_date_str}")
    return start_date_str, stop_date_str


def run_command(command_list, step_description):
    """Executes a quick command (as a list) and waits for it to complete."""
    logging.info(f"--- Starting Step: {step_description} ---")
    try:
        process = subprocess.run(
            command_list, check=True, text=True, capture_output=True
        )
        if process.stdout:
            logging.debug(f"Command stdout:\n{process.stdout.strip()}")
        logging.info(f"✅ SUCCESS: {step_description} completed.")
    except subprocess.CalledProcessError as e:
        logging.error(
            f"❌ FAILED: {step_description} failed with exit code {e.returncode}."
        )
        if e.stdout:
            logging.error(f"--- STDOUT ---\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"--- STDERR ---\n{e.stderr.strip()}")
        sys.exit(1)


def stream_command(command_list, step_description):
    """Executes a long-running command (as a list) and streams its output."""
    logging.info(f"--- Starting Step: {step_description} ---")
    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with process.stdout:
        for line in iter(process.stdout.readline, ""):
            # Your filter to remove tqdm noise
            if "1/1" not in line and "0/1" not in line and line.strip():
                logging.info(f"[CONTAINER] {line.strip()}")
    return_code = process.wait()
    if return_code == 0:
        logging.info(f"✅ SUCCESS: {step_description} completed.")
    else:
        logging.error(
            f"❌ FAILED: The container process failed with exit code {return_code}."
        )
        sys.exit(1)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Wrapper script to run S1/SWOT  SWH collocation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--startdate",
        type=str,
        default=None,
        required=False,
        help="The start date in YYYYMMDD format [optional].",
    )
    parser.add_argument(
        "--stopdate",
        type=str,
        default=None,
        required=False,
        help="The stop date in YYYYMMDD format [optional].",
    )
    parser.add_argument(
        "--confpath", help="path of the config.yml you want to use", required=True
    )
    args = parser.parse_args()

    logging.info("Starting the S1/SWOT collocation wrapper script.")
    logging.info("load the config file from :%s", args.confpath)
    stream = open(args.confpath)
    conf = load(stream, Loader=Loader)
    # --- Configuration ---
    DOCKER_BINARY_PATH = conf["DOCKER_BINARY_PATH"]
    DOCKER_IMAGE = conf["DOCKER_IMAGE"]
    HOST_DATAWORK = conf["HOST_DATAWORK"]
    HOST_SOURCES_DIR = conf["HOST_SOURCES_DIR"]
    HOST_SOURCES_DATA = conf["HOST_SOURCES_DATA"]
    HOST_OUTPUT_DIR = conf["HOST_SEASTATE_COLOC_OUTPUT_DIR"]
    CONTAINER_SCRIPT_PATH = "coloc_seastate_SWOT_S1_sequential"
    SWOT_L2_AVISO_DIR = conf["SWOT_L2_AVISO_DIR"]
    # --- End of Configuration ---
    if args.startdate and args.stopdate:
        if not is_valid_date_format(args.startdate) or not is_valid_date_format(
            args.stopdate
        ):
            logging.error("Invalid date format. Please use YYYYMMDD.")
            sys.exit(1)
        start_date, stop_date = args.startdate, args.stopdate
        logging.info(f"Using user-provided date range: {start_date} to {stop_date}")
    elif not args.startdate and not args.stopdate:
        start_date, stop_date = calculate_default_date_range()
    else:
        logging.error(
            "Invalid arguments. You must provide both --startdate and --stopdate, or neither."
        )
        sys.exit(1)

    # <--- NEW: Get the current user's UID and GID to run the container with
    user_spec = f"{os.getuid()}:{os.getgid()}"
    logging.info(f"Will run container as user spec (UID:GID): {user_spec}")

    # --- Step 1: Pull Docker Image ---
    pull_command = [DOCKER_BINARY_PATH, "pull", DOCKER_IMAGE]
    run_command(pull_command, f"Pull Docker image '{DOCKER_IMAGE}'")

    # --- Step 2: Run Collocation Script in Container ---
    docker_run_command = [
        DOCKER_BINARY_PATH,
        "run",
        "--rm",
        "--user",
        user_spec,  # <--- NEW: Add the --user flag
        "-e",
        "HOME=/tmp",
        "-v",
        f"{HOST_SOURCES_DIR}:{HOST_SOURCES_DIR}",
        "-v",
        f"{HOST_DATAWORK}:{HOST_DATAWORK}",
        "-v",
        f"{HOST_SOURCES_DATA}:{HOST_SOURCES_DATA}",
        "-v",
        f"{HOST_OUTPUT_DIR}:{HOST_OUTPUT_DIR}",
        "-v",
        f"{SWOT_L2_AVISO_DIR}:{SWOT_L2_AVISO_DIR}",
        DOCKER_IMAGE,
        # "python", "-u", CONTAINER_SCRIPT_PATH,
        CONTAINER_SCRIPT_PATH,
        "--startdate",
        start_date,
        "--stopdate",
        stop_date,
        "--outputdir",
        HOST_OUTPUT_DIR,
        "--confpath",
        args.confpath,
        "--groupsar",
        "intraburst",
        "--overwrite",
    ]
    stream_command(docker_run_command, "Execute collocation script in a new container")

    logging.info("🎉 All steps completed successfully. Script finished.")
    logging.info("Recall output directory: %s", HOST_OUTPUT_DIR)


if __name__ == "__main__":
    main()
