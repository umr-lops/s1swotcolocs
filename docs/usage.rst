=====
Usage
=====

This page describes the main command‑line entrypoints provided by the library.
All scripts accept ``--help`` for a full list of options.

--------------
Configuration
--------------

Most scripts rely on a YAML configuration file (e.g., ``config.yml`` or ``localconfig.yml``) to define input/output paths and parameters.
A typical configuration file looks like:

.. code-block:: yaml

   SWOT_L2_AVISO_DIR: "/path/to/SWOT_L2_KARIN_LR_WindWave_AVISO"
   S1_WV_ROOT: "/path/to/L2_daily/0.12"
   WV_MATCHUP_OUTPUT_DIR: "/path/to/matchups_wv"
   CACHE_CDSE: "/path/to/cache"
   DELTA_HOURS: 6
   MAX_AREA_SIZE: 200
   TOLERANCE_SIMPLIFICATION: 0.1

-------------------
Main Entrypoints
-------------------

coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel
-------------------------------------------

**Purpose**:
Create meta‑collocation files between Sentinel‑1 IW/EW products and SWOT Level‑3 data.
This is the recommended entrypoint for IW/EW collocations.

**Basic usage**:

.. code-block:: bash

   coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel \
       --startmonth 20250616 \
       --stopmonth 20250616 \
       --confpath src/s1swotcolocs/localconfig.yml \
       --outputdir /tmp/

**Options**:

- ``--startmonth`` (YYYYMMDD) – first day to process.
- ``--stopmonth`` (YYYYMMDD) – last day to process.
- ``--confpath`` – path to the YAML configuration file.
- ``--outputdir`` – root directory where NetCDF collocation files will be written.
- ``--mode`` – choose ``IW`` (default) or ``EW``.
- ``--verbose`` – increase log verbosity.

coloc_seastate_SWOT_S1
----------------------

**Purpose**:
Collocate Sentinel‑1 ocean state products (e.g., wave spectra) with SWOT KaRIn data.
This entrypoint processes data in a single‑pass mode (all available files).

**Basic usage**:

.. code-block:: bash

   coloc_seastate_SWOT_S1 --confpath config.yml --start-date 2025-01-01 --stop-date 2025-01-31

**Options**:

- ``--confpath`` – YAML configuration file.
- ``--start-date`` / ``--stop-date`` – date range (YYYY-MM-DD).
- ``--output-dir`` – override the output directory from the config.

coloc_seastate_SWOT_S1_sequential
---------------------------------

**Purpose**:
Same as above, but processes the date range sequentially (better for large datasets with limited memory).

**Basic usage**:

.. code-block:: bash

   coloc_seastate_SWOT_S1_sequential --confpath config.yml --start-date 2025-01-01 --stop-date 2025-01-31

**Options**: same as `coloc_seastate_SWOT_S1`.

download_l2_swot_karin_lr_ssh_windwave_cnes
-------------------------------------------

**Purpose**:
Download SWOT L2 KaRIn WindWave products from the CNES AVISO data centre (requires credentials).

**Basic usage**:

.. code-block:: bash

   download_l2_swot_karin_lr_ssh_windwave_cnes --config config.yml --start-date 2025-01-01 --stop-date 2025-01-10 --output /path/to/download

**Options**:

- ``--config`` – YAML file containing AVISO credentials and settings.
- ``--start-date`` / ``--stop-date`` – date range.
- ``--output`` – local directory for downloaded files.
- ``--parallel`` – number of parallel downloads (optional).

convert_matchups_nc_to_stac
---------------------------

**Purpose**:
Convert the NetCDF collocation files (produced by `coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel`) into STAC‑compliant JSON items.

**Basic usage**:

.. code-block:: bash

   convert_matchups_nc_to_stac --input /path/to/netcdf_files --output /path/to/stac_items

**Options**:

- ``--input`` – directory containing NetCDF collocation files.
- ``--output`` – directory where STAC JSON items will be written.
- ``--catalog-url`` – base URL for the STAC catalog (optional).

matchups_wv_karin
-----------------

**Purpose**:
Collocate Sentinel‑1 Wave Mode (WV) Level‑2 products with SWOT KaRIn L2 LR SSH data.
This script is memory‑efficient (only SWOT swath‑edge pixels are loaded) and does not use CDSE queries – it resolves S1‑WV files directly from a known directory tree.

**Basic usage**:

.. code-block:: bash

   matchups_wv_karin --conf-file config.yml --start-date 2026-01-01 --stop-date 2026-01-10 --output /path/to/matchups

**Options**:

- ``--conf-file`` – YAML configuration file (must contain ``SWOT_L2_AVISO_DIR``, ``S1_WV_ROOT``, and optionally ``WV_MATCHUP_OUTPUT_DIR``).
- ``--start-date`` / ``--stop-date`` – only process SWOT files within this date range (based on filename timestamps).
- ``--output`` – output directory (overrides config).
- ``--max-time-diff`` – maximum allowed time difference between S1‑WV and SWOT (in minutes, default 360).
- ``--dev`` – process only the first 2 SWOT files (for testing).
- ``--log-level`` – set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

**Expected S1‑WV file tree**:

.. code-block:: text

   <S1_WV_ROOT>/<yyyy>/<doy>/S1{A,C,D}_WV_L2D_enriched_LOPS_<yyyymmdd>_daily_IPF_*.nc

**Output**:

- STAC‑like JSON matchups under ``<output>/json_matchups/<yyyy>/<doy>/``.
- A summary CSV at ``<output>/test_json_matchups.csv``.
- Optional debug images (if enabled in the code).

-------------------
Additional Scripts
-------------------

The following scripts are **not** exposed as entrypoints but are still part of the library:

- ``coloc_SWOT_L3_with_S1_CDSE_TOPS.py`` – core collocation logic (used by the wrapper).
- ``coloc_SWOT_L3_with_S1_CDSE_TOPS_sequential_wrapper.py`` – wrapper for processing a range of dates in sequential mode (can be invoked directly, but not installed as a script).
- ``coloc_SWOT_L3_with_S1_CDSE_TOPS_prun.py`` – parallel processing wrapper (internal).
- ``illustrate_coloc_swh_file.py`` / ``illustrate_coloc_swh_file_pyplot.py`` – visualisation helpers (not exposed).
- ``pickup_best_swot_file.py`` – utility for selecting the most recent SWOT product version.

For advanced usage, you can import these modules directly:

.. code-block:: python

    from s1swotcolocs import coloc_SWOT_L3_with_S1_CDSE_TOPS as swot_coloc

    # ... call functions as needed

-----------------------------------
Complete Example Workflow
-----------------------------------

1. **Download SWOT data** (if not already available):

   .. code-block:: bash

      download_l2_swot_karin_lr_ssh_windwave_cnes --config config.yml --start-date 2025-01-01 --stop-date 2025-01-10

2. **Run IW/EW collocation**:

   .. code-block:: bash

      coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel --startmonth 20250101 --stopmonth 20250110 --confpath config.yml --outputdir ./coloc_output

3. **Convert to STAC**:

   .. code-block:: bash

      convert_matchups_nc_to_stac --input ./coloc_output --output ./stac_output

4. **Run WV collocation** (for S1 Wave Mode):

   .. code-block:: bash

      matchups_wv_karin --conf-file config.yml --start-date 2025-01-01 --stop-date 2025-01-10 --output ./wv_matchups --max-time-diff 60

-------------------------
Configuration Reference
-------------------------

The YAML configuration file typically includes:

.. code-block:: yaml

   # SWOT directories
   SWOT_L2_AVISO_DIR: "/path/to/SWOT_L2_KARIN_LR_WindWave_AVISO"
   SWOT_L3_AVISO_DIR: "/path/to/SWOT_L3_LR_SSH_expert"

   # S1-WV directory (for WV collocation)
   S1_WV_ROOT: "/path/to/L2_daily/0.12"

   # Output directories
   WV_MATCHUP_OUTPUT_DIR: "/path/to/matchups_wv"
   HOST_META_COLOC_OUTPUT_DIR: "/path/to/meta_coloc"

   # CDSE cache
   CACHE_CDSE: "/path/to/cache"

   # Parameters
   DELTA_HOURS: 6                # time window for colocation (hours)
   MAX_AREA_SIZE: 200            # maximum polygon area (deg²)
   TOLERANCE_SIMPLIFICATION: 0.1 # geometry simplification tolerance

   # Docker/Apptainer (if used)
   DOCKER_BINARY_PATH: "/usr/bin/docker"
   DOCKER_IMAGE: "my-registry/s1swotcolocs:latest"
   APPTAINER_BINARY_PATH: "/usr/bin/singularity"

--------------
Troubleshooting
--------------

- **Missing `s1ifr` dependency**: The library depends on a private package. Ensure you have access to the GitLab package registry. In CI, use:
  .. code-block:: bash

     pip install s1ifr --index-url https://gitlab-ci-token:$CI_JOB_TOKEN@gitlab.ifremer.fr/api/v4/projects/4991/packages/pypi/simple

- **Memory errors**: Use the `sequential` variants (e.g., `coloc_seastate_SWOT_S1_sequential`) for large date ranges.
- **SWOT polygon building fails**: Increase `TOLERANCE_SIMPLIFICATION` or reduce `MAX_AREA_SIZE` in the config.

For more details, refer to the :doc:`installation` and :doc:`contributing` pages.
