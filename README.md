# s1swotcolocs

[![CI](https://github.com/umr-lops/s1swotcolocs/actions/workflows/ci.yaml/badge.svg)](https://github.com/umr-lops/s1swotcolocs/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/s1swotcolocs.svg)](https://pypi.org/project/s1swotcolocs/)
[![Python versions](https://img.shields.io/pypi/pyversions/s1swotcolocs.svg)](https://pypi.org/project/s1swotcolocs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![codecov](https://codecov.io/gh/umr-lops/s1swotcolocs/branch/main/graph/badge.svg)](https://codecov.io/gh/umr-lops/s1swotcolocs)

Python lib to create co-locations between Sentinel-1 (IW, EW, WV) products and SWOT KaRIn swath.

-   Free software: MIT license
-   Documentation: [docs/](docs/).

# Features

-   Find temporal and spatial co-locations between Sentinel-1 (S1) Level-1/Level-2 products and SWOT Level-2/Level-3 SSH data.
-   Process S1 **IW** (Interferometric Wide swath), **EW** (Extra Wide swath), and **WV** (Wave mode) acquisitions.
-   Interface with CDSE (Copernicus Data Space Ecosystem) for S1 IW/EW data discovery.
-   Direct file-tree based collocation for S1-WV daily enriched products (no CDSE query required).
-   Generate output as STAC-like JSON items (WV mode) or NetCDF files (IW/EW mode).
-   Configurable time difference and overlap percentage criteria.
-   Utility functions for geospatial operations and data handling relevant to S1 and SWOT.

# Usage

```python
import s1swotcolocs
```

## Creating meta-data colocation files (IW/EW)

```bash
# using the command-line entry point
coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel \
    --startdate 20250616 --stopdate 20250616 \
    --confpath src/s1swotcolocs/config.yml \
    --outputdir /tmp/
```

## WV-KaRIn matchups (new)

The script `matchups_WV_KaRIn_v2.py` (entrypoint `matchups_wv_karin`) collocates Sentinel-1 Wave Mode (WV) Level-2 products with SWOT KaRIn Level-2 LR SSH data.

**Key features**:

-   Memory-efficient: only SWOT swath-edge pixels are loaded.
-   No CDSE query - S1-WV daily files are located directly from a known file tree.
-   Outputs STAC-like JSON files per matchup, organised by year and day-of-year.
-   Supports date filtering (`--start-date`, `--stop-date`) and a development mode (`--dev`).
-   Configuration via YAML file (`--conf-file`).

**File tree expected for S1-WV daily files**:

```text
<S1_WV_ROOT>/<yyyy>/<doy>/S1{A,C,D}_WV_L2D_enriched_LOPS_<yyyymmdd>_daily_IPF_*.nc
```

**Example usage**:

```bash
matchups_wv_karin --conf-file config.yml \
    --start-date 2026-01-01 --stop-date 2026-01-10 \
    --dev --output /path/to/output --log-level DEBUG
```

**Configuration file example** (`config.yml`):

```yaml
SWOT_L2_AVISO_DIR: "/path/to/SWOT_L2_KARIN_LR_WindWave_AVISO"
S1_WV_ROOT: "/path/to/L2_daily/0.12"
WV_MATCHUP_OUTPUT_DIR: "/path/to/matchups_wv"
```

**Output**:

-   JSON matchup files under `output_dir/json_matchups/<yyyy>/<doy>/`.
-   A summary CSV (`test_json_matchups.csv`) with all matchups.
-   Optionally, debug images if `debug_image=True` in the code.

## Illustration of a co-location between SWOT KaRIn swath and Sentinel-1 IW swath

![coloc_swot_iw](docs/_static/figures/illustrate_coloc_s1_swot_iw.png)

## Illustration of a co-location between SWOT KaRIn reduced swath and Sentinel-1 IW swath

For triple co-location (KaRIn, SWOT-nadir, IW Level-2 WAV) purposes it can be important to limit the KaRIn swath to the low incidence part.

![coloc_swot_iw](docs/_static/figures/illustrate_coloc_s1_swot_iw_nadir.png)

## New: WV / SWOT KaRIn co-locations

The WV collocation module is now fully operational and documented. See the **WV-KaRIn matchups** section above for details.
