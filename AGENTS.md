# AGENTS.md - s1swotcolocs

## Project Overview
Python library for creating co-locations between Sentinel-1 (S1) images and SWOT KaRin swath data.

## Core Commands
- **Tests**: `make test` (pytest) or `make test-all` (tox).
- **Linting/Formatting**: Use `pre-commit run --all-files`. The project uses Ruff and Black.
- **Documentation**: `make docs` to generate Sphinx HTML.
- **Build/Install**: `make dist` for distribution packages; `make install` for local installation.

## Development Workflow
1. Run `pre-commit run --all-files` before committing.
2. Verify changes with `pytest`.
3. For comprehensive testing across Python versions, use `tox`.

## Architecture & Structure
- **Source**: `src/s1swotcolocs/`
- **Tests**: `tests/`
- **Config**: `pyproject.toml` (dependencies and scripts), `ruff.toml` (linting).

## Key Constraints & Quirks
- **Dependencies**: Uses a private PyPI index for some packages: `https://gitlab.ifremer.fr/api/v4/projects/4991/packages/pypi/simple`.
- **Scripts**: Several CLI tools are defined in `pyproject.toml` (e.g., `coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel`).
- **Verification Order**: Recommended `pre-commit -> pytest -> make docs`.
