"""
cdse_parquet_query.py — local replacement for `cdsodatacli.query.fetch_data`.

Instead of querying the live CDSE OData API
(``https://catalogue.dataspace.copernicus.eu/odata/v1/Products``), this module
reads the S1 product metadata pre-dumped on Ifremer disk as one parquet file
per day (the ``daily_parquet_files_v2`` directory produced by the
sentinel1-workbench pipeline, e.g.
``/scale/project/lops-siam-sentinel1-workbench/data/ifremer/metadata_cdse_sentinel1/daily_parquet_files_v2``).

Each ``s1_YYYY-MM-DD.parquet`` file is a full CDSE OData ``Products`` dump for
the products published that day: same column set as the JSON API returns
(``Id``, ``Name``, ``ContentDate``, ``Footprint``, ``GeoFootprint``,
``Attributes``, ...), plus a couple of derived columns that are dropped here.

The OData filter semantics of ``cdsodatacli.query.fetch_data`` are replicated
exactly:

* ``OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(...))``  →
  shapely ``footprint.intersects(query_geometry)``
* ``productType eq '<producttype>'``  (from the ``Attributes`` string
  attributes, e.g. ``SLC``)
* ``operationalMode eq '<sensormode>'`` (e.g. ``IW`` / ``EW``)
* ``ContentDate/Start gt '<start_datetime>'`` and
  ``ContentDate/Start lt '<end_datetime>'`` (strict inequalities, on the
  product acquisition start time)

The post-processing pipeline is then **reused as-is** from ``cdsodatacli.query``
(``remove_duplicates`` → ``multy_to_poly`` → ``sea_percent``), so the returned
GeoDataFrame is column-for-column the same as the live OData path (plus the
``id_original_query`` of the query gdf row, as the live path also does).
"""

import collections
import datetime
import json
import logging
import os
import threading

import cdsodatacli.query
import pandas as pd
import shapely

app_logger = logging.getLogger(__name__)

# Columns that exist in the daily parquet dumps but not in the OData JSON
# response and that no downstream consumer needs.
_PARQUET_ONLY_COLUMNS = ("WVImagettes", "geometry")

# One S1 product is published at most ~1 day after acquisition; a 2-day margin
# on each side of the query window is more than enough and keeps the number of
# daily files to load very small (a 2 h query window → ≤ 5 files).
_DAILY_FILE_MARGIN_DAYS = 2

# Bounded LRU cache of preprocessed daily frames (RAM ~ a few hundred MB).
_MAX_CACHED_DAYS = 8

# Slicing period large enough that no realistic query window is ever split by
# `cdsodatacli.query.normalize_gdf` (which defaults to 1 week otherwise).
_NO_SLICING_PERIOD = datetime.timedelta(days=365 * 10)

_cache_lock = threading.Lock()
_daily_cache: "collections.OrderedDict[pd.Timestamp, pd.DataFrame]" = (
    collections.OrderedDict()
)


def _parse_attributes(attr_str):
    """Extract (operationalMode, productType) from the OData Attributes JSON."""
    try:
        attrs = json.loads(attr_str)
    except (TypeError, ValueError):
        return None, None
    op_mode = None
    prod_type = None
    if isinstance(attrs, list):
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            name = attr.get("Name")
            value = attr.get("Value")
            if name == "operationalMode":
                op_mode = value
            elif name == "productType":
                prod_type = value
    return op_mode, prod_type


def _name_prefix_fallback(name):
    """Fallback mode/type from the SAFE name: S1D_<IW|EW>_<SLC|GRDH|...>."""
    if not isinstance(name, str):
        return None, None
    parts = name.split("_")
    if len(parts) < 3 or not name.startswith("S1"):
        return None, None
    return parts[1], parts[2]


def _load_daily_frame(parquet_dir, day):
    """Load (and preprocess) one daily parquet file, with a bounded LRU cache.

    Precomputed columns:
      * ``_ContentStart_dt`` — product acquisition start (UTC datetime64[ns])
      * ``_op_mode`` / ``_prod_type`` — operationalMode / productType
      * ``_fp_geom`` — shapely footprint geometry (same parse as
        ``cdsodatacli.query.multy_to_poly`` applies to ``Footprint``)
    """
    fpath = os.path.join(parquet_dir, f"s1_{day:%Y-%m-%d}.parquet")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"daily CDSE parquet file not found: {fpath}")

    with _cache_lock:
        if day in _daily_cache:
            _daily_cache.move_to_end(day)
            return _daily_cache[day]

    frame = pd.read_parquet(fpath)

    for col in _PARQUET_ONLY_COLUMNS:
        if col in frame.columns:
            frame = frame.drop(columns=[col])

    # Product acquisition start time (OData filter is on ContentDate/Start).
    frame["_ContentStart_dt"] = (
        frame["ContentDate"]
        .astype(str)
        .str.extract(r'"Start":\s*"([^"]+)"', expand=False)
        .pipe(pd.to_datetime, utc=True, errors="coerce")
    )

    # operationalMode / productType from the Attributes JSON, with a SAFE-name
    # prefix fallback (S1D_IW_SLC / S1D_EW_GRDH / ...).
    parsed = frame["Attributes"].map(_parse_attributes)
    frame["_op_mode"] = parsed.map(lambda t: t[0])
    frame["_prod_type"] = parsed.map(lambda t: t[1])
    has_mode = frame["_op_mode"].notna()
    if not has_mode.all():
        fallback = frame.loc[~has_mode, "Name"].map(_name_prefix_fallback)
        frame.loc[~has_mode, "_op_mode"] = fallback.map(lambda t: t[0])
        frame.loc[~has_mode, "_prod_type"] = fallback.map(lambda t: t[1])

    # Footprint geometry, parsed exactly like cdsodatacli.query.multy_to_poly.
    wkt = frame["Footprint"].str.split(";", expand=True)[1].str.strip().str[:-1]
    frame["_fp_geom"] = shapely.from_wkt(wkt, on_invalid="ignore")
    n_bad_geom = int(frame["_fp_geom"].isna().sum())
    if n_bad_geom:
        app_logger.warning(
            "daily parquet %s: %i unparseable footprint geometry(ies) dropped "
            "from the candidate pool.",
            os.path.basename(fpath),
            n_bad_geom,
        )
        frame = frame[frame["_fp_geom"].notna()]

    with _cache_lock:
        _daily_cache[day] = frame
        _daily_cache.move_to_end(day)
        while len(_daily_cache) > _MAX_CACHED_DAYS:
            _daily_cache.popitem(last=False)

    return frame


def _candidate_frames(parquet_dir, start_dt, end_dt):
    """Concatenate the daily frames covering (start_dt, end_dt) with margins."""
    first_day = pd.Timestamp(start_dt).date() - datetime.timedelta(
        days=_DAILY_FILE_MARGIN_DAYS
    )
    last_day = pd.Timestamp(end_dt).date() + datetime.timedelta(
        days=_DAILY_FILE_MARGIN_DAYS
    )
    frames = []
    day = first_day
    while day <= last_day:
        stamp = pd.Timestamp(day)
        try:
            frames.append(_load_daily_frame(parquet_dir, stamp))
        except FileNotFoundError:
            app_logger.warning(
                "daily CDSE parquet file missing for %s — products acquired "
                "that day cannot be matched (gap in the local metadata dump).",
                f"{day:%Y-%m-%d}",
            )
        day += datetime.timedelta(days=1)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _filter_frame(frame, row):
    """Apply the OData-equivalent filters of one query gdf row to a candidate frame."""
    start_dt = row["start_datetime"]
    end_dt = row["end_datetime"]

    mask = (frame["_ContentStart_dt"] > start_dt) & (frame["_ContentStart_dt"] < end_dt)
    if "sensormode" in row and pd.notna(row["sensormode"]):
        mask &= frame["_op_mode"] == row["sensormode"]
    if "producttype" in row and pd.notna(row["producttype"]):
        # The CDSE OData `eq` on StringAttribute/Value is effectively a
        # case-insensitive substring match, and the public catalog excludes
        # `_PRIVATE` (development) products. Reproduce both: the productType
        # must contain the requested type and must not be a _PRIVATE product.
        # (Verified to reproduce the live OData selection exactly, IW and EW.)
        pt = frame["_prod_type"].fillna("")
        mask &= pt.str.contains(str(row["producttype"]), case=False) & ~pt.str.contains(
            "_PRIVATE", case=False
        )
    if not mask.any():
        return None

    sub = frame.loc[mask]
    # bbox prefilter, then the exact OData.CSC.Intersects equivalent.
    query_geom = row["geometry"]
    qminx, qminy, qmaxx, qmaxy = shapely.bounds(query_geom)
    bboxes = shapely.bounds(sub["_fp_geom"])
    bbox_mask = (
        (bboxes[:, 0] <= qmaxx)
        & (bboxes[:, 2] >= qminx)
        & (bboxes[:, 1] <= qmaxy)
        & (bboxes[:, 3] >= qminy)
    )
    sub = sub.loc[bbox_mask]
    if sub.empty:
        return None
    sub = sub.loc[shapely.intersects(sub["_fp_geom"], query_geom)]
    return sub if len(sub) else None


def fetch_data_from_parquet(gdf, parquet_dir, min_sea_percent=None):
    """Local (parquet) equivalent of ``cdsodatacli.query.fetch_data``.

    Args:
        gdf (gpd.GeoDataFrame): same contract as ``cdsodatacli.query.fetch_data``
            (``start_datetime``, ``end_datetime``, ``id_query``, optional
            ``sensormode`` / ``producttype`` / ``geometry``).
        parquet_dir (str): directory containing the daily
            ``s1_YYYY-MM-DD.parquet`` metadata files.
        min_sea_percent (float, optional): same meaning as in the live path.

    Returns:
        gpd.GeoDataFrame (or None if no product matched): identical column set
        and post-processing as the live OData path.
    """
    if gdf is None or len(gdf) == 0:
        app_logger.warning("fetch_data_from_parquet: empty GeoDataFrame")
        return None
    if not parquet_dir or not os.path.isdir(parquet_dir):
        raise ValueError(
            f"CDSE_METADATA_SOURCE=parquet but CDSE_PARQUET_DIR is not an "
            f"existing directory: {parquet_dir!r}"
        )

    # Drop NaT-time rows before normalize_gdf (which cannot handle them —
    # the live path guards against NaT in do_cdse_query instead).
    gdf_in = gdf.copy()
    nat_mask = gdf_in["start_datetime"].isna() | gdf_in["end_datetime"].isna()
    if nat_mask.any():
        app_logger.warning(
            "fetch_data_from_parquet: dropping %i row(s) with NaT start or "
            "end datetime.",
            int(nat_mask.sum()),
        )
        gdf_in = gdf_in.loc[~nat_mask]
        if len(gdf_in) == 0:
            return None

    # Same normalization (tz, geometry validation, id_original_query) as the
    # live path, with a period large enough that no time slicing happens —
    # the local pool has no OData $top=1000 limit to work around.
    gdf_norm = cdsodatacli.query.normalize_gdf(
        gdf=gdf_in, timedelta_slice=_NO_SLICING_PERIOD
    )
    if gdf_norm is None or len(gdf_norm) == 0:
        app_logger.error("fetch_data_from_parquet: gdf is empty after normalize")
        return None

    collected = None
    for idx in gdf_norm.index:
        row = gdf_norm.loc[idx]
        if pd.isna(row["start_datetime"]) or pd.isna(row["end_datetime"]):
            app_logger.warning(
                "fetch_data_from_parquet: NaT start or end datetime on "
                "query %s — skipping.",
                row.get("id_original_query"),
            )
            continue
        if not isinstance(row["geometry"], shapely.geometry.base.BaseGeometry):
            app_logger.warning(
                "fetch_data_from_parquet: invalid geometry on query %s — skipping.",
                row.get("id_original_query"),
            )
            continue

        candidates = _candidate_frames(
            parquet_dir, row["start_datetime"], row["end_datetime"]
        )
        if candidates is None:
            continue
        sub = _filter_frame(candidates, row)
        if sub is None:
            continue
        sub = sub.copy()
        # Same as the live path: tag each product with the query id of the row
        # that matched it (downstream save_meta_coloc_output reads token [1]
        # as the SWOT file path).
        sub["id_original_query"] = row["id_original_query"]
        if collected is None:
            collected = sub
        else:
            collected = pd.concat([collected, sub], ignore_index=True)

    if collected is None or collected.empty:
        return None

    # Same post-processing pipeline as the live path (pure local operations).
    data_dedup = cdsodatacli.query.remove_duplicates(safes_ori=collected)
    full_data = cdsodatacli.query.multy_to_poly(collected_data=data_dedup)
    if full_data is not None and not full_data.empty and min_sea_percent is not None:
        full_data = cdsodatacli.query.sea_percent(
            collected_data=full_data, min_sea_percent=min_sea_percent
        )
    if full_data is None or full_data.empty:
        return None

    # Downstream (save_meta_coloc_output) reads ContentDate via `.str["Start"]`,
    # i.e. expects the OData dict form; the parquet stores a JSON string.
    full_data = full_data.copy()
    full_data["ContentDate"] = full_data["ContentDate"].map(
        lambda v: json.loads(v) if isinstance(v, str) else v
    )
    return full_data


def clear_cache():
    """Drop all cached daily frames (for tests)."""
    with _cache_lock:
        _daily_cache.clear()
