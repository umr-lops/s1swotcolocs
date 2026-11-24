import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from s1swotcolocs.cdse_parquet_query import fetch_data_from_parquet
from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import (
    do_cdse_query,
    treat_one_day_wrapper,
)

# ── Geometry helpers ─────────────────────────────────────────────────────────
# Off-shore box (mid-Atlantic, off Namibia), ~100 % sea.
OCEAN = (8.0, -24.0, 10.5, -21.5)
# Query box surrounding OCEAN and also covering the Namibian coast.
QUERY = (8.0, -25.0, 16.0, -15.0)
# Far away, no intersection with QUERY.
FAR = (10.0, 10.0, 11.0, 11.0)
# Inland Namibia → ~0 % sea (excluded by min_sea_percent=10).
LAND = (14.0, -22.0, 15.8, -20.2)

WINDOW_START = "2026-03-15T10:00:00.000000Z"
WINDOW_END = "2026-03-15T14:00:00.000000Z"

P1_NAME = "S1D_IW_SLC__1SSH_20260315T010000_20260315T010026_004200_005432_0001.SAFE"
P9_NAME = "S1D_IW_SLC__1SSH_20260315T020000_20260315T020026_004200_005432_0009.SAFE"


def _box_wkt(x0, y0, x1, y1):
    return (
        f"geography'SRID=4326;POLYGON (({x0} {y0}, {x1} {y0}, "
        f"{x1} {y1}, {x0} {y1}, {x0} {y0}))'"
    )


def _attributes(mode, producttype):
    return json.dumps(
        [
            {
                "@odata.type": "#OData.CSC.StringAttribute",
                "Name": "operationalMode",
                "Value": mode,
                "ValueType": "String",
            },
            {
                "@odata.type": "#OData.CSC.StringAttribute",
                "Name": "productType",
                "Value": producttype,
                "ValueType": "String",
            },
        ]
    )


def _content_date(start):
    return json.dumps({"Start": start, "End": start.replace("00Z", "26Z")})


def _row(pid, name, mode, producttype, box, start, mod_date):
    return {
        "Id": 1000 + pid,
        "Name": name,
        "ContentDate": _content_date(start),
        "Footprint": _box_wkt(*box),
        "GeoFootprint": "{}",
        "Attributes": _attributes(mode, producttype),
        "ModificationDate": mod_date,
        "OriginDate": mod_date,
        "S3Path": "s3://fake",
        "WVImagettes": float("nan"),
        "geometry": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
    }


def _write_daily(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def _make_parquet_dir(tmpdir):
    """Synthetic daily parquet dumps:
    s1_2026-03-15.parquet:
      P1  IW SLC  in window  ocean box      (expected match)
      P8  IW SLC  in window  ocean box      same Name as P1, newer ModificationDate
      P2  EW SLC  in window  ocean box      (wrong mode)
      P3  IW GRD  in window  ocean box      (wrong product type)
      P4  IW SLC  before window             ocean box
      P5  IW SLC  at window end (strict <)  ocean box
      P6  IW SLC  in window  far box        (no intersection)
      P7  IW SLC  in window  land box       (sea_percent < 10)
    s1_2026-03-16.parquet:
      P9  IW SLC  in window  ocean box      (published next day: margin check)
    """
    os.makedirs(tmpdir, exist_ok=True)
    day1 = [
        _row(
            1,
            P1_NAME,
            "IW",
            "IW_SLC__1S",
            OCEAN,
            "2026-03-15T11:00:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        _row(
            8,
            P1_NAME,
            "IW",
            "IW_SLC__1S",
            OCEAN,
            "2026-03-15T11:00:00.000000Z",
            "2026-03-16T02:00:00.000000Z",
        ),
        _row(
            2,
            "S1D_EW_SLC__1SAH_20260315T010000_20260315T010026_004200_005432_0002.SAFE",
            "EW",
            "EW_SLC__1S",
            OCEAN,
            "2026-03-15T11:30:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        _row(
            3,
            "S1D_IW_GRDH_1SDH_20260315T010000_20260315T010026_004200_005432_0003.SAFE",
            "IW",
            "IW_GRDH_1S",
            OCEAN,
            "2026-03-15T11:30:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        _row(
            4,
            "S1D_IW_SLC__1SSH_20260314T235900_20260314T235926_004200_005432_0004.SAFE",
            "IW",
            "IW_SLC__1S",
            OCEAN,
            "2026-03-15T09:59:59.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        _row(
            5,
            "S1D_IW_SLC__1SSH_20260315T055900_20260315T055926_004200_005432_0005.SAFE",
            "IW",
            "IW_SLC__1S",
            OCEAN,
            "2026-03-15T14:00:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        _row(
            6,
            "S1D_IW_SLC__1SSH_20260315T030000_20260315T030026_004200_005432_0006.SAFE",
            "IW",
            "IW_SLC__1S",
            FAR,
            "2026-03-15T12:00:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        _row(
            7,
            "S1D_IW_SLC__1SSH_20260315T040000_20260315T040026_004200_005432_0007.SAFE",
            "IW",
            "IW_SLC__1S",
            LAND,
            "2026-03-15T12:00:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
        # _PRIVATE (development) product: must be excluded even though it
        # matches mode, type (contains SLC), time and geometry.
        _row(
            10,
            "S1D_IW_SLC__1ADH_20260315T040000_20260315T040026_004200_005432_0010.SAFE",
            "IW",
            "IW_SLC__1A_PRIVATE",
            OCEAN,
            "2026-03-15T12:00:00.000000Z",
            "2026-03-15T20:00:00.000000Z",
        ),
    ]
    day2 = [
        _row(
            9,
            P9_NAME,
            "IW",
            "IW_SLC__1S",
            OCEAN,
            "2026-03-15T12:30:00.000000Z",
            "2026-03-16T03:00:00.000000Z",
        ),
    ]
    _write_daily(os.path.join(tmpdir, "s1_2026-03-15.parquet"), day1)
    _write_daily(os.path.join(tmpdir, "s1_2026-03-16.parquet"), day2)
    return tmpdir


def _query_gdf():
    geom = Polygon(
        [
            (QUERY[0], QUERY[1]),
            (QUERY[2], QUERY[1]),
            (QUERY[2], QUERY[3]),
            (QUERY[0], QUERY[3]),
        ]
    )
    return gpd.GeoDataFrame(
        {
            "start_datetime": pd.to_datetime(WINDOW_START, utc=True),
            "end_datetime": pd.to_datetime(WINDOW_END, utc=True),
            "collection": ["SENTINEL-1"],
            "sensormode": ["IW"],
            "producttype": ["SLC"],
            "id_query": [
                "SWOT SWOT_L2_LR_SSH_WindWave_000_000_20250315T100000_20250315T110000_PID0_01.nc 20250315T100000 20250315T110000"
            ],
        },
        geometry=[geom],
        crs="EPSG:4326",
    )


class TestFetchDataFromParquet(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="cdse_parquet_test_")
        self.parquet_dir = os.path.join(self.tmpdir, "daily")
        _make_parquet_dir(self.parquet_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        from s1swotcolocs import cdse_parquet_query

        cdse_parquet_query.clear_cache()

    def test_filters_and_dedup(self):
        """Mode, product type, strict time window, intersection and sea
        percent filters, plus Name dedup keeping the newest ModificationDate
        and the publication-lag margin on the daily files."""
        res = fetch_data_from_parquet(
            _query_gdf(), self.parquet_dir, min_sea_percent=10
        )
        self.assertIsNotNone(res)
        names = set(res["Name"])
        self.assertEqual(names, {P1_NAME, P9_NAME})
        # Dedup kept the P8 row (newer ModificationDate) for P1_NAME.
        mod = res.loc[res["Name"] == P1_NAME, "ModificationDate"].iloc[0]
        self.assertEqual(mod, "2026-03-16T02:00:00.000000Z")
        # id_original_query tagged with the query gdf row's id.
        self.assertTrue(
            (res["id_original_query"] == _query_gdf()["id_query"].iloc[0]).all()
        )
        # ContentDate is in the OData dict form downstream expects.
        cd0 = res["ContentDate"].iloc[0]
        self.assertIsInstance(cd0, dict)
        self.assertIn("Start", cd0)
        # geometry column is shapely (built by multy_to_poly).
        self.assertTrue(res.geometry.iloc[0].intersects(_query_gdf().geometry.iloc[0]))
        # parquet-only column dropped.
        self.assertNotIn("WVImagettes", res.columns)

    def test_no_min_sea_percent_keeps_land_product(self):
        res = fetch_data_from_parquet(_query_gdf(), self.parquet_dir)
        self.assertIsNotNone(res)
        # P7 (land box) is kept when no sea percent filter is applied.
        self.assertEqual(len(res), 3)

    def test_no_match_returns_none(self):
        gdf = _query_gdf()
        gdf.loc[0, "sensormode"] = "EW"  # only SLC products in the pool are IW/EW-mixed
        # keep IW SLC products in the pool but ask for EW in the FAR box:
        gdf2 = gdf.copy()
        gdf2.loc[0, "start_datetime"] = pd.Timestamp("2020-01-01T00:00:00Z")
        gdf2.loc[0, "end_datetime"] = pd.Timestamp("2020-01-01T01:00:00Z")
        res = fetch_data_from_parquet(gdf2, self.parquet_dir, min_sea_percent=10)
        self.assertIsNone(res)

    def test_missing_daily_files_do_not_crash(self):
        gdf = _query_gdf()
        gdf.loc[0, "start_datetime"] = pd.Timestamp("2026-03-04T10:00:00Z")
        gdf.loc[0, "end_datetime"] = pd.Timestamp("2026-03-04T14:00:00Z")
        res = fetch_data_from_parquet(gdf, self.parquet_dir, min_sea_percent=10)
        self.assertIsNone(res)

    def test_empty_gdf_returns_none(self):
        self.assertIsNone(fetch_data_from_parquet(gpd.GeoDataFrame(), self.parquet_dir))

    def test_nat_window_returns_none(self):
        gdf = _query_gdf()
        gdf.loc[0, "start_datetime"] = pd.NaT
        self.assertIsNone(fetch_data_from_parquet(gdf, self.parquet_dir))

    def test_bad_parquet_dir_raises(self):
        with self.assertRaises(ValueError):
            fetch_data_from_parquet(_query_gdf(), "/nonexistent/dir")
        with self.assertRaises(ValueError):
            fetch_data_from_parquet(_query_gdf(), None)


class TestDoCdseQueryDispatch(unittest.TestCase):
    def _gdf(self):
        gdf = _query_gdf()
        return gdf

    @patch("s1swotcolocs.cdse_parquet_query.fetch_data_from_parquet")
    def test_parquet_source_calls_local_fetcher(self, mock_pq):
        mock_pq.return_value = "LOCAL_RESULT"
        res = do_cdse_query(
            self._gdf(),
            metadata_source="parquet",
            parquet_dir="/some/dir",
        )
        self.assertEqual(res, "LOCAL_RESULT")
        mock_pq.assert_called_once()
        kwargs = mock_pq.call_args.kwargs
        self.assertEqual(kwargs["parquet_dir"], "/some/dir")
        self.assertEqual(kwargs["min_sea_percent"], 10)

    @patch("cdsodatacli.query.fetch_data")
    def test_live_source_calls_odata_fetcher(self, mock_odata):
        mock_odata.return_value = "LIVE_RESULT"
        res = do_cdse_query(self._gdf())
        self.assertEqual(res, "LIVE_RESULT")
        mock_odata.assert_called_once()

    @patch("cdsodatacli.query.fetch_data")
    def test_explicit_live_source_calls_odata_fetcher(self, mock_odata):
        mock_odata.return_value = "LIVE_RESULT"
        res = do_cdse_query(self._gdf(), metadata_source="live")
        self.assertEqual(res, "LIVE_RESULT")
        mock_odata.assert_called_once()

    @patch("s1swotcolocs.cdse_parquet_query.fetch_data_from_parquet")
    def test_nat_guard_applies_to_both_sources(self, mock_pq):
        gdf = _query_gdf()
        gdf.loc[0, "start_datetime"] = pd.NaT
        self.assertIsNone(do_cdse_query(gdf, metadata_source="parquet"))
        mock_pq.assert_not_called()


class TestTreatOneDayWrapperConfigValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="coloc_conf_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _confpath(self, extra):
        conf = {
            "SWOT_L2_AVISO_DIR": self.tmpdir,
            "CACHE_CDSE": self.tmpdir,
            "DELTA_HOURS": 1,
            "MAX_AREA_SIZE": 200,
            "TOLERANCE_SIMPLIFICATION": 0.1,
        }
        conf.update(extra)
        import yaml

        fpath = os.path.join(self.tmpdir, "conf.yml")
        with open(fpath, "w") as f:
            yaml.safe_dump(conf, f)
        return fpath

    def test_parquet_source_missing_dir_raises(self):
        confpath = self._confpath(
            {"CDSE_METADATA_SOURCE": "parquet", "CDSE_PARQUET_DIR": "/nonexistent"}
        )
        with self.assertRaises(ValueError):
            treat_one_day_wrapper(
                day2treat="20250101",
                outputdir=os.path.join(self.tmpdir, "out"),
                mode="IW",
                confpath=confpath,
            )

    def test_unknown_source_raises(self):
        confpath = self._confpath({"CDSE_METADATA_SOURCE": "bogus"})
        with self.assertRaises(ValueError):
            treat_one_day_wrapper(
                day2treat="20250101",
                outputdir=os.path.join(self.tmpdir, "out"),
                mode="IW",
                confpath=confpath,
            )

    def test_default_source_is_live_and_backwards_compatible(self):
        """A config without CDSE_METADATA_SOURCE still runs (live path, here
        with no SWOT files so it is a fast no-op)."""
        confpath = self._confpath({})
        cpt = treat_one_day_wrapper(
            day2treat="20250101",
            outputdir=os.path.join(self.tmpdir, "out"),
            mode="IW",
            confpath=confpath,
            disable_tqdm=True,
        )
        self.assertIsInstance(cpt, dict)


if __name__ == "__main__":
    unittest.main()
