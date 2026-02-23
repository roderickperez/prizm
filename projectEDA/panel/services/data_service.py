from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from projectEDA import ProjectEDA_constants as constants


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = safe_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(text, errors="coerce").date()
    except Exception:
        return None


def _normalize_token(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in safe_text(value))
    return " ".join(cleaned.split())


def _load_mnemonics_master() -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(__file__).resolve().parents[2] / "mnemonics_master.json"
    if not path.exists():
        return {}, {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    alias_to_group: dict[str, str] = {}
    for group_name, payload in data.items():
        alias_to_group[_normalize_token(group_name)] = group_name

        if isinstance(payload, dict):
            canonical = safe_text(payload.get("canonical"), "")
            if canonical:
                alias_to_group[_normalize_token(canonical)] = group_name
            for alias in payload.get("aliases", []) or []:
                token = _normalize_token(alias)
                if token:
                    alias_to_group[token] = group_name

    return data, alias_to_group


def normalize_log_group(log_name: str, alias_to_group: dict[str, str]) -> str:
    token = _normalize_token(log_name)
    if token in alias_to_group:
        return alias_to_group[token]

    compact = token.replace(" ", "")
    for alias_token, group_name in alias_to_group.items():
        if alias_token.replace(" ", "") == compact:
            return group_name

    upper = safe_text(log_name).upper()
    if upper.startswith("GR"):
        return "Gamma Ray"
    if upper.startswith("RH") or "DENS" in upper:
        return "Density"
    if upper.startswith("DT"):
        return "Sonic (DT)"
    if "CALI" in upper or upper.startswith("CAL"):
        return "Caliper"
    if "RES" in upper or upper.startswith(("RD", "RM", "RS", "RXO", "RT")):
        return "Resistivity (Best)"
    return "Other"


@dataclass
class ExtractedData:
    project_name: str
    wells: pd.DataFrame
    logs: pd.DataFrame
    tops: pd.DataFrame


class ProjectEDADataService:
    def __init__(self, db_path: Path | str, test_mode: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.test_mode = bool(test_mode)
        self.mnemonics_master, self.alias_to_group = _load_mnemonics_master()
        self._init_db()

    def _conn(self):
        return duckdb.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key VARCHAR,
                    value VARCHAR
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS wells (
                    well_guid VARCHAR,
                    well_name VARCHAR,
                    uwi VARCHAR,
                    x DOUBLE,
                    y DOUBLE,
                    latitude DOUBLE,
                    longitude DOUBLE,
                    location VARCHAR,
                    spud_date DATE
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    well_guid VARCHAR,
                    well_name VARCHAR,
                    log_name VARCHAR,
                    log_group VARCHAR,
                    mnemonic_group VARCHAR,
                    md DOUBLE,
                    value DOUBLE,
                    unit VARCHAR
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS tops (
                    well_guid VARCHAR,
                    well_name VARCHAR,
                    top_name VARCHAR,
                    md DOUBLE,
                    tvd DOUBLE
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS log_standardization (
                    log_name VARCHAR,
                    log_group VARCHAR,
                    priority INTEGER
                )
                """
            )
            cols = con.execute("PRAGMA table_info('logs')").df()["name"].tolist()
            if "mnemonic_group" not in cols:
                con.execute("ALTER TABLE logs ADD COLUMN mnemonic_group VARCHAR")

    def _read_selections(self) -> dict[str, Any]:
        path = Path(str(constants.SELECTIONS_FILE))
        if not path.exists():
            env_path = Path(str(Path().cwd()))
            _ = env_path
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def refresh_from_petrel(self, force: bool = False) -> dict[str, Any]:
        if self.test_mode:
            counts = self.get_counts()
            if force or counts["wells"] == 0:
                extracted = self._empty_data("Project EDA (Test Mode)")
                self._persist_extracted(extracted)
                self._seed_standardization()
            return self.get_counts()

        counts = self.get_counts()
        if not force and counts["wells"] > 0:
            return counts

        extracted = self._extract_data()
        self._persist_extracted(extracted)
        self._seed_standardization()
        return self.get_counts()

    def _extract_data(self) -> ExtractedData:
        try:
            from cegalprizm.pythontool import PetrelConnection
        except Exception:
            return self._synthetic_data("Synthetic Project")

    def _empty_data(self, project_name: str) -> ExtractedData:
        wells = pd.DataFrame(
            columns=["well_guid", "well_name", "uwi", "x", "y", "latitude", "longitude", "location", "spud_date"]
        )
        logs = pd.DataFrame(
            columns=["well_guid", "well_name", "log_name", "log_group", "mnemonic_group", "md", "value", "unit"]
        )
        tops = pd.DataFrame(columns=["well_guid", "well_name", "top_name", "md", "tvd"])
        return ExtractedData(project_name=project_name, wells=wells, logs=logs, tops=tops)

        try:
            ptp = PetrelConnection(allow_experimental=True)
            project_name = safe_text(ptp.get_current_project_name(), "Unknown Project")
            selections = self._read_selections()
            scope_mode = int(selections.get("scope_mode", 2))
            selected_well_guids = set(map(str, selections.get("well_guids", []) or []))
            selected_log_names = set(map(str, selections.get("global_log_names", []) or []))

            if scope_mode == 1 and selected_well_guids:
                wells = [w for w in ptp.get_petrelobjects_by_guids(list(selected_well_guids)) if w is not None]
            elif scope_mode == 1:
                wells = []
            else:
                wells = list(getattr(ptp, "wells", []))

            well_rows: list[dict[str, Any]] = []
            log_rows: list[dict[str, Any]] = []
            top_rows: list[dict[str, Any]] = []

            for well in wells:
                well_guid = safe_text(getattr(well, "guid", getattr(well, "id", "")), "")
                well_name = safe_text(getattr(well, "petrel_name", ""), "Unknown")
                uwi = safe_text(getattr(well, "uwi", ""), "")
                coords = getattr(well, "wellhead_coordinates", None) or []
                x = safe_float(coords[0]) if len(coords) > 0 else math.nan
                y = safe_float(coords[1]) if len(coords) > 1 else math.nan
                lat = safe_float(getattr(well, "latitude", math.nan))
                lon = safe_float(getattr(well, "longitude", math.nan))
                location = safe_text(getattr(well, "path", ""), "")
                spud = None
                try:
                    stats = well.retrieve_stats()
                    spud = safe_date(stats.get("Spud Date") if isinstance(stats, dict) else None)
                except Exception:
                    spud = None

                well_rows.append(
                    {
                        "well_guid": well_guid,
                        "well_name": well_name,
                        "uwi": uwi,
                        "x": x,
                        "y": y,
                        "latitude": lat,
                        "longitude": lon,
                        "location": location,
                        "spud_date": spud,
                    }
                )

                for log in list(getattr(well, "logs", []) or []):
                    log_name = safe_text(getattr(log, "petrel_name", ""), "")
                    if selected_log_names and log_name not in selected_log_names:
                        continue
                    try:
                        df = log.as_dataframe()
                    except Exception:
                        continue
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    if "MD" not in df.columns or "Value" not in df.columns:
                        continue

                    tmp = df[["MD", "Value"]].copy()
                    tmp["MD"] = pd.to_numeric(tmp["MD"], errors="coerce")
                    tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce")
                    tmp = tmp.dropna(subset=["MD", "Value"])
                    if tmp.empty:
                        continue

                    if len(tmp) > constants.MAX_POINTS_PER_LOG:
                        idx = np.linspace(0, len(tmp) - 1, constants.MAX_POINTS_PER_LOG).astype(int)
                        tmp = tmp.iloc[idx]

                    log_group = normalize_log_group(log_name, self.alias_to_group)
                    unit = safe_text(getattr(log, "unit", ""), "")

                    tmp["well_guid"] = well_guid
                    tmp["well_name"] = well_name
                    tmp["log_name"] = log_name
                    tmp["log_group"] = log_group
                    tmp["mnemonic_group"] = log_group
                    tmp["unit"] = unit
                    tmp = tmp.rename(columns={"MD": "md", "Value": "value"})
                    log_rows.extend(tmp.to_dict(orient="records"))

            marker_collections = list(getattr(ptp, "markercollections", []) or [])
            for mc in marker_collections:
                try:
                    mdf = mc.as_dataframe(include_unconnected_markers=False)
                except Exception:
                    continue
                if not isinstance(mdf, pd.DataFrame) or mdf.empty:
                    continue

                well_col = "Well identifier (Well name)"
                top_col = "Surface"
                md_col = "MD"
                if not {well_col, top_col, md_col}.issubset(mdf.columns):
                    continue

                tmp = mdf[[well_col, top_col, md_col]].copy()
                tmp = tmp.rename(
                    columns={well_col: "well_name", top_col: "top_name", md_col: "md"}
                )
                tmp["md"] = pd.to_numeric(tmp["md"], errors="coerce")
                tmp = tmp.dropna(subset=["well_name", "top_name", "md"])
                tmp["well_guid"] = ""
                tmp["tvd"] = np.nan
                top_rows.extend(tmp[["well_guid", "well_name", "top_name", "md", "tvd"]].to_dict(orient="records"))

            wells_df = pd.DataFrame(well_rows)
            logs_df = pd.DataFrame(log_rows)
            tops_df = pd.DataFrame(top_rows)
            return ExtractedData(project_name=project_name, wells=wells_df, logs=logs_df, tops=tops_df)
        except Exception:
            return self._synthetic_data("Synthetic Project")

    def _synthetic_data(self, project_name: str) -> ExtractedData:
        wells = pd.DataFrame(
            [
                {
                    "well_guid": "W1",
                    "well_name": "Well-A",
                    "uwi": "UWI-A",
                    "x": 10.0,
                    "y": 20.0,
                    "latitude": 58.1,
                    "longitude": 11.2,
                    "location": "Synthetic",
                    "spud_date": datetime(2021, 1, 10).date(),
                },
                {
                    "well_guid": "W2",
                    "well_name": "Well-B",
                    "uwi": "UWI-B",
                    "x": 11.5,
                    "y": 21.1,
                    "latitude": 58.2,
                    "longitude": 11.4,
                    "location": "Synthetic",
                    "spud_date": datetime(2022, 4, 16).date(),
                },
            ]
        )

        md = np.arange(1000, 2100, 5)
        logs_rows = []
        for _, w in wells.iterrows():
            for log_name, scale in [("GR", 100), ("RHO", 2.2), ("DT", 120)]:
                vals = np.random.normal(loc=scale, scale=max(scale * 0.05, 0.2), size=len(md))
                for m, v in zip(md, vals):
                    logs_rows.append(
                        {
                            "well_guid": w["well_guid"],
                            "well_name": w["well_name"],
                            "log_name": log_name,
                            "log_group": normalize_log_group(log_name, self.alias_to_group),
                            "mnemonic_group": normalize_log_group(log_name, self.alias_to_group),
                            "md": float(m),
                            "value": float(v),
                            "unit": "",
                        }
                    )
        logs = pd.DataFrame(logs_rows)

        tops = pd.DataFrame(
            [
                {"well_guid": "W1", "well_name": "Well-A", "top_name": "Top_A", "md": 1200.0, "tvd": np.nan},
                {"well_guid": "W1", "well_name": "Well-A", "top_name": "Base_A", "md": 1850.0, "tvd": np.nan},
                {"well_guid": "W2", "well_name": "Well-B", "top_name": "Top_A", "md": 1300.0, "tvd": np.nan},
                {"well_guid": "W2", "well_name": "Well-B", "top_name": "Base_A", "md": 1900.0, "tvd": np.nan},
            ]
        )

        return ExtractedData(project_name=project_name, wells=wells, logs=logs, tops=tops)

    def _persist_extracted(self, extracted: ExtractedData) -> None:
        wells = extracted.wells if isinstance(extracted.wells, pd.DataFrame) else pd.DataFrame()
        logs = extracted.logs if isinstance(extracted.logs, pd.DataFrame) else pd.DataFrame()
        tops = extracted.tops if isinstance(extracted.tops, pd.DataFrame) else pd.DataFrame()

        with self._conn() as con:
            con.execute("DELETE FROM metadata")
            con.execute("DELETE FROM wells")
            con.execute("DELETE FROM logs")
            con.execute("DELETE FROM tops")

            con.execute("INSERT INTO metadata VALUES ('project_name', ?)", [extracted.project_name])

            if not wells.empty:
                con.register("wells_df", wells)
                con.execute("INSERT INTO wells SELECT * FROM wells_df")
            if not logs.empty:
                con.register("logs_df", logs)
                con.execute("INSERT INTO logs SELECT * FROM logs_df")
            if not tops.empty:
                con.register("tops_df", tops)
                con.execute("INSERT INTO tops SELECT * FROM tops_df")

    def _seed_standardization(self) -> None:
        with self._conn() as con:
            existing = con.execute("SELECT COUNT(*) FROM log_standardization").fetchone()[0]
            if existing > 0:
                return
            con.execute(
                """
                INSERT INTO log_standardization
                SELECT
                    log_name,
                    COALESCE(mnemonic_group, log_group),
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(mnemonic_group, log_group)
                        ORDER BY COUNT(*) DESC, log_name
                    ) AS priority
                FROM logs
                GROUP BY log_name, log_group, mnemonic_group
                """
            )

    def get_project_name(self) -> str:
        with self._conn() as con:
            row = con.execute("SELECT value FROM metadata WHERE key='project_name' LIMIT 1").fetchone()
            return row[0] if row else "Unknown Project"

    def get_counts(self) -> dict[str, int]:
        with self._conn() as con:
            wells = con.execute("SELECT COUNT(*) FROM wells").fetchone()[0]
            logs = con.execute("SELECT COUNT(DISTINCT log_name) FROM logs").fetchone()[0]
            tops = con.execute("SELECT COUNT(DISTINCT top_name) FROM tops").fetchone()[0]
        return {"wells": int(wells), "logs": int(logs), "tops": int(tops)}

    def get_wells(self) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute("SELECT * FROM wells ORDER BY well_name").df()

    def get_tops(self) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute("SELECT * FROM tops ORDER BY well_name, md").df()

    def get_gantt(self) -> pd.DataFrame:
        with self._conn() as con:
            df = con.execute(
                """
                SELECT well_name, spud_date
                FROM wells
                WHERE spud_date IS NOT NULL
                ORDER BY spud_date
                """
            ).df()
        if df.empty:
            return df
        df["start"] = pd.to_datetime(df["spud_date"])
        df["finish"] = df["start"] + pd.to_timedelta(1, unit="D")
        return df

    def get_logs_catalog(self) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute(
                """
                SELECT
                    log_name,
                    log_group,
                    COALESCE(mnemonic_group, log_group) AS mnemonic_group,
                    COUNT(DISTINCT well_name) AS wells,
                    COUNT(*) AS rows,
                    MIN(md) AS md_min,
                    MAX(md) AS md_max
                FROM logs
                GROUP BY log_name, log_group, mnemonic_group
                ORDER BY mnemonic_group, log_name
                """
            ).df()

    def get_log_standardization(self) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute(
                """
                SELECT log_name, log_group, priority
                FROM log_standardization
                ORDER BY log_group, priority, log_name
                """
            ).df()

    def get_mnemonic_groups(self) -> list[str]:
        if self.mnemonics_master:
            return sorted(self.mnemonics_master.keys())
        with self._conn() as con:
            df = con.execute("SELECT DISTINCT log_group FROM logs ORDER BY log_group").df()
        return df["log_group"].dropna().astype(str).tolist() if not df.empty else []

    def save_log_standardization(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        payload = df[["log_name", "log_group", "priority"]].copy()
        payload["priority"] = pd.to_numeric(payload["priority"], errors="coerce").fillna(1).astype(int)
        with self._conn() as con:
            con.execute("DELETE FROM log_standardization")
            con.register("std_df", payload)
            con.execute("INSERT INTO log_standardization SELECT * FROM std_df")

    def get_depth_range(self, log_names: list[str] | None = None) -> tuple[float, float]:
        with self._conn() as con:
            if log_names:
                placeholders = ",".join(["?"] * len(log_names))
                row = con.execute(
                    f"SELECT MIN(md), MAX(md) FROM logs WHERE log_name IN ({placeholders})",
                    log_names,
                ).fetchone()
            else:
                row = con.execute("SELECT MIN(md), MAX(md) FROM logs").fetchone()

        if not row or row[0] is None or row[1] is None:
            return (0.0, 0.0)
        return (float(row[0]), float(row[1]))

    def get_logs_data(
        self,
        well_names: list[str] | None = None,
        log_names: list[str] | None = None,
        depth_min: float | None = None,
        depth_max: float | None = None,
    ) -> pd.DataFrame:
        query = "SELECT well_name, log_name, md, value FROM logs WHERE 1=1"
        params: list[Any] = []
        if well_names:
            placeholders = ",".join(["?"] * len(well_names))
            query += f" AND well_name IN ({placeholders})"
            params.extend(well_names)
        if log_names:
            placeholders = ",".join(["?"] * len(log_names))
            query += f" AND log_name IN ({placeholders})"
            params.extend(log_names)
        if depth_min is not None:
            query += " AND md >= ?"
            params.append(depth_min)
        if depth_max is not None:
            query += " AND md <= ?"
            params.append(depth_max)
        query += " ORDER BY well_name, log_name, md"

        with self._conn() as con:
            return con.execute(query, params).df()

    def get_top_names(self) -> list[str]:
        with self._conn() as con:
            df = con.execute("SELECT DISTINCT top_name FROM tops ORDER BY top_name").df()
        return df["top_name"].dropna().astype(str).tolist() if not df.empty else []

    def get_depth_window_from_tops(self, top_name: str, base_name: str) -> tuple[float, float] | None:
        with self._conn() as con:
            top_row = con.execute("SELECT AVG(md) FROM tops WHERE top_name=?", [top_name]).fetchone()
            base_row = con.execute("SELECT AVG(md) FROM tops WHERE top_name=?", [base_name]).fetchone()
        if not top_row or not base_row or top_row[0] is None or base_row[0] is None:
            return None
        a, b = float(top_row[0]), float(base_row[0])
        return (min(a, b), max(a, b))

    def get_completeness(self, log_names: list[str], depth_min: float, depth_max: float) -> pd.DataFrame:
        if not log_names:
            return pd.DataFrame()

        wells = self.get_wells()[["well_name"]]
        all_pairs = wells.assign(key=1).merge(pd.DataFrame({"log_name": log_names, "key": 1}), on="key").drop(columns=["key"])

        with self._conn() as con:
            placeholders = ",".join(["?"] * len(log_names))
            available = con.execute(
                f"""
                SELECT well_name, log_name, COUNT(*) AS cnt
                FROM logs
                WHERE log_name IN ({placeholders})
                  AND md >= ?
                  AND md <= ?
                GROUP BY well_name, log_name
                """,
                [*log_names, depth_min, depth_max],
            ).df()

        merged = all_pairs.merge(available, on=["well_name", "log_name"], how="left")
        merged["is_available"] = merged["cnt"].fillna(0).astype(int) > 0
        matrix = merged.pivot(index="well_name", columns="log_name", values="is_available").fillna(False)
        matrix = matrix.reset_index()
        return matrix

    def get_single_log_stats(
        self, well_name: str, log_name: str, depth_min: float | None = None, depth_max: float | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.get_logs_data([well_name], [log_name], depth_min, depth_max)
        if df.empty:
            return df, pd.DataFrame()
        stats = (
            df["value"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .rename("value")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "stat"})
        )
        return df, stats

    def get_multi_well_crossplot(
        self, x_log: str, y_log: str, depth_min: float | None = None, depth_max: float | None = None
    ) -> pd.DataFrame:
        x_df = self.get_logs_data(log_names=[x_log], depth_min=depth_min, depth_max=depth_max)
        y_df = self.get_logs_data(log_names=[y_log], depth_min=depth_min, depth_max=depth_max)
        if x_df.empty or y_df.empty:
            return pd.DataFrame()

        x_agg = x_df.groupby("well_name", as_index=False)["value"].mean().rename(columns={"value": "x_value"})
        y_agg = y_df.groupby("well_name", as_index=False)["value"].mean().rename(columns={"value": "y_value"})
        out = x_agg.merge(y_agg, on="well_name", how="inner")
        return out
