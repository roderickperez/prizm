from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


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

    parsed = datetime.fromisoformat(text) if "T" in text else None
    return parsed.date() if parsed else None


def normalize_token(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in safe_text(value))
    return " ".join(cleaned.split())


def load_mnemonics_master(project_root: Path | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    root = project_root or Path(__file__).resolve().parent
    path = root / "mnemonics_master.json"
    if not path.exists():
        return {}, {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    alias_to_group: dict[str, str] = {}
    for group_name, meta in payload.items():
        alias_to_group[normalize_token(group_name)] = group_name

        if isinstance(meta, dict):
            canonical = safe_text(meta.get("canonical"), "")
            if canonical:
                alias_to_group[normalize_token(canonical)] = group_name

            for alias in meta.get("aliases", []) or []:
                token = normalize_token(alias)
                if token:
                    alias_to_group[token] = group_name

    return payload, alias_to_group


def normalize_log_group(log_name: str, alias_to_group: dict[str, str]) -> str:
    token = normalize_token(log_name)
    if token in alias_to_group:
        return alias_to_group[token]

    compact = token.replace(" ", "")
    for alias, group in alias_to_group.items():
        if alias.replace(" ", "") == compact:
            return group

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
