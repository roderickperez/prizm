from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


SOURCE_FILE = Path(__file__).resolve().parent / "2026_02_02_Checkshots_to_be_loaded_into_Petrel_LFA.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "wells"


REQUIRED_COLUMNS = ["Well Name", "MD (ft)", "TWT (msec)"]
OUTPUT_COLUMNS = ["Well Name", "Measured Depth from KB (ft)", "Raw pick time (ms)"]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown_well"


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {SOURCE_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(SOURCE_FILE)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=["Well Name", "MD (ft)", "TWT (msec)"])
    df["Well Name"] = df["Well Name"].astype(str).str.strip()
    df["MD (ft)"] = pd.to_numeric(df["MD (ft)"], errors="coerce")
    df["TWT (msec)"] = pd.to_numeric(df["TWT (msec)"], errors="coerce")
    df = df.dropna(subset=["MD (ft)", "TWT (msec)"])

    df = df.rename(columns={
        "MD (ft)": "Measured Depth from KB (ft)",
        "TWT (msec)": "Raw pick time (ms)",
    })

    grouped = df.groupby("Well Name", sort=True)
    exported = 0

    for well_name, well_df in grouped:
        well_df = well_df[OUTPUT_COLUMNS].sort_values(by="Measured Depth from KB (ft)").reset_index(drop=True)
        stem = safe_name(well_name)

        csv_path = OUTPUT_DIR / f"{stem}.csv"
        txt_path = OUTPUT_DIR / f"{stem}.txt"

        well_df.to_csv(csv_path, index=False)
        well_df.to_csv(txt_path, index=False, sep="\t")
        exported += 1

    print(f"Exported {exported} wells to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
