#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DHW_CSV = Path("output/dhw_stochastic_model/dhw_tank_input_2024_scaled_2000.csv")
DEFAULT_WEATHER_CSV = Path("data/FullDS_Findhorn_clean.csv")
DEFAULT_OUTPUT_CSV = Path("output/forecast_input.csv")

TIME_CANDIDATES = ["timestamp", "Time", "time"]
T_OUT_CANDIDATES = ["t_out_c", "T_out [C]", "Tout", "outdoor_temp_c"]
T_AMB_CANDIDATES = ["t_amb_c", "T_amb [C]", "ambient_c", "ambient_temp_c"]
ST_POWER_CANDIDATES = ["st_power_kw", "ST Power [kW]", "solar_power_kw"]
ST_KWH_CANDIDATES = ["st_kwh", "solar_kwh", "st_energy_kwh"]
ENCODINGS = ["utf-8", "cp1252", "latin-1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a forecast input CSV by combining synthetic DHW with historical weather/solar signals."
    )
    parser.add_argument("--dhw-csv", type=Path, default=DEFAULT_DHW_CSV)
    parser.add_argument("--weather-csv", type=Path, default=DEFAULT_WEATHER_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--dhw-mode",
        choices=["size", "energy", "both"],
        default="both",
        help="Which DHW demand fields to keep in output.",
    )
    parser.add_argument(
        "--include-ambient",
        action="store_true",
        help="Include t_amb_c in output when available.",
    )
    parser.add_argument(
        "--include-solar",
        action="store_true",
        help="Include st_power_kw and/or st_kwh in output when available.",
    )
    return parser.parse_args()


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def read_csv_with_encoding_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Unable to decode {path} with encodings {ENCODINGS}. Last error: {last_error}",
    )


def parse_timestamp(df: pd.DataFrame, preferred_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    time_col = preferred_col if preferred_col in out.columns else first_present(out, TIME_CANDIDATES)
    if time_col is None:
        raise ValueError("No timestamp column found.")

    # Day-first parsing handles historical CSV format DD/MM/YYYY HH:MM.
    out["timestamp"] = pd.to_datetime(out[time_col], errors="coerce", dayfirst=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def build_forecast_input(
    dhw: pd.DataFrame,
    weather: pd.DataFrame,
    dhw_mode: str,
    include_ambient: bool,
    include_solar: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    source_map: dict[str, str] = {}

    # Base output timeline comes from synthetic DHW input.
    out = dhw[["timestamp"]].copy()

    if dhw_mode in {"size", "both"} and "dhw_draw_size_l" in dhw.columns:
        out["dhw_draw_size_l"] = pd.to_numeric(dhw["dhw_draw_size_l"], errors="coerce").fillna(0.0)
        source_map["dhw_draw_size_l"] = "dhw"
    if dhw_mode in {"energy", "both"} and "dhw_draw_energy_kwh" in dhw.columns:
        out["dhw_draw_energy_kwh"] = pd.to_numeric(dhw["dhw_draw_energy_kwh"], errors="coerce").fillna(0.0)
        source_map["dhw_draw_energy_kwh"] = "dhw"

    if ("dhw_draw_size_l" not in out.columns) and ("dhw_draw_energy_kwh" not in out.columns):
        raise ValueError("DHW CSV does not include requested demand columns.")

    w = weather.copy()

    t_out_col = first_present(w, T_OUT_CANDIDATES)
    if t_out_col is None:
        raise ValueError("Weather CSV must include an outdoor temperature column (t_out_c / T_out [C]).")

    keep_cols = ["timestamp", t_out_col]
    source_map["t_out_c"] = t_out_col

    t_amb_col = first_present(w, T_AMB_CANDIDATES)
    if include_ambient and t_amb_col is not None:
        keep_cols.append(t_amb_col)
        source_map["t_amb_c"] = t_amb_col

    if include_solar:
        st_power_col = first_present(w, ST_POWER_CANDIDATES)
        st_kwh_col = first_present(w, ST_KWH_CANDIDATES)
        if st_power_col is not None:
            keep_cols.append(st_power_col)
            source_map["st_power_kw"] = st_power_col
        if st_kwh_col is not None:
            keep_cols.append(st_kwh_col)
            source_map["st_kwh"] = st_kwh_col

    w = w[keep_cols].copy()
    rename_map = {t_out_col: "t_out_c"}
    if include_ambient and t_amb_col is not None:
        rename_map[t_amb_col] = "t_amb_c"
    if include_solar:
        if "st_power_kw" in source_map:
            rename_map[source_map["st_power_kw"]] = "st_power_kw"
        if "st_kwh" in source_map:
            rename_map[source_map["st_kwh"]] = "st_kwh"
    w = w.rename(columns=rename_map)

    merged = out.merge(w, on="timestamp", how="left")

    # Keep basic weather continuity simple and explicit.
    merged["t_out_c"] = pd.to_numeric(merged["t_out_c"], errors="coerce")
    merged["t_out_c"] = merged["t_out_c"].interpolate(limit_direction="both")

    if "t_amb_c" in merged.columns:
        merged["t_amb_c"] = pd.to_numeric(merged["t_amb_c"], errors="coerce")
        merged["t_amb_c"] = merged["t_amb_c"].interpolate(limit_direction="both")

    if "st_power_kw" in merged.columns:
        merged["st_power_kw"] = pd.to_numeric(merged["st_power_kw"], errors="coerce").fillna(0.0)
    if "st_kwh" in merged.columns:
        merged["st_kwh"] = pd.to_numeric(merged["st_kwh"], errors="coerce").fillna(0.0)

    return merged, source_map


def main() -> None:
    args = parse_args()

    dhw_raw, dhw_enc = read_csv_with_encoding_fallback(args.dhw_csv)
    weather_raw, weather_enc = read_csv_with_encoding_fallback(args.weather_csv)

    dhw = parse_timestamp(dhw_raw, preferred_col="timestamp")
    weather = parse_timestamp(weather_raw)

    forecast, source_map = build_forecast_input(
        dhw=dhw,
        weather=weather,
        dhw_mode=args.dhw_mode,
        include_ambient=args.include_ambient,
        include_solar=args.include_solar,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)

    print("Forecast input CSV created")
    print(f"DHW source: {args.dhw_csv}")
    print(f"DHW encoding: {dhw_enc}")
    print(f"Weather source: {args.weather_csv}")
    print(f"Weather encoding: {weather_enc}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(forecast)}")
    print("Output columns:")
    for c in forecast.columns:
        src = source_map.get(c, "derived")
        print(f"  - {c} (from {src})")


if __name__ == "__main__":
    main()
