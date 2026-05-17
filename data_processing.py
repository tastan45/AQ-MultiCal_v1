import re
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

from config import POLLUTANT_DISPLAY_LABELS, POLLUTANT_DISPLAY_UNITS


# -----------------------------------------------------------------------------
# Robust CSV loading utilities
# -----------------------------------------------------------------------------

def _rewind_file(file_obj):
    """Reset Streamlit UploadedFile / file-like object before repeated reads."""
    try:
        file_obj.seek(0)
    except Exception:
        pass


def _read_csv_auto(file_obj) -> pd.DataFrame:
    """Read CSV files with common delimiters automatically and efficiently.

    Supports comma, semicolon, tab, and pipe-delimited files. The delimiter is
    inferred from a small sample first, so large CSV files are not parsed several
    times.
    """
    if file_obj is None:
        return pd.DataFrame()

    _rewind_file(file_obj)
    sample_text = ""
    try:
        sample = file_obj.read(8192)
        if isinstance(sample, bytes):
            sample_text = sample.decode("utf-8-sig", errors="ignore")
        else:
            sample_text = str(sample)
    finally:
        _rewind_file(file_obj)

    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    delimiters = [";", ",", "\t", "|"]
    sep = max(delimiters, key=lambda d: first_line.count(d)) if first_line else ","

    try:
        return pd.read_csv(file_obj, sep=sep)
    except Exception:
        # Last resort for unusual files. This is slower but more flexible.
        _rewind_file(file_obj)
        return pd.read_csv(file_obj, sep=None, engine="python")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", str(col).replace("\ufeff", "").strip().lower()
               .replace(" ", "_").replace("-", "_").replace(".", ""))
        for col in df.columns
    ]
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking object columns, including decimal-comma values."""
    df = df.copy()
    for col in df.columns:
        if col == "timestamp":
            continue
        if df[col].dtype == "object":
            cleaned = df[col].astype(str).str.strip()
            # Convert decimal comma only when it looks like a decimal separator.
            cleaned = cleaned.str.replace(",", ".", regex=False)
            converted = pd.to_numeric(cleaned, errors="coerce")
            # Keep conversion if at least half of non-empty values are numeric.
            non_empty = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna()
            if len(non_empty) == 0 or converted.notna().sum() >= max(1, int(0.5 * len(non_empty))):
                df[col] = converted
        else:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df


def _prepare_basic_df(file_obj) -> pd.DataFrame:
    df = _read_csv_auto(file_obj)
    df = _normalize_columns(df)

    timestamp_candidates = [
        c for c in df.columns
        if c == "timestamp" or "timestamp" in c or c in {"time", "date", "datetime", "tarih", "zaman"}
    ]
    if not timestamp_candidates:
        raise ValueError("No timestamp/date/time column was found in the uploaded file.")

    time_col = timestamp_candidates[0]
    if time_col != "timestamp":
        df = df.rename(columns={time_col: "timestamp"})

    # Robust timestamp parsing.
    # IMPORTANT:
    # Files formatted as YYYY-MM-DD must NOT be parsed with dayfirst=True.
    # Otherwise dates such as 2025-01-11 may be interpreted as 2025-11-01.
    timestamp_text = df["timestamp"].astype(str).str.strip()
    non_empty_ts = timestamp_text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna()

    iso_like_ratio = 0.0
    if len(non_empty_ts) > 0:
        iso_like_ratio = non_empty_ts.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean()

    if iso_like_ratio >= 0.8:
        # ISO-like format: YYYY-MM-DD or YYYY/MM/DD
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=False)
    else:
        # TR/EU-like formats such as DD.MM.YYYY or DD/MM/YYYY
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
        if parsed.notna().sum() == 0:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=False)

    df["timestamp"] = parsed
    df = df.dropna(subset=["timestamp"])
    df = _convert_numeric_columns(df)
    return df


def _is_reference_col(col: str) -> bool:
    return (
        col.startswith("reference_")
        or col.startswith("ref_")
        or col.endswith("_reference")
        or col.endswith("_ref")
        or "reference" in col
        or col in {"ref", "baseline", "reference"}
    )


def _is_temperature_col(col: str) -> bool:
    return any(token in col for token in ["temp", "temperature", "sicaklik", "sıcaklık"])


def _is_humidity_col(col: str) -> bool:
    return any(token in col for token in ["hum", "humidity", "nem"])


def _detect_reference_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c != "timestamp" and _is_reference_col(c)]
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_candidates:
        return numeric_candidates[0]
    if candidates:
        return candidates[0]
    raise ValueError("No reference column was found. Use a name such as reference_CO2, CO2_Reference, reference_PM25, or PM25_Reference.")


def _infer_pollutant_from_reference(ref_col: str) -> str:
    pollutant = ref_col
    pollutant = re.sub(r"(^reference_|^ref_|_reference$|_ref$|reference|ref)", "", pollutant).strip("_")
    if not pollutant:
        pollutant = "pollutant"
    return pollutant.upper().replace("PM2_5", "PM25").replace("PM25", "PM25")


def _derive_location_from_raw_col(raw_col: str, pollutant_name: str) -> str:
    loc = raw_col
    tokens = ["raw", "sensor", pollutant_name.lower(), "pollutant", "value", "measurement"]
    for token in tokens:
        loc = re.sub(rf"(^{token}_|_{token}$|{token})", "", loc)
    loc = loc.strip("_")
    return loc or "sensor"


def _detect_raw_cols(df: pd.DataFrame, ref_col: str) -> List[str]:
    excluded = {"timestamp", ref_col}
    numeric_cols = [
        c for c in df.columns
        if c not in excluded
        and not _is_temperature_col(c)
        and not _is_humidity_col(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    preferred = [c for c in numeric_cols if any(token in c for token in ["raw", "sensor", "lcs", "lowcost", "pollutant"])]
    return preferred if preferred else numeric_cols


def _single_file_to_long_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """Convert one-file format to platform long format.

    Supported examples:
      timestamp;CO2_Reference;CO2_Raw
      timestamp,reference_CO2,kitchen_CO2,livingroom_CO2
      timestamp,reference_PM25,raw_PM25,temp,hum
    """
    ref_col = _detect_reference_col(df)
    pollutant_name_upper = _infer_pollutant_from_reference(ref_col)
    pollutant_label = POLLUTANT_DISPLAY_LABELS.get(pollutant_name_upper, pollutant_name_upper)
    display_unit = POLLUTANT_DISPLAY_UNITS.get(pollutant_name_upper, pollutant_name_upper)

    raw_cols = _detect_raw_cols(df, ref_col)
    if not raw_cols:
        raise ValueError("No raw sensor column was found. Use a name such as CO2_Raw, raw_CO2, sensor_CO2, or provide one numeric raw column besides the reference column.")

    temp_cols = [c for c in df.columns if _is_temperature_col(c) and pd.api.types.is_numeric_dtype(df[c])]
    hum_cols = [c for c in df.columns if _is_humidity_col(c) and pd.api.types.is_numeric_dtype(df[c])]
    default_temp = temp_cols[0] if temp_cols else None
    default_hum = hum_cols[0] if hum_cols else None

    records = []
    base = df[["timestamp", ref_col] + raw_cols + ([default_temp] if default_temp else []) + ([default_hum] if default_hum else [])].copy()

    for raw_col in raw_cols:
        loc = _derive_location_from_raw_col(raw_col, pollutant_name_upper)
        temp_col = None
        hum_col = None
        # Prefer location-specific temp/humidity if available, otherwise global one.
        loc_prefix = loc.split("_")[0]
        loc_temp_candidates = [c for c in temp_cols if c.startswith(loc_prefix)]
        loc_hum_candidates = [c for c in hum_cols if c.startswith(loc_prefix)]
        temp_col = loc_temp_candidates[0] if loc_temp_candidates else default_temp
        hum_col = loc_hum_candidates[0] if loc_hum_candidates else default_hum

        df_loc = pd.DataFrame({
            "timestamp": base["timestamp"],
            "reference_pollutant": base[ref_col],
            "raw_pollutant": base[raw_col],
            # If temp/humidity are not supplied, use a neutral constant so the
            # existing model pipeline still works even when checkboxes are on.
            "raw_temp": base[temp_col] if temp_col else 0.0,
            "raw_humidity": base[hum_col] if hum_col else 0.0,
            "location": loc,
        })
        records.append(df_loc)

    out = pd.concat(records, ignore_index=True)
    out = out.dropna(subset=["timestamp", "reference_pollutant", "raw_pollutant"])
    return out, pollutant_label, display_unit


# -----------------------------------------------------------------------------
# Main platform function
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner="Merging data files...")
def merge_and_prepare_data(pollutant_data, temp_data=None, hum_data=None):
    """Prepare uploaded data for AQ-MultiCal.

    Supports both:
      1) Legacy 3-file format: pollutant + temperature + humidity CSVs.
      2) Single-file format: timestamp + one reference column + one/multiple raw sensor columns
         with optional temperature/humidity columns.
    """
    df_p = _prepare_basic_df(pollutant_data)

    # Single-file mode: no separate temp/humidity files are supplied.
    if temp_data is None or hum_data is None:
        return _single_file_to_long_format(df_p)

    # Legacy 3-file mode, now with robust CSV/date/numeric parsing.
    df_t = _prepare_basic_df(temp_data)
    df_h = _prepare_basic_df(hum_data)

    for df in [df_p, df_t, df_h]:
        # Standardize common environmental column names while preserving location prefixes.
        df.rename(columns=lambda x: re.sub(r'temp(erature)?', 'temp', x), inplace=True)
        df.rename(columns=lambda x: re.sub(r'hum(idity)?', 'hum', x), inplace=True)

    ref_col = _detect_reference_col(df_p)
    pollutant_name_upper = _infer_pollutant_from_reference(ref_col)
    pollutant_name_raw = pollutant_name_upper.lower()
    pollutant_label = POLLUTANT_DISPLAY_LABELS.get(pollutant_name_upper, pollutant_name_upper)
    display_unit = POLLUTANT_DISPLAY_UNITS.get(pollutant_name_upper, pollutant_name_upper)

    df_p = df_p.rename(columns={ref_col: "reference_pollutant"})
    # Convert raw pollutant columns to *_pollutant for legacy location matching.
    rename_map = {}
    for col in df_p.columns:
        if col in {"timestamp", "reference_pollutant"}:
            continue
        if pd.api.types.is_numeric_dtype(df_p[col]) and not _is_temperature_col(col) and not _is_humidity_col(col):
            rename_map[col] = col.replace(f"_{pollutant_name_raw}", "_pollutant").replace(f"{pollutant_name_raw}_", "")
            if not rename_map[col].endswith("_pollutant"):
                rename_map[col] = f"{rename_map[col].strip('_')}_pollutant"
    df_p = df_p.rename(columns=rename_map)

    df_merged = df_p.merge(df_t, on='timestamp', how='inner').merge(df_h, on='timestamp', how='inner')

    # Pollutant locations must be inferred from pollutant columns only.
    # Environmental files may be provided as global columns such as temp_raw / hum_raw
    # rather than location-prefixed columns. If locations are inferred from all columns,
    # temperature/humidity can be missed for single-sensor files.
    pollutant_cols = [
        col for col in df_merged.columns
        if col.endswith('_pollutant') and col != 'reference_pollutant'
    ]
    locations = sorted(list(set([col.rsplit('_pollutant', 1)[0] for col in pollutant_cols])))

    # Detect global environmental columns from separate TEMP/HUM files.
    # Examples after normalization: temp_raw, temperature_raw -> temp_raw;
    # hum_raw, humidity_raw -> hum_raw. Reference temp/hum columns are ignored.
    global_temp_candidates = [
        c for c in df_merged.columns
        if c != 'timestamp'
        and not _is_reference_col(c)
        and pd.api.types.is_numeric_dtype(df_merged[c])
        and _is_temperature_col(c)
    ]
    global_hum_candidates = [
        c for c in df_merged.columns
        if c != 'timestamp'
        and not _is_reference_col(c)
        and pd.api.types.is_numeric_dtype(df_merged[c])
        and _is_humidity_col(c)
    ]

    # Prefer raw environmental columns over reference environmental columns.
    global_temp_candidates = sorted(global_temp_candidates, key=lambda c: (0 if 'raw' in c else 1, c))
    global_hum_candidates = sorted(global_hum_candidates, key=lambda c: (0 if 'raw' in c else 1, c))
    global_temp_col = global_temp_candidates[0] if global_temp_candidates else None
    global_hum_col = global_hum_candidates[0] if global_hum_candidates else None

    all_locations_data = []
    for loc in locations:
        pollutant_col = f'{loc}_pollutant'

        # Prefer location-specific environmental columns if they exist;
        # otherwise use global temp/humidity columns.
        location_temp_candidates = [c for c in global_temp_candidates if c.startswith(f'{loc}_')]
        location_hum_candidates = [c for c in global_hum_candidates if c.startswith(f'{loc}_')]
        temp_col = location_temp_candidates[0] if location_temp_candidates else global_temp_col
        hum_col = location_hum_candidates[0] if location_hum_candidates else global_hum_col

        if pollutant_col not in df_merged.columns:
            continue
        df_loc = pd.DataFrame({
            'timestamp': df_merged['timestamp'],
            'reference_pollutant': df_merged['reference_pollutant'],
            'raw_pollutant': df_merged[pollutant_col],
            'raw_temp': df_merged[temp_col] if temp_col in df_merged.columns else 0.0,
            'raw_humidity': df_merged[hum_col] if hum_col in df_merged.columns else 0.0,
            'location': loc,
        })
        all_locations_data.append(df_loc)

    if not all_locations_data:
        # Fall back to single-file interpretation if legacy location matching fails.
        return _single_file_to_long_format(df_p.rename(columns={"reference_pollutant": ref_col}))

    df_final_long = pd.concat(all_locations_data, ignore_index=True)
    return df_final_long.dropna(subset=['timestamp', 'reference_pollutant', 'raw_pollutant']), pollutant_label, display_unit
