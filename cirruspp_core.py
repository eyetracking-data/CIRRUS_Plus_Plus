from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, savgol_filter, welch
from scipy.spatial.distance import jensenshannon
from scipy.stats import kurtosis, median_abs_deviation, skew, wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

PREPROCESSING_SETTINGS = {
    "non_finite_signals": "Treat positive and negative infinity as missing (NaN).",
    "knn_distance_scaling": "StandardScaler per signal on observed values; return imputed values in original units and preserve observed values.",
    "moving_median_boundary": "Extend each finite segment with its nearest endpoint values (no zero padding).",
}

# ---------------------------------------------------------------------------
# Data loading and domain inference
# ---------------------------------------------------------------------------
MISSING_TOKENS = {
    "", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none",
    "-", "--", "---", "?", "nan", "NaN", "#VALUE!", "Inf", "-Inf",
    "infinity", "Infinity",
}

ID_PATTERNS = [
    r"^unnamed", r"^id$", r"participant", r"subject", r"trial", r"session",
    r"export", r"stimulus", r"condition", r"label", r"target", r"class",
    r"^row$", r"index", r"^sample$", r"sample_?id", r"^frame$", r"tracking ratio",
    r"^missing$", r"^validity", r"^is_valid$", r"^valid$",
]
TIME_PATTERNS = [
    r"^timestamp$", r"time", r"recordingtime", r"recording_time", r"device_time",
    r"system_time", r"elapsed", r"duration",
    r"date", r"clock",
]
DOMAIN_KEYWORDS = {
    "eye_tracking": ("gaze", "pupil", "fixation", "saccade", "por", "eye"),
    "industrial": ("temperature", "pressure", "vibration", "flow", "rpm", "machine"),
    "energy": ("load", "power", "energy", "solar", "consumption", "demand"),
    "physiological": ("ecg", "eeg", "ppg", "heart", "pulse", "resp", "eda", "motion"),
    "finance": ("price", "return", "volume", "open", "close", "high", "low"),
    "network": ("cpu", "latency", "request", "error", "traffic", "memory", "packet"),
}

DOMAIN_LABELS = {
    "eye_tracking": "Eye-tracking",
    "industrial": "Industrial sensors",
    "energy": "Energy",
    "physiological": "Physiological / wearable",
    "finance": "Finance",
    "network": "Network / system monitoring",
}

DOMAIN_CONFIG = {
    "eye_tracking": {
        "description": "Preserve gaze trajectories, saccades, short-gap continuity, pupil dynamics, and physical plausibility.",
        "quality_weights": {"completeness": .23, "continuity": .24, "plausibility": .17, "distribution": .12, "stability": .16, "correlation": .08},
        "loss_weights": {"reconstruction": .22, "clean_harm": .15, "dynamics": .20, "distribution": .08, "frequency": .05, "event": .15, "detection": .10, "retention": .05},
    },
    "industrial": {
        "description": "Preserve operating-state transitions, detect sensor faults, and avoid hiding drift or saturation.",
        "quality_weights": {"completeness": .17, "continuity": .16, "plausibility": .23, "distribution": .08, "stability": .25, "correlation": .11},
        "loss_weights": {"reconstruction": .15, "clean_harm": .10, "dynamics": .10, "distribution": .05, "frequency": .10, "event": .20, "detection": .20, "retention": .10},
    },
    "energy": {
        "description": "Preserve daily and weekly seasonality, genuine demand peaks, and longer-term load structure.",
        "quality_weights": {"completeness": .20, "continuity": .16, "plausibility": .12, "distribution": .10, "stability": .30, "correlation": .12},
        "loss_weights": {"reconstruction": .15, "clean_harm": .15, "dynamics": .10, "distribution": .05, "frequency": .25, "event": .20, "detection": .05, "retention": .05},
    },
    "physiological": {
        "description": "Preserve local waveform, quasi-periodicity, amplitude variation, and activity-related events.",
        "quality_weights": {"completeness": .17, "continuity": .22, "plausibility": .12, "distribution": .09, "stability": .30, "correlation": .10},
        "loss_weights": {"reconstruction": .15, "clean_harm": .15, "dynamics": .20, "distribution": .05, "frequency": .25, "event": .15, "detection": .03, "retention": .02},
    },
    "finance": {
        "description": "Preserve heavy tails, volatility clusters, and genuine jumps; aggressive outlier removal is especially risky.",
        "quality_weights": {"completeness": .14, "continuity": .10, "plausibility": .08, "distribution": .30, "stability": .30, "correlation": .08},
        "loss_weights": {"reconstruction": .05, "clean_harm": .25, "dynamics": .10, "distribution": .20, "frequency": .00, "event": .30, "detection": .05, "retention": .05},
    },
    "network": {
        "description": "Preserve bursts, incidents, regime changes, and service-level peaks while detecting measurement failures.",
        "quality_weights": {"completeness": .18, "continuity": .15, "plausibility": .20, "distribution": .07, "stability": .27, "correlation": .13},
        "loss_weights": {"reconstruction": .10, "clean_harm": .15, "dynamics": .10, "distribution": .05, "frequency": .05, "event": .25, "detection": .20, "retention": .10},
    },
}

# Clean real-eye-tracking winner frequencies from the documented quick pilot.
# These are domain-level context, NOT probabilities for the current uploaded file.
REAL_EYE_TRACKING_PILOT = {
    "imputation": [("linear", .85), ("locf", .125), ("mean", .025)],
    "outlier": [("zscore", .70), ("iqr", .20), ("mad", .10)],
    "smoothing": [("moving_median", .50), ("none", .30), ("butterworth", .15), ("savgol", .05)],
}

@dataclass(frozen=True)
class LoadedData:
    frame: pd.DataFrame
    separator: str
    decimal: str
    encoding: str
    source_name: str


def _normalise_object_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.replace(list(MISSING_TOKENS), pd.NA)


def auto_convert_numeric(df: pd.DataFrame, threshold: float = .80) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        # An identifier may be mostly numeric yet contain meaningful labels
        # such as "Unidentified(Pos)" / "Unidentified(Neg)". Converting it with
        # errors="coerce" would irreversibly merge distinct recordings.
        id_name = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if id_name in {"participant", "participantid", "subject", "subjectid",
                       "recording", "recordingid", "session", "sessionid"}:
            continue
        s = _normalise_object_series(out[col])
        if s.notna().sum() == 0:
            continue
        cleaned = (s.str.replace("\u00a0", "", regex=False)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace(r"(?<=\d)([a-zA-Z%]+)$", "", regex=True))
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted[s.notna()].notna().mean() >= threshold:
            out[col] = converted
    return out


def _decode_sample(data: bytes) -> tuple[str, str]:
    sample = data[:200_000]
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return sample.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return sample.decode("latin-1", errors="replace"), "latin-1"


def _infer_separator(sample: str, filename: str) -> str:
    if filename.lower().endswith(".tsv"):
        return "\t"
    try:
        return csv.Sniffer().sniff(sample[:50_000], delimiters=",;\t|").delimiter
    except csv.Error:
        lines = [line for line in sample.splitlines()[:20] if line.strip()]
        candidates = [",", ";", "\t", "|"]
        scores = {}
        for sep in candidates:
            counts = [line.count(sep) for line in lines]
            scores[sep] = (np.median(counts) if counts else 0, -np.std(counts) if counts else 0)
        return max(candidates, key=lambda sep: scores[sep])


def _infer_decimal(sample: str, separator: str) -> str:
    if separator == ",":
        return "."
    comma_decimal = len(re.findall(r"(?<!\d)\d+,\d+|\d+,\d+(?!\d)", sample[:100_000]))
    dot_decimal = len(re.findall(r"(?<!\d)\d+\.\d+|\d+\.\d+(?!\d)", sample[:100_000]))
    return "," if comma_decimal > dot_decimal else "."


def read_table_bytes(data: bytes, filename: str, separator: str = "Auto", decimal: str = "Auto") -> LoadedData:
    sample, encoding = _decode_sample(data)
    sep = _infer_separator(sample, filename) if separator == "Auto" else {"Tab": "\t"}.get(separator, separator)
    dec = _infer_decimal(sample, sep) if decimal == "Auto" else decimal
    try:
        frame = pd.read_csv(io.BytesIO(data), sep=sep, decimal=dec, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        encoding = "latin-1"
        frame = pd.read_csv(io.BytesIO(data), sep=sep, decimal=dec, encoding=encoding, low_memory=False)
    if frame.shape[1] < 2:
        frame = pd.read_csv(io.BytesIO(data), sep=None, engine="python", decimal=dec, encoding=encoding)
        sep = "auto"
    frame = auto_convert_numeric(frame)
    frame.columns = [str(c).strip() for c in frame.columns]
    return LoadedData(frame, sep, dec, encoding, filename)


def likely_time_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(re.search(p, str(c).lower()) for p in TIME_PATTERNS)]


def preferred_timestamp_column(df: pd.DataFrame) -> str | None:
    """Pick a clearly named, *validated* time column before prompting the user.

    Avoid treating an unlabelled row index as actual recording time. If no named
    column validates automatically (e.g. a unitless numeric 'timestamp'),
    suggest the named candidate and require the user to choose the unit.
    """
    candidates = likely_time_columns(df)
    for column in candidates:
        try:
            trial = canonical_frame(df, [], 1.0, column, "Auto")
            validate_timestamps(trial.timestamp)
            return column
        except (ValueError, TypeError, OverflowError):
            continue
    return candidates[0] if candidates else None


def numeric_signal_columns(df: pd.DataFrame, include_identifier_like: bool = False) -> list[str]:
    cols = list(df.select_dtypes(include=[np.number]).columns)
    if include_identifier_like:
        return cols
    result = []
    for col in cols:
        lower = str(col).lower()
        if any(re.search(p, lower) for p in ID_PATTERNS + TIME_PATTERNS):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        result.append(col)
    return result


def infer_domain(df: pd.DataFrame, signal_cols: list[str]) -> tuple[str, dict[str, int]]:
    text = " ".join(str(c).lower() for c in signal_cols + list(df.columns))
    scores = {d: sum(text.count(k) for k in keys) for d, keys in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return (best if scores[best] else "eye_tracking"), scores


def exact_duplicate_signal_groups(df: pd.DataFrame, columns: list[str]) -> list[list[str]]:
    """Find exactly equal numeric channels (including matching NaN positions).

    This is deliberately conservative: only channels that are numerically equal
    at every row are grouped. The user may still select them manually.
    """
    cols = [c for c in columns if c in df.columns]
    groups: list[list[str]] = []
    used: set[str] = set()
    arrays = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols}
    for i, c in enumerate(cols):
        if c in used:
            continue
        group = [c]
        for d in cols[i + 1:]:
            if d in used:
                continue
            if np.all(np.isclose(arrays[c], arrays[d], equal_nan=True)):
                group.append(d); used.add(d)
        if len(group) > 1:
            groups.append(group); used.update(group)
    return groups


def distinct_signal_defaults(df: pd.DataFrame, columns: list[str], max_n: int = 5) -> list[str]:
    """Choose default signals while skipping exact duplicates."""
    chosen: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for c in columns:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        duplicate = False
        for kept in chosen:
            if np.all(np.isclose(arr, arrays[kept], equal_nan=True)):
                duplicate = True
                break
        if not duplicate:
            chosen.append(c); arrays[c] = arr
        if len(chosen) >= max_n:
            break
    return chosen

def _eye_side_columns(df: pd.DataFrame, side: str) -> dict[str, str | None]:
    low = {str(c).lower(): str(c) for c in df.columns}
    side_low = side.lower()
    def pick(tokens: tuple[str, ...], suffix: str | None = None):
        for c in df.columns:
            s = str(c).lower()
            if side_low not in s:
                continue
            if all(t in s for t in tokens) and (suffix is None or suffix in s):
                return str(c)
        return None
    # Prefer explicit Point-of-Regard channels.
    x = pick(("point of regard",), " x") or pick(("gaze",), " x")
    y = pick(("point of regard",), " y") or pick(("gaze",), " y")
    category = pick(("category",))
    pupil = pick(("pupil",))
    return {"x": x, "y": y, "category": category, "pupil": pupil}

def eye_tracking_validity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise transparent gaze-availability cues without modifying data."""
    rows = []
    n = max(1, len(df))
    for side in ("Left", "Right"):
        cols = _eye_side_columns(df, side)
        xcol, ycol, ccol = cols["x"], cols["y"], cols["category"]
        if not xcol or not ycol:
            continue
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        zero_pair = x.eq(0) & y.eq(0)
        explicit_missing = x.isna() | y.isna()
        blink = pd.Series(False, index=df.index)
        if ccol:
            blink = df[ccol].astype("string").str.strip().str.lower().str.contains("blink", na=False)
        domain_unavailable = zero_pair | blink | explicit_missing
        rows.append({
            "eye": side, "x_column": xcol, "y_column": ycol,
            "explicit_missing_pairs": int(explicit_missing.sum()),
            "paired_0_0": int(zero_pair.sum()), "paired_0_0_%": 100 * float(zero_pair.mean()),
            "blink_labels": int(blink.sum()), "blink_%": 100 * float(blink.mean()),
            "combined_unavailable": int(domain_unavailable.sum()),
            "combined_unavailable_%": 100 * float(domain_unavailable.mean()),
        })
    return pd.DataFrame(rows)

def apply_eye_tracking_validity(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Interpret gaze availability in an explicit, reversible analysis copy.

    Raw input is never overwritten. Only paired (0,0) gaze coordinates and,
    in the domain-aware mode, same-eye Blink-labelled gaze coordinates are set
    to NaN. A single x=0 or y=0 is never treated as missing by itself.
    """
    out = df.copy()
    meta: dict[str, Any] = {"mode": mode, "affected": []}
    if mode == "Raw numeric values":
        return out, meta
    for side in ("Left", "Right"):
        cols = _eye_side_columns(out, side)
        xcol, ycol, ccol = cols["x"], cols["y"], cols["category"]
        if not xcol or not ycol:
            continue
        x = pd.to_numeric(out[xcol], errors="coerce")
        y = pd.to_numeric(out[ycol], errors="coerce")
        mask = x.eq(0) & y.eq(0)
        reasons = {"paired_0_0": int(mask.sum())}
        if mode == "Domain-aware gaze validity (0,0 + Blink labels)" and ccol:
            blink = out[ccol].astype("string").str.strip().str.lower().str.contains("blink", na=False)
            reasons["blink_labels"] = int(blink.sum())
            mask = mask | blink
        before_missing = int(out[[xcol, ycol]].isna().any(axis=1).sum())
        out.loc[mask, [xcol, ycol]] = np.nan
        after_missing = int(out[[xcol, ycol]].isna().any(axis=1).sum())
        meta["affected"].append({"eye": side, "x": xcol, "y": ycol, **reasons,
                                 "unavailable_pairs_after": after_missing,
                                 "newly_interpreted_pairs": max(0, after_missing-before_missing)})
    return out, meta


def rank_signal_columns(columns: list[str], domain: str) -> list[str]:
    priorities = {
        "eye_tracking": ("gaze point x", "gaze point y", "point of regard", "gaze_x", "gaze_y", "pupil", "gaze", "fixation", "saccade"),
        "industrial": ("temperature", "pressure", "vibration", "flow", "rpm"),
        "energy": ("load", "power", "consumption", "demand", "solar", "temperature"),
        "physiological": ("ecg", "ppg", "pulse", "heart", "resp", "eda", "sensor", "motion"),
        "finance": ("log_return", "return", "price", "volume", "close", "open"),
        "network": ("request", "cpu", "latency", "error", "traffic", "memory"),
    }.get(domain, ())
    def score(c: str):
        low = str(c).lower().replace(" ", "_")
        for i, token in enumerate(priorities):
            if token.replace(" ", "_") in low:
                return i, len(c), low
        return len(priorities) + 1, len(c), low
    return sorted(columns, key=score)


TIMESTAMP_UNITS = {
    "Auto": None,
    "Seconds (s)": (1., "seconds"),
    "Milliseconds (ms)": (1e-3, "milliseconds"),
    "Microseconds (µs)": (1e-6, "microseconds"),
    "Nanoseconds (ns)": (1e-9, "nanoseconds"),
    "Minutes (min)": (60., "minutes"),
    "Hours (h)": (3600., "hours"),
    "Days (d)": (86400., "days"),
    "Date / time": None,
}


def _timestamp_values(series: pd.Series, unit_choice: str):
    """Parse date/time separately from numeric timestamps; retain unavailable rows."""
    if unit_choice not in TIMESTAMP_UNITS:
        raise ValueError(f"Unknown timestamp unit {unit_choice!r}")
    numeric = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series.dtype)
    looks_like_date = series.astype(str).str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}", regex=True).any()
    if unit_choice == "Date / time" or (unit_choice == "Auto" and (is_datetime or (numeric.notna().sum() < .8*series.notna().sum() and looks_like_date))):
        # Numeric clocks must never be accidentally read as Unix nanoseconds.
        source = series if is_datetime else series.astype(str).where(pd.to_numeric(series,errors="coerce").isna())
        parsed = pd.to_datetime(source, format="mixed", errors="coerce", utc=True).reset_index(drop=True)
        valid = parsed.dropna()
        origin = valid.iloc[0] if len(valid) else pd.NaT
        values = (parsed-origin).dt.total_seconds()
        return values, {"origin": origin.isoformat() if pd.notna(origin) else None,
                        "format": "date_time", "timezone_interpretation": "Explicit offsets converted to UTC; dates without a timezone assumed UTC. Ambiguous numeric dates use month-first parsing; ISO 8601 is recommended."}
    return numeric.where(np.isfinite(numeric)), {}


def _timestamp_scale(dt: float, name: str, unit_choice: str = "Auto") -> tuple[float, str]:
    if unit_choice not in TIMESTAMP_UNITS:
        raise ValueError(f"Unknown timestamp unit {unit_choice!r}")
    if unit_choice != "Auto":
        return TIMESTAMP_UNITS[unit_choice]
    name = str(name).lower()
    if "micro" in name or re.search(r"(?:^|[^a-z])(?:us|usec|µs|μs)(?:$|[^a-z])", name):
        mult, unit = 1e-6, "microseconds"
    elif "nano" in name or re.search(r"(?:^|[^a-z])(?:ns|nsec)(?:$|[^a-z])", name):
        mult, unit = 1e-9, "nanoseconds"
    elif "milli" in name or re.search(r"(?:^|[^a-z])(?:ms|msec)(?:$|[^a-z])", name):
        mult, unit = 1e-3, "milliseconds"
    elif re.search(r"minute|(?:^|_)min(?:$|_)",name):
        mult, unit = 60., "minutes"
    elif re.search(r"hour|(?:^|_)h(?:$|_)",name):
        mult, unit = 3600., "hours"
    elif re.search(r"day|(?:^|_)d(?:$|_)",name):
        mult, unit = 86400., "days"
    elif re.search(r"second|(?:^|[^a-z])(?:s|sec)(?:$|[^a-z])",name):
        mult, unit = 1.0, "seconds"
    else:
        raise ValueError("Timestamp unit is ambiguous. Select Seconds, Milliseconds, Microseconds or another explicit unit; it cannot be determined from the interval alone.")
    return mult, unit


def infer_sampling_rate(df: pd.DataFrame, timestamp_col: str | None, timestamp_unit: str = "Auto") -> tuple[float | None, str]:
    if not timestamp_col or timestamp_col not in df.columns:
        return None, "No timestamp selected"
    if timestamp_unit == "Auto" and df.attrs.get("timestamp_unit"):
        timestamp_unit = df.attrs["timestamp_unit"]
    values, parsed_meta = _timestamp_values(df[timestamp_col], timestamp_unit)
    s = values.dropna()
    if len(s) < 2:
        return None, "Too few timestamp values"
    diffs = s.diff().dropna().to_numpy(float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) < 1:
        return None, "No stable positive timestamp increments"
    dt = float(np.median(diffs))
    mult, unit = (1., "date/time") if parsed_meta else _timestamp_scale(dt, timestamp_col, timestamp_unit)
    fs = 1.0 / (dt * mult)
    if not (1e-9 <= fs <= 10000):
        return None, f"Inferred rate {fs:g} Hz is outside the supported range"
    source = "auto-detected unit" if timestamp_unit == "Auto" else "manually selected unit"
    return float(fs), f"Inferred from median timestamp step ({unit}; {source})"


def canonical_frame(df: pd.DataFrame, signal_cols: list[str], fs: float, timestamp_col: str | None = None, timestamp_unit: str = "Auto") -> pd.DataFrame:
    """Use actual timestamps only. fs is retained for call compatibility, never as a fallback."""
    if timestamp_col is None:
        candidates = likely_time_columns(df)
        timestamp_col = candidates[0] if candidates else None
    if timestamp_col is None or timestamp_col not in df:
        raise ValueError("Select a timestamp column. A recording with usable timestamps is required.")
    if timestamp_unit == "Auto" and df.attrs.get("timestamp_unit"):
        timestamp_unit = df.attrs["timestamp_unit"]
    times = None
    metadata = {}
    if timestamp_col is not None and timestamp_col in df:
        numeric, parsed_meta = _timestamp_values(df[timestamp_col], timestamp_unit)
        finite = numeric.dropna()
        steps = finite.diff().dropna()
        positive = steps[steps > 0]
        if len(finite) >= 2 and len(positive):
            median_step = float(positive.median())
            scale, unit = (1., "date/time") if parsed_meta else _timestamp_scale(median_step, timestamp_col, timestamp_unit)
            # Subtract the origin before conversion to preserve precision for
            # large integer device clocks. Missing timestamps stay unavailable.
            times = (numeric - finite.iloc[0]) * scale
            metadata = {
                "source": "selected_timestamp", "column": timestamp_col,
                "unit_selection": timestamp_unit,
                "original_unit": unit, "seconds_per_original_unit": scale,
                "origin": finite.iloc[0].item() if hasattr(finite.iloc[0], "item") else finite.iloc[0],
                "missing_timestamps": int(numeric.isna().sum()),
                "non_increasing_steps": int((steps <= 0).sum()),
                "irregular_intervals": bool((np.abs(positive-median_step) > .05*median_step).any()),
                "description": "Selected timestamps converted to seconds relative to the first available timestamp. Original observation order, intervals and jumps are preserved; no resampling or timestamp filling.",
                **parsed_meta,
            }
        else:
            raise ValueError("The timestamp column needs at least two usable values and a positive time interval. Correct the column or unit; no artificial time axis is generated.")
    out = pd.DataFrame({"timestamp": times})
    for c in signal_cols:
        if c != "timestamp":
            # Normalize nullable pandas Int64/Float64 columns to NumPy float64.
            # Otherwise comparisons and Plotly markers can receive ambiguous pd.NA.
            numeric_signal = pd.to_numeric(df[c], errors="coerce")
            signal_values = numeric_signal.to_numpy(dtype=float, na_value=np.nan)
            signal_values[~np.isfinite(signal_values)] = np.nan
            out[c] = signal_values
    out.attrs["time_axis"] = metadata
    metadata["diagnostics"] = timestamp_diagnostics(out.timestamp)
    out.attrs["time_processing"] = {"max_imputation_gap_seconds": max(.5, 10 * metadata["diagnostics"]["median_interval_seconds"]), "irregular_smoothing": "disabled"}
    return out


def timestamp_diagnostics(timestamp: pd.Series) -> dict[str, Any]:
    times = timestamp.to_numpy(dtype=float)
    steps = np.diff(times)
    finite = np.isfinite(times)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    median = float(np.median(positive)) if len(positive) else np.nan
    effective_interval = median
    estimation = "inverse median of positive intervals"
    # Coarsely quantised device clocks can report alternating 1 and 2 ms
    # increments even when actual frequency is ~700 Hz. In this specific
    # pattern, 1/median would incorrectly produce 1000 Hz. Use a trimmed
    # mean in place of that biased median, excluding acquisition interruptions.
    # Full original timestamps and the *actual median* are not modified.
    if len(positive) > 100:
        q05, q80, q99 = np.quantile(positive, [.05, .80, .99])
        if q05 > 0 and median <= 2.05*q05 and q80 > 1.35*q05 and q99 <= 3.2*q05:
            clean_intervals = positive[positive <= q99]
            if clean_intervals.size:
                effective_interval = float(np.mean(clean_intervals))
                estimation = "trimmed-mean interval for coarsely quantised clock"
    return {"missing_timestamps": int((~finite).sum()),
            "duplicate_steps": int((steps == 0).sum()), "backward_steps": int((steps < 0).sum()),
            "median_interval_seconds": median,
            "effective_interval_seconds": effective_interval,
            "sampling_rate_estimation": estimation,
            "sampling_rate_hz": 1 / effective_interval if effective_interval > 0 else np.nan,
            "duration_seconds": float(times[-1] - times[0]) if len(times) and finite.all() else np.nan,
            "irregular_intervals": bool(len(positive) and np.any(np.abs(positive - median) > .05 * median)),
            "regularity_tolerance": .05,
            "valid": bool(len(times) >= 2 and finite.all() and np.all(steps > 0))}


def validate_timestamps(timestamp: pd.Series) -> dict[str, Any]:
    diagnostics = timestamp_diagnostics(timestamp)
    if not diagnostics["valid"]:
        raise ValueError(f"Correct the timestamp column before continuing: {diagnostics['missing_timestamps']} missing/invalid values, {diagnostics['duplicate_steps']} duplicate steps and {diagnostics['backward_steps']} backward steps. Timestamps must be complete and strictly increasing; split clock resets into separate recordings. Rows are not reordered or filled automatically.")
    return diagnostics

# ---------------------------------------------------------------------------
# Profiling and intrinsic quality index
# ---------------------------------------------------------------------------
def signal_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "timestamp"]


def _runs(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.array([], dtype=int)
    padded = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def _robust_outlier_rate(x: np.ndarray, threshold: float = 3.5) -> float:
    med = np.nanmedian(x)
    mad = median_abs_deviation(x, nan_policy="omit", scale=1.0)
    if not np.isfinite(mad) or mad <= 0:
        return 0.0
    rz = 0.67448975 * (x - med) / mad
    return float(np.nanmean(np.abs(rz) > threshold))


def _iqr_outlier_rate(x: np.ndarray, factor: float = 1.5) -> float:
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return 0.0
    return float(np.nanmean((x < q1 - factor * iqr) | (x > q3 + factor * iqr)))


def _spectral_entropy(x: np.ndarray, fs: float) -> tuple[float, float]:
    x = pd.Series(x).interpolate(limit_direction="both").fillna(0).to_numpy(float)
    if len(x) < 16 or np.nanstd(x) == 0:
        return 0.0, 0.0
    _, p = welch(x - np.mean(x), fs=max(fs, 1e-9), nperseg=min(256, len(x)))
    p = np.maximum(p, 0)
    if p.sum() <= 0:
        return 0.0, 0.0
    p = p / p.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return float(ent), float(p.max())


def _drift_score(x: np.ndarray) -> float:
    s = pd.Series(x).interpolate(limit_direction="both")
    if len(s) < 30 or s.std() == 0:
        return 0.0
    w = max(10, len(s) // 10)
    r = s.rolling(w, min_periods=max(5, w // 3)).mean().dropna()
    if r.empty:
        return 0.0
    return float(np.clip((r.max() - r.min()) / (4 * s.std()), 0, 1))


def _feature_profile(s: pd.Series, fs: float, timestamp: pd.Series | None = None) -> Dict[str, float | bool | int]:
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    missing = ~np.isfinite(x)
    runs = _runs(missing)
    durations = runs / fs
    if timestamp is not None and len(timestamp) > 1:
        times = timestamp.to_numpy(float)
        edges = np.diff(np.r_[False, missing, False].astype(int))
        durations = np.array([(times[stop] if stop < len(times) else times[-1] + 1/fs) - times[start]
                              for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))])
    valid = x[~missing]
    n = len(x)
    if len(valid) < 4:
        return {
            "n": n, "missing_rate": float(missing.mean()),
            "longest_missing_run_seconds": float(durations.max()) if len(runs) else 0.0,
            "missing_run_p95_seconds": float(np.quantile(durations, .95)) if len(runs) else 0.0,
            "missing_burstiness": 0.0, "outlier_iqr_rate": 0.0, "outlier_mad_rate": 0.0,
            "skewness": 0.0, "kurtosis_excess": 0.0, "lag1_autocorr": 0.0,
            "drift_score": 0.0, "local_volatility": 0.0, "spectral_entropy": 0.0,
            "dominant_frequency_ratio": 0.0, "zero_variance": True,
        }
    run_concentration = float(runs.max() / max(1, runs.sum())) if len(runs) else 0.0
    mean_run_ratio = float(runs.mean() / max(1, n)) if len(runs) else 0.0
    burst = float(np.clip(.65 * run_concentration + 8 * .35 * mean_run_ratio, 0, 1))
    interp = pd.Series(x).interpolate(limit_direction="both")
    lag1 = float(interp.autocorr(lag=1)) if len(interp) > 2 else 0.0
    if not np.isfinite(lag1):
        lag1 = 0.0
    local_vol = float(np.nanstd(np.diff(interp)) / (np.nanstd(interp) + 1e-12))
    spec_ent, dom = _spectral_entropy(x, fs)
    return {
        "n": n, "missing_rate": float(missing.mean()),
        "longest_missing_run_seconds": float(durations.max()) if len(runs) else 0.0,
        "missing_run_p95_seconds": float(np.quantile(durations, .95)) if len(runs) else 0.0,
        "missing_burstiness": burst, "outlier_iqr_rate": _iqr_outlier_rate(valid),
        "outlier_mad_rate": _robust_outlier_rate(valid),
        "skewness": float(skew(valid, bias=False, nan_policy="omit")),
        "kurtosis_excess": float(kurtosis(valid, fisher=True, bias=False, nan_policy="omit")),
        "lag1_autocorr": lag1, "drift_score": _drift_score(x),
        "local_volatility": local_vol, "spectral_entropy": spec_ent,
        "dominant_frequency_ratio": dom, "zero_variance": bool(np.nanstd(valid) <= 1e-12),
    }


def profile_dataframe(df: pd.DataFrame, fs: float, aggregate: bool = True) -> pd.DataFrame | Dict[str, Any]:
    cols = signal_columns(df)
    rows = []
    corr = df[cols].corr().abs() if len(cols) > 1 else None
    timing = timestamp_diagnostics(df.timestamp) if "timestamp" in df else None
    for c in cols:
        row: Dict[str, Any] = {"feature": c, **_feature_profile(df[c], fs, df.get("timestamp"))}
        if timing and (not timing["valid"] or timing["irregular_intervals"]):
            row["spectral_entropy"] = np.nan
            row["dominant_frequency_ratio"] = np.nan
        if corr is not None:
            vals = corr.loc[c].drop(c, errors="ignore")
            row["corr_mean_abs"] = float(vals.mean()) if not vals.empty else 0.0
        else:
            row["corr_mean_abs"] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if not aggregate:
        return out
    if out.empty:
        return {}
    agg: Dict[str, Any] = {"n_features": len(out)}
    for c in out.columns:
        if c in {"feature", "zero_variance"}:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            finite_values = out[c].dropna()
            agg[c] = float(finite_values.median()) if len(finite_values) else np.nan
            agg[f"max_{c}"] = float(finite_values.max()) if len(finite_values) else np.nan
    agg["zero_variance_fraction"] = float(out["zero_variance"].mean())
    if timing:
        agg["irregular_timestamps"] = timing["irregular_intervals"]
        agg["smoothing_requires_resampling"] = timing["irregular_intervals"] and df.attrs.get("time_processing", {}).get("irregular_smoothing", "disabled") != "resample"
    return agg


def quality_components(profile: Dict[str, float]) -> Dict[str, float]:
    missing = profile.get("missing_rate", 0.0)
    long_gap = profile.get("missing_run_p95_seconds", 0.0)
    burst = profile.get("missing_burstiness", 0.0)
    outlier = max(profile.get("outlier_iqr_rate", 0.0), profile.get("outlier_mad_rate", 0.0))
    sk = abs(profile.get("skewness", 0.0))
    ku = max(0.0, profile.get("kurtosis_excess", 0.0))
    drift = profile.get("drift_score", 0.0)
    vol = profile.get("local_volatility", 0.0)
    corr = profile.get("corr_mean_abs", 0.0)
    return {
        "completeness": 100 * (1 - np.clip(missing / .45, 0, 1)),
        "continuity": 100 * (1 - np.clip(.55 * burst + .45 * long_gap/(1 + long_gap), 0, 1)),
        "plausibility": 100 * (1 - np.clip(outlier / .25, 0, 1)),
        "distribution": 100 * (1 - np.clip(.6 * sk / 3 + .4 * ku / 10, 0, 1)),
        "stability": 100 * (1 - np.clip(.65 * drift + .35 * vol/(1 + vol), 0, 1)),
        "correlation": 100 * np.clip(.25 + .75 * corr, 0, 1),
    }


def quality_index(profile: Dict[str, float], weights: Dict[str, float]) -> float:
    comp = quality_components(profile)
    return float(sum(weights[k] * comp[k] for k in weights) / sum(weights.values()))


def quality_breakdown(profile: Dict[str, float], weights: Dict[str, float]) -> pd.DataFrame:
    comp = quality_components(profile)
    total = sum(weights.values())
    labels = {
        "completeness": "Completeness", "continuity": "Continuity",
        "plausibility": "Outlier burden", "distribution": "Distribution shape",
        "stability": "Temporal stability", "correlation": "Cross-signal consistency",
    }
    rows = []
    for k, score in comp.items():
        w = weights[k] / total
        rows.append({"component": labels[k], "score_0_100": score, "domain_weight": w, "weighted_points": score * w})
    return pd.DataFrame(rows)


def profile_table(frame: pd.DataFrame, fs: float, weights: dict[str, float]) -> pd.DataFrame:
    table = profile_dataframe(frame, fs=fs, aggregate=False)
    rows = []
    for _, row in table.iterrows():
        p = row.to_dict()
        comps = quality_components(p)
        rows.append({**p, **{f"quality_{k}": v for k, v in comps.items()}, "quality_index": quality_index(p, weights)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
PIPELINE_ORDERS = {
    "outlier_imputation_scaling_smoothing": "Outlier treatment → Missing value imputation → Normalization → Optional methods",
    "imputation_outlier_scaling_smoothing": "Missing value imputation → Outlier treatment → Normalization → Optional methods",
}

METHOD_OPTIONS = {
    "outlier": ["none", "zscore", "iqr", "mad", "winsorize", "isolation_forest"],
    "imputation": ["none", "mean", "median", "locf", "linear", "knn", "mice"],
    "scaling": ["none", "standard", "minmax", "robust"],
    "smoothing": ["none", "moving_median", "savgol", "butterworth"],
}

@dataclass(frozen=True)
class PipelineSpec:
    imputation: str = "none"
    outlier: str = "none"
    smoothing: str = "none"
    scaling: str = "none"
    def name(self) -> str:
        return f"imp={self.imputation}|out={self.outlier}|smooth={self.smoothing}|scale={self.scaling}"
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def temporal_fill_mask(x: pd.DataFrame, timestamp: pd.Series, maximum: float) -> pd.DataFrame:
    """Only fill gaps bounded by observations no more than maximum seconds apart."""
    times = timestamp.to_numpy(dtype=float)
    allowed = x.notna().copy()
    for c in x:
        missing = ~np.isfinite(x[c].to_numpy(float))
        edges = np.diff(np.r_[False, missing, False].astype(int))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if start > 0 and stop < len(times) and times[stop] - times[start-1] <= maximum + 1e-12:
                allowed.iloc[start:stop, allowed.columns.get_loc(c)] = True
    return allowed


def apply_imputation(x: pd.DataFrame, method: str, seed: int = 42, *, timestamp: pd.Series | None = None,
                     max_gap_seconds: float | None = None, parameters: dict | None = None) -> pd.DataFrame:
    x = x.replace([np.inf, -np.inf], np.nan)
    if timestamp is not None:
        validate_timestamps(timestamp)
        if method == "linear":
            working = x.copy()
            working.index = timestamp.to_numpy(float)
            output = working.interpolate(method="index", limit_area="inside")
            output.index = x.index
        else:
            output = apply_imputation(x, method, seed, parameters=parameters)
        if parameters is not None and method != "none":
            parameters.update(max_gap_seconds=max_gap_seconds, fill_boundary_gaps=False)
            if method == "linear":
                parameters["interpolation"] = "actual_timestamp_distance"
        if method != "none":
            maximum = max_gap_seconds if max_gap_seconds is not None else np.inf
            output = output.where(temporal_fill_mask(x, timestamp, maximum))
        return output
    method = method.lower()
    if method == "none": return x.copy()
    if method in {"mean", "median"}:
        imputer = SimpleImputer(strategy=method)
        output = pd.DataFrame(imputer.fit_transform(x), index=x.index, columns=x.columns)
        if parameters is not None:
            parameters.update(strategy=method, fill_values=dict(zip(x.columns, imputer.statistics_.tolist())))
        return output
    if parameters is not None and method == "locf":
        parameters["direction"] = "forward"
    if method == "locf": return x.ffill().bfill()
    if method == "linear": return x.interpolate(method="linear", limit_direction="both")
    if method == "knn":
        n_neighbors = max(2, min(7, int(np.sqrt(max(4, len(x)))) // 4))
        if parameters is not None:
            parameters.update(n_neighbors=n_neighbors, weights="distance", metric="nan_euclidean",
                              distance_scaling="StandardScaler", observed_values_preserved=True)
        # Neighbour distances must not depend on the signals' physical units.
        # Entirely unobserved channels cannot be reconstructed from neighbours.
        usable = x.columns[x.notna().any()]
        output = x.copy()
        if not len(usable):
            return output
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x[usable])
        if parameters is not None:
            parameters.update(distance_scaler_signals=list(usable),
                              distance_scaler_mean=scaler.mean_.tolist(), distance_scaler_scale=scaler.scale_.tolist())
        filled = KNNImputer(n_neighbors=n_neighbors, weights="distance").fit_transform(scaled)
        estimates = pd.DataFrame(scaler.inverse_transform(filled), index=x.index, columns=usable)
        output[usable] = x[usable].where(x[usable].notna(), estimates)
        return output
    if method == "mice":
        imp = IterativeImputer(max_iter=8, random_state=seed, sample_posterior=False, initial_strategy="median", skip_complete=True)
        output = pd.DataFrame(imp.fit_transform(x), index=x.index, columns=x.columns)
        if parameters is not None:
            parameters.update(max_iter=8, iterations_run=int(imp.n_iter_), initial_strategy="median",
                              sample_posterior=False, skip_complete=True, estimator="BayesianRidge",
                              tol=imp.tol, imputation_order=imp.imputation_order)
        return output
    raise ValueError(f"Unknown imputation method {method!r}")


def detect_outliers(x: pd.DataFrame, method: str, seed: int = 42, *, parameters: dict | None = None) -> pd.DataFrame:
    method = method.lower()
    mask = pd.DataFrame(False, index=x.index, columns=x.columns)
    if method in {"none", "winsorize"}: return mask
    if method == "isolation_forest":
        if parameters is not None:
            parameters.update(n_estimators=150, contamination="auto", min_samples=16,
                              replacement="NaN",
                              affected_signals="all_selected_signals_on_flagged_rows",
                              missing_for_detection="row_interpolation_then_median_then_zero")
        filled = x.interpolate(limit_direction="both").fillna(x.median()).fillna(0)
        if len(filled) < 16:
            if parameters is not None:
                parameters["status"] = "skipped_insufficient_observations"
            return mask
        detector = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=1)
        pred = detector.fit_predict(filled)
        if parameters is not None:
            parameters.update(max_samples=detector.max_samples, max_samples_fitted=int(detector.max_samples_),
                              max_features=detector.max_features, bootstrap=detector.bootstrap,
                              offset=float(detector.offset_), status="applied")
        mask.loc[pred == -1, :] = True
        return mask
    if parameters is not None:
        parameters.update(min_available_samples=8, replacement="NaN", signals={})
        parameters.update({"zscore": {"threshold": 3., "ddof": 0}, "iqr": {"factor": 1.5},
                           "mad": {"threshold": 3.5, "consistency_factor": .67448975}}.get(method, {}))
    for c in x.columns:
        s = x[c]; valid = s.dropna()
        if len(valid) < 8:
            if parameters is not None:
                parameters["signals"][c] = {"status": "skipped_insufficient_observations"}
            continue
        if method == "zscore":
            sd = valid.std(ddof=0)
            if parameters is not None:
                parameters["signals"][c] = {"mean": float(valid.mean()), "std": float(sd), "applied": bool(sd > 0)}
            if sd > 0: mask[c] = ((s - valid.mean()).abs()/sd) > 3
        elif method == "iqr":
            q1, q3 = valid.quantile([.25, .75]); iqr = q3 - q1
            if parameters is not None:
                parameters["signals"][c] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                                             "lower": float(q1-1.5*iqr), "upper": float(q3+1.5*iqr), "applied": bool(iqr > 0)}
            if iqr > 0: mask[c] = (s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)
        elif method == "mad":
            med = valid.median(); mad = median_abs_deviation(valid, scale=1.0)
            if parameters is not None:
                parameters["signals"][c] = {"median": float(med), "mad": float(mad), "applied": bool(mad > 0)}
            if mad > 0: mask[c] = (0.67448975*(s-med)/mad).abs() > 3.5
        else: raise ValueError(f"Unknown outlier method {method!r}")
    return mask.fillna(False)


def apply_outlier_handling(x: pd.DataFrame, method: str, seed: int = 42, *, parameters: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = method.lower()
    if method == "none": return x.copy(), pd.DataFrame(False, index=x.index, columns=x.columns)
    if method == "winsorize":
        if parameters is not None:
            parameters.update(quantiles=[.01, .99], replacement="clip", signals={})
        out = x.copy(); changed = pd.DataFrame(False, index=x.index, columns=x.columns)
        for c in x.columns:
            lo, hi = x[c].quantile([.01, .99]); clipped = x[c].clip(lo, hi)
            if parameters is not None:
                parameters["signals"][c] = {"lower": float(lo), "upper": float(hi)}
            changed[c] = ~np.isclose(clipped, x[c], equal_nan=True); out[c] = clipped
        return out, changed
    detected = detect_outliers(x, method, seed, parameters=parameters)
    # Leave detected observations missing for the explicit imputation stage.
    return x.mask(detected), detected


def apply_smoothing(x: pd.DataFrame, method: str, fs: float, *, parameters: dict | None = None) -> pd.DataFrame:
    method = method.lower()
    if method == "none": return x.copy()
    out = x.copy()
    for c in x.columns:
        s = x[c].interpolate(limit_direction="both").to_numpy(float)
        if len(s) < 9: continue
        settings = {}
        if method == "moving_median":
            k = min(9, len(s) if len(s) % 2 else len(s)-1); k = max(3, k); out[c] = median_filter(s, size=k, mode="nearest")
            settings = {"window_samples": k, "boundary": "nearest"}
        elif method == "savgol":
            w = min(21, len(s) if len(s) % 2 else len(s)-1); w = max(5, w); out[c] = savgol_filter(s, w, polyorder=min(3, w-2), mode="interp")
            settings = {"window_samples": w, "polynomial_order": min(3, w-2), "boundary": "interp"}
        elif method == "butterworth":
            nyq = max(fs/2, 1e-9); cutoff = min(.20*fs, .45*fs); wn = cutoff/nyq
            b, a = butter(3, wn, btype="low"); padlen = 3*(max(len(a), len(b))-1)
            out[c] = filtfilt(b, a, s) if len(s) > padlen else s
            settings = {"order": 3, "cutoff_hz": cutoff, "normalized_cutoff": wn,
                        "b": b.tolist(), "a": a.tolist(), "padlen": padlen,
                        "applied": len(s) > padlen}
        else: raise ValueError(f"Unknown smoothing method {method!r}")
        if parameters is not None:
            parameters[c] = settings
    # Temporary interpolation supplies the filter with a complete working
    # signal, but filtering must not silently replace explicit imputation.
    return out.where(x.notna())


def smooth_timestamped(x, method, timestamp, fs, settings, *, parameters=None):
    """Filter finite segments, optionally on an internal uniform grid; preserve observation times."""
    if method == "none":
        return x.copy()
    timing = validate_timestamps(timestamp)
    irregular = timing["irregular_intervals"]
    if irregular and settings.get("irregular_smoothing", "disabled") != "resample":
        raise ValueError("Smoothing needs regular timestamps. Enable 'Resample internally for smoothing' in Time processing, or select None for smoothing.")
    maximum = settings.get("max_imputation_gap_seconds")
    if parameters is not None:
        parameters.update(min_segment_samples=9, internal_resampling=bool(irregular),
                          sampling_rate_hz=fs, signals={})
    times = timestamp.to_numpy(float)
    out = x.copy()
    for c in x:
        signal_log = {"segments_skipped": 0, "settings_used": []}
        if parameters is not None:
            parameters["signals"][c] = signal_log
        valid = np.flatnonzero(np.isfinite(x[c].to_numpy(float)))
        if not len(valid):
            continue
        # Split across unfilled signal gaps and acquisition interruptions.
        breaks = (np.diff(valid) != 1)
        if maximum is not None:
            breaks |= np.diff(times[valid]) > maximum
        for positions in np.split(valid, np.flatnonzero(breaks) + 1):
            if len(positions) < 9:
                signal_log["segments_skipped"] += 1
                continue
            segment_times = times[positions]
            segment_values = x[c].iloc[positions].to_numpy(float)
            segment_log = {} if parameters is not None else None
            if irregular:
                count = max(2, int(np.ceil((segment_times[-1] - segment_times[0]) * fs)) + 1)
                if count > 1_000_000:
                    raise ValueError("The smoothing grid is too large. Reduce the maximum gap or split the recording.")
                grid = segment_times[0] + np.arange(count) / fs
                # Uniform acquisition-rate grid, not a lower-rate output/downsampling operation.
                grid_values = np.interp(grid, segment_times, segment_values)
                filtered = apply_smoothing(pd.DataFrame({c: grid_values}), method, fs, parameters=segment_log)[c].to_numpy()
                values = np.interp(segment_times, grid, filtered)
            else:
                values = apply_smoothing(pd.DataFrame({c: segment_values}), method, fs, parameters=segment_log)[c].to_numpy()
            if segment_log is not None:
                actual = segment_log.get(c, {"applied": False})
                entry = next((item for item in signal_log["settings_used"] if item["parameters"] == actual), None)
                if entry is None:
                    signal_log["settings_used"].append({"parameters": actual, "segments": 1})
                else:
                    entry["segments"] += 1
            out.iloc[positions, out.columns.get_loc(c)] = values
    return out.where(x.notna())


def apply_scaling(x: pd.DataFrame, method: str) -> tuple[pd.DataFrame, Any | None]:
    method = method.lower()
    if method == "none": return x.copy(), None
    scaler = StandardScaler() if method == "standard" else MinMaxScaler() if method == "minmax" else RobustScaler() if method == "robust" else None
    if scaler is None: raise ValueError(f"Unknown scaling method {method!r}")
    return pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns), scaler


def changed_fraction(before: pd.DataFrame, after: pd.DataFrame) -> float:
    a = before.to_numpy(float); b = after.to_numpy(float)
    return float(1 - np.mean(np.isclose(a, b, equal_nan=True)))


def execute_pipeline(frame: pd.DataFrame, spec: PipelineSpec, fs: float, order: str = "outlier_imputation_scaling_smoothing", seed: int = 42, *, capture_imputation: bool = False, capture_physical_output: bool = False, metric_stage: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Older saved order keys retain their first-two-step choice; normalization
    # now always precedes optional filtering.
    order={"outlier_imputation_smoothing_scaling":"outlier_imputation_scaling_smoothing",
           "imputation_outlier_smoothing_scaling":"imputation_outlier_scaling_smoothing"}.get(order,order)
    if order not in PIPELINE_ORDERS:
        raise ValueError(f"Unknown pipeline order {order!r}")
    cols = signal_columns(frame); raw = frame[cols].replace([np.inf, -np.inf], np.nan); x = raw.copy()
    timing = validate_timestamps(frame["timestamp"])
    settings = frame.attrs.get("time_processing", {})
    fs = timing["sampling_rate_hz"]
    method_parameters = {stage: {} for stage in ("outlier", "imputation", "scaling", "smoothing")}
    stage_changes: dict[str, float] = {}; detected = pd.DataFrame(False, index=x.index, columns=x.columns)
    imputation_details = {}
    stage_metrics = {}
    def measure_stage(name, before, after):
        if name != metric_stage:
            return
        stage_metrics["stage_changed_fraction"] = changed_fraction(before, after)
        if name == "imputation":
            stage_metrics["imputed_fraction"] = float((before.isna() & after.notna()).to_numpy().mean())
        elif name == "outlier":
            stage_metrics["outlier_affected_fraction"] = float((detected & before.notna()).to_numpy().mean())
        elif name == "smoothing":
            left = aggregate_profile(before, fs).get("local_volatility", 0.)
            right = aggregate_profile(after, fs).get("local_volatility", 0.)
            stage_metrics.update(volatility_before=left, volatility_after=right, volatility_delta=right-left)
        elif name == "scaling":
            stage_metrics.update(scaled_min_median=float(after.min().median()),
                                 scaled_max_median=float(after.max().median()),
                                 scaled_std_median=float(after.std(ddof=0).median()))
    def apply_and_track(name: str, fn):
        nonlocal x
        before = x.copy(); x = fn(x); stage_changes[name] = changed_fraction(before, x)
        measure_stage(name, before, x)
        if capture_imputation and name == "imputation":
            imputation_details.update(imputation_input=before, imputation_output=x.copy())
    if order == "outlier_imputation_scaling_smoothing":
        before = x.copy(); x, detected = apply_outlier_handling(x, spec.outlier, seed, parameters=method_parameters["outlier"]); stage_changes["outlier"] = changed_fraction(before, x)
        measure_stage("outlier", before, x)
        apply_and_track("imputation", lambda z: apply_imputation(z, spec.imputation, seed, timestamp=frame.timestamp, max_gap_seconds=settings.get("max_imputation_gap_seconds"), parameters=method_parameters["imputation"]))
    else:
        apply_and_track("imputation", lambda z: apply_imputation(z, spec.imputation, seed, timestamp=frame.timestamp, max_gap_seconds=settings.get("max_imputation_gap_seconds"), parameters=method_parameters["imputation"]))
        before = x.copy(); x, detected = apply_outlier_handling(x, spec.outlier, seed, parameters=method_parameters["outlier"]); stage_changes["outlier"] = changed_fraction(before, x)
        measure_stage("outlier", before, x)
    pre_scaled = x.copy(); scaled, scaler = apply_scaling(x, spec.scaling); stage_changes["scaling"] = changed_fraction(pre_scaled, scaled)
    measure_stage("scaling", pre_scaled, scaled)
    x = scaled
    if scaler is not None:
        method_parameters["scaling"] = scaler.get_params()
    apply_and_track("smoothing", lambda z: smooth_timestamped(z, spec.smoothing, frame.timestamp, fs, settings, parameters=method_parameters["smoothing"]))
    out = frame.copy(); out[cols] = x
    # Remove only the change of units from the actual final output. This keeps
    # cleaning diagnostics and physical-unit plots faithful to filters that
    # ran after normalization, rather than rerunning a different pipeline.
    physical = pd.DataFrame(scaler.inverse_transform(x),index=x.index,columns=x.columns) if scaler is not None else x.copy()
    physical_frame = frame.copy(); physical_frame[cols] = physical
    return out, {
        "spec": asdict(spec), "order": order, "detected_mask": detected,
        "stage_changed_fraction": stage_changes,
        "cleaning_changed_fraction": changed_fraction(raw, physical),
        "scaling_changed_fraction": stage_changes["scaling"],
        "overall_changed_fraction": changed_fraction(raw, x),
        "scaler": type(scaler).__name__ if scaler is not None else None,
        "seed": seed,
        "method_parameters": method_parameters,
        "preprocessing_settings": dict(PREPROCESSING_SETTINGS),
        "time_processing": dict(settings), "timestamp_diagnostics": timing,
        "scaler_parameters": {name: getattr(scaler, name).tolist() for name in
                              ("mean_", "scale_", "center_", "min_", "data_min_", "data_max_", "data_range_")
                              if scaler is not None and getattr(scaler, name, None) is not None},
        **({"comparison_stage_metrics":stage_metrics} if metric_stage else {}),
        **imputation_details,
        **({"physical_output":physical_frame} if capture_physical_output else {}),
    }


def aggregate_profile(frame: pd.DataFrame, fs: float) -> dict[str, Any]:
    return profile_dataframe(frame, fs, aggregate=True)


def profile_change_table(before: pd.DataFrame, after: pd.DataFrame, fs: float, weights: dict[str, float]) -> pd.DataFrame:
    left = profile_table(before, fs, weights).set_index("feature"); right = profile_table(after, fs, weights).set_index("feature")
    metrics = list(left.columns.intersection(right.columns))
    rows = []
    for f in left.index.intersection(right.index):
        row = {"feature": f}
        for m in metrics:
            row[f"before_{m}"] = left.loc[f, m]; row[f"after_{m}"] = right.loc[f, m]
            if m != "zero_variance":
                row[f"delta_{m}"] = right.loc[f, m] - left.loc[f, m]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_stage_methods(frame: pd.DataFrame, stage: str, methods: list[str], fs: float, weights: dict[str, float], base_spec: PipelineSpec, order: str, *, imputation_details: dict | None = None, visual_outputs: dict | None = None, execution_parameters: dict | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    outputs = {}; rows = []; original_q = quality_index(aggregate_profile(frame, fs), weights)
    for method in methods:
        kwargs = asdict(base_spec); kwargs[stage] = method; spec = PipelineSpec(**kwargs)
        result, meta = execute_pipeline(frame, spec, fs, order, capture_imputation=stage=="imputation" and imputation_details is not None, capture_physical_output=visual_outputs is not None, metric_stage=stage)
        if execution_parameters is not None:
            execution_parameters[method] = meta["method_parameters"]
        if visual_outputs is not None:
            visual_outputs[method] = result if stage=="scaling" else meta["physical_output"]
        if stage=="imputation" and imputation_details is not None:
            imputation_details[method] = {"input": meta["imputation_input"], "output": meta["imputation_output"]}
        outputs[method] = result; p = aggregate_profile(result, fs); q = quality_index(p, weights)
        rows.append({"method": method, "quality_index": q, "quality_delta": q-original_q, "missing_rate": p.get("missing_rate", 0), "outlier_rate": max(p.get("outlier_iqr_rate",0), p.get("outlier_mad_rate",0)), "local_volatility": p.get("local_volatility",0), "cleaning_changed_fraction": meta["cleaning_changed_fraction"], "overall_changed_fraction": meta["overall_changed_fraction"]})
        rows[-1].update(meta["comparison_stage_metrics"])
        rows[-1]["stage_changed_fraction"] = meta["stage_changed_fraction"][stage]
        rows[-1]["remaining_missing_fraction"] = float(result[signal_columns(result)].isna().to_numpy().mean())
    return outputs, pd.DataFrame(rows).sort_values("quality_index", ascending=False)

# ---------------------------------------------------------------------------
# Profile-specific recommendations + clearly separate domain evidence
# ---------------------------------------------------------------------------
STAGE_VALIDATION = {
    "imputation": {"status":"candidate", "grouped_cv_mean":.5018, "majority":.7455, "holdout":.80, "holdout_name":"ETDD70/Dyslexia", "reason":"Single-dataset hold-out was good, but grouped cross-validation remained below the majority baseline."},
    "outlier": {"status":"unstable", "grouped_cv_mean":.20, "majority":.64, "holdout":.0, "holdout_name":"ETDD70/Dyslexia", "reason":"Neither grouped validation nor the external eye-tracking hold-out supports a stable rule yet."},
    "smoothing": {"status":"candidate", "grouped_cv_mean":.6267, "majority":.4933, "holdout":.0, "holdout_name":"ETDD70/Dyslexia", "reason":"Grouped validation exceeded the baseline, but the external hold-out failed."},
}

@dataclass
class StageRecommendation:
    stage: str; method: str; status: str; rule_trace: list[str]; caution: str; domain_support: list[tuple[str,float]]; validation: dict[str,Any]
    def as_dict(self): return asdict(self)


def _imputation_rule(p):
    trace=[]; n=p.get("n",p.get("max_n",0)); longest=p.get("longest_missing_run_seconds",0); max_vol=p.get("max_local_volatility",p.get("local_volatility",0)); burst=p.get("missing_burstiness",0)
    if n <= 182.5: return "mean", [f"n={n:.0f} ≤ 182.5"]
    trace.append(f"n={n:.0f} > 182.5")
    if longest <= .1693:
        trace.append(f"longest missing run={longest:.3f}s ≤ 0.169s")
        if max_vol <= .3987: trace.append(f"max local volatility={max_vol:.3f} ≤ 0.399"); return "linear", trace
        trace.append(f"max local volatility={max_vol:.3f} > 0.399"); return "locf", trace
    trace.append(f"longest missing run={longest:.3f}s > 0.169s")
    if burst <= .248: trace.append(f"missing burstiness={burst:.3f} ≤ 0.248"); return "locf", trace
    trace.append(f"missing burstiness={burst:.3f} > 0.248"); return "linear", trace


def _outlier_rule(p):
    trace=[]; ent=p.get("spectral_entropy",0); iqr=p.get("max_outlier_iqr_rate",p.get("outlier_iqr_rate",0)); mad=p.get("max_outlier_mad_rate",p.get("outlier_mad_rate",0))
    if not np.isfinite(ent):
        method = "iqr" if iqr <= .0233 else "mad" if mad <= .0762 else "zscore"
        return method, ["Frequency diagnostics are unavailable for irregular timestamps; use the IQR/MAD branch without a spectral-entropy condition.", f"max IQR outlier rate={iqr:.3%}", f"max MAD outlier rate={mad:.3%}"]
    if ent <= .8184:
        trace.append(f"spectral entropy={ent:.3f} ≤ 0.818")
        if iqr <= .0233: trace.append(f"max IQR outlier rate={iqr:.3%} ≤ 2.33%"); return "iqr", trace
        trace.append(f"max IQR outlier rate={iqr:.3%} > 2.33%")
        if mad <= .0762: trace.append(f"max MAD outlier rate={mad:.3%} ≤ 7.62%"); return "mad", trace
        trace.append(f"max MAD outlier rate={mad:.3%} > 7.62%"); return "zscore", trace
    trace.append(f"spectral entropy={ent:.3f} > 0.818"); return "winsorize", trace


def _smoothing_rule(p):
    trace=[]; vol=p.get("max_local_volatility",p.get("local_volatility",0)); sk=p.get("skewness",0); dom=p.get("dominant_frequency_ratio",0)
    if vol > .5748: return "none", [f"max local volatility={vol:.3f} > 0.575"]
    trace.append(f"max local volatility={vol:.3f} ≤ 0.575")
    if sk > .1067: trace.append(f"skewness={sk:.3f} > 0.107"); return "butterworth", trace
    trace.append(f"skewness={sk:.3f} ≤ 0.107")
    if dom <= .4698: trace.append(f"dominant-frequency ratio={dom:.3f} ≤ 0.470"); return "savgol", trace
    trace.append(f"dominant-frequency ratio={dom:.3f} > 0.470"); return "moving_median", trace


def recommend_stage(stage: str, profile: dict[str,float], domain: str) -> StageRecommendation:
    method, trace = _imputation_rule(profile) if stage=="imputation" else _outlier_rule(profile) if stage=="outlier" else _smoothing_rule(profile)
    guard=""
    if stage=="imputation" and profile.get("missing_rate",0)<=0: method="none"; guard="No missing values were observed in the selected signals."
    if stage=="outlier" and domain=="finance": method="none"; guard="Finance guardrail: heavy tails and jumps may be genuine; automatic removal is disabled by default."
    if stage=="outlier" and profile.get("kurtosis_excess",0)>8 and profile.get("outlier_mad_rate",0)<.01: method="none"; guard="High kurtosis with few isolated MAD outliers suggests heavy tails/regimes rather than simple spikes."
    if stage=="smoothing" and domain in {"finance","network"} and method!="none": method="none"; guard=f"{DOMAIN_LABELS[domain]} guardrail: preserve jumps/bursts by default."
    if stage=="smoothing" and profile.get("smoothing_requires_resampling",False): method="none"; guard="Irregular timestamps require explicit internal resampling before smoothing."
    if guard: trace.append(guard)
    validation=dict(STAGE_VALIDATION[stage]); caution=(guard+" " if guard else "")+validation["reason"]
    support = REAL_EYE_TRACKING_PILOT.get(stage, []) if domain=="eye_tracking" else []
    return StageRecommendation(stage,method,validation["status"],trace,caution,support,validation)


def recommend_scaling(profile: dict[str,float], domain: str) -> StageRecommendation:
    outlier=max(profile.get("outlier_iqr_rate",0),profile.get("outlier_mad_rate",0)); ku=profile.get("kurtosis_excess",0)
    if domain=="finance": method="none"; trace=["Finance auto-profile preserves interpretable scale by default."]
    elif outlier>.03 or ku>3: method="robust"; trace=[f"Tail diagnostics (outliers={outlier:.2%}, excess kurtosis={ku:.2f}) favour RobustScaler."]
    else: method="standard"; trace=["Auto mode selects StandardScaler for a comparatively regular distribution."]
    return StageRecommendation("scaling",method,"profile_based",trace,"Scaling is chosen from domain and distribution diagnostics rather than ranked as a universal cleaning method.",[],{})

# ---------------------------------------------------------------------------
# Ground-truth / pseudo-ground-truth validation
# ---------------------------------------------------------------------------
@dataclass
class CorruptionResult:
    data: pd.DataFrame; combined_mask: pd.DataFrame; parameters: Dict[str,Any]


def inject_corruption(clean: pd.DataFrame, corruption: str, severity: float, seed: int=42) -> CorruptionResult:
    rng=np.random.default_rng(seed); data=clean.copy(deep=True); cols=signal_columns(clean); mask=pd.DataFrame(False,index=clean.index,columns=cols); n=len(clean)
    if corruption=="mcar_missing":
        rate=float(np.clip(severity,0,.8))
        for c in cols:
            idx=rng.choice(n,size=min(max(1,int(round(n*rate))),n),replace=False); data.loc[idx,c]=np.nan; mask.loc[idx,c]=True
        return CorruptionResult(data,mask,{"rate":rate})
    if corruption=="block_missing":
        rate=float(np.clip(severity,0,.7))
        for c in cols:
            remaining=max(1,int(round(n*rate))); typical=max(3,int(n*max(.005,rate/8))); attempts=0
            while remaining>0 and attempts<100:
                length=int(np.clip(rng.lognormal(np.log(typical),.55),2,max(3,n//4))); length=min(length,remaining); start=int(rng.integers(0,max(1,n-length+1)))
                data.loc[start:start+length-1,c]=np.nan; mask.loc[start:start+length-1,c]=True; remaining-=length; attempts+=1
        return CorruptionResult(data,mask,{"rate":rate})
    if corruption=="spikes":
        rate=float(np.clip(severity,.0005,.2))
        for c in cols:
            x=clean[c].astype(float); scale=float(np.nanstd(x)) or 1.; k=max(1,int(round(n*rate))); idx=rng.choice(n,size=min(k,n),replace=False)
            data.loc[idx,c]=x.iloc[idx].to_numpy()+rng.choice([-1.,1.],len(idx))*rng.uniform(4.,9.,len(idx))*scale; mask.loc[idx,c]=True
        return CorruptionResult(data,mask,{"rate":rate})
    if corruption=="drift":
        frac=float(np.clip(.25+severity,.2,.9)); length=max(10,int(n*frac)); start=int(rng.integers(0,max(1,n-length+1))); ramp=np.linspace(0,1,length)
        for c in cols:
            scale=float(np.nanstd(clean[c])) or 1.; amount=scale*(.8+4*severity)*rng.choice([-1,1]); data.loc[start:start+length-1,c]=clean.loc[start:start+length-1,c].to_numpy()+amount*ramp; mask.loc[start:start+length-1,c]=True
        return CorruptionResult(data,mask,{"start":start,"length":length})
    if corruption=="level_shift":
        start=int(rng.integers(max(5,n//5),max(6,4*n//5)))
        for c in cols:
            scale=float(np.nanstd(clean[c])) or 1.; amount=scale*(.8+4.5*severity)*rng.choice([-1,1]); data.loc[start:,c]=clean.loc[start:,c]+amount; mask.loc[start:,c]=True
        return CorruptionResult(data,mask,{"start":start})
    if corruption=="stuck_at":
        length=max(3,int(n*float(np.clip(severity,.01,.35)))); start=int(rng.integers(0,max(1,n-length+1)))
        for c in cols:
            value=float(clean[c].iloc[max(0,start-1)]); data.loc[start:start+length-1,c]=value; mask.loc[start:start+length-1,c]=True
        return CorruptionResult(data,mask,{"start":start,"length":length})
    if corruption=="high_frequency_noise":
        for c in cols:
            scale=float(np.nanstd(clean[c])) or 1.; noise=rng.normal(0,scale*(.05+1.4*severity),n); alternating=((-1.)**np.arange(n))*scale*.25*severity
            data[c]=clean[c].to_numpy()+noise+alternating; mask[c]=True
        return CorruptionResult(data,mask,{"severity":severity})
    if corruption=="clipping":
        q=float(np.clip(.01+severity/3,.01,.18))
        for c in cols:
            lo,hi=clean[c].quantile([q,1-q]).to_numpy(); clipped=clean[c].clip(lo,hi); data[c]=clipped; mask[c]=~np.isclose(clipped.to_numpy(),clean[c].to_numpy(),equal_nan=True)
        return CorruptionResult(data,mask,{"tail_quantile":q})
    raise ValueError(corruption)


def complete_window(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    cols=signal_columns(frame); valid=frame[cols].notna().all(axis=1)
    if valid.sum()<min(length,32): return frame.dropna(subset=cols).head(length).reset_index(drop=True)
    groups=(valid!=valid.shift(fill_value=False)).cumsum(); best=[]
    for _,idx in valid[valid].groupby(groups[valid]).groups.items():
        indices=list(idx)
        if len(indices)>len(best): best=indices
    return frame.loc[best[:length]].reset_index(drop=True) if best else frame.dropna(subset=cols).head(length).reset_index(drop=True)


def _bounded(x: float) -> float:
    if not np.isfinite(x): return np.nan
    x=max(0.,float(x)); return x/(1+x)


def _fill(s: pd.Series) -> np.ndarray:
    return s.interpolate(limit_direction="both").fillna(s.median()).fillna(0).to_numpy(float)


def _spectral_js(a,b,fs):
    if len(a)<16 or len(b)<16:return 0.
    nseg=min(256,len(a),len(b)); _,pa=welch(a-np.mean(a),fs=max(fs,1e-9),nperseg=nseg); _,pb=welch(b-np.mean(b),fs=max(fs,1e-9),nperseg=nseg)
    pa=pa/(pa.sum()+1e-12); pb=pb/(pb.sum()+1e-12); return float(jensenshannon(pa+1e-12,pb+1e-12,base=2.)**2)


def _f1(true,pred):
    true=true.astype(bool).ravel(); pred=pred.astype(bool).ravel(); tp=np.sum(true&pred); fp=np.sum(~true&pred); fn=np.sum(true&~pred)
    precision=tp/(tp+fp) if tp+fp else 1.; recall=tp/(tp+fn) if tp+fn else 1.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.; return precision,recall,f1


def evaluate_reconstruction(clean, corrupted, processed, corruption_mask, fs, event_mask=None, detected_mask=None, detection_applicable=False):
    cols=[c for c in signal_columns(clean) if c in processed and c in corrupted]; per=[]; all_true=[]; all_pred=[]
    for c in cols:
        y=_fill(clean[c]); z=_fill(processed[c]); mask=corruption_mask[c].to_numpy(bool) if c in corruption_mask else np.zeros(len(y),bool); scale=np.std(y)+1e-12
        damaged=mask & np.isfinite(y)&np.isfinite(z); intact=(~mask)&np.isfinite(y)&np.isfinite(z)
        rec=np.sqrt(np.mean((z[damaged]-y[damaged])**2))/scale if damaged.any() else 0.; harm=np.sqrt(np.mean((z[intact]-y[intact])**2))/scale if intact.any() else 0.
        dy,dz=np.diff(y),np.diff(z); dyn=np.sqrt(np.mean((dz-dy)**2))/(np.std(dy)+1e-12); wd=wasserstein_distance(y,z)/scale
        skerr=abs(float(skew(z,bias=False))-float(skew(y,bias=False))); kuerr=abs(float(kurtosis(z,fisher=True,bias=False))-float(kurtosis(y,fisher=True,bias=False)))
        dist=.55*_bounded(wd)+.20*_bounded(skerr)+.25*_bounded(kuerr/3); freq=_spectral_js(y,z,fs)
        event_loss=np.nan
        if event_mask is not None and c in event_mask and event_mask[c].any():
            em=event_mask[c].to_numpy(bool); ids=np.clip(np.flatnonzero(em)-1,0,len(dy)-1); clean_amp=np.mean(np.abs(dy[ids]))+1e-12; proc_amp=np.mean(np.abs(dz[ids]))+1e-12; event_loss=_bounded(abs(np.log(proc_amp/clean_amp)))
        per.append({"reconstruction":_bounded(rec),"clean_harm":_bounded(harm),"dynamics":_bounded(dyn),"distribution":float(np.clip(dist,0,1)),"frequency":float(np.clip(freq,0,1)),"event":event_loss})
        if detection_applicable and detected_mask is not None and c in detected_mask: all_true.append(mask); all_pred.append(detected_mask[c].to_numpy(bool))
    metrics={k:float(np.nanmean([r[k] for r in per])) if per and not all(np.isnan([r[k] for r in per])) else np.nan for k in ["reconstruction","clean_harm","dynamics","distribution","frequency","event"]}
    if all_true:
        precision,recall,f1=_f1(np.concatenate(all_true),np.concatenate(all_pred)); metrics.update({"det_precision":precision,"det_recall":recall,"det_f1":f1,"detection":1-f1})
    else: metrics.update({"det_precision":np.nan,"det_recall":np.nan,"det_f1":np.nan,"detection":np.nan})
    metrics["retention"]=float(1-processed[cols].notna().to_numpy().mean()) if cols else 1.; return metrics


def domain_weighted_loss(metrics: Dict[str,float], weights: Dict[str,float]) -> float:
    active=[k for k,w in weights.items() if w>0 and k in metrics and np.isfinite(metrics[k])]
    total=sum(weights[k] for k in active)
    return float(sum(weights[k]*metrics[k] for k in active)/total) if total>0 else np.nan


def compact_pipeline_grid(recommended: PipelineSpec, alternatives: dict[str,list[str]], max_pipelines: int=10) -> list[PipelineSpec]:
    values={}
    for stage in ("imputation","outlier","smoothing"):
        first=getattr(recommended,stage); vals=[first]+[v for v in alternatives.get(stage,[]) if v!=first]; values[stage]=vals[:2]
    specs=[PipelineSpec(i,o,s,"none") for i,o,s in product(values["imputation"],values["outlier"],values["smoothing"])]
    return specs[:max_pipelines]


def controlled_validation(clean, specs, corruption, severity, fs, loss_weights, event_mask=None, seed=42):
    injected=inject_corruption(clean,corruption,severity,seed); rows=[]; outputs={}; detection_applicable=corruption in {"spikes","clipping"}
    for spec in specs:
        processed,meta=execute_pipeline(injected.data,spec,fs); metrics=evaluate_reconstruction(clean,injected.data,processed,injected.combined_mask,fs,event_mask,meta["detected_mask"],detection_applicable)
        loss=domain_weighted_loss(metrics,loss_weights); name=spec.name(); outputs[name]=processed; rows.append({"pipeline":name,"loss":loss,**asdict(spec),**metrics})
    return injected.data,pd.DataFrame(rows).sort_values("loss").reset_index(drop=True),outputs

# ---------------------------------------------------------------------------
# Minimal synthetic demos (for a usable standalone app without external files)
# ---------------------------------------------------------------------------
def _random_missing_mask(n: int, n_signals: int, rate: float, seed: int) -> np.ndarray:
    """Mix scattered losses with random-length gaps, meeting a global cell budget.

    Channels may have different missing rates. Gap starts are shuffled across
    channels and time; block lengths are sampled on a logarithmic scale so both
    short and long gaps occur. The timestamp is never included in this mask.
    """
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Missing-value rate must be between 0 and 1.")
    mask=np.zeros((n_signals,n),dtype=bool)
    remaining=int(round(mask.size*rate))
    if remaining == 0:
        return mask.T
    rng=np.random.default_rng(np.random.SeedSequence([seed, 7291]))
    max_gap=max(2,n//8)
    for flat_index in rng.permutation(mask.size):
        if remaining == 0:
            break
        channel,start=divmod(int(flat_index),n)
        if mask[channel,start]:
            continue
        length=1 if rng.random()<.4 else int(round(np.exp(rng.uniform(np.log(2),np.log(max_gap)))))
        length=min(length,remaining,n-start)
        # Stop at an existing gap rather than counting overlapping cells twice.
        occupied=np.flatnonzero(mask[channel,start:start+length])
        if occupied.size:
            length=int(occupied[0])
        mask[channel,start:start+length]=True
        remaining-=length
    return mask.T


def generate_domain_series(domain: str, n: int=2400, seed: int=42, missing_rate: float=0.0) -> tuple[pd.DataFrame,pd.DataFrame,float]:
    rng=np.random.default_rng(seed)
    fs=120. if domain=="eye_tracking" else 10. if domain=="industrial" else 50. if domain=="physiological" else 1. if domain=="finance" else 1/60 if domain=="network" else 1/900
    t=np.arange(n)/fs; event=pd.DataFrame(index=range(n))
    if domain=="eye_tracking":
        gx=960+350*np.sin(.8*t)+rng.normal(0,8,n); gy=540+240*np.cos(.6*t)+rng.normal(0,7,n); pupil=3.5+.15*np.sin(.2*t)+rng.normal(0,.03,n); df=pd.DataFrame({"timestamp":t,"gaze_x":gx,"gaze_y":gy,"pupil_left":pupil,"pupil_right":pupil+rng.normal(0,.02,n)}); event=pd.DataFrame(False,index=df.index,columns=signal_columns(df))
    elif domain=="industrial":
        df=pd.DataFrame({"timestamp":t,"temperature":65+2*np.sin(.1*t)+rng.normal(0,.2,n),"pressure":4.5+.15*np.sin(.08*t)+rng.normal(0,.02,n),"vibration":.9+.2*np.abs(np.sin(1.8*t))+rng.normal(0,.03,n),"flow":100+6*np.sin(.1*t)+rng.normal(0,.7,n)}); event=pd.DataFrame(False,index=df.index,columns=signal_columns(df))
    elif domain=="finance":
        ret=rng.standard_t(5,n)*.01; price=100*np.exp(np.cumsum(ret)); df=pd.DataFrame({"timestamp":t,"price":price,"log_return":ret,"volume":np.exp(10+5*np.abs(ret)+rng.normal(0,.3,n))}); event=pd.DataFrame(False,index=df.index,columns=signal_columns(df))
    else:
        df=pd.DataFrame({"timestamp":t,"signal_a":np.sin(.2*t)+rng.normal(0,.05,n),"signal_b":.7*np.sin(.2*t+.5)+rng.normal(0,.06,n),"signal_c":np.cos(.05*t)+rng.normal(0,.04,n)}); event=pd.DataFrame(False,index=df.index,columns=signal_columns(df))
    cols=signal_columns(df)
    missing=_random_missing_mask(n,len(cols),missing_rate,seed)
    df[cols]=df[cols].mask(missing)
    df.attrs["timestamp_unit"] = "Seconds (s)"
    return df,event,fs
