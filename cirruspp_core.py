from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, savgol_filter, welch
from scipy.spatial.distance import jensenshannon
from scipy.stats import kurtosis, median_abs_deviation, skew, wasserstein_distance
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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
]
TIME_PATTERNS = [
    r"^timestamp$", r"time", r"recordingtime", r"recording_time", r"device_time",
    r"system_time", r"elapsed", r"duration",
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


def infer_sampling_rate(df: pd.DataFrame, timestamp_col: str | None) -> tuple[float | None, str]:
    if not timestamp_col or timestamp_col not in df.columns:
        return None, "No timestamp selected"
    s = pd.to_numeric(df[timestamp_col], errors="coerce").dropna()
    if len(s) < 4:
        return None, "Too few timestamp values"
    diffs = np.diff(s.to_numpy(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) < 3:
        return None, "No stable positive timestamp increments"
    dt = float(np.median(diffs))
    name = str(timestamp_col).lower()
    if any(t in name for t in ("micro", "_us", "usec", "µs")):
        mult, unit = 1e-6, "microseconds"
    elif any(t in name for t in ("nano", "_ns", "nsec")):
        mult, unit = 1e-9, "nanoseconds"
    elif any(t in name for t in ("milli", "_ms", "msec")):
        mult, unit = 1e-3, "milliseconds"
    elif dt < 1:
        mult, unit = 1.0, "seconds"
    elif dt < 1000:
        if dt >= 60 and abs(dt / 60 - round(dt / 60)) < 1e-6:
            mult, unit = 1.0, "seconds"
        else:
            mult, unit = 1e-3, "milliseconds"
    elif dt < 1e6:
        mult, unit = 1e-6, "microseconds"
    else:
        mult, unit = 1e-9, "nanoseconds"
    fs = 1.0 / (dt * mult)
    if not (1e-5 <= fs <= 10000):
        return None, f"Inferred rate {fs:g} Hz is outside the supported range"
    return float(fs), f"Inferred from median timestamp step ({unit})"


def canonical_frame(df: pd.DataFrame, signal_cols: list[str], fs: float) -> pd.DataFrame:
    # CIRRUS++ uses a clean sample-time axis in seconds internally. The original
    # timestamp column is preserved unchanged in the full-data export.
    out = pd.DataFrame({"timestamp": np.arange(len(df), dtype=float) / max(fs, 1e-12)})
    for c in signal_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").reset_index(drop=True)
    return out

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


def _feature_profile(s: pd.Series, fs: float) -> Dict[str, float | bool | int]:
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    missing = ~np.isfinite(x)
    runs = _runs(missing)
    valid = x[~missing]
    n = len(x)
    if len(valid) < 4:
        return {
            "n": n, "missing_rate": float(missing.mean()),
            "longest_missing_run_seconds": float(runs.max()/fs) if len(runs) else 0.0,
            "missing_run_p95_seconds": float(np.quantile(runs, .95)/fs) if len(runs) else 0.0,
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
        "longest_missing_run_seconds": float(runs.max()/fs) if len(runs) else 0.0,
        "missing_run_p95_seconds": float(np.quantile(runs, .95)/fs) if len(runs) else 0.0,
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
    for c in cols:
        row: Dict[str, Any] = {"feature": c, **_feature_profile(df[c], fs)}
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
            agg[c] = float(out[c].median())
            agg[f"max_{c}"] = float(out[c].max())
    agg["zero_variance_fraction"] = float(out["zero_variance"].mean())
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


def apply_imputation(x: pd.DataFrame, method: str, seed: int = 42) -> pd.DataFrame:
    method = method.lower()
    if method == "none": return x.copy()
    if method == "mean": return pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(x), index=x.index, columns=x.columns)
    if method == "median": return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), index=x.index, columns=x.columns)
    if method == "locf": return x.ffill().bfill()
    if method == "linear": return x.interpolate(method="linear", limit_direction="both")
    if method == "knn":
        n_neighbors = max(2, min(7, int(np.sqrt(max(4, len(x)))) // 4))
        return pd.DataFrame(KNNImputer(n_neighbors=n_neighbors, weights="distance").fit_transform(x), index=x.index, columns=x.columns)
    if method == "mice":
        imp = IterativeImputer(max_iter=8, random_state=seed, sample_posterior=False, initial_strategy="median", skip_complete=True)
        return pd.DataFrame(imp.fit_transform(x), index=x.index, columns=x.columns)
    raise ValueError(f"Unknown imputation method {method!r}")


def detect_outliers(x: pd.DataFrame, method: str, seed: int = 42) -> pd.DataFrame:
    method = method.lower()
    mask = pd.DataFrame(False, index=x.index, columns=x.columns)
    if method in {"none", "winsorize"}: return mask
    if method == "isolation_forest":
        filled = x.interpolate(limit_direction="both").fillna(x.median()).fillna(0)
        if len(filled) < 16: return mask
        pred = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=1).fit_predict(filled)
        mask.loc[pred == -1, :] = True
        return mask
    for c in x.columns:
        s = x[c]; valid = s.dropna()
        if len(valid) < 8: continue
        if method == "zscore":
            sd = valid.std(ddof=0)
            if sd > 0: mask[c] = ((s - valid.mean()).abs()/sd) > 3
        elif method == "iqr":
            q1, q3 = valid.quantile([.25, .75]); iqr = q3 - q1
            if iqr > 0: mask[c] = (s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)
        elif method == "mad":
            med = valid.median(); mad = median_abs_deviation(valid, scale=1.0)
            if mad > 0: mask[c] = (0.67448975*(s-med)/mad).abs() > 3.5
        else: raise ValueError(f"Unknown outlier method {method!r}")
    return mask.fillna(False)


def apply_outlier_handling(x: pd.DataFrame, method: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = method.lower()
    if method == "none": return x.copy(), pd.DataFrame(False, index=x.index, columns=x.columns)
    if method == "winsorize":
        out = x.copy(); changed = pd.DataFrame(False, index=x.index, columns=x.columns)
        for c in x.columns:
            lo, hi = x[c].quantile([.01, .99]); clipped = x[c].clip(lo, hi)
            changed[c] = ~np.isclose(clipped, x[c], equal_nan=True); out[c] = clipped
        return out, changed
    detected = detect_outliers(x, method, seed)
    out = x.mask(detected)
    out = out.interpolate(method="linear", limit_direction="both").fillna(out.median())
    return out, detected


def apply_smoothing(x: pd.DataFrame, method: str, fs: float) -> pd.DataFrame:
    method = method.lower()
    if method == "none": return x.copy()
    out = x.copy()
    for c in x.columns:
        s = x[c].interpolate(limit_direction="both").to_numpy(float)
        if len(s) < 9: continue
        if method == "moving_median":
            k = min(9, len(s) if len(s) % 2 else len(s)-1); k = max(3, k); out[c] = medfilt(s, kernel_size=k)
        elif method == "savgol":
            w = min(21, len(s) if len(s) % 2 else len(s)-1); w = max(5, w); out[c] = savgol_filter(s, w, polyorder=min(3, w-2), mode="interp")
        elif method == "butterworth":
            nyq = max(fs/2, 1e-9); cutoff = min(.20*fs, .45*fs); wn = cutoff/nyq
            b, a = butter(3, wn, btype="low"); padlen = 3*(max(len(a), len(b))-1)
            out[c] = filtfilt(b, a, s) if len(s) > padlen else s
        else: raise ValueError(f"Unknown smoothing method {method!r}")
    return out


def apply_scaling(x: pd.DataFrame, method: str) -> tuple[pd.DataFrame, Any | None]:
    method = method.lower()
    if method == "none": return x.copy(), None
    scaler = StandardScaler() if method == "standard" else MinMaxScaler() if method == "minmax" else RobustScaler() if method == "robust" else None
    if scaler is None: raise ValueError(f"Unknown scaling method {method!r}")
    return pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns), scaler


def changed_fraction(before: pd.DataFrame, after: pd.DataFrame) -> float:
    a = before.to_numpy(float); b = after.to_numpy(float)
    return float(1 - np.mean(np.isclose(a, b, equal_nan=True)))


def execute_pipeline(frame: pd.DataFrame, spec: PipelineSpec, fs: float, order: str = "imputation_outlier_smoothing_scaling", seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = signal_columns(frame); raw = frame[cols].copy(); x = raw.copy()
    stage_changes: dict[str, float] = {}; detected = pd.DataFrame(False, index=x.index, columns=x.columns)
    def apply_and_track(name: str, fn):
        nonlocal x
        before = x.copy(); x = fn(x); stage_changes[name] = changed_fraction(before, x)
    if order == "outlier_imputation_smoothing_scaling":
        before = x.copy(); x, detected = apply_outlier_handling(x, spec.outlier, seed); stage_changes["outlier"] = changed_fraction(before, x)
        apply_and_track("imputation", lambda z: apply_imputation(z, spec.imputation, seed))
    else:
        apply_and_track("imputation", lambda z: apply_imputation(z, spec.imputation, seed))
        before = x.copy(); x, detected = apply_outlier_handling(x, spec.outlier, seed); stage_changes["outlier"] = changed_fraction(before, x)
    apply_and_track("smoothing", lambda z: apply_smoothing(z, spec.smoothing, fs))
    pre_scaled = x.copy(); scaled, scaler = apply_scaling(x, spec.scaling); stage_changes["scaling"] = changed_fraction(pre_scaled, scaled)
    out = frame.copy(); out[cols] = scaled
    return out, {
        "spec": asdict(spec), "order": order, "detected_mask": detected,
        "stage_changed_fraction": stage_changes,
        "cleaning_changed_fraction": changed_fraction(raw, pre_scaled),
        "scaling_changed_fraction": stage_changes["scaling"],
        "overall_changed_fraction": changed_fraction(raw, scaled),
        "scaler": type(scaler).__name__ if scaler is not None else None,
    }


def aggregate_profile(frame: pd.DataFrame, fs: float) -> dict[str, Any]:
    return profile_dataframe(frame, fs, aggregate=True)


def profile_change_table(before: pd.DataFrame, after: pd.DataFrame, fs: float, weights: dict[str, float]) -> pd.DataFrame:
    left = profile_table(before, fs, weights).set_index("feature"); right = profile_table(after, fs, weights).set_index("feature")
    metrics = ["quality_index", "missing_rate", "outlier_iqr_rate", "outlier_mad_rate", "skewness", "kurtosis_excess", "lag1_autocorr", "drift_score", "local_volatility", "spectral_entropy"]
    rows = []
    for f in left.index.intersection(right.index):
        row = {"feature": f}
        for m in metrics:
            row[f"before_{m}"] = left.loc[f, m]; row[f"after_{m}"] = right.loc[f, m]; row[f"delta_{m}"] = right.loc[f, m] - left.loc[f, m]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_stage_methods(frame: pd.DataFrame, stage: str, methods: list[str], fs: float, weights: dict[str, float], base_spec: PipelineSpec, order: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    outputs = {}; rows = []; original_q = quality_index(aggregate_profile(frame, fs), weights)
    for method in methods:
        kwargs = asdict(base_spec); kwargs[stage] = method; spec = PipelineSpec(**kwargs)
        result, meta = execute_pipeline(frame, spec, fs, order)
        outputs[method] = result; p = aggregate_profile(result, fs); q = quality_index(p, weights)
        rows.append({"method": method, "quality_index": q, "quality_delta": q-original_q, "missing_rate": p.get("missing_rate", 0), "outlier_rate": max(p.get("outlier_iqr_rate",0), p.get("outlier_mad_rate",0)), "local_volatility": p.get("local_volatility",0), "cleaning_changed_fraction": meta["cleaning_changed_fraction"], "overall_changed_fraction": meta["overall_changed_fraction"]})
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
    if guard: trace.append(guard)
    validation=dict(STAGE_VALIDATION[stage]); caution=(guard+" " if guard else "")+validation["reason"]
    support = REAL_EYE_TRACKING_PILOT.get(stage, []) if domain=="eye_tracking" else []
    return StageRecommendation(stage,method,validation["status"],trace,caution,support,validation)


def recommend_scaling(profile: dict[str,float], domain: str, goal: str) -> StageRecommendation:
    outlier=max(profile.get("outlier_iqr_rate",0),profile.get("outlier_mad_rate",0)); ku=profile.get("kurtosis_excess",0)
    if goal=="Preserve physical units": method="none"; trace=["The selected objective requires original physical units."]
    elif goal=="Bounded model input [0,1]": method="minmax"; trace=["The selected objective explicitly requires a bounded [0,1] range."]
    elif goal=="Distance/gradient-based model":
        if outlier>.03 or ku>3: method="robust"; trace=[f"Tail diagnostics (outliers={outlier:.2%}, excess kurtosis={ku:.2f}) favour RobustScaler."]
        else: method="standard"; trace=["No strong tail warning; StandardScaler is a reasonable model-oriented default."]
    else:
        if domain=="finance": method="none"; trace=["Finance auto-profile preserves interpretable scale by default."]
        elif outlier>.03 or ku>3: method="robust"; trace=[f"Tail diagnostics (outliers={outlier:.2%}, excess kurtosis={ku:.2f}) favour RobustScaler."]
        else: method="standard"; trace=["Auto mode selects StandardScaler for a comparatively regular distribution."]
    return StageRecommendation("scaling",method,"goal_based",trace,"Scaling is chosen from the analysis goal rather than ranked as a universal cleaning method.",[],{})

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
# Leakage-aware exploratory downstream validation
# ---------------------------------------------------------------------------
def downstream_validation(raw: pd.DataFrame, processed: pd.DataFrame, features: list[str], target: pd.Series) -> pd.DataFrame:
    y=target.reset_index(drop=True); valid=y.notna(); n=int(valid.sum())
    if n<40: return pd.DataFrame()
    yv=y[valid]; classification=yv.nunique()<=max(20,int(np.sqrt(n)))
    rows=[]
    for label,frame in (("Raw",raw),("Processed",processed)):
        X=frame.loc[valid,features].reset_index(drop=True); yy=yv.reset_index(drop=True)
        splitter=TimeSeriesSplit(n_splits=min(5,max(2,n//30))); scores=[]
        for tr,te in splitter.split(X):
            if classification:
                ytr=yy.iloc[tr].astype(str); yte=yy.iloc[te].astype(str)
                if ytr.nunique()<2 or yte.nunique()<2: continue
                model=SkPipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=-1))])
                model.fit(X.iloc[tr],ytr); scores.append(balanced_accuracy_score(yte,model.predict(X.iloc[te])))
                metric="Balanced accuracy"
            else:
                ynum=pd.to_numeric(yy,errors="coerce"); ok=np.isfinite(ynum.to_numpy(float))
                if not ok.all(): continue
                model=SkPipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=120,random_state=42,n_jobs=-1))])
                model.fit(X.iloc[tr],ynum.iloc[tr]); scores.append(r2_score(ynum.iloc[te],model.predict(X.iloc[te]))); metric="R²"
        if scores: rows.append({"data":label,"metric":metric,"cv_mean":float(np.mean(scores)),"cv_sd":float(np.std(scores)),"n_splits_used":len(scores)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Minimal synthetic demos (for a usable standalone app without external files)
# ---------------------------------------------------------------------------
def generate_domain_series(domain: str, n: int=2400, seed: int=42) -> tuple[pd.DataFrame,pd.DataFrame,float]:
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
    return df,event,fs
