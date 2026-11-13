"""Auditable import assistance for heterogeneous real-world eye-tracking tables.

Import suggestions are not a substitute for external acquisition metadata.
The raw frame is never modified; only an explicit, labelled analysis copy is
changed when the operator enables validity interpretation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def timestamp_unit_guidance(df: pd.DataFrame, column: str) -> dict | None:
    """Suggest, but never guarantee, units for unlabelled numeric clocks.

    Tobii Pro Lab recording durations are in milliseconds and let us verify
    the ratio of recorded timestamp span to the duration metadata. For generic
    eye-tracking data, propose a plausible rate as a convenience only.
    """
    if not column or column not in df or len(df) < 3:
        return None
    name = str(column).lower()
    if any(unit in name for unit in ("[ms]", "(ms)", "millisecond", "microsecond", "µs", "usec", "nanosecond")):
        return None
    s = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    finite = s[np.isfinite(s)]
    if len(finite) < 10:
        return None
    steps = np.diff(finite)
    pos = steps[(steps > 0) & np.isfinite(steps)]
    if len(pos) < 5:
        return None
    med = float(np.median(pos))
    # Mean of non-gap intervals corrects rate estimates for 1/2 ms quantised clocks.
    typical = med
    if len(pos)>100:
        q05, q80, q99 = np.quantile(pos, [.05, .80, .99])
        if q05>0 and med <= 2.05*q05 and q80 > 1.35*q05 and q99 <= 3.2*q05:
            typical = float(np.mean(pos[pos <= q99]))
    if "Recording duration" in df.columns and "recording" in name:
        durations = pd.to_numeric(df["Recording duration"], errors="coerce").dropna()
        if len(durations):
            d = float(durations.iloc[0])
            span = float(finite[-1] - finite[0])
            if d > 0 and span > 0:
                ratio = span / d
                # Duration is given in ms by Tobii. Values in us are ~1000x.
                if 950 < ratio < 1050:
                    return {"unit": "Microseconds (µs)", "source": "recording duration metadata", "confidence": "verified",
                            "detail": "The timestamp span matches the export's millisecond recording duration at a 1000:1 ratio."}
                if .95 < ratio < 1.05:
                    return {"unit": "Milliseconds (ms)", "source": "recording duration metadata", "confidence": "verified",
                            "detail": "The timestamp span matches the export's millisecond recording duration."}
    lower_names = " ".join(str(c).lower() for c in df.columns)
    if not any(token in lower_names for token in ("gaze", "pupil", "eye", "fixation")):
        return None
    candidates = [(unit, 1 / (typical * scale)) for unit, scale in
                  (("Seconds (s)", 1.), ("Milliseconds (ms)", .001), ("Microseconds (µs)", .000001), ("Nanoseconds (ns)", 1e-9))]
    plausible = [(unit, rate) for unit, rate in candidates if 30 <= rate <= 2000]
    if len(plausible) != 1:
        return None
    unit, rate = plausible[0]
    return {"unit": unit, "source": "plausible eye-tracking sampling range", "confidence": "unverified",
            "detail": f"The observed timestamp increments are consistent with approximately {rate:.0f} Hz. This is a suggestion, not proof of the clock unit; check the acquisition documentation."}


def data_validity_options(df: pd.DataFrame) -> list[dict]:
    """Identify *explicitly documented in-file* invalid-data indicators."""
    options = []
    if "Sensor" in df and (df["Sensor"].astype("string").str.strip() == "Eye Tracker").any():
        non_gaze = int((df["Sensor"].astype("string").str.strip() != "Eye Tracker").sum())
        if non_gaze:
            options.append({"key": "eye_tracker_rows", "rows": non_gaze,
                            "description": "Use Eye Tracker rows only (exclude mouse/event-only rows from signal analysis)."})
    if "missing" in df and pd.api.types.is_numeric_dtype(df["missing"]):
        flags = pd.to_numeric(df["missing"], errors="coerce")
        count = int(flags.eq(1).sum())
        if count:
            options.append({"key": "missing_flags", "rows": count,
                            "description": "Interpret 'missing = 1' as unavailable signal measurements."})
    if "Validity left" in df or "Validity right" in df:
        invalid = pd.Series(False, index=df.index)
        for side in ("left", "right"):
            col = f"Validity {side}"
            if col in df:
                invalid |= df[col].astype("string").str.strip().str.casefold().eq("invalid").fillna(False)
        if invalid.any():
            options.append({"key": "validity_flags", "rows": int(invalid.sum()),
                            "description": "Interpret explicit Invalid eye validity labels as unavailable gaze channels."})
    return options


def gaze_rows_only(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return *analysis view*, never overwrite original uploaded rows."""
    if "Sensor" not in df:
        return df.copy(), {"applied": False, "excluded_rows": 0}
    keep = df["Sensor"].astype("string").str.strip().eq("Eye Tracker").fillna(False).to_numpy(dtype=bool)
    result = df.iloc[np.flatnonzero(keep)].reset_index(drop=True).copy()
    return result, {"applied": True, "excluded_rows": int((~keep).sum()), "kept_rows": int(keep.sum()),
                    "reason": "Sensor == Eye Tracker; mouse and event rows remain in original uploaded file"}


def interpret_validity_flags(df: pd.DataFrame, signals: list[str],
                             *, missing_flag: bool = False, eye_validity: bool = False) -> tuple[pd.DataFrame, dict]:
    """Mask explicitly invalid samples in *selected* signals only (time unchanged)."""
    out = df.copy()
    changes: list[dict] = []
    if missing_flag and "missing" in df:
        invalid = pd.to_numeric(df["missing"], errors="coerce").eq(1).fillna(False)
        for col in signals:
            if col in out:
                before = int(pd.to_numeric(out[col], errors="coerce").isna().sum())
                out.loc[invalid, col] = np.nan
                changes.append({"signal": col, "reason": "missing == 1", "flagged_rows": int(invalid.sum()),
                                "new_missing": int(pd.to_numeric(out[col], errors="coerce").isna().sum())-before})
    if eye_validity:
        for side in ("left", "right"):
            key = f"Validity {side}"
            if key not in df:
                continue
            invalid = df[key].astype("string").str.strip().str.casefold().eq("invalid").fillna(False)
            for col in signals:
                low = col.lower()
                if col in out and side in low and any(token in low for token in ("gaze", "pupil", "eye position")):
                    before = int(pd.to_numeric(out[col], errors="coerce").isna().sum())
                    out.loc[invalid, col] = np.nan
                    changes.append({"signal": col, "reason": f"{key} == Invalid", "flagged_rows": int(invalid.sum()),
                                    "new_missing": int(pd.to_numeric(out[col], errors="coerce").isna().sum())-before})
    return out, {"mode": "explicit validity flags", "changes": changes, "raw_preserved": True}


def paired_zero_gaze_signals(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Detect, but do not automatically reinterpret, paired zero gaze channels."""
    pairs = []
    names = list(df.columns)
    for x in names:
        low = str(x).lower()
        if not ("gaze" in low or "point of regard" in low):
            continue
        candidate = None
        if "gaze_x_" in low:
            candidate = x.replace("gaze_x_", "gaze_y_")
        elif low.endswith(" x"):
            candidate = x[:-2] + " Y"
        elif " x [" in low:
            candidate = x.replace(" X [", " Y [")
        elif " x (" in low:
            candidate = x.replace(" X (", " Y (")
        if candidate and candidate in names:
            xv = pd.to_numeric(df[x],errors="coerce")
            yv = pd.to_numeric(df[candidate],errors="coerce")
            n = int((xv.eq(0) & yv.eq(0)).sum())
            if n:
                pairs.append((x, candidate, n))
    return pairs


def interpret_paired_zero_gaze(df: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Optional, auditable assumption: (0,0) gaze means unavailable."""
    out = df.copy()
    change = []
    for x, y, n in paired_zero_gaze_signals(df):
        invalid = pd.to_numeric(df[x],errors="coerce").eq(0) & pd.to_numeric(df[y],errors="coerce").eq(0)
        for c in (x,y):
            if c in signals:
                existing = pd.to_numeric(out[c],errors="coerce").isna().sum()
                out.loc[invalid,c] = np.nan
                change.append({"signal":c,"reason":"optional paired (0,0) assumption", "flagged_rows":n,
                               "new_missing":int(pd.to_numeric(out[c],errors="coerce").isna().sum()-existing)})
    return out,change
