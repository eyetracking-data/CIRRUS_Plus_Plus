"""Definitions of the columns in Explore all diagnostic measurements."""

import streamlit as st


MEASUREMENT_DEFINITIONS = [
    ("Signal", "The name of the selected numeric signal column.",
     "Each table row describes one signal before pipeline execution."),
    ("Quality score", "A 0–100 heuristic combining six component scores using the active domain weights.",
     "Higher means a higher heuristic score, not proven signal accuracy. The full component formulas are explained under Understand this score in Data overview."),
    ("Missing (%)", "100 × number of missing/non-finite values / total samples. NaN and positive or negative infinity count as missing; numeric zeros remain observations.",
     "The share of samples with no available numeric value. A low percentage does not establish measurement validity."),
    ("p95 gap (s)", "The 95th percentile of consecutive missing-run durations. Each duration runs from the first missing sample to the next observed sample; at the recording end, one median timestamp interval is added after the final sample. Zero if no gaps exist.",
     "Describes the upper end of gap durations; it is not the longest gap. Interruptions without missing rows are not counted as missing-value runs."),
    ("IQR flags (%)", "100 × flagged available values / available values. With IQR = Q3 − Q1, flag values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. Zero IQR gives no flags.",
     "Statistical extremes relative to the middle 50% of observations. Flags do not prove measurement errors."),
    ("MAD flags (%)", "100 × flagged available values / available values. MAD is median(abs(value − median)); flag abs(0.67448975 × (value − median) / MAD) > 3.5. Zero MAD gives no flags.",
     "Statistical extremes relative to median absolute deviation. These profiling flags do not execute the selected outlier-treatment method."),
    ("Gap concentration", "A 0–1 heuristic: clip(0.65 × longest gap length / all missing samples + 2.8 × mean gap length / total samples, 0, 1). Gap lengths here are sample counts. Zero if there are no gaps.",
     "Higher values indicate missing samples concentrated into gaps. This is neither a percentage nor a duration."),
    ("Skewness", "Bias-corrected standardized third central moment of available values, without gap interpolation.",
     "Positive: a longer tail toward larger values. Negative: a longer tail toward smaller values. Near zero does not establish normality."),
    ("Excess kurtosis", "Bias-corrected standardized fourth central moment relative to the normal-distribution reference; normal excess kurtosis is zero. Computed from available values.",
     "Positive indicates heavier tails; negative indicates lighter tails. Rare genuine events can produce heavy tails."),
    ("Drift", "After temporary gap filling, compute rolling means with window max(10, floor(N/10)) and at least max(5, floor(window/3)) observations. Drift = clip(range of rolling means / (4 × sample standard deviation), 0, 1). Returns zero for fewer than 30 samples, constant signals or unavailable rolling means.",
     "Measures slow changes in the local average relative to signal variation. Genuine trends can also produce high drift."),
    ("Local volatility", "Population standard deviation of first differences / (population standard deviation of the temporarily filled signal + 1e−12).",
     "Higher values indicate stronger changes between adjacent samples relative to overall variability. This is not a percentage."),
    ("Mean absolute correlation", "For this signal, average absolute Pearson correlations with the other selected signals, using pairwise available observations without interpolation. Undefined pairs are omitted; with one signal, the fallback is zero.",
     "Values near one indicate strong linear relationships, including opposite directions. Independent signals can still be valid; insufficient overlap can make the result undefined."),
    ("Longest gap (s)", "The longest consecutive missing-run duration, measured on the timestamp axis using the same duration convention as p95 gap. Zero if there are no gaps.",
     "The longest observed interruption represented by missing samples; it can differ substantially from p95 gap."),
    ("Lag-1 autocorrelation", "Pearson correlation between adjacent samples after temporary gap filling. Undefined correlation falls back to zero.",
     "High values can reflect smooth dynamics; low or negative values can reflect fast or alternating changes. Neither establishes signal quality."),
    ("Spectral entropy", "After temporary gap filling and mean removal, estimate Welch power at the median-derived sampling rate with segments up to 256 samples. Normalize power bins p to sum to one; entropy = −sum(p × log(p + 1e−12)) / log(number of bins).",
     "Near zero: concentrated power. Near one: diffuse power. Neither direction is universally better. Unavailable for irregular timestamps."),
    ("Dominant-frequency share (%)", "100 × the largest normalized Welch power bin, using the same spectrum as spectral entropy.",
     "The share of power in the strongest frequency bin, not a frequency in Hz. Unavailable for irregular timestamps."),
]


def render_signal_calculation_guide():
    st.button("← Back to analysis", key="close_signal_calculation_guide",
              on_click=st.session_state.update, kwargs={"signal_calculation_guide_open": False})
    st.title("Diagnostic measurements explained")
    st.write("Definitions, calculations and interpretation of the columns in Explore all diagnostic measurements. Each measurement describes one selected signal before pipeline execution.")
    for label, definition, interpretation in MEASUREMENT_DEFINITIONS:
        st.markdown(f"### {label}")
        st.markdown(f"**Definition & calculation:** {definition}\n\n**Interpretation:** {interpretation}")
    st.caption("Temporal and spectral diagnostics temporarily fill gaps by linear interpolation over row positions with endpoint filling. This does not modify your data or use the pipeline gap-duration limit. Autocorrelation, drift and volatility remain sample-based for irregular timestamps. With fewer than four available observations, several diagnostics return fallback zeros; these are not evidence of a clean signal. Constant or sparse signals may yield undefined values. For regular timestamps, spectral diagnostics also return zero for fewer than 16 samples or zero signal power.")
