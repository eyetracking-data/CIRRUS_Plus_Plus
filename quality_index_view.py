"""Read-only explanation of the current intrinsic quality calculation."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from cirruspp_core import quality_breakdown, quality_index
from quality_visuals import COMPONENT_COLORS, component_rgba, render_quality_calculation


# Meaning comes before implementation details; none of these descriptions change scoring.
COMPONENT_GUIDE = {
    "Completeness": (
        "How much of the signal is available?",
        "Checks whether the recording contains usable numbers where measurements are expected. A high score means fewer missing observations; it does not tell us whether the observed values are correct.",
        ["missing_rate"],
    ),
    "Continuity": (
        "How interrupted is the recording?",
        "Looks at the pattern of missing observations. Scattered missing points and a long uninterrupted gap can have the same missing percentage, but very different consequences for interpreting a movement.",
        ["missing_burstiness", "missing_run_p95_seconds"],
    ),
    "Outlier burden": (
        "How many values stand out from the rest?",
        "Measures how often observations are unusually far from the signal's typical range. A high component score means fewer statistical flags. A flag is a reason to inspect a value, not proof of a measurement error.",
        ["outlier_iqr_rate", "outlier_mad_rate"],
    ),
    "Distribution shape": (
        "Are the values asymmetric or heavy-tailed?",
        "Describes the shape of the value distribution, without considering when values occurred. Strong asymmetry or heavy tails lower this component, although both can be natural properties of a real signal.",
        ["skewness", "kurtosis_excess"],
    ),
    "Temporal stability": (
        "Does the signal drift or change abruptly?",
        "Combines slow changes in the local average with rapid sample-to-sample variation. A high score indicates less of these changes under this heuristic. Real movements and events can also lower the score.",
        ["drift_score", "local_volatility"],
    ),
    "Cross-signal consistency": (
        "How strongly do the signals move together?",
        "Measures the strength of relationships between selected signals. Both matching and opposite directions count because absolute correlation is used. Strong relationships raise the score, but independent signals can still be perfectly valid.",
        ["corr_mean_abs"],
    ),
}

DIAGNOSTICS = {
    "missing_rate": ("Missing values", "The share of observations with missing or non-finite numeric values.", "0% means no missing observations. Higher percentages mean less observed data.", "percent"),
    "missing_burstiness": ("Gap concentration", "A 0–1 heuristic combining the largest gap's share of all missing observations with the average gap length relative to the recording.", "Higher values mean stronger concentration into gaps. It is not a missing-data percentage.", "number"),
    "missing_run_p95_seconds": ("95th-percentile gap duration", "Measure missing runs on the timestamp axis: this is their 95th-percentile duration. The final sample uses the median interval. It is zero when there are no gaps.", "Longer durations mean longer interruptions near the upper end of the gap distribution. This is not the longest gap.", "seconds"),
    "outlier_iqr_rate": ("IQR outlier rate", "The share of available observations far outside the middle 50% of values, using the interquartile-range rule.", "Higher percentages mean more statistically extreme values. Genuine events may also be flagged.", "percent"),
    "outlier_mad_rate": ("MAD outlier rate", "The share of available observations unusually far from the median, measured against the median absolute deviation.", "Higher percentages mean more extreme values relative to the signal's typical deviation from its median.", "percent"),
    "skewness": ("Skewness", "Describes asymmetry: positive values indicate a longer tail toward larger values; negative values indicate a longer tail toward smaller values.", "Near zero means roughly balanced tails. It does not imply a normal distribution. The index penalizes the magnitude after taking the median across signals.", "number"),
    "kurtosis_excess": ("Excess kurtosis", "Describes tail weight relative to a normal distribution. Positive values indicate heavier tails; negative values indicate lighter tails.", "Zero is the normal-distribution reference. Only positive excess kurtosis is penalized here; heavy tails can reflect genuine rare events.", "number"),
    "drift_score": ("Drift", "A 0–1 measure of how much the rolling average moves relative to the signal's overall variability.", "Near zero means little movement in the local average; larger values mean stronger slow change. Real trends can create drift.", "number"),
    "local_volatility": ("Local volatility", "Compares variability between consecutive observations with variability of the signal as a whole.", "Larger values indicate stronger rapid changes relative to the overall signal variation. This can reflect noise or meaningful fast events; the value is not a percentage.", "number"),
    "corr_mean_abs": ("Mean absolute correlation", "For each signal, averages the strengths of its Pearson correlations with the other selected signals. The index then uses the median of these per-signal averages.", "Values near 1 mean strong linear relationships, including opposite directions; near 0 means weak linear relationships. With one selected signal, the implementation uses 0 as a fallback.", "number"),
}

MEASUREMENT_LABELS = {
    "feature": "Signal", "samples": "Samples", "finite_values": "Available values",
    "missing_values": "Missing values (count)", "gap_count": "Number of gaps",
    "longest_gap_samples": "Longest gap (samples)", "mean_gap_samples": "Mean gap (samples)",
    "p95_gap_samples": "95th-percentile gap (samples)", "q1": "25th percentile (Q1)",
    "q3": "75th percentile (Q3)", "iqr": "Interquartile range", "median": "Median value",
    "mad": "Median absolute deviation", "iqr_lower_bound": "IQR lower limit",
    "iqr_upper_bound": "IQR upper limit", "iqr_flagged": "IQR flagged (count)",
    "mad_flagged": "MAD flagged (count)", "central_moment_2": "2nd central moment",
    "central_moment_3": "3rd central moment", "central_moment_4": "4th central moment",
    "rolling_window_samples": "Rolling window (samples)", "rolling_mean_range": "Range of rolling means",
    "sample_std": "Signal std. dev. (sample)", "population_std": "Signal std. dev. (population)",
    "difference_std": "First-difference std. dev.",
    **{key: value[0] for key, value in DIAGNOSTICS.items()},
}


def _format_diagnostic(value, kind):
    if not np.isfinite(value):
        return "Undefined"
    if kind == "percent":
        return f"{value:.2%}"
    return f"{value:.4g}" + (" s" if kind == "seconds" else "")


def _measurement_table(frame):
    """Display rates as percentages and keep calculation inputs at full precision."""
    shown = frame.copy()
    for key, (_, _, _, kind) in DIAGNOSTICS.items():
        if key in shown and kind in {"percent", "seconds"}:
            shown[key] = shown[key].map(lambda value: _format_diagnostic(value, kind))
    st.dataframe(shown.rename(columns=MEASUREMENT_LABELS), hide_index=True, width="stretch")


def _render_diagnostic_meaning(key, profile, component):
    label, meaning, interpretation, kind = DIAGNOSTICS[key]
    st.markdown(
        f'<article class="meaning-measurement" style="--meaning-color:{COMPONENT_COLORS[component]};--meaning-tint:{component_rgba(component,.045)}">'
        f'<div class="meaning-measurement-body"><h4>{escape(label)}</h4><p>{escape(meaning)}</p></div>'
        f'<div class="meaning-measurement-value"><span>Current median</span><strong>{escape(_format_diagnostic(profile[key],kind))}</strong></div>'
        f'<p class="meaning-measurement-reading"><b>How to read it</b><br>{escape(interpretation)}</p></article>',
        unsafe_allow_html=True,
    )

def _signal_measurements(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Expose the underlying measurements without changing the scoring code."""
    rows = []
    for name in features.feature:
        x = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(x)
        finite = x[~missing]
        edges = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
        runs = edges[1::2] - edges[::2]
        q1, q3 = np.percentile(finite, [25, 75]) if len(finite) else (np.nan, np.nan)
        median = np.median(finite) if len(finite) else np.nan
        mad = np.median(np.abs(finite - median)) if len(finite) else np.nan
        iqr = q3 - q1
        interp = pd.Series(x).interpolate(limit_direction="both")
        window = max(10, len(x) // 10)
        rolling = interp.rolling(window, min_periods=max(5, window // 3)).mean().dropna()
        centered = finite - finite.mean() if len(finite) else finite
        rows.append({
            "feature": name, "samples": len(x), "finite_values": len(finite),
            "missing_values": int(missing.sum()), "gap_count": len(runs),
            "longest_gap_samples": int(runs.max()) if len(runs) else 0,
            "mean_gap_samples": float(runs.mean()) if len(runs) else 0.,
            "p95_gap_samples": float(np.quantile(runs, .95)) if len(runs) else 0.,
            "q1": q1, "q3": q3, "iqr": iqr, "median": median, "mad": mad,
            "iqr_lower_bound": q1 - 1.5 * iqr, "iqr_upper_bound": q3 + 1.5 * iqr,
            "iqr_flagged": int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum()) if len(finite) >= 4 and iqr > 0 else 0,
            "mad_flagged": int((np.abs(.67448975 * (finite - median) / mad) > 3.5).sum()) if len(finite) >= 4 and mad > 0 else 0,
            "central_moment_2": float(np.mean(centered**2)) if len(finite) else np.nan,
            "central_moment_3": float(np.mean(centered**3)) if len(finite) else np.nan,
            "central_moment_4": float(np.mean(centered**4)) if len(finite) else np.nan,
            "rolling_window_samples": window,
            "rolling_mean_range": float(rolling.max() - rolling.min()) if len(rolling) else np.nan,
            "sample_std": float(interp.std()),
            "population_std": float(np.nanstd(interp)),
            "difference_std": float(np.nanstd(np.diff(interp))) if len(interp) > 1 else np.nan,
        })
    return pd.DataFrame(rows).merge(features, on="feature", validate="one_to_one")


def render_quality_details(*, source_name, domain_label, validity_mode, timestamp_choice,
                           fs, canonical, profile_features, profile_agg, weights):
    st.button("← Back to Data overview", on_click=st.session_state.update,
              kwargs={"quality_details_open": False}, key="close_quality_details")
    st.markdown("""<style>
.quality-heading {font-size:1.9rem;font-weight:650;letter-spacing:-.035em;line-height:1.2;margin:.5rem 0;}
.quality-explainer {border:1px solid rgba(68,119,170,.24);border-left:4px solid #4477aa;border-radius:12px;padding:22px 24px;margin:16px 0 20px;color:var(--text-color);background:rgba(68,119,170,.035);}
.quality-explainer-lead {font-size:1.15rem;font-weight:600;line-height:1.5;margin:0 0 12px;max-width:90ch;}
.quality-explainer p {font-size:.95rem;line-height:1.7;margin:0 0 12px;max-width:100ch;}
.quality-explainer p:last-child {margin-bottom:0;}
.quality-explainer-reading {border-top:1px solid rgba(68,119,170,.18);padding-top:12px;}
.quality-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 8px;}
.quality-card {border:1px solid rgba(128,145,170,.24);border-top:3px solid var(--component-color);border-radius:12px;padding:16px;box-sizing:border-box;display:flex;flex-direction:column;min-width:0;background:linear-gradient(180deg,var(--component-tint),transparent 110px);}
.quality-card-header {display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px;}
.quality-card-title {font-size:.93rem;font-weight:650;line-height:1.35;}
.quality-card-score {font-size:1.5rem;font-weight:650;letter-spacing:-.035em;white-space:nowrap;color:var(--component-color);}
.quality-card-score span {font-size:.75rem;font-weight:400;color:var(--text-color);opacity:.6;letter-spacing:0;}
.quality-card-meter {height:5px;border-radius:999px;background:var(--component-track);margin:0 0 16px;overflow:hidden;}
.quality-card-meter-fill {height:100%;border-radius:inherit;background:var(--component-color);}
.quality-card-question {font-size:.85rem;line-height:1.4;margin:0 0 10px;}
.quality-card-inputs {display:grid;gap:4px;margin-bottom:12px;}
.quality-card-input {display:flex;justify-content:space-between;gap:10px;font-size:.76rem;line-height:1.4;}
.quality-card-input span {opacity:.72;}.quality-card-input b {font-weight:550;white-space:nowrap;}
.quality-card-weight {display:flex;justify-content:space-between;gap:10px;border-top:1px solid rgba(128,145,170,.2);padding-top:9px;margin-top:auto;font-size:.76rem;}
.quality-card-weight span {opacity:.7;}
.meaning-intro {font-size:.9rem;line-height:1.6;margin:0 0 16px;}
.st-key-quality_component_picker {margin:0 0 16px;}
.st-key-quality_component_picker button {min-height:52px;justify-content:flex-start;padding:12px 16px;border-radius:10px;}
.st-key-quality_component_picker button p {font-size:.84rem;font-weight:500;white-space:normal;text-align:left;}
.meaning-overview {border:1px solid rgba(128,145,170,.22);border-left:4px solid var(--meaning-color);background:var(--meaning-tint);border-radius:12px;padding:22px 24px;margin:0 0 6px;}
.meaning-component {color:var(--meaning-color);font-size:.8rem;font-weight:650;margin-bottom:8px;}
.meaning-overview h3 {font-size:1.3rem;line-height:1.35;letter-spacing:-.025em;margin:0 0 12px;padding:0;}
.meaning-overview p {font-size:.9rem;line-height:1.65;margin:0;max-width:95ch;}
.meaning-measurement {display:grid;grid-template-columns:minmax(0,1fr) minmax(140px,180px);gap:20px 28px;border:1px solid rgba(128,145,170,.22);border-radius:12px;padding:22px 24px;}
.meaning-measurement h4 {font-size:1rem;font-weight:600;margin:0 0 10px;padding:0;}
.meaning-measurement-body p {font-size:.87rem;line-height:1.65;margin:0;max-width:95ch;}
.meaning-measurement-value {display:flex;flex-direction:column;justify-content:center;gap:8px;background:var(--meaning-tint);border-radius:10px;padding:16px;}
.meaning-measurement-value span {font-size:.73rem;}
.meaning-measurement-value strong {color:var(--meaning-color);font-size:1.8rem;font-weight:600;letter-spacing:-.03em;line-height:1.2;}
.meaning-measurement-reading {grid-column:1/-1;border-top:1px solid rgba(128,145,170,.15);padding-top:14px;margin:0;font-size:.82rem;line-height:1.6;}
.calculation-intro {font-size:.95rem;line-height:1.65;margin:0 0 22px;max-width:90ch;}
.calculation-step {display:flex;gap:12px;align-items:center;margin:24px 0 12px;}
.calculation-step-number {display:grid;place-items:center;width:32px;height:32px;flex-shrink:0;border-radius:10px;background:rgba(68,119,170,.09);color:#4477aa;font-size:.85rem;font-weight:650;}
.calculation-step h3 {font-size:1.2rem;font-weight:600;margin:0;padding:0;letter-spacing:-.02em;}
.calculation-step-copy {font-size:.88rem;line-height:1.65;margin:0 0 16px;max-width:100ch;}
[class*="st-key-calculation_component_"] {border-radius:12px;padding:18px;}
.calculation-component-header {display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:12px;}
.calculation-component-name {font-size:.95rem;font-weight:650;}
.calculation-component-value {font-size:1.65rem;font-weight:600;letter-spacing:-.025em;white-space:nowrap;}
.calculation-component-value small {font-size:.72rem;font-weight:400;color:var(--text-color);opacity:.65;}
.calculation-component-contribution {font-size:.82rem;padding:10px 12px;border-radius:8px;line-height:1.5;}
.st-key-calculation_result {padding:22px;border-radius:14px;border-color:rgba(68,119,170,.24);}
.calculation-convention {font-size:.82rem;line-height:1.6;padding:12px 16px;background:rgba(128,145,170,.05);border-radius:10px;margin:0 0 16px;}
@media(max-width:640px) {.meaning-measurement {grid-template-columns:1fr;padding:18px;}.meaning-overview {padding:18px;}}
.st-key-quality_next_steps {margin-top:16px;}
.st-key-quality_open_measurements button,.st-key-quality_open_calculation button {
position:relative;min-height:100px;padding:20px 62px 20px 22px;border-radius:12px;
border:1px solid rgba(68,119,170,.22);background:rgba(68,119,170,.025);
text-align:left;justify-content:flex-start;box-shadow:0 2px 8px rgba(15,23,42,.025);
transition:background .15s,border-color .15s,box-shadow .15s,transform .15s;}
.st-key-quality_open_measurements button p,.st-key-quality_open_calculation button p {
font-size:.82rem;line-height:1.55;text-align:left;margin:0;white-space:normal;overflow:visible;text-overflow:clip;}
.st-key-quality_open_measurements button strong,.st-key-quality_open_calculation button strong {
display:block;font-size:1rem;font-weight:600;line-height:1.35;margin-bottom:6px;letter-spacing:-.015em;}
.st-key-quality_open_measurements button::after,.st-key-quality_open_calculation button::after {
content:"→";position:absolute;right:20px;top:50%;transform:translateY(-50%);
display:grid;place-items:center;width:30px;height:30px;border-radius:50%;
background:rgba(68,119,170,.08);color:#4477aa;font-size:1.1rem;}
.st-key-quality_open_measurements button:hover,.st-key-quality_open_calculation button:hover {
border-color:#4477aa;background:rgba(68,119,170,.06);color:var(--text-color);
box-shadow:0 4px 12px rgba(68,119,170,.09);transform:translateY(-2px);}
.st-key-quality_open_measurements button:focus-visible,.st-key-quality_open_calculation button:focus-visible {
outline:2px solid #4477aa;outline-offset:3px;}
@media(max-width:1100px) {.quality-grid {grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:640px) {.quality-grid {grid-template-columns:1fr;}.quality-heading {font-size:1.6rem;}}
</style>""", unsafe_allow_html=True)
    st.markdown('<h1 class="quality-heading">Understand your quality score</h1>', unsafe_allow_html=True)
    breakdown = quality_breakdown(profile_agg, weights)
    total = quality_index(profile_agg, weights)
    overview, meaning, calculation = st.tabs(
        ["Score overview", "Measurements & meaning", "Full calculation"],
        key="quality_details_tab", on_change="rerun",
    )
    with overview:
        st.markdown('''<section class="quality-explainer" aria-label="What the quality score means">
<div class="quality-explainer-lead">The quality score summarizes six aspects of your selected signals in one number from 0 to 100.</div>
<p>It looks at missing values, gaps, statistical outliers, distribution shape, changes over time, and relationships between signals. Each aspect receives its own score.</p>
<p>These six scores are combined using <strong>domain weights</strong>: aspects that matter more for the active domain have a greater influence on the result.</p>
<p class="quality-explainer-reading"><strong>How to read the result:</strong> Higher values indicate a more favorable diagnostic profile under these criteria.</p>
</section>''', unsafe_allow_html=True)
        st.subheader("Six components, one weighted score")
        st.markdown("Each card shows a component’s score, the measurements behind it, and its weighted contribution to your total.")
        if len(profile_features) == 1:
            st.caption("Only one signal is selected: cross-signal consistency uses a fallback score of 25/100 because no signal pair is available.")
        cards = []
        for _, row in breakdown.iterrows():
            question, _, inputs = COMPONENT_GUIDE[row.component]
            card_style = f"--component-color:{COMPONENT_COLORS[row.component]};--component-tint:{component_rgba(row.component,.055)};--component-track:{component_rgba(row.component,.13)};"
            meter = (
                f'<div class="quality-card-meter" role="meter" aria-label="{escape(row.component)} score" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{row.score_0_100:.4f}"><div class="quality-card-meter-fill" style="width:{np.clip(row.score_0_100,0,100):.4f}%"></div></div>'
                if np.isfinite(row.score_0_100) else '<div class="quality-card-meter" aria-label="Score unavailable"></div>'
            )
            observed = "".join(
                f'<div class="quality-card-input"><span>{escape(DIAGNOSTICS[key][0])}</span><b>{escape(_format_diagnostic(profile_agg[key], DIAGNOSTICS[key][3]))}</b></div>'
                for key in inputs
            )
            cards.append(f'<div class="quality-card" style="{card_style}"><div class="quality-card-header"><div class="quality-card-title">{escape(row.component)}</div><div class="quality-card-score">{row.score_0_100:.1f}<span> / 100</span></div></div>{meter}<div class="quality-card-question">{escape(question)}</div><div class="quality-card-inputs">{observed}</div><div class="quality-card-weight"><span>{row.domain_weight:.0%} domain weight</span><b>{row.weighted_points:.2f} points</b></div></div>')
        st.markdown('<div class="quality-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)
        with st.container(key="quality_next_steps"):
            explain_measurements, explain_calculation = st.columns(2,gap="medium")
        explain_measurements.button(
            "**Understand the measurements**\nExplore what each diagnostic tells you about your signals.", key="quality_open_measurements",
            on_click=st.session_state.update,
            kwargs={"quality_details_tab": "Measurements & meaning"}, width="stretch",
        )
        explain_calculation.button(
            "**Follow the calculation**\nTrace each component score, its weight, and the final sum.", key="quality_open_calculation",
            on_click=st.session_state.update,
            kwargs={"quality_details_tab": "Full calculation"}, width="stretch",
        )
    measurements = _signal_measurements(canonical, profile_features)
    p = profile_agg
    f = lambda value: f"{value:.8g}"
    definitions = [
        ("Completeness", ["samples", "finite_values", "missing_values", "missing_rate"],
         "For each signal, missing rate = non-finite values / sample count. Non-finite includes NaN and infinity.",
         "100 × (1 − clip(missing_rate / 0.45))",
         f"100 × (1 − clip({f(p['missing_rate'])} / 0.45))",
         "The score reaches zero at 45% missingness. This threshold is a heuristic, not a universal acceptability limit."),
        ("Continuity", ["samples", "missing_values", "gap_count", "longest_gap_samples", "mean_gap_samples", "p95_gap_samples", "missing_burstiness", "missing_run_p95_seconds"],
         "A gap is a consecutive run of non-finite values. Per signal: gap_concentration = clip(0.65 × longest gap / total missing values + 2.8 × mean gap / sample count). With no gaps it is zero. Higher gap concentration means missing observations are more strongly grouped into gaps; it is not a percentage of missing data. The 95th-percentile gap duration uses elapsed timestamps from each run start to the next observed position; a run ending at the final sample uses the median interval for that last sample. The median gap concentration and median duration are used below.",
         "100 × (1 − clip(0.55 × gap_concentration + 0.45 × gap_seconds / (1 + gap_seconds)))",
         f"100 × (1 − clip(0.55 × {f(p['missing_burstiness'])} + 0.45 × {f(p['missing_run_p95_seconds'])} / (1 + {f(p['missing_run_p95_seconds'])})))",
         "The same missing fraction can yield different continuity scores when gaps have different lengths and concentrations."),
        ("Outlier burden", ["finite_values", "q1", "q3", "iqr", "iqr_lower_bound", "iqr_upper_bound", "iqr_flagged", "outlier_iqr_rate", "median", "mad", "mad_flagged", "outlier_mad_rate"],
         "On finite observations, Q1 and Q3 are the 25th and 75th percentiles; IQR = Q3 − Q1. IQR flags lie strictly outside Q1 − 1.5 × IQR and Q3 + 1.5 × IQR. MAD is median(|x − median(x)|); MAD flags satisfy |0.67448975 × (x − median) / MAD| > 3.5. Each rate is flagged count / finite count. If IQR or MAD is zero, its rate is zero. Take the median of each rate across signals, then the larger of those two medians.",
         "100 × (1 − clip(max(IQR_rate, MAD_rate) / 0.25))",
         f"100 × (1 − clip(max({f(p['outlier_iqr_rate'])}, {f(p['outlier_mad_rate'])}) / 0.25))",
         "Statistical extremes can be real events. Fewer flagged values do not establish better signal fidelity."),
        ("Distribution shape", ["finite_values", "central_moment_2", "central_moment_3", "central_moment_4", "skewness", "kurtosis_excess"],
         "For each signal, use finite values and central moments mr = mean((x − mean(x))^r). With n finite values, bias-corrected skewness = sqrt(n × (n−1)) / (n−2) × m3 / m2^1.5. Bias-corrected excess kurtosis = ((n²−1) × m4 / m2² − 3 × (n−1)²) / ((n−2) × (n−3)). These match scipy.stats.skew(bias=False) and kurtosis(fisher=True, bias=False). Take the median of each signed diagnostic, then apply absolute skewness and non-negative excess kurtosis below.",
         "100 × (1 − clip(0.6 × |skewness| / 3 + 0.4 × max(0, excess_kurtosis) / 10))",
         f"100 × (1 − clip(0.6 × |{f(p['skewness'])}| / 3 + 0.4 × max(0, {f(p['kurtosis_excess'])}) / 10))",
         "Asymmetric or heavy-tailed distributions can be legitimate. Constant signals can have undefined moments; this is not evidence of high quality."),
        ("Temporal stability", ["rolling_window_samples", "rolling_mean_range", "sample_std", "population_std", "difference_std", "drift_score", "local_volatility"],
         "For these diagnostics only, missing values are linearly interpolated in both directions; the input is unchanged. The rolling window is max(10, floor(sample count / 10)), with at least max(5, floor(window / 3)) observations. Drift = clip(range of rolling means / (4 × sample standard deviation)). Drift is zero for fewer than 30 samples, zero standard deviation or no rolling means. Volatility = population standard deviation of first differences / (population standard deviation of interpolated signal + 10⁻¹²). Take the median drift and median volatility across signals.",
         "100 × (1 − clip(0.65 × drift + 0.35 × volatility / (1 + volatility)))",
         f"100 × (1 − clip(0.65 × {f(p['drift_score'])} + 0.35 × {f(p['local_volatility'])} / (1 + {f(p['local_volatility'])})))",
         "True signal changes can lower this component. A smoother signal is not automatically more accurate."),
        ("Cross-signal consistency", ["corr_mean_abs"],
         "Compute absolute Pearson correlations between selected signal pairs, using pairwise available observations. For each signal, average its correlations with the other signals, excluding itself and undefined pairs. Then take the median of these per-signal means. With one signal the correlation input is zero, giving a component score of 25, not a measured consistency result.",
         "100 × clip(0.25 + 0.75 × mean_absolute_correlation)",
         f"100 × clip(0.25 + 0.75 × {f(p['corr_mean_abs'])})",
         "Correlated channels and duplicate signals can inflate this score. Independent signals are not necessarily poor quality."),
    ]
    with meaning:
        st.subheader("What does each measurement tell you?")
        st.markdown('<p class="meaning-intro">Choose a component to explore what it measures and how to interpret your current values.</p>',unsafe_allow_html=True)
        selected = st.session_state.get("quality_meaning_component","Completeness")
        if selected not in COMPONENT_GUIDE:
            selected = "Completeness"
        picker_styles = []
        with st.container(key="quality_component_picker"):
            for start in (0,3):
                columns = st.columns(3,gap="small")
                for index,(column,component) in enumerate(zip(columns,list(COMPONENT_GUIDE)[start:start+3]),start=start):
                    picker_key = f"quality_meaning_select_{index}"
                    color = COMPONENT_COLORS[component]
                    active = component == selected
                    picker_styles.append(
                        f'.st-key-{picker_key} button {{border-color:{color if active else "rgba(128,145,170,.22)"};'
                        f'background:{component_rgba(component,.1 if active else .025)};color:var(--text-color);}}'
                        f'.st-key-{picker_key} button::before {{content:"";width:8px;height:8px;flex-shrink:0;border-radius:50%;background:{color};margin-right:8px;}}'
                        f'.st-key-{picker_key} button:hover {{border-color:{color};background:{component_rgba(component,.1)};}}'
                    )
                    column.button(component,key=picker_key,width="stretch",on_click=st.session_state.update,
                                  kwargs={"quality_meaning_component":component})
        st.markdown('<style>'+''.join(picker_styles)+'</style>',unsafe_allow_html=True)
        question, interpretation, inputs = COMPONENT_GUIDE[selected]
        st.markdown(
            f'<section class="meaning-overview" style="--meaning-color:{COMPONENT_COLORS[selected]};--meaning-tint:{component_rgba(selected,.05)}">'
            f'<div class="meaning-component">{escape(selected)}</div><h3>{escape(question)}</h3><p>{escape(interpretation)}</p></section>',
            unsafe_allow_html=True,
        )
        for key in inputs:
            _render_diagnostic_meaning(key, profile_agg, selected)
        if selected == "Cross-signal consistency" and len(profile_features) == 1:
            st.info("Only one signal is selected. Cross-signal consistency cannot be measured; the implementation uses correlation 0, which gives a fallback component score of 25/100.")
        with st.expander("Compare these measurements across your signals"):
            _measurement_table(profile_features[["feature"] + inputs])
            st.caption("The cards show the median across selected signals. The table shows each individual signal, so differences remain visible.")
        _, _, _, _, _, caution = next(d for d in definitions if d[0] == selected)
        st.caption(caution)

    with calculation:
        st.subheader("How your quality index is calculated")
        st.markdown('<p class="calculation-intro">Follow three steps: combine measurements across your signals, calculate six component scores, then apply the domain weights and add the contributions.</p>',unsafe_allow_html=True)
        st.markdown('<div class="calculation-step"><span class="calculation-step-number">01</span><h3>Combine measurements across signals</h3></div>',unsafe_allow_html=True)
        st.markdown('<p class="calculation-step-copy">Each diagnostic is measured separately for every selected signal. Its <strong>median across signals</strong> is then used in the component formula. Undefined values are ignored. <strong>With two defined values, the median equals their mean.</strong> For larger even counts, it is the mean of the two central values.</p>',unsafe_allow_html=True)
        with st.expander("Inspect the per-signal values and the medians used"):
            st.write("Component formulas use these aggregated diagnostics; the final index is not the average of individual signal scores.")
            diagnostic_columns = list(DIAGNOSTICS)[:10]
            diagnostics = profile_features[["feature"] + diagnostic_columns].copy()
            diagnostics.loc[len(diagnostics)] = ["Median used for scoring"] + [profile_agg[k] for k in diagnostic_columns]
            _measurement_table(diagnostics)

        st.markdown('<div class="calculation-step"><span class="calculation-step-number">02</span><h3>Calculate the six component scores</h3></div>',unsafe_allow_html=True)
        st.markdown('<p class="calculation-step-copy">Each component has its own formula and a score from 0 to 100. Open a card to follow the formula with your measured values.</p><div class="calculation-convention"><strong>Reading the formulas:</strong> Rates are fractions: 10% becomes 0.1. <code>clip(x)</code> limits a value to the range 0–1.</div>',unsafe_allow_html=True)
        styles = []
        for start in (0,2,4):
            component_columns = st.columns(2,gap="medium")
            for index,(column,definition) in enumerate(zip(component_columns,definitions[start:start+2]),start=start):
                label, columns, explanation, formula, substitution, caution = definition
                row = breakdown.loc[breakdown.component == label].iloc[0]
                color = COMPONENT_COLORS[label]
                styles.append(f'.st-key-calculation_component_{index} {{border-top:3px solid {color};}}')
                with column, st.container(border=True,key=f"calculation_component_{index}",height="stretch"):
                    st.markdown(
                        f'<div class="calculation-component-header"><span class="calculation-component-name">{escape(label)}</span>'
                        f'<span class="calculation-component-value" style="color:{color}">{row.score_0_100:.2f}<small> / 100</small></span></div>'
                        f'<div class="calculation-component-contribution" style="background:{component_rgba(label,.07)}">'
                        f'{row.score_0_100:.2f} × {row.domain_weight:.1%} weight = <strong style="color:{color}">{row.weighted_points:.2f} points</strong></div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Formula & measured values"):
                        st.markdown("**Component formula**")
                        st.code(formula, language=None, wrap_lines=True)
                        st.markdown("**Your values inserted**")
                        st.code(f"{substitution} = {f(row.score_0_100)}", language=None, wrap_lines=True)
                        st.markdown("**Measurements behind this formula**")
                        st.write(explanation)
                        _measurement_table(measurements[["feature"] + columns])
                        if label == "Cross-signal consistency":
                            st.write("Absolute pairwise Pearson correlations (diagonal excluded from scoring):")
                            st.dataframe(canonical[list(profile_features.feature)].corr().abs(), width="stretch")
                            st.caption("Pearson r = sum((x−mean(x)) × (y−mean(y))) / sqrt(sum((x−mean(x))²) × sum((y−mean(y))²)), evaluated on each pair's available rows.")
                        st.write(f"**Weighted contribution:** {row.score_0_100:.8g} × {row.domain_weight:.8g} = **{row.weighted_points:.8g} points**.")
                        st.caption(caution)
        st.markdown('<style>'+''.join(styles)+'</style>',unsafe_allow_html=True)

        st.markdown('<div class="calculation-step"><span class="calculation-step-number">03</span><h3>Weight the components and add them up</h3></div>',unsafe_allow_html=True)
        st.markdown('<p class="calculation-step-copy">Domain weights determine each component’s influence. Multiply each score by its normalized weight, then add the six contributions.</p>',unsafe_allow_html=True)
        with st.container(border=True,key="calculation_result"):
            render_quality_calculation(profile_agg,weights,key="full_calculation_contributions")
            with st.expander("Show the sum at full precision"):
                st.code(" + ".join(f"{v:.12g}" for v in breakdown.weighted_points) + f" = {total:.12g}", language=None, wrap_lines=True)
                st.write("Normalized weight = configured weight / sum of configured weights. The calculation uses full precision; displayed values are rounded.")
