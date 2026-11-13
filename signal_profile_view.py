"""Signal comparison and descriptive diagnostics, without changing analysis data."""
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quality_visuals import quality_component_chart, render_quality_calculation, COMPONENT_COLORS, component_rgba
from quality_index_view import DIAGNOSTICS


EXTRA_DIAGNOSTICS = {
    "longest_missing_run_seconds": ("Longest gap", "Duration of the longest consecutive run of unavailable values.", "", "seconds"),
    "lag1_autocorr": ("Lag-1 autocorrelation", "Similarity between consecutive samples. High values can reflect smooth dynamics; they do not establish signal quality.", "", "number"),
    "spectral_entropy": ("Spectral entropy", "Spread of spectral power: near 0 means concentrated, near 1 means diffuse. Neither direction is universally better.", "", "number"),
    "dominant_frequency_ratio": ("Dominant-frequency share", "Share of spectral power in the strongest frequency bin. This is a proportion, not a frequency in Hz.", "", "percent"),
}
GROUPS = [
    ("Completeness", ["missing_rate"]),
    ("Cross-signal consistency", ["corr_mean_abs"]),
    ("Outlier burden", ["outlier_iqr_rate", "outlier_mad_rate"]),
    ("Distribution shape", ["skewness", "kurtosis_excess"]),
    ("Continuity", ["longest_missing_run_seconds", "missing_run_p95_seconds", "missing_burstiness"]),
    ("Temporal stability", ["lag1_autocorr", "drift_score", "local_volatility"]),
]
DIAGNOSTIC_COMPONENTS = {
    "missing_rate":"Completeness",
    "longest_missing_run_seconds":"Continuity", "missing_run_p95_seconds":"Continuity", "missing_burstiness":"Continuity",
    "outlier_iqr_rate":"Outlier burden", "outlier_mad_rate":"Outlier burden",
    "skewness":"Distribution shape", "kurtosis_excess":"Distribution shape",
    "lag1_autocorr":"Temporal stability", "drift_score":"Temporal stability", "local_volatility":"Temporal stability",
    "corr_mean_abs":"Cross-signal consistency",
}


def _style(fig, height):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#64748b"),
                      margin=dict(l=12, r=24, t=16, b=20),
                      legend=dict(orientation="h", y=1.08, x=0),
                      hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,145,170,.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,145,170,.12)", zeroline=False)
    return fig


def _signal_plot(values, timestamp, max_points, markers, show_missing=True):
    # Preserve gaps and every highlighted observation, including its neighbours.
    missing = values.isna().to_numpy()
    boundaries = np.flatnonzero(np.diff(missing.astype(int)))
    base = np.linspace(0, len(values)-1, min(len(values), max_points)).astype(int)
    flagged_by_method = {}
    valid = values.dropna()
    if len(valid) >= 4:
        q1, q3 = valid.quantile([.25, .75]); iqr = q3-q1
        median = valid.median(); mad = (valid-median).abs().median()
        masks = {"IQR": ((values < q1-1.5*iqr) | (values > q3+1.5*iqr)) if iqr > 0 else values.notna() & False,
                 "MAD": ((.67448975*(values-median)/mad).abs() > 3.5) if mad > 0 else values.notna() & False}
        # Comparisons on pandas nullable numeric data yield nullable booleans.
        # Missing observations cannot be outliers and must never reach np.nonzero as pd.NA.
        flagged_by_method = {
            method: np.flatnonzero(masks[method].fillna(False).to_numpy(dtype=bool))
            for method in markers
        }
    flagged = np.unique(np.concatenate(list(flagged_by_method.values()))) if flagged_by_method else np.array([], dtype=int)
    neighbours = np.clip(np.r_[flagged-1, flagged, flagged+1], 0, len(values)-1)
    idx = np.unique(np.r_[base, boundaries, boundaries+1, neighbours])
    fig = go.Figure()
    if show_missing and missing.any():
        edges = np.diff(np.r_[False, missing, False].astype(int))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        times = timestamp.to_numpy(dtype=float)
        steps = np.diff(times)
        positive_steps = steps[np.isfinite(steps) & (steps > 0)]
        step = float(np.median(positive_steps)) if len(positive_steps) else 1.0
        band_x, band_y = [], []
        for start, stop in zip(starts, stops):
            left = (times[start-1]+times[start])/2 if start else times[0]-step/2
            right = (times[stop-1]+times[stop])/2 if stop < len(times) else times[-1]+step/2
            if not np.isfinite(left) or not np.isfinite(right):
                continue
            band_x.extend([left, right, right, left, left, None])
            band_y.extend([0, 0, 1, 1, 0, None])
        fig.add_trace(go.Scatter(x=band_x, y=band_y, yaxis="y2", mode="lines",
                                line=dict(width=0), fill="toself", fillcolor="rgba(220,53,69,.16)",
                                name="Missing values", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=timestamp.iloc[idx], y=values.iloc[idx], mode="lines+markers",
                            name="Observed signal", connectgaps=False,
                            marker=dict(size=3, color="#2563eb"), line=dict(color="#2563eb", width=2)))
    for method, flagged in flagged_by_method.items():
        fig.add_trace(go.Scatter(x=timestamp.iloc[flagged], y=values.iloc[flagged],
                                mode="markers", name=f"{method} flags",
                                marker=dict(color=COMPONENT_COLORS["Outlier burden"],
                                            size=15 if method == "IQR" else 8,
                                            line=dict(width=2),
                                            symbol="circle-open" if method == "IQR" else "diamond-open")))
    _style(fig, 400)
    fig.update_layout(hovermode="x unified", xaxis_title="Relative time (s)", yaxis_title="Signal value",
                      yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True,
                                  showgrid=False, zeroline=False))
    return fig


def _distribution(values):
    valid = values.dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.22, .78], vertical_spacing=.06)
    fig.add_trace(go.Box(x=valid, y0=0, orientation="h", name="Quartiles & IQR", boxpoints=False,
                         hoverinfo="skip", quartilemethod="linear", marker_color=COMPONENT_COLORS["Distribution shape"],
                         fillcolor=component_rgba("Distribution shape",.15)), row=1, col=1)
    if len(valid):
        q1, median, q3 = valid.quantile([.25, .5, .75]); iqr = q3-q1
        lower = valid[valid >= q1-1.5*iqr].min()
        upper = valid[valid <= q3+1.5*iqr].max()
        fig.add_trace(go.Scatter(
            x=[lower, q1, median, q3, upper], y=[0]*5, mode="markers",
            marker=dict(size=18, color="rgba(37,99,235,0)"),
            text=["Lower whisker", "Q1 · 25th percentile", "Median · 50th percentile", "Q3 · 75th percentile", "Upper whisker"],
            customdata=[iqr]*5,
            hovertemplate="%{text}: %{x:.4g}<br>IQR (Q3 − Q1): %{customdata:.4g}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Histogram(x=valid, nbinsx=45, marker_color=COMPONENT_COLORS["Distribution shape"],
                              hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"), row=2, col=1)
    _style(fig, 355)
    fig.update_layout(showlegend=False, bargap=.06,
                      hoverlabel=dict(bgcolor="white", font=dict(color="#334155"), namelength=0))
    fig.update_xaxes(title_text="Signal value", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(showticklabels=False, range=[-.6, .6], row=1, col=1)
    return fig


def render_signal_profile(*, canonical, profile_features, selected_signals, weights, max_plot_points):
    st.markdown("""<style>
.st-key-signal_profile h3 {font-size:1.15rem;letter-spacing:-.02em;}
.st-key-signal_profile [data-testid="stMetricValue"] {font-size:1.7rem;letter-spacing:-.03em;}
.profile-focus-name {font-size:1rem;font-weight:600;overflow-wrap:anywhere;margin:0 0 8px;}
.profile-component-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:0 0 24px;align-items:start;}
.profile-component-card {min-width:0;display:flex;flex-direction:column;overflow:hidden;border:1px solid var(--component-border);border-top:4px solid var(--component-color);border-radius:14px;background:var(--background-color,#fff);box-shadow:0 3px 12px rgba(15,23,42,.035);}
.profile-component-heading {display:flex;align-items:center;gap:12px;padding:20px 22px 16px;background:linear-gradient(110deg,var(--component-tint),transparent);}
.profile-component-body {padding:0 22px 18px;display:flex;flex-direction:column;flex:1;}
.profile-component-dot {width:10px;height:10px;flex-shrink:0;border-radius:50%;background:var(--component-color);box-shadow:0 0 0 5px var(--component-tint);}
.profile-component-heading .profile-group-heading {color:var(--text-color);font-size:1rem;font-weight:600;line-height:1.4;letter-spacing:-.02em;padding:0;margin:0;}
.profile-metric {padding:14px 0;border-bottom:1px solid rgba(128,145,170,.14);}
.profile-metric:last-child {border-bottom:0;padding-bottom:0;}
.profile-metric-top {display:flex;justify-content:space-between;align-items:center;gap:20px;font-size:.83rem;line-height:1.5;}
.profile-metric-top > span {opacity:.78;}
.profile-metric-top strong {font-weight:600;white-space:nowrap;color:var(--text-color);font-size:1.22rem;letter-spacing:-.025em;font-variant-numeric:tabular-nums;}
.profile-metric p {font-size:.79rem;line-height:1.6;margin:8px 0 0;}
.profile-component-note {font-size:.75rem;line-height:1.6;margin:14px 0 0;opacity:.75;}
.profile-component-help {margin-top:14px;padding-top:0;font-size:.76rem;line-height:1.6;}
.profile-component-help summary {cursor:pointer;opacity:.75;list-style:none;transition:opacity 150ms ease;}
.profile-component-help summary::-webkit-details-marker {display:none;}
.profile-component-help summary::before {content:"";display:inline-block;width:5px;height:5px;margin-right:9px;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;transform:translateY(-1px) rotate(-45deg);transition:transform 160ms ease;}
.profile-component-help[open] summary::before {transform:translateY(-3px) rotate(45deg);}
.profile-component-help summary:hover {opacity:1;}
.profile-component-help summary:focus-visible {outline:2px solid var(--primary-color,#2563eb);outline-offset:4px;border-radius:3px;}
.profile-component-help p {margin:10px 0 0;}
.profile-component-help b {font-weight:600;}
.profile-frequency-card {padding:22px 24px;margin-bottom:18px;border:1px solid rgba(128,145,170,.22);border-radius:14px;}
.profile-frequency-intro {font-size:.8rem;line-height:1.6;opacity:.75;margin:4px 0 18px;}
.profile-frequency-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
.profile-frequency-stat {min-width:0;padding:17px 20px;background:rgba(128,145,170,.055);border:1px solid rgba(128,145,170,.14);border-radius:10px;}
.profile-frequency-label {font-size:.8rem;font-weight:500;margin-bottom:6px;}
.profile-frequency-value {font-size:1.65rem;font-weight:600;letter-spacing:-.035em;line-height:1.3;font-variant-numeric:tabular-nums;}
.profile-frequency-hint {font-size:.74rem;line-height:1.6;opacity:.7;margin:5px 0 0;}
.profile-group-heading {font-size:1rem;font-weight:650;letter-spacing:-.02em;margin:0 0 10px;}
.profile-group-tags {display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px;}
.profile-group-tag {font-size:.68rem;font-weight:550;padding:3px 8px;border-radius:999px;}
.profile-section-copy {font-size:.87rem;line-height:1.6;margin:0 0 16px;}
.st-key-profile_time_card,.st-key-profile_distribution_card,.st-key-profile_quality_card,[class*="st-key-profile_measurements_"] {border-radius:14px;padding:20px;border-color:rgba(128,145,170,.22);}
.st-key-profile_distribution_card {background:transparent;}
.st-key-profile_time_controls {background:rgba(128,145,170,.045);border-radius:10px;padding:12px;}
.profile-intro {font-size:.92rem;line-height:1.65;margin:0 0 18px;max-width:95ch;}
.profile-inspect-intro {font-size:.88rem;line-height:1.6;margin:0 0 12px;max-width:95ch;}
.profile-page-heading {font-size:1.65rem;font-weight:650;letter-spacing:-.035em;line-height:1.25;margin:0 0 8px;}
.st-key-profile_comparison_panel {padding:20px;border-radius:14px;border-color:rgba(128,145,170,.22);}
.st-key-profile_inspection_panel {padding:22px;border-radius:14px;border-color:rgba(68,119,170,.25);background:rgba(68,119,170,.035);margin-top:6px;}
.profile-step-heading {display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.profile-step-number {display:grid;place-items:center;width:32px;height:32px;flex-shrink:0;border-radius:9px;background:rgba(68,119,170,.09);color:#4477aa;font-size:.8rem;font-weight:650;}
.profile-step-heading h3 {font-size:1.1rem;font-weight:600;line-height:1.35;letter-spacing:-.02em;margin:0;padding:0;}
.profile-step-count {font-size:.75rem;margin-left:auto;white-space:nowrap;opacity:.65;}
.profile-comparison-note {font-size:.8rem;line-height:1.6;margin:0 0 12px;}
.profile-selection-copy {font-size:.86rem;line-height:1.65;margin:0;max-width:65ch;}
.st-key-profile_inspection_panel [data-testid="stSelectbox"] {margin-top:0;}
.st-key-signal_profile [data-testid="stButtonGroup"] button {transition:background-color 150ms ease,border-color 150ms ease,color 150ms ease;}
@media(prefers-reduced-motion:reduce) {
    .profile-component-help summary,.profile-component-help summary::before,
    .st-key-signal_profile [data-testid="stButtonGroup"] button {transition:none;}
}
@media(max-width:640px) {.st-key-profile_comparison_panel,.st-key-profile_inspection_panel {padding:16px;}.profile-step-count {display:none;}.profile-component-grid,.profile-frequency-grid {grid-template-columns:1fr;}.profile-frequency-card {padding:18px;}.profile-metric-top {flex-wrap:wrap;gap:8px;}}
</style>""", unsafe_allow_html=True)
    with st.container(key="signal_profile"):
        st.markdown('<h2 class="profile-page-heading">Signal profile</h2><p class="profile-intro">Compare signal quality at a glance, then choose one signal for a closer look.</p>',unsafe_allow_html=True)
        table = profile_features.copy()
        columns = {"feature": "Signal", "quality_index": "Quality score", "missing_rate": "Missing (%)",
                   "missing_run_p95_seconds": "p95 gap (s)", "outlier_iqr_rate": "IQR flags (%)",
                   "outlier_mad_rate": "MAD flags (%)"}
        columns.update({key: info[0] + (" (%)" if info[3] == "percent" else " (s)" if info[3] == "seconds" else "")
                        for key, info in {**DIAGNOSTICS, **EXTRA_DIAGNOSTICS}.items() if key not in columns})
        for key in columns:
            if key in {"missing_rate", "outlier_iqr_rate", "outlier_mad_rate", "dominant_frequency_ratio"}:
                table[key] *= 100
        with st.container(border=True,key="profile_comparison_panel"):
            st.markdown(f'<div class="profile-step-heading"><span class="profile-step-number">01</span><h3>Compare your signals</h3><span class="profile-step-count">{len(selected_signals)} selected signals</span></div><p class="profile-comparison-note">Start with quality scores, missing values, gaps, and statistical flags. Sort any column to compare signals.</p>',unsafe_allow_html=True)
            summary_keys=["feature","quality_index","missing_rate","longest_missing_run_seconds","outlier_mad_rate"]
            st.dataframe(
                table[summary_keys].rename(columns=columns),hide_index=True,width="stretch",height="content",
                column_config={
                    "Signal":st.column_config.TextColumn(width=200),
                    "Quality score":st.column_config.ProgressColumn(width=125,min_value=0,max_value=100,format="%.1f",help="Domain-weighted quality index, from 0 to 100"),
                    "Missing (%)":st.column_config.NumberColumn(width=100,format="%.2f",help="Share of unavailable signal values"),
                    "Longest gap (s)":st.column_config.NumberColumn(width=120,format="%.3f",help="Duration of the longest consecutive run of unavailable values, in seconds"),
                    "MAD flags (%)":st.column_config.NumberColumn(width=110,format="%.2f",help="Share of available values flagged by the MAD rule"),
                },
            )
            with st.expander("Explore all diagnostic measurements"):
                st.button("Diagnostic measurements explained →", key="open_signal_calculation_guide_profile",
                          on_click=st.session_state.update,
                          kwargs={"signal_calculation_guide_open": True, "method_guide_open": False, "quality_details_open": False})
                st.dataframe(table[list(columns)].rename(columns=columns),hide_index=True,width="stretch",height="content",
                             column_config={label:st.column_config.NumberColumn(format="%.2f") for key,label in columns.items() if key!="feature"})
        with st.container(border=True,key="profile_inspection_panel"):
            st.markdown('<div class="profile-step-heading"><span class="profile-step-number">02</span><h3>Take a closer look</h3></div>',unsafe_allow_html=True)
            description,selection=st.columns([1.2,1],gap="large",vertical_alignment="center")
            description.markdown('<p class="profile-selection-copy">Choose a signal to explore its time course, distribution, missing values, and quality components.</p>',unsafe_allow_html=True)
            with selection:
                focus = st.selectbox("Signal to inspect", selected_signals, index=None,
                                     placeholder="Choose a signal to explore",key="profile_inspect_signal")
        if focus is None:
            return
        fp = profile_features.loc[profile_features.feature == focus].iloc[0].to_dict()
        values = pd.to_numeric(canonical[focus], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 4:
            st.warning("Fewer than four available observations: several diagnostics and the quality score rely on fallback values. Interpret them with caution.")
        elif values.dropna().nunique() <= 1:
            st.info("This signal is constant. Several temporal and distribution diagnostics are undefined or use fallback values.")
        with st.container(border=True,key="profile_time_card"):
            st.markdown("### Signal over time")
            st.markdown(f'<p class="profile-focus-name">{escape(focus)}</p>', unsafe_allow_html=True)
            markers = st.multiselect("Highlight statistical flags", ["IQR", "MAD"], default=[],
                                     key="profile_flag_methods", placeholder="Choose IQR or MAD to inspect flagged observations")
            show_missing = st.checkbox("Highlight missing values", value=False, key="profile_show_missing")
            st.plotly_chart(_signal_plot(values, canonical.timestamp, max_plot_points, markers, show_missing), width="stretch")
            st.caption("Shaded blue bands mark missing or non-finite numeric samples; gaps remain unconnected. Highlighted outliers and their neighbours are always retained in the blue trace, even when other points are downsampled.")
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True, height="stretch", key="profile_distribution_card"):
            st.markdown("### Distribution")
            st.caption("How available values are distributed, regardless of their order in time.")
            valid = values.dropna()
            shape = st.columns(2)
            with shape[0]:
                st.metric("Skewness", f"{fp['skewness']:.3f}" if np.isfinite(fp['skewness']) else "Undefined")
                st.caption("Positive: longer right tail. Negative: longer left tail. Near zero: roughly symmetric.")
            with shape[1]:
                st.metric("Excess kurtosis", f"{fp['kurtosis_excess']:.3f}" if np.isfinite(fp['kurtosis_excess']) else "Undefined")
                st.caption("Tail weight relative to a normal distribution: positive means heavier tails, negative lighter tails.")
            st.caption("Both describe distribution shape; neither establishes a measurement error.")
            st.plotly_chart(_distribution(values), width="stretch", config={"displayModeBar": False})
            if len(valid):
                q1, median, q3 = valid.quantile([.25, .5, .75])
                st.caption(f"Q1: {q1:.4g} · Median: {median:.4g} · Q3: {q3:.4g} · IQR: {q3-q1:.4g}")
            else:
                st.caption("No available values to plot.")
            st.caption("The box spans Q1–Q3; its width is the IQR. Whiskers reach the outermost values within 1.5 × IQR of the box.")
        with right, st.container(border=True, height="stretch", key="profile_quality_card"):
            st.markdown("### Quality score")
            st.caption("Diagnostic summary using the active domain weights.")
            st.metric("Selected signal · quality index", f"{fp['quality_index']:.1f} / 100")
            quality_view = st.session_state.get("signal_quality_chart_type", "Bar chart")
            if quality_view == "Index calculation":
                render_quality_calculation(fp, weights, key="signal_quality_calculation")
            else:
                st.plotly_chart(quality_component_chart(fp,weights),width="stretch",key="signal_quality_components",config={"displayModeBar":False})
                st.caption("Bars show the six component scores from 0 to 100 before weighting, using the same colors as Data overview. Hover to see weights and contributions.")
            st.segmented_control("Chart type", ["Bar chart", "Index calculation"], default="Bar chart",
                                 key="signal_quality_chart_type", label_visibility="collapsed",
                                 on_change=lambda: None)
            st.caption("A higher index means a higher score under this heuristic, not proof of more accurate data. Cross-signal consistency measures relationships with the other selected signals; independent signals can still be valid.")
        st.markdown("### Detailed measurements")
        st.markdown('<p class="profile-section-copy">Explore the measurements behind this signal. Colors match the quality components; frequency measurements are additional diagnostics.</p>',unsafe_allow_html=True)
        guide = {**DIAGNOSTICS, **EXTRA_DIAGNOSTICS}
        def measurement_html(key, *, show_description=True):
            name, description, _, kind = guide[key]
            if key == "corr_mean_abs":
                description = "Mean absolute Pearson correlation with the other selected signals. Both matching and opposite directions count; independent signals can still be valid."
            value = fp.get(key, np.nan)
            formatted = "Undefined" if not np.isfinite(value) else f"{value:.2%}" if kind == "percent" else f"{value:.3f} s" if kind == "seconds" else f"{value:.3f}"
            explanation = f'<p>{escape(description)}</p>' if show_description else ""
            return f'<div class="profile-metric"><div class="profile-metric-top"><span>{escape(name)}</span><strong>{formatted}</strong></div>{explanation}</div>'

        cards = []
        for component, keys in GROUPS:
            note = ""
            if component == "Temporal stability":
                note = "Autocorrelation and local volatility use internal gap interpolation. Your data is unchanged."
            elif component == "Cross-signal consistency":
                note = "With one selected signal, correlation uses a fallback of 0 rather than a measured relationship."
            cards.append(
                f'<article class="profile-component-card" style="--component-color:{COMPONENT_COLORS[component]};--component-tint:{component_rgba(component,.08)};--component-border:{component_rgba(component,.20)}">'
                f'<div class="profile-component-heading"><span class="profile-component-dot" aria-hidden="true"></span><h4 class="profile-group-heading">{escape(component)}</h4></div>'
                + '<div class="profile-component-body">'
                + ''.join(measurement_html(key, show_description=False) for key in keys)
                + '<details class="profile-component-help"><summary>About these measurements</summary>'
                + ''.join(f'<p><b>{escape(guide[key][0])}.</b> {escape(guide[key][1])}</p>' for key in keys)
                + (f'<p class="profile-component-note">{escape(note)}</p>' if note else "")
                + '</details></div></article>'
            )
        st.markdown('<div class="profile-component-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
        if canonical.attrs.get("time_axis", {}).get("diagnostics", {}).get("irregular_intervals"):
            st.caption("Frequency diagnostics are unavailable for irregular timestamps. Row-lag autocorrelation, drift and local volatility remain sample-based; interpret them with care.")
        frequency_stats = []
        for key, hint in (("spectral_entropy", "0 = concentrated · 1 = diffuse"),
                          ("dominant_frequency_ratio", "Power in the strongest frequency bin")):
            value = fp.get(key, np.nan)
            formatted = "Undefined" if not np.isfinite(value) else f"{value:.2%}" if key == "dominant_frequency_ratio" else f"{value:.3f}"
            frequency_stats.append(
                f'<div class="profile-frequency-stat"><div class="profile-frequency-label">{escape(guide[key][0])}</div>'
                f'<div class="profile-frequency-value">{formatted}</div><p class="profile-frequency-hint">{escape(hint)}</p></div>'
            )
        st.markdown(
            '<section class="profile-frequency-card"><h4 class="profile-group-heading">Frequency diagnostics</h4>'
            '<p class="profile-frequency-intro">Additional signal characteristics · Not included in the quality index</p>'
            '<div class="profile-frequency-grid">' + ''.join(frequency_stats) + '</div>'
            '<details class="profile-component-help"><summary>About these measurements</summary>'
            + ''.join(f'<p><b>{escape(guide[key][0])}.</b> {escape(guide[key][1])}</p>' for key in ("spectral_entropy", "dominant_frequency_ratio"))
            + '<p class="profile-component-note">Spectral diagnostics use internally filled gaps. Your data is unchanged.</p></details></section>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="profile-section-copy" style="margin-top:20px">Ready to choose your next steps? '
            'Continue to <strong>3 · Recommendation</strong> to see preprocessing suggestions based on your signal diagnostics and active domain profile.</p>',
            unsafe_allow_html=True,
        )
