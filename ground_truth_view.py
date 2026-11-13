"""Reference validation with RMSE in each signal's original units."""
from __future__ import annotations

import hashlib
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

from cirruspp_core import (
    METHOD_OPTIONS, PipelineSpec, canonical_frame, compact_pipeline_grid,
    complete_window, execute_pipeline, inject_corruption,
)

CORRUPTIONS = {
    "mcar_missing": ("Random missing values", "Removes individual values throughout the window. The percentage is applied separately to each signal."),
    "block_missing": ("Consecutive gaps", "Removes consecutive blocks, simulating tracking interruptions. The percentage is a target; overlapping blocks can reduce the actual missing fraction."),
    "spikes": ("Isolated spikes", "Adds positive or negative spikes at the selected percentage of samples per signal. Each spike is 4–9 reference standard deviations high."),
    "high_frequency_noise": ("High-frequency noise", "Adds random and alternating noise to every sample. Strength controls amplitude relative to the reference signal's standard deviation."),
}
EVALUATION_VERSION = "rmse-per-signal-1"


def candidate_specs(table, method_label):
    """Validate user-edited methods and evaluate identical pipelines only once."""
    specs = []
    seen = set()
    for _, row in table.iterrows():
        if all(pd.isna(row.get(s.title())) or row.get(s.title()) == "" for s in ("outlier", "imputation", "smoothing")):
            continue
        methods = {}
        for stage in ("outlier", "imputation", "smoothing"):
            choices = {method_label(method): method for method in METHOD_OPTIONS[stage]}
            value = row.get(stage.title())
            if pd.isna(value) or value == "":
                methods[stage] = "none"
                continue
            if not isinstance(value, str) or value not in choices:
                raise ValueError("Choose an available method, or leave the cell empty to skip that stage.")
            methods[stage] = choices[value]
        spec = PipelineSpec(**methods, scaling="none")
        if spec.name() not in seen:
            specs.append(spec)
            seen.add(spec.name())
    if not specs:
        raise ValueError("Add at least one candidate pipeline to run validation.")
    return specs


def reconstruction_rmse(clean, original, processed, corruption_mask):
    """Never fill gaps for scoring; report finite-pair coverage explicitly."""
    rows = []
    for feature in clean.columns:
        if feature == "timestamp" or feature not in processed or feature not in original:
            continue
        reference = clean[feature].to_numpy(float)
        after = processed[feature].to_numpy(float)
        eligible = np.isfinite(reference)
        affected = corruption_mask[feature].to_numpy(bool) & eligible
        intact = ~corruption_mask[feature].to_numpy(bool) & eligible
        pairs = eligible & np.isfinite(after)
        def error(values, mask):
            valid = mask & np.isfinite(values)
            return float(np.sqrt(np.mean(np.square(values[valid] - reference[valid])))) if valid.any() else np.nan
        coverage = float(pairs.sum() / eligible.sum()) if eligible.any() else 0.
        rmse = error(after, eligible)
        rows.append({"feature": feature, "rmse": rmse, "affected_rmse": error(after, affected),
                     "intact_rmse": error(after, intact), "coverage": coverage,
                     "affected_coverage": float((pairs & affected).sum() / affected.sum()) if affected.any() else np.nan,
                     "remaining_missing": int((eligible & ~np.isfinite(after)).sum()),
                     "reference_samples": int(eligible.sum()), "affected_samples": int(affected.sum()),
                     "intact_samples": int(intact.sum()),
                     "complete": coverage == 1})
    return pd.DataFrame(rows)


def run_rmse_validation(clean, specs, corruption, severity, fs, *, execution_parameters=None):
    injected = inject_corruption(clean, corruption, severity, 42)
    rows, outputs = [], {}
    for spec in specs:
        output, meta = execute_pipeline(injected.data, spec, fs)
        name = spec.name()
        if execution_parameters is not None:
            execution_parameters[name] = meta["method_parameters"]
        outputs[name] = output
        metrics = reconstruction_rmse(clean, injected.data, output, injected.combined_mask)
        for row in metrics.to_dict("records"):
            rows.append({"pipeline": name, **spec.to_dict(), **row})
    return injected.data, pd.DataFrame(rows), outputs


def number(value):
    if not np.isfinite(value):
        return "Unavailable"
    return f"{value:.4g}"


def signal_unit(feature):
    match = re.search(r"\[([^\]]+)\]", feature)
    return match.group(1) if match else "original units"


def rmse_comparison_plot(chart, unit):
    """Use explicit row positions so color groups cannot change ranking order."""
    chart = chart.copy()
    if "Coverage" not in chart:
        chart["Coverage"] = 1.
    fig = go.Figure()
    for label, color in [("Your pipeline", "#2563eb"), ("Alternative", "#93b4f5")]:
        positions = [i for i, value in enumerate(chart.Type) if value == label]
        if not positions:
            continue
        subset = chart.iloc[positions]
        fig.add_trace(go.Bar(
            x=subset.RMSE, y=positions, name=label, orientation="h", width=.72,
            marker_color=color, text=[number(v) for v in subset.RMSE],
            textposition="outside", textfont=dict(color="#334155"), cliponaxis=False,
            customdata=subset[["Candidate", "Coverage"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>RMSE: %{x:.6g}<br>Coverage: %{customdata[1]:.1%}<extra>%{fullData.name}</extra>",
        ))
    finite = chart.RMSE[np.isfinite(chart.RMSE)]
    maximum = float(finite.max()) if not finite.empty else 0.
    for i, value in enumerate(chart.RMSE):
        if not np.isfinite(value):
            fig.add_annotation(x=0, y=i, text="Unavailable", showarrow=False,
                               xanchor="left", font=dict(color="#64748b"))
    fig.update_layout(
        title=f"RMSE · {unit} · inspect coverage", height=max(340, 44 * len(chart) + 120),
        barmode="overlay", bargap=.28, margin=dict(l=55, r=35, t=55, b=100),
        xaxis=dict(title="RMSE", range=[0, maximum * 1.18 if maximum > 0 else 1],
                   showgrid=True, gridcolor="rgba(148,163,184,.15)", zeroline=False),
        yaxis=dict(title=None, tickmode="array", tickvals=list(range(len(chart))),
                   ticktext=[f"{row.Candidate} · {row.Coverage:.0%} coverage" if row.Coverage < 1 else row.Candidate
                             for row in chart.itertuples()], range=[len(chart) - .5, -.5],
                   autorange=False, showgrid=False, zeroline=False),
        legend=dict(title_text="", orientation="h", x=0, y=-.23, yanchor="top"),
        uniformtext=dict(minsize=11, mode="show"),
    )
    return fig


def show_results(result, method_label, line_plot, max_plot_points):
    st.subheader("Validation results")
    controlled = result["kind"] == "controlled"
    features = list(dict.fromkeys(result["results"].feature))
    feature = st.selectbox("Evaluation signal", features, key="gt_feature")
    unit = signal_unit(feature)
    rows = result["results"].loc[result["results"].feature == feature].copy()
    if controlled:
        rows = rows.sort_values("rmse", na_position="last")
        labels = {p: f"P{i + 1}" for i, p in enumerate(dict.fromkeys(result["results"].pipeline))}
        ranking, traces, detail = st.tabs(["Candidate comparison", "Signal reconstruction", "RMSE explained"])
        with ranking:
            active_pipeline = result.get("active_pipeline", result["results"].pipeline.iloc[0])
            chart_rows = [{"Candidate": labels[r.pipeline], "RMSE": r.rmse, "Coverage": r.coverage,
                           "Type": "Your pipeline" if r.pipeline == active_pipeline else "Alternative"}
                          for r in rows.itertuples()]
            if chart_rows:
                chart = pd.DataFrame(chart_rows)
                fig = rmse_comparison_plot(chart, unit)
                st.plotly_chart(fig, width="stretch")
                available = rows.loc[np.isfinite(rows.rmse) & rows.complete]
                if not available.empty:
                    best = available.iloc[0]
                    methods = " → ".join(method_label(best[stage]) for stage in ["outlier", "imputation", "smoothing"])
                    st.markdown(f"**Lowest RMSE among fully reconstructed candidates:** {labels[best.pipeline]} · {methods} · RMSE {number(best.rmse)}")
                else:
                    st.info("No candidate reconstructs the full reference window. No lowest-RMSE candidate is highlighted; compare coverage and affected-sample errors below.")
                if not rows.complete.all():
                    st.warning("Incomplete candidates are scored only where output values exist. An RMSE of 0 can mean gaps were left unfilled; it does not prove successful reconstruction. RMSE values with different coverage are not directly comparable.")
            else:
                st.info("No candidate results are available.")
            table = pd.DataFrame({"Candidate": rows.pipeline.map(labels), "RMSE ↓": rows.rmse,
                "Coverage": rows.coverage, "Remaining gaps": rows.remaining_missing,
                "Reconstruction": np.where(rows.complete, "Complete", "Incomplete")})
            for stage in ["outlier", "imputation", "smoothing"]:
                table[stage.title()] = rows[stage].map(method_label)
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "RMSE ↓": st.column_config.NumberColumn(format="%.4g"),
                "Coverage": st.column_config.NumberColumn(format="percent")})
            st.caption("All candidates use outlier treatment → imputation → optional smoothing. Scaling is disabled to preserve original units.")
    else:
        traces, detail = st.tabs(["Signal reconstruction", "RMSE explained"])
    with traces:
        series = {"Reference": result["clean"][feature],
                  "Corrupted" if controlled else "Original input": result["corrupted"][feature]}
        if controlled:
            candidates = list(labels)
            selection_key = "gt_candidates"
            selection_context = tuple(candidates)
            if st.session_state.get("_gt_overlay_context") != selection_context:
                st.session_state["_gt_overlay_context"] = selection_context
                st.session_state[selection_key] = [rows.pipeline.iloc[0]]
            selected = st.multiselect("Overlay candidate pipelines", candidates,
                format_func=lambda p: f"{labels[p]} · " + " · ".join(method_label(rows.loc[rows.pipeline == p, stage].iloc[0])
                    for stage in ["outlier", "imputation", "smoothing"]), key=selection_key)
            for candidate in selected:
                series[labels[candidate]] = result["outputs"][candidate][feature]
                metric = rows.loc[rows.pipeline == candidate].iloc[0]
                if not metric.complete:
                    st.caption(f"{labels[candidate]}: {int(metric.remaining_missing)} reference samples have no finite output value.")
            if not selected:
                st.caption("Select one or more pipelines to overlay their reconstructions.")
        else:
            series["Active cleaning"] = result["processed"][feature]
            metric = rows.iloc[0]
            if not metric.complete:
                st.caption(f"{int(metric.remaining_missing)} reference samples have no finite output value.")
        fig = line_plot(series, result["clean"].timestamp, max_plot_points, f"Reconstruction · {feature}")
        if controlled:
            # Candidate identity, rather than selection order, determines its color.
            for trace in fig.data:
                if trace.name in labels.values():
                    index = int(trace.name[1:]) - 1
                    color = qualitative.Dark24[index % len(qualitative.Dark24)]
                    trace.update(line_color=color, marker_color=color)
        st.plotly_chart(fig, width="stretch")
        if controlled:
            st.caption("Each P-number matches the comparison table. Click a legend entry to hide a trace; double-click to isolate it.")
        st.caption("Compare the output trajectory with the reference, particularly around gaps and altered samples.")
    with detail:
        st.markdown("**How RMSE is calculated**")
        st.latex(r"\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}")
        st.write("At each evaluated sample, subtract the reference from the processed value, square the difference, average these squared differences, then take the square root. The result has the same unit as the signal and emphasizes larger errors.")
        st.markdown("**Affected samples** measure reconstruction of damaged values. **Intact samples** show how much cleaning changes values outside the corruption mask. With noise at every sample, no intact-sample RMSE is available.")
        if not controlled:
            st.caption("For an uploaded reference, affected samples are those where the input differs from the reference under the numerical comparison tolerance; they are not independently annotated faults.")
        st.write("Only finite reference/output pairs are scored. Coverage is their share of all finite reference samples. A missing subset is shown as unavailable, never as zero error. A zero overall RMSE with incomplete coverage can still leave every damaged sample unreconstructed. Compare RMSE directly only for the same evaluated samples; the highlighted candidate must reconstruct the full reference window.")
        overview = rows.copy()
        columns = ["feature", "rmse", "affected_rmse", "intact_rmse", "coverage", "remaining_missing"]
        if controlled:
            overview["Candidate"] = overview.pipeline.map(labels)
            columns = ["Candidate"] + columns
        st.dataframe(overview[columns],
                     hide_index=True, width="stretch", column_config={
                         "feature": "Signal", "rmse": st.column_config.NumberColumn("RMSE", format="%.4g"),
                         "affected_rmse": st.column_config.NumberColumn("Affected RMSE", format="%.4g"),
                         "intact_rmse": st.column_config.NumberColumn("Intact RMSE", format="%.4g"),
                         "coverage": st.column_config.NumberColumn("Coverage", format="percent"),
                         "remaining_missing": "Remaining gaps"})


def render_ground_truth(*, canonical, selected_signals, source_name, fs, domain_cfg, recs,
                        active_spec, order_key, analysis_context_key, separator, decimal,
                        load_upload, longest_complete_run_length, method_label, line_plot, max_plot_points):
    st.markdown("## Reference validation")
    st.write("Evaluate how closely a preprocessing pipeline reconstructs a reference signal using controlled corruption or an aligned clean recording.")
    flow = st.columns(3)
    for col, title, text in zip(flow, ["1 · Choose a reference", "2 · Run validation", "3 · Compare results"],
                                ["Use a complete window from your data or upload an aligned clean recording.", "Compare candidate pipelines or evaluate your active cleaning against the reference.", "Inspect RMSE, reconstructed signals and remaining missing values."]):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.caption(text)
    mode = st.radio("Validation source", ["Inject corruption into a complete window", "Upload clean reference for current noisy data"], horizontal=True,
                    format_func=lambda x: "Controlled experiment" if x.startswith("Inject") else "Clean reference upload")
    if st.session_state.get("ground_truth"):
        st.session_state["ground_truth"]["current"] = False
    if st.session_state.get("clean_ref_result"):
        st.session_state["clean_ref_result"]["current"] = False
    if mode.startswith("Inject"):
        max_complete = longest_complete_run_length(canonical)
        defaults = {"imputation": ["linear", "locf"], "outlier": ["zscore", "iqr"], "smoothing": ["none", "moving_median"]}
        alternatives = {s: [m for m, _ in recs[s].domain_support] or defaults[s] for s in defaults}
        spec = PipelineSpec(active_spec.imputation, active_spec.outlier, active_spec.smoothing, "none")
        smoothing_blocked = canonical.attrs.get("time_axis", {}).get("diagnostics", {}).get("irregular_intervals", False) and canonical.attrs.get("time_processing", {}).get("irregular_smoothing", "disabled") != "resample"
        if smoothing_blocked:
            spec = PipelineSpec(spec.imputation, spec.outlier, "none", "none")
            alternatives["smoothing"] = ["none"]
        specs = compact_pipeline_grid(spec, alternatives, 10)
        settings, explanation = st.columns([1.65, 1], gap="large")
        with settings, st.container(border=True):
            st.markdown("### Experiment settings")
            corruption = st.selectbox("Corruption", list(CORRUPTIONS), format_func=lambda k: CORRUPTIONS[k][0])
            st.caption(CORRUPTIONS[corruption][1])
            if corruption == "high_frequency_noise":
                severity = st.slider("Noise strength", .01, .40, .10, .01, key="gt_noise_strength")
                st.caption(f"Random noise standard deviation: {(.05 + 1.4 * severity):.3f} × reference SD; alternating amplitude: {(.25 * severity):.3f} × reference SD. Every sample is affected.")
            else:
                label = "Affected samples (%)" if corruption == "spikes" else "Missing values (%)"
                percentage = st.slider(label, 1, 20 if corruption == "spikes" else 40, 10, 1, key=f"gt_amount_{corruption}")
                severity = percentage / 100
            if max_complete < 32:
                st.warning("At least 32 consecutive complete samples across the selected signals are needed. Adjust the signal selection or validity interpretation to make a longer window available.")
                window_length = 32
            elif max_complete == 32:
                window_length = 32
                st.caption("Reference window: 32 samples.")
            else:
                upper = min(3000, max_complete)
                window_length = st.slider("Reference-window samples", 32, upper, min(400, upper), 1)
            preview_window = complete_window(canonical, window_length) if max_complete >= 32 else None
            window_duration = float(preview_window.timestamp.iloc[-1] - preview_window.timestamp.iloc[0]) if preview_window is not None else 0.
            st.caption(f"Available complete run: {max_complete:,} samples · Selected window duration: {window_duration:.3g} s from actual timestamps.")
        with explanation, st.container(border=True):
            st.markdown("### What will be compared?")
            st.write("A continuous complete window serves as the reference. A reproducible corruption is added, then each candidate tries to recover the original values.")
            st.caption("Start with your active cleaning methods and compact alternatives, then edit the candidate table below. Changes here apply only to this validation experiment.")
            st.markdown("**Execution order**")
            st.caption("Outlier treatment → imputation → optional smoothing. Scaling is disabled to retain comparable physical units.")
        default_rows = pd.DataFrame([{s.title(): method_label(getattr(p, s)) for s in ("outlier", "imputation", "smoothing")} for p in specs])
        seed = (analysis_context_key, tuple(p.name() for p in specs))
        if st.session_state.get("_gt_candidate_seed") != seed:
            st.session_state["_gt_candidate_seed"] = seed
            st.session_state.pop("gt_candidate_editor", None)
        candidate_error = None
        with st.expander("Preview candidate methods", expanded=True):
            st.caption("Click a cell to choose a method. Add a row at the bottom or select rows to delete them. Changing the analysis or active cleaning methods restores the suggested candidates.")
            st.caption("Each stage is optional: select None or leave its cell empty to skip it. For imputation only, choose an Imputation method and leave Outlier and Smoothing unused. Completely empty rows are ignored.")
            if st.button("Reset candidate methods", key="gt_reset_candidates"):
                st.session_state.pop("gt_candidate_editor", None)
            edited = st.data_editor(default_rows, key="gt_candidate_editor", num_rows="dynamic",
                hide_index=True, width="stretch",
                column_config={s.title(): st.column_config.SelectboxColumn(s.title(),
                    options=[method_label(m) for m in METHOD_OPTIONS[s]], required=False,
                    help="None or an empty cell skips this stage. You can run a pipeline with only one stage.")
                    for s in ("outlier", "imputation", "smoothing")})
            try:
                specs = candidate_specs(edited, method_label)
                if smoothing_blocked and any(p.smoothing != "none" for p in specs):
                    raise ValueError("Enable Resample internally for smoothing in Time processing, or set candidate Smoothing to None.")
                st.caption(f"{len(specs)} distinct candidate pipelines · Scaling disabled · Identical rows are evaluated once.")
            except ValueError as exc:
                candidate_error = str(exc)
                specs = []
                st.warning(candidate_error)
        run = st.button("Run controlled validation", type="primary", width="stretch",
                        disabled=max_complete < 32 or candidate_error is not None)
        expected = (EVALUATION_VERSION, analysis_context_key, corruption, severity, window_length, tuple(p.name() for p in specs))
        if run:
            clean = complete_window(canonical, window_length)
            validation_parameters = {}
            with st.spinner("Corrupting the reference and evaluating candidates…"):
                corrupted, results, outputs = run_rmse_validation(clean, specs, corruption, severity, fs, execution_parameters=validation_parameters)
            st.session_state["ground_truth"] = {"kind": "controlled", "key": expected, "clean": clean,
                "corrupted": corrupted, "results": results, "outputs": outputs, "metric": EVALUATION_VERSION,
                "corruption": corruption, "strength": severity, "seed": 42, "active_pipeline": spec.name(),
                "method_parameters": validation_parameters}
        result = st.session_state.get("ground_truth")
        if result:
            result["current"] = result.get("key") == expected
        if result and result.get("kind") == "controlled" and result.get("key") == expected:
            show_results(result, method_label, line_plot, max_plot_points)
        elif result:
            st.info("The experiment settings or analysis have changed. Run validation to refresh the results.")
        else:
            st.info("Ready to compare: choose a corruption and window, then run validation to see candidate rankings and reconstructed signals.")
    else:
        with st.container(border=True):
            st.markdown("### Reference recording")
            st.write("Upload the clean version of the current recording. Signal names must match, and rows must represent the same observations in the same order.")
            upload = st.file_uploader("Clean reference CSV/TSV", type=["csv", "tsv", "txt"], key="clean_ref")
            confirm = st.checkbox("I confirm that the clean and noisy files are aligned sample-for-sample (same observation order).")
            expected = None
            common = []
            if upload is not None:
                data = upload.getvalue()
                loaded = load_upload(data, upload.name, separator, decimal)
                common = [c for c in selected_signals if c in loaded.frame.columns]
                n = min(len(loaded.frame), len(canonical))
                expected = (EVALUATION_VERSION, analysis_context_key, hashlib.sha256(data).hexdigest(), separator, decimal, active_spec.name(), order_key)
                if not common:
                    st.error("No matching selected signal columns were found.")
                else:
                    st.caption(f"{len(common)} matching signals · {n:,} paired rows · {fs:g} Hz")
                    if len(common) != len(selected_signals):
                        st.warning("Only matching signals will be evaluated. Missing from reference: " + ", ".join(c for c in selected_signals if c not in common))
                    if len(loaded.frame) != len(canonical):
                        st.warning(f"File lengths differ. Evaluation uses the first {n:,} rows of both files; extra rows are excluded. No automatic synchronization is performed.")
            st.caption("Your active cleaning methods and stage order are evaluated with scaling disabled. Confirm alignment before proceeding.")
            smoothing_blocked = canonical.attrs.get("time_axis", {}).get("diagnostics", {}).get("irregular_intervals", False) and canonical.attrs.get("time_processing", {}).get("irregular_smoothing", "disabled") != "resample" and active_spec.smoothing != "none"
            if smoothing_blocked:
                st.warning("Enable internal resampling for smoothing, or select None for smoothing in Pipeline lab.")
            run = st.button("Evaluate active cleaning against reference", type="primary", disabled=not (common and confirm) or smoothing_blocked, width="stretch")
        if run:
            # The user confirms sample alignment; use the actual observation times of the noisy recording.
            clean = canonical.iloc[:n][["timestamp"]].reset_index(drop=True).copy()
            clean.attrs = dict(canonical.attrs)
            for feature in common:
                clean[feature] = pd.to_numeric(loaded.frame[feature].iloc[:n], errors="coerce").to_numpy()
            noisy = canonical.iloc[:n][["timestamp"] + common].reset_index(drop=True)
            if n < 32 or not np.isfinite(clean[common].to_numpy(float)).all():
                st.error("The paired reference needs at least 32 rows and finite values for every matching signal.")
            else:
                spec = PipelineSpec(active_spec.imputation, active_spec.outlier, active_spec.smoothing, "none")
                with st.spinner("Evaluating active cleaning against the reference…"):
                    pred, meta = execute_pipeline(noisy, spec, fs, order_key)
                    mask = pd.DataFrame({c: ~np.isclose(clean[c].to_numpy(float), noisy[c].to_numpy(float), equal_nan=True) for c in common})
                    metrics = reconstruction_rmse(clean, noisy, pred, mask)
                st.session_state["clean_ref_result"] = {"kind": "reference", "key": expected, "clean": clean,
                    "corrupted": noisy, "processed": pred, "results": metrics, "metric": EVALUATION_VERSION,
                    "reference_name": upload.name, "reference_sha256": hashlib.sha256(data).hexdigest(),
                    "alignment_confirmed": True, "paired_rows": n, "signals": common,
                    "pipeline": spec.to_dict(), "order": order_key, "seed": meta["seed"],
                    "method_parameters": meta["method_parameters"]}
        result = st.session_state.get("clean_ref_result")
        if confirm and expected is not None and result and result.get("key") == expected:
            result["current"] = True
            show_results(result, method_label, line_plot, max_plot_points)
        elif result:
            st.info("Upload and confirm the matching reference, then evaluate the current cleaning settings to refresh results.")
    st.caption("Next: continue to Export to download your executed final pipeline and its audit. Validation results stay in this view and are not included in the JSON export; validation candidates do not replace the final pipeline.")
