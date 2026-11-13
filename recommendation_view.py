"""Recommendation source picker and constrained AI pipeline presentation."""
import os
import json
from html import escape

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ai_recommendation import INPUT_MODES, MODELS, USER_GOALS, DEFAULT_USER_GOAL, CUSTOM_USER_GOAL, MAX_GOAL_LENGTH, build_request_body, generate_suggestion, save_suggestion, suggestion_id, matching_suggestions, AI_PIPELINE_SOURCES
from cirruspp_core import PIPELINE_ORDERS

STAGE_LABELS = {"outlier": "Outlier treatment", "imputation": "Missing value imputation",
                "scaling": "Normalization", "smoothing": "Optional methods"}
STAGE_COLORS = {"outlier": "#B57A25", "imputation": "#4477AA",
                "scaling": "#8870AD", "smoothing": "#278579"}


def render_stage_styles():
    st.markdown("""<style>
.pipeline-stage {min-width:0;display:flex;flex-direction:column;border:1px solid var(--stage-border);border-top:3px solid var(--stage-color);border-radius:14px;padding:18px;background:linear-gradient(150deg,var(--stage-tint),transparent 75%);box-sizing:border-box;}
.pipeline-stage-top {display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:16px;}
.pipeline-stage-number {display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:var(--stage-tint);color:var(--stage-color);font-size:.78rem;font-weight:650;}
.pipeline-stage-status {font-size:.68rem;opacity:.65;}
.pipeline-stage-label {font-size:.8rem;line-height:1.5;min-height:3em;}
.pipeline-stage-label small {display:block;font-size:.72rem;opacity:.65;}
.pipeline-stage-method {font-size:1.2rem;font-weight:600;letter-spacing:-.025em;line-height:1.4;min-height:2.8em;margin:10px 0 14px;overflow-wrap:anywhere;}
.pipeline-stage-reason {border-top:1px solid var(--stage-border);padding-top:12px;font-size:.78rem;line-height:1.65;margin-top:auto;}
.pipeline-stage-reason summary {cursor:pointer;opacity:.8;}
.pipeline-stage-reason summary:focus-visible {outline:2px solid var(--stage-color);outline-offset:4px;border-radius:3px;}
.pipeline-stage-reason p {margin:12px 0 0;}
.pipeline-stage-reason .recommendation-rules {margin-top:12px;}
.pipeline-stage-reason .recommendation-rule-values strong {color:var(--stage-color);}
@media(max-width:640px) {.pipeline-stage-label,.pipeline-stage-method {min-height:0;}}
</style>""", unsafe_allow_html=True)


def stage_card_html(*, stage, number, method, method_name, explanation):
    """Shared stage identity and layout for heuristic and AI recommendations."""
    color = STAGE_COLORS[stage]
    red, green, blue = (int(color[i:i+2], 16) for i in (1, 3, 5))
    style = f"--stage-color:{color};--stage-tint:rgba({red},{green},{blue},.08);--stage-border:rgba({red},{green},{blue},.24)"
    detail = '<small>Smoothing</small>' if stage == 'smoothing' else ''
    status = 'Not applied' if method == 'none' else 'Selected'
    return (
        f'<article class="pipeline-stage" style="{style}">'
        f'<div class="pipeline-stage-top"><span class="pipeline-stage-number">{number:02d}</span><span class="pipeline-stage-status">{status}</span></div>'
        f'<div class="pipeline-stage-label">{escape(STAGE_LABELS[stage])}{detail}</div>'
        f'<div class="pipeline-stage-method">{escape(method_name)}</div>'
        f'<details class="pipeline-stage-reason"><summary>Why this recommendation?</summary>{explanation}</details></article>'
    )


def render_request_preview(context, model):
    body = build_request_body(context, model)
    transmitted = json.loads(body["input"])
    with st.expander("What is sent to OpenAI?"):
        st.caption("Preview for the next request with the current model and input settings. Opening this preview does not send a request. Tables show the actual input content; technical field names match the JSON.")
        st.markdown("**Analysis settings**")
        settings = {k: v for k, v in transmitted.items()
                    if k not in {"aggregate_diagnostics", "signal_diagnostics", "signal_data", "available_methods", "execution_orders"}}
        st.dataframe(pd.DataFrame([
            {"Field": k, "Value": json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else str(v)}
            for k, v in settings.items()
        ]), hide_index=True, width="stretch")
        st.caption("Time information sent: sampling rate, recording duration, interval regularity and active time-processing settings. Original timestamp column names, units, origins and detailed validity checks stay local. Submitted sample times, when included, are relative seconds.")
        st.markdown("**Diagnostics per signal**")
        st.caption("Calculated from the complete selected recording, before pipeline execution. Signal names are included.")
        st.dataframe(pd.DataFrame(transmitted.get("signal_diagnostics", [])), hide_index=True, width="stretch")
        st.markdown("**Aggregate diagnostics**")
        st.caption("Includes medians and maxima across selected signals, as provided in the request.")
        st.dataframe(pd.DataFrame([
            {"Diagnostic": k, "Value": v} for k, v in transmitted.get("aggregate_diagnostics", {}).items()
        ]), hide_index=True, width="stretch")
        data = transmitted.get("signal_data")
        st.markdown("**Individual signal values**")
        if data:
            st.caption(f"All {data['included_rows']:,} submitted rows are available in this scrollable table, out of {data['total_rows']:,} recording rows. Selection: {data['selection']}.")
            st.dataframe(pd.DataFrame(data["rows"], columns=data["columns"]), hide_index=True, width="stretch", height=320)
            st.caption("Sample times are relative seconds. Original row indices identify omitted intervals.")
        else:
            st.info("Diagnostics only: no individual signal values or time-series rows are sent.")
        st.markdown("**Allowed methods and execution orders**")
        st.dataframe(pd.DataFrame([
            {"Stage": stage, "Allowed methods": ", ".join(methods)}
            for stage, methods in transmitted.get("available_methods", {}).items()
        ]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([
            {"Order": key, "Steps": value} for key, value in transmitted.get("execution_orders", {}).items()
        ]), hide_index=True, width="stretch")
        with st.expander("How the OpenAI call works · full request"):
            st.markdown("""
1. **Prepare locally:** CIRRUS++ calculates diagnostics and constructs the input shown above.
2. **Send on click:** Generate / Regenerate suggestion makes one HTTPS POST from the Streamlit server to `https://api.openai.com/v1/responses`, with a 60-second timeout. Your API key authenticates the request in the Authorization header; it is excluded from this preview.
3. **Request a constrained answer:** Instructions explain the methods and their fixed parameters. A strict JSON schema requires one method per stage, an execution order, reasons and cautions. No code execution or web-search tools are supplied.
4. **Check the answer:** CIRRUS++ rejects incomplete, refused, malformed or unsupported suggestions and checks compatibility with all-missing channels. A valid suggestion is saved in the current session with token usage and an estimated cost. No conversation history is attached to the next request.
5. **Apply explicitly:** Use AI-based pipeline selects the suggestion; Run final pipeline executes the implemented methods on the Streamlit server.
""")
            st.caption("store=false is sent with the request. It is not a guarantee of zero provider retention. The local data fingerprint is excluded from the request.")
            st.markdown("**Exact instructions**")
            st.code(body["instructions"], language="text")
            st.markdown("**Complete JSON request body (without authentication header)**")
            st.code(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False), language="json")


def server_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except (StreamlitSecretNotFoundError, FileNotFoundError):
            key = ""
    return key


def use_recommendation(source, context_key):
    st.session_state["final_pipeline_source"] = source
    st.session_state["final_pipeline_context"] = context_key
    if source == "AI-based":
        st.session_state["final_ai_result"] = context_key
        results=dict(st.session_state.get("ai_pipeline_results",{}))
        legacy=st.session_state.get("ai_pipeline_result")
        if legacy:
            results.setdefault(suggestion_id(legacy),legacy)
        result=results[context_key]
        mode=result.get("input_mode","diagnostics_data" if "signal_data" in result["context"] else "diagnostics")
        st.session_state["final_pipeline_source"]=next(label for label,value in AI_PIPELINE_SOURCES.items() if value==mode)
    st.session_state["workflow_tabs"] = "5 · Pipeline lab"


def remember_session_key():
    st.session_state["_ai_session_api_key"] = st.session_state["ai_api_key_input"].strip()


def remember_input_mode():
    st.session_state["_ai_input_mode"] = st.session_state["ai_input_mode"]


def remember_goal():
    st.session_state["_ai_goal"] = st.session_state["ai_goal"]
    st.session_state.pop("ai_pipeline_error", None)


def remember_custom_goal():
    st.session_state["_ai_custom_goal"] = st.session_state["ai_custom_goal"]
    st.session_state.pop("ai_pipeline_error", None)


def render_ai_result(result, context_key, method_label):
    """Present an existing suggestion without modifying it or making an API call."""
    render_stage_styles()
    st.markdown("""<style>
.st-key-ai_pipeline_suggestion {padding:24px;border-radius:16px;border-color:rgba(68,119,170,.22);}
.ai-result-header {display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:12px;}
.ai-result-kicker {font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;opacity:.6;margin-bottom:5px;}
.ai-result-header h3 {font-size:1.25rem;font-weight:650;letter-spacing:-.025em;margin:0;padding:0;}
.ai-result-cost {font-size:.75rem;white-space:nowrap;border:1px solid rgba(128,145,170,.2);border-radius:8px;padding:7px 10px;}
.ai-result-cost span {opacity:.6;margin-right:5px;}
.ai-result-meta {display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:18px;}
.ai-result-meta span {font-size:.7rem;border-radius:6px;padding:4px 8px;background:rgba(128,145,170,.07);}
.ai-result-meta time {font-size:.7rem;opacity:.6;padding:4px 0;}
.ai-result-lead {font-size:.9rem;line-height:1.7;max-width:95ch;margin:0 0 18px;}
.ai-pipeline-flow {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0;}
.st-key-ai_result_action {border-top:1px solid rgba(128,145,170,.18);padding-top:18px;margin-top:8px;}
.ai-result-action-copy {font-size:.79rem;line-height:1.65;margin:0;max-width:75ch;}
.ai-result-action-copy strong {display:block;font-weight:600;}
.ai-result-action-copy span {opacity:.7;}
@media(max-width:1050px) {.ai-pipeline-flow {grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:640px) {.st-key-ai_pipeline_suggestion {padding:18px;}.ai-pipeline-flow {grid-template-columns:1fr;}.ai-result-header {flex-wrap:wrap;}}
</style>""",unsafe_allow_html=True)
    suggestion = result["suggestion"]
    input_mode = INPUT_MODES[result.get("input_mode", "diagnostics_data" if "signal_data" in result["context"] else "diagnostics")]
    generated = result["created_at"][:16].replace("T", " ") + " UTC"
    # A short verbatim lead keeps existing long responses usable. The full
    # summary remains available below, without another model request.
    import re
    lead = re.split(r"(?<=[.!?])\s+(?=[A-Z])", suggestion["summary"], maxsplit=1)[0]
    if len(lead) > 300:
        lead = lead[:300].rsplit(" ",1)[0] + "…"
    with st.container(border=True, key="ai_pipeline_suggestion"):
        st.markdown(
            '<div class="ai-result-header"><div><div class="ai-result-kicker">Your suggested workflow</div><h3>AI-suggested pipeline</h3></div>'
            f'<div class="ai-result-cost"><span>Est. cost</span>${result["estimated_cost_usd"]:.5f}</div></div>'
            f'<div class="ai-result-meta"><span>OpenAI · {escape(MODELS.get(result["model"],result["model"]))}</span><span>{escape(input_mode)}</span><time>{escape(generated)}</time></div>'
            f'<p class="ai-result-lead">{escape(lead)}</p>',unsafe_allow_html=True,
        )
        goal = result["context"].get("user_goal")
        if goal:
            st.caption(f"Preprocessing goal: {goal}")
        steps = []
        for number, stage in enumerate(suggestion["order"].split("_"),start=1):
            steps.append(stage_card_html(stage=stage, number=number, method=suggestion[stage],
                                        method_name=method_label(suggestion[stage]),
                                        explanation=f'<p>{escape(suggestion["reasons"][stage])}</p>'))
        st.markdown('<div class="ai-pipeline-flow">'+''.join(steps)+'</div>',unsafe_allow_html=True)
        if suggestion["summary"] != lead:
            with st.expander("Overall rationale"):
                st.write(suggestion["summary"])
        if suggestion["outlier"] not in {"none", "winsorize"} and (suggestion["imputation"] == "none" or suggestion["order"].startswith("imputation")):
            st.warning("This pipeline leaves detected outliers missing because no imputation follows outlier treatment.")
        if suggestion["cautions"]:
            with st.expander("Limitations to review"):
                for caution in suggestion["cautions"]:
                    st.write("• " + caution)
        with st.container(key="ai_result_action"):
            copy, action = st.columns([2,1],gap="medium",vertical_alignment="center")
            copy.markdown('<p class="ai-result-action-copy"><strong>Review and run in Pipeline lab</strong><span>This transfers the suggested methods. Processing starts only when you run the pipeline. AI suggestions are not validated reconstruction results.</span></p>',unsafe_allow_html=True)
            action.button("Use AI-based pipeline",key="use_ai_pipeline",type="primary",width="stretch",
                          on_click=use_recommendation,args=("AI-based",suggestion_id(result)))
        with st.expander("Request details & token usage"):
            usage=result["usage"]
            st.caption(f"{usage.get('input_tokens',0):,} input tokens · {usage.get('output_tokens',0):,} output tokens · Estimated API cost: ${result['estimated_cost_usd']:.5f}")
            st.caption("Cost estimates use configured standard rates and may differ from your invoice.")
            st.json({"model":result["resolved_model"],"prompt_version":result["prompt_version"],
                     "diagnostics":{k:v for k,v in result["context"].items() if k!="data_fingerprint"},"response":suggestion})


def render_recommendations(*, recs, context, context_key, method_label, heuristic_card, method_guide):
    render_stage_styles()
    st.markdown("""<style>
.recommendation-heading {font-size:1.65rem;font-weight:650;letter-spacing:-.035em;margin:0 0 8px;}
.recommendation-intro {font-size:.92rem;line-height:1.65;max-width:85ch;margin:0 0 18px;}
.st-key-ai_model_panel,.st-key-ai_input_panel {padding:22px;border-radius:14px;border-color:rgba(128,145,170,.22);}
.ai-panel-heading {display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.ai-panel-step {display:grid;place-items:center;width:28px;height:28px;border-radius:8px;background:rgba(68,119,170,.09);color:#4477aa;font-size:.72rem;font-weight:650;flex-shrink:0;}
.ai-panel-heading h3 {font-size:1rem;font-weight:600;letter-spacing:-.02em;margin:0;padding:0;}
.ai-provider {margin-left:auto;font-size:.7rem;border:1px solid rgba(128,145,170,.22);border-radius:999px;padding:3px 9px;}
.ai-panel-copy {font-size:.8rem;line-height:1.6;opacity:.7;margin:0 0 18px;}
.ai-input-summary {background:rgba(68,119,170,.045);border:1px solid rgba(68,119,170,.14);border-radius:10px;padding:16px 18px;margin:8px 0 12px;}
.ai-input-summary strong {display:block;font-size:.95rem;font-weight:600;margin-bottom:5px;}
.ai-input-summary p {font-size:.78rem;line-height:1.6;margin:0;opacity:.8;}
.st-key-ai_context_panel {padding:16px 18px;margin-bottom:16px;border-radius:10px;background:rgba(128,145,170,.035);}
.st-key-ai_generate_panel {margin-top:6px;padding:16px 20px;border-radius:12px;border-color:rgba(68,119,170,.18);background:rgba(68,119,170,.035);}
@media(max-width:640px) {.st-key-ai_model_panel,.st-key-ai_input_panel {padding:18px;}.st-key-ai_generate_panel {padding:16px;}}
@media(max-width:760px) {
.st-key-ai_recommendation_settings [data-testid="stHorizontalBlock"] {flex-direction:column;}
.st-key-ai_recommendation_settings [data-testid="stColumn"] {width:100% !important;flex:1 1 100% !important;}
}
</style>""",unsafe_allow_html=True)
    st.markdown('<h2 class="recommendation-heading">Recommendation</h2><p class="recommendation-intro">Find a suitable preprocessing pipeline for your signals with AI guidance or transparent heuristic rules.</p>',unsafe_allow_html=True)
    source = st.segmented_control("Recommendation approach", ["AI-suggested pipeline", "Heuristic"],
                                  default="AI-suggested pipeline", key="recommendation_approach")
    if source != "AI-suggested pipeline":
        st.caption("Transparent profile-based rules derived from your selected signals.")
        st.warning("These heuristic rules have limited validation and may not generalize to your dataset. Review the suggested methods and compare their effects on your signals before running the pipeline.")
        with st.container(key="recommendation_grid"):
            for number, (col, (stage, label)) in enumerate(zip(st.columns(4), STAGE_LABELS.items()), start=1):
                with col:
                    heuristic_card(recs[stage], number=number, label=label)
        st.button("Use heuristic pipeline", key="use_heuristic_pipeline", type="primary",
                  on_click=use_recommendation, args=("Heuristic", context_key))
        with st.expander("Heuristic rules & thresholds"):
            st.markdown("""
These fixed decision rules select one method per stage for all selected signals.
Within each table, use the first matching row. Domain guardrails below override these choices.
Unless marked **maximum**, diagnostics use the **median across selected signals**.
**Sample count** means the number of input rows (samples per signal), including rows with missing values.

**Missing value imputation**

| Condition | Suggested method |
| --- | --- |
| Sample count ≤ 182.5 | Mean imputation |
| Sample count > 182.5 and longest missing run ≤ 0.1693 s and maximum local volatility ≤ 0.3987 | Linear interpolation |
| Sample count > 182.5 and longest missing run ≤ 0.1693 s and maximum local volatility > 0.3987 | Last observation carried forward (LOCF) |
| Sample count > 182.5 and longest missing run > 0.1693 s and missing burstiness ≤ 0.248 | LOCF |
| Sample count > 182.5 and longest missing run > 0.1693 s and missing burstiness > 0.248 | Linear interpolation |

**Outlier treatment**

| Condition | Suggested method |
| --- | --- |
| Spectral entropy > 0.8184 | Winsorization |
| Spectral entropy ≤ 0.8184 and maximum IQR outlier rate ≤ 2.33% | IQR |
| Spectral entropy ≤ 0.8184, maximum IQR outlier rate > 2.33% and maximum MAD outlier rate ≤ 7.62% | MAD |
| Spectral entropy ≤ 0.8184, maximum IQR outlier rate > 2.33% and maximum MAD outlier rate > 7.62% | Z-score |

**Optional smoothing**

| Condition | Suggested method |
| --- | --- |
| Maximum local volatility > 0.5748 | None |
| Maximum local volatility ≤ 0.5748 and skewness > 0.1067 | Butterworth |
| Maximum local volatility ≤ 0.5748, skewness ≤ 0.1067 and dominant-frequency ratio ≤ 0.4698 | Savitzky–Golay |
| Maximum local volatility ≤ 0.5748, skewness ≤ 0.1067 and dominant-frequency ratio > 0.4698 | Moving median |

**Normalization**

| Condition | Suggested method |
| --- | --- |
| Larger of the median IQR and median MAD outlier rates > 3%, or median excess kurtosis > 3 | RobustScaler |
| Otherwise | StandardScaler |

**Guardrails and overrides**

- **Imputation:** median missing rate ≤ 0 → None.
- **Outliers:** Finance domain → None. In any domain, median excess kurtosis > 8 together with median MAD outlier rate < 1% also → None.
- **Smoothing:** Finance or Network domain → None, preserving jumps and bursts by default. Irregular timestamps → None unless internal resampling is enabled.
- **Frequency diagnostics:** unavailable for irregular timestamps. The outlier rule then uses its IQR/MAD branch without the spectral-entropy condition.
- **Normalization:** Finance domain → None, preserving the original scale.

The longest-missing-run diagnostic is the median of each signal's longest run,
measured on the actual timestamp axis (the final sample uses the median interval). The rules above choose methods;
they are separate from the detection cutoffs and filter parameters used when those methods run.
""")
    else:
        import hashlib
        import json
        analysis_context = {k: v for k, v in context.items() if k != "signal_data"}
        analysis_key = hashlib.sha256(json.dumps(analysis_context, sort_keys=True, allow_nan=False).encode()).hexdigest()
        saved = matching_suggestions(st.session_state, analysis_key)
        with st.container(key="ai_recommendation_settings"):
            axis = context.get("time_axis", {})
            time_label = str(axis.get("column")) if axis.get("source") == "selected_timestamp" else "Observation timestamps"
            with st.container(border=True, key="ai_context_panel"):
                domain_column, time_column, goal_column = st.columns([1, 1, 2], gap="medium", vertical_alignment="center")
                domain_column.caption("Domain")
                domain_column.write(context["domain"].replace("_", " ").title())
                time_column.caption("Time source")
                time_column.write(time_label)
                with goal_column:
                    if "ai_goal" not in st.session_state:
                        st.session_state["ai_goal"] = st.session_state.get("_ai_goal", DEFAULT_USER_GOAL)
                    goal_options = {**USER_GOALS, CUSTOM_USER_GOAL: "Custom"}
                    st.selectbox("Preprocessing goal", list(goal_options), key="ai_goal",
                                 format_func=goal_options.get, on_change=remember_goal,
                                 help="Guides the AI recommendation. Changing the goal requires a matching suggestion; it does not run or apply a pipeline. Heuristic rules are unchanged.")
                    if st.session_state["ai_goal"] == CUSTOM_USER_GOAL:
                        if "ai_custom_goal" not in st.session_state:
                            st.session_state["ai_custom_goal"] = st.session_state.get("_ai_custom_goal", "")
                        st.text_area("Your goal or constraint", key="ai_custom_goal", max_chars=MAX_GOAL_LENGTH,
                                     on_change=remember_custom_goal,
                                     placeholder="Enter your own preprocessing goal or constraint…",
                                     help="Sent exactly as entered as user_goal. No explanation is added. The text is also recorded in the audit.")
            model_panel, input_panel = st.columns([1, 1.15], gap="medium")
            with model_panel, st.container(border=True, key="ai_model_panel", height="stretch"):
                st.markdown('<div class="ai-panel-heading"><span class="ai-panel-step">01</span><h3>Model &amp; access</h3><span class="ai-provider">OpenAI</span></div><p class="ai-panel-copy">Choose a model and provide an API key.</p>',unsafe_allow_html=True)
                model = st.selectbox("Model", list(MODELS), format_func=MODELS.get, key="ai_model")
                entered_key = st.text_input(
                    "OpenAI API key", type="password", key="ai_api_key_input",
                    value=st.session_state.get("_ai_session_api_key", ""),
                    placeholder="sk-…", on_change=remember_session_key,
                    help="Used only for this session. Not written to files or included in exports. Clear the field to remove your key.",
                ).strip()
                configured_key = server_api_key()
                api_key = entered_key or configured_key
                if entered_key:
                    st.caption("Session key ready · Usage is billed to your OpenAI project.")
                elif configured_key:
                    st.caption("Server key available · Add your own key to use your project.")
                else:
                    st.caption("Your key is used only for this session.")
            with input_panel, st.container(border=True, key="ai_input_panel", height="stretch"):
                st.markdown('<div class="ai-panel-heading"><span class="ai-panel-step">02</span><h3>Recommendation input</h3></div><p class="ai-panel-copy">Choose what the model can inspect.</p>',unsafe_allow_html=True)
                st.radio(
                    "Information sent to AI", list(INPUT_MODES), format_func=INPUT_MODES.get,
                    index=list(INPUT_MODES).index(st.session_state.get("_ai_input_mode", "diagnostics")),
                    key="ai_input_mode", on_change=remember_input_mode, horizontal=True,
                    label_visibility="collapsed",
                )
                signal_data = context.get("signal_data")
                if signal_data:
                    all_rows = signal_data["selection"] == "all_rows"
                    title = f"All {signal_data['total_rows']:,} rows included" if all_rows else f"{signal_data['included_rows']:,} of {signal_data['total_rows']:,} rows included"
                    description = "Diagnostics plus actual values of your selected signals." if all_rows else "Diagnostics plus a time overview and local windows at gaps and extreme values."
                    axis = signal_data["time_axis"]
                    time_label = str(axis["column"]) if axis["source"] == "selected_timestamp" else "Observation timestamps"
                else:
                    title = "Diagnostics only"
                    description = "Signal quality measurements and context, without individual signal values."
                    axis = context.get("time_axis", {})
                    time_label = str(axis.get("column")) if axis.get("source") == "selected_timestamp" else "Observation timestamps"
                st.markdown(
                    f'<div class="ai-input-summary"><strong>{escape(title)}</strong><p>{escape(description)}</p></div>',unsafe_allow_html=True,
                )
            render_request_preview(context, model)
            result = saved.get(context_key + ":" + model)
            if saved:
                with st.expander(f"Saved suggestions for this analysis ({len(saved)})"):
                    st.caption("Saved in this session. Choose a model and input above to view its result. All matching suggestions are also available in Pipeline lab.")
                    for saved_result in saved.values():
                        saved_mode = saved_result.get("input_mode", "diagnostics_data" if "signal_data" in saved_result["context"] else "diagnostics")
                        st.write(f"{MODELS.get(saved_result['model'],saved_result['model'])} · {INPUT_MODES[saved_mode]}")
            with st.container(border=True, key="ai_generate_panel", horizontal_alignment="center", vertical_alignment="center"):
                goal_ready = bool(context.get("user_goal", "").strip())
                generate = st.button("Regenerate suggestion" if result else "Generate suggestion", key="generate_ai_pipeline", type="primary", disabled=not bool(api_key) or not goal_ready, width="content")
                if not goal_ready:
                    st.caption("Enter your goal or constraint above to continue.")
                if result:
                    st.caption("Makes a new API request and replaces this saved suggestion.")
                if not api_key:
                    st.caption("Enter an API key above to continue.")
            if generate:
                try:
                    with st.spinner("Reviewing diagnostics and choosing a pipeline…"):
                        result = generate_suggestion(context, context_key, model, api_key)
                    save_suggestion(st.session_state, result)
                    st.session_state.pop("ai_pipeline_error", None)
                    st.rerun()
                except ValueError as exc:
                    st.session_state["ai_pipeline_error"] = str(exc)
            if st.session_state.get("ai_pipeline_error"):
                st.error(st.session_state["ai_pipeline_error"])
        result = st.session_state.get("ai_pipeline_results", {}).get(context_key + ":" + model)
        if result is None:
            legacy = st.session_state.get("ai_pipeline_result")
            if legacy and legacy["context_key"] == context_key and legacy["model"] == model:
                result = legacy
        valid = result is not None and result["context_key"] == context_key
        if result and not valid:
            st.warning("The saved AI suggestion belongs to different data or analysis settings. Generate a new pipeline for the current context.")
        elif valid:
            render_ai_result(result, context_key, method_label)
