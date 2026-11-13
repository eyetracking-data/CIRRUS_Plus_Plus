from __future__ import annotations

import json
import re
from itertools import cycle
import base64
import importlib
import sys
import time
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cirruspp_core as _cirruspp_core
# Refresh local core code on rerun, just like the dependent view modules below.
importlib.reload(_cirruspp_core)
from cirruspp_core import *  # noqa: F403,F401
from cirruspp_core import PIPELINE_ORDERS
# Refresh local views in dependency order: Streamlit reruns can otherwise retain
# cached modules whose exported functions no longer match the current files.
for _view_module in ("quality_visuals", "quality_index_view", "signal_profile_view", "ai_recommendation", "recommendation_view", "ground_truth_view", "export_view", "method_guide_view", "signal_calculation_guide_view"):
    if _view_module in sys.modules:
        importlib.reload(sys.modules[_view_module])
    else:
        importlib.import_module(_view_module)
from quality_index_view import render_quality_details
from signal_profile_view import render_signal_profile
from quality_visuals import COMPONENT_COLORS, render_quality_calculation, quality_component_chart
from ai_recommendation import build_context, apply_pipeline_source, matching_suggestions, MODELS, INPUT_MODES, AI_PIPELINE_SOURCES, DEFAULT_USER_GOAL
from recommendation_view import render_recommendations, stage_card_html
from ground_truth_view import render_ground_truth
from export_view import build_exports, render_export
from method_guide_view import render_method_guide
from signal_calculation_guide_view import render_signal_calculation_guide
from datetime import datetime, timezone
from recording_selection import detect_recording_blocks, select_recording
from source_adapter import (timestamp_unit_guidance, data_validity_options, gaze_rows_only,
                            interpret_validity_flags, paired_zero_gaze_signals, interpret_paired_zero_gaze)

st.set_page_config(page_title="CIRRUS++ | Domain-aware preprocessing", page_icon="☁️", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.comparison-flow {display:flex;align-items:center;gap:12px;margin:8px 0;}
.comparison-flow-step {flex:1;min-width:0;padding:14px 16px;border:1px solid rgba(128,145,170,.22);border-radius:10px;background:rgba(128,145,170,.035);}
.comparison-flow-step span {display:block;font-size:.75rem;opacity:.7;margin-bottom:6px;}
.comparison-flow-step strong {font-size:.9rem;font-weight:600;overflow-wrap:anywhere;}
.comparison-flow-varies {border:2px solid #4477aa;background:rgba(68,119,170,.09);}
.comparison-flow-arrow {opacity:.45;}
@media(max-width:760px) {.comparison-flow {flex-direction:column;align-items:stretch;}.comparison-flow-arrow {text-align:center;transform:rotate(90deg);}}

.block-container {padding-top:1rem; padding-bottom:3rem; max-width:1550px;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
.brand {display:flex;flex-direction:row;flex-wrap:nowrap;align-items:center;gap:.5rem;white-space:nowrap;font-size:1.5rem;font-weight:600;
letter-spacing:-.025em;color:var(--text-color);line-height:1.15;margin:0 0 .5rem;padding:0;}
.brand-icon {display:flex;align-items:center;color:#2563eb;font-size:1.85rem;line-height:1;}
.brand-plus {color:#2563eb;}
.st-key-analysis_header {padding-top:2rem;}
.st-key-signal_entry {max-width:1100px;margin:1.5rem auto;}
.signal-entry-heading h1 {font-size:clamp(1.7rem,3vw,2.3rem);letter-spacing:-.04em;margin:0;padding:0 0 .6rem;}
.signal-entry-heading p {opacity:.7;line-height:1.6;margin:0 0 1.25rem;max-width:75ch;}
.st-key-signal_picker {border-radius:16px;border-color:rgba(128,145,170,.22);padding:22px;}
.st-key-signal_entry [data-testid="stMetricValue"] {font-size:1.4rem;}
.st-key-data_entry {max-width:760px;margin:1.5rem auto 2rem;}
@media(min-width:900px) and (min-height:800px) {
.st-key-data_entry {margin-top:clamp(3rem,calc(30svh - 180px),16rem);}
[data-testid="stMain"]:has([role="tablist"]) .st-key-data_entry {margin-top:1.5rem;}}
.entry-heading h1 {font-size:clamp(1.8rem,4vw,2.6rem);font-weight:700;letter-spacing:-.045em;
line-height:1.15;margin:0;padding:0 0 .75rem;}
.entry-heading p {font-size:1rem;line-height:1.65;opacity:.7;margin:0 0 1.2rem;}
.st-key-data_entry [data-testid="stRadio"] [role="radiogroup"] {gap:12px;flex-wrap:wrap;}
.st-key-data_entry [data-testid="stRadioGroup"] > div {flex:1;min-width:200px;}
.st-key-data_entry [data-testid="stRadio"] label[data-baseweb="radio"],
.st-key-data_entry label[data-testid="stRadioOption"] {
display:flex;box-sizing:border-box;width:100%;cursor:pointer;
flex:1;min-width:200px;margin:0;padding:16px 18px;border:1px solid #dce2eb;border-radius:12px;}
.st-key-data_entry [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked),
.st-key-data_entry label[data-testid="stRadioOption"]:has(input:checked) {
border-color:#2563eb;background:rgba(37,99,235,.055);box-shadow:0 0 0 1px #2563eb;}
.st-key-data_entry label[data-testid="stRadioOption"]:has(input:focus-visible) {outline:2px solid #2563eb;outline-offset:4px;}
.st-key-data_entry label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {background:#2563eb;}
.st-key-data_entry [data-testid="stBaseButton-primary"] {background:#2563eb;border-color:#2563eb;}
.st-key-data_entry [data-testid="stFileUploaderDropzone"] {
border:1.5px dashed #bdcce1;border-radius:14px;padding:28px 20px;
background:rgba(37,99,235,.025);min-height:140px;}
.st-key-data_entry [data-testid="stExpander"] {border-color:#e4e8ef;border-radius:10px;}
.entry-path {display:flex;gap:12px;align-items:center;justify-content:center;
font-size:.78rem;opacity:.6;padding:1.25rem 0 .25rem;}
.entry-path b {font-weight:500;}.entry-path span {opacity:.5;}
@media(max-width:640px) {.st-key-data_entry {margin-top:.5rem;}
.brand {margin-bottom:.4rem;}.entry-path {gap:8px;font-size:.7rem;}}
.stage-card {border:1px solid rgba(59,130,246,.25);border-radius:16px;padding:.9rem;background:rgba(239,246,255,.60);min-height:185px}
.recommendation-stage {display:flex;align-items:baseline;gap:8px;min-height:48px;margin-bottom:8px;}
.recommendation-number {color:#2563eb;font-size:.95rem;font-weight:600;}
.recommendation-title {font-size:1.05rem;font-weight:600;line-height:1.4;letter-spacing:-.015em;}
.recommendation-card {border:1px solid rgba(59,130,246,.25);border-radius:14px;
padding:16px;background:rgba(239,246,255,.60);box-sizing:border-box;}
.recommendation-method {display:flex;align-items:center;font-size:1.5rem;font-weight:650;
line-height:1.3;letter-spacing:-.025em;min-height:52px;margin:8px 0;overflow-wrap:anywhere;}
.recommendation-copy {font-size:.82rem;line-height:1.6;color:#475569;min-height:64px;}
.recommendation-card .pill {font-size:.75rem;margin:0;}
.recommendation-detail {display:block;font-size:.78rem;font-weight:400;opacity:.65;}
.recommendation-rules {display:grid;gap:10px;}
.recommendation-rule {padding:10px 12px;border:1px solid rgba(128,145,170,.18);border-radius:9px;background:rgba(128,145,170,.04);}
.recommendation-rule-label {font-size:.75rem;line-height:1.45;opacity:.75;margin-bottom:6px;}
.recommendation-rule-values {display:flex;align-items:center;gap:10px;font-size:.9rem;line-height:1.4;font-variant-numeric:tabular-nums;white-space:nowrap;}
.recommendation-rule-values strong {font-weight:650;color:#4477aa;}
.recommendation-rule-values > span:first-of-type {opacity:.6;}
.recommendation-rule-note {font-size:.8rem;line-height:1.6;}
.st-key-recommendation_footer {margin-top:20px;}
.st-key-recommendation_footer > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {flex:3 1 calc(75% - .25rem);}
.st-key-recommendation_footer > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {flex:1 1 calc(25% - .75rem);}
.recommendation-next {display:flex;align-items:center;justify-content:flex-end;min-height:42px;
font-size:.9rem;line-height:1.5;color:var(--text-color);}
.recommendation-next strong {font-weight:600;}
@media(max-width:640px) {.recommendation-next {justify-content:flex-start;}}
@media(max-width:1100px) {
.st-key-recommendation_grid [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
.st-key-recommendation_grid [data-testid="stColumn"] {flex:1 1 calc(50% - 1rem);width:calc(50% - 1rem);min-width:0;}}
@media(max-width:640px) {
.st-key-recommendation_grid [data-testid="stColumn"] {flex-basis:100%;width:100%;}}
@media(max-width:640px) {.recommendation-stage {min-height:0;}.recommendation-copy {min-height:0;}.recommendation-method {min-height:0;margin:16px 0;}}
.pill {display:inline-block;padding:.18rem .52rem;margin:.08rem;border-radius:999px;border:1px solid #93c5fd;background:#eff6ff;font-size:.82rem}
.subtle {font-size:.86rem;color:#475569}.good{color:#166534}.warn{color:#92400e}
.info-card {border-left:4px solid #60a5fa;padding:.65rem .9rem;background:#eff6ff;border-radius:8px;margin:.3rem 0 .7rem 0}
.st-key-quality_components_panel h3,.st-key-domain_weights_panel h3,.st-key-data_preview_panel h3 {font-size:1.15rem;letter-spacing:-.02em;padding:0;}
.st-key-current_quality_score {background:rgba(68,119,170,.035);border-color:rgba(128,145,170,.18);border-radius:14px;padding:20px;justify-content:center;}
.st-key-current_quality_score [data-testid="stVerticalBlock"] {gap:12px;}
.st-key-current_quality_score [data-testid="stMetricValue"] {font-size:2.65rem;font-weight:600;letter-spacing:-.045em;}
.st-key-current_quality_score [data-testid="stBaseButton-tertiary"] {padding-left:0;color:#4477aa;}
.st-key-data_summary_panel {border:1px solid rgba(128,145,170,.2);border-radius:14px;padding:20px;container-type:inline-size;}
.dataset-stat-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:16px 0 18px;}
.dataset-stat {padding:0 12px 0 0;border-right:1px solid rgba(128,145,170,.16);}
.dataset-stat:last-child {border-right:none;}
.dataset-stat dt {font-size:.72rem;line-height:1.4;opacity:.65;margin:0 0 7px;}
.dataset-stat dd {font-size:1.5rem;font-weight:600;line-height:1.2;letter-spacing:-.03em;margin:0;}
.dataset-domain {border-top:1px solid rgba(128,145,170,.16);padding-top:14px;}
.dataset-domain-title {font-size:.85rem;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;}
.dataset-domain-title span {opacity:.6;}
.dataset-domain-title strong {font-weight:600;}
.dataset-domain p {font-size:.8rem;line-height:1.55;opacity:.7;margin:7px 0 0;}
@container(max-width:440px) {.dataset-stat-grid {grid-template-columns:repeat(2,minmax(0,1fr));}.dataset-stat dt {min-height:0;}}
@media(max-width:900px) {
 .st-key-overview_summary [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 .st-key-overview_summary [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100%;width:100%;min-width:0;}
}
.st-key-current_quality_score p.quality-index-explanation {font-size:.74rem;line-height:1.75;margin:10px 0 0;max-width:65ch;font-weight:400;}
.quality-summary-label {font-size:.88rem;margin:0;}
.quality-summary-number {font-size:2.65rem;font-weight:600;letter-spacing:-.045em;line-height:1.15;white-space:nowrap;}
.quality-summary-number span {font-size:.95rem;font-weight:400;letter-spacing:0;opacity:.65;}
.st-key-score_value_link [data-testid="stColumn"] {min-width:0;}
.st-key-score_value_link button p {font-size:.78rem;white-space:normal;text-align:left;line-height:1.4;}
.quality-index-explanation-note {font-size:.8rem;line-height:1.5;opacity:.65;margin:6px 0 0;}
.overview-heading {font-size:1.65rem;font-weight:650;letter-spacing:-.04em;margin:0;line-height:1.25;}
.overview-intro {font-size:.88rem;opacity:.65;margin:6px 0 16px;}
.st-key-quality_components_panel,.st-key-domain_weights_panel,.st-key-data_preview_panel {
border-radius:16px;border-color:rgba(128,145,170,.2);padding:20px;box-shadow:0 3px 14px rgba(15,23,42,.025);}
.st-key-comparison_setup {
background:rgba(37,99,235,.035);border-color:rgba(37,99,235,.17);border-radius:16px;padding:20px;}
.st-key-comparison_action {border-radius:14px;border-color:rgba(128,145,170,.22);}
.st-key-comparison_pipeline_panel [data-testid="stExpander"] {background:#fff;color:#1f2937;border-color:rgba(128,145,170,.23);border-radius:12px;overflow:hidden;}
.st-key-comparison_pipeline_panel [data-testid="stExpander"] > details > summary {background:#fff;color:#1f2937;}
.st-key-comparison_pipeline_panel [data-testid="stExpanderDetails"] {background:#fff;color:#1f2937;}
.st-key-quality_components_panel [data-testid="stVerticalBlock"],
.st-key-domain_weights_panel [data-testid="stVerticalBlock"] {gap:10px;}
.st-key-quality_panel_header,.st-key-weights_panel_header {min-height:0;}
.chart-heading {font-size:1.15rem;font-weight:650;letter-spacing:-.02em;line-height:1.35;margin:9px 0 12px;}
.chart-description {font-size:.82rem;line-height:1.6;opacity:.65;margin:0 0 4px;}
.st-key-quality_components_panel [data-testid="stButtonGroup"] {margin-top:6px;margin-bottom:8px;}
.st-key-quality_components_panel [data-testid="stButtonGroup"] button {border-radius:8px;font-size:.78rem;}
.panel-kicker {font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.55;margin:0 0 5px;}
.panel-meta {font-size:.76rem;opacity:.6;margin:6px 0 0;}
.preview-meta {font-size:.76rem;opacity:.6;text-align:right;margin:4px 0;}
@media(max-width:1400px) {
 .st-key-overview_charts [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 .st-key-overview_charts [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100%;width:100%;min-width:0;}
}
@media(max-width:640px) {
 .st-key-current_quality_score,.st-key-quality_components_panel,.st-key-domain_weights_panel,.st-key-data_preview_panel {padding:16px;}
 .preview-meta {text-align:left;}
}
.st-key-workflow_tabs > div > [role="tablist"],
.st-key-workflow_tabs > [data-baseweb="tab-list"] {
 display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;background:rgba(128,145,170,.055);
 border:1px solid rgba(128,145,170,.16);border-radius:14px;
 padding:8px;margin-bottom:24px;overflow-x:auto;scrollbar-width:thin;}
.st-key-workflow_tabs > div > [role="tablist"] > [role="tab"],
.st-key-workflow_tabs > [data-baseweb="tab-list"] [role="tab"] {
 height:auto;min-height:58px;min-width:0;justify-content:center;padding:10px 12px;border-radius:9px;
 border:1px solid rgba(128,145,170,.2);background:var(--background-color);
 color:var(--text-color);white-space:normal;text-align:center;
 transition:background .15s,border-color .15s,box-shadow .15s;}
.st-key-workflow_tabs > div > [role="tablist"] > [role="tab"] p,
.st-key-workflow_tabs > [data-baseweb="tab-list"] [role="tab"] p {
 font-size:.85rem;font-weight:500;line-height:1.4;}
.st-key-workflow_tabs > div > [role="tablist"] > [role="tab"]:hover,
.st-key-workflow_tabs > [data-baseweb="tab-list"] [role="tab"]:hover {
 background:rgba(37,99,235,.07);border-color:rgba(37,99,235,.35);}
.st-key-workflow_tabs > div > [role="tablist"] > [role="tab"][aria-selected="true"],
.st-key-workflow_tabs > [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {
 background:#2563eb;color:#fff;border-color:#2563eb;
 box-shadow:0 2px 6px rgba(37,99,235,.18);}
.st-key-workflow_tabs > div > [role="tablist"] > [role="tab"]:focus-visible {
 outline:2px solid #2563eb;outline-offset:3px;}
.st-key-workflow_tabs > div > [role="tablist"] .react-aria-SelectionIndicator,
.st-key-workflow_tabs > [data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
.st-key-workflow_tabs > [data-baseweb="tab-list"] [data-baseweb="tab-border"] {display:none;}
@media(max-width:1400px) {
 .st-key-workflow_tabs > div > [role="tablist"],
 .st-key-workflow_tabs > [data-baseweb="tab-list"] {grid-template-columns:repeat(4,minmax(0,1fr));}}
@media(max-width:640px) {
 .st-key-workflow_tabs > div > [role="tablist"],
 .st-key-workflow_tabs > [data-baseweb="tab-list"] {grid-template-columns:repeat(2,minmax(0,1fr));}}
[class*="st-key-workflow_footer_"] {margin-top:24px;padding-top:16px;border-top:1px solid rgba(128,145,170,.18);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

METHOD_LABELS = {
    "none":"None", "mean":"Mean", "median":"Median", "locf":"LOCF", "linear":"Linear interpolation",
    "knn":"KNN imputation", "mice":"MICE", "zscore":"Z-score", "iqr":"IQR", "mad":"MAD",
    "winsorize":"Winsorization", "isolation_forest":"Isolation Forest", "moving_median":"Moving median",
    "savgol":"Savitzky–Golay", "butterworth":"Butterworth", "standard":"StandardScaler",
    "minmax":"MinMaxScaler", "robust":"RobustScaler",
}
IMP_METHODS=METHOD_OPTIONS["imputation"]
OUT_METHODS=METHOD_OPTIONS["outlier"]
SMOOTH_METHODS=METHOD_OPTIONS["smoothing"]
SCALE_METHODS=METHOD_OPTIONS["scaling"]
LABEL_TO_DOMAIN={v:k for k,v in DOMAIN_LABELS.items()}  # noqa: F405

# Allocate contrasting colors to the traces actually selected, regardless of
# method names. The same selection/order gives the same colors in every view.
COMPARISON_COLORS=["#D55E00","#009E73","#882255","#CC79A7","#8C6D00","#56B4E9","#333333"]

def comparison_colors(names):
    palette=cycle(["#0072B2",*COMPARISON_COLORS] if "Raw" not in names else COMPARISON_COLORS)
    return {name:"#0072B2" if name=="Raw" else next(palette) for name in names}

def ml(m:str)->str: return METHOD_LABELS.get(m,m.replace("_"," ").title())
def safe_json(x:Any)->bytes: return json.dumps(x,indent=2,ensure_ascii=False,default=str).encode("utf-8")

def downsample(n:int,max_points:int):
    return np.arange(n) if n<=max_points else np.linspace(0,n-1,max_points).astype(int)

def line_plot(series:dict[str,pd.Series], timestamp:pd.Series, max_points:int, title:str):
    n=min(len(timestamp),*(len(s) for s in series.values())); base_idx=downsample(n,max_points); fig=go.Figure()
    colors=comparison_colors(list(series))
    dashes=["solid","dash","dot","dashdot","longdash","longdashdot"]
    for i,(name,s) in enumerate(series.items()):
        color=colors[name]
        values=pd.to_numeric(s.iloc[:n],errors="coerce").replace([np.inf,-np.inf],np.nan)
        missing=values.isna().to_numpy()
        boundaries=np.flatnonzero(missing[:-1]!=missing[1:])
        # Keep both sides of every gap, including isolated observed samples,
        # even when regular display downsampling would skip them.
        idx=np.unique(np.r_[base_idx,boundaries,boundaries+1])
        fig.add_trace(go.Scatter(x=timestamp.iloc[idx],y=values.iloc[idx],mode="lines+markers",name=name,marker=dict(size=3,color=color),line=dict(color=color,width=2.6 if i==0 else 2.2,dash=dashes[i%len(dashes)]),connectgaps=False))
    fig.update_layout(height=390,title=title,margin=dict(l=35,r=20,t=55,b=35),legend=dict(orientation="h"),xaxis_title="Relative time (s)")
    return fig

def imputation_plot(raw:pd.Series, outputs:dict[str,pd.Series], stage_input:pd.Series,
                    timestamp:pd.Series, max_points:int, title:str):
    """Show actual imputed segments, anchored to adjacent observed samples."""
    fig=line_plot({"Raw":raw},timestamp,max_points,title)
    colors=comparison_colors(["Raw",*(ml(method) for method in outputs)])
    observed=np.isfinite(stage_input.to_numpy(dtype=float))
    base_idx=downsample(len(stage_input),max_points)
    dashes=["dash","dot","dashdot","longdash","longdashdot"]
    for number,(method,output) in enumerate(outputs.items()):
        color=colors[ml(method)]
        values=output.to_numpy(dtype=float)
        filled=~observed & np.isfinite(values)
        edges=np.diff(np.r_[False,filled,False].astype(int))
        plot_x=[]; plot_y=[]
        for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            sampled=base_idx[np.searchsorted(base_idx,start):np.searchsorted(base_idx,stop)]
            indices=np.unique(np.r_[start,sampled,stop-1])
            if start>0 and observed[start-1]: indices=np.r_[start-1,indices]
            if stop<len(observed) and observed[stop]: indices=np.r_[indices,stop]
            plot_x.extend(timestamp.iloc[indices].tolist()+[None])
            plot_y.extend([float(stage_input.iloc[i]) if observed[i] else float(values[i]) for i in indices]+[None])
        fig.add_trace(go.Scatter(x=plot_x,y=plot_y,mode="lines+markers",name=ml(method),
                                marker=dict(size=4,color=color),line=dict(color=color,width=2.4,dash=dashes[number%len(dashes)]),connectgaps=False))
    return fig

def method_difference_window(outputs:dict[str,pd.Series], pad:int=60, width:int=320)->tuple[int,int] | None:
    """Find a local window where compared method traces actually disagree."""
    if len(outputs)<2:
        return None

    arrays=[]
    for value in outputs.values():
        # Defensive handling: this helper expects one signal per method.
        # If a one-column DataFrame reaches it, safely reduce it to a Series.
        if isinstance(value,pd.DataFrame):
            if value.shape[1] != 1:
                return None
            value=value.iloc[:,0]
        arr=pd.to_numeric(value,errors="coerce").to_numpy(dtype=float)
        arrays.append(arr)

    n=min(len(a) for a in arrays)
    if n == 0:
        return None
    arrays=[a[:n] for a in arrays]

    ref=arrays[0]
    mask=np.zeros(n,dtype=bool)
    for b in arrays[1:]:
        finite_ref=np.isfinite(ref); finite_b=np.isfinite(b)
        mask |= finite_ref != finite_b
        both=finite_ref & finite_b
        mask[both] |= ~np.isclose(ref[both],b[both],rtol=1e-7,atol=1e-9)

    idx=np.flatnonzero(mask)
    if not len(idx):
        return None
    center=int(idx[0])
    start=max(0,center-pad)
    end=min(n,max(start+width,center+pad))
    return start,end

def focused_line_plot(raw:pd.Series, outputs:dict[str,pd.Series], timestamp:pd.Series, title:str, *, imputation_input:pd.Series | None=None):
    compared=outputs if imputation_input is None else {k:v.where(imputation_input.isna()) for k,v in outputs.items()}
    win=method_difference_window(compared)
    if win is None: return None
    if imputation_input is not None:
        start,end=win
        return imputation_plot(raw.iloc[start:end],{k:v.iloc[start:end] for k,v in outputs.items()},
                               imputation_input.iloc[start:end],timestamp.iloc[start:end],5000,title+f" · samples {start}–{end-1}")
    start,end=win; series={"Raw":raw.iloc[start:end]}
    series.update({ml(k):v.iloc[start:end] for k,v in outputs.items()})
    ts=timestamp.iloc[start:end].reset_index(drop=True); series={k:v.reset_index(drop=True) for k,v in series.items()}
    return line_plot(series,ts,5000,title+f" · samples {start}–{end-1}")

def method_guide():
    render_method_guide(fs=fs,time_processing=time_processing)

def longest_complete_run_length(frame:pd.DataFrame)->int:
    cols=[c for c in frame.columns if c!="timestamp"]
    if not cols or frame.empty: return 0
    mask=frame[cols].notna().all(axis=1).to_numpy(bool)
    if not mask.any(): return 0
    padded=np.r_[False,mask,False].astype(int); edges=np.flatnonzero(np.diff(padded))
    runs=edges[1::2]-edges[::2]
    return int(runs.max()) if len(runs) else 0

def metric_change_guide():
    st.caption("Before = original input. After = processed output in original units, with normalization removed for comparison.")
    guide=pd.DataFrame([
        ["Quality index", "Domain-weighted diagnostic score (0–100). Higher does not guarantee a more accurate signal."],
        ["Missing rate", "Fraction of unavailable values. Lower means fewer gaps."],
        ["IQR / MAD outlier rate", "Fraction of statistically unusual values detected by each rule. Genuine events can also be flagged."],
        ["Skewness", "Distribution asymmetry. Values near zero indicate a more symmetric distribution."],
        ["Excess kurtosis", "Tail weight relative to a normal distribution. Large positive values can reflect extreme observations."],
        ["Lag-1 autocorrelation", "Similarity between adjacent samples. Smoothing often increases it."],
        ["Drift score", "Slow movement of the rolling mean relative to overall variability."],
        ["Local volatility", "Sample-to-sample fluctuation relative to overall variability. Lower means smoother."],
        ["Spectral entropy", "How widely signal power is spread across frequencies. Lower means more concentrated power."],
    ],columns=["Metric","Meaning"])
    st.dataframe(guide,hide_index=True,width="stretch")


def domain_weight_chart(weights:dict[str,float]):
    labels={"completeness":"Completeness","continuity":"Continuity","plausibility":"Outlier burden","distribution":"Distribution shape","stability":"Temporal stability","correlation":"Cross-signal consistency"}
    df=pd.DataFrame({"component":[labels[k] for k in weights],"weight":list(weights.values())})
    fig=go.Figure(go.Pie(labels=df.component,values=df.weight,sort=False,
                        textinfo="percent",textposition="inside",textfont=dict(size=14,color="white"),
                        marker=dict(colors=[COMPONENT_COLORS[label] for label in df.component],
                                    line=dict(color="white",width=2)),
                        hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.update_layout(height=340,showlegend=True,
                      legend=dict(orientation="h",x=.5,xanchor="center",y=-.08,yanchor="top"),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0,r=0,t=10,b=70),font=dict(size=12,color="#64748b"))
    return fig

def recommendation_card(rec, *, number, label):
    metric_labels={"n":"Sample count","longest missing run":"Longest missing run",
                   "max local volatility":"Maximum local volatility","missing burstiness":"Missing burstiness",
                   "spectral entropy":"Spectral entropy","max IQR outlier rate":"Maximum IQR outlier rate",
                   "max MAD outlier rate":"Maximum MAD outlier rate","skewness":"Skewness",
                   "dominant-frequency ratio":"Dominant-frequency ratio"}
    rules=[]
    for trace in rec.rule_trace:
        match=re.fullmatch(r"([^=]+)=\s*(.+?)\s*(≤|≥|<|>)\s*(.+)",trace)
        if match:
            metric,value,operator,threshold=match.groups()
            metric=metric_labels.get(metric.strip(),metric.strip().capitalize())
            rules.append(f'<div class="recommendation-rule"><div class="recommendation-rule-label">{escape(metric)}</div><div class="recommendation-rule-values"><strong>{escape(value)}</strong><span>{escape(operator)}</span><span>{escape(threshold)}</span></div></div>')
        else:
            rules.append(f'<div class="recommendation-rule recommendation-rule-note">{escape(trace)}</div>')
    explanation='<div class="recommendation-rules">'+''.join(rules)+'</div>'
    st.markdown(stage_card_html(stage=rec.stage,number=number,method=rec.method,
                               method_name=ml(rec.method),explanation=explanation),unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_upload(data:bytes,filename:str,sep:str,dec:str): return read_table_bytes(data,filename,sep,dec)  # noqa: F405
@st.cache_data(show_spinner=False)
def synth(domain:str,n:int,seed:int,missing_rate:float): return generate_domain_series(domain,n,seed,missing_rate)  # noqa: F405

# ---------------------------------------------------------------------------
# Data entry
# ---------------------------------------------------------------------------
if "analysis_data" not in st.session_state:
    with st.container(key="data_entry"):
        st.markdown('<div class="brand"><span class="brand-icon" aria-hidden="true">☁</span><span>CIRRUS<span class="brand-plus">++</span></span></div>',unsafe_allow_html=True)
        st.markdown('<div class="entry-heading"><h1>Explore your time-series data</h1><p>Understand your time-series data, review suggested cleaning steps, and export your results.</p></div>',unsafe_allow_html=True)
        source_mode=st.radio(
            "Choose your data source",["Upload CSV/TSV","Synthetic demo"],index=0,
            format_func=lambda value: {"Upload CSV/TSV":"Upload a file","Synthetic demo":"Try example data"}[value],
            horizontal=True,label_visibility="collapsed",
        )
        parser_note=""; source_event_mask=None
        if source_mode=="Synthetic demo":
            st.caption("Explore generated time-series data without uploading a file.")
            with st.expander("Example data settings"):
                dlab=st.selectbox("Example domain",list(DOMAIN_LABELS.values()))  # noqa: F405
                n=st.slider("Number of samples",400,8000,2400,200); seed=st.number_input("Random seed",0,1_000_000,42)
                missing_percent=st.slider("Missing values (%)",0,50,10,1,help="Percentage of signal values to remove across all signals. Timestamps stay complete.")
                st.caption("Randomly placed single missing values and gaps of varying lengths. Change the random seed to create a different pattern.")
            raw_df,source_event_mask,synthetic_fs=synth(LABEL_TO_DOMAIN[dlab],n,int(seed),missing_percent/100); source_name=f"synthetic_{LABEL_TO_DOMAIN[dlab]}.csv"
            separator=decimal="Auto"
        else:
            st.caption("A CSV, TSV or TXT file with one sample per row and signals in columns.")
            up=st.file_uploader("Upload your time-series data",type=["csv","tsv","txt"],label_visibility="collapsed")
            with st.expander("Import settings"):
                st.caption("File formatting is detected automatically. Adjust these settings if the preview looks incorrect.")
                separator=st.selectbox("Column separator",["Auto",",",";","Tab","|"])
                decimal=st.selectbox("Decimal separator",["Auto",".",","])
            if up is None:
                st.markdown('<div class="entry-path"><b>1 &nbsp; Load data</b><span>→</span><b>2 &nbsp; Inspect &amp; clean</b><span>→</span><b>3 &nbsp; Export results</b></div>',unsafe_allow_html=True)
                st.stop()
            upload_bytes=up.getvalue()
            try:
                loaded=load_upload(upload_bytes,up.name,separator,decimal)
            except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
                st.error("This file could not be read. Check that it contains a CSV or TSV table, or adjust the import settings.")
                with st.expander("Import error details"):
                    st.code(str(exc))
                st.stop()
            raw_df=loaded.frame; source_name=loaded.source_name; parser_note=f"separator={loaded.separator!r}, decimal={loaded.decimal!r}, encoding={loaded.encoding}"

        with st.container(border=True):
            st.write("**Your data at a glance**")
            st.caption(source_name)
            summary=st.columns(3)
            summary[0].metric("Samples",f"{len(raw_df):,}")
            summary[1].metric("Columns",len(raw_df.columns))
            summary[2].metric("Detected numeric signals",len(numeric_signal_columns(raw_df,False)))  # noqa: F405
            uploaded_blocks = detect_recording_blocks(raw_df) if source_mode != "Synthetic demo" else []
            if uploaded_blocks:
                st.info(
                    f"Multiple recordings detected: {len(uploaded_blocks)} contiguous blocks "
                    f"identified by '{uploaded_blocks[0].source_column}'. "
                    "After Inspect data, select one recording in the sidebar. "
                    "Different participants/sessions are not joined into one time series."
                )
            if source_mode=="Synthetic demo":
                example_signals=signal_columns(raw_df)  # noqa: F405
                actual_missing=raw_df[example_signals].isna().to_numpy().mean()
                st.caption(f"Missing signal values: {actual_missing:.1%} · Includes isolated values and gaps of varying lengths. Timestamps are complete.")
            with st.expander("Preview first rows"):
                st.dataframe(raw_df.head(10),width="stretch",hide_index=True)

        st.caption("Check the preview, then continue to inspect signal quality. Your data is not cleaned at this step.")
        if st.button("Inspect data",type="primary"):
            st.session_state["analysis_data"]={
                "frame":raw_df,"source_mode":source_mode,"source_name":source_name,
                "parser_note":parser_note,"separator":separator,"decimal":decimal,
                "synthetic_fs":synthetic_fs if source_mode=="Synthetic demo" else None,
            }
            st.rerun()
        st.stop()

data=st.session_state["analysis_data"]
raw_df=data["frame"]
source_mode=data["source_mode"]; source_name=data["source_name"]
parser_note=data["parser_note"]; separator=data["separator"]; decimal=data["decimal"]
synthetic_fs=data["synthetic_fs"]

with st.container(key="analysis_header"):
    heading,change=st.columns([5,1],vertical_alignment="center")
    with heading:
        st.markdown('<div class="brand"><span class="brand-icon" aria-hidden="true">☁</span><span>CIRRUS<span class="brand-plus">++</span></span></div>',unsafe_allow_html=True)
        st.caption(f"Inspecting {source_name}")
    with change:
        if st.button("Change data",width="stretch"):
            for key in list(st.session_state):
                if key in {"analysis_data","compare_result","pipeline","ground_truth","clean_ref_result","quality_details_open","method_guide_open","signal_calculation_guide_open","ai_pipeline_result","ai_pipeline_results","final_ai_result","ai_pipeline_error","timestamp_unit","timestamp_column","max_imputation_gap","limit_imputation_gap","resample_smoothing","selected_signal_columns","pending_signal_columns","signal_entry_editor","recording_segment","gaze_rows_only","missing_flag_mask","eye_validity_mask","paired_gaze_zero_mask","_timestamp_unit_suggested_for"} or key.startswith("final_pipeline_"):
                    st.session_state.pop(key,None)
            st.rerun()

signal_selection_notice=st.empty()
# A CSV may contain recordings from multiple participants. Select a contiguous
# block before evaluating timestamps; validating the concatenation would produce
# false "backward steps" at participant/session boundaries. Do not concatenate
# repeated participant IDs separated by other recordings.
all_recording_blocks = detect_recording_blocks(raw_df) if source_mode != "Synthetic demo" else []
if all_recording_blocks:
    def _recording_changed():
        # Any previous diagnostics/pipelines refer to another recording.
        reset = {"compare_result", "pipeline", "ground_truth", "clean_ref_result",
                 "quality_details_open", "method_guide_open", "signal_calculation_guide_open",
                 "ai_pipeline_result", "ai_pipeline_results", "final_ai_result",
                 "ai_pipeline_error", "timestamp_unit", "timestamp_column",
                 "max_imputation_gap", "limit_imputation_gap", "resample_smoothing",
                 "selected_signal_columns", "pending_signal_columns", "signal_entry_editor",
                 "gaze_rows_only", "missing_flag_mask", "eye_validity_mask", "paired_gaze_zero_mask","_timestamp_unit_suggested_for"}
        for key in list(st.session_state):
            if key in reset or key.startswith("final_pipeline_"):
                st.session_state.pop(key, None)

    with st.sidebar:
        st.header("Recording selection")
        st.caption(f"{len(all_recording_blocks)} contiguous recordings detected. The first is selected by default; choose another here.")
        recording_choice = st.selectbox(
            "Recording / participant", options=list(range(len(all_recording_blocks))),
            index=0, key="recording_segment", on_change=_recording_changed,
            placeholder="Choose one recording",
            format_func=lambda idx: (
                f"#{idx + 1} · {all_recording_blocks[idx].source_column} "
                f"{all_recording_blocks[idx].label} "
                f"({all_recording_blocks[idx].n_rows:,} samples)"
            ),
        )
    if recording_choice is None:
        signal_selection_notice.info(
            f"This file contains {len(all_recording_blocks)} different recordings. "
            "Choose a single recording in the sidebar before proceeding. "
            "Samples and timestamps will be preserved in their original order."
        )
        st.stop()
    selected_block = all_recording_blocks[recording_choice]
    raw_df = select_recording(raw_df, selected_block)
    source_name = f"{source_name} [recording {recording_choice + 1}: {selected_block.source_column} {selected_block.label}]"
    st.caption(
        f"Selected recording {recording_choice + 1}/{len(all_recording_blocks)}: "
        f"{selected_block.n_rows:,} original rows from "
        f"{selected_block.source_column} {selected_block.label}. No joining, sorting, "
        "timestamp filling or resampling was performed."
    )
# Tobii exports contain event and mouse rows with repeated timestamps.
# A Streamlit-visible, reversible choice selects the actual gaze sample stream.
import_selection_meta = {"raw_rows_after_recording_choice": len(raw_df)}
if "Sensor" in raw_df and "Eye Tracker" in raw_df["Sensor"].astype("string").unique():
    with st.sidebar:
        with st.container(border=True):
            st.caption("Mixed sensor/event stream")
            gaze_only = st.checkbox("Analyse Eye Tracker rows only", key="gaze_rows_only", value=True,
                help="Tobii event/mouse rows may share timestamps. This excludes them from the analysis view but leaves the source file unchanged.")
    if gaze_only:
        raw_df, subset_meta = gaze_rows_only(raw_df)
        import_selection_meta["gaze_stream"] = subset_meta
        st.caption(f"Eye Tracker analysis stream: {len(raw_df):,} gaze samples. "
                   f"{subset_meta['excluded_rows']:,} mouse/event rows excluded from analysis, not deleted from the original data.")
if "pending_signal_columns" in st.session_state:
    st.session_state["selected_signal_columns"]=st.session_state.pop("pending_signal_columns")
with st.sidebar:
    st.caption("Choose what to analyse here. Review your data and domain assumptions in Data overview.")

    st.header("Data settings")
    time_options=list(raw_df.columns)
    if ("timestamp_column" not in st.session_state
            or st.session_state["timestamp_column"] not in time_options):
        st.session_state["timestamp_column"] = preferred_timestamp_column(raw_df)  # noqa: F405
    def _timestamp_column_changed():
        # A manually selected unit for another column must not silently carry over.
        st.session_state["timestamp_unit"] = "Auto"
    timestamp_choice=st.selectbox("Timestamp column",time_options,index=None,
        key="timestamp_column",on_change=_timestamp_column_changed,
        placeholder="Select the actual recording time column")
    timestamp_col=timestamp_choice
    unit_guidance = timestamp_unit_guidance(raw_df, timestamp_col)
    if ("timestamp_unit" not in st.session_state or
            (st.session_state["timestamp_unit"] == "Auto" and
             st.session_state.get("_timestamp_unit_suggested_for") != timestamp_col)):
        # Only preselect on initial load/column switch. Manual choice remains authoritative.
        if unit_guidance:
            st.session_state["timestamp_unit"] = unit_guidance["unit"]
        st.session_state["_timestamp_unit_suggested_for"] = timestamp_col
    timestamp_unit=st.selectbox("Timestamp unit",list(TIMESTAMP_UNITS),key="timestamp_unit",
        help="An import suggestion may preselect the unit. Verify it against your recording metadata; changing it never changes the source values.")
    if unit_guidance and timestamp_unit == unit_guidance["unit"]:
        label = "Verified against recording metadata" if unit_guidance["confidence"] == "verified" else "Suggested; verify acquisition unit"
        st.info(f"{label}: {unit_guidance['unit']}. {unit_guidance['detail']}")
    if timestamp_col is None:
        signal_selection_notice.info(
            "Choose a recording timestamp column and its unit in Data settings. "
            "No timestamp was detected reliably. A row index is not automatically treated as time. "
            "If your file has only sample numbers, add timestamps using the documented sampling rate before importing."
        )
        st.stop()
    if str(timestamp_col).strip().lower().startswith(("unnamed", "index", "row", "sample")):
        st.warning("This column looks like a row/sample index, not recording time. "
                   "The chosen unit is applied literally and can produce an incorrect sampling rate. "
                   "Prefer actual recorded timestamps for temporal diagnostics.")
    try:
        time_frame=canonical_frame(raw_df,[],1.,timestamp_col,timestamp_unit)
        time_checks=validate_timestamps(time_frame.timestamp)
        fs=float(time_checks["sampling_rate_hz"])
        if not 1e-9 <= fs <= 10000:
            raise ValueError(f"The timestamp unit gives {fs:g} Hz, outside the supported range. Check the unit or recording.")
    except (ValueError, TypeError, OverflowError) as exc:
        if "unit is ambiguous" in str(exc).lower():
            signal_selection_notice.info(
                f"The time unit of '{timestamp_col}' is not specified in the file. "
                "Select its actual unit under Timestamp unit (e.g. milliseconds). "
                "CIRRUS++ will not guess the sampling rate from numeric values."
            )
        else:
            signal_selection_notice.error(f"Timestamp setup: {exc}")
        st.stop()
    st.caption(f"Time unit: {time_frame.attrs['time_axis']['original_unit']} · Median timestamp step: {time_checks['median_interval_seconds']:.6g} s · Estimated sampling rate: {fs:.6g} Hz · Duration: {time_checks['duration_seconds']:.6g} s")
    if time_checks['sampling_rate_estimation'] != "inverse median of positive intervals":
        st.caption("Sampling rate estimate accounts for quantised timestamp increments; verify against the acquisition settings.")
    if time_checks["irregular_intervals"]:
        st.warning("Timestamp intervals vary by more than 5% from the median. Original observation times are retained.")
    else:
        st.caption("Timestamp intervals are regular within a 5% tolerance.")
    with st.expander("Time processing"):
        limit_imputation_gap=st.checkbox("Limit gap duration",value=False,key="limit_imputation_gap",
            help="Only fill gaps up to a chosen duration. Leave unchecked to fill internal gaps without a duration limit.")
        max_imputation_gap=None
        if limit_imputation_gap:
            max_imputation_gap=st.number_input("Maximum gap to fill (s)",min_value=.001,value=float(time_frame.attrs["time_processing"]["max_imputation_gap_seconds"]),step=.1,key="max_imputation_gap",
                help="Imputation fills only gaps bounded by observed values within this time span. Leading/trailing gaps and longer gaps remain missing. This limit applies to all imputation methods.")
        resample_smoothing=st.checkbox("Resample internally for smoothing",value=False,key="resample_smoothing",
            help="Required for irregular timestamps when smoothing is selected. A regular grid at the median-derived sampling rate is used within finite signal segments; filtered values are returned at the original times. No rows are added to the export.")
    time_processing={"max_imputation_gap_seconds":max_imputation_gap,
                     "irregular_smoothing":"resample" if resample_smoothing else "disabled"}
    possible=[c for c in numeric_signal_columns(raw_df) if c != timestamp_col]  # noqa: F405
    if not possible:
        signal_selection_notice.warning("No usable numeric signals detected. Check your file's column types and names. Identifier-like, time-like and constant numeric columns are excluded.")
        st.stop()
    preliminary,_=infer_domain(raw_df,possible)  # noqa: F405
    possible=rank_signal_columns(possible,preliminary)  # noqa: F405
    duplicate_groups=exact_duplicate_signal_groups(raw_df,possible)  # noqa: F405
    st.session_state["selected_signal_columns"]=[c for c in st.session_state.get("selected_signal_columns",[]) if c in possible]
    selected_signals=st.multiselect("Signal columns",possible,placeholder="Choose signals to analyse",key="selected_signal_columns")
    if duplicate_groups:
        dup_text="; ".join(" = ".join(g) for g in duplicate_groups)
        st.caption("Exact duplicate numeric channels detected. Choose one representative from each group: "+dup_text)
    if not selected_signals:
        with signal_selection_notice.container():
            with st.container(key="signal_entry"):
                st.markdown('<div class="panel-kicker">Set up your analysis</div><div class="signal-entry-heading"><h1>Which signals would you like to explore?</h1><p>Select the columns you want to analyse.</p></div>',unsafe_allow_html=True)
                stats=st.columns(3)
                stats[0].metric("Samples",f"{len(raw_df):,}")
                stats[1].metric("Columns in your file",f"{len(raw_df.columns):,}")
                stats[2].metric("Numeric signal candidates",f"{len(possible):,}")
                with st.container(border=True,key="signal_picker"):
                    st.subheader("Choose your signal columns")
                    st.caption("Tick the columns to include. You can adjust your selection later in the sidebar.")
                    # Cache the overview across selections; previews use evenly
                    # spaced raw rows and finite-value counts, without cleaning input.
                    @st.cache_data(show_spinner=False)
                    def signal_selection_overview(frame,columns):
                        rows=[]
                        positions=np.linspace(0,len(frame)-1,min(48,len(frame)),dtype=int) if len(frame) else []
                        for column in columns:
                            values=pd.to_numeric(frame[column],errors="coerce").to_numpy(dtype=float,na_value=np.nan)
                            finite=np.isfinite(values)
                            sampled=values[positions]
                            valid_preview=sampled[np.isfinite(sampled)]
                            low=float(valid_preview.min()) if len(valid_preview) else 0.
                            high=float(valid_preview.max()) if len(valid_preview) else 0.
                            commands=[]
                            connected=False
                            for i,value in enumerate(sampled):
                                if not np.isfinite(value):
                                    connected=False
                                    continue
                                x=4+192*i/max(1,len(sampled)-1)
                                y=16 if high==low else 28-24*((value/2-low/2)/(high/2-low/2))
                                commands.append(f'{"L" if connected else "M"}{x:.1f},{y:.1f}')
                                # A tiny horizontal mark also makes isolated samples visible.
                                commands.append(f'L{x+.6:.1f},{y:.1f}')
                                connected=True
                            svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="32" viewBox="0 0 200 32"><path d="{" ".join(commands)}" fill="none" stroke="#2563eb" stroke-width="1.6" stroke-linecap="round"/></svg>'
                            preview="data:image/svg+xml;base64,"+base64.b64encode(svg.encode()).decode()
                            rows.append({"Signal column":column,"Preview":preview,
                                         "Available (%)":round(100*finite.mean(),1) if len(values) else 0.,
                                         "Missing / non-finite":int((~finite).sum()),
                                         "Minimum":float(values[finite].min()) if finite.any() else None,
                                         "Maximum":float(values[finite].max()) if finite.any() else None})
                        return pd.DataFrame(rows)
                    overview=signal_selection_overview(raw_df,possible)
                    chosen=st.dataframe(overview,key="signal_entry_editor",hide_index=True,width="stretch",
                        height=min(460,38+35*len(possible)),on_select="rerun",selection_mode="multi-row",
                        column_config={
                            "Signal column":st.column_config.TextColumn(width="medium"),
                            "Preview":st.column_config.ImageColumn("Signal preview",help="Up to 48 evenly spaced raw samples in row order. Each column has its own vertical scale; gaps remain missing.",width="medium"),
                            "Available (%)":st.column_config.ProgressColumn("Available",min_value=0,max_value=100,format="%.1f%%",help="Finite numeric values in the input."),
                            "Minimum":st.column_config.NumberColumn(format="%.4g"),
                            "Maximum":st.column_config.NumberColumn(format="%.4g"),
                        })
                    st.caption("Previews use raw values and an individual scale per signal. Availability excludes missing and infinite values.")
                    picked=overview.iloc[chosen.selection.rows]["Signal column"].tolist()
                    st.caption(f"{len(picked)} of {len(possible)} signals selected")
                    if st.button("Analyse selected signals →",type="primary",disabled=not picked):
                        st.session_state["pending_signal_columns"]=picked
                        st.rerun()
                if duplicate_groups:
                    st.info("Some channels contain identical values. Choose one representative per group: "+dup_text)
                with st.expander("Preview original data · first 10 rows"):
                    st.dataframe(raw_df.head(10),hide_index=True,width="stretch")
                st.caption("Need a different time axis or additional numeric columns? Open Data settings in the sidebar.")
        st.stop()
    selected_dup=[g for g in duplicate_groups if sum(c in selected_signals for c in g)>1]
    if selected_dup:
        st.warning("Selected signals include exact duplicates. This can inflate cross-signal consistency; keep both only if you intentionally want both channels in the analysis.")

    inferred_domain,domain_scores=infer_domain(raw_df,selected_signals)  # noqa: F405
    domain_option=st.selectbox("Domain profile",["Auto-detect"]+list(DOMAIN_LABELS.values()))  # noqa: F405
    domain=inferred_domain if domain_option=="Auto-detect" else LABEL_TO_DOMAIN[domain_option]
    st.caption(f"Active: {DOMAIN_LABELS[domain]}")  # noqa: F405
    validity_mode="Raw numeric values"
    validity_meta={"mode":validity_mode,"affected":[]}

max_plot_points=5000

analysis_df=raw_df.copy()
with st.sidebar:
    validity_options = {x["key"]: x for x in data_validity_options(raw_df)}
    if "missing_flags" in validity_options or "validity_flags" in validity_options:
        st.subheader("Validity interpretation")
        st.caption("Explicit file flags can mark unavailable signal values. Raw values stay unchanged; only the analysis copy is masked.")
    missing_flag_enabled = (st.checkbox("Use missing = 1 as unavailable", value=True, key="missing_flag_mask")
                            if "missing_flags" in validity_options else False)
    eye_validity_enabled = (st.checkbox("Use eye validity flags", value=True, key="eye_validity_mask")
                            if "validity_flags" in validity_options else False)
    zero_pairs = paired_zero_gaze_signals(raw_df)
    if zero_pairs:
        st.caption("Paired zero gaze coordinates detected: " + "; ".join(
            f"{x}/{y}: {count:,}" for x,y,count in zero_pairs[:3]))
    paired_zero_enabled = (st.checkbox("Treat paired gaze (0,0) as missing", value=False,
                                   key="paired_gaze_zero_mask",
                                   help="Optional assumption; (0,0) can sometimes be valid screen coordinates. Raw values are never changed.")
                           if zero_pairs else False)
analysis_df, validity_meta = interpret_validity_flags(analysis_df,selected_signals,
    missing_flag=missing_flag_enabled, eye_validity=eye_validity_enabled)
if paired_zero_enabled:
    analysis_df, zero_changes = interpret_paired_zero_gaze(analysis_df,selected_signals)
    validity_meta["changes"].extend(zero_changes)
validity_meta["source_row_selection"] = import_selection_meta
validity_mode = "Explicit file validity flags / optional gaze-zero interpretation" if (missing_flag_enabled or eye_validity_enabled or paired_zero_enabled) else "Raw numeric values"
canonical=canonical_frame(analysis_df,selected_signals,fs,timestamp_col,timestamp_unit)  # noqa: F405
canonical.attrs["time_processing"]=time_processing
time_axis=canonical.attrs["time_axis"]
with st.sidebar:
    st.caption(f"Time axis: {timestamp_col}, converted from {time_axis['original_unit']} to relative seconds.")
    if time_axis.get("format")=="date_time":
        st.caption("Date/time offsets are normalized to UTC. Dates without a timezone are assumed UTC; use ISO dates.")
domain_cfg=DOMAIN_CONFIG[domain]  # noqa: F405
profile_agg=aggregate_profile(canonical,fs)  # noqa: F405
profile_features=profile_table(canonical,fs,domain_cfg["quality_weights"])  # noqa: F405
q_index=quality_index(profile_agg,domain_cfg["quality_weights"])  # noqa: F405
recs={s:recommend_stage(s,profile_agg,domain) for s in ["imputation","outlier","smoothing"]}  # noqa: F405
recs["scaling"]=recommend_scaling(profile_agg,domain)  # noqa: F405

# One final selection is shared by Pipeline lab, validation and export.
# Keep its state separate from the exploratory comparison configuration.
ai_goal=st.session_state.get("_ai_goal",DEFAULT_USER_GOAL)
ai_custom_goal=st.session_state.get("_ai_custom_goal","")
ai_context,analysis_context_key=build_context(canonical,profile_agg,profile_features,domain,fs,validity_mode,timestamp_choice,user_goal=ai_goal,custom_goal=ai_custom_goal)
ai_context_key=analysis_context_key
if st.session_state.get("_ai_input_mode","diagnostics")=="diagnostics_data":
    ai_context,ai_context_key=build_context(canonical,profile_agg,profile_features,domain,fs,validity_mode,timestamp_choice,"diagnostics_data",user_goal=ai_goal,custom_goal=ai_custom_goal)
if "final_pipeline_source" not in st.session_state:
    st.session_state["final_pipeline_source"]="Heuristic" if st.session_state.get("final_pipeline_recommended",False) else "Custom"
ai_results=matching_suggestions(st.session_state,analysis_context_key)
ai_available=bool(ai_results)
final_ai_groups={label:{key:result for key,result in ai_results.items()
    if result.get("input_mode","diagnostics_data" if "signal_data" in result["context"] else "diagnostics")==mode}
    for label,mode in AI_PIPELINE_SOURCES.items()}
source_options=["Custom","Heuristic"]+[label for label,results in final_ai_groups.items() if results]
# Migrate sessions that still use the previous combined AI source.
if st.session_state["final_pipeline_source"]=="AI-based":
    previous=ai_results.get(st.session_state.get("final_ai_result"))
    if previous is None and ai_results:
        previous=next(reversed(ai_results.values()))
    previous_mode=previous.get("input_mode","diagnostics_data" if "signal_data" in previous["context"] else "diagnostics") if previous else None
    st.session_state["final_pipeline_source"]=next((label for label,mode in AI_PIPELINE_SOURCES.items() if mode==previous_mode),"AI-based")
ai_selection_stale=st.session_state["final_pipeline_source"] not in source_options
if ai_selection_stale:
    st.session_state["final_pipeline_source"]="Custom"
    st.session_state["ai_selection_stale"]=True
final_ai_candidates=final_ai_groups.get(st.session_state["final_pipeline_source"],{})
if final_ai_candidates:
    selected_ai=st.session_state.get("final_ai_result")
    st.session_state["final_ai_result"]=selected_ai if selected_ai in final_ai_candidates else next(reversed(final_ai_candidates))
elif "final_ai_result" in st.session_state:
    # Preserve the last selection while the saved-suggestion widget is hidden.
    st.session_state["final_ai_result"]=st.session_state["final_ai_result"]
ai_result=final_ai_candidates.get(st.session_state.get("final_ai_result"))
if ai_result:
    st.session_state["final_pipeline_context"]=ai_result["context_key"]
    st.session_state.pop("ai_selection_stale",None)
elif st.session_state["final_pipeline_source"]=="Heuristic":
    st.session_state.pop("ai_selection_stale",None)
apply_pipeline_source(st.session_state,st.session_state["final_pipeline_source"],recs,ai_result)
for final_stage,rec in recs.items():
    key=f"final_pipeline_{final_stage}"
    st.session_state[key]=st.session_state.get(key,rec.method)
st.session_state["final_pipeline_order"]=st.session_state.get("final_pipeline_order",PIPELINE_ORDERS["outlier_imputation_scaling_smoothing"])
def clear_final_pipeline_recommendation():
    st.session_state["final_pipeline_source"]="Custom"
    st.session_state.pop("final_pipeline_context",None)
    st.session_state.pop("ai_selection_stale",None)
with st.sidebar:
    st.subheader("Preprocessing method guide")
    st.button("Explore preprocessing methods →",key="open_method_guide",width="stretch",
        on_click=st.session_state.update,kwargs={"method_guide_open":True,"signal_calculation_guide_open":False,"quality_details_open":False})
active_spec=PipelineSpec(**{stage:st.session_state[f"final_pipeline_{stage}"] for stage in recs})  # noqa: F405
stage_order=st.session_state["final_pipeline_order"]
order_key=next(key for key,label in PIPELINE_ORDERS.items() if label==stage_order)  # noqa: F405
pipeline_key=json.dumps({"context":analysis_context_key,"spec":asdict(active_spec),"order":order_key},sort_keys=True)

guide_open=st.session_state.get("method_guide_open",False) or st.session_state.get("signal_calculation_guide_open",False)
if st.session_state.get("quality_details_open", False) and not guide_open:
    render_quality_details(
        source_name=source_name, domain_label=DOMAIN_LABELS[domain], validity_mode=validity_mode,
        timestamp_choice=timestamp_choice, fs=fs, canonical=canonical,
        profile_features=profile_features, profile_agg=profile_agg,
        weights=domain_cfg["quality_weights"],
    )
    st.stop()

# Keep workflow widgets mounted while the reference page is visible so that
# navigation does not discard method selections, uploaded references or results.
if st.session_state.get("method_guide_open",False):
    method_guide()
elif st.session_state.get("signal_calculation_guide_open",False):
    render_signal_calculation_guide()
if guide_open:
    st.markdown('<style>.st-key-workflow_content {display:none;}</style>',unsafe_allow_html=True)
with st.container(key="workflow_content"):
    # ---------------------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------------------
    WORKFLOW_TABS=["1 · Data overview","2 · Signal profile","3 · Recommendation","4 · Method comparison","5 · Pipeline lab","6 · Reference validation","7 · Export"]

    def switch_workflow_tab(label):
        st.session_state["workflow_tabs"]=label

    def continue_to_method_comparison(selection):
        st.session_state["comparison_mode"]="Isolated stage"
        st.session_state["pending_comparison_pipeline"]=selection
        st.session_state["comparison_pipeline_source"]=selection["source"]
        st.session_state["comparison_order"]=PIPELINE_ORDERS[selection["order"]]
        if selection.get("ai_result_id"):
            st.session_state["comparison_ai_result"]=selection["ai_result_id"]
        for pipeline_stage,method in selection["pipeline"].items():
            st.session_state[f"comparison_context_{pipeline_stage}"]=method
        selected_stage=st.session_state.get("compare_stage","imputation")
        st.session_state["comparison_methods"]=[selection["pipeline"][selected_stage]]
        st.session_state["workflow_tabs"]="4 · Method comparison"

    # Preserve the selected page in sessions opened before the navigation rename.
    renamed_tabs={"1 · Data & domain":"1 · Data overview",
                  "4 · Step compare":"4 · Method comparison",
                  "6 · Ground truth":"6 · Reference validation"}
    previous_tab=st.session_state.get("workflow_tabs")
    if previous_tab in renamed_tabs:
        st.session_state["workflow_tabs"]=renamed_tabs[previous_tab]

    tabs=st.tabs(WORKFLOW_TABS,key="workflow_tabs",on_change="rerun")

    with tabs[0]:
        st.markdown('<h2 class="overview-heading">Data quality overview</h2><p class="overview-intro">Review your dataset, inspect its quality scores, and see how domain weights shape the overall index.</p>',unsafe_allow_html=True)
        with st.container(key="overview_summary"):
            dataset_summary,quality_summary=st.columns([1.7,1],gap="medium")
        with dataset_summary:
            with st.container(border=True,key="data_summary_panel",height="stretch"):
                st.markdown('<div class="panel-kicker">Dataset overview</div>',unsafe_allow_html=True)
                st.markdown(
                    f'<dl class="dataset-stat-grid"><div class="dataset-stat"><dt>Rows</dt><dd>{len(raw_df):,}</dd></div>'
                    f'<div class="dataset-stat"><dt>Columns</dt><dd>{raw_df.shape[1]}</dd></div>'
                    f'<div class="dataset-stat"><dt>Analysed signals</dt><dd>{len(selected_signals)}</dd></div>'
                    f'<div class="dataset-stat"><dt>Sampling rate</dt><dd>{fs:.4g} Hz</dd></div></dl>'
                    f'<div class="dataset-domain"><div class="dataset-domain-title"><span>Active Domain:</span><strong>{escape(DOMAIN_LABELS[domain])}</strong></div>'
                    f'<p>{escape(domain_cfg["description"])}</p></div>',unsafe_allow_html=True,
                )
        with quality_summary:
            with st.container(border=True,key="current_quality_score",height="stretch"):
                st.markdown('<div class="panel-kicker">Quality summary</div>',unsafe_allow_html=True)
                st.markdown('<p class="quality-summary-label">Intrinsic quality index</p>',unsafe_allow_html=True)
                with st.container(key="score_value_link"):
                    score_value,score_link=st.columns([1.3,1],gap="small",vertical_alignment="center")
                    score_value.markdown(f'<div class="quality-summary-number">{q_index:.1f}<span> / 100</span></div>',unsafe_allow_html=True)
                    score_link.button("Understand this score", on_click=st.session_state.update,
                                      kwargs={"quality_details_open": True}, key="open_quality_details",type="tertiary")
                st.markdown('<p class="quality-index-explanation">Combines the scores for completeness, continuity, outlier burden, distribution shape, temporal stability, and cross-signal consistency using domain weights.</p>',unsafe_allow_html=True)
        with st.container(key="overview_charts"):
            left,right=st.columns(2,gap="medium")
        with left:
            with st.container(border=True,key="quality_components_panel",height="stretch"):
                with st.container(key="quality_panel_header"):
                    st.markdown('<div class="panel-kicker">Measured quality</div><h3 class="chart-heading">Quality index components</h3><p class="chart-description">Six component scores, each on a 0–100 scale.</p>',unsafe_allow_html=True)
                quality_view=st.session_state.get("quality_component_chart_type","Bar chart")
                if quality_view=="Index calculation":
                    render_quality_calculation(profile_agg,domain_cfg["quality_weights"],key="aggregate_quality_calculation")
                else:
                    st.plotly_chart(quality_component_chart(profile_agg,domain_cfg["quality_weights"]),width="stretch",config={"displayModeBar":False})
                st.segmented_control("Chart type",["Bar chart","Index calculation"],default="Bar chart",
                                     key="quality_component_chart_type",label_visibility="collapsed",
                                     on_change=lambda:None)
        with right:
            with st.container(border=True,key="domain_weights_panel",height="stretch"):
                with st.container(key="weights_panel_header"):
                    st.markdown('<div class="panel-kicker">Domain perspective</div><h3 class="chart-heading">Domain weights</h3><p class="chart-description">How strongly each component influences the index.</p>',unsafe_allow_html=True)
                st.plotly_chart(domain_weight_chart(domain_cfg["quality_weights"]),width="stretch",config={"displayModeBar":False})
        with st.container(border=True,key="data_preview_panel"):
            preview_title,preview_meta=st.columns([2,1],vertical_alignment="center")
            preview_title.subheader("Data preview")
            preview_meta.markdown(f'<p class="preview-meta">{min(10,len(raw_df))} input rows · {len(selected_signals)} selected signals</p>',unsafe_allow_html=True)
            preview_columns=list(dict.fromkeys(([timestamp_col] if timestamp_col is not None else [])+selected_signals))
            st.caption("Selected signals and time axis, as recorded in your original file.")
            st.dataframe(raw_df[preview_columns].head(10),width="stretch",height=280,hide_index=True)
            with st.expander("Data preview · All columns"):
                st.caption("First 10 input rows across all columns. Scroll horizontally to explore the complete table.")
                st.dataframe(raw_df.head(10),width="stretch",height=280,hide_index=True)
            if duplicate_groups:
                with st.expander("Exact duplicate-signal check"):
                    st.write("CIRRUS++ checks numeric channels for exact equality. Counting the same numeric channel twice can inflate cross-signal consistency; choose one representative unless you intentionally need both channels.")
                    st.dataframe(pd.DataFrame({"exact_duplicate_group":[" = ".join(g) for g in duplicate_groups]}),hide_index=True,width="stretch")
        with st.expander("How was the domain detected?"):
            if domain_option=="Auto-detect":
                st.write("Auto-detection is intentionally simple and auditable: CIRRUS++ scores domain-related keywords in column names (for example gaze/pupil, temperature/pressure, price/return). It is not a learned domain classifier.")
                st.json(domain_scores)
            else: st.write("You manually selected the active domain profile; the automatic suggestion is not used.")

    with tabs[1]:
        render_signal_profile(
            canonical=canonical, profile_features=profile_features,
            selected_signals=selected_signals, weights=domain_cfg["quality_weights"],
            max_plot_points=max_plot_points,
        )

    with tabs[2]:
        render_recommendations(
            recs=recs, context=ai_context, context_key=ai_context_key,
            method_label=ml, heuristic_card=recommendation_card, method_guide=method_guide,
        )
        comparison_handoff={"source":"Current pipeline","pipeline":active_spec.to_dict(),"order":order_key}
        if st.session_state.get("recommendation_approach")=="Heuristic":
            comparison_handoff={"source":"Heuristic","pipeline":{s:r.method for s,r in recs.items()},
                                "order":"outlier_imputation_scaling_smoothing"}
        else:
            shown_ai=ai_results.get(ai_context_key+":"+st.session_state.get("ai_model",next(iter(MODELS))))
            if shown_ai:
                shown_mode=shown_ai.get("input_mode","diagnostics_data" if "signal_data" in shown_ai["context"] else "diagnostics")
                comparison_handoff={
                    "source":next(label for label,mode in AI_PIPELINE_SOURCES.items() if mode==shown_mode),
                    "pipeline":{stage:shown_ai["suggestion"][stage] for stage in recs},
                    "order":shown_ai["suggestion"]["order"],
                    "ai_result_id":ai_context_key+":"+shown_ai["model"],
                }

    with tabs[3]:
        st.subheader("Method comparison")
        st.caption("Compare preprocessing methods on their own or within your current pipeline to see how they change your signals.")
        with st.container(border=True,key="comparison_setup"):
            mode_labels={"Isolated stage":"Selected step only","Pipeline context":"Within current pipeline"}
            mode=st.radio("Comparison mode",list(mode_labels),horizontal=True,format_func=mode_labels.get,key="comparison_mode",
                help="Compare a step on its own, or replace only that step within the pipeline selected in Pipeline lab.")
            step_labels={"outlier":"Outlier treatment","imputation":"Missing value imputation","scaling":"Normalization","smoothing":"Smoothing"}
            stage=st.selectbox(
                "Stage",["outlier","imputation","scaling","smoothing"],index=1,key="compare_stage",format_func=step_labels.get)
            comparison_pools={"imputation":IMP_METHODS,"outlier":OUT_METHODS,"smoothing":SMOOTH_METHODS,"scaling":SCALE_METHODS}
            pool=comparison_pools[stage]
            st.session_state["comparison_methods"]=[method for method in st.session_state.get("comparison_methods",[]) if method in pool]
            methods=st.multiselect("Methods",pool,key="comparison_methods",format_func=ml,placeholder="Choose methods to compare",
                help="Include None to compare with skipping this step. All selected signals are processed.")
            comparison_base=PipelineSpec()  # noqa: F405
            comparison_order_key=order_key
            if mode=="Pipeline context":
                # Hidden comparison widgets may be cleaned up by Streamlit.
                # Restore the transferred pipeline once when context is opened.
                pending=st.session_state.pop("pending_comparison_pipeline",None)
                if pending:
                    source=pending["source"]
                    if source in AI_PIPELINE_SOURCES and pending.get("ai_result_id") not in ai_results:
                        source="Custom"
                    st.session_state["comparison_pipeline_source"]=source
                    st.session_state["comparison_order"]=PIPELINE_ORDERS[pending["order"]]
                    if pending.get("ai_result_id"):
                        st.session_state["comparison_ai_result"]=pending["ai_result_id"]
                    for pipeline_stage,method in pending["pipeline"].items():
                        st.session_state[f"comparison_context_{pipeline_stage}"]=method
                comparison_flow=st.container()
                with st.container(key="comparison_pipeline_panel"):
                    with st.expander("Adjust pipeline for this comparison",expanded=False):
                        st.caption("Choose the pipeline for the fixed steps. The selected comparison step still varies between your chosen methods.")
                        def clear_comparison_recommendation():
                            st.session_state["comparison_pipeline_source"]="Custom"
                        comparison_ai_modes={
                            "AI suggestion (Diagnostics only)":"diagnostics",
                            "AI suggestion (Diagnostics + data)":"diagnostics_data",
                        }
                        mode_results={label:{key:result for key,result in ai_results.items()
                            if result.get("input_mode","diagnostics_data" if "signal_data" in result["context"] else "diagnostics")==input_mode}
                            for label,input_mode in comparison_ai_modes.items()}
                        comparison_sources=["Current pipeline","Custom","Heuristic"]+[label for label,results in mode_results.items() if results]
                        previous_source=st.session_state.get("comparison_pipeline_source","Current pipeline")
                        stale_comparison_source=previous_source not in comparison_sources
                        st.session_state["comparison_pipeline_source"]="Custom" if stale_comparison_source else previous_source
                        comparison_source=st.selectbox("Comparison pipeline source",comparison_sources,key="comparison_pipeline_source",
                            help="Current pipeline follows your selection in Pipeline lab. Custom lets you set the fixed steps. Heuristic uses rule-based recommendations. AI options use saved suggestions matching the current data and analysis settings; no API request is made here.")
                        if stale_comparison_source:
                            st.info("The saved AI pipeline no longer matches this analysis. Its settings are retained as Custom.")
                        comparison_suggestion=None
                        if comparison_source in comparison_ai_modes:
                            candidates=mode_results[comparison_source]
                            if st.session_state.get("comparison_ai_result") not in candidates:
                                st.session_state["comparison_ai_result"]=next(reversed(candidates))
                            chosen=st.selectbox("Saved comparison suggestion",list(candidates),key="comparison_ai_result",
                                format_func=lambda key:MODELS.get(candidates[key]["model"],candidates[key]["model"]))
                            comparison_suggestion=candidates[chosen]["suggestion"]
                        if comparison_source=="Current pipeline":
                            st.session_state["comparison_order"]=stage_order
                            for fixed_stage in recs:
                                st.session_state[f"comparison_context_{fixed_stage}"]=getattr(active_spec,fixed_stage)
                        if comparison_source=="Heuristic" or comparison_suggestion:
                            selected_order=comparison_suggestion["order"] if comparison_suggestion else "outlier_imputation_scaling_smoothing"
                            st.session_state["comparison_order"]=PIPELINE_ORDERS[selected_order]
                            for fixed_stage in recs:
                                st.session_state[f"comparison_context_{fixed_stage}"]=comparison_suggestion[fixed_stage] if comparison_suggestion else recs[fixed_stage].method
                        if st.session_state.get("comparison_order") not in PIPELINE_ORDERS.values():
                            st.session_state["comparison_order"]=stage_order
                        comparison_order=st.selectbox("Comparison execution order",list(PIPELINE_ORDERS.values()),
                            key="comparison_order",on_change=clear_comparison_recommendation)
                        comparison_order_key=next(key for key,label in PIPELINE_ORDERS.items() if label==comparison_order)
                        st.caption("Manual changes switch the source to Custom. These settings apply only to this comparison.")
                        comparison_choices={}
                        context_columns=st.columns(2,gap="large")
                        for position,context_stage in enumerate(comparison_order_key.split("_")):
                            with context_columns[position%2]:
                                if context_stage==stage:
                                    st.text_input(step_labels[context_stage],"Varies between selected methods",disabled=True,key=f"comparison_varies_{context_stage}")
                                    comparison_choices[context_stage]="none"
                                else:
                                    context_pool=comparison_pools[context_stage]
                                    st.session_state.setdefault(f"comparison_context_{context_stage}",getattr(active_spec,context_stage))
                                    comparison_choices[context_stage]=st.selectbox(step_labels[context_stage],context_pool,
                                        format_func=ml,key=f"comparison_context_{context_stage}",
                                        on_change=clear_comparison_recommendation)
                        comparison_base=PipelineSpec(**comparison_choices)  # noqa: F405
                flow_steps=[]
                for flow_stage in comparison_order_key.split("_"):
                    varies=flow_stage==stage
                    method_text=(" / ".join(ml(method) for method in methods) or "Choose methods") if varies else ml(getattr(comparison_base,flow_stage))
                    flow_steps.append(
                        f'<div class="comparison-flow-step{" comparison-flow-varies" if varies else ""}">'
                        f'<span>{escape(step_labels[flow_stage])}{" · varies" if varies else " · fixed"}</span>'
                        f'<strong>{escape(method_text)}</strong></div>')
                with comparison_flow:
                    st.markdown('<div class="comparison-flow">'+ '<span class="comparison-flow-arrow">→</span>'.join(flow_steps)+'</div>',unsafe_allow_html=True)
                    st.caption("Only the highlighted step varies. Other steps stay fixed · "+comparison_source)
        comparison_key=json.dumps({"context":pipeline_key,"spec":asdict(comparison_base),"order":comparison_order_key},sort_keys=True)
        comparison_result_key=(mode,stage,tuple(methods),comparison_key,"stage-metrics-2")
        with st.container(border=True,key="comparison_action"):
            action, guidance=st.columns([1,3],gap="medium",vertical_alignment="center")
            with action:
                run_comparison=st.button("Run comparison",type="primary",key="run_compare",disabled=not methods or (time_checks["irregular_intervals"] and not resample_smoothing and (any(m!="none" for m in methods) if stage=="smoothing" else comparison_base.smoothing!="none")),width="stretch")
            with guidance:
                if not methods:
                    st.markdown("**Select at least one method to continue.**")
                    st.caption("Choose two or more to compare alternatives; include None to skip this step.")
                else:
                    st.markdown(f"**Ready to compare · {len(methods)} {'method' if len(methods)==1 else 'methods'}**")
                    st.caption("Results appear below. Your final pipeline stays unchanged.")
        if time_checks["irregular_intervals"] and not resample_smoothing:
            st.caption("For smoothing comparisons, enable Resample internally for smoothing in Time processing, or compare methods without smoothing.")
        if run_comparison and methods:
            base=comparison_base
            # Plot the actual final output in original units without rerunning a
            # different pipeline with normalization disabled.
            visual_outs={}; comparison_parameters={}
            imputation_details={} if stage=="imputation" else None
            with st.spinner("Comparing methods..."):
                outs,cmp=compare_stage_methods(canonical,stage,methods,fs,domain_cfg["quality_weights"],base,comparison_order_key,imputation_details=imputation_details,visual_outputs=visual_outs,execution_parameters=comparison_parameters)  # noqa: F405
            st.session_state["compare_result"]={"key":comparison_result_key,"outputs":outs,"visual_outputs":visual_outs,"table":cmp,"imputation_details":imputation_details,
                "context":{"mode":mode,"stage":stage,"methods":methods,
                           "pipeline":comparison_base.to_dict(),"order":comparison_order_key,
                           "seed":42,"method_parameters":comparison_parameters}}
        cr=st.session_state.get("compare_result")
        if methods and cr and cr["key"]!=comparison_result_key:
            st.caption("Comparison settings have changed. Run comparison again to see updated results.")
        if cr and cr["key"]==comparison_result_key:
            feature=st.selectbox("Signal",selected_signals,key="compare_feature",
                help="Changes the preview only. Results summarize all selected signals.")
            visual_outs=cr.get("visual_outputs",cr["outputs"])
            if stage!="scaling":
                feature_visual_outs={k:v[feature] for k,v in visual_outs.items()}
                imputation_input=None
                imputed_only=False
                if stage=="imputation":
                    if cr.get("imputation_details"):
                        view=st.radio("Signal view",["Imputed values only","Full processed signal"],horizontal=True,key="comparison_imputation_view")
                        imputed_only=view=="Imputed values only"
                        if imputed_only:
                            details=cr["imputation_details"]
                            imputation_input=next(iter(details.values()))["input"][feature]
                            feature_visual_outs={k:v["output"][feature] for k,v in details.items()}
                    else:
                        st.info("Run comparison again to enable the imputed-values-only view.")
                plot_title=f"{mode_labels[mode]} · {step_labels[stage]} · {feature}"
                if imputed_only:
                    st.caption("Colored method traces show only values filled by the imputation step, with adjacent observed samples as anchors. Later processing is excluded from this view; the metrics still describe the complete compared pipeline.")
                    if not imputation_input.isna().any():
                        st.info("No missing values reach the imputation step for this feature, so there are no imputed segments to display.")
                    fig=imputation_plot(canonical[feature],feature_visual_outs,imputation_input,canonical.timestamp,max_plot_points,plot_title+" · imputed values only")
                else:
                    st.caption("The actual pipeline output is converted back to original units for this plot so it can be compared with the raw signal. Normalization still runs before optional methods; the table describes the configured final output.")
                    series={"Raw":canonical[feature]}; series.update({ml(k):v for k,v in feature_visual_outs.items()})
                    fig=line_plot(series,canonical.timestamp,max_plot_points,plot_title+" · full signal")
                st.plotly_chart(fig,width="stretch")
            else:
                st.warning("Scaling methods change numerical units. A raw-vs-scaled overlay is therefore not a like-for-like visual comparison; use the method table and inspect the scaled values separately.")
                series={ml(k):v[feature] for k,v in visual_outs.items()}; st.plotly_chart(line_plot(series,canonical.timestamp,max_plot_points,f"Scaling outputs · {feature}"),width="stretch")
            st.subheader("Comparison results")
            st.caption("Step effects are measured at the selected step across all selected signals. Remaining missing values and quality scores describe the final output; the plot above shows one feature.")
            table=cr["table"].copy(); table["method"]=table.method.map(ml)
            table["cleaning_changed_%"]=100*table["cleaning_changed_fraction"]; table["overall_changed_%"]=100*table["overall_changed_fraction"]
            for metric in ("imputed_fraction","outlier_affected_fraction","remaining_missing_fraction","stage_changed_fraction"):
                if metric in table:
                    table[metric]*=100
            main_columns={
                "imputation":["method","quality_delta","remaining_missing_fraction","imputed_fraction"],
                "outlier":["method","quality_delta","outlier_affected_fraction","remaining_missing_fraction"],
                "smoothing":["method","quality_delta","volatility_delta","stage_changed_fraction"],
                "scaling":["method","scaled_min_median","scaled_max_median","scaled_std_median"],
            }
            column_config={
                "imputed_fraction":st.column_config.NumberColumn("Values filled (%)",format="%.2f",help="Missing cells filled at the imputation step, as a percentage of all selected signal cells. Includes gaps created by earlier outlier treatment."),
                "outlier_affected_fraction":st.column_config.NumberColumn("Flagged / clipped (%)",format="%.2f",help="Observed cells flagged by the selected detector, or actually clipped by winsorization, at the outlier step. Percentage of all selected signal cells, not rows."),
                "remaining_missing_fraction":st.column_config.NumberColumn("Remaining gaps (%)",format="%.2f",help="Missing cells in the final output as a percentage of all selected signal cells; not a median across signals."),
                "stage_changed_fraction":st.column_config.NumberColumn("Changed by this step (%)",format="%.2f",help="Percentage of all selected signal cells changed by this step alone."),
                "volatility_before":st.column_config.NumberColumn("Volatility before filter",format="%.3f"),
                "volatility_after":st.column_config.NumberColumn("Volatility after filter",format="%.3f"),
                "volatility_delta":st.column_config.NumberColumn("Δ Local volatility",format="%+.3f",help="Median local volatility after minus before smoothing. Negative means smoother; it does not establish greater accuracy."),
                "scaled_min_median":st.column_config.NumberColumn("Median signal minimum",format="%.3f",help="Median of the per-signal minima immediately after normalization, before optional smoothing."),
                "scaled_max_median":st.column_config.NumberColumn("Median signal maximum",format="%.3f",help="Median of the per-signal maxima immediately after normalization, before optional smoothing."),
                "scaled_std_median":st.column_config.NumberColumn("Median signal SD",format="%.3f",help="Median per-signal population standard deviation immediately after normalization. Not a pooled SD across different signals."),
                "method":st.column_config.TextColumn("Method"),
                "quality_index":st.column_config.NumberColumn("Quality index",format="%.2f",help="Domain-weighted intrinsic diagnostic score, 0–100; not reconstruction accuracy."),
                "quality_delta":st.column_config.NumberColumn("Δ Quality index",format="%+.2f",help="Change from the unprocessed baseline, in score points. In pipeline context this includes all configured steps."),
                "missing_rate":st.column_config.NumberColumn("Missing values (%)",format="%.2f",help="Median remaining missing-value percentage across selected signals. Zero does not guarantee every signal is complete."),
                "outlier_rate":st.column_config.NumberColumn("Outlier rate (%)",format="%.2f",help="Larger of the median IQR- and MAD-based statistical outlier rates. Lower is not automatically better."),
                "local_volatility":st.column_config.NumberColumn("Local volatility",format="%.3f",help="Median normalized first-difference variability across selected signals. Lower means smoother, not necessarily more accurate."),
                "cleaning_changed_%":st.column_config.NumberColumn("Changed by cleaning (%)",format="%.2f",help="Percentage of all selected values changed by imputation, outlier handling and smoothing, excluding scaling."),
                "overall_changed_%":st.column_config.NumberColumn("Changed overall (%)",format="%.2f",help="Percentage of all selected values numerically changed, including scaling."),
            }
            display_table=table.copy()
            display_table[["missing_rate","outlier_rate"]]*=100
            if stage=="scaling":
                display_table=display_table.sort_values("method")
                st.caption("Range and standard deviation are summarized per signal immediately after normalization, before optional smoothing. Choose scaling to match your analysis objective.")
            st.dataframe(display_table[main_columns[stage]],width="stretch",hide_index=True,column_config=column_config)
            with st.expander("More diagnostics"):
                st.caption("Full-output diagnostics and additional step measurements. Changed overall includes normalization and may therefore approach 100%.")
                st.dataframe(display_table,width="stretch",hide_index=True,column_config=column_config)
            with st.expander("How to read the comparison table"):
                guides={
                    "imputation":"**Values filled (%)** counts missing cells filled at the imputation step, including gaps created by earlier outlier treatment. **Remaining gaps (%)** measures cells still missing after the complete run.",
                    "outlier":"**Flagged / clipped (%)** counts observed cells marked by the detector or changed by winsorization. **Remaining gaps (%)** shows whether gaps remain after later steps. Counts refer to signal cells, not rows; Isolation Forest can mark all channels in a row.",
                    "smoothing":"**Δ Local volatility** is the median volatility after filtering minus before filtering. Negative values mean smoother signals. **Changed by this step (%)** counts values changed by the filter alone.",
                    "scaling":"**Median signal minimum / maximum / SD** summarize the per-signal ranges and population standard deviations immediately after normalization, before smoothing. Individual signals may differ from these medians. Judge scaling by your analysis objective.",
                }
                st.markdown(guides[stage])
                if stage!="scaling":
                    st.markdown("**Δ Quality index** is the change from the unprocessed baseline in score points, not percent. In pipeline mode it includes the complete configured pipeline. A higher score does not establish reconstruction accuracy.")
                with st.expander("What do the additional metrics mean?"):
                    st.markdown("""
    - **Quality index (0–100):** a summary of the measured data quality, weighted for your selected domain. A higher score does not guarantee a more accurate signal.
    - **Missing values (%):** calculate the missing percentage for each signal, then take the middle value. **Remaining gaps (%)** instead counts missing values across all selected signals together.
    - **Outlier rate (%):** the percentage of values that still look statistically unusual after processing. Uses the larger of two estimates (IQR and MAD); unusual values can also be real events.
    - **Local volatility:** how strongly values fluctuate from one sample to the next, relative to the signal's overall variation. Lower values indicate a smoother signal.
    - **Changed by cleaning (%):** how many values differ from the original after filling gaps, treating outliers and smoothing, with normalization removed from the comparison.
    - **Changed overall (%):** how many values differ from the original including normalization. This can reach 100% simply because normalization changes the numerical scale.
    """)
            st.markdown("**Ready to build your pipeline?** In **5 · Pipeline lab**, combine your preferred methods, review their execution order, and run the complete workflow.")

    with tabs[4]:
        st.subheader("Pipeline lab")
        st.caption("Choose your final pipeline after comparing methods. Processing starts only when you click Run final pipeline.")
        st.subheader("Choose final pipeline")
        pipeline_labels={"outlier":"Outlier treatment","imputation":"Missing value imputation","scaling":"Normalization","smoothing":"Optional methods (smoothing)"}
        final_pools={"outlier":OUT_METHODS,"imputation":IMP_METHODS,"scaling":SCALE_METHODS,"smoothing":SMOOTH_METHODS}
        with st.container(border=True,key="final_pipeline_settings"):
            st.selectbox("Pipeline source",source_options,key="final_pipeline_source",
                help="Custom lets you choose methods and execution order. Heuristic and AI sources supply the order. Manual method changes switch to Custom. Nothing runs automatically.")
            if st.session_state["final_pipeline_source"]=="Heuristic":
                st.warning("Heuristic pipeline selected. These rules have limited validation and may not generalize to your dataset. Review the methods and compare their effects on your signals before running the pipeline.")
            elif st.session_state["final_pipeline_source"] not in AI_PIPELINE_SOURCES:
                st.warning("No AI-suggested pipeline is selected. You are using a custom pipeline; review the methods and their execution order before running it.")
            if st.session_state["final_pipeline_source"]=="Custom":
                st.selectbox("Final execution order",list(PIPELINE_ORDERS.values()),key="final_pipeline_order",on_change=clear_final_pipeline_recommendation)
            if final_ai_candidates:
                st.selectbox("Saved AI suggestion",list(final_ai_candidates),key="final_ai_result",
                    format_func=lambda key:MODELS.get(final_ai_candidates[key]["model"],final_ai_candidates[key]["model"]))
            if not ai_available:
                st.caption("Generate an AI pipeline on page 3 to enable AI selection.")
            if st.session_state.get("ai_selection_stale"):
                st.warning("Your AI suggestion belongs to a different analysis context. The methods are retained as Custom; generate a new suggestion on page 3.")
            st.caption(f"Selected pipeline: {st.session_state['final_pipeline_source']}")
            final_columns=st.columns(2,gap="large")
            for position,final_stage in enumerate(order_key.split("_")):
                with final_columns[position%2]:
                    st.selectbox(pipeline_labels[final_stage],final_pools[final_stage],format_func=ml,
                        key=f"final_pipeline_{final_stage}",on_change=clear_final_pipeline_recommendation)
            st.caption("This selection is used for the final run, validation and export. Your comparison settings stay separate.")
        with st.expander("What happens when I click Run final pipeline?"):
            st.markdown("""
    CIRRUS++ applies outlier treatment and missing-value imputation in the selected order, followed by normalization and optional smoothing. By default, outlier treatment comes first. Outlier detectors (Z-score/IQR/MAD/Isolation Forest) mark detected values as NaN; the selected imputation method then fills these and existing gaps. Winsorization clips the 1st/99th percentiles instead. Without a subsequent imputation step, marked values remain missing; optional filters do not silently fill them.

    The app reports **cleaning changes separately from scaling transformations**. This matters because a scaler can numerically transform 100% of values without meaning that 100% of observations were repaired.
    """)
            st.caption("Linear interpolation uses actual time intervals. All imputation methods leave leading/trailing gaps missing and respect the optional maximum gap in Time processing. Smoothing preserves unfilled gaps and splits at acquisition interruptions exceeding the enabled gap limit. Moving median and Savitzky–Golay use up to 9 and 21 samples on the processing grid.")
        st.markdown(" ".join(f'<span class="pill">{pipeline_labels[stage]}: {ml(getattr(active_spec,stage))}</span>' for stage in order_key.split("_")),unsafe_allow_html=True)
        st.caption("Review your final selection above before running. Selected methods and execution order are recorded in the export.")
        if active_spec.outlier not in {"none","winsorize"} and (active_spec.imputation=="none" or order_key.startswith("imputation")):
            st.caption("Detected outliers will remain NaN because this configuration has no imputation after outlier treatment.")
        smoothing_blocked=time_checks["irregular_intervals"] and not resample_smoothing and active_spec.smoothing!="none"
        if smoothing_blocked:
            st.warning("Enable Resample internally for smoothing in Time processing, or choose None for smoothing.")
        if st.button("Run final pipeline",type="primary",key="run_final_pipeline",width="stretch",disabled=smoothing_blocked):
            start=time.perf_counter()
            with st.spinner("Executing pipeline..."):
                processed,meta=execute_pipeline(canonical,active_spec,fs,order_key,capture_physical_output=True)  # noqa: F405
                cleaned=meta.pop("physical_output")
            st.session_state["pipeline"]={"key":pipeline_key,"processed":processed,"cleaned":cleaned,"meta":meta,"elapsed":time.perf_counter()-start,
                "source":st.session_state["final_pipeline_source"],
                "executed_at_utc":datetime.now(timezone.utc).isoformat(),
                "ai_recommendation":ai_result if st.session_state["final_pipeline_source"] in AI_PIPELINE_SOURCES else None}
        pr=st.session_state.get("pipeline")
        if pr is None or pr["key"]!=pipeline_key:
            st.warning("The current final pipeline has not been run yet. Click **Run final pipeline** above.")
        else:
            st.caption(f"Last run source: {pr.get('source','Custom')}")
            processed=pr["processed"]; cleaned=pr["cleaned"]; meta=pr["meta"]; after_q=quality_index(aggregate_profile(cleaned,fs),domain_cfg["quality_weights"])  # noqa: F405
            p=st.columns(4); p[0].metric("Before intrinsic quality",f"{q_index:.1f}/100"); p[1].metric("After cleaning",f"{after_q:.1f}/100",f"{after_q-q_index:+.1f}"); p[2].metric("Cleaning changes",f"{meta['cleaning_changed_fraction']:.2%}"); p[3].metric("Scaling transforms",f"{meta['scaling_changed_fraction']:.2%}")
            stages=pd.DataFrame([{"stage":k.title(),"cells changed vs previous stage":v} for k,v in meta["stage_changed_fraction"].items()]); st.dataframe(stages.style.format({"cells changed vs previous stage":"{:.2%}"}),hide_index=True,width="stretch")
            feat=st.selectbox("Preview feature",selected_signals,key="pipeline_feature"); st.plotly_chart(line_plot({"Raw":canonical[feat],"Processed (original units)":cleaned[feat]},canonical.timestamp,max_plot_points,f"Processing effect · {feat}"),width="stretch")
            if active_spec.scaling!="none": st.caption(f"The final output is converted back from {ml(active_spec.scaling)} to original units for this overlay and the cleaning diagnostics. The pipeline runs normalization before optional methods; the processed data export retains the normalized output.")
            change=profile_change_table(canonical,cleaned,fs,domain_cfg["quality_weights"])  # noqa: F405
            st.subheader("Metric changes by feature")
            before_after_columns=[column for column in change.columns if column=="feature" or column.startswith(("before_","after_"))]
            st.dataframe(change[before_after_columns].round(6),width="stretch",hide_index=True)
            with st.expander("What do the before/after metrics mean?"):
                metric_change_guide()

    with tabs[5]:
        render_ground_truth(
            canonical=canonical, selected_signals=selected_signals, source_name=source_name,
            fs=fs, domain_cfg=domain_cfg, recs=recs, active_spec=active_spec, order_key=order_key,
            analysis_context_key=analysis_context_key, separator=separator, decimal=decimal,
            load_upload=load_upload, longest_complete_run_length=longest_complete_run_length,
            method_label=ml, line_plot=line_plot, max_plot_points=max_plot_points,
        )

    with tabs[6]:
        export_comparison=st.session_state.get("compare_result")
        if not methods or not export_comparison or export_comparison.get("key")!=comparison_result_key:
            export_comparison=None
        elif not export_comparison.get("context"):
            # A matching result key proves these settings also apply to results
            # retained in sessions opened before export context was introduced.
            export_comparison={**export_comparison,"context":{
                "mode":mode,"stage":stage,"methods":methods,"feature":feature,
                "pipeline":comparison_base.to_dict(),"order":comparison_order_key}}
        bundle=build_exports(
            raw=raw_df,canonical=canonical,run=st.session_state.get("pipeline"),pipeline_key=pipeline_key,
            spec=active_spec,order_key=order_key,stage_order=stage_order,
            selected_source=st.session_state["final_pipeline_source"],
            context={"source":source_name,"source_mode":source_mode,"domain":domain,
                     "domain_label":DOMAIN_LABELS[domain],"sampling_rate_hz":fs,"signals":selected_signals,
                     "analysis_context_key":analysis_context_key,"data_fingerprint":ai_context["data_fingerprint"],
                     "timestamp_column":timestamp_col,"timestamp_unit_selection":timestamp_unit if timestamp_col else None,
                     "time_axis":time_axis,"time_processing":time_processing,"parser_note":parser_note,
                     "selected_ai_goal":ai_context["user_goal"],
                     "eye_tracking_validity_mode":validity_mode,"eye_tracking_validity_meta":validity_meta,
                     "exact_duplicate_signal_groups":duplicate_groups},
            weights=domain_cfg["quality_weights"],recommendations=recs,
            comparison=export_comparison,ai_suggestions=list(ai_results.values()),
            selected_ai_result=ai_result if st.session_state["final_pipeline_source"] in AI_PIPELINE_SOURCES else None,
        )
        render_export(bundle)

    for index,tab in enumerate(tabs):
        with tab:
            with st.container(key=f"workflow_footer_{index}"):
                if index==4:
                    st.markdown("**Next: validate reconstruction.** The before/after diagnostics show how your pipeline changes the signals. To assess how well preprocessing recovers a reference, continue to **6 · Reference validation** for an optional experiment with controlled corruption or an aligned clean reference.")
                previous,next_step=st.columns([2,1] if index==0 else [1,1],gap="large",vertical_alignment="center")
                if index==0:
                    previous.markdown("**Next: explore your signals.** Compare their quality and inspect individual signals in detail.")
                if index>0:
                    previous.button(f"← {WORKFLOW_TABS[index-1]}",key=f"workflow_previous_{index}",
                                    on_click=switch_workflow_tab,args=(WORKFLOW_TABS[index-1],),width="stretch")
                if index<len(WORKFLOW_TABS)-1:
                    next_step.button(f"Continue to {WORKFLOW_TABS[index+1]} →",key=f"workflow_next_{index}",
                                     on_click=continue_to_method_comparison if index==2 else switch_workflow_tab,
                                     args=(comparison_handoff,) if index==2 else (WORKFLOW_TABS[index+1],),
                                     type="primary",width="stretch")

# Legal notice
st.link_button("Impressum", "https://www.uni-regensburg.de/impressum")
