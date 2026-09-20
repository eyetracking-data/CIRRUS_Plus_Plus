from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
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

from cirruspp_core import *  # noqa: F403,F401

st.set_page_config(page_title="CIRRUS++ | Domain-aware preprocessing", page_icon="☁️", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.block-container {padding-top:1rem; padding-bottom:3rem; max-width:1550px;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
.hero {padding:1.15rem 1.35rem; border-radius:20px; color:white;
background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 58%,#38bdf8 100%); margin-bottom:.8rem;}
.hero h1 {margin:0 0 .15rem 0; font-size:2.05rem}.hero p{margin:0;color:#e2e8f0}
.stage-card {border:1px solid rgba(59,130,246,.25);border-radius:16px;padding:.9rem;background:rgba(239,246,255,.60);min-height:185px}
.pill {display:inline-block;padding:.18rem .52rem;margin:.08rem;border-radius:999px;border:1px solid #93c5fd;background:#eff6ff;font-size:.82rem}
.subtle {font-size:.86rem;color:#475569}.good{color:#166534}.warn{color:#92400e}
.info-card {border-left:4px solid #60a5fa;padding:.65rem .9rem;background:#eff6ff;border-radius:8px;margin:.3rem 0 .7rem 0}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>☁️ CIRRUS++</h1><p>Domain-aware, transparent preprocessing for multivariate time-series data.</p></div>', unsafe_allow_html=True)

METHOD_LABELS = {
    "none":"None", "mean":"Mean", "median":"Median", "locf":"LOCF", "linear":"Linear interpolation",
    "knn":"KNN imputation", "mice":"MICE", "zscore":"Z-score", "iqr":"IQR", "mad":"MAD",
    "winsorize":"Winsorization", "isolation_forest":"Isolation Forest", "moving_median":"Moving median",
    "savgol":"Savitzky–Golay", "butterworth":"Butterworth", "standard":"StandardScaler",
    "minmax":"MinMaxScaler", "robust":"RobustScaler",
}
IMP_METHODS=["none","mean","median","locf","linear","knn","mice"]
OUT_METHODS=["none","zscore","iqr","mad","winsorize","isolation_forest"]
SMOOTH_METHODS=["none","moving_median","savgol","butterworth"]
SCALE_METHODS=["none","standard","minmax","robust"]
LABEL_TO_DOMAIN={v:k for k,v in DOMAIN_LABELS.items()}  # noqa: F405

def ml(m:str)->str: return METHOD_LABELS.get(m,m.replace("_"," ").title())
def safe_json(x:Any)->bytes: return json.dumps(x,indent=2,ensure_ascii=False,default=str).encode("utf-8")

def downsample(n:int,max_points:int):
    return np.arange(n) if n<=max_points else np.linspace(0,n-1,max_points).astype(int)

def line_plot(series:dict[str,pd.Series], timestamp:pd.Series, max_points:int, title:str):
    n=min(len(s) for s in series.values()); idx=downsample(n,max_points); fig=go.Figure()
    dashes=["solid","dash","dot","dashdot","longdash","longdashdot"]
    for i,(name,s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=timestamp.iloc[idx],y=pd.to_numeric(s.iloc[idx],errors="coerce"),mode="lines",name=name,line=dict(width=2.6 if i==0 else 1.8,dash=dashes[i%len(dashes)]),connectgaps=False))
    fig.update_layout(height=390,title=title,margin=dict(l=35,r=20,t=55,b=35),legend=dict(orientation="h"),xaxis_title="Sample time (s)")
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

def focused_line_plot(raw:pd.Series, outputs:dict[str,pd.Series], timestamp:pd.Series, title:str):
    win=method_difference_window(outputs)
    if win is None: return None
    start,end=win; series={"Raw":raw.iloc[start:end]}
    series.update({ml(k):v.iloc[start:end] for k,v in outputs.items()})
    ts=timestamp.iloc[start:end].reset_index(drop=True); series={k:v.reset_index(drop=True) for k,v in series.items()}
    return line_plot(series,ts,5000,title+f" · samples {start}–{end-1}")

def method_guide():
    st.markdown("""
**Imputation**
- **Mean / Median:** replace missing values with the column mean or median.
- **LOCF:** carries the last observed value forward; only a leading gap is back-filled so the series is complete.
- **Linear interpolation:** connects neighbouring observed values with a straight line.
- **KNN:** estimates a missing value from similar multivariate rows.
- **MICE:** iteratively predicts each incomplete feature from the others.

**Outlier handling**
- **Z-score:** flags values more than 3 standard deviations from the mean.
- **IQR:** flags values outside `Q1 − 1.5×IQR` and `Q3 + 1.5×IQR`.
- **MAD:** robust median-based outlier detection using a modified z-score.
- **Winsorization:** clips each signal to its 1st/99th percentiles instead of deleting points.
- **Isolation Forest:** multivariate anomaly detector; anomalous rows are flagged jointly.

For Z-score, IQR, MAD and Isolation Forest, flagged values are temporarily set missing and repaired by linear interpolation with a median fallback.

**Smoothing**
- **Moving median:** robust local median filter (up to 9 samples).
- **Savitzky–Golay:** local polynomial smoother (up to 21 samples) that can preserve peak shape better than a moving average.
- **Butterworth:** third-order low-pass filter that attenuates high-frequency variation.

**Scaling**
- **StandardScaler:** centres by the mean and scales by standard deviation.
- **MinMaxScaler:** maps each feature to `[0,1]`.
- **RobustScaler:** centres/scales using median and IQR and is less sensitive to extremes.

**None** disables the respective stage. Method names describe what is executed in this app; they are not claims that a method is universally appropriate.
""")

def longest_complete_run_length(frame:pd.DataFrame)->int:
    cols=[c for c in frame.columns if c!="timestamp"]
    if not cols or frame.empty: return 0
    mask=frame[cols].notna().all(axis=1).to_numpy(bool)
    if not mask.any(): return 0
    padded=np.r_[False,mask,False].astype(int); edges=np.flatnonzero(np.diff(padded))
    runs=edges[1::2]-edges[::2]
    return int(runs.max()) if len(runs) else 0

def metric_change_guide():
    guide=pd.DataFrame([
        ["Δ intrinsic quality index","after − before","Positive means the domain-weighted intrinsic diagnostic score increased. It does not prove better reconstruction."],
        ["Δ missing rate","after − before","Negative means fewer unavailable values; usually caused by imputation."],
        ["Δ IQR / MAD outlier rate","after − before","Negative means fewer values are statistically flagged as extreme. Genuine events may also disappear."],
        ["Δ skewness","after − before","Shows how distribution asymmetry changed; movement toward 0 is not automatically better."],
        ["Δ excess kurtosis","after − before","Negative means less heavy-tailed/peaked relative to a normal distribution; this may reflect cleaning or loss of genuine extremes."],
        ["Δ lag-1 autocorrelation","after − before","Positive means adjacent samples became more similar; smoothing often increases it."],
        ["Δ drift score","after − before","Negative means less slow rolling-mean movement relative to variability."],
        ["Δ local volatility","after − before","Negative means smaller first-difference variability, i.e. a smoother signal. Too large a decrease can mean over-smoothing."],
        ["Δ spectral entropy","after − before","Negative means power became more concentrated in fewer frequencies; smoothing can reduce it."],
    ],columns=["Metric","Delta definition","How to read the change"])
    st.dataframe(guide,hide_index=True,width="stretch")
    st.caption("All deltas are descriptive. A desirable direction depends on the signal, domain and task; Ground truth is required when reconstruction fidelity is the question.")

def quality_radar(profile:dict[str,float]):
    comp=quality_components(profile)  # noqa: F405
    labelmap={"completeness":"Completeness","continuity":"Continuity","plausibility":"Outlier burden","distribution":"Distribution shape","stability":"Temporal stability","correlation":"Cross-signal consistency"}
    labels=[labelmap[k] for k in comp]; vals=list(comp.values()); fig=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=labels+[labels[0]],fill="toself")); fig.update_layout(height=355,showlegend=False,polar=dict(radialaxis=dict(range=[0,100],visible=True)),margin=dict(l=30,r=30,t=30,b=30)); return fig

def domain_weight_chart(weights:dict[str,float]):
    labels={"completeness":"Completeness","continuity":"Continuity","plausibility":"Outlier burden","distribution":"Distribution shape","stability":"Temporal stability","correlation":"Cross-signal consistency"}
    df=pd.DataFrame({"component":[labels[k] for k in weights],"weight":list(weights.values())}); fig=px.bar(df,x="component",y="weight",text=df.weight.map(lambda x:f"{x:.2f}"),title="Domain weights used in the intrinsic quality index"); fig.update_layout(height=320,showlegend=False,margin=dict(l=30,r=20,t=55,b=90)); return fig

def recommendation_card(rec):
    status={"candidate":"Pilot rule","unstable":"Unstable pilot rule","goal_based":"Goal-based"}.get(rec.status,rec.status)
    st.markdown(f'<div class="stage-card"><span class="pill">{status}</span><h3 style="margin:.45rem 0 .2rem 0">{ml(rec.method)}</h3><div class="subtle"><b>Recommendation for this dataset profile</b><br>Derived from the measured signal profile and transparent rule thresholds.</div></div>',unsafe_allow_html=True)
    with st.expander("Why this recommendation?"):
        for x in rec.rule_trace: st.write("• "+x)
        st.caption("These thresholds are pilot rules. The recommendation is profile-specific, not a probability that this method is universally best.")
    if rec.stage!="scaling":
        with st.expander("Domain pilot context"):
            if rec.domain_support:
                st.write("Across the documented **real eye-tracking pilot scenarios**, winner frequencies were:")
                support=pd.DataFrame([{"method":ml(m),"winner_frequency":s} for m,s in rec.domain_support])
                st.dataframe(support.style.format({"winner_frequency":"{:.1%}"}),hide_index=True,width="stretch")
                st.caption("This is domain-level context. It is deliberately separated from the recommendation for the current uploaded dataset.")
            else:
                st.write("No comparable real-data winner-frequency table is shown for this domain. Its profile is currently supported by literature/synthetic domain logic rather than a real-data frequency claim.")
        with st.expander("Validation detail"):
            v=rec.validation; st.write(rec.caution)
            st.caption(f"Grouped CV: {v['grouped_cv_mean']:.3f}; majority baseline: {v['majority']:.3f}; {v['holdout_name']} hold-out: {v['holdout']:.3f}.")
    else:
        with st.expander("How scaling is chosen"):
            st.write(rec.caution)

@st.cache_data(show_spinner=False)
def load_upload(data:bytes,filename:str,sep:str,dec:str): return read_table_bytes(data,filename,sep,dec)  # noqa: F405
@st.cache_data(show_spinner=False)
def synth(domain:str,n:int,seed:int): return generate_domain_series(domain,n,seed)  # noqa: F405

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source_mode=st.radio("Source",["Upload CSV/TSV","Synthetic demo"],index=0)
    parser_note=""; source_event_mask=None
    if source_mode=="Synthetic demo":
        dlab=st.selectbox("Synthetic domain",list(DOMAIN_LABELS.values()))  # noqa: F405
        n=st.slider("Samples",400,8000,2400,200); seed=st.number_input("Seed",0,1_000_000,42)
        raw_df,source_event_mask,synthetic_fs=synth(LABEL_TO_DOMAIN[dlab],n,int(seed)); source_name=f"synthetic_{LABEL_TO_DOMAIN[dlab]}.csv"
        separator=decimal="Auto"
    else:
        up=st.file_uploader("Time-series file",type=["csv","tsv","txt"]); separator=st.selectbox("Separator",["Auto",",",";","Tab","|"]); decimal=st.selectbox("Decimal",["Auto",".",","])
        if up is None:
            st.info("Upload a time-series table. The synthetic demo remains available without a file."); st.stop()
        loaded=load_upload(up.getvalue(),up.name,separator,decimal); raw_df=loaded.frame; source_name=loaded.source_name; parser_note=f"separator={loaded.separator!r}, decimal={loaded.decimal!r}, encoding={loaded.encoding}"

    st.header("Signals")
    time_candidates=likely_time_columns(raw_df)  # noqa: F405
    time_options=["Sample index"]+list(raw_df.columns); default_t=time_options.index(time_candidates[0]) if time_candidates else 0
    timestamp_choice=st.selectbox("Timestamp column",time_options,index=default_t); timestamp_col=None if timestamp_choice=="Sample index" else timestamp_choice
    include_ids=st.checkbox("Include identifier-like numeric columns",False)
    possible=numeric_signal_columns(raw_df,include_ids)  # noqa: F405
    if not possible: st.error("No usable numeric signals detected."); st.stop()
    preliminary,_=infer_domain(raw_df,possible)  # noqa: F405
    possible=rank_signal_columns(possible,preliminary)  # noqa: F405
    duplicate_groups=exact_duplicate_signal_groups(raw_df,possible)  # noqa: F405
    default_signals=distinct_signal_defaults(raw_df,possible,5)  # noqa: F405
    selected_signals=st.multiselect("Signal columns",possible,default=default_signals)
    if duplicate_groups:
        dup_text="; ".join(" = ".join(g) for g in duplicate_groups)
        st.caption("Exact duplicate numeric channels detected. Auto-selection keeps one representative: "+dup_text)
    if not selected_signals: st.warning("Select at least one signal."); st.stop()
    selected_dup=[g for g in duplicate_groups if sum(c in selected_signals for c in g)>1]
    if selected_dup:
        st.warning("Selected signals include exact duplicates. This can inflate cross-signal consistency; keep both only if you intentionally want both channels in the analysis.")

    inferred_fs,fs_note=infer_sampling_rate(raw_df,timestamp_col)  # noqa: F405
    if source_mode=="Synthetic demo" and inferred_fs is None: inferred_fs=synthetic_fs; fs_note="Synthetic sampling rate"
    use_inferred=st.checkbox("Use inferred sampling rate",value=inferred_fs is not None,disabled=inferred_fs is None)
    default_fs=float(inferred_fs) if inferred_fs is not None else 1.0
    fs=default_fs if use_inferred else st.number_input("Sampling rate (Hz)",min_value=.00001,max_value=10000.,value=max(.00001,default_fs),format="%.6f")
    st.caption(f"{fs_note}; active rate: {fs:.6g} Hz")

    inferred_domain,domain_scores=infer_domain(raw_df,selected_signals)  # noqa: F405
    domain_option=st.selectbox("Domain profile",["Auto-detect"]+list(DOMAIN_LABELS.values()))  # noqa: F405
    domain=inferred_domain if domain_option=="Auto-detect" else LABEL_TO_DOMAIN[domain_option]
    st.caption(f"Active: {DOMAIN_LABELS[domain]}")  # noqa: F405
    validity_mode="Raw numeric values"
    validity_summary=pd.DataFrame(); validity_meta={"mode":validity_mode,"affected":[]}
    if domain=="eye_tracking":
        validity_summary=eye_tracking_validity_summary(raw_df)  # noqa: F405
        with st.expander("Eye-tracking validity / missingness",expanded=True):
            modes=["Raw numeric values","Treat paired gaze (0,0) as unavailable","Domain-aware gaze validity (0,0 + Blink labels)"]
            recommended=2 if not validity_summary.empty and validity_summary["blink_labels"].sum()>0 else (1 if not validity_summary.empty and validity_summary["paired_0_0"].sum()>0 else 0)
            validity_mode=st.selectbox("Interpret gaze availability",modes,index=recommended)
            st.caption("Only paired gaze coordinates (x=0 AND y=0) are treated as unavailable. A single x=0 or y=0 remains a valid numeric coordinate. In the domain-aware mode, same-eye Blink labels are also treated as unavailable gaze. Raw input is never overwritten.")
            if not validity_summary.empty:
                st.dataframe(validity_summary[["eye","paired_0_0_%","blink_%","combined_unavailable_%"]].style.format({"paired_0_0_%":"{:.1f}%","blink_%":"{:.1f}%","combined_unavailable_%":"{:.1f}%"}),hide_index=True,width="stretch")
    scaling_goal=st.selectbox("Scaling objective",["Auto","Preserve physical units","Distance/gradient-based model","Bounded model input [0,1]"])
    max_plot_points=st.slider("Maximum plotted points",500,15000,5000,500)

analysis_df,validity_meta=apply_eye_tracking_validity(raw_df,validity_mode) if domain=="eye_tracking" else (raw_df.copy(),{"mode":"Not applicable","affected":[]})  # noqa: F405
canonical=canonical_frame(analysis_df,selected_signals,fs)  # noqa: F405
domain_cfg=DOMAIN_CONFIG[domain]  # noqa: F405
profile_agg=aggregate_profile(canonical,fs)  # noqa: F405
profile_features=profile_table(canonical,fs,domain_cfg["quality_weights"])  # noqa: F405
q_index=quality_index(profile_agg,domain_cfg["quality_weights"])  # noqa: F405
recs={s:recommend_stage(s,profile_agg,domain) for s in ["imputation","outlier","smoothing"]}  # noqa: F405
recs["scaling"]=recommend_scaling(profile_agg,domain,scaling_goal)  # noqa: F405

with st.sidebar:
    st.header("Active pipeline")
    def sel(label,pool,rec): return st.selectbox(label,pool,index=pool.index(rec) if rec in pool else 0,format_func=ml)
    manual_imp=sel("Imputation",IMP_METHODS,recs["imputation"].method); manual_out=sel("Outlier handling",OUT_METHODS,recs["outlier"].method); manual_smooth=sel("Smoothing",SMOOTH_METHODS,recs["smoothing"].method); manual_scale=sel("Scaling",SCALE_METHODS,recs["scaling"].method)
    stage_order=st.selectbox("Stage order",["Imputation → Outlier → Smoothing → Scaling","Outlier → Imputation → Smoothing → Scaling"])
    with st.expander("What do the preprocessing methods do?"):
        method_guide()
active_spec=PipelineSpec(manual_imp,manual_out,manual_smooth,manual_scale)  # noqa: F405
order_key="outlier_imputation_smoothing_scaling" if stage_order.startswith("Outlier") else "imputation_outlier_smoothing_scaling"
pipeline_key=json.dumps({"source":source_name,"signals":selected_signals,"spec":asdict(active_spec),"order":order_key,"fs":fs,"validity_mode":validity_mode},sort_keys=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs=st.tabs(["1 · Data & domain","2 · Signal profile","3 · Recommendation","4 · Step compare","5 · Pipeline lab","6 · Ground truth","7 · Export"])

with tabs[0]:
    m=st.columns(5); m[0].metric("Rows",f"{len(raw_df):,}"); m[1].metric("Columns",raw_df.shape[1]); m[2].metric("Analysed signals",len(selected_signals)); m[3].metric("Sampling rate",f"{fs:.3g} Hz"); m[4].metric("Intrinsic quality index",f"{q_index:.1f}/100")
    st.markdown('<div class="info-card"><b>What the quality index means:</b> a domain-weighted summary of intrinsic diagnostics in the selected signals. It is not ground truth, not a probability, and not proof that a dataset is “90% correct”.</div>',unsafe_allow_html=True)
    with st.expander("How is the intrinsic quality index calculated?"):
        st.write("CIRRUS++ maps the dataset-level profile to six 0–100 components and then takes the weighted average using the active domain profile.")
        qb=quality_breakdown(profile_agg,domain_cfg["quality_weights"])  # noqa: F405
        st.dataframe(qb.style.format({"score_0_100":"{:.1f}","domain_weight":"{:.0%}","weighted_points":"{:.2f}"}),hide_index=True,width="stretch")
        st.markdown("""
**Exact component transforms**

- **Completeness:** penalises the fraction of missing values; reaches 0 at 45% missingness.
- **Continuity:** combines missing-value burstiness (55%) with the 95th-percentile gap duration (45%).
- **Outlier burden:** uses the larger of the IQR- and MAD-based outlier rates; reaches 0 at 25% statistical outliers.
- **Distribution shape:** penalises strong absolute skewness and positive excess kurtosis.
- **Temporal stability:** combines the drift score (65%) and local volatility (35%).
- **Cross-signal consistency:** increases with mean absolute cross-channel correlation.

The final score is `sum(domain weight × component score) / sum(weights)`.

**Important limitation:** these are intrinsic diagnostics, not universal truths. A real saccade can look like a jump, heavy tails can be legitimate, and high correlation is not automatically “better”. This is why CIRRUS++ keeps the index separate from the Ground truth validation.
""")
    left,right=st.columns([1,1.1])
    with left:
        st.subheader("Active domain profile"); st.markdown(f"### {DOMAIN_LABELS[domain]}"); st.write(domain_cfg["description"]); st.plotly_chart(domain_weight_chart(domain_cfg["quality_weights"]),width="stretch")
        with st.expander("How was the domain detected?"):
            if domain_option=="Auto-detect":
                st.write("Auto-detection is intentionally simple and auditable: CIRRUS++ scores domain-related keywords in column names (for example gaze/pupil, temperature/pressure, price/return). It is not a learned domain classifier.")
                st.json(domain_scores)
            else: st.write("You manually selected the active domain profile; the automatic suggestion is not used.")
        if domain=="eye_tracking":
            with st.expander("How are eye-tracking validity and missing values interpreted?",expanded=True):
                st.write(f"**Active interpretation:** {validity_mode}")
                st.write("The original file is preserved. CIRRUS++ creates an analysis copy and can reinterpret paired gaze `(0,0)` values and same-eye Blink labels as unavailable gaze. This prevents format-specific invalid gaze markers from being mistaken for ordinary numeric measurements.")
                if not validity_summary.empty:
                    st.dataframe(validity_summary.style.format({"paired_0_0_%":"{:.2f}%","blink_%":"{:.2f}%","combined_unavailable_%":"{:.2f}%"}),hide_index=True,width="stretch")
                if validity_meta.get("affected"): st.json(validity_meta["affected"])
        if duplicate_groups:
            with st.expander("Exact duplicate-signal check"):
                st.write("CIRRUS++ checks exact equality before choosing the default signals. Exact duplicates are skipped in the automatic default because counting the same numeric channel twice can inflate cross-signal consistency.")
                st.dataframe(pd.DataFrame({"exact_duplicate_group":[" = ".join(g) for g in duplicate_groups]}),hide_index=True,width="stretch")
    with right:
        st.subheader("Data preview"); st.caption(source_name); st.dataframe(raw_df.head(30),width="stretch",height=300)
        analysis_missing={c:float(pd.to_numeric(analysis_df[c],errors="coerce").isna().mean()*100) if c in analysis_df.columns and c in selected_signals else np.nan for c in raw_df.columns}
        schema=pd.DataFrame({"column":raw_df.columns,"dtype":raw_df.dtypes.astype(str).values,"raw_missing_%":raw_df.isna().mean().mul(100).round(3).values,"analysis_missing_%":[round(analysis_missing[c],3) if np.isfinite(analysis_missing[c]) else np.nan for c in raw_df.columns],"selected_signal":[c in selected_signals for c in raw_df.columns]}); st.dataframe(schema,width="stretch",height=260)
        with st.expander("Import details"):
            st.write(parser_note or "Synthetic data: no external parser was used."); st.write(f"Timestamp selection: {timestamp_choice}"); st.write(f"Sampling-rate note: {fs_note}")

with tabs[1]:
    st.subheader("Signal profile")
    st.caption("Profile individual signals before choosing or comparing preprocessing methods.")
    show=profile_features.copy(); show["missing_%"]=100*show["missing_rate"]; show["iqr_outlier_%"]=100*show["outlier_iqr_rate"]; show["mad_outlier_%"]=100*show["outlier_mad_rate"]
    display=["feature","quality_index","missing_%","missing_run_p95_seconds","iqr_outlier_%","mad_outlier_%","skewness","kurtosis_excess","lag1_autocorr","drift_score","local_volatility","spectral_entropy","dominant_frequency_ratio","corr_mean_abs"]
    st.dataframe(show[display].round(4),width="stretch",height=290)
    focus=st.selectbox("Inspect feature",selected_signals,key="profile_focus"); fp=profile_features.loc[profile_features.feature==focus].iloc[0].to_dict()
    c=st.columns(4); c[0].metric("Missing",f"{fp['missing_rate']:.2%}",f"p95 gap {fp['missing_run_p95_seconds']:.3f}s"); c[1].metric("MAD outliers",f"{fp['outlier_mad_rate']:.2%}",f"IQR {fp['outlier_iqr_rate']:.2%}"); c[2].metric("Skewness",f"{fp['skewness']:.3f}",f"Excess kurtosis {fp['kurtosis_excess']:.3f}"); c[3].metric("Lag-1 autocorr.",f"{fp['lag1_autocorr']:.3f}",f"Volatility {fp['local_volatility']:.3f}")
    p=st.columns([1.15,1,1]);
    with p[0]: st.plotly_chart(line_plot({"Raw":canonical[focus]},canonical.timestamp,max_plot_points,f"Signal over time · {focus}"),width="stretch")
    with p[1]: st.plotly_chart(px.histogram(pd.to_numeric(canonical[focus],errors="coerce").dropna(),nbins=45,marginal="box",title=f"Distribution · {focus}"),width="stretch")
    with p[2]: st.plotly_chart(quality_radar(fp),width="stretch")
    with st.expander("What do the profile metrics mean?"):
        st.markdown("""
- **Missing rate:** share of unavailable values. **p95 gap:** duration below which 95% of missing runs fall.
- **IQR outlier rate:** share outside `Q1 − 1.5×IQR` / `Q3 + 1.5×IQR`. **MAD outlier rate:** robust modified-z criterion based on the median absolute deviation.
- **Skewness:** asymmetry of the value distribution. **Excess kurtosis:** tail/peak shape relative to a normal distribution; high values do not prove an error.
- **Lag-1 autocorrelation:** similarity between consecutive samples.
- **Drift score:** strength of slow movement in a rolling mean relative to signal variability.
- **Local volatility:** standard deviation of first differences divided by signal standard deviation.
- **Spectral entropy:** how broadly signal power is spread across frequencies (0 concentrated, 1 diffuse).
- **Dominant-frequency ratio:** share of spectral power carried by the strongest frequency bin.
- **Mean absolute correlation:** average absolute correlation with the other selected signals.

The radar shows the six **unweighted intrinsic components for this signal**. The overall quality index additionally applies the active domain weights.
""")

with tabs[2]:
    st.subheader("Recommendations")
    st.caption("Profile-specific recommendations and domain-level pilot context are shown separately so that a domain frequency cannot be mistaken for confidence in the current dataset.")
    with st.expander("Quick guide: what do the available preprocessing methods do?"):
        method_guide()
    cols=st.columns(4)
    for col,stage in zip(cols,["imputation","outlier","smoothing","scaling"]):
        with col: st.markdown(f"### {stage.title()}"); recommendation_card(recs[stage])
    st.divider(); st.subheader("Active pipeline after manual review"); st.markdown(" ".join(f'<span class="pill">{s.title()}: {ml(getattr(active_spec,s))}</span>' for s in ["imputation","outlier","smoothing","scaling"]),unsafe_allow_html=True); st.caption(f"Order: {stage_order}. Manual choices override recommendations and are recorded in the export.")

with tabs[3]:
    st.subheader("Step compare")
    mode=st.radio("Comparison mode",["Isolated stage","Pipeline context"],horizontal=True)
    if mode=="Isolated stage": st.info("Only the selected stage is active; all other preprocessing stages are set to None. Use this to isolate a method's intrinsic effect.")
    else: st.info("The selected stage varies while the other stages stay exactly as configured in the Active pipeline. Use this to compare methods inside the combined pipeline context.")
    stage=st.selectbox("Stage",["imputation","outlier","smoothing","scaling"],key="compare_stage"); pool={"imputation":IMP_METHODS,"outlier":OUT_METHODS,"smoothing":SMOOTH_METHODS,"scaling":SCALE_METHODS}[stage]
    default=list(dict.fromkeys(["none",getattr(active_spec,stage)]+pool[1:3]))[:4]; methods=st.multiselect("Methods",pool,default=default,format_func=ml); feature=st.selectbox("Feature",selected_signals,key="compare_feature")
    with st.expander("What does the quality-index bar chart mean?"):
        st.write("For each compared method CIRRUS++ executes the requested configuration, profiles the resulting signals, calculates the same six intrinsic components, and applies the current domain weights. A higher bar means a better **intrinsic diagnostic score**, not proven reconstruction accuracy. Ground truth validation is the place for known-reference ranking.")
        if stage=="scaling": st.warning("Linear scaling usually leaves most shape-based intrinsic diagnostics nearly unchanged. Scaling should primarily be judged by the downstream analysis objective, not by this quality index.")
    with st.expander("What do the methods in this comparison do?"):
        method_guide()
    st.caption("Tip: in the full-series plot several methods may overlap almost perfectly. CIRRUS++ therefore adds a second zoom plot at the first region where the selected methods actually differ.")
    if st.button("Run comparison",type="primary",key="run_compare") and methods:
        base=PipelineSpec() if mode=="Isolated stage" else active_spec  # noqa: F405
        # For imputation/outlier/smoothing, create a second visual-only result with scaling disabled.
        # Otherwise StandardScaler can compress all processed traces around zero while the raw trace remains in pixels,
        # making the chart appear to contain only one line. Metrics still use the requested full pipeline context.
        visual_base=base if stage=="scaling" else PipelineSpec(base.imputation,base.outlier,base.smoothing,"none")  # noqa: F405
        with st.spinner("Comparing methods..."):
            outs,cmp=compare_stage_methods(canonical,stage,methods,fs,domain_cfg["quality_weights"],base,order_key)  # noqa: F405
            visual_outs,_=compare_stage_methods(canonical,stage,methods,fs,domain_cfg["quality_weights"],visual_base,order_key)  # noqa: F405
        st.session_state["compare_result"]={"key":(mode,stage,tuple(methods),feature,pipeline_key),"outputs":outs,"visual_outputs":visual_outs,"table":cmp}
    cr=st.session_state.get("compare_result")
    if cr and cr["key"]==(mode,stage,tuple(methods),feature,pipeline_key):
        visual_outs=cr.get("visual_outputs",cr["outputs"])
        if stage!="scaling":
            st.caption("The comparison plot is shown **before scaling** so raw and method traces stay in the same physical units. The table below still reflects the requested full pipeline context.")
            feature_visual_outs={k:v[feature] for k,v in visual_outs.items()}
            series={"Raw":canonical[feature]}; series.update({ml(k):v for k,v in feature_visual_outs.items()}); st.plotly_chart(line_plot(series,canonical.timestamp,max_plot_points,f"{mode} · {stage.title()} · {feature} · full signal"),width="stretch")
            zoom=focused_line_plot(canonical[feature],feature_visual_outs,canonical.timestamp,"Zoom on first region changed/filled by the compared methods")
            if zoom is not None:
                st.plotly_chart(zoom,width="stretch")
                st.caption("This second plot automatically zooms to the first region where the compared methods actually disagree with each other. Different dash patterns help reveal traces that overlap in the full-series view.")
            else:
                st.info("For this feature and configuration the selected methods produce the same numerical trace, so there is no region in which separate lines can be shown.")
        else:
            st.warning("Scaling methods change numerical units. A raw-vs-scaled overlay is therefore not a like-for-like visual comparison; use the method table and inspect the scaled values separately.")
            series={ml(k):v[feature] for k,v in visual_outs.items()}; st.plotly_chart(line_plot(series,canonical.timestamp,max_plot_points,f"Scaling outputs · {feature}"),width="stretch")
        c1,c2=st.columns([1.3,1]); table=cr["table"].copy(); table["method"]=table.method.map(ml)
        table["cleaning_changed_%"]=100*table["cleaning_changed_fraction"]; table["overall_changed_%"]=100*table["overall_changed_fraction"]
        show_cols=["method","quality_index","quality_delta","missing_rate","outlier_rate","local_volatility","cleaning_changed_%","overall_changed_%"]
        with c1: st.dataframe(table[show_cols].round(5),width="stretch")
        with c2:
            fig=px.bar(table,x="method",y="quality_index",text_auto=".2f",title="Domain-weighted intrinsic quality index"); fig.update_layout(height=350,showlegend=False); st.plotly_chart(fig,width="stretch")
        with st.expander("How to read the comparison table"):
            st.markdown("""
- **quality_index / quality_delta:** intrinsic heuristic after the method and its change from the unprocessed baseline. Higher is not automatically more faithful.
- **missing_rate:** remaining unavailable fraction after the compared configuration.
- **outlier_rate:** larger of IQR- and MAD-based statistical outlier rates after processing.
- **local_volatility:** normalized first-difference variability; lower generally means a smoother signal.
- **cleaning_changed_%:** fraction changed by imputation/outlier/smoothing, excluding scaling.
- **overall_changed_%:** fraction numerically changed including scaling.
""")

with tabs[4]:
    st.subheader("Pipeline lab")
    st.caption("Run the complete active pipeline deliberately. Nothing is executed just because this tab was opened or a sidebar setting changed.")
    with st.expander("What happens when I click Run active pipeline?"):
        st.markdown("""
CIRRUS++ applies the selected stages in the chosen order. Outlier detectors (Z-score/IQR/MAD/Isolation Forest) identify values and repair detected points using linear interpolation with a median fallback; Winsorization clips the 1st/99th percentiles. Smoothing then runs if selected, followed by scaling.

The app reports **cleaning changes separately from scaling transformations**. This matters because a scaler can numerically transform 100% of values without meaning that 100% of observations were repaired.
""")
        st.caption("LOCF uses forward filling and a backward fill only for an initial leading gap. Moving median and Savitzky–Golay use sample-based windows (up to 9 and 21 samples respectively).")
    if st.button("Run active pipeline",type="primary",width="stretch"):
        start=time.perf_counter()
        with st.spinner("Executing pipeline..."):
            processed,meta=execute_pipeline(canonical,active_spec,fs,order_key)  # noqa: F405
            clean_spec=PipelineSpec(active_spec.imputation,active_spec.outlier,active_spec.smoothing,"none")  # noqa: F405
            cleaned,_=execute_pipeline(canonical,clean_spec,fs,order_key)  # noqa: F405
        st.session_state["pipeline"]={"key":pipeline_key,"processed":processed,"cleaned":cleaned,"meta":meta,"elapsed":time.perf_counter()-start}
    pr=st.session_state.get("pipeline")
    if pr is None or pr["key"]!=pipeline_key:
        st.warning("The current active pipeline has not been run yet. Click **Run active pipeline** above.")
    else:
        processed=pr["processed"]; cleaned=pr["cleaned"]; meta=pr["meta"]; after_q=quality_index(aggregate_profile(cleaned,fs),domain_cfg["quality_weights"])  # noqa: F405
        p=st.columns(4); p[0].metric("Before intrinsic quality",f"{q_index:.1f}/100"); p[1].metric("After cleaning",f"{after_q:.1f}/100",f"{after_q-q_index:+.1f}"); p[2].metric("Cleaning changes",f"{meta['cleaning_changed_fraction']:.2%}"); p[3].metric("Scaling transforms",f"{meta['scaling_changed_fraction']:.2%}")
        stages=pd.DataFrame([{"stage":k.title(),"cells changed vs previous stage":v} for k,v in meta["stage_changed_fraction"].items()]); st.dataframe(stages.style.format({"cells changed vs previous stage":"{:.2%}"}),hide_index=True,width="stretch")
        feat=st.selectbox("Preview feature",selected_signals,key="pipeline_feature"); st.plotly_chart(line_plot({"Raw":canonical[feat],"After cleaning (before scaling)":cleaned[feat]},canonical.timestamp,max_plot_points,f"Cleaning effect · {feat}"),width="stretch")
        if active_spec.scaling!="none": st.caption(f"Scaling ({ml(active_spec.scaling)}) is intentionally excluded from this overlay because raw and scaled values use different numerical units.")
        change=profile_change_table(canonical,cleaned,fs,domain_cfg["quality_weights"])  # noqa: F405
        st.subheader("Metric changes by feature")
        with st.expander("What are these metrics, and what does a change mean?",expanded=True):
            metric_change_guide()
        compact=change[["feature","before_quality_index","after_quality_index","delta_quality_index","delta_missing_rate","delta_outlier_mad_rate","delta_kurtosis_excess","delta_local_volatility"]].copy()
        compact["delta_missing_pp"]=100*compact.pop("delta_missing_rate")
        compact["delta_MAD_outlier_pp"]=100*compact.pop("delta_outlier_mad_rate")
        compact=compact.rename(columns={"before_quality_index":"quality_before","after_quality_index":"quality_after","delta_quality_index":"delta_quality","delta_kurtosis_excess":"delta_excess_kurtosis","delta_local_volatility":"delta_local_volatility"})
        st.dataframe(compact.round(5),width="stretch")
        with st.expander("Show all before/after metric changes"):
            st.dataframe(change.round(6),width="stretch")
        with st.expander("Processed data preview"):
            rows=st.slider("Rows shown",10,min(500,max(10,len(processed))),min(100,max(10,len(processed))),10); pv=processed[["timestamp"]+selected_signals].head(rows); st.dataframe(pv.round(6),width="stretch")
        with st.expander("Optional downstream before/after check"):
            st.caption("Exploratory only. CIRRUS++ uses forward TimeSeriesSplit folds and fits the imputer/model on training data only to reduce temporal leakage.")
            target_candidates=[c for c in raw_df.columns if c not in selected_signals]; target=st.selectbox("Target column",["None"]+target_candidates)
            if target!="None" and st.button("Run downstream check"):
                res=downstream_validation(canonical,cleaned,selected_signals,raw_df[target])  # noqa: F405
                if res.empty: st.warning("Not enough usable time-ordered samples/classes for validation.")
                else: st.dataframe(res.round(5),width="stretch"); st.plotly_chart(px.bar(res,x="data",y="cv_mean",error_y="cv_sd",title=res.metric.iloc[0]),width="stretch")

with tabs[5]:
    st.subheader("Ground truth")
    st.markdown('<div class="info-card"><b>What this tab is for:</b> intrinsic scores cannot tell whether preprocessing reconstructed the unknown truth. Here you can create a controlled pseudo-ground-truth experiment or supply a real clean reference.</div>',unsafe_allow_html=True)
    with st.expander("How to use this tab"):
        st.markdown("""
**Controlled corruption** (no second file needed):
1. CIRRUS++ finds a complete window in your current data and treats that window as a temporary reference.
2. It injects a known corruption (missingness, spikes, drift, etc.).
3. Several compact pipeline candidates try to reconstruct the reference.
4. The original window is known, so **lower domain-weighted loss is better**.

**Clean reference upload:** use this only if you have a clean version aligned sample-for-sample with the noisy file. CIRRUS++ does not silently resynchronise two unrelated recordings.
""")
    with st.expander("How is the domain-weighted loss calculated?"):
        st.write("The loss combines reconstruction error, harm to intact samples, dynamics, distribution, frequency structure, event preservation, detection performance when applicable, and retention. Only metrics that are actually available are included; unavailable event/detection terms are dropped and the remaining domain weights are renormalised.")
        st.json(domain_cfg["loss_weights"])
    mode=st.radio("Validation source",["Inject corruption into a complete window","Upload clean reference for current noisy data"],horizontal=True)
    if mode.startswith("Inject"):
        max_complete=longest_complete_run_length(canonical)
        c=st.columns(3); corruption=c[0].selectbox("Corruption",["mcar_missing","block_missing","spikes","drift","level_shift","stuck_at","high_frequency_noise","clipping"]); severity=c[1].slider("Severity",.01,.40,.10,.01)
        if max_complete < 32:
            c[2].warning("No complete reference window of at least 32 samples is available under the current validity interpretation/signals.")
            window_length=32
        else:
            upper=min(3000,max_complete); default_window=min(400,upper); step=10 if upper<200 else 50
            window_length=c[2].slider("Reference-window samples",32,upper,default_window,step)
            c[2].caption(f"Longest currently available complete run: {max_complete} samples ({max_complete/fs:.2f} s).")
        alt={}
        defaults={"imputation":["linear","locf"],"outlier":["zscore","iqr"],"smoothing":["none","moving_median"]}
        for s in ["imputation","outlier","smoothing"]:
            support=[m for m,_ in recs[s].domain_support]; alt[s]=support or defaults[s]
        validation_spec=PipelineSpec(active_spec.imputation,active_spec.outlier,active_spec.smoothing,"none")  # noqa: F405
        specs=compact_pipeline_grid(validation_spec,alt,10)  # noqa: F405
        st.caption(f"{len(specs)} compact candidate pipelines will be compared. Scaling is disabled because reconstruction metrics are scale-sensitive.")
        if st.button("Run controlled validation",type="primary"):
            clean=complete_window(canonical,window_length)  # noqa: F405
            if len(clean)<32: st.error("No sufficiently long complete reference window was found.")
            else:
                with st.spinner("Injecting corruption and evaluating candidates..."):
                    corrupted,results,outputs=controlled_validation(clean,specs,corruption,severity,fs,domain_cfg["loss_weights"],None,42)  # noqa: F405
                st.session_state["ground_truth"]={"kind":"controlled","key":(source_name,tuple(selected_signals),corruption,severity,window_length),"clean":clean,"corrupted":corrupted,"results":results,"outputs":outputs}
        gt=st.session_state.get("ground_truth")
        expected=(source_name,tuple(selected_signals),corruption,severity,window_length)
        if gt and gt.get("kind")=="controlled" and gt.get("key")==expected:
            results=gt["results"]; st.dataframe(results.round(5),width="stretch",height=300); best=results.iloc[0].pipeline; feat=st.selectbox("Validation feature",selected_signals,key="gt_feature"); st.plotly_chart(line_plot({"Reference":gt["clean"][feat],"Corrupted":gt["corrupted"][feat],"Best pipeline":gt["outputs"][best][feat]},gt["clean"].timestamp,max_plot_points,f"Pseudo-ground-truth reconstruction · {feat}"),width="stretch"); fig=px.bar(results.head(10),x="pipeline",y="loss",title="Domain-weighted reconstruction loss · lower is better"); fig.update_layout(xaxis_tickangle=-35,height=410); st.plotly_chart(fig,width="stretch")
    else:
        clean_up=st.file_uploader("Clean reference CSV/TSV",type=["csv","tsv","txt"],key="clean_ref")
        confirm=st.checkbox("I confirm that the clean and noisy files are aligned sample-for-sample (same observation order).")
        if clean_up is not None:
            clean_loaded=load_upload(clean_up.getvalue(),clean_up.name,separator,decimal); common=[c for c in selected_signals if c in clean_loaded.frame.columns]
            if not common: st.error("No selected signal columns with matching names were found in the clean reference.")
            else: st.caption(f"Matching signals: {', '.join(common)}. Rows will be truncated to the common length; no automatic time warping/resynchronisation is performed.")
        if clean_up is not None and confirm and st.button("Evaluate active cleaning against reference",type="primary"):
            n=min(len(clean_loaded.frame),len(canonical)); clean=canonical_frame(clean_loaded.frame.iloc[:n],common,fs); noisy=canonical.iloc[:n][["timestamp"]+common].reset_index(drop=True); eval_spec=PipelineSpec(active_spec.imputation,active_spec.outlier,active_spec.smoothing,"none"); pred,meta=execute_pipeline(noisy,eval_spec,fs,order_key)  # noqa: F405
            mask=pd.DataFrame({c:~np.isclose(clean[c].to_numpy(float),noisy[c].to_numpy(float),equal_nan=True) for c in common}); metrics=evaluate_reconstruction(clean,noisy,pred,mask,fs,event_mask=None,detected_mask=meta["detected_mask"],detection_applicable=False); loss=domain_weighted_loss(metrics,domain_cfg["loss_weights"])  # noqa: F405
            st.session_state["clean_ref_result"]={"loss":loss,"metrics":metrics}
        if "clean_ref_result" in st.session_state: st.metric("Domain-weighted loss",f"{st.session_state['clean_ref_result']['loss']:.4f}"); st.json(st.session_state["clean_ref_result"]["metrics"])

with tabs[6]:
    st.subheader("Export")
    st.caption("Exports are tied to the last explicitly executed Pipeline Lab run. This prevents silently exporting a pipeline that was never run.")
    pr=st.session_state.get("pipeline"); valid_run=pr is not None and pr.get("key")==pipeline_key
    with st.expander("What does each export contain?"):
        st.markdown("""
- **Processed full CSV:** your complete original table, including all untouched columns and the original timestamp column; only the selected signal columns are replaced by the executed pipeline output.
- **Pipeline JSON:** selected methods and stage order.
- **Audit JSON:** source/domain/sampling information, profile metrics, intrinsic-quality breakdown, recommendations, validation context, manual pipeline, and stage-wise change fractions.
- **Audit Markdown:** compact human-readable report.
""")
    if not valid_run: st.warning("Run the current pipeline once in **Pipeline lab** before exporting processed data.")
    processed=pr["processed"] if valid_run else canonical; cleaned=pr["cleaned"] if valid_run else canonical; meta=pr["meta"] if valid_run else {}
    after_profile=aggregate_profile(cleaned,fs)  # noqa: F405
    audit={"tool":"CIRRUS++","app_version":"1.5.0-demo-ready","source":source_name,"domain":domain,"domain_label":DOMAIN_LABELS[domain],"sampling_rate_hz":fs,"signals":selected_signals,"eye_tracking_validity_mode":validity_mode,"eye_tracking_validity_meta":validity_meta,"exact_duplicate_signal_groups":duplicate_groups,"quality_index_semantics":"domain-weighted intrinsic diagnostic heuristic; not ground truth","quality_index_before":q_index,"quality_index_after_cleaning":quality_index(after_profile,domain_cfg["quality_weights"]),"quality_breakdown_before":quality_breakdown(profile_agg,domain_cfg["quality_weights"]).to_dict("records"),"profile_before":profile_agg,"profile_after_cleaning":after_profile,"recommendations":{s:r.as_dict() for s,r in recs.items()},"executed_pipeline":asdict(active_spec) if valid_run else None,"stage_order":stage_order if valid_run else None,"execution_meta":{k:v for k,v in meta.items() if k!="detected_mask"} if valid_run else None}
    if st.session_state.get("ground_truth",{}).get("kind")=="controlled": audit["controlled_validation_results"]=st.session_state["ground_truth"]["results"].to_dict("records")
    full_processed=raw_df.copy()
    if valid_run:
        for c in selected_signals: full_processed[c]=processed[c].to_numpy()
    md=f"""# CIRRUS++ preprocessing audit\n\n- Source: `{source_name}`\n- Domain: **{DOMAIN_LABELS[domain]}**\n- Sampling rate: `{fs:.6g} Hz`\n- Analysed signals: {', '.join(selected_signals)}\n- Intrinsic quality before: `{q_index:.2f}/100`\n- Intrinsic quality after cleaning: `{audit['quality_index_after_cleaning']:.2f}/100`\n- Pipeline: `{active_spec.name() if valid_run else 'NOT RUN'}`\n- Order: `{stage_order if valid_run else 'NOT RUN'}`\n\nThe intrinsic quality index is a transparent domain-weighted diagnostic heuristic, not a ground-truth accuracy score. Domain pilot frequencies shown in the Recommendation tab are context and are not probabilities for this dataset.\n"""
    e=st.columns(4)
    e[0].download_button("Processed full CSV",full_processed.to_csv(index=False).encode("utf-8"),"cirruspp_processed_full.csv","text/csv",disabled=not valid_run,width="stretch")
    e[1].download_button("Pipeline JSON",safe_json({"pipeline":asdict(active_spec),"order":stage_order}),"cirruspp_pipeline.json","application/json",width="stretch")
    e[2].download_button("Audit JSON",safe_json(audit),"cirruspp_audit.json","application/json",width="stretch")
    e[3].download_button("Audit Markdown",md.encode("utf-8"),"cirruspp_audit.md","text/markdown",width="stretch")
    with st.expander("Preview audit JSON"): st.json(audit,expanded=False)

st.divider(); st.caption("CIRRUS++ · Domain-aware and auditable preprocessing for multivariate time-series data.")

# Legal notice
st.link_button("Impressum", "https://www.uni-regensburg.de/impressum")
