"""User-facing reference for the implemented preprocessing methods."""

import pandas as pd
import streamlit as st

from recommendation_view import STAGE_COLORS


def render_method_guide(*, fs, time_processing):
    st.button("← Back to analysis", key="close_method_guide",
              on_click=st.session_state.update, kwargs={"method_guide_open": False})
    st.title("Preprocessing methods")
    st.write("Explore what each method does and the settings CIRRUS++ actually uses. These settings apply to both heuristic and AI-suggested pipelines.")
    st.caption("None skips a stage. This page explains the methods; choose and run your pipeline in Pipeline lab.")

    def section(stage, title, rows):
        color = STAGE_COLORS[stage]
        st.markdown(f'<h3 style="border-left:4px solid {color};padding-left:12px;margin-top:24px">{title}</h3>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows, columns=["Method", "How it works", "Implementation settings"]),
                     hide_index=True, width="stretch", height="content")

    section("imputation", "Missing value imputation", [
        ("Mean", "Replaces missing values with the signal mean.", "Calculated separately for each signal."),
        ("Median", "Replaces missing values with the signal median.", "Calculated separately for each signal."),
        ("LOCF", "Carries the last observed value forward within the permitted gap duration.", "Leading and trailing gaps remain missing."),
        ("Linear interpolation", "Connects neighbouring observations using their actual timestamp distances.", "Interpolates only between observed endpoints."),
        ("KNN", "Estimates missing values from similar multivariate rows, which may be far apart in time.", "2–7 neighbours depending on row count; distances use internally standardized signals. Estimates return to original units; observed values stay unchanged."),
        ("MICE", "Iteratively predicts each incomplete signal from the others.", "Up to 8 iterations; median initialization; random seed 42."),
    ])
    maximum = time_processing["max_imputation_gap_seconds"]
    if maximum is None:
        st.info("Gap duration is unlimited. All imputation methods leave leading/trailing gaps missing. Enable Limit gap duration in Time processing to restrict which internal gaps can be filled.")
    else:
        st.info(f"Current maximum gap to fill: {maximum:g} s. All imputation methods leave leading/trailing gaps and gaps whose bounding observations exceed this duration missing. Change this limit in Time processing in the sidebar.")

    section("outlier", "Outlier treatment", [
        ("Z-score", "Flags large deviations from the signal mean.", "Absolute Z-score > 3; population standard deviation (ddof=0)."),
        ("IQR", "Flags values outside the interquartile fences.", "Below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR."),
        ("MAD", "Flags large deviations from the median using a robust modified Z-score.", "Absolute modified Z-score > 3.5."),
        ("Winsorization", "Clips extreme values instead of replacing them with missing values.", "1st and 99th percentiles, separately for each signal."),
        ("Isolation Forest", "Detects anomalous multivariate rows; every selected signal in a flagged row is marked missing.", "150 trees; contamination=auto; random seed 42."),
    ])
    st.caption("Z-score, IQR, MAD and Isolation Forest replace flagged values with NaN. Subsequent imputation can fill them together with existing gaps, subject to the configured gap limit. Without subsequent imputation they remain missing. Statistical flags can also correspond to real signal events.")

    section("smoothing", "Smoothing", [
        ("Moving median", "Applies a local median filter.", "Window of up to 9 samples; nearest endpoint values extend each segment at its boundaries."),
        ("Savitzky–Golay", "Fits local polynomials to smooth the signal.", "Odd window of up to 21 samples; polynomial degree 3."),
        ("Butterworth", "Attenuates high-frequency variation with a low-pass filter.", "Third-order filter; cutoff 0.2 × sampling rate; forward/backward application."),
    ])
    st.caption(f"Current sampling rate: {fs:g} Hz. The Butterworth cutoff is {0.2 * fs:g} Hz. Sample-count windows cover different time spans at different sampling rates.")
    st.caption("Smoothing preserves remaining missing values and processes finite segments separately. Acquisition interruptions longer than the gap limit split segments; segments with fewer than 9 observations remain unchanged. For irregular timestamps, internal resampling must be explicitly enabled in Time processing. Filtered values are returned at the original observation times.")

    section("scaling", "Normalization", [
        ("StandardScaler", "Centres by the mean and scales by standard deviation.", "Calculated separately for each signal."),
        ("MinMaxScaler", "Maps each signal's observed range to a common range.", "Target range [0, 1]."),
        ("RobustScaler", "Centres by the median and scales using the interquartile range.", "Median and 25th–75th percentile range, separately for each signal."),
    ])
    st.caption("Normalization precedes optional smoothing. A subsequent filter can change the final range or standard deviation. Pipeline plots show the cleaning effect in original units; the processed CSV retains the actual normalized output.")
