# CIRRUS++ — two live demonstration workflows

Both paper examples use the **same real autism eye-tracking recording**, locally identified as `1.csv`. See [DATASET.md](DATASET.md) for the source. The dataset is not bundled with the repository. Do not claim that the intrinsic quality score verifies true gaze accuracy.

## Before starting

1. Open https://cirrus.fids.ur.de/ or start `streamlit run app.py` locally.
2. In **Source**, select **Upload CSV/TSV** and upload the selected autism recording (`1.csv` in the local copy used for the paper).
3. Check the sidebar. Select the eye-tracking domain profile if necessary, map the recording's time column, and select gaze and pupil signals.
4. Check that the file has **34,630 rows**, the analysis shows approximately **50 Hz**, and note the active interpretation of gaze validity. Different app versions/validity settings can change selected metrics. The hosted screenshots used in the paper are the reference for the illustrated outputs.

## Demo 1 — Interactive Signal Quality Profiling

1. Open **2 · Signal profile**.
2. Inspect the per-feature diagnostics table. Contrast selected gaze-coordinate and pupil-diameter channels; do not infer that the same missingness implies identical overall quality.
3. In **Inspect feature**, select **Point of Regard Left X [px]**.
4. Point out the displayed completeness, robust outlier rate, skewness and lag-one autocorrelation; inspect the **signal**, **distribution** and **radar** views.
5. Example values from the captured hosted configuration: about **0.06% explicit missing**, **0.00% robust outliers**, **0.210 skewness**, and **0.857 lag-one autocorrelation**. Feature-level quality indices differ across signals.

**Takeaway:** quality is multidimensional, feature-dependent and descriptive; low missingness is not proof that all observations are valid.

## Demo 2 — Interactive Outlier Method Comparison

1. Open **4 · Step compare**.
2. Choose the **outlier** stage. For the paper's illustration, select **Z-score** and **Isolation Forest**. Use **Point of Regard Left Y [px]** in the **Feature** selector.
3. Check **Comparison mode** and other active pipeline settings; method effects and scores depend on this context.
4. Click **Run comparison**. Inspect the raw-versus-processed overlay, the focused difference zoom where available, the comparison table and the quality-index chart.
5. In the captured demonstration configuration, **Isolation Forest ≈ 93.87 (Δ +3.11)** and **Z-score ≈ 92.00 (Δ +1.24)** on the *intrinsic* quality index. Values may vary with data selection, parameters, interpretation and software version.

**Takeaway:** the techniques produce different interventions and diagnostic scores, **not** a verified winner against a clean ground-truth recording. A high-amplitude gaze movement can be real.

## Optional continuation (not one of the two paper demos)

Use **5 · Pipeline lab** to execute a chosen full preprocessing pipeline explicitly, inspect stage effects, then **7 · Export** to obtain the processed CSV and audit files. **6 · Ground truth** is a separate optional module for controlled synthetic corruption or truly aligned clean-reference files; it is **not** used to make claims about actual ground truth for the autism recording.

Screenshots: [Signal profile](docs/screenshots/demo1_signal_profile.png) · [Outlier comparison](docs/screenshots/demo2_outlier_comparison.png)
