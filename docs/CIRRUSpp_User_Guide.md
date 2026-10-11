# CIRRUS++ User Guide

Domain-aware and auditable preprocessing for multivariate time-series data.

Documentation basis: supplied `app.py` and `cirruspp_core.py` (audit version `1.5.0-demo-ready`).

---

## 1. Purpose and scope

CIRRUS++ is a Streamlit research demonstrator for transparent, domain-aware preprocessing of multivariate time-series data. It combines file/schema inspection, feature profiling, explainable recommendations, method comparison, explicit pipeline execution, optional reference-based validation, and auditable exports.

This guide documents the supplied application code (app.py and cirruspp_core.py) and the seven tabs visible in the interface. It distinguishes intrinsic diagnostic improvements from verified recovery of the underlying physical signal. A higher intrinsic quality index is never, by itself, proof of correct repair.

The paper demonstrations use one autism eye-tracking recording and focus on signal-level profiling and outlier-method comparison. The optional Ground truth tab is a different workflow; it does not imply that an independent clean version of the real recording exists.


## 2. Access, installation, and start

Hosted application: https://cirrus.fids.ur.de/ . Source repository: https://github.com/eyetracking-data/CIRRUS_Plus_Plus . A hosted deployment may differ from a locally launched revision; check its interface before using a workflow.

Local installation: install a compatible Python environment, open the repository root, run "python -m pip install -r requirements.txt", then "streamlit run app.py". Open the local address shown by Streamlit in your browser. On Windows, the repository also provides run_windows.bat and START_CIRRUSPP.bat launchers.

Package requirements are provided in requirements.txt. The application is a research demo, not a fully validated clinical or production data-cleaning service.

- If you have no recording, select Synthetic demo in the sidebar. Otherwise choose Upload CSV/TSV and upload a CSV, TSV, or TXT table.
- Keep the original file unchanged and separately retain the pipeline JSON and audit JSON for reproducibility.
- The web server reads the uploaded data for the active session; consult your institutional data-handling rules before uploading sensitive recordings.

## 3. Sidebar: data source and signals

The sidebar controls the shared configuration for all seven tabs. Source offers Upload CSV/TSV or Synthetic demo. Upload settings allow Separator = Auto, comma, semicolon, tab or vertical bar; Decimal = Auto, period or comma. The data loader attempts to detect encoding and convert usable numeric columns.

Timestamp column can be a detected time column or Sample index. The sampling rate can be inferred from the timestamp values when possible, or entered manually when Use inferred sampling rate is disabled. Always verify units and sample rate for temporal diagnostics.

Signal columns is a multiselect of usable numeric columns. By default, identifier-like columns, time-like columns and constant numeric columns are excluded. The checkbox Include identifier-like numeric columns overrides that filtering. Exact duplicate numerical signals are detected, and default selection retains one representative; you may intentionally select both, but this can bias cross-signal metrics.

The domain profile can be auto-detected or selected manually. The eye-tracking code additionally implements optional missingness interpretations for paired gaze coordinates (0,0) and blink labels, but the hosted interface used for the paper demos did not expose this setting. Do not rely on this control for the two documented live demos.

Scaling objective can be Auto, Preserve physical units, Distance/gradient-based model or Bounded model input [0,1]. Maximum plotted points controls chart downsampling, not the underlying preprocessing.


## 4. Tab 1 — Data & domain

This overview reports source size, selected signals, sampling rate, inferred/selected domain, intrinsic quality index, quality-component breakdown and a data preview. It is the place to verify that a time-series file was interpreted correctly before comparing methods.

Quality is intrinsic and heuristic: the tool combines metrics associated with completeness, plausible values, continuity, signal stability and cross-signal consistency using domain-dependent weights. This quality index should be read as a dashboard summary, not as data fidelity against ground truth.

The code has domain-aware eye-tracking validity functions. In particular, paired zeros (both X and Y equal zero) can be treated as unavailable, and blink labels may be considered. This is a domain assumption, not universal proof that those entries are invalid. Depending on deployment version the option may not be visible.

- Confirm timestamp column, frequency and units; errors here distort gap durations and frequency diagnostics.
- Review duplicated channels and the five auto-selected signals. The user can change the selection.
- Keep the original file as source-of-record; reinterpretation should be documented in the audit.

## 5. Tab 2 — Signal profile

The Signal profile tab includes a feature-level diagnostics table and an Inspect feature selector. It computes missing fraction, missing-run lengths, robust/IQR outlier proportions, skewness and excess kurtosis, lag-one autocorrelation, local volatility, drift, spectral entropy, dominant frequency and cross-feature relationships where applicable.

The selected feature is shown through a time-series plot, distribution view and radar summary. Different metrics answer different questions: low missingness does not imply low noise, and a statistical outlier does not necessarily represent a sensor error. Lag-one autocorrelation captures adjacent-sample dependence; it is sensitive to sampling rate and dynamics.

Features absent from the sidebar or Inspect feature picker usually fail the current numeric-signal eligibility checks, are constant, or resemble identifier/time columns. The UI may also auto-select only five signals by default; the complete candidate list can contain more. If an expected field is missing, inspect the imported data type and Include identifier-like numeric columns option. Do not assume all excluded features are invalid.

- For the documented demo, select Point of Regard Left X [px] and inspect its displayed metrics and plots.
- Compare the per-feature table with pupil-diameter and other selected gaze features, where present.
- Interpret a low outlier estimate only as a result of that detector and threshold, not as proof of validity.

## 6. Tab 3 — Recommendation

This tab presents guidance separately for imputation, outlier handling, smoothing and scaling. Recommendation cards describe the suggested method and associated rule evidence, caution and alternatives. Suggestions are advisory and do not execute transformations automatically.

Recommendations may abstain if conditions do not support a stable choice. The app separates profile-derived guidance from domain-level pilot frequencies, which are not calibrated probabilities that a method is best on the current file.

The active pipeline remains user-controlled via sidebar stage selectors. A selected method may differ from a recommendation, and that manual selection is reflected in the exported audit.

- Read any cautions/guardrails before applying a method, especially on fast eye-movement events.
- Distinguish the selected active pipeline from the guidance cards.
- Use Step compare to inspect observable consequences before running the full pipeline.

## 7. Available imputation methods

None leaves the values untouched. Mean and Median replace missing values with per-column central values; they ignore temporal continuity. LOCF forward-fills missing values and backfills initial missing values; it produces plateaus and may hide transitions. Linear interpolation connects observed neighbors linearly, extending values toward boundaries where needed.

KNN imputation uses distance-weighted neighboring samples (the neighbor count is chosen by the code). MICE uses iterative multivariate imputation with a median initialization, fixed iteration cap and deterministic random state. These multivariate strategies can exploit relationships across channels, but may alter physical dynamics and are not a time-aware predictive ground-truth model.

The app processes selected numeric channels; imputation should be judged against missingness mechanism, gap duration, time resolution, and downstream use. Short gaps and long blinks are not interchangeable.


## 8. Available outlier handling methods

None makes no outlier changes. Z-score flags values more than 3 standard deviations from a column mean. IQR flags points outside Q1 − 1.5×IQR or Q3 + 1.5×IQR. MAD uses a median-centered robust standardized deviation threshold of 3.5. Winsorization clips each channel at its 1st and 99th percentiles.

Isolation Forest is a multivariate detector, implemented with 150 trees, contamination=auto and fixed random state. The current code flags observations at the row level and masks all selected channels for flagged rows. This can make its intervention substantially different from a univariate detector.

For Z-score, IQR, MAD and Isolation Forest, flagged entries are not simply deleted: they are replaced by missing values, then repaired through linear interpolation with median fallback. Winsorization clips rather than interpolates. These details matter when interpreting the overlaid outputs and change fractions.

Large but valid saccades may be detected as outliers. Therefore compare raw and processed dynamics and do not equate a higher intrinsic index with more faithful gaze reconstruction.


## 9. Available smoothing and scaling methods

Smoothing options are None, Moving median (up to nine-point kernel), Savitzky–Golay (up to 21-point window with a polynomial of degree up to three) and a third-order low-pass Butterworth filter implemented with forward/backward filtering where enough samples are available. Smoothing may suppress noise but also alter genuine short events and transition slopes.

Scaling options are None, StandardScaler (mean centering and standard-deviation normalization), MinMaxScaler (rescaling to a bounded numerical range), and RobustScaler (median/interquartile-range based). Scaling is not a cleaning operation and usually changes numerical units. Use it for an analysis objective rather than to chase a higher intrinsic quality score.

The pipeline can execute Imputation → Outlier → Smoothing → Scaling or Outlier → Imputation → Smoothing → Scaling. The first two stages therefore can change each other’s inputs and resulting flags.


## 10. Tab 4 — Step compare

Comparison mode determines whether only the selected stage runs (Isolated stage) or whether the other stages stay at their active pipeline settings (Pipeline context). Select one of imputation, outlier, smoothing, scaling; select two or more methods; choose a Feature; then click Run comparison.

CIRRUS++ displays a full-series overlay and, if alternatives actually differ, a second plot zooming to the first local region of disagreement. This is especially helpful for rare changes that are invisible on the full trace. If curves are identical, the app says so.

The comparison table provides quality_index, quality_delta, missing_rate, outlier_rate, local_volatility, cleaning_changed_% and overall_changed_%, together with a domain-weighted quality-index bar chart. The chart is an intrinsic score, not reference-based accuracy. For non-scaling stages, plots omit scaling to preserve physical units even when evaluation metrics use the full pipeline context.

Step compare does not overwrite the active pipeline or export an executed processed dataset. Use sidebar controls if you decide to apply a method, then execute it in Pipeline lab.


## 11. Tab 5 — Pipeline lab

Click Run active pipeline to deliberately execute the configured sequence. Opening this tab alone does not run anything. The app reports intrinsic quality before and after cleaning, cleaning_changed_fraction and scaling_changed_fraction separately. This separation prevents a scaler from being misreported as mass data repair.

It shows per-stage changed-cell fractions, selected feature overlays (raw versus after cleaning, before scaling), before/after feature-level metrics and a preview of processed rows. A target-bearing dataset may additionally permit an exploratory downstream before/after check with forward temporal splits.

Re-running is necessary after changing a relevant pipeline or input configuration; the Export tab checks that a valid current run exists before offering the processed full CSV.


## 12. Tab 6 — Ground truth (optional)

This module supports two different validation sources. In Inject corruption into a complete window, the app extracts a fully observed window, injects a controlled disturbance and measures how candidate pipelines reconstruct the held-out original window. It is a pseudo-ground-truth experiment: the retained original window is a reference only relative to the synthetic corruption.

Available corruptions in the code: MCAR missingness, block missingness, spikes, drift, level shift, stuck-at, high-frequency noise and clipping. Severity, sample-window length and candidates can be explored. The app ranks at most ten compact candidate pipelines with a domain-weighted reconstruction loss; lower loss is better under the chosen synthetic experiment.

The second mode, Upload clean reference for current noisy data, requires a genuinely clean reference aligned sample-for-sample with the noisy data. The code does not time-warp or resynchronize two files. The confirmation checkbox matters: matching row indices do not themselves establish matching observations.

This tab is not used by the two live paper demos. The real autism recording alone does not provide an independently observed clean gaze trajectory.


## 13. Tab 7 — Export and audit

After executing a current Pipeline lab run, Processed full CSV exports the original complete table, including columns that were not selected for processing; selected signals are replaced by the processed outputs. The original input is not overwritten.

Pipeline JSON captures the selected imputation, outlier, smoothing and scaling methods and the stage order. Audit JSON includes source name, domain, signal selection, sampling rate, validity-mode context, duplicate-channel findings, baseline and post-cleaning quality, component breakdown, recommendations, executed configuration and stage-wise effects. Audit Markdown provides a concise human-readable synopsis.

Pipeline JSON, audit JSON and audit Markdown are available as configuration/report exports. Processed full CSV is disabled until the current pipeline has been explicitly run. A comparison alone is insufficient.

Keep all outputs with the original input and note the app revision used; settings and numerical results may differ between deployment versions.


## 14. Reproducible live demo A — Signal quality profiling

Dataset: Cilia et al., Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder (2022), Figshare DOI https://doi.org/10.6084/m9.figshare.20113592 . The selected recording is named 1.csv in the local/downloaded copy used during the demo; the repository does not redistribute the recording. It contains 34,630 rows, including 21 separator rows, and is used without participant filtering or resampling.

Upload the file; verify the time column, approximately 50 Hz inferred sampling rate, eye-tracking profile and selected gaze/pupil channels. Open 2 · Signal profile. Inspect the diagnostics table and select Point of Regard Left X [px]. The documented hosted screenshot reported Missing approximately 0.06%, Robust outliers 0.00%, Skewness 0.210, Lag-1 autocorrelation 0.857. Values are tied to the captured application state.

Show the signal trace, distribution plot and radar quality view; then highlight differences between selected features, including pupil versus gaze channels. The takeaway is that completeness is only one dimension of signal quality.


## 15. Reproducible live demo B — Outlier method comparison

Use the same recording and unchanged selected input configuration. Open 4 · Step compare. Select Outlier as stage, Z-score and Isolation Forest as methods, and Point of Regard Left Y [px] as the feature. Run comparison. Use the full overlay, zoomed disagreement region where available, the method table and domain-weighted index bar chart.

The captured hosted example showed Isolation Forest quality index approximately 93.87 (delta +3.11) versus Z-score approximately 92.00 (delta +1.24). Do not describe this as reconstruction accuracy or universal method superiority: the result reflects this recording, configuration, method handling and intrinsic score.

If the two methods do not differ in a different configuration, verify the current comparison mode, feature, signal selection, validity interpretation, stage order and application revision. A separate imputation comparison on this file may show almost no visible difference because the source has few explicitly missing numeric entries.


## 16. Interpreting metrics responsibly

Missing rate describes absent numeric values under the current interpretation; it is not automatically the fraction of scientifically unusable observations. Gap durations additionally depend on sample rate. IQR/MAD outlier fractions reflect chosen statistical definitions and can flag legitimate events.

Lag-one autocorrelation describes adjacent sample similarity. Local volatility summarizes first-difference variation; smoothing tends to reduce it but may destroy valuable fast signal changes. Spectral entropy captures frequency-spread characteristics, not direct fidelity.

The intrinsic quality index is an aggregate heuristic with domain weights. Improvements can result from smoothing or removing extreme movements, even if those movements are genuine. Evaluate intervention proportions, actual plots, assumptions and downstream objectives. Only known-reference experiments can measure reconstruction error against an available reference, and synthetic tests remain assumption-dependent.


## 17. Troubleshooting and practical notes

File not accepted: verify CSV/TSV format, separator, decimal mark, encoding, and that at least one meaningful numeric signal exists. Empty signal picker: inspect column types, constant columns, identifier/time filtering and Include identifier-like numeric columns.

Wrong gap durations or spectral features: confirm timestamp units and inferred sampling frequency. Plot looks flat or methods overlap: raise Maximum plotted points, inspect the automatic difference zoom, and choose a feature with actual affected values. Scaling overlays have different units; compare their numeric results separately.

Processed export disabled: explicitly run the current pipeline in Pipeline lab first, and rerun after changing the configuration. Clean-reference validation looks suspicious: check sample-by-sample alignment; the program does not automatically align recordings.

Hosted versus local mismatch: the code archive and deployment may reflect different revisions. For replicable publication screenshots, report the interface revision and settings; do not assert a control exists in the hosted build solely because it appears in source code.


## 18. Sources, credit and reproducibility checklist

Application and code: https://github.com/eyetracking-data/CIRRUS_Plus_Plus . Hosted application: https://cirrus.fids.ur.de/ . Original research data: https://doi.org/10.6084/m9.figshare.20113592 . Dataset and redistribution information: DATASET.md in the repository.

This user guide was prepared against the supplied application sources; it does not certify the clinical validity of preprocessing outcomes, nor does it replace reading the code and examining licensing/privacy requirements. For an actual scientific record, save the source file identifier, application version/commit, selected channels, timestamp settings, domain, validity mode, stage order, methods, any random seed, pipeline JSON, and audit JSON.

