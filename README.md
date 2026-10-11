# CIRRUS++

**Domain-aware and auditable preprocessing guidance for multivariate time-series data.**

CIRRUS++ is a Streamlit research demonstrator for data-quality profiling, explainable preprocessing recommendations, stage-wise comparisons, pipeline execution and audit export. Its intrinsic quality index is a **heuristic**: a higher value does **not** prove faithful reconstruction.

- **Live application:** https://cirrus.fids.ur.de/
- **Source repository:** https://github.com/eyetracking-data/CIRRUS_Plus_Plus
- **Two tested demonstration workflows:** [DEMO_SCRIPT.md](DEMO_SCRIPT.md)
- **Dataset provenance:** [DATASET.md](DATASET.md)
- **Detailed guide:** forthcoming (`docs/CIRRUSpp_User_Guide.pdf`; not included in this release)

## Installation

Requires Python 3.11–3.13 (the dependencies also permit compatible newer versions, but these are not verified here).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows users can alternatively double-click `run_windows.bat` (or `START_CIRRUSPP.bat`): it attempts to locate or install Python, create `.venv`, install requirements and start Streamlit. Note that this may install software through `winget`; inspect the script if you prefer manual installation. The browser generally opens at `http://localhost:8501`.

## Using CIRRUS++

Load your own numeric CSV/TSV/TXT table via **Upload CSV/TSV**, or select **Synthetic demo** (which needs no external dataset). In the sidebar, review the timestamp, selected signals, inferred sampling rate, domain profile, optional eye-tracking validity interpretation, scaling objective and active pipeline.

| Tab | Purpose |
| --- | --- |
| **1 · Data & domain** | Source/schema inspection, domain and intrinsic quality overview. |
| **2 · Signal profile** | Per-feature completeness, outliers, distribution, temporal and spectral diagnostics; signal, distribution and radar views. |
| **3 · Recommendation** | Profile-specific stage recommendations, conditions, guardrails and separate pilot evidence. |
| **4 · Step compare** | Run isolated-stage or pipeline-context method comparisons and inspect traces, difference zoom and metric deltas. |
| **5 · Pipeline lab** | Explicitly execute the selected stage sequence; inspect cleaning and scaling separately. |
| **6 · Ground truth** | Optional controlled corruption of a complete window or evaluation against a user-supplied aligned clean reference. **Not part of the two paper demos.** |
| **7 · Export** | Processed full CSV (after execution), pipeline JSON, audit JSON and audit Markdown. |

The active pipeline supports imputation, outlier handling, smoothing, scaling and an order choice for the first two stages. **Step compare does not modify the active pipeline.**

## Paper demonstrations

The paper's two illustrated workflows use the same real autism eye-tracking recording:

1. **Interactive signal quality profiling:** inspect per-feature metrics, selected gaze/pupil features, signal/distribution/radar plots.
2. **Interactive outlier method comparison:** select the **outlier** stage, methods **Z-score** and **Isolation Forest**, and the feature **Point of Regard Left Y [px]**. Compare overlays and intrinsic metrics; interpret them with caution.

See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for details and [docs/screenshots/](docs/screenshots/) for screenshots from the hosted demonstration. Neither demo uses a clean gaze reference or asserts ground-truth accuracy.

## Data licensing and research limitations

The original real-data recording is **not redistributed** in this repository. See [DATASET.md](DATASET.md) for the public dataset DOI and file identification. This tool is a research demonstrator, not a validated universal recommender. Statistical outliers may correspond to real eye movements. Profile-specific and pilot suggestions can be unstable, and controlled corruption yields only pseudo-ground truth.

## Repository files

- `app.py` — Streamlit interface
- `cirruspp_core.py` — preprocessing, profiling, recommendations and validation functions
- `requirements.txt` — Python requirements
- `run_windows.bat`, `START_CIRRUSPP.bat` — Windows launchers
- `DEMO_SCRIPT.md`, `DATASET.md` — reproducible demonstration instructions and dataset provenance
- `paper_figures/architecture_cirruspp.png` — architecture figure
- `docs/screenshots/` — screenshots of the hosted interface used in the demos

**Code license:** no license file was present in the supplied source archive. Do not assume permission to reuse or redistribute the code beyond what applicable law permits; add an explicit license if intended.
