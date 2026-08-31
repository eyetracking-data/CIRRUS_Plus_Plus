# CIRRUS++

**Domain-aware, transparent, and auditable preprocessing for multivariate time-series data.**

CIRRUS++ is an interactive Streamlit demonstrator for inspecting and challenging preprocessing decisions rather than hiding them behind an automatic cleaning step.

The current demo build supports:

- CSV/TSV upload and automatic schema inspection
- timestamp and signal selection
- exact duplicate-channel detection
- domain-aware eye-tracking validity interpretation
- signal profiling and an explainable intrinsic quality index
- profile-specific preprocessing guidance
- separate domain-level pilot context
- isolated-stage and pipeline-context method comparison
- complete-pipeline execution with cleaning/scaling separated
- feature-level metric-change explanations
- controlled pseudo-ground-truth validation
- optional exploratory downstream validation
- processed-data, pipeline, and audit export

## Quick start on Windows

1. Download or clone the repository.
2. Double-click `run_windows.bat`.
3. On first start, CIRRUS++ creates a local `.venv` and installs the required packages.
4. The application opens in the browser at `http://localhost:8501`.

`START_CIRRUSPP.bat` is identical and can be used as an alternative launcher.

The launcher searches several common Python installations and can use a `python_path.txt` file if needed.

## Manual start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Interface

CIRRUS++ is organized into seven functional tabs:

1. **Data & domain** — load data, map signals, inspect duplicates, sampling rate, domain profile, and validity semantics.
2. **Signal profile** — inspect missingness, gap structure, outlier diagnostics, distributional properties, temporal dynamics, spectral measures, and cross-signal relations.
3. **Recommendation** — inspect profile-specific guidance, alternatives, rule traces, guardrails, abstention, and separately displayed domain-level pilot context.
4. **Step compare** — compare methods in isolation or inside the configured pipeline. A focused zoom highlights the first region where outputs actually differ.
5. **Pipeline lab** — deliberately execute the complete pipeline and inspect cleaning changes, scaling transformations, feature-level metric deltas, processed values, and optional downstream effects.
6. **Ground truth** — inject controlled corruption into a complete reference window or compare noisy data with a supplied clean reference.
7. **Export** — export the complete processed table, pipeline configuration, and audit information.

## Eye-tracking validity interpretation

For the demo eye-tracking format, CIRRUS++ can distinguish between raw numerical completeness and domain-aware gaze availability.

The domain-aware mode can interpret paired gaze coordinates `(0,0)` and compatible blink labels as unavailable gaze observations. A single `x=0` or `y=0` is not automatically treated as missing.

The original uploaded file is not modified. The setting changes the analytical representation used for profiling and preprocessing.

CIRRUS++ also detects exact duplicate numeric channels and keeps one representative in the automatic signal selection to avoid unintentionally inflating cross-signal measures.

## Two live demonstration scenarios

The current demo paper uses one real autism eye-tracking recording as a continuous example.

### Scenario A — Missing, or a measurement?

The audience first sees a nearly complete numerical file and decides whether gaze validity should remain raw or be interpreted using domain semantics.

Prepared-example values:

- explicit missingness: approximately **0.06%**
- paired `(0,0)` gaze values: approximately **29.11%**
- blink labels: approximately **26.54%**
- domain-aware unavailable gaze: approximately **30.23%**
- intrinsic quality index: approximately **88.2 → 72.5**

The data do not change; their interpretation does.

### Scenario B — Which imputation should we trust?

CIRRUS++ retains a fully observed window from the same recording, injects a controlled missing block, and compares candidate imputations such as Linear, LOCF, and Mean against the retained reference.

The scenario illustrates that profile-specific guidance and domain-level pilot evidence are informative but can be challenged through controlled validation.

## Scientific interpretation

CIRRUS++ intentionally separates:

**measurement → interpretation → guidance → validation → decision**

An intrinsic score is not ground truth, a statistical outlier is not automatically a fault, and a recommendation is not a guarantee of optimal reconstruction.

## Repository integration

This package contains the current application files for the final demo-paper build. Existing empirical-study folders in the public repository can remain unchanged. Replace/update the root application files with the files in this package and keep the existing research artifacts, notebooks, pilot tables, tests, and documentation alongside them.

## Paper figures

`paper_figures/` contains:

- `architecture_cirruspp.png` — decision architecture
- `demo_scenarios.png` — the two interactive demo scenarios

## Files

```text
app.py
cirruspp_core.py
requirements.txt
run_windows.bat
START_CIRRUSPP.bat
.streamlit/config.toml
README.md
DEMO_SCRIPT.md
CHANGELOG.md
paper_figures/
```

## Scope and limitations

CIRRUS++ is a demonstrator, not a production-ready universal preprocessing recommender. Current real-data empirical evidence is strongest for eye tracking. Domain weights and the intrinsic quality index are transparent analytical heuristics. Controlled complete windows provide pseudo-ground truth rather than certified physical truth, and validity rules depend on dataset semantics.

These limitations are intentionally reflected in the interface through explicit explanations, pilot/unstable states, guardrails, alternatives, and user override.
