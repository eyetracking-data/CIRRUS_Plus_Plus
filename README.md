# CIRRUS++ — Research demonstrator and reproducibility materials

**CIRRUS++** is an interactive, domain-aware and auditable preprocessing system for multivariate time series. It combines diagnostic profiling, explicit pipeline execution, heuristic and optional AI-assisted guidance, method comparison, and reference validation. Its six-component intrinsic quality index is a **screening heuristic**, not a guarantee of reconstruction accuracy.

- **Live application:** https://cirrus.fids.ur.de/
- **Repository:** https://github.com/eyetracking-data/CIRRUS_Plus_Plus
- **Reference dataset ETDD70:** https://doi.org/10.5281/zenodo.13332134
- **Study paper:** EDBT 2027 demo submission (see manuscript for full bibliography)

## Where to find everything

| Location | Content |
|---|---|
| Root (`app.py`, `cirruspp_core.py`, `*_view.py`, adapters, etc.) | Current modular Streamlit app, preprocessing engine, quality-index and heuristic functions, AI integration. |
| `tests/` | Regression tests for heterogeneous CSV/TSV inputs and recording selection. |
| `heuristic/` | **How guidance and weights were developed**: original weight-rationale PDF, a concise one-page PDF on heuristic provenance, empirical rule-induction pilot notebook, study protocol, publication-run plan, and a concise status report. |
| `evaluation/` | **Quality-index evaluation**: cross-domain PDF, evaluation Python scripts, 33,600-row raw candidate table for five synthetic domains, summary tables, analysis notebook, and figures. |
| `DATASET.md` and `DEMO_SCRIPT.md` | ETDD70 provenance and two paper demo scenarios. |

## Run CIRRUS++

Python 3.11–3.13 is preferred (not every platform and Python version has been tested):

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, `run_windows.bat` is an optional launcher; check its behavior before allowing installations. Run basic import regression tests with:

```bash
python -m unittest discover -s tests -v
```

**Optional AI:** Enter your own OpenAI API key via the app settings. The repository contains **no API keys**. Users choose whether to send diagnostics only or also sample values; suggestions are not automatically executed. Treat outputs as proposals subject to user review.

## Application tabs

1. **Data overview**: inspect selected input/recording, timestamp semantics, domain profile and intrinsic index.
2. **Signal profile**: inspect missingness, outliers, distribution and temporal diagnostics.
3. **Recommendation**: compare heuristic guidance and optional AI-assisted proposals.
4. **Method comparison**: apply alternative single-stage methods in isolation or pipeline context, without changing the active pipeline.
5. **Pipeline lab**: configure and explicitly execute a stage sequence.
6. **Reference validation**: optionally compare to aligned clean data or inject known corruption into complete windows; this is a controlled test, not a claim that entire real recordings are error-free.
7. **Export**: download processed data, configuration and audit information.

## Scientific status and limitations (important)

**Do not conflate the two distinct experimental strands:**

- **Heuristic/rule development** (`heuristic/`): controlled candidate-method experiments and a **pilot** of interpretable rule induction. A pilot with 572 successful method evaluations documents how rules *can* be derived and validated. It **does not establish** that a final learned recommender is validated or replaces all active built-in recommendations. Existing domain weights are explained by a **retrospective evidence-informed rationale**, not historically fitted through optimization.
- **Intrinsic quality-index evaluation** (`evaluation/`): an exploratory **synthetic** benchmark described in the PDF as 40,320 pipeline evaluations across six domains. **Only the 33,600 underlying pipeline records for the five non-eye-tracking domains are included here as a complete CSV**; the separate 6,720 eye-tracking evaluations are reported in the PDF but their individual rows are not contained in these provided artifacts. The evaluation uses a **locally transcribed** implementation; direct parity with the included, newer `cirruspp_core.py` remains unverified. Do not describe all 40,320 results as fully reproducible from this package without supplying the missing eye-tracking run and verifying parity.

The underlying synthetic runs use one limited corruption family (random missing values and spikes), five seeds and 0/5/10/15% severity. The output-equivalent pipeline configurations are correlated. A higher intrinsic score may screen out poor options but does not ensure the minimum reconstruction error.

The `heuristic/` notebooks were recovered from the original research materials. Some notebook cells rely on the separate `cirruspp_benchmark` source package and configurations **not present** in this assembled folder. They are preserved as documented research artifacts, **not claimed to run standalone**. The application code and the five-domain evaluation scripts are separate runnable components, subject to installed dependencies. The generation/aggregation pipeline is documented in `evaluation/README.md`.

**Data ethics and licensing:** no raw ETDD70 participant files or other real recordings are redistributed. Download them from the original archival source under its terms. The repository contains no virtual environment and no runtime secrets. No code license is included; add one if external reuse is intended.

