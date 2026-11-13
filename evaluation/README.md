# Cross-domain synthetic quality-index evaluation

**Core report:** `Quality_Index_Cross_Domain_Report.pdf`. The implemented experiment comprises five 480-point synthetic families (industrial, energy, physiological, finance and network) plus a separately run 1,200-point eye-tracking family.

- Five domains: `results_all.csv` (33,600 rows), `summary_all.csv`, `baselines_all.csv`, `components_all.csv`, and additional aggregated/diagnostic outputs.
- Source programs: `cirrus_eval.py`, `run_full_five_domains.py`, `orchestrate.py`, `summarize_and_package.py`, and the parity test.
- Five-domain executed results: `Five_Domains_Executed_Analysis.ipynb`, charts, and `README_Original_Benchmark.md`.
- The **separate eye-tracking raw 6,720-row table is not provided** with the currently available files. Therefore the 40,320 total from the report is documented but not completely represented by raw records here.

## Five-domain reproduction

Install compatible versions of numpy, pandas and scikit-learn as required by the scripts; examine original README and `metadata.json` for exact versions and commands. Run from `evaluation/`:

```bash
python run_full_five_domains.py --domain industrial --seed 42 --n 480 --output runs
python orchestrate.py
python summarize_and_package.py
```

Some scripts assume the working directory and require time and memory. They generate artificial data and do not download real participant records.

## Validation status

- Original synthetic benchmark matched its earlier overlapping results (`phase1_reproduction_1875_rows.csv`).
- **Direct parity with the newer production module remains open.** Run the provided test on the root module, examine failures and investigate differences before asserting reproduction of deployment-time quality scores:

```bash
python test_upstream_parity.py --module ../cirruspp_core.py
```

- The index is a domain-weighted *intrinsic diagnostic*, not a guaranteed proxy for low RMSE. The top-scoring candidate was seldom the RMSE optimum.
- All 40,320 reported outputs are correlated transformations of fewer independent scenarios; no universal statistical guarantee is implied.
