# CIRRUS++ Heuristic Development and Evidence Status

This directory documents **two related but distinct research components**. They must not be conflated in the paper.

1. **Domain-specific weighting of the intrinsic quality index:** See [`Quality_Index_Domain_Weighting_Rationale.pdf`](Quality_Index_Domain_Weighting_Rationale.pdf). This report provides an evidence-informed, retrospective explanation of the six existing domain-weight vectors in terms of dissertation research, eye-tracking evidence, and domain-specific preservation objectives. It is explicitly an **ex-post rationale**, **not** evidence that the weights were originally learned through exhaustive statistical optimization.
2. **Experimental development of method recommendations:** See [`Rule_Induction_Pilot_Executed.ipynb`](Rule_Induction_Pilot_Executed.ipynb), [`Master_Benchmark_Template.ipynb`](Master_Benchmark_Template.ipynb), [`Empirical_Study_Protocol.md`](Empirical_Study_Protocol.md), and [`Publication_Run_Plan.md`](Publication_Run_Plan.md). They document an approach for learning small, auditable method-selection rules for imputation, outlier handling, and smoothing from controlled corruption experiments.

The one-page overview [`Heuristic_Development_and_Evidence_Status.pdf`](Heuristic_Development_and_Evidence_Status.pdf) summarizes the provenance and scientific limits of these materials.

## Experimental workflow

**Domain and objective** → synthetic series with known clean reference or fully observed real-data window → controlled corruption at fixed seeds/severities → evaluate competing methods on the *same* degraded input → measure reconstruction error and domain-specific preservation criteria → identify suitable or near-equivalent candidate methods → train shallow decision trees using observable profile features → assess performance on held-out recordings and datasets.

Candidate features include missingness, gap durations, autocorrelation, variance, robust outliers, skewness, and kurtosis. The target is **not simply the smallest raw training-set error**; plausibility, preservation goals, and unwanted distortions should also be considered.

## Current status and limitations

- The documented **pilot comprises 572 successful candidate-method evaluations**: 132 synthetic cases and 440 cases using real eye-tracking windows. It demonstrates technical feasibility and exposes weaknesses in the rule-induction approach. It is **not sufficient external validation** of a universal recommender.
- Some experimental rules performed worse than simple majority-class baselines under grouped validation. They must **not** be advertised as robust across domains.
- **The deterministic heuristic recommendations currently implemented in `../cirruspp_core.py` are distinct from the pilot's experimental learned rules.** These notebooks do not demonstrate that learned rule models have replaced the deployed recommendation logic.
- Some preserved notebooks import a separate `cirruspp_benchmark` package and configurations that **are not included** in this assembled release. Accordingly, they are methodological research artifacts **rather than standalone executable notebooks**.
- No real participant recordings are redistributed.

## Distinction from the quality-index evaluation

The separate [`../evaluation/`](../evaluation/) study tests whether the intrinsic quality index helps screen groups of preprocessing candidates under controlled synthetic corruption. That study does **not** validate the learned method-selection rules. Stronger claims about such rules require group-wise cross-validation, untouched dataset holdouts, and appropriate baseline comparisons.
