# 12. Publication-Run Plan

## Why the full run must be staged

A literal Cartesian product over every available file, three windows, five seeds, four severities, and all methods can produce millions of method evaluations. This is computationally wasteful and may overweight large datasets.

The final design should balance datasets and participants rather than letting corpus size determine the result.

## Recommended three-tier plan

### Tier 1 — Calibration pilot

Purpose:

- validate all adapters;
- inspect achieved corruption severity;
- tune no model on final holdouts;
- estimate runtime and memory.

Suggested size:

- five files per dataset;
- up to three windows per file;
- three seeds;
- two severity levels.

### Tier 2 — Balanced main benchmark

Purpose:

- fit rules and analyze method suitability.

Suggested sampling:

- equal number of independent recordings per dataset;
- up to three non-overlapping windows per recording;
- five seeds;
- all four severity levels;
- predefined train datasets/files.

The exact file count should be chosen from a power/stability analysis and computational budget, then frozen before examining holdout performance.

### Tier 3 — Untouched validation

Purpose:

- final leave-one-dataset-out evaluation;
- confidence and abstention calibration.

Requirements:

- files not used during implementation decisions;
- no threshold or weight adjustment after evaluation;
- report all results, including zero-accuracy or negative findings.

## Required analyses

### Method-level

- component-wise loss distributions;
- average rank and critical differences;
- oracle margins;
- runtime and memory;
- failure rates;
- method-by-severity interaction.

### Rule-level

- grouped top-1 and top-2 accuracy;
- regret of predicted method;
- coverage-risk curve under abstention;
- rotating leave-one-dataset-out results;
- confusion matrices;
- majority and global-method baselines;
- deployment-feature ablation.

### Weight sensitivity

- recommendation support under perturbed weights;
- scenario-level instability;
- comparison of domain-aware and equal-weight rankings.

### Feature ablation

At minimum compare:

1. missingness/outlier/skewness only, approximating CIRRUS 1.0;
2. plus kurtosis;
3. plus temporal metrics;
4. plus spectral metrics;
5. plus multivariate correlation;
6. full deployment-safe profile.

This directly quantifies whether kurtosis and the additional CIRRUS++ metrics improve recommendations.

## Recommended deployment acceptance criteria

A stage model should not be integrated as an automatic recommendation unless it:

- beats a training-fold majority/global-method baseline in grouped validation;
- achieves acceptable regret on every rotated holdout;
- shows stable recommendations under weight perturbation;
- uses only deployment-available predictors;
- has adequate support in the relevant profile region;
- can abstain on out-of-distribution profiles.

Until then, CIRRUS++ should display benchmark comparisons rather than claim an empirically reliable automatic choice.
