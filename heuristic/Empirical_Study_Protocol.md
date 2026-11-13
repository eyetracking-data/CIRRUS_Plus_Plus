# CIRRUS++ Empirical Study: Controlled, Domain-Aware Rule Induction for Time-Series Preprocessing

## Abstract

CIRRUS++ aims to replace opaque or hand-written preprocessing recommendations with an auditable empirical decision layer for multivariate time-series data. The benchmark combines six controlled synthetic signal families with pseudo-clean reference windows from four heterogeneous real eye-tracking datasets. Known corruptions are injected with exact masks, candidate methods are evaluated against unchanged references, and multiple error dimensions are combined under explicit domain objectives. Small decision rules are then induced from universal signal profiles and evaluated with recording-grouped cross-validation and leave-one-dataset-out holdouts. This archive documents the executed diagnostic pilot and its findings; the original standalone benchmark package and full quick-run data are not included. The quick run validates the software and reveals that the current rule learner is not yet publication-ready: imputation achieves 0.80 on the selected ETDD70 holdout but underperforms a grouped majority baseline in cross-validation, while outlier and smoothing selection achieve 0.00 on the same holdout. The artifact therefore documents both positive method-level results and negative rule-level findings, and defines the publication protocol required before automatic recommendations are integrated into CIRRUS++.

## 1. Motivation

Time-series preprocessing is not neutral. Imputation can flatten dynamics, outlier handling can remove genuine events, smoothing can suppress meaningful peaks, and scaling can alter the assumptions of downstream algorithms. A method that performs well on one signal structure may fail on another, and an isolated stage winner may not remain preferable inside a complete pipeline.

CIRRUS 1.0 addressed this problem with transparent profiling and manually formulated recommendation rules. CIRRUS++ keeps the interactive and explainable interface but moves the empirical basis into a separate, reproducible benchmark. The central design principle is that a recommendation must be traceable to:

- observable profile characteristics;
- a declared domain objective;
- controlled experiments with known references;
- validation on unseen groups;
- an applicability and uncertainty statement.

## 2. Study design

The experiment has two complementary branches.

### Synthetic branch

Six generators model selected structural properties of eye tracking, industrial sensing, energy, physiological/wearable signals, finance, and network monitoring. Each generator produces a clean multivariate series and exact masks for genuine events such as saccades, operating-state transitions, demand peaks, activity episodes, market jumps, or network incidents.

### Real branch

Four eye-tracking adapters map Autism, Desktop Activities, ETDD70/Dyslexia, and EyeT4Empathy exports to a canonical schema. The benchmark extracts fully observed six-second gaze windows without prior imputation. These windows are treated as pseudo-clean references: they provide unchanged observed values but are not claimed to represent physical sensor truth.

### Controlled degradation

The framework supports MCAR missingness, block missingness, spikes, drift, level shifts, stuck-at faults, high-frequency noise, clipping, and mixtures. Every injection returns an exact affected-position mask. The current quick run uses MCAR missingness, block missingness, spikes, and high-frequency noise. The publication configurations add severity grids and clipping; drift, level shift, and stuck-at remain profiling and extension targets until dedicated correction methods are defined.

### Candidate methods

Imputation candidates are mean, median, LOCF, linear interpolation, KNN, and MICE. Outlier candidates are no handling, z-score, IQR, MAD, winsorization, and Isolation Forest, with a fixed linear repair policy after detection. Smoothing candidates are no smoothing, moving median, Savitzky–Golay, and Butterworth filtering. Scaling is implemented but deliberately excluded from intrinsic raw-scale ranking because its value depends on downstream model requirements.

## 3. Universal profiling

The profile layer measures:

- missing rate;
- longest and 95th-percentile gap duration in seconds;
- missingness burstiness;
- IQR and MAD outlier rates;
- bias-corrected skewness;
- bias-corrected excess kurtosis;
- lag-1 autocorrelation;
- local volatility;
- drift score;
- spectral entropy;
- dominant-frequency concentration;
- mean absolute inter-channel correlation.

Kurtosis is explicitly included, but not interpreted as proof of contamination. High kurtosis can be caused by injected spikes, genuine saccades, demand peaks, market jumps, heavy tails, or mixed regimes. Its value lies in interaction with temporal persistence, local volatility, outlier rates, and domain objectives.

## 4. Evaluation

Each processed signal is compared with the clean reference on eight components:

1. reconstruction error at affected values;
2. unintended harm at unaffected values;
3. distortion of first-difference dynamics;
4. distributional distortion using Wasserstein distance, skewness, and excess-kurtosis changes;
5. spectral distortion using Jensen–Shannon distance between Welch spectra;
6. attenuation or amplification of genuine events;
7. detection loss based on one minus F1 against the exact corruption mask;
8. remaining missing-value fraction.

Each domain provides transparent weights over these components. The weighted loss defines the strict oracle method per scenario. Methods within 0.02 absolute loss of the oracle form a near-tie set; the simplest and fastest member is selected as the practical winner. Both oracle and practical winner are retained, together with regret and second-best margin.

## 5. Rule induction

The current diagnostic learner is a shallow decision-tree classifier with maximum depth four and minimum leaf size four. Dataset identity is excluded, but the diagnostic quick model still includes injected corruption type and source type. These variables are unavailable for an arbitrary uploaded series and must be removed from any deployment model. Raw sample count must also be handled carefully because it can proxy sampling rate or dataset family.

Validation includes grouped cross-validation, leave-one-dataset-out holdout, bootstrap winner support, and a comparison against grouped majority baselines. The final publication study will add rotating holdouts, top-2 accuracy, predicted-method regret, weight sensitivity, feature ablation, and abstention curves.

## 6. Current quick-run evidence

The archived quick run reports 572 successful candidate-method evaluations and no execution failures. The original `results/quick/` and `results/notebook_real_quick/` directories are not included in this repository.

### Real-data coverage

Twenty files were loaded, five per dataset, totaling 1,307,712 rows. Median inferred sampling rates are approximately 50.22 Hz for Autism, 28.57 Hz for Desktop Activities, 250 Hz for ETDD70/Dyslexia, and 120.02 Hz for EyeT4Empathy. Every file yielded a complete six-second gaze reference window.

### Method-level patterns

Across 52 combined imputation scenarios, practical winners are linear interpolation in 39 scenarios, LOCF in 12, and mean imputation in one. Linear interpolation wins 34 of 40 real scenarios. Across 26 outlier scenarios, z-score wins 17, IQR five, MAD two, and winsorization two. Across 26 smoothing scenarios, moving median wins 13, no smoothing nine, Butterworth three, and Savitzky–Golay one.

Near ties are frequent. Mean near-tie-set sizes are 1.63 for imputation, 1.46 for outlier handling, and 1.85 for smoothing. Practical-winner oracle regret is correspondingly small in the quick run. This supports a UI that exposes alternatives rather than forcing a single apparently certain answer.

### Rule-level diagnostics

The current grouped cross-validation means are approximately 0.502 for imputation, 0.200 for outlier selection, and 0.627 for smoothing. Grouped majority baselines are approximately 0.745, 0.640, and 0.493 respectively. On the ETDD70 holdout, accuracy is 0.800 for imputation and 0.000 for both outlier and smoothing.

These results mean:

- the imputation tree does not improve on the dominant-class baseline, despite its apparently good ETDD70 score;
- the outlier tree is clearly unusable;
- the smoothing tree shows some cross-validation signal but fails to transfer to ETDD70;
- current rule bundles are diagnostic artifacts only.

The current imputation tree’s first split on sample count illustrates the risk of learning dataset or sampling-rate proxies rather than a meaningful preprocessing principle.

## 7. Publication protocol

The final study will use a balanced design rather than every possible file-window combination. Equal or controlled numbers of recordings per dataset will prevent large corpora from dominating. Multiple non-overlapping windows, five seeds, and four severity levels will be used. Final holdout files and datasets will remain untouched until all generators, weights, thresholds, and predictors are frozen.

Required final analyses include:

- component-level method trade-offs;
- average ranks and effect sizes;
- method-by-severity interactions;
- runtime and failure rates;
- rotating leave-one-dataset-out validation;
- top-1, top-2, and regret metrics;
- majority and global-method baselines;
- deployment-safe feature ablation;
- explicit contribution of kurtosis;
- domain-weight sensitivity;
- applicability and abstention evaluation;
- a restricted complete-pipeline interaction experiment.

Automatic CIRRUS++ integration is gated on beating simple baselines, stable holdout regret, deployment-safe predictors, weight stability, and empirical support in the query profile region.

## 8. Reproducibility and paper integration

This repository includes documented exploratory rule-induction notebooks, methodological reports and the CIRRUS++ application code. It also contains the evaluation scripts and complete candidate table for the five non-eye-tracking synthetic domains. The original standalone `cirruspp_benchmark` package, full rule-pilot run directories, trained rule bundles, and the separate eye-tracking raw evaluation table are not supplied. Accordingly, the rule-induction notebooks serve as transparent research records rather than a complete independently executable reproduction. The negative grouped-validation results are retained and do not establish a validated deployed learned recommender.
