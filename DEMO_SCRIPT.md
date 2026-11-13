# Two paper demonstration scenarios — ETDD70

**Application:** https://cirrus.fids.ur.de/  
**Dataset:** https://doi.org/10.5281/zenodo.13332134

The illustrated screenshots/paper refer to **subject 1021, task T1**, right-eye gaze X/Y and pupil diameter, with 18,472 selected observations. These values are tied to the paper configuration and must be matched to the hosted application version for reproduction.

## Scenario A: Interactive guided pipeline selection (no ground truth)

1. Import the ETDD70 recording and select the appropriate timestamp unit, task/recording, signal channels and eye-tracking domain.
2. Examine Data overview and Signal profile; inspect component diagnostics underlying the weighted quality index.
3. Compare heuristic and optional AI guidance. In the illustrated configuration the AI suggested winsorization and robust scaling, and no missing-value imputation or smoothing. AI suggestions are never applied automatically.
4. Inspect Method comparison, configure Pipeline lab and explicitly run the chosen transformations; compare visual/diagnostic effects.

## Scenario B: Controlled reference validation

1. Select a fully observed window for the chosen channel and retain it as pseudo-reference.
2. Introduce controlled missingness and compare candidate preprocessing pipelines in Reference validation.
3. Inspect reconstruction RMSE together with **coverage** and original/reconstructed signal traces.
4. Illustrated paper outcome: P2 (RMSE 1.572; 90% coverage) vs P8 (RMSE 2.833; 100% coverage), in the published screenshot configuration. These RMSE values are not directly rankable without accounting for different coverage.
5. Export candidate settings and an audit record.

**Important:** Scenario B's candidates are not automatically the same as the AI suggestion in Scenario A. These examples do **not** establish a global optimum for the real recording. The six-domain synthetic study is independent of this demonstration.
