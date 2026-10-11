# Source data and provenance

The published demo uses an autism eye-tracking recording from the dataset by Cilia et al. (2022), *Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder*.

**Official DOI:** https://doi.org/10.6084/m9.figshare.20113592

The recording was called **`1.csv` in the authors' downloaded/local copy**. We have not independently established that this is the original archival filename on Figshare; do not present the local path `Eye-tracking Output/1.csv` as a repository file. The working copy used for the paper has **34,630 rows including 21 separator rows**, approximately **50 Hz** based on a median timestamp interval near 20 ms, and eye-tracking columns such as `Point of Regard Left X [px]`, `Point of Regard Left Y [px]`, and `Pupil Diameter Left [mm]`.

### Reproduce

1. Follow the DOI and retrieve the original dataset, subject to its terms.
2. Identify the matching recording based on the structural identifiers above (including row count and signal column names), and upload it to CIRRUS++ without participant filtering or resampling.
3. Follow [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

This repository **does not include the original participant data**. All numeric figures and results are descriptive for the specified analysis configuration. In particular, the paper's intrinsic quality indices are not ground-truth validation statistics.
