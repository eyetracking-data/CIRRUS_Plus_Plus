# ETDD70 dataset provenance and reproducibility

The EDBT 2027 demonstration uses **ETDD70**, an eye-tracking dataset from reading tasks.

- **Official dataset archive:** https://doi.org/10.5281/zenodo.13332134
- **Acquisition:** eye tracking, 250 Hz reported for the relevant recordings.
- **Illustrated example (paper):** subject 1021, task T1 (syllable reading), 18,472 observations in the selected analysis configuration; right-eye gaze X/Y and pupil diameter are discussed.
- **Use:** two complementary interactive scenarios: guidance without reference, and controlled corruption/reference validation.

The paper figures and numerical examples depend on exact signal selections, validity settings, preprocessing settings and app version. For exact replication, verify them against the original input. ETDD70 is a real recorded dataset; **a fully observed window used in a controlled corruption test is only a pseudo-reference**, not independent proof of gaze accuracy. The synthetic six-domain evaluation is separate from the real ETDD70 demonstration.

Raw participant files are **not redistributed** with this repository. Retrieve the dataset from Zenodo under its terms. Exact reproduction also requires identifying the archival filename, checksum and participant/task configuration.
