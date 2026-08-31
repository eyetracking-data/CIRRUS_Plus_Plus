# CIRRUS++ live demo script

## Dataset

Use the prepared autism eye-tracking CSV used in the demo paper.

Expected basic properties:

- 34,630 rows
- approximately 50 Hz
- four automatically selected non-duplicate signals in the final build

## Scenario A — Missing, or a measurement?

1. Load the autism recording.
2. Start with raw numeric interpretation.
3. Point out the very low explicit missingness.
4. Open the eye-tracking validity explanation.
5. Show the paired `(0,0)` gaze frequency and blink-label frequency.
6. Ask the audience whether these should remain numerical observations or be interpreted as unavailable gaze.
7. Activate domain-aware validity.
8. Reveal the changed unavailable-gaze estimate and intrinsic quality index.

Prepared values used in the paper figure:

- explicit missing: ~0.06%
- paired `(0,0)` gaze: ~29.11%
- blink labels: ~26.54%
- domain-aware unavailable gaze: ~30.23%
- intrinsic quality index: ~88.2 → ~72.5

Key message:

> The data stay the same; the interpretation changes.

## Scenario B — Which imputation should we trust?

1. Keep the same autism recording.
2. Open Ground Truth.
3. Select controlled block missingness.
4. Use a complete reference window supported by the current dataset.
5. Compare at least Linear, LOCF, and Mean.
6. Ask the audience to choose a method before revealing the result.
7. Run controlled validation.
8. Inspect both the quantitative loss and the visible reconstruction.

Key message:

> Guidance must be challengeable.

Do not present a profile rule, pilot frequency, intrinsic score, or one controlled run as universal proof that a method is best.

## Closing

Show that the final user choice can be accepted, changed, disabled, or overridden, and that the resulting pipeline and audit context can be exported.
