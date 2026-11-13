"""Targeted regressions for real CSV imports and nullable outlier markers.

Run from the project folder: python -m unittest discover -s tests -v
The UI itself needs Streamlit; a tiny module stub keeps these pure plotting
and data tests runnable in environments without the Streamlit dependency.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import streamlit  # noqa: F401
except ImportError:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

from cirruspp_core import (
    DOMAIN_CONFIG,
    auto_convert_numeric,
    canonical_frame,
    preferred_timestamp_column,
    profile_table,
    validate_timestamps,
)
from signal_profile_view import _signal_plot
from recording_selection import detect_recording_blocks, select_recording


class RealCSVRegressions(unittest.TestCase):
    def test_mixed_participant_labels_are_preserved_on_import(self):
        raw = pd.DataFrame({"Participant": ["50", "52", "Unidentified(Neg)",
                                            "Unidentified(Pos)", "49"],
                            "RecordingTime [ms]": [100., 120., 140., 160., 180.],
                            "gaze_x": ["100", "120", "-", "130", "125"]})
        result = auto_convert_numeric(raw)
        self.assertEqual(result["Participant"].tolist(), raw["Participant"].tolist())
        self.assertTrue(pd.api.types.is_numeric_dtype(result["gaze_x"]))

    def test_multi_recording_detection_keeps_contiguous_blocks(self):
        frame = pd.DataFrame({
            "Participant": ["50"] * 200 + ["52"] * 220 + ["50"] * 230,
            "RecordingTime [ms]": (list(range(0, 4000, 20)) +
                                   list(range(500, 4900, 20)) +
                                   list(range(100, 4700, 20))),
            "gaze_x": np.arange(650),
        })
        blocks = detect_recording_blocks(frame)
        self.assertEqual([(b.label, b.n_rows) for b in blocks],
                         [("50", 200), ("52", 220), ("50", 230)])
        for b in blocks:
            self.assertTrue((select_recording(frame, b)["RecordingTime [ms]"].diff().dropna() > 0).all())
        self.assertEqual(frame.loc[200, "RecordingTime [ms]"], 500)

    def test_not_splitting_a_single_recording_or_per_row_identifiers(self):
        single = pd.DataFrame({"Participant": ["50"] * 200, "RecordingTime [ms]": np.arange(200) * 20})
        self.assertEqual(detect_recording_blocks(single), [])
        per_row = pd.DataFrame({"Participant": [str(x) for x in range(200)], "RecordingTime [ms]": np.arange(200) * 20})
        self.assertEqual(detect_recording_blocks(per_row), [])

    def test_nullable_integer_outlier_flags_do_not_raise(self):
        values = pd.Series([100, 101, pd.NA, 101, 100, 500, 102, pd.NA, 103, 101], dtype="Int64")
        timestamps = pd.Series(np.arange(len(values)) / 50)
        fig = _signal_plot(values, timestamps, 5000, ["IQR", "MAD"], True)
        self.assertIn("IQR flags", [trace.name for trace in fig.data])
        self.assertIn("MAD flags", [trace.name for trace in fig.data])
        for trace in fig.data:
            if trace.name.endswith("flags"):
                self.assertFalse(any(pd.isna(value) for value in trace.y))

    def test_nullable_columns_become_standard_float_with_nan(self):
        frame = pd.DataFrame({
            "RecordingTime [ms]": [0, 20, 40, 60, 80, 100],
            "gaze_x": pd.Series([3, 4, pd.NA, 5, 130, 6], dtype="Int64"),
        })
        result = canonical_frame(frame, ["gaze_x"], 50, "RecordingTime [ms]", "Auto")
        self.assertEqual(result["gaze_x"].dtype, np.dtype("float64"))
        self.assertTrue(np.isnan(result.loc[2, "gaze_x"]))
        self.assertAlmostEqual(validate_timestamps(result["timestamp"])["sampling_rate_hz"], 50)
        self.assertTrue(pd.api.types.is_extension_array_dtype(frame["gaze_x"].dtype))

    def test_actual_time_column_preferred_over_generic_timestamp(self):
        frame = pd.DataFrame({
            "timestamp": [0, 1, 2, 3, 4],  # ambiguous numeric time without unit
            "RecordingTime [ms]": [0, 20, 40, 60, 80],
            "gaze": [1, 2, 3, 4, 5],
        })
        self.assertEqual(preferred_timestamp_column(frame), "RecordingTime [ms]")

    def test_unlabelled_row_index_is_not_silently_interpreted_as_seconds(self):
        frame = pd.DataFrame({"Unnamed: 0": [0, 1, 2, 3, 4], "gaze_x": [4, 5, 3, 7, 9]})
        self.assertIsNone(preferred_timestamp_column(frame))

    def test_unitless_named_timestamp_is_not_guessed(self):
        frame = pd.DataFrame({"timestamp": [0, 10, 20, 30], "gaze": [1, 2, 3, 4]})
        self.assertEqual(preferred_timestamp_column(frame), "timestamp")
        with self.assertRaisesRegex(ValueError, "unit is ambiguous"):
            canonical_frame(frame, ["gaze"], 1, "timestamp", "Auto")

    def test_realistic_mixed_data_can_be_profiled_and_plotted(self):
        raw = pd.DataFrame({
            "RecordingTime [ms]": np.arange(400) * 20,
            "Point of Regard Left X [px]": pd.Series(
                ["", "250", "270", "1200", "NA", "260", "261", "260"] * 50,
                dtype="string",
            ),
        })
        converted = auto_convert_numeric(raw)
        actual_time = preferred_timestamp_column(converted)
        self.assertEqual(actual_time, "RecordingTime [ms]")
        signal = "Point of Regard Left X [px]"
        canonical = canonical_frame(converted, [signal], 50, actual_time, "Auto")
        profile = profile_table(canonical, 50, DOMAIN_CONFIG["eye_tracking"]["quality_weights"])
        self.assertEqual(len(profile), 1)
        fig = _signal_plot(canonical[signal], canonical["timestamp"], 120, ["IQR", "MAD"], True)
        self.assertGreaterEqual(len(fig.data), 3)


if __name__ == "__main__":
    unittest.main()
