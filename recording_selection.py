"""Conservative selection of contiguous recordings in uploaded multi-session CSVs.

A *recording* is never formed by concatenating disjoint blocks with the same
participant ID. The application processes only the selected interval in its
original row order and preserves all original timestamp and signal values.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecordingBlock:
    start: int  # inclusive, positional
    stop: int   # exclusive, positional
    label: str
    source_column: str

    @property
    def n_rows(self) -> int:
        return self.stop - self.start


_RECORDING_COLUMNS = (
    # Prefer the finest reliable contiguous measurement unit to a participant ID:
    # a single participant may have many independent recordings/trials.
    "recording name", "recording id", "recordingid", "recording", "video",
    "session id", "sessionid", "session", "task", "trial",
    "participant", "participant id", "participantid", "subject", "subject id", "subjectid",
)


def detect_recording_blocks(df: pd.DataFrame, *, min_run_length: int = 30,
                            max_blocks: int = 150) -> list[RecordingBlock]:
    """Find sufficiently long, contiguous ID-labelled recording blocks.

    Do not split on arbitrary backward time jumps: those can also represent
    faulty timestamps. Users must inspect/repair those cases separately.
    Empty output means no reliably identifiable multi-recording structure.
    """
    if len(df) < 2 * min_run_length:
        return []
    columns = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for name in _RECORDING_COLUMNS:
        column = columns.get(re.sub(r"[^a-z0-9]", "", name))
        if column is None:
            continue
        # Missing IDs are kept in their own contiguous blocks. Their original
        # values remain untouched; no changes are made to the input CSV.
        ids = df[column].astype("string").fillna("(unknown)").str.strip().to_numpy(dtype=str)
        if ids.size == 0:
            continue
        change_positions = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        boundaries = np.r_[0, change_positions, len(df)]
        lengths = np.diff(boundaries)
        if not (2 <= len(lengths) <= max_blocks and np.median(lengths) >= min_run_length
                and (lengths >= min_run_length).mean() >= .8):
            continue
        return [RecordingBlock(int(a), int(b), ids[a], str(column))
                for a, b in zip(boundaries[:-1], boundaries[1:])]
    return []


def select_recording(df: pd.DataFrame, block: RecordingBlock) -> pd.DataFrame:
    """Return selected original rows, preserving samples and timestamp values."""
    if not (0 <= block.start < block.stop <= len(df)):
        raise ValueError("Selected recording is outside the uploaded dataset")
    return df.iloc[block.start:block.stop].copy().reset_index(drop=True)
