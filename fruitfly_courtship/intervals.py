from __future__ import annotations

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "video_id",
    "behavior",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "confidence",
    "method",
]


def _segments(mask: list[bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if (not value) and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _merge_segments(segments: list[tuple[int, int]], fps: float, merge_gap_s: float) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    max_gap_frames = int(round(merge_gap_s * fps))
    for start, end in segments[1:]:
        previous_start, previous_end = merged[-1]
        gap = start - previous_end - 1
        if gap <= max_gap_frames:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def frames_to_events(
    frames: pd.DataFrame,
    mask: pd.Series,
    behavior: str,
    fps: float,
    min_duration_s: float,
    merge_gap_s: float,
    method: str,
) -> pd.DataFrame:
    if len(frames) != len(mask):
        raise ValueError("frames and mask must have the same length")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    rows: list[dict[str, object]] = []
    ordered = frames.reset_index(drop=True)
    bool_mask = [bool(value) for value in mask.reset_index(drop=True).fillna(False)]

    for start_index, end_index in _merge_segments(_segments(bool_mask), fps=fps, merge_gap_s=merge_gap_s):
        start_row = ordered.iloc[start_index]
        end_row = ordered.iloc[end_index]
        start_time_s = float(start_row["time_s"])
        end_time_s = float(end_row["time_s"]) + (1.0 / fps)
        duration_s = end_time_s - start_time_s
        if duration_s < min_duration_s:
            continue
        confidence = float(ordered.iloc[start_index : end_index + 1]["tracking_min_score"].mean())
        rows.append(
            {
                "video_id": str(start_row["video_id"]),
                "behavior": behavior,
                "start_frame": int(start_row["frame"]),
                "end_frame": int(end_row["frame"]),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": duration_s,
                "confidence": confidence,
                "method": method,
            }
        )

    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
