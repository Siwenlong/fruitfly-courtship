from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


POSE_COLUMNS = [
    "video_id",
    "frame",
    "time_s",
    "individual",
    "keypoint",
    "x",
    "y",
    "score",
]

BASE_REQUIRED_COLUMNS = ["video_id", "frame", "individual", "keypoint", "x", "y", "score"]
IDENTIFIER_COLUMNS = ["video_id", "individual", "keypoint"]
FINITE_NUMERIC_COLUMNS = ["time_s", "x", "y", "score"]
NATURAL_KEY_COLUMNS = ["video_id", "frame", "individual", "keypoint"]
EXPECTED_INDIVIDUALS = ["female", "male"]
FRAME_ERROR = (
    "Pose CSV frame values must be numeric, finite, integer-valued, non-null, and non-negative"
)


def _require_no_null_values(pose: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if pose[column].isna().any():
            raise ValueError(f"Pose CSV column {column} contains missing/null values")


def _coerce_integer_frame(frame: pd.Series) -> pd.Series:
    try:
        numeric = pd.to_numeric(frame, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(FRAME_ERROR) from exc

    values = numeric.to_numpy(dtype=float)
    if (
        numeric.isna().any()
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or (values < 0).any()
    ):
        raise ValueError(FRAME_ERROR)
    return numeric.astype(int)


def _coerce_finite_numeric(
    values: pd.Series,
    column: str,
    *,
    allow_negative: bool = True,
) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pose CSV column {column} must be numeric and finite") from exc

    numeric_values = numeric.to_numpy(dtype=float)
    if numeric.isna().any() or not np.isfinite(numeric_values).all():
        raise ValueError(f"Pose CSV column {column} must be numeric and finite")
    if not allow_negative and (numeric_values < 0).any():
        raise ValueError(f"Pose CSV column {column} must be non-negative")
    return numeric.astype(float)


def _validate_fps(fps: float) -> float:
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be finite and positive") from exc
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be finite and positive")
    return fps_value


def load_pose_csv(path: str | Path, fps: float | None = None) -> pd.DataFrame:
    """Load a normalized long-form pose CSV with one keypoint observation per row."""
    pose_path = Path(path)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose CSV not found: {pose_path}")

    pose = pd.read_csv(pose_path)
    missing_columns = [column for column in BASE_REQUIRED_COLUMNS if column not in pose.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Pose CSV missing required columns: {joined}")

    _require_no_null_values(pose, IDENTIFIER_COLUMNS)
    pose["frame"] = _coerce_integer_frame(pose["frame"])

    if "time_s" not in pose.columns:
        if fps is None:
            raise ValueError("time_s column is missing; pass fps to derive timestamps from frame")
        pose["time_s"] = pose["frame"].astype(float) / _validate_fps(fps)

    pose = pose[POSE_COLUMNS].copy()
    pose["video_id"] = pose["video_id"].astype(str)
    pose["individual"] = pose["individual"].astype(str)
    pose["keypoint"] = pose["keypoint"].astype(str)
    for column in FINITE_NUMERIC_COLUMNS:
        pose[column] = _coerce_finite_numeric(
            pose[column],
            column,
            allow_negative=column != "time_s",
        )

    if pose.duplicated(NATURAL_KEY_COLUMNS).any():
        raise ValueError("Duplicate pose rows for video_id, frame, individual, keypoint")

    return pose.reset_index(drop=True)


def infer_fps(pose: pd.DataFrame) -> float:
    """Infer one global FPS, assuming all videos in the pose table share that FPS."""
    frame_time_values = pose[["video_id", "frame", "time_s"]].copy()
    frame_time_values["frame"] = _coerce_integer_frame(frame_time_values["frame"])
    frame_time_values["time_s"] = _coerce_finite_numeric(
        frame_time_values["time_s"],
        "time_s",
        allow_negative=False,
    )
    frame_time_values = frame_time_values.drop_duplicates()
    distinct_timestamps = frame_time_values.groupby(["video_id", "frame"], sort=False)[
        "time_s"
    ].nunique(dropna=False)
    if (distinct_timestamps > 1).any():
        raise ValueError("Pose contains conflicting time_s values for the same video_id and frame")

    frame_times = frame_time_values.sort_values(["video_id", "frame"])
    fps_values: list[float] = []
    for _, group in frame_times.groupby("video_id", sort=False):
        frame_values = group["frame"].to_numpy(dtype=float)
        time_values = group["time_s"].to_numpy(dtype=float)
        frame_diffs = np.diff(frame_values)
        time_diffs = np.diff(time_values)
        if (time_diffs <= 0).any():
            raise ValueError(
                "Pose contains non-increasing time_s values across increasing frames"
            )
        fps_values.extend(
            float(frame_delta / time_delta)
            for frame_delta, time_delta in zip(frame_diffs, time_diffs)
        )
    if not fps_values:
        raise ValueError("Cannot infer FPS from fewer than two distinct frame timestamps")
    return float(np.median(fps_values))


def required_keypoints_present(
    pose: pd.DataFrame,
    keypoints: list[str],
    individuals: list[str] | None = None,
) -> dict[str, list[str]]:
    """Report missing required keypoints for each expected individual."""
    checked_individuals = EXPECTED_INDIVIDUALS if individuals is None else individuals
    missing: dict[str, list[str]] = {}
    for individual in checked_individuals:
        observed = set(pose.loc[pose["individual"] == individual, "keypoint"].unique())
        absent = [keypoint for keypoint in keypoints if keypoint not in observed]
        if absent:
            missing[individual] = absent
    return missing
