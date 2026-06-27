from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_LEVEL1_KEYPOINTS = ["head", "thorax", "abdomen_tip", "left_wing_tip", "right_wing_tip"]
REQUIRED_COLUMNS = {"video_id", "frame", "time_s", "individual", "keypoint", "x", "y", "score"}
FRAME_ERROR = "Pose table frame values must be finite, integer-valued, and non-negative"
TIME_ERROR = "Pose table time_s values must be finite and non-negative"


def _vector(values: Iterable[float]) -> np.ndarray:
    return np.array(list(values), dtype=float)


def angle_deg(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    a = _vector(vector_a)
    b = _vector(vector_b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
        return float("nan")
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    if not np.isfinite(cosine):
        return float("nan")
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _point(frame_rows: pd.DataFrame, individual: str, keypoint: str) -> np.ndarray:
    match = frame_rows[
        (frame_rows["individual"] == individual)
        & (frame_rows["keypoint"] == keypoint)
    ]
    if match.empty:
        return np.array([np.nan, np.nan], dtype=float)
    row = match.iloc[0]
    return np.array([float(row["x"]), float(row["y"])], dtype=float)


def _score(frame_rows: pd.DataFrame, individual: str, keypoint: str) -> float:
    match = frame_rows[
        (frame_rows["individual"] == individual)
        & (frame_rows["keypoint"] == keypoint)
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["score"])


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
        return float("nan")
    return float(np.linalg.norm(point_a - point_b))


def _safe_divide(value: float, denominator: float) -> float:
    if denominator == 0.0 or np.isnan(value) or np.isnan(denominator):
        return float("nan")
    return float(value / denominator)


def _validate_grouping_keys(pose: pd.DataFrame) -> pd.DataFrame:
    if pose.empty:
        raise ValueError("Pose table contains no rows")
    if pose["video_id"].isna().any():
        raise ValueError("Pose table video_id values must be non-null")

    validated = pose.copy()
    try:
        frame = pd.to_numeric(validated["frame"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(FRAME_ERROR) from exc
    frame_values = frame.to_numpy(dtype=float)
    if (
        frame.isna().any()
        or not np.isfinite(frame_values).all()
        or not np.equal(frame_values, np.floor(frame_values)).all()
        or (frame_values < 0).any()
    ):
        raise ValueError(FRAME_ERROR)
    validated["frame"] = frame.astype(int)

    try:
        time_s = pd.to_numeric(validated["time_s"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(TIME_ERROR) from exc
    time_values = time_s.to_numpy(dtype=float)
    if time_s.isna().any() or not np.isfinite(time_values).all() or (time_values < 0).any():
        raise ValueError(TIME_ERROR)
    validated["time_s"] = time_s.astype(float)

    return validated


def _frame_features(frame_rows: pd.DataFrame, video_id: str, frame: int, time_s: float) -> dict[str, float | int | str]:
    male_head = _point(frame_rows, "male", "head")
    male_thorax = _point(frame_rows, "male", "thorax")
    male_abdomen = _point(frame_rows, "male", "abdomen_tip")
    male_left_wing = _point(frame_rows, "male", "left_wing_tip")
    male_right_wing = _point(frame_rows, "male", "right_wing_tip")

    female_head = _point(frame_rows, "female", "head")
    female_thorax = _point(frame_rows, "female", "thorax")
    female_abdomen = _point(frame_rows, "female", "abdomen_tip")

    male_axis = male_head - male_abdomen
    female_axis = female_head - female_abdomen
    male_body_length_px = _distance(male_head, male_abdomen)

    left_wing_angle = angle_deg(male_left_wing - male_thorax, male_axis)
    right_wing_angle = angle_deg(male_right_wing - male_thorax, male_axis)
    wing_asymmetry = abs(left_wing_angle - right_wing_angle)

    male_to_female = female_thorax - male_thorax
    distance_px = _distance(male_thorax, female_thorax)
    distance_body_lengths = _safe_divide(distance_px, male_body_length_px)
    heading_error = angle_deg(male_axis, male_to_female)

    male_to_female_posterior = female_abdomen - male_thorax
    female_posterior_to_male = male_thorax - female_abdomen
    copulation_distance_px = _distance(male_thorax, female_abdomen)
    copulation_distance_body_lengths = _safe_divide(copulation_distance_px, male_body_length_px)
    male_to_female_posterior_angle = angle_deg(male_axis, male_to_female_posterior)
    female_posterior_angle = angle_deg(female_axis, female_posterior_to_male)

    scores = [
        _score(frame_rows, individual, keypoint)
        for individual in ["male", "female"]
        for keypoint in REQUIRED_LEVEL1_KEYPOINTS
    ]
    tracking_min_score = float("nan") if any(not np.isfinite(score) for score in scores) else min(scores)

    return {
        "video_id": video_id,
        "frame": frame,
        "time_s": time_s,
        "male_thorax_x": float(male_thorax[0]),
        "male_thorax_y": float(male_thorax[1]),
        "female_thorax_x": float(female_thorax[0]),
        "female_thorax_y": float(female_thorax[1]),
        "male_body_length_px": male_body_length_px,
        "left_wing_angle_deg": left_wing_angle,
        "right_wing_angle_deg": right_wing_angle,
        "wing_asymmetry_deg": wing_asymmetry,
        "distance_body_lengths": distance_body_lengths,
        "heading_error_deg": heading_error,
        "copulation_distance_body_lengths": copulation_distance_body_lengths,
        "male_to_female_posterior_angle_deg": male_to_female_posterior_angle,
        "female_posterior_angle_deg": female_posterior_angle,
        "tracking_min_score": tracking_min_score,
    }


def extract_features(pose: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(pose.columns))
    if missing:
        raise ValueError(f"Pose table missing required columns: {', '.join(missing)}")
    pose = _validate_grouping_keys(pose)
    if pose.duplicated(["video_id", "frame", "individual", "keypoint"]).any():
        raise ValueError("Duplicate pose rows for video_id, frame, individual, keypoint")

    frame_times = pose[["video_id", "frame", "time_s"]].drop_duplicates()
    distinct_timestamps = frame_times.groupby(["video_id", "frame"], sort=False)[
        "time_s"
    ].nunique(dropna=False)
    if (distinct_timestamps > 1).any():
        raise ValueError("Pose contains conflicting time_s values for the same video_id and frame")
    for _, group in frame_times.sort_values(["video_id", "frame"]).groupby("video_id", sort=False):
        if (group["time_s"].diff().dropna() <= 0).any():
            raise ValueError(
                "Pose contains non-increasing time_s values across increasing frames within video_id"
            )

    feature_rows: list[dict[str, float | int | str]] = []
    grouped = pose.groupby(["video_id", "frame", "time_s"], sort=True)
    for (video_id, frame, time_s), frame_rows in grouped:
        feature_rows.append(
            _frame_features(
                frame_rows=frame_rows,
                video_id=str(video_id),
                frame=int(frame),
                time_s=float(time_s),
            )
        )

    features = pd.DataFrame(feature_rows).sort_values(["video_id", "frame"]).reset_index(drop=True)
    features["male_speed_body_lengths_s"] = 0.0
    features["relative_speed_body_lengths_s"] = 0.0

    for _, index in features.groupby("video_id", sort=False).groups.items():
        ordered_index = list(index)
        group = features.loc[ordered_index].sort_values("frame")
        dt = group["time_s"].diff()
        male_dx = group["male_thorax_x"].diff()
        male_dy = group["male_thorax_y"].diff()
        male_step_px = np.sqrt(male_dx.pow(2) + male_dy.pow(2))
        body_length = group["male_body_length_px"].replace(0.0, np.nan)
        male_speed = male_step_px / dt / body_length

        distance_px = np.sqrt(
            (group["female_thorax_x"] - group["male_thorax_x"]).pow(2)
            + (group["female_thorax_y"] - group["male_thorax_y"]).pow(2)
        )
        relative_speed = distance_px.diff().abs() / dt / body_length

        if not group.empty:
            male_speed.iloc[0] = 0.0
            relative_speed.iloc[0] = 0.0

        features.loc[group.index, "male_speed_body_lengths_s"] = male_speed.to_numpy()
        features.loc[group.index, "relative_speed_body_lengths_s"] = relative_speed.to_numpy()

    return features
