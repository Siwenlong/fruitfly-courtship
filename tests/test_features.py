import math

import pandas as pd
import pytest

from fruitfly_courtship.features import angle_deg, extract_features


def _frame_rows(frame: int, time_s: float, male_y: float) -> list[dict[str, object]]:
    male_points = {
        "abdomen_tip": (0.0, male_y - 1.0),
        "thorax": (0.0, male_y),
        "head": (0.0, male_y + 1.0),
        "left_wing_tip": (0.2, male_y + 0.9),
        "right_wing_tip": (2.0, male_y),
    }
    female_points = {
        "abdomen_tip": (0.0, 3.0),
        "thorax": (0.0, 4.0),
        "head": (0.0, 5.0),
        "left_wing_tip": (-0.2, 4.0),
        "right_wing_tip": (0.2, 4.0),
    }
    rows: list[dict[str, object]] = []
    for individual, points in [("male", male_points), ("female", female_points)]:
        for keypoint, (x, y) in points.items():
            rows.append(
                {
                    "video_id": "fly_001",
                    "frame": frame,
                    "time_s": time_s,
                    "individual": individual,
                    "keypoint": keypoint,
                    "x": x,
                    "y": y,
                    "score": 0.95,
                }
            )
    return rows


def _with_video_id(rows: list[dict[str, object]], video_id: str) -> list[dict[str, object]]:
    return [{**row, "video_id": video_id} for row in rows]


def _pose_with_grouping_key(column: str, value: object) -> pd.DataFrame:
    rows = _frame_rows(0, 0.0, 1.0)
    for row in rows:
        row[column] = value
    return pd.DataFrame(rows)


def test_angle_deg_handles_right_angle():
    assert angle_deg((1, 0), (0, 1)) == pytest.approx(90.0)
    assert angle_deg((0, 2), (0, 5)) == pytest.approx(0.0)
    assert math.isnan(angle_deg((0, 0), (1, 0)))


def test_angle_deg_returns_nan_for_non_finite_vectors():
    assert math.isnan(angle_deg((float("nan"), 1), (1, 0)))
    assert math.isnan(angle_deg((float("inf"), 1), (1, 0)))


def test_extract_features_computes_wing_and_chasing_geometry():
    pose = pd.DataFrame(_frame_rows(0, 0.0, 1.0) + _frame_rows(1, 0.1, 1.2))

    features = extract_features(pose)

    first = features.iloc[0]
    second = features.iloc[1]
    assert first["video_id"] == "fly_001"
    assert first["frame"] == 0
    assert first["right_wing_angle_deg"] > 80.0
    assert first["left_wing_angle_deg"] < 20.0
    assert first["wing_asymmetry_deg"] > 70.0
    assert first["distance_body_lengths"] == pytest.approx(1.5)
    assert first["heading_error_deg"] == pytest.approx(0.0)
    assert second["male_speed_body_lengths_s"] == pytest.approx(1.0)
    assert first["copulation_distance_body_lengths"] == pytest.approx(1.0)
    assert first["male_to_female_posterior_angle_deg"] == pytest.approx(0.0)
    assert first["female_posterior_angle_deg"] == pytest.approx(180.0)
    assert first["tracking_min_score"] == pytest.approx(0.95)


def test_extract_features_propagates_missing_keypoint_to_angles():
    pose = pd.DataFrame(
        row
        for row in _frame_rows(0, 0.0, 1.0)
        if not (row["individual"] == "male" and row["keypoint"] == "left_wing_tip")
    )

    features = extract_features(pose)

    first = features.iloc[0]
    assert math.isnan(first["left_wing_angle_deg"])
    assert math.isnan(first["wing_asymmetry_deg"])


def test_extract_features_rejects_duplicate_natural_key_rows():
    rows = _frame_rows(0, 0.0, 1.0)
    rows.append({**rows[0], "x": 10.0})
    pose = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="Duplicate pose rows"):
        extract_features(pose)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("time_s", float("nan"), "time_s"),
        ("video_id", None, "video_id"),
        ("frame", 0.5, "frame"),
        ("frame", -1, "frame"),
        ("time_s", -0.1, "time_s"),
    ],
)
def test_extract_features_rejects_invalid_grouping_keys(
    column: str,
    value: object,
    message: str,
):
    pose = _pose_with_grouping_key(column, value)

    with pytest.raises(ValueError, match=message):
        extract_features(pose)


@pytest.mark.parametrize(("first_time_s", "second_time_s"), [(0.0, 0.0), (0.2, 0.1)])
def test_extract_features_rejects_non_increasing_timestamps(
    first_time_s: float,
    second_time_s: float,
):
    pose = pd.DataFrame(
        _frame_rows(0, first_time_s, 1.0) + _frame_rows(1, second_time_s, 1.2)
    )

    with pytest.raises(ValueError, match="non-increasing time_s"):
        extract_features(pose)


def test_extract_features_preserves_nan_speed_for_zero_body_length_after_first_frame():
    rows = _frame_rows(0, 0.0, 1.0) + _frame_rows(1, 0.1, 1.2)
    for row in rows:
        if row["frame"] == 1 and row["individual"] == "male" and row["keypoint"] in {
            "head",
            "abdomen_tip",
        }:
            row["x"] = 0.0
            row["y"] = 1.2
    pose = pd.DataFrame(rows)

    features = extract_features(pose)

    assert features.iloc[0]["male_speed_body_lengths_s"] == pytest.approx(0.0)
    assert math.isnan(features.iloc[1]["male_speed_body_lengths_s"])
    assert math.isnan(features.iloc[1]["relative_speed_body_lengths_s"])


def test_extract_features_rejects_empty_pose_table():
    pose = pd.DataFrame(
        columns=["video_id", "frame", "time_s", "individual", "keypoint", "x", "y", "score"]
    )

    with pytest.raises(ValueError, match="Pose table contains no rows"):
        extract_features(pose)


def test_extract_features_marks_tracking_score_nan_when_required_keypoint_missing():
    pose = pd.DataFrame(
        row
        for row in _frame_rows(0, 0.0, 1.0)
        if not (row["individual"] == "female" and row["keypoint"] == "right_wing_tip")
    )

    features = extract_features(pose)

    assert math.isnan(features.iloc[0]["tracking_min_score"])


def test_extract_features_resets_first_frame_speed_per_video():
    fly_001_rows = _with_video_id(
        _frame_rows(0, 0.0, 1.0) + _frame_rows(1, 0.1, 1.2),
        "fly_001",
    )
    fly_002_rows = _with_video_id(
        _frame_rows(0, 0.0, 2.0) + _frame_rows(1, 0.1, 2.2),
        "fly_002",
    )
    pose = pd.DataFrame(fly_001_rows + fly_002_rows)

    features = extract_features(pose)

    fly_001 = features[features["video_id"] == "fly_001"].reset_index(drop=True)
    fly_002 = features[features["video_id"] == "fly_002"].reset_index(drop=True)
    assert fly_001.iloc[0]["male_speed_body_lengths_s"] == pytest.approx(0.0)
    assert fly_002.iloc[0]["male_speed_body_lengths_s"] == pytest.approx(0.0)
    assert fly_001.iloc[1]["male_speed_body_lengths_s"] == pytest.approx(1.0)
    assert fly_002.iloc[1]["male_speed_body_lengths_s"] == pytest.approx(1.0)
