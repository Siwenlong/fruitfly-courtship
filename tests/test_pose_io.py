from pathlib import Path

import pandas as pd
import pytest

from fruitfly_courtship.pose_io import infer_fps, load_pose_csv, required_keypoints_present


def _write_pose(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _valid_pose_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "video_id": "fly_001",
        "frame": 0,
        "time_s": 0.0,
        "individual": "male",
        "keypoint": "head",
        "x": 1.0,
        "y": 2.0,
        "score": 0.9,
    }
    row.update(overrides)
    return row


def test_load_pose_csv_accepts_long_format_with_time(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            {
                "video_id": "fly_001",
                "frame": 0,
                "time_s": 0.0,
                "individual": "male",
                "keypoint": "head",
                "x": 1.0,
                "y": 2.0,
                "score": 0.9,
            },
            {
                "video_id": "fly_001",
                "frame": 0,
                "time_s": 0.0,
                "individual": "female",
                "keypoint": "head",
                "x": 4.0,
                "y": 5.0,
                "score": 0.8,
            },
        ],
    )

    pose = load_pose_csv(path)

    assert list(pose.columns) == [
        "video_id",
        "frame",
        "time_s",
        "individual",
        "keypoint",
        "x",
        "y",
        "score",
    ]
    assert pose.loc[0, "frame"] == 0
    assert pose.loc[0, "score"] == pytest.approx(0.9)


def test_load_pose_csv_derives_time_from_fps(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            {
                "video_id": "fly_001",
                "frame": 3,
                "individual": "male",
                "keypoint": "head",
                "x": 1.0,
                "y": 2.0,
                "score": 0.9,
            }
        ],
    )

    pose = load_pose_csv(path, fps=30)

    assert pose.loc[0, "time_s"] == pytest.approx(0.1)


def test_load_pose_csv_requires_time_or_fps(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            {
                "video_id": "fly_001",
                "frame": 0,
                "individual": "male",
                "keypoint": "head",
                "x": 1.0,
                "y": 2.0,
                "score": 0.9,
            }
        ],
    )

    with pytest.raises(ValueError, match="time_s column is missing"):
        load_pose_csv(path)


def test_load_pose_csv_rejects_missing_required_score_column(tmp_path: Path):
    path = tmp_path / "pose.csv"
    row = _valid_pose_row()
    del row["score"]
    _write_pose(path, [row])

    with pytest.raises(ValueError, match="Pose CSV missing required columns"):
        load_pose_csv(path)


def test_load_pose_csv_rejects_duplicate_natural_key_rows(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row(), _valid_pose_row(x=2.0, y=3.0, score=0.7)])

    with pytest.raises(ValueError, match="Duplicate pose rows"):
        load_pose_csv(path)


@pytest.mark.parametrize("column", ["video_id", "individual", "keypoint"])
def test_load_pose_csv_rejects_missing_identifier_values(tmp_path: Path, column: str):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row(**{column: None})])

    with pytest.raises(ValueError, match=column):
        load_pose_csv(path)


@pytest.mark.parametrize("frame", [None, "not-a-frame", 1.5, float("inf")])
def test_load_pose_csv_rejects_invalid_frame_values(tmp_path: Path, frame: object):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row(frame=frame)])

    with pytest.raises(ValueError, match="frame"):
        load_pose_csv(path)


def test_load_pose_csv_rejects_negative_frame_values(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row(frame=-1)])

    with pytest.raises(ValueError, match="frame.*non-negative"):
        load_pose_csv(path)


@pytest.mark.parametrize("column", ["time_s", "x", "y", "score"])
@pytest.mark.parametrize("value", [None, "not-a-number", float("inf")])
def test_load_pose_csv_rejects_non_numeric_or_non_finite_values(
    tmp_path: Path,
    column: str,
    value: object,
):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row(**{column: value})])

    with pytest.raises(ValueError, match=column):
        load_pose_csv(path)


def test_load_pose_csv_rejects_negative_time_values(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row(time_s=-0.1)])

    with pytest.raises(ValueError, match="time_s.*non-negative"):
        load_pose_csv(path)


@pytest.mark.parametrize("fps", [0, -30, float("nan"), float("inf")])
def test_load_pose_csv_rejects_invalid_fps_when_deriving_time(tmp_path: Path, fps: float):
    path = tmp_path / "pose.csv"
    row = _valid_pose_row()
    del row["time_s"]
    _write_pose(path, [row])

    with pytest.raises(ValueError, match="fps"):
        load_pose_csv(path, fps=fps)


def test_required_keypoints_present_reports_missing_items(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            {
                "video_id": "fly_001",
                "frame": 0,
                "time_s": 0.0,
                "individual": "male",
                "keypoint": "head",
                "x": 1.0,
                "y": 2.0,
                "score": 0.9,
            }
        ],
    )
    pose = load_pose_csv(path)

    missing = required_keypoints_present(pose, ["head", "thorax"])

    assert missing == {"female": ["head", "thorax"], "male": ["thorax"]}


def test_required_keypoints_present_respects_explicit_empty_individuals(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(path, [_valid_pose_row()])
    pose = load_pose_csv(path)

    assert required_keypoints_present(pose, ["head"], individuals=[]) == {}


def test_infer_fps_uses_median_frame_spacing(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            {
                "video_id": "fly_001",
                "frame": frame,
                "time_s": time_s,
                "individual": "male",
                "keypoint": "head",
                "x": 1.0,
                "y": 2.0,
                "score": 0.9,
            }
            for frame, time_s in zip([0, 1, 2, 3], [0.0, 0.05, 0.10, 1.10])
        ],
    )
    pose = load_pose_csv(path)

    assert infer_fps(pose) == pytest.approx(20.0)


def test_infer_fps_uses_delta_frame_over_delta_time_for_sparse_frames(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            _valid_pose_row(frame=frame, time_s=time_s)
            for frame, time_s in zip([0, 2, 4], [0.0, 2 / 30, 4 / 30])
        ],
    )
    pose = load_pose_csv(path)

    assert infer_fps(pose) == pytest.approx(30.0)


@pytest.mark.parametrize("time_values", [[0.0, 0.0, 1 / 30], [0.0, 0.1, 0.05]])
def test_infer_fps_rejects_non_increasing_timestamps_for_increasing_frames(
    tmp_path: Path,
    time_values: list[float],
):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            _valid_pose_row(frame=frame, time_s=time_s)
            for frame, time_s in zip([0, 1, 2], time_values)
        ],
    )
    pose = load_pose_csv(path)

    with pytest.raises(ValueError, match="non-increasing"):
        infer_fps(pose)


@pytest.mark.parametrize(
    ("frame", "time_s", "message"),
    [
        (1.5, 0.1, "frame"),
        (float("inf"), 0.1, "frame"),
        (-1, 0.1, "frame.*non-negative"),
        (1, "not-a-time", "time_s"),
        (1, float("inf"), "time_s"),
        (1, -0.1, "time_s.*non-negative"),
    ],
)
def test_infer_fps_validates_frame_and_time_when_called_directly(
    frame: object,
    time_s: object,
    message: str,
):
    pose = pd.DataFrame(
        [
            {"video_id": "fly_001", "frame": 0, "time_s": 0.0},
            {"video_id": "fly_001", "frame": frame, "time_s": time_s},
        ]
    )

    with pytest.raises(ValueError, match=message):
        infer_fps(pose)


def test_infer_fps_rejects_conflicting_timestamps_for_same_frame(tmp_path: Path):
    path = tmp_path / "pose.csv"
    _write_pose(
        path,
        [
            _valid_pose_row(frame=0, time_s=0.0, individual="male"),
            _valid_pose_row(frame=0, time_s=0.1, individual="female"),
            _valid_pose_row(frame=1, time_s=0.05, individual="male"),
        ],
    )
    pose = load_pose_csv(path)

    with pytest.raises(ValueError, match="conflicting"):
        infer_fps(pose)
