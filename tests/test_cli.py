from pathlib import Path

import pandas as pd

from fruitfly_courtship.cli import main


def _rows_for_frame(frame: int, time_s: float, male_y: float) -> list[dict[str, object]]:
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


def test_cli_analyze_writes_outputs(tmp_path: Path):
    pose_path = tmp_path / "pose.csv"
    rows: list[dict[str, object]] = []
    for frame in range(6):
        rows.extend(_rows_for_frame(frame, frame / 10, 1.0 + frame * 0.2))
    pd.DataFrame(rows).to_csv(pose_path, index=False)

    config_path = tmp_path / "rules.yaml"
    config_path.write_text(
        "\n".join(
            [
                "min_keypoint_score: 0.3",
                "wing_extension:",
                "  min_duration_s: 0.1",
                "chasing:",
                "  min_duration_s: 0.1",
                "  min_speed_body_lengths_s: 0.0",
                "copulation_attempt:",
                "  min_duration_s: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "outputs"

    exit_code = main(["analyze", "--pose-csv", str(pose_path), "--config", str(config_path), "--out-dir", str(out_dir)])

    assert exit_code == 0
    events = pd.read_csv(out_dir / "events.csv")
    assert {"wing_extension", "chasing", "copulation_attempt"}.issubset(set(events["behavior"]))
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "features.csv").exists()
    assert (out_dir / "qc" / "tracking_quality.csv").exists()
    assert (out_dir / "qc" / "fly_001_timeline.png").exists()
