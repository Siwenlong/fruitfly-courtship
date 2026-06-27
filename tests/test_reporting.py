from pathlib import Path

import pandas as pd

from fruitfly_courtship.reporting import (
    plot_timeline,
    summarize_events,
    write_outputs,
    write_tracking_quality,
)


def test_summarize_events_counts_duration_and_latency():
    events = pd.DataFrame(
        {
            "video_id": ["fly_001", "fly_001", "fly_001"],
            "behavior": ["wing_extension", "wing_extension", "chasing"],
            "start_time_s": [1.0, 4.0, 2.0],
            "end_time_s": [2.0, 5.5, 3.0],
            "duration_s": [1.0, 1.5, 1.0],
            "confidence": [0.9, 0.8, 0.7],
            "method": ["rules_v1", "rules_v1", "rules_v1"],
        }
    )

    summary = summarize_events(events)

    wing = summary[summary["behavior"] == "wing_extension"].iloc[0]
    assert wing["event_count"] == 2
    assert wing["total_duration_s"] == 2.5
    assert wing["latency_to_first_s"] == 1.0
    assert wing["mean_event_duration_s"] == 1.25


def test_write_outputs_creates_expected_files(tmp_path: Path):
    events = pd.DataFrame(
        {
            "video_id": ["fly_001"],
            "behavior": ["wing_extension"],
            "start_time_s": [1.0],
            "end_time_s": [2.0],
            "duration_s": [1.0],
            "confidence": [0.9],
            "method": ["rules_v1"],
        }
    )
    features = pd.DataFrame({"video_id": ["fly_001"], "frame": [0], "time_s": [0.0]})

    write_outputs(events=events, features=features, out_dir=tmp_path)

    assert (tmp_path / "events.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "features.csv").exists()
    assert (tmp_path / "qc" / "fly_001_timeline.png").exists()


def test_write_tracking_quality_creates_keypoint_summary(tmp_path: Path):
    pose = pd.DataFrame(
        {
            "video_id": ["fly_001", "fly_001"],
            "frame": [0, 0],
            "time_s": [0.0, 0.0],
            "individual": ["male", "male"],
            "keypoint": ["head", "thorax"],
            "x": [1.0, 1.0],
            "y": [2.0, 1.0],
            "score": [0.9, 0.2],
        }
    )

    write_tracking_quality(pose=pose, out_dir=tmp_path, min_keypoint_score=0.3)

    quality = pd.read_csv(tmp_path / "qc" / "tracking_quality.csv")
    assert set(quality["keypoint"]) == {"head", "thorax"}
    assert quality.loc[quality["keypoint"] == "thorax", "low_score_fraction"].iloc[0] == 1.0


def test_plot_timeline_handles_empty_events(tmp_path: Path):
    plot_timeline(events=pd.DataFrame(), out_dir=tmp_path)

    assert (tmp_path / "qc" / "all_videos_timeline.png").exists()
