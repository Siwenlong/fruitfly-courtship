import pandas as pd

from fruitfly_courtship.config import CourtshipConfig
from fruitfly_courtship.detectors import detect_behaviors


def test_detect_behaviors_finds_level1_events():
    features = pd.DataFrame(
        {
            "video_id": ["fly_001"] * 6,
            "frame": list(range(6)),
            "time_s": [frame / 10 for frame in range(6)],
            "left_wing_angle_deg": [10, 10, 10, 10, 10, 10],
            "right_wing_angle_deg": [80, 80, 80, 80, 10, 10],
            "wing_asymmetry_deg": [70, 70, 70, 70, 0, 0],
            "distance_body_lengths": [2, 2, 2, 2, 2, 2],
            "heading_error_deg": [20, 20, 20, 20, 20, 20],
            "male_speed_body_lengths_s": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            "copulation_distance_body_lengths": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0],
            "male_to_female_posterior_angle_deg": [20, 20, 20, 20, 100, 100],
            "female_posterior_angle_deg": [160, 160, 160, 160, 60, 60],
            "relative_speed_body_lengths_s": [0.2, 0.2, 0.2, 0.2, 2.0, 2.0],
            "tracking_min_score": [0.95] * 6,
        }
    )
    config = CourtshipConfig()

    events = detect_behaviors(features, config)

    assert set(events["behavior"]) == {
        "orientation",
        "chasing",
        "copulation_attempt",
        "wing_extension",
    }
    assert events.loc[events["behavior"] == "wing_extension", "duration_s"].iloc[0] >= 0.2
    assert events.loc[events["behavior"] == "copulation_attempt", "duration_s"].iloc[0] >= 0.1


def test_detect_behaviors_finds_v2_courtship_sequence_events():
    frame_count = 12
    features = pd.DataFrame(
        {
            "video_id": ["fly_001"] * frame_count,
            "frame": list(range(frame_count)),
            "time_s": [frame / 10 for frame in range(frame_count)],
            "left_wing_angle_deg": [10.0] * frame_count,
            "right_wing_angle_deg": [80.0] * frame_count,
            "wing_asymmetry_deg": [70.0] * frame_count,
            "wing_angle_change_deg_s": [300.0] * frame_count,
            "distance_body_lengths": [1.0] * frame_count,
            "heading_error_deg": [20.0] * frame_count,
            "male_speed_body_lengths_s": [0.4] * frame_count,
            "front_leg_to_female_body_body_lengths": [0.2] * frame_count,
            "mouth_to_female_posterior_body_lengths": [0.2] * frame_count,
            "abdomen_bending_angle_deg": [120.0] * frame_count,
            "copulation_distance_body_lengths": [0.8] * frame_count,
            "male_to_female_posterior_angle_deg": [20.0] * frame_count,
            "female_posterior_angle_deg": [160.0] * frame_count,
            "relative_speed_body_lengths_s": [0.05] * frame_count,
            "tracking_min_score": [0.95] * frame_count,
        }
    )

    events = detect_behaviors(features, CourtshipConfig())

    assert set(events["behavior"]) == {
        "orientation",
        "chasing",
        "tapping",
        "wing_extension",
        "wing_vibration",
        "licking",
        "abdomen_bending",
        "copulation_attempt",
        "copulation",
    }
