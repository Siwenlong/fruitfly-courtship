import pandas as pd
import pytest

from fruitfly_courtship.intervals import frames_to_events


def test_frames_to_events_removes_short_events_and_merges_close_events():
    frames = pd.DataFrame(
        {
            "video_id": ["fly_001"] * 8,
            "frame": list(range(8)),
            "time_s": [frame / 10 for frame in range(8)],
            "tracking_min_score": [0.9] * 8,
        }
    )
    mask = pd.Series([False, True, True, False, True, True, True, False])

    events = frames_to_events(
        frames=frames,
        mask=mask,
        behavior="wing_extension",
        fps=10.0,
        min_duration_s=0.2,
        merge_gap_s=0.15,
        method="rules_v1",
    )

    assert len(events) == 1
    event = events.iloc[0]
    assert event["behavior"] == "wing_extension"
    assert event["start_time_s"] == pytest.approx(0.1)
    assert event["end_time_s"] == pytest.approx(0.7)
    assert event["duration_s"] == pytest.approx(0.6)
    assert event["confidence"] == pytest.approx(0.9)
