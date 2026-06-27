from __future__ import annotations

import pandas as pd

from fruitfly_courtship.config import CourtshipConfig
from fruitfly_courtship.intervals import EVENT_COLUMNS, frames_to_events
from fruitfly_courtship.pose_io import infer_fps


def _valid_tracking(features: pd.DataFrame, min_keypoint_score: float) -> pd.Series:
    return features["tracking_min_score"].ge(min_keypoint_score)


def _wing_extension_mask(features: pd.DataFrame, config: CourtshipConfig) -> pd.Series:
    rule = config.wing_extension
    left_extended = features["left_wing_angle_deg"].ge(rule.angle_deg)
    right_extended = features["right_wing_angle_deg"].ge(rule.angle_deg)
    asymmetric = features["wing_asymmetry_deg"].ge(rule.asymmetry_deg)
    return (left_extended | right_extended) & asymmetric & _valid_tracking(features, config.min_keypoint_score)


def _chasing_mask(features: pd.DataFrame, config: CourtshipConfig) -> pd.Series:
    rule = config.chasing
    return (
        features["distance_body_lengths"].le(rule.max_distance_body_lengths)
        & features["heading_error_deg"].le(rule.max_heading_error_deg)
        & features["male_speed_body_lengths_s"].ge(rule.min_speed_body_lengths_s)
        & _valid_tracking(features, config.min_keypoint_score)
    )


def _copulation_attempt_mask(features: pd.DataFrame, config: CourtshipConfig) -> pd.Series:
    rule = config.copulation_attempt
    return (
        features["copulation_distance_body_lengths"].le(rule.max_distance_body_lengths)
        & features["male_to_female_posterior_angle_deg"].le(rule.max_male_to_female_posterior_angle_deg)
        & features["female_posterior_angle_deg"].ge(rule.min_female_posterior_angle_deg)
        & features["relative_speed_body_lengths_s"].le(rule.max_relative_speed_body_lengths_s)
        & _valid_tracking(features, config.min_keypoint_score)
    )


def detect_behaviors(features: pd.DataFrame, config: CourtshipConfig) -> pd.DataFrame:
    fps = float(config.fps) if config.fps is not None else infer_fps(features)
    event_tables: list[pd.DataFrame] = []

    for _, group in features.groupby("video_id", sort=False):
        ordered = group.sort_values("frame").reset_index(drop=True)
        event_tables.append(
            frames_to_events(
                frames=ordered,
                mask=_wing_extension_mask(ordered, config),
                behavior="wing_extension",
                fps=fps,
                min_duration_s=config.wing_extension.min_duration_s,
                merge_gap_s=config.wing_extension.merge_gap_s,
                method="rules_v1",
            )
        )
        event_tables.append(
            frames_to_events(
                frames=ordered,
                mask=_chasing_mask(ordered, config),
                behavior="chasing",
                fps=fps,
                min_duration_s=config.chasing.min_duration_s,
                merge_gap_s=config.chasing.merge_gap_s,
                method="rules_v1",
            )
        )
        event_tables.append(
            frames_to_events(
                frames=ordered,
                mask=_copulation_attempt_mask(ordered, config),
                behavior="copulation_attempt",
                fps=fps,
                min_duration_s=config.copulation_attempt.min_duration_s,
                merge_gap_s=config.copulation_attempt.merge_gap_s,
                method="rules_v1",
            )
        )

    non_empty = [table for table in event_tables if not table.empty]
    if not non_empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    events = pd.concat(non_empty, ignore_index=True)
    return events.sort_values(["video_id", "start_time_s", "behavior"]).reset_index(drop=True)
