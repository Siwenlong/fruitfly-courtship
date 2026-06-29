from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WingExtensionRule:
    angle_deg: float = 55.0
    asymmetry_deg: float = 25.0
    min_duration_s: float = 0.2
    merge_gap_s: float = 0.1


@dataclass(frozen=True)
class OrientationRule:
    max_distance_body_lengths: float = 5.0
    max_heading_error_deg: float = 70.0
    min_duration_s: float = 0.2
    merge_gap_s: float = 0.2


@dataclass(frozen=True)
class ChasingRule:
    max_distance_body_lengths: float = 4.0
    max_heading_error_deg: float = 60.0
    min_speed_body_lengths_s: float = 0.25
    min_duration_s: float = 0.5
    merge_gap_s: float = 0.2


@dataclass(frozen=True)
class TappingRule:
    max_front_leg_distance_body_lengths: float = 0.45
    min_duration_s: float = 0.05
    merge_gap_s: float = 0.1


@dataclass(frozen=True)
class WingVibrationRule:
    min_wing_angle_deg: float = 55.0
    min_wing_angle_change_deg_s: float = 250.0
    min_duration_s: float = 0.1
    merge_gap_s: float = 0.1


@dataclass(frozen=True)
class LickingRule:
    max_mouth_to_female_posterior_body_lengths: float = 0.6
    min_duration_s: float = 0.1
    merge_gap_s: float = 0.1


@dataclass(frozen=True)
class AbdomenBendingRule:
    max_bending_angle_deg: float = 145.0
    min_duration_s: float = 0.1
    merge_gap_s: float = 0.1


@dataclass(frozen=True)
class CopulationAttemptRule:
    max_distance_body_lengths: float = 1.5
    max_male_to_female_posterior_angle_deg: float = 70.0
    min_female_posterior_angle_deg: float = 120.0
    max_relative_speed_body_lengths_s: float = 1.5
    min_duration_s: float = 0.1
    merge_gap_s: float = 0.2


@dataclass(frozen=True)
class CopulationRule:
    max_distance_body_lengths: float = 1.2
    max_male_to_female_posterior_angle_deg: float = 60.0
    min_female_posterior_angle_deg: float = 130.0
    max_relative_speed_body_lengths_s: float = 0.25
    min_duration_s: float = 1.0
    merge_gap_s: float = 0.5


@dataclass(frozen=True)
class CourtshipConfig:
    fps: float | None = None
    min_keypoint_score: float = 0.3
    orientation: OrientationRule = field(default_factory=OrientationRule)
    wing_extension: WingExtensionRule = field(default_factory=WingExtensionRule)
    chasing: ChasingRule = field(default_factory=ChasingRule)
    tapping: TappingRule = field(default_factory=TappingRule)
    wing_vibration: WingVibrationRule = field(default_factory=WingVibrationRule)
    licking: LickingRule = field(default_factory=LickingRule)
    abdomen_bending: AbdomenBendingRule = field(default_factory=AbdomenBendingRule)
    copulation_attempt: CopulationAttemptRule = field(default_factory=CopulationAttemptRule)
    copulation: CopulationRule = field(default_factory=CopulationRule)


def _field_names(dataclass_type: type[Any]) -> set[str]:
    return {item.name for item in fields(dataclass_type)}


def _replace_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    allowed = _field_names(type(instance))
    for key in updates:
        if key not in allowed:
            raise ValueError(f"Unknown config key: {key}")
    return replace(instance, **updates)


def _apply_section(instance: Any, section_name: str, values: dict[str, Any]) -> Any:
    current_section = getattr(instance, section_name)
    updated_section = _replace_dataclass(current_section, values)
    return replace(instance, **{section_name: updated_section})


def load_config(path: str | Path | None) -> CourtshipConfig:
    config = CourtshipConfig()
    if path is None:
        return config

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a mapping at the top level")

    top_level_updates: dict[str, Any] = {}
    section_names = {
        "orientation",
        "wing_extension",
        "chasing",
        "tapping",
        "wing_vibration",
        "licking",
        "abdomen_bending",
        "copulation_attempt",
        "copulation",
    }
    allowed_top_level = _field_names(CourtshipConfig)

    for key, value in raw.items():
        if key not in allowed_top_level:
            raise ValueError(f"Unknown config key: {key}")
        if key in section_names:
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{key}' must be a mapping")
            config = _apply_section(config, key, value)
        else:
            top_level_updates[key] = value

    if top_level_updates:
        config = _replace_dataclass(config, top_level_updates)

    return config
