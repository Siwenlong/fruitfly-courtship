from pathlib import Path

import pytest

from fruitfly_courtship.config import CourtshipConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_contains_level1_rules():
    config = load_config(None)

    assert isinstance(config, CourtshipConfig)
    assert config.min_keypoint_score == pytest.approx(0.3)
    assert config.orientation.max_heading_error_deg == pytest.approx(70.0)
    assert config.wing_extension.angle_deg == pytest.approx(55.0)
    assert config.chasing.max_distance_body_lengths == pytest.approx(4.0)
    assert config.tapping.max_front_leg_distance_body_lengths == pytest.approx(0.45)
    assert config.wing_vibration.min_wing_angle_change_deg_s == pytest.approx(250.0)
    assert config.licking.max_mouth_to_female_posterior_body_lengths == pytest.approx(0.6)
    assert config.abdomen_bending.max_bending_angle_deg == pytest.approx(145.0)
    assert config.copulation_attempt.min_female_posterior_angle_deg == pytest.approx(120.0)
    assert config.copulation.min_duration_s == pytest.approx(1.0)


def test_yaml_overrides_nested_values(tmp_path: Path):
    config_path = tmp_path / "rules.yaml"
    config_path.write_text(
        "\n".join(
            [
                "fps: 30",
                "min_keypoint_score: 0.25",
                "wing_extension:",
                "  angle_deg: 62",
                "  min_duration_s: 0.3",
                "chasing:",
                "  max_heading_error_deg: 45",
                "tapping:",
                "  max_front_leg_distance_body_lengths: 0.35",
                "wing_vibration:",
                "  min_wing_angle_change_deg_s: 300",
                "licking:",
                "  max_mouth_to_female_posterior_body_lengths: 0.5",
                "abdomen_bending:",
                "  max_bending_angle_deg: 135",
                "copulation_attempt:",
                "  max_distance_body_lengths: 1.2",
                "copulation:",
                "  min_duration_s: 2.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.fps == pytest.approx(30.0)
    assert config.min_keypoint_score == pytest.approx(0.25)
    assert config.wing_extension.angle_deg == pytest.approx(62.0)
    assert config.wing_extension.min_duration_s == pytest.approx(0.3)
    assert config.chasing.max_heading_error_deg == pytest.approx(45.0)
    assert config.tapping.max_front_leg_distance_body_lengths == pytest.approx(0.35)
    assert config.wing_vibration.min_wing_angle_change_deg_s == pytest.approx(300.0)
    assert config.licking.max_mouth_to_female_posterior_body_lengths == pytest.approx(0.5)
    assert config.abdomen_bending.max_bending_angle_deg == pytest.approx(135.0)
    assert config.copulation_attempt.max_distance_body_lengths == pytest.approx(1.2)
    assert config.copulation_attempt.min_female_posterior_angle_deg == pytest.approx(120.0)
    assert config.copulation.min_duration_s == pytest.approx(2.0)


def test_unknown_config_section_raises_clear_error(tmp_path: Path):
    config_path = tmp_path / "rules.yaml"
    config_path.write_text("unknown_section:\n  value: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(config_path)


def test_unknown_nested_config_key_raises_clear_error(tmp_path: Path):
    config_path = tmp_path / "rules.yaml"
    config_path.write_text("wing_extension:\n  bogus: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(config_path)


def test_empty_config_file_loads_defaults(tmp_path: Path):
    config_path = tmp_path / "rules.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_config(config_path) == CourtshipConfig()


@pytest.mark.parametrize("yaml_text", ["false", "0", "[]"])
def test_falsy_non_mapping_config_raises_clear_error(tmp_path: Path, yaml_text: str):
    config_path = tmp_path / "rules.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Config file must contain a mapping at the top level"):
        load_config(config_path)


def test_checked_in_level1_rules_match_defaults():
    assert load_config(PROJECT_ROOT / "config" / "level1_rules.yaml") == CourtshipConfig()


def test_checked_in_v2_rules_match_defaults():
    assert load_config(PROJECT_ROOT / "config" / "v2_rules.yaml") == CourtshipConfig()
