from __future__ import annotations

import argparse
from pathlib import Path

from fruitfly_courtship.config import load_config
from fruitfly_courtship.detectors import detect_behaviors
from fruitfly_courtship.features import extract_features
from fruitfly_courtship.pose_io import load_pose_csv, required_keypoints_present
from fruitfly_courtship.reporting import write_outputs, write_tracking_quality


LEVEL1_KEYPOINTS = ["head", "thorax", "abdomen_tip", "left_wing_tip", "right_wing_tip"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fruitfly-courtship")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Detect Level 1 courtship events from pose CSV")
    analyze.add_argument("--pose-csv", required=True, help="Long-format pose CSV")
    analyze.add_argument("--config", default=None, help="YAML rule configuration")
    analyze.add_argument("--out-dir", required=True, help="Output directory")
    return parser


def _format_missing_keypoints(missing: dict[str, list[str]]) -> str:
    parts = [f"{individual}: {', '.join(keypoints)}" for individual, keypoints in missing.items()]
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        config = load_config(args.config)
        pose = load_pose_csv(args.pose_csv, fps=config.fps)
        missing = required_keypoints_present(pose, LEVEL1_KEYPOINTS)
        if missing:
            raise ValueError(f"Missing required Level 1 keypoints: {_format_missing_keypoints(missing)}")

        features = extract_features(pose)
        events = detect_behaviors(features, config)
        out_dir = Path(args.out_dir)
        write_outputs(events=events, features=features, out_dir=out_dir)
        write_tracking_quality(pose=pose, out_dir=out_dir, min_keypoint_score=config.min_keypoint_score)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
