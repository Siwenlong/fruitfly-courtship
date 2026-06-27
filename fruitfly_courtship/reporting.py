from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BEHAVIOR_COLORS = {
    "wing_extension": "#1f77b4",
    "chasing": "#2ca02c",
    "copulation_attempt": "#d62728",
}


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "video_id",
        "behavior",
        "event_count",
        "total_duration_s",
        "latency_to_first_s",
        "mean_event_duration_s",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        events.groupby(["video_id", "behavior"], as_index=False)
        .agg(
            event_count=("behavior", "size"),
            total_duration_s=("duration_s", "sum"),
            latency_to_first_s=("start_time_s", "min"),
            mean_event_duration_s=("duration_s", "mean"),
        )
        .sort_values(["video_id", "behavior"])
        .reset_index(drop=True)
    )
    return summary[columns]


def plot_timeline(events: pd.DataFrame, out_dir: str | Path) -> None:
    output_dir = Path(out_dir)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if events.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No detected events", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(qc_dir / "all_videos_timeline.png", dpi=150)
        plt.close(fig)
        return

    for video_id, group in events.groupby("video_id", sort=False):
        behaviors = list(dict.fromkeys(group["behavior"].tolist()))
        fig_height = max(2.0, 0.7 * len(behaviors) + 1.0)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y_positions = {behavior: index for index, behavior in enumerate(behaviors)}
        for _, event in group.iterrows():
            y = y_positions[event["behavior"]]
            ax.broken_barh(
                [(event["start_time_s"], event["duration_s"])],
                (y - 0.3, 0.6),
                facecolors=BEHAVIOR_COLORS.get(event["behavior"], "#7f7f7f"),
            )
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(behaviors)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{video_id} courtship event timeline")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(qc_dir / f"{video_id}_timeline.png", dpi=150)
        plt.close(fig)


def write_tracking_quality(pose: pd.DataFrame, out_dir: str | Path, min_keypoint_score: float) -> pd.DataFrame:
    output_dir = Path(out_dir)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    quality = (
        pose.assign(low_score=pose["score"] < min_keypoint_score)
        .groupby(["video_id", "individual", "keypoint"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            low_score_fraction=("low_score", "mean"),
        )
        .sort_values(["video_id", "individual", "keypoint"])
        .reset_index(drop=True)
    )
    quality.to_csv(qc_dir / "tracking_quality.csv", index=False)
    return quality


def write_outputs(events: pd.DataFrame, features: pd.DataFrame, out_dir: str | Path) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "events.csv", index=False)
    summarize_events(events).to_csv(output_dir / "summary.csv", index=False)
    features.to_csv(output_dir / "features.csv", index=False)
    plot_timeline(events, output_dir)
