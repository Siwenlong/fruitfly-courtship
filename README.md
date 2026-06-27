# Fruit Fly Courtship Analysis

This project detects Level 1 male Drosophila courtship behaviors from tracked pose coordinates:

- `wing_extension`
- `chasing`
- `copulation_attempt`

The pipeline expects pose tracking to be performed upstream with SLEAP, DeepLabCut, or another tracker. Export or convert the tracked coordinates into the long CSV format below.

## Input Pose CSV

Required columns:

```text
video_id,frame,time_s,individual,keypoint,x,y,score
```

Supported individuals:

```text
male
female
```

Required keypoints per individual:

```text
head
thorax
abdomen_tip
left_wing_tip
right_wing_tip
```

Example rows:

```csv
video_id,frame,time_s,individual,keypoint,x,y,score
fly_001,0,0.000,male,head,101.2,88.4,0.95
fly_001,0,0.000,male,thorax,100.8,102.1,0.96
fly_001,0,0.000,male,abdomen_tip,100.5,118.0,0.94
fly_001,0,0.000,male,left_wing_tip,91.0,103.0,0.91
fly_001,0,0.000,male,right_wing_tip,122.0,103.4,0.93
```

If `time_s` is not present, pass `fps` in the YAML configuration so timestamps can be derived from `frame`.

## Run Analysis

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the Level 1 detector:

```bash
fruitfly-courtship analyze \
  --pose-csv data/pose/fly_pose.csv \
  --config config/level1_rules.yaml \
  --out-dir outputs
```

Equivalent module command:

```bash
python -m fruitfly_courtship.cli analyze \
  --pose-csv data/pose/fly_pose.csv \
  --config config/level1_rules.yaml \
  --out-dir outputs
```

## Outputs

```text
outputs/events.csv
outputs/summary.csv
outputs/features.csv
outputs/qc/tracking_quality.csv
outputs/qc/<video_id>_timeline.png
```

`events.csv` contains one row per detected behavior bout:

```text
video_id,behavior,start_frame,end_frame,start_time_s,end_time_s,duration_s,confidence,method
```

`summary.csv` contains one row per video and behavior:

```text
video_id,behavior,event_count,total_duration_s,latency_to_first_s,mean_event_duration_s
```

## Rule Tuning

Edit `config/level1_rules.yaml` to tune duration, distance, angle, and tracking-confidence thresholds. For a first validation pass, adjust one threshold at a time and inspect the generated timeline PNGs against manually reviewed clips.

## Validation Recommendation

Start with 5 to 10 videos and manually review obvious examples of:

- `wing_extension`
- `chasing`
- `copulation_attempt`
- background intervals with no target behavior

Treat `orientation`, `tapping`, `licking`, `abdomen_bending`, `copulation`, and `courtship_song` as annotation labels until the video resolution, frame rate, and optional audio are sufficient to detect them reliably.
