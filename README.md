# Fruit Fly Courtship Analysis

This project detects male Drosophila courtship behaviors from tracked pose coordinates.
The v2 rule set follows the common courtship sequence:

- `orientation`
- `chasing`
- `tapping`
- `wing_extension`
- `wing_vibration`
- `licking`
- `abdomen_bending`
- `copulation_attempt`
- `copulation`

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

Optional male keypoints improve v2 calls:

```text
left_front_leg_tip
right_front_leg_tip
proboscis
```

Accepted aliases include `left_foreleg_tip`, `right_foreleg_tip`, `left_front_tarsus`,
`right_front_tarsus`, `mouthparts`, and `mouth`.

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

Run the v2 courtship detector:

```bash
fruitfly-courtship analyze \
  --pose-csv data/pose/fly_pose.csv \
  --config config/v2_rules.yaml \
  --out-dir outputs
```

Equivalent module command:

```bash
python -m fruitfly_courtship.cli analyze \
  --pose-csv data/pose/fly_pose.csv \
  --config config/v2_rules.yaml \
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

Edit `config/v2_rules.yaml` to tune duration, distance, angle, and tracking-confidence thresholds. For a first validation pass, adjust one threshold at a time and inspect the generated timeline PNGs against manually reviewed clips.

## Validation Recommendation

Start with 5 to 10 videos and manually review obvious examples of:

- `orientation`
- `chasing`
- `tapping`
- `wing_extension`
- `wing_vibration`
- `licking`
- `abdomen_bending`
- `copulation_attempt`
- `copulation`
- background intervals with no target behavior

Treat `wing_vibration` as a video proxy for courtship song unless synchronized audio is available.
`tapping`, `licking`, and `abdomen_bending` are most reliable when the optional front-leg,
mouth/proboscis, and abdomen keypoints are labeled consistently.
