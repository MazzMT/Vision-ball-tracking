# balltrack

Classical soccer-ball tracking for broadcast video (SoccerNet and similar).
No neural networks: OpenCV, NumPy and SciPy only. Handles every kind of shot,
including close-ups and slow-motion replays.

## Install

```
pip install opencv-python numpy scipy
```

The labeling tool opens a window, so it needs `opencv-python`, not
`opencv-python-headless`. Tracking and evaluation work with either.

## Workflow

```
# 1. track: writes out/<clip>_ball.csv and out/<clip>_annotated.mp4 (green box on the ball)
python -m balltrack track clip.mp4

# 2. label ground truth by clicking (resumable; progress is saved on every click)
python -m balltrack label clip.mp4 --out clip_gt.csv

# 3. evaluate
python -m balltrack evaluate out/clip_ball.csv clip_gt.csv

# optional: video with predictions (green box) and ground truth (yellow cross)
python -m balltrack render clip.mp4 out/clip_ball.csv --gt clip_gt.csv --out check.mp4

# optional: just list the detected shots
python -m balltrack shots clip.mp4
```

For a fair result on the new 1-minute clip, tune parameters on your SoccerNet
clips, freeze them, then run `track` once on the new clip and evaluate it. Tuning
on the clip you report on makes the numbers meaningless.

## How it works

| Step | What happens | Where |
|---|---|---|
| 1. Shot changes | HSV histograms on a 4x4 grid per frame. A **hard cut** is a jump that is large in absolute terms and compared with the local median (so pans don't trigger it). A **dissolve or logo wipe** is a run of moderate jumps with a large total; those frames are not tracked. Each shot is tracked independently. | `shots.py` |
| 2. Ball size from players | Players are non-grass blobs that do not touch the image border. Ball diameter = player height / 8, using the 75th percentile of blob heights, because blobs are often fragments (a head, a torso) that only err short. Hough then searches 0.5–1.8× that size, since the estimate is rough. If fewer than 5 players are found (close-ups), a wide size range is used instead. | `tracker.calibrate` |
| 3. Green channel prior | A ball in play is seen against grass. For every candidate, the share of the ring around it that is grass (pixels where G is clearly above both R and B) multiplies its score. Socks, shorts, the goal net, spare balls by the boards, scoreboard text and crowd specks are not surrounded by grass, so they score near zero. | `tracker.measure`, `tracker.score` |
| 4. Hough + calibrated colour | The frame is rescaled so the ball is about 6 px in radius, then `cv2.HoughCircles`. A small ball gets few Hough votes, so circles are ranked by how well their centre matches the ball colour, not by votes. Each circle is snapped onto the ball-coloured blob, then scored on colour (starting from a warm-white prior), contrast with its surroundings, size, **isolation** (a ball's colour ends at its edge, while lines and crowd dots continue past the circle) and the green prior. The factors are multiplied, so each one must be good. A candidate on the same screen spot (within 2 px) in more than 60% of a shot's frames is a logo or scoreboard and is dropped; the ball never stays that still, even when the camera follows it. | `tracker.detect`, `tracker.score` |
| 5. Search field | The expected position is the last position plus velocity. The search field is deliberately generous (4 ball diameters, growing by 1.5 diameters per frame the ball is not seen), so camera pans and sudden kicks stay inside it; after 30 frames the whole frame is searched. | `tracker.link` |
| 6. Visibility | Viterbi DP over the best 12 candidates per frame plus a **no-ball** state. The search field is the transition cost, so a ball that follows a smooth path beats look-alikes that jump between players, and later evidence can correct earlier mistakes. Visible runs shorter than 5 frames are dropped. Then ball colour and size are re-calibrated from the confident detections and the DP is run again. | `tracker.link`, `tracker.pick` |
| 7. Output | CSV with x, y per frame, and a video with a green box on the ball. | `video.py` |

## Files

```
balltrack/config.py    every parameter, with comments
balltrack/shots.py     step 1
balltrack/tracker.py   steps 2-6
balltrack/video.py     video and CSV reading/writing, annotated video
balltrack/label.py     ground-truth labeling tool
balltrack/evaluate.py  metrics
examples/make_synthetic_clip.py   synthetic clip with exact ground truth (smoke test)
```

## Output CSV (`*_ball.csv`)

One row per frame:

| column | meaning |
|---|---|
| `frame`, `time` | frame index (from 0) and seconds |
| `shot_id` | shot number; -1 for transition frames (dissolves, wipes) |
| `visible` | 1 if the ball was found |
| `x`, `y`, `radius` | ball centre and radius in original-video pixels (empty if not visible) |
| `confidence` | 0 to 1: the appearance score of the chosen candidate |
| `ball_diameter` | the shot's estimated ball diameter in pixels |
| `green_frac` | share of the frame that is pitch (used to group results by shot type) |

## Labeling tool

| input | action |
|---|---|
| left click | ball centre; jumps ahead 5 frames (`--step`), never past a cut |
| right click or `x` | ball not visible on this frame |
| `a` / `d` | previous / next frame |
| `w` / `s` | forward / back one step |
| `n` | next shot |
| `u` | remove this frame's label |
| `-` / `=` | smaller / larger ball circle; the diameter is stored with each click |
| `q` or Esc | save and quit |

Positions are interpolated between two labelled frames of the same shot, never
across a cut. Between a "visible" and a "not visible" label the frames stay
unknown and are not scored, so when the ball goes behind a player or out of
frame, right-click the frames where it is hidden. A magnifier shows the area
around the cursor, and the bar at the bottom shows labelled frames (green
visible, red not visible) and cuts (white). Setting the circle to the ball's size
matters because errors are measured in ball diameters.

The ground-truth CSV has columns `frame, status, x, y, diameter, source`, where
status is `visible`, `not_visible` or `unknown`.

## Metrics

Only frames with a known label are scored. Errors are in **ball diameters**, so
the same threshold means the same thing in a wide shot and in a close-up.

* `found`: share of ball-visible frames where the tracker reported a ball
* `false_alarm`: share of no-ball frames where it reported one anyway
* `median_err`: median distance from the true centre, when both agree the ball is visible
* `F1@t`: a detection counts if it is within t diameters

Results are printed overall, by shot type (wide / medium / close-up, from the
share of pitch in the frame), for transitions, and per shot.

## Tuning

Change any parameter without editing code: `--set name=value` (repeatable).
For example `python -m balltrack track clip.mp4 --set no_ball_cost=0.6 --set min_run=3`.
If the ball isn't white, give its colour: `--ball-color 255,220,0`.

The ones most worth tuning on your SoccerNet clips:

| parameter | effect |
|---|---|
| `no_ball_cost` | lower = reports the ball more often (more found, more false alarms) |
| `min_score` | candidates below this are ignored |
| `spill_ref` | how strict the isolation test is; raise it if balls near lines or feet are missed |
| `candidates` | candidates per frame given to the DP; more helps in crowded scenes |
| `green_margin`, `green_full` | green prior: how green a pixel must be, and how much grass around a candidate counts as fully on grass |
| `search_radius`, `search_grow` | how far the ball may move between frames, in diameters |
| `hough_param2` | lower = more Hough circles (higher recall, slower) |
| `workers` | parallel processes for detection; 0 (default) uses all CPU cores, 1 turns it off |
| `player_ratio` | player height / ball diameter |
| `cut_min`, `cut_k`, `fade_sum` | shot-change sensitivity; check with `python -m balltrack shots` |

## Testing on SoccerNet clips with ground truth

[SoccerNet-Tracking](https://arxiv.org/abs/2204.06918) has 30-second 1080p clips in which the ball has a box in every frame. They are from the main camera only, with no replays, so they test open play. The script downloads a few sequences and converts each to a video plus a ground-truth CSV:

```
pip install huggingface_hub
python examples/soccernet_tracking.py soccernet --count 3
python -m balltrack track soccernet/SNMOT-116.mp4
python -m balltrack evaluate out/SNMOT-116_ball.csv soccernet/SNMOT-116_gt.csv
```

(use the names the script prints). It reads only the files it needs from the
archive. If you already have the sequences unzipped, use
`--from-folder path/to/test` instead. Frames without a ball box count as "ball
not visible". `huggingface_hub` is only needed for this download, not for the tracker.

## Smoke test

`examples/make_synthetic_clip.py` draws a 390-frame clip with a wide panning
shot, a cut to a zoomed camera, a logo wipe and a slow-motion replay. It includes
an occlusion, a lofted pass, white socks, crowd dots, pitch lines and a
scoreboard.

```
python examples/make_synthetic_clip.py synth
python -m balltrack track synth/synth.mp4
python -m balltrack evaluate out/synth_ball.csv synth/synth_gt.csv
```

Result: shots and the wipe detected exactly, ball found in 77% of ball-visible
frames, no false alarms, no detections in the wrong place, median error 0.05
diameters, F1 0.87. Most misses are in the zoomed shot and where the ball
touches a player. This only shows that the pieces work together. Real
footage has compression, motion blur and real occlusions, so expect lower
numbers on SoccerNet.

## Known limitations

* **Resolution matters most.** At 640x360 the ball is about 3 px across and
  looks like a white sock or a patch of the goal net. Use the highest
  resolution you can get (720p or 1080p); the tracker downscales anything wider
  than 1280 px itself.

* A ball that touches a player, a white line or the crowd fails the isolation
  test, and a ball in the air against the crowd fails the green prior. The DP bridges short gaps, but long contact is lost.
* Fast kicks blur the ball into a streak that Hough may miss; the search field
  re-acquires it afterwards.
* In shots with few visible players the size is unknown until the first
  confident detections, so the start of such shots is weaker.
