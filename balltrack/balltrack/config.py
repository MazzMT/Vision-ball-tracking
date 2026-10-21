"""Every tunable parameter in one place.

Sizes are relative to the estimated ball or to the frame wherever possible, so
values tuned on SoccerNet transfer to a new clip.
Override from the command line with  --set name=value  (repeatable).
"""
import ast
from dataclasses import dataclass, fields


@dataclass
class Config:
    proc_width: int = 1280            # wider frames are downscaled for processing
    workers: int = 10                  # parallel processes for detection; 0 = all CPU cores, 1 = off
    ball_color: tuple = None          # (R, G, B) start colour; None = generic white ball

    # 1. Shot changes (histogram analysis)
    cut_min: float = 0.06             # minimum histogram distance for a hard cut
    cut_k: float = 4.0                # ... and k times the local median distance
    fade_lo: float = 0.04             # per-frame change that may belong to a dissolve/wipe
    fade_sum: float = 0.30            # total change needed over the dissolve/wipe
    fade_len: tuple = (4, 30)         # allowed dissolve/wipe length (frames)
    min_shot_len: int = 5             # shorter shots are treated as transitions

    # 2. Ball size from player size
    player_ratio: float = 8.0         # player height / ball diameter (1.75 m / 0.22 m)
    min_radius: float = 2.2           # smallest ball radius (px) considered
    min_players: int = 5              # player blobs needed to trust the estimate
    fallback_radius: tuple = (0.003, 0.03)   # radius range / frame width if no players
    size_range: tuple = (0.5, 1.8)    # Hough searches radii from/to these multiples of the estimate
    size_tol: float = 0.7             # size tolerance (log scale) in the score

    # 3. Green channel prior
    green_margin: int = 8             # a pixel is grass if G > R + this and G > B + this
    green_full: float = 0.85          # share of grass around a candidate that counts as fully on grass
    green_skip: float = 0.2           # circles with less grass nearby are skipped before measuring (speed)

    # 4. Hough circles + calibrated colour
    hough_radius: float = 6.0         # frame is rescaled so the ball radius is ~this many px
    hough_param1: float = 100.0       # Canny upper threshold
    hough_param2: float = 7.0         # accumulator votes needed (lower = more circles)
    max_circles: int = 300            # Hough circles measured per frame (best colour match first)
    max_candidates: int = 20          # candidates kept per frame
    contrast_ref: float = 50.0        # Lab distance at which disc/ring contrast saturates
    spill_ref: float = 0.30           # share of the ring that may look like the ball
    graphic_share: float = 0.6        # a candidate on the same spot in more of a shot's frames is a logo

    # 5. Search field + 6. DP visibility
    candidates: int = 12              # best candidates per frame passed to the DP
    min_score: float = 0.25           # candidates below this are ignored
    search_radius: float = 8.0        # search radius in ball diameters just after a detection
    search_grow: float = 1.5          # ... growth per frame without a detection
    max_gap: int = 30                 # after this many missed frames, search the whole frame
    no_ball_cost: float = 0.8         # DP cost per frame of "ball not visible"
    drop_cost: float = 0.03           # extra cost to switch from a track to "not visible"
    new_track_cost: float = 0.7       # cost of starting a track anywhere
    fit_bonus: float = 0.25           # reward for a candidate inside the search field
    min_run: int = 5                  # visible runs shorter than this are dropped
    recalib_conf: float = 0.6         # detections used to re-calibrate ball colour and size


def load_config(overrides=(), ball_color=None):
    cfg = Config()
    if ball_color:
        cfg.ball_color = tuple(int(v) for v in ball_color.split(","))
    names = {f.name for f in fields(cfg)}
    for item in overrides or ():
        name, _, value = item.partition("=")
        if name not in names:
            raise SystemExit(f"unknown parameter '{name}'. Valid: {sorted(names)}")
        setattr(cfg, name, ast.literal_eval(value))
    return cfg
