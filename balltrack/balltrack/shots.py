"""1. Shot changes from colour-histogram analysis.

Each frame is summarised by HSV histograms on a 4x4 grid of cells (so two
different wide cameras, which share a global histogram, still differ).
  * hard cut      : a single-frame jump that is large, and large compared with the
                    local median jump (so fast pans do not trigger it).
  * dissolve/wipe : a run of moderate jumps whose total is large. These frames
                    become a 'transition' segment and are not tracked.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from .video import read_frames

GRID, BINS = 4, (12, 6, 6)


@dataclass
class Segment:
    start: int
    end: int        # exclusive
    shot_id: int    # -1 for transitions


def signature(frame):
    small = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.0), cv2.COLOR_BGR2HSV)
    cells = []
    for rows in np.array_split(hsv, GRID, axis=0):
        for cell in np.array_split(rows, GRID, axis=1):
            h = cv2.calcHist([cell], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
            cells.append(gaussian_filter(h, 1.2, mode=("wrap", "nearest", "nearest")))  # soft binning
    sig = np.array(cells).reshape(GRID * GRID, -1)
    return sig / sig.sum(axis=1, keepdims=True)


def distance(a, b):
    """Mean Hellinger distance over the grid cells (0 = identical, 1 = disjoint)."""
    return np.sqrt(np.clip(1 - np.sqrt(a * b).sum(-1), 0, 1)).mean(-1)


def segment(sigs, cfg):
    n = len(sigs)
    d = np.zeros(n)
    d[1:] = distance(sigs[1:], sigs[:-1])
    local = median_filter(d, size=25, mode="nearest")
    cut = (d > cfg.cut_min) & (d > cfg.cut_k * np.maximum(local, 0.01))

    fade = np.zeros(n, bool)
    moving = (d > cfg.fade_lo) & ~cut
    t = 1
    while t < n:
        if moving[t]:
            e = t
            while e + 1 < n and (moving[e + 1] or (e + 2 < n and moving[e + 2])):
                e += 1
            if cfg.fade_len[0] <= e - t + 1 <= cfg.fade_len[1] and d[t:e + 1].sum() > cfg.fade_sum:
                fade[t:e + 1] = True
            t = e
        t += 1

    bounds = [0] + [t for t in range(1, n) if cut[t] or fade[t] != fade[t - 1]] + [n]
    segs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        is_shot = not fade[s] and e - s >= cfg.min_shot_len
        if not is_shot and segs and segs[-1].shot_id == -1:
            segs[-1].end = e                          # merge neighbouring transition pieces
            continue
        segs.append(Segment(s, e, sum(g.shot_id >= 0 for g in segs) if is_shot else -1))
    return segs


def segment_video(path, cfg):
    return segment(np.array([signature(f) for f in read_frames(path)]), cfg)
