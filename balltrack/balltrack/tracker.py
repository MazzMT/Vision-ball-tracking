"""Ball detection, steps 2-6 of the pipeline. Each shot is handled on its own:

  per shot    grass colour and ball size (from player size)
  per frame   Hough circles (best colour match first), each snapped onto the
              ball-coloured blob and scored on:
                colour     close to the (calibrated) ball colour
                contrast   stands out from its surroundings
                size       close to the expected ball size
                isolation  its colour ends at its edge (not part of a line or sock)
                green      GREEN CHANNEL PRIOR: the ring around it is grass, i.e.
                           green-dominant pixels (G clearly above R and B). A ball
                           in play is seen against the pitch; socks, shorts, the goal
                           net, spare balls by the boards and the crowd are not.
  per shot    candidates on the same screen pixel all shot long (logos) dropped,
              then Viterbi DP over the best candidates per frame + a 'no ball' state.
              The transition cost is a generous search field around where the
              ball should be, given its last position and velocity. Then ball
              colour and size are re-calibrated from the confident detections,
              and the DP is run again.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, repeat
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .shots import segment_video
from .video import PRED_COLUMNS, read_frames, video_info, write_csv

# Lab mean, sigma. Broadcast white balls are often lit warm (b > 128), hence the wider b.
WHITE = (np.array([225.0, 128.0, 138.0]), np.array([45.0, 14.0, 25.0]))
FEATURES = 9      # candidate columns: x, y, r, L, a, b, contrast, spill, green
CALIB_FRAMES = 50


# --------------------------------------------------------------------------
# 2. per-shot calibration: grass colour and ball size from player size
# --------------------------------------------------------------------------
def grass(hsv, hue):
    return cv2.inRange(hsv, (hue - 12, 40, 40), (hue + 12, 255, 255))


def player_heights(frame, hue):
    """Heights (px) of player-shaped non-grass blobs that do not touch the image
    border (crowd and boards do)."""
    mask = cv2.morphologyEx(grass(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), hue),
                            cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(255 - mask)
    H, W = mask.shape
    return [h for x, y, w, h, area in stats[1:]
            if x > 0 and y > 0 and x + w < W and y + h < H                  # not touching the border
            and 0.02 * H < h < 0.5 * H and 1.2 <= h / w <= 5 and area >= 0.3 * w * h]


def calibrate(frames, cfg):
    W = frames[0].shape[1]
    samples = frames[:: max(1, len(frames) // 8)]
    hsv = [cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
           for f in samples]
    px = np.concatenate([h.reshape(-1, 3) for h in hsv])
    greenish = (px[:, 0] >= 25) & (px[:, 0] <= 95) & (px[:, 1] >= 40) & (px[:, 2] >= 40)
    hue = int(np.bincount(px[greenish, 0]).argmax()) if greenish.mean() > 0.03 else None
    green_frac = 0.0 if hue is None else float(np.mean([(grass(h, hue) > 0).mean() for h in hsv]))

    heights = [h for f in samples for h in player_heights(f, hue)] if hue is not None else []
    if len(heights) >= cfg.min_players:
        # 75th percentile: blobs are often fragments (head, torso), so they only err short
        radius = max(cfg.min_radius, float(np.percentile(heights, 75)) / (2 * cfg.player_ratio))
        r_range, size_tol = (cfg.size_range[0] * radius, cfg.size_range[1] * radius), cfg.size_tol
    else:
        r_range = (cfg.fallback_radius[0] * W, cfg.fallback_radius[1] * W)
        radius, size_tol = float(np.sqrt(r_range[0] * r_range[1])), 1.0

    if cfg.ball_color:
        r, g, b = cfg.ball_color
        color = (cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float), WHITE[1] * 1.3)
    else:
        color = WHITE
    return SimpleNamespace(green_frac=green_frac, radius=radius, r_range=r_range, size_tol=size_tol,
                           players=len(heights) >= cfg.min_players, color=color,
                           scale=float(np.clip(cfg.hough_radius / radius, 0.25, 3.0)))


# --------------------------------------------------------------------------
# 3 + 4. candidates: Hough circles, measured and scored
# --------------------------------------------------------------------------
def measure(lab, green, x, y, r):
    """For a circle: median colour inside it, contrast with the surrounding ring,
    'spill' (share of the ring that looks like the disc), 'green' (share of the
    ring that is grass), and the centre/radius of the ball-coloured blob."""
    R = int(2.6 * r) + 1
    x0, y0 = int(x) - R, int(y) - R
    if x0 < 0 or y0 < 0 or x0 + 2 * R + 1 > lab.shape[1] or y0 + 2 * R + 1 > lab.shape[0]:
        return None
    patch = lab[y0:y0 + 2 * R + 1, x0:x0 + 2 * R + 1].reshape(-1, 3)
    yy, xx = (g.ravel() for g in np.mgrid[y0:y0 + 2 * R + 1, x0:x0 + 2 * R + 1])
    dist = np.hypot(xx - x, yy - y)
    disc, ring = dist <= 0.7 * r, (dist >= 1.5 * r) & (dist <= 2.5 * r)   # margin: Hough radii are approximate
    if disc.sum() < 3 or ring.sum() < 8:
        return None
    inner, outer = np.median(patch[disc], axis=0), np.median(patch[ring], axis=0)
    ball_like = np.linalg.norm(patch - inner, axis=1) < np.linalg.norm(patch - outer, axis=1)
    blob = ball_like & (dist <= 1.5 * r)
    if blob.sum() < 3:
        return None
    contrast = np.percentile(np.linalg.norm(patch[ring] - inner, axis=1), 25)
    on_grass = green[y0:y0 + 2 * R + 1, x0:x0 + 2 * R + 1].ravel()[ring].mean()
    return (*inner, contrast, ball_like[ring].mean(), on_grass,
            xx[blob].mean(), yy[blob].mean(), np.sqrt(blob.sum() / np.pi))


def detect(frame, S, cfg):
    """All candidates of one frame as rows of FEATURES (proc-resolution pixels)."""
    s = S.scale
    work = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    r0, r1 = S.r_range[0] * s, S.r_range[1] * s
    circles = cv2.HoughCircles(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, dp=1,
                               minDist=max(2.0, r0), param1=cfg.hough_param1, param2=cfg.hough_param2,
                               minRadius=max(1, int(r0)), maxRadius=int(np.ceil(r1)))
    if circles is None:
        return np.zeros((0, FEATURES))
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    b, g, r = (work[..., i].astype(np.int16) for i in range(3))
    green = ((g > r + cfg.green_margin) & (g > b + cfg.green_margin)).astype(np.float32)

    # Small balls get few Hough votes, so Hough's own ranking is useless here.
    # Rank every circle by how well its centre matches the ball colour instead.
    circles = circles[0, :, :3]
    cx = np.clip(circles[:, 0].astype(int), 0, work.shape[1] - 1)
    cy = np.clip(circles[:, 1].astype(int), 0, work.shape[0] - 1)
    mean, sigma = S.color
    order = np.argsort((((lab[cy, cx] - mean) / sigma) ** 2).mean(axis=1))
    # Hough returns many circles around the same spot (different radii): keep the
    # best-coloured one per spot, so the budget goes to different spots.
    cell = max(2, int(r0))
    _, first = np.unique((cy[order] // cell) * 100000 + cx[order] // cell, return_index=True)
    circles = circles[order[np.sort(first)]]
    # Green channel prior, cheap version first: skip circles with almost no grass
    # around them before measuring them properly (they would score ~0 anyway).
    k = int(5 * r0) | 1
    grass_near = cv2.blur(green, (k, k))
    circles = circles[grass_near[np.clip(circles[:, 1].astype(int), 0, work.shape[0] - 1),
                                 np.clip(circles[:, 0].astype(int), 0, work.shape[1] - 1)] >= cfg.green_skip]
    circles = circles[:cfg.max_circles]

    rows = []
    for x, y, rad in circles:
        m = measure(lab, green, x, y, rad)
        if m is None:
            continue
        x, y, rad = m[6], m[7], float(np.clip(m[8], r0, r1))   # snap position and size onto the blob
        m = measure(lab, green, x, y, rad)
        if m is not None:
            rows.append([x / s, y / s, rad / s, *m[:6]])
    rows = np.array(rows).reshape(-1, FEATURES)
    keep = []                                   # several circles often snap onto the same blob
    for i in np.argsort(-score(rows, S, cfg)):
        if len(keep) == cfg.max_candidates:
            break
        if len(keep) == 0 or np.all(np.hypot(*(rows[keep, :2] - rows[i, :2]).T) > rows[keep, 2]):
            keep.append(i)
    return rows[keep]


def score(c, S, cfg):
    """Score per candidate, 0..1. Each factor must be good: they are multiplied."""
    mean, sigma = S.color
    color = np.exp(-0.5 * (((c[:, 3:6] - mean) / sigma) ** 2).mean(axis=1))
    contrast = np.clip(c[:, 6] / cfg.contrast_ref, 0, 1)
    isolated = np.exp(-c[:, 7] / cfg.spill_ref)
    green = np.clip(c[:, 8] / cfg.green_full, 0, 1)
    size = np.exp(-0.5 * (np.log(c[:, 2] / S.radius) / S.size_tol) ** 2)
    return color * isolated * green * (contrast + 0.7 * size) / 1.7


# --------------------------------------------------------------------------
# 5 + 6. search field + Viterbi DP with a 'no ball' state
# --------------------------------------------------------------------------
def link(frames, ball_d, cfg):
    """frames: list of (xy (K,2), score (K,)). States per frame: the K candidates
    and NONE (last). Each state remembers the last seen position, velocity and
    frames since seen of its best path; that memory is the search field for the
    next frame: candidates near the predicted position are cheap, far ones cost
    as much as starting a new track. Returns the chosen candidate per frame (-1 = none)."""
    back = []
    for t, (xy, sc) in enumerate(frames):
        K = len(sc)
        node = 2.0 * (1.0 - sc)
        if t == 0:
            cost = np.r_[node + cfg.new_track_cost, cfg.no_ball_cost]
            pos, vel = np.r_[xy, [[0.0, 0.0]]], np.zeros((K + 1, 2))
            gap = np.r_[np.zeros(K), 1e9]
            back.append(np.full(K + 1, -1))
            continue
        known = gap < cfg.max_gap
        pred = pos + vel * (np.minimum(gap, cfg.max_gap) + 1)[:, None]          # constant-velocity guess
        radius = ball_d * (cfg.search_radius + cfg.search_grow * np.minimum(gap, cfg.max_gap))
        D = np.hypot(*(xy[None, :, :] - pred[:, None, :]).transpose(2, 0, 1))  # (states, K)
        trans = np.minimum(0.5 * (D / radius[:, None]) ** 2, cfg.new_track_cost + cfg.fit_bonus) - cfg.fit_bonus
        trans[~known] = cfg.new_track_cost
        total = cost[:, None] + trans
        best = total.argmin(axis=0)
        none_total = cost + np.where(gap == 0, cfg.drop_cost, 0.0)
        j = int(none_total.argmin())
        step = (xy - pos[best]) / (gap[best] + 1)[:, None]
        new_vel = np.where(known[best][:, None], 0.6 * step + 0.4 * vel[best], 0.0)
        cost = np.r_[total[best, np.arange(K)] + node, none_total[j] + cfg.no_ball_cost]
        pos, vel = np.r_[xy, pos[j:j + 1]], np.r_[new_vel, vel[j:j + 1]]
        gap = np.r_[np.zeros(K), gap[j] + 1]
        back.append(np.r_[best, j])

    path, s = np.full(len(frames), -1), int(cost.argmin())
    for t in range(len(frames) - 1, -1, -1):
        if s < len(frames[t][1]):
            path[t] = s
        s = back[t][s]
    return path


def pick(cands, S, cfg):
    """Best few candidates per frame -> DP -> (candidate index, score) per frame.
    Visible runs shorter than `min_run` frames are dropped (flicker)."""
    frames, kept = [], []
    for c in cands:
        sc = score(c, S, cfg)
        keep = [i for i in np.argsort(-sc)[:cfg.candidates] if sc[i] >= cfg.min_score]
        frames.append((c[keep, :2], sc[keep]))
        kept.append(keep)
    path = link(frames, 2 * S.radius, cfg)
    picks = [(kept[t][k], frames[t][1][k]) if k >= 0 else (-1, 0.0) for t, k in enumerate(path)]
    t = 0
    while t < len(picks):
        e = t
        while e < len(picks) and picks[e][0] >= 0:
            e += 1
        if 0 < e - t < cfg.min_run:
            picks[t:e] = [(-1, 0.0)] * (e - t)
        t = e + 1
    return picks


def recalibrate(cands, picks, S, cfg):
    """Ball colour and size from the shot's own confident detections."""
    rows = np.array([cands[t][k] for t, (k, conf) in enumerate(picks) if k >= 0 and conf >= cfg.recalib_conf])
    if len(rows) < 10:
        return False
    mean = np.median(rows[:, 3:6], axis=0)
    spread = 1.4826 * np.median(np.abs(rows[:, 3:6] - mean), axis=0)
    S.color = (mean, np.maximum([20.0, 6.0, 6.0], 2 * spread))
    S.radius, S.size_tol = float(np.median(rows[:, 2])), cfg.size_tol
    return True


def drop_screen_graphics(cands, cfg):
    """Logos and scoreboards sit on the same screen spot for the whole shot; the
    ball never does, even when the camera follows it. Drop candidates that have
    a candidate within 2 px of them in more than `graphic_share` of the frames."""
    pts = np.concatenate([c[:, :2] for c in cands])
    frame_of = np.concatenate([np.full(len(c), t) for t, c in enumerate(cands)])
    if len(pts) == 0:
        return cands
    near = cKDTree(pts).query_ball_point(pts, r=2.0)
    fixed = np.array([len(set(frame_of[n])) > cfg.graphic_share * len(cands) for n in near])
    out, i = [], 0
    for c in cands:
        out.append(c[~fixed[i:i + len(c)]])
        i += len(c)
    return out


def track_shot(frames, cfg, pool=None):
    """frames: list of one shot's frames. Returns (per-frame results, calibration).
    With a process pool, frames are detected in parallel."""
    S = calibrate(frames[:CALIB_FRAMES], cfg)
    if pool is None:
        cands = [detect(f, S, cfg) for f in frames]
    else:
        cands = list(pool.map(detect, frames, repeat(S), repeat(cfg), chunksize=8))
    cands = drop_screen_graphics(cands, cfg)
    picks = pick(cands, S, cfg)
    if recalibrate(cands, picks, S, cfg):
        picks = pick(cands, S, cfg)
    results = []
    for c, (k, conf) in zip(cands, picks):
        if k >= 0:
            results.append(dict(visible=1, x=c[k, 0], y=c[k, 1], radius=c[k, 2], confidence=conf))
        else:
            results.append(dict(visible=0, x=np.nan, y=np.nan, radius=np.nan, confidence=0.0))
    return results, S


def track_video(path, out_csv, cfg):
    info = video_info(path)
    scale = min(1.0, cfg.proc_width / info["width"])
    segments = segment_video(path, cfg)
    print(f"{segments[-1].end} frames, {sum(s.shot_id >= 0 for s in segments)} shots")
    frames = read_frames(path, scale)
    rows = []
    workers = cfg.workers or os.cpu_count() or 1
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    for seg in segments:
        shot_frames = list(islice(frames, seg.end - seg.start))
        if seg.shot_id >= 0:
            results, S = track_shot(shot_frames, cfg, pool)
            ball_d, green = 2 * S.radius / scale, S.green_frac
            seen = np.mean([r["visible"] for r in results])
            print(f"  shot {seg.shot_id:3d}  frames {seg.start:5d}-{seg.end - 1:<5d}  pitch {green:4.0%}  "
                  f"ball ~{ball_d:4.1f}px ({'players' if S.players else 'fallback'})  visible {seen:4.0%}")
        else:
            results = [dict(visible=0, x=np.nan, y=np.nan, radius=np.nan, confidence=0.0)] * len(shot_frames)
            ball_d, green = np.nan, np.nan
            print(f"  transition  frames {seg.start:5d}-{seg.end - 1}")
        for i, r in enumerate(results):
            frame = seg.start + i
            rows.append(dict(frame=frame, time=frame / info["fps"], shot_id=seg.shot_id,
                             visible=r["visible"], x=r["x"] / scale, y=r["y"] / scale,
                             radius=r["radius"] / scale, confidence=float(r["confidence"]),
                             ball_diameter=ball_d, green_frac=green))
    if pool is not None:
        pool.shutdown()
    write_csv(out_csv, rows, PRED_COLUMNS)
    print(f"wrote {out_csv}")
