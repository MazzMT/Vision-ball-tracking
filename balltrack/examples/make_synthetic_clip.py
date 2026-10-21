"""Make a synthetic test clip with exact ground truth (a smoke test, not a benchmark).

  shot 0  wide shot, panning camera, scoreboard + round logo, ball passed along the
          ground, hidden behind a player for 10 frames, then lofted over the pitch
  shot 1  hard cut to a zoomed-in camera
  wipe    logo wipe (transition)
  shot 2  slow-motion replay (each frame shown twice) of part of shot 0

White socks, white crowd dots and pitch lines act as look-alikes.

    python examples/make_synthetic_clip.py synth/
writes synth/synth.mp4 and synth/synth_gt.csv (same format as the labeler).
"""
import csv
import os
import sys

import cv2
import numpy as np

VW, VH, FPS = 960, 540, 25
WW, WH = 3000, 1300
BALL_R = 3.0
KEYS = [(0, 900, 620), (50, 1000, 640), (70, 1000, 640), (95, 1550, 560), (130, 1750, 700),
        (150, 1950, 520), (175, 2200, 420), (220, 2350, 600), (260, 2500, 700), (300, 2400, 820)]
HIDDEN = range(108, 118)
rng = np.random.default_rng(0)


def make_world():
    w = np.zeros((WH, WW, 3), np.uint8)
    w[:] = (52, 140, 62)
    for x in range(0, WW, 300):
        w[:, x + 150:x + 300] = (46, 128, 56)                       # mowing stripes
    tex = cv2.resize(rng.normal(0, 5, (WH // 4, WW // 4)).astype(np.float32), (WW, WH))
    w = np.clip(w + tex[..., None], 0, 255).astype(np.uint8)
    w[:260] = cv2.resize(rng.integers(20, 110, (52, WW // 5, 3)).astype(np.uint8), (WW, 260),
                         interpolation=cv2.INTER_NEAREST)           # crowd
    for _ in range(900):
        cv2.circle(w, (int(rng.integers(0, WW)), int(rng.integers(10, 250))), 3, (235, 235, 235), -1)
    for x in range(0, WW, 220):                                     # advertising boards
        cv2.rectangle(w, (x, 262), (x + 210, 300), (40 + x % 200, 60, 200), -1)
        cv2.putText(w, "ADS", (x + 20, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    white = (235, 235, 235)
    cv2.rectangle(w, (200, 380), (2800, 1220), white, 5)
    cv2.line(w, (1500, 380), (1500, 1220), white, 5)
    cv2.circle(w, (1500, 800), 150, white, 5)
    cv2.rectangle(w, (200, 600), (560, 1000), white, 5)
    cv2.rectangle(w, (2440, 600), (2800, 1000), white, 5)
    return w


PLAYERS = [dict(x=rng.uniform(350, 2650), y=rng.uniform(450, 1150), a=rng.uniform(20, 70),
                s=rng.uniform(0.03, 0.07), ph=rng.uniform(0, 6.28), team=i % 2) for i in range(18)]


def ball_at(t):
    ts, xs, ys = zip(*KEYS)
    x, y = np.interp(t, ts, xs), np.interp(t, ts, ys)
    if 150 <= t <= 175:                                             # lofted pass
        u = (t - 150) / 25
        y -= 220 * 4 * u * (1 - u)
    return x, y


def draw_world(world, t):
    img = world.copy()
    for p in PLAYERS:
        x = int(p["x"] + p["a"] * np.sin(p["s"] * t + p["ph"]))
        y = int(p["y"] + p["a"] * np.cos(p["s"] * t + p["ph"]))
        cv2.ellipse(img, (x, y - 18), (7, 13), 0, 0, 360, (30, 30, 200) if p["team"] else (200, 90, 30), -1)
        cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 3), (20, 20, 20), -1)
        cv2.line(img, (x - 3, y + 3), (x - 4, y + 16), (120, 160, 210), 3)
        cv2.line(img, (x + 3, y + 3), (x + 4, y + 16), (120, 160, 210), 3)
        cv2.rectangle(img, (x - 6, y + 10), (x - 2, y + 17), (240, 240, 240), -1)   # white socks
        cv2.rectangle(img, (x + 2, y + 10), (x + 6, y + 17), (240, 240, 240), -1)
        cv2.circle(img, (x, y - 35), 4, (120, 160, 210), -1)
    bx, by = ball_at(t)
    visible = int(t) not in HIDDEN
    if visible:
        cv2.circle(img, (int(round(bx)), int(round(by))), int(BALL_R), (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(bx)), int(round(by))), 1, (150, 150, 150), -1)
    return img, (bx, by), visible


class Camera:
    def __init__(self, zoom, follow_y):
        self.zoom, self.follow_y, self.c = zoom, follow_y, None

    def matrix(self, ball):
        target = np.array([ball[0], 650 + (ball[1] - 650) * self.follow_y])
        self.c = target if self.c is None else 0.88 * self.c + 0.12 * target
        z = self.zoom
        cx = np.clip(self.c[0], VW / (2 * z), WW - VW / (2 * z))
        cy = np.clip(self.c[1], VH / (2 * z), WH - VH / (2 * z))
        return np.array([[z, 0, -z * (cx - VW / (2 * z))], [0, z, -z * (cy - VH / (2 * z))]], np.float32)


def finish(img):
    cv2.rectangle(img, (10, 10), (210, 44), (240, 240, 240), -1)       # scoreboard
    cv2.putText(img, "HOM 1-0 AWY", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    cv2.circle(img, (VW - 40, 36), 12, (245, 245, 245), -1)             # round channel logo
    cv2.circle(img, (VW - 40, 36), 5, (40, 40, 200), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.6).astype(np.float32) + rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    world = make_world()
    frames, gt = [], []

    def shoot(t, cam, label=True):
        img, ball, vis = draw_world(world, t)
        M = cam.matrix(ball)
        x, y = M @ np.array([ball[0], ball[1], 1.0])
        inside = vis and 0 <= x < VW and 0 <= y < VH
        row = dict(status="visible" if inside else "not_visible", x=f"{x:.2f}" if inside else "",
                   y=f"{y:.2f}" if inside else "", diameter=f"{2 * BALL_R * cam.zoom:.2f}" if inside else "")
        return finish(cv2.warpAffine(img, M, (VW, VH))), row if label else dict(status="unknown")

    cam = Camera(1.0, 0.3)
    for t in range(220):
        f, r = shoot(t, cam); frames.append(f); gt.append(r)
    cam = Camera(2.2, 0.6)
    for t in range(220, 290):
        f, r = shoot(t, cam); frames.append(f); gt.append(r)
    last, cam = frames[-1], Camera(1.5, 0.5)
    for i in range(100):
        f, r = shoot(95 + i // 2, cam, label=i >= 12)
        if i < 12:                                                    # logo wipe
            x1 = int(-336 + (i + 1) / 13 * (VW + 672))
            wiped = last.copy()
            wiped[:, :max(0, x1)] = f[:, :max(0, x1)]
            cv2.rectangle(wiped, (x1, 0), (x1 + 336, VH), (180, 60, 20), -1)
            cv2.rectangle(wiped, (x1 + 158, 0), (x1 + 178, VH), (240, 240, 240), -1)
            f = wiped
        frames.append(f); gt.append(r)

    writer = cv2.VideoWriter(os.path.join(out_dir, "synth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW, VH))
    for f in frames:
        writer.write(f)
    writer.release()
    with open(os.path.join(out_dir, "synth_gt.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "status", "x", "y", "diameter"])
        w.writeheader()
        for i, r in enumerate(gt):
            w.writerow(dict(frame=i, **r))
    print(f"wrote {out_dir}/synth.mp4 ({len(frames)} frames) and synth_gt.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth")
