"""Click-to-label ground truth (OpenCV window).

  left click      ball centre (then jumps ahead `step` frames, never past a cut)
  right click / x ball not visible on this frame
  a / d           previous / next frame       w / s   forward / back `step` frames
  n               next shot                   u       remove the label on this frame
  - / =           smaller / larger ball circle (its diameter is stored with the click)
  q or Esc        save and quit (progress is also saved on every click)

Between two labelled frames of the same shot the ball position is interpolated.
Between a 'visible' and a 'not visible' label the frames are unknown and are
skipped by the evaluator, so mark occlusions with explicit right clicks.
Re-opening the same output file resumes where you left off.
"""
import os

import cv2
import numpy as np

from .shots import segment, signature
from .video import GT_COLUMNS, read_csv, read_frames, write_csv


def ground_truth(clicks, segments, n):
    rows = [dict(frame=i, status="unknown") for i in range(n)]
    for seg in segments:
        frames = sorted(f for f in clicks if seg.start <= f < seg.end)
        for a, b in zip(frames, frames[1:] + [None]):
            rows[a] = dict(frame=a, **clicks[a])
            if b is None or clicks[a]["status"] != clicks[b]["status"]:
                continue
            for f in range(a + 1, b):
                u = (f - a) / (b - a)
                row = {k: clicks[a][k] + u * (clicks[b][k] - clicks[a][k]) for k in ("x", "y", "diameter")} \
                    if clicks[a]["status"] == "visible" else {}
                rows[f] = dict(frame=f, status=clicks[a]["status"], **row)
    return rows


def label(video, out_csv, cfg, step=5, display_width=1280):
    print("reading video ...")
    jpgs, sigs = [], []
    for frame in read_frames(video):
        scale = min(1.0, display_width / frame.shape[1])
        sigs.append(signature(frame))
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        jpgs.append(cv2.imencode(".jpg", small)[1])      # compressed frame cache
    n, segments = len(jpgs), segment(np.array(sigs), cfg)
    seg_of = np.concatenate([[i] * (s.end - s.start) for i, s in enumerate(segments)])

    clicks = {}
    if os.path.exists(out_csv):
        prev = read_csv(out_csv)
        clicks = {f: {k: r[k] for k in ("status", "x", "y", "diameter")} for f, r in prev.items()
                  if r.get("source", "click") == "click" and r["status"] in ("visible", "not_visible")}
        print(f"resumed {len(clicks)} labels")
    state = dict(frame=0, mouse=(0, 0), diameter=12.0)

    def save():
        rows = ground_truth(clicks, segments, n)
        for r in rows:
            r["source"] = "click" if r["frame"] in clicks else ""
        write_csv(out_csv, rows, GT_COLUMNS + ["source"])

    def set_label(status, x=np.nan, y=np.nan):
        f = state["frame"]
        clicks[f] = dict(status=status, x=x / scale, y=y / scale, diameter=state["diameter"]) \
            if status == "visible" else dict(status=status, x=np.nan, y=np.nan, diameter=np.nan)
        save()
        state["frame"] = min(f + step, segments[seg_of[f]].end - 1) if f < segments[seg_of[f]].end - 1 else min(f + 1, n - 1)

    def on_mouse(event, x, y, *_):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            set_label("visible", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            set_label("not_visible")

    cv2.namedWindow("label")
    cv2.setMouseCallback("label", on_mouse)
    while True:
        f = state["frame"]
        img = cv2.imdecode(jpgs[f], cv2.IMREAD_COLOR)
        clean = img.copy()
        rows = ground_truth(clicks, segments, n)
        r = rows[f]
        if r["status"] == "visible":
            cv2.circle(img, (int(r["x"] * scale), int(r["y"] * scale)), int(r["diameter"] * scale / 2) + 1,
                       (0, 255, 0) if f in clicks else (0, 200, 200), 1)
        mx, my = state["mouse"]
        cv2.circle(img, (mx, my), int(state["diameter"] * scale / 2) + 1, (255, 255, 0), 1)
        # 6x magnifier around the cursor, top-right corner
        crop = cv2.copyMakeBorder(clean, 20, 20, 20, 20, cv2.BORDER_CONSTANT)[my:my + 40, mx:mx + 40]
        if crop.shape[:2] == (40, 40):
            mag = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(mag, (120, 120), (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
            img[5:245, img.shape[1] - 245:img.shape[1] - 5] = mag
        seg = segments[seg_of[f]]
        text = (f"frame {f}/{n - 1}  {'shot ' + str(seg.shot_id) if seg.shot_id >= 0 else 'transition'} "
                f"[{seg.start}-{seg.end - 1}]  label: {r['status']}{' (click)' if f in clicks else ''}  labels: {len(clicks)}")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(img, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        # timeline: green = visible, red = not visible, grey = unknown, white ticks = cuts
        bar = np.full((14, n, 3), 80, np.uint8)
        for row in rows:
            bar[:, row["frame"]] = {"visible": (0, 200, 0), "not_visible": (0, 0, 200)}.get(row["status"], (80, 80, 80))
        for s in segments[1:]:
            bar[:, s.start] = 255
        bar = cv2.resize(bar, (img.shape[1], 14), interpolation=cv2.INTER_NEAREST)
        cv2.line(bar, (f * img.shape[1] // n, 0), (f * img.shape[1] // n, 13), (0, 255, 255), 2)
        cv2.imshow("label", np.vstack([img, bar]))

        key = cv2.waitKey(30) & 0xFF
        moves = {ord("d"): 1, ord("a"): -1, ord("w"): step, ord("s"): -step}
        if key in moves:
            state["frame"] = int(np.clip(f + moves[key], 0, n - 1))
        elif key == ord("n"):
            state["frame"] = min(seg.end, n - 1)
        elif key == ord("x"):
            set_label("not_visible")
        elif key == ord("u"):
            clicks.pop(f, None)
            save()
        elif key == ord("-"):
            state["diameter"] = max(2.0, state["diameter"] - 1)
        elif key in (ord("="), ord("+")):
            state["diameter"] += 1
        elif key in (ord("q"), 27):
            save()
            break
    cv2.destroyAllWindows()
    print(f"saved {out_csv}")
