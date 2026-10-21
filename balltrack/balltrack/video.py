"""Video reading/writing, CSV files and the annotated output video."""
import csv
import math

import cv2

PRED_COLUMNS = ["frame", "time", "shot_id", "visible", "x", "y", "radius",
                "confidence", "ball_diameter", "green_frac"]
GT_COLUMNS = ["frame", "status", "x", "y", "diameter"]


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frames(path, scale=1.0):
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield frame
    cap.release()


def write_csv(path, rows, columns):
    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.2f}"
        return v
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: fmt(r.get(k, "")) for k in columns} for r in rows)


def read_csv(path):
    """frame -> row dict; numeric fields become floats (empty -> nan)."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if v != "" else math.nan
                except ValueError:
                    row[k] = v
            out[int(row["frame"])] = row
    return out


def render(video_path, pred_csv, out_path, gt_csv=None, min_box=20):
    """Green box around the ball on every frame where it is visible.
    With gt_csv, ground truth is drawn as a small yellow cross."""
    info = video_info(video_path)
    pred = read_csv(pred_csv)
    gt = read_csv(gt_csv) if gt_csv else {}
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                             (info["width"], info["height"]))
    for i, frame in enumerate(read_frames(video_path)):
        g = gt.get(i)
        if g and g["status"] == "visible":
            cv2.drawMarker(frame, (int(g["x"]), int(g["y"])), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        p = pred.get(i)
        if p and p["visible"] == 1:
            half = max(min_box, 2.6 * p["radius"]) / 2
            cv2.rectangle(frame, (int(p["x"] - half), int(p["y"] - half)),
                          (int(p["x"] + half), int(p["y"] + half)), (0, 255, 0), 2)
        writer.write(frame)
    writer.release()
    print(f"wrote {out_path}")
