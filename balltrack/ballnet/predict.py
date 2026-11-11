"""Track the ball in a video with a trained BallNet.

    python -m ballnet.predict videos\\clip.mp4 E:\\soccernet\\ballnet.pt --out-dir out_nn

Writes out_nn/<clip>_ball.csv (same columns as balltrack) and
out_nn/<clip>_annotated.mp4, so you can compare with
    python -m balltrack evaluate out_nn\\<clip>_ball.csv labels\\<clip>_gt.csv

Frames are scaled to the size the model was trained at (stored in the checkpoint). Shots are found with
balltrack's histogram analysis, and the LSTM memory never crosses a cut. Within
a shot the video is processed in windows as long as the training sequences,
each started half a window early, so every frame is predicted with the same
amount of history the network saw during training.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from balltrack.config import load_config
from balltrack.shots import segment_video
from balltrack.tracker import calibrate
from balltrack.video import PRED_COLUMNS, read_frames, render, video_info, write_csv

from .model import BallNet, decode, to_input

def predict_shot(model, small, seq_len, device):
    """small: list of resized frames of one shot -> [(confidence, x, y)] in resized pixels."""
    n, half = len(small), max(1, seq_len // 2)
    out = [None] * n
    for t0 in range(0, n, half):
        a, b = max(0, t0 - half), min(n, t0 + half)
        x = to_input(np.stack(small[a:b]))[None].to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(x)[0].float())[0, :, 0].cpu().numpy()
        for t in range(t0, b):
            out[t] = decode(prob[t - a])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", default="out_nn")
    ap.add_argument("--threshold", type=float, help="ball visible if confidence >= this (default: from training)")
    ap.add_argument("--no-video", action="store_true", help="only write the CSV")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = BallNet(pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    threshold = a.threshold if a.threshold is not None else ckpt["threshold"]
    seq_len = ckpt.get("seq_len", 8)
    W, H = ckpt.get("size", (640, 360))
    print(f"model from epoch {ckpt['epoch']} (validation F1 {ckpt['val_f1']:.3f}), {W}x{H}, threshold {threshold}, {device}")

    cfg = load_config()
    info = video_info(a.video)
    sx, sy = info["width"] / W, info["height"] / H
    segments = segment_video(a.video, cfg)
    print(f"{segments[-1].end} frames, {sum(s.shot_id >= 0 for s in segments)} shots")

    frames, rows = read_frames(a.video), []
    for seg in segments:
        small, first = [], []                  # resized copies; full size only for calibration
        for i in range(seg.end - seg.start):
            f = next(frames)
            small.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
            if i < 50:
                first.append(f)
        results = [(0.0, np.nan, np.nan)] * len(small)
        ball_d, green = np.nan, np.nan
        if seg.shot_id >= 0:
            results = predict_shot(model, small, seq_len, device)
            # pitch share and expected ball size, only for the CSV (shot-type table, box size)
            S = calibrate(first, cfg)
            ball_d, green = 2 * S.radius, S.green_frac
            seen = np.mean([c >= threshold for c, _, _ in results])
            print(f"  shot {seg.shot_id:3d}  frames {seg.start:5d}-{seg.end - 1:<5d}  visible {seen:4.0%}")
        for i, (c, x, y) in enumerate(results):
            f = seg.start + i
            vis = seg.shot_id >= 0 and c >= threshold
            rows.append(dict(frame=f, time=f / info["fps"], shot_id=seg.shot_id, visible=int(vis),
                             x=x * sx if vis else np.nan, y=y * sy if vis else np.nan,
                             radius=ball_d / 2 if vis else np.nan, confidence=float(c),
                             ball_diameter=ball_d, green_frac=green))

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, os.path.splitext(os.path.basename(a.video))[0])
    write_csv(stem + "_ball.csv", rows, PRED_COLUMNS)
    print(f"wrote {stem}_ball.csv")
    if not a.no_video:
        render(a.video, stem + "_ball.csv", stem + "_annotated.mp4")


if __name__ == "__main__":
    main()
