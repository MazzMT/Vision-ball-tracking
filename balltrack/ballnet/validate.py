"""Run a trained BallNet (or the classical tracker) on the validation sequences and score it.

    python -m ballnet.validate E:\\soccernet\\ballnet.pt --out-dir out\\nn
    python -m ballnet.validate E:\\soccernet\\ballnet.pt --method classical --out-dir out\\classical

With --method classical the same sequences are run through balltrack's classical
tracker instead (the checkpoint is then only used to find the sequences), so both
methods are scored on identical clips with identical ground truth. Each sequence
is one camera shot, so it is tracked as a single shot.

The validation sequences are the ones held out during training (stored in the
checkpoint); they are read from the prepared frames (frames.npy, at the
training size, e.g. 1280x720), not from a video. For every sequence it writes
    <out-dir>/<SNMOT-xxx>_ball.csv        the network's predictions (balltrack format)
    <out-dir>/<SNMOT-xxx>_gt.csv          SoccerNet's ground truth (labeler format)
    <out-dir>/<SNMOT-xxx>_annotated.mp4   green box = prediction, yellow cross = true ball
and all_ball.csv / all_gt.csv with every sequence after each other (each
sequence is one "shot" there), then prints balltrack's evaluation table for all
of them together. Positions are in the prepared frames' pixels.
Other sequences:
    --from-zip E:\\soccernet\\test.zip   every prepared sequence from that zip, e.g. the
                                       test split: 3 matches the model never trained on
    --clips <folder> <folder> ...      any prepared sequence folders
"""
import argparse
import os
import zipfile

import cv2
import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor

from balltrack.config import load_config
from balltrack.evaluate import evaluate
from balltrack.tracker import track_shot
from balltrack.video import GT_COLUMNS, PRED_COLUMNS, write_csv

from .model import BallNet
from .predict import predict_shot
from .prepare import sequences

FPS = 25.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--method", default="nn", choices=["nn", "classical"])
    ap.add_argument("--out-dir", default=os.path.join("out", "nn"))
    ap.add_argument("--clips", nargs="+", help="prepared sequence folders (default: validation clips from training)")
    ap.add_argument("--from-zip", help="score the prepared sequences that come from this SoccerNet zip (e.g. test.zip)")
    ap.add_argument("--threshold", type=float, help="ball visible if confidence >= this (default: from training)")
    ap.add_argument("--no-video", action="store_true", help="only write the CSV files")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    if a.method == "nn":
        model = BallNet(pretrained=False).to(device).eval()
        model.load_state_dict(ckpt["model"])
    else:
        cfg = load_config()
        workers = cfg.workers or os.cpu_count() or 1
        pool = ProcessPoolExecutor(workers) if workers > 1 else None
    threshold = a.threshold if a.threshold is not None else ckpt["threshold"]
    clips = a.clips or ckpt["val_clips"]
    if a.from_zip:
        root = os.path.dirname(os.path.normpath(ckpt["val_clips"][0]))     # the prepared_<W>x<H> folder
        with zipfile.ZipFile(a.from_zip) as z:
            names = sorted(sequences(z))
        clips = [os.path.join(root, n) for n in names if os.path.exists(os.path.join(root, n, "ball.npy"))]
        print(f"{len(clips)} of the {len(names)} sequences in {os.path.basename(a.from_zip)} are prepared in {root}")
        if not clips:
            raise SystemExit("nothing to score")
    if a.method == "nn":
        print(f"model from epoch {ckpt['epoch']}, threshold {threshold}, {device}, {len(clips)} sequences")
    else:
        print(f"classical tracker (balltrack), {len(clips)} sequences")
    os.makedirs(a.out_dir, exist_ok=True)

    all_pred, all_gt = [], []
    for shot, clip in enumerate(clips):
        name = os.path.basename(os.path.normpath(clip))
        frames = np.load(os.path.join(clip, "frames.npy"), mmap_mode="r")
        ball = np.load(os.path.join(clip, "ball.npy"))
        ball_d = float(np.median(ball[ball[:, 3] > 0, 2])) if ball[:, 3].any() else np.nan
        if a.method == "nn":
            out = predict_shot(model, [np.array(f) for f in frames], ckpt.get("seq_len", 8), device)
            results = [dict(visible=int(c >= threshold), x=x, y=y, radius=ball_d / 2, confidence=c) for c, x, y in out]
            green = np.nan
        else:
            results, S = track_shot([np.array(f) for f in frames], cfg, pool)
            green = S.green_frac

        pred, gt = [], []
        for i, (r, (bx, by, bd, bv)) in enumerate(zip(results, ball)):
            vis = bool(r["visible"])
            pred.append(dict(frame=i, time=i / FPS, shot_id=shot, visible=int(vis),
                             x=float(r["x"]) if vis else np.nan, y=float(r["y"]) if vis else np.nan,
                             radius=float(r["radius"]) if vis else np.nan,
                             confidence=float(r["confidence"]), ball_diameter=ball_d, green_frac=green))
            gt.append(dict(frame=i, status="visible", x=float(bx), y=float(by), diameter=float(bd)) if bv > 0
                      else dict(frame=i, status="not_visible", x=np.nan, y=np.nan, diameter=np.nan))
        stem = os.path.join(a.out_dir, name)
        write_csv(stem + "_ball.csv", pred, PRED_COLUMNS)
        write_csv(stem + "_gt.csv", gt, GT_COLUMNS)

        if not a.no_video:
            h, w = frames.shape[1:3]
            writer = cv2.VideoWriter(stem + "_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            for f, p, g in zip(frames, pred, gt):
                img = np.array(f)
                if g["status"] == "visible":
                    cv2.drawMarker(img, (int(g["x"]), int(g["y"])), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
                if p["visible"]:
                    half = max(20, 2.6 * p["radius"]) / 2
                    cv2.rectangle(img, (int(p["x"] - half), int(p["y"] - half)),
                                  (int(p["x"] + half), int(p["y"] + half)), (0, 255, 0), 2)
                writer.write(img)
            writer.release()

        found = sum(1 for p, g in zip(pred, gt) if p["visible"] and g["status"] == "visible"
                    and np.hypot(p["x"] - g["x"], p["y"] - g["y"]) <= 2 * g["diameter"])
        print(f"  {name}: ball visible in {sum(g['status'] == 'visible' for g in gt)} frames, "
              f"predicted in {sum(p['visible'] for p in pred)}, within 2 diameters in {found}  -> {stem}_*")

        off = len(all_gt)                              # one long sequence: renumber the frames
        all_pred += [dict(r, frame=r["frame"] + off, time=(r["frame"] + off) / FPS) for r in pred]
        all_gt += [dict(r, frame=r["frame"] + off) for r in gt]

    if a.method == "classical" and pool is not None:
        pool.shutdown()
    write_csv(os.path.join(a.out_dir, "all_ball.csv"), all_pred, PRED_COLUMNS)
    write_csv(os.path.join(a.out_dir, "all_gt.csv"), all_gt, GT_COLUMNS)
    print(f"\nAll {len(clips)} sequences together (each sequence is one shot):")
    evaluate(os.path.join(a.out_dir, "all_ball.csv"), os.path.join(a.out_dir, "all_gt.csv"))


if __name__ == "__main__":
    main()
