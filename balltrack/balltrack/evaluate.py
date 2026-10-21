"""Score predictions against labelled ground truth.

Frames whose ground truth is 'unknown' are skipped. Distances are measured in
ball diameters (the diameter stored with each click), so the same threshold
means the same thing in a wide shot and in a close-up.

For a threshold t:
  TP = ball visible, predicted within t diameters
  FP = predicted visible, but no ball or more than t diameters away
  FN = ball visible, but not predicted within t diameters
"""
import numpy as np

from .video import read_csv


def metrics(gt_vis, pred_vis, err, thresholds):
    m = dict(frames=len(gt_vis), ball_visible=int(gt_vis.sum()),
             found=np.mean(pred_vis[gt_vis]) if gt_vis.any() else np.nan,
             false_alarm=np.mean(pred_vis[~gt_vis]) if (~gt_vis).any() else np.nan,
             median_err=np.median(err[gt_vis & pred_vis]) if (gt_vis & pred_vis).any() else np.nan)
    for t in thresholds:
        tp = np.sum(err <= t)
        p = tp / pred_vis.sum() if pred_vis.any() else np.nan
        r = tp / gt_vis.sum() if gt_vis.any() else np.nan
        m[f"F1@{t}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
        m[f"P@{t}"], m[f"R@{t}"] = p, r
    return m


def evaluate(pred_csv, gt_csv, thresholds=(0.5, 1, 2, 4)):
    pred, gt = read_csv(pred_csv), read_csv(gt_csv)
    frames = [f for f in sorted(gt) if gt[f]["status"] in ("visible", "not_visible") and f in pred]
    gt_vis = np.array([gt[f]["status"] == "visible" for f in frames])
    pred_vis = np.array([pred[f]["visible"] == 1 for f in frames])
    diam = np.array([gt[f]["diameter"] if gt[f]["diameter"] > 0 else pred[f]["ball_diameter"] for f in frames])
    err = np.hypot([pred[f]["x"] - gt[f]["x"] for f in frames], [pred[f]["y"] - gt[f]["y"] for f in frames]) / diam
    err = np.where(gt_vis & pred_vis, err, np.inf)           # error in ball diameters

    green = np.array([pred[f]["green_frac"] for f in frames])
    shot = np.array([pred[f]["shot_id"] for f in frames])
    groups = {"all": np.ones(len(frames), bool),
              "wide (pitch >= 50%)": green >= 0.5,
              "medium (pitch 20-50%)": (green >= 0.2) & (green < 0.5),
              "close-up (pitch < 20%)": green < 0.2,
              "transitions": shot < 0}
    groups.update({f"shot {int(s)}": shot == s for s in sorted(set(shot)) if s >= 0})

    cols = ["frames", "ball_visible", "found", "false_alarm", "median_err"] + [f"F1@{t}" for t in thresholds]
    print(f"{'':24s}" + "".join(f"{c:>13s}" for c in cols))
    report = {}
    for name, m in groups.items():
        if m.any():
            report[name] = metrics(gt_vis[m], pred_vis[m], err[m], thresholds)
            print(f"{name:24s}" + "".join(
                f"{v:13d}" if isinstance(v, int) else f"{v:13.3f}" for v in (report[name][c] for c in cols)))
    print("\nfound = share of ball-visible frames where a ball was reported; "
          "false_alarm = share of no-ball frames where a ball was reported;\n"
          "median_err and F1@t are in ball diameters.")
    return report
