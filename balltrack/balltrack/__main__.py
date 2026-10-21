"""python -m balltrack {track,label,evaluate,render,shots} ..."""
import argparse
import os

from .config import load_config


def main():
    ap = argparse.ArgumentParser(prog="balltrack", description="Classical soccer-ball tracker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text in [("track", "track the ball: CSV + annotated video"),
                            ("label", "click-to-label ground truth"),
                            ("shots", "list detected shots")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("video")
        p.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a config value")
        p.add_argument("--ball-color", metavar="R,G,B", help="ball colour if not white")
    sub.choices["track"].add_argument("--out-dir", default="out")
    sub.choices["label"].add_argument("--out", required=True, help="ground-truth CSV to write")
    sub.choices["label"].add_argument("--step", type=int, default=5)
    p = sub.add_parser("evaluate", help="score predictions against ground truth")
    p.add_argument("pred")
    p.add_argument("gt")
    p = sub.add_parser("render", help="draw predictions (and optional ground truth) on the video")
    p.add_argument("video")
    p.add_argument("pred")
    p.add_argument("--out", required=True)
    p.add_argument("--gt")
    a = ap.parse_args()

    if a.cmd == "track":
        from .tracker import track_video
        from .video import render
        os.makedirs(a.out_dir, exist_ok=True)
        stem = os.path.join(a.out_dir, os.path.splitext(os.path.basename(a.video))[0])
        track_video(a.video, stem + "_ball.csv", load_config(a.set, a.ball_color))
        render(a.video, stem + "_ball.csv", stem + "_annotated.mp4")
    elif a.cmd == "label":
        from .label import label
        label(a.video, a.out, load_config(a.set, a.ball_color), a.step)
    elif a.cmd == "shots":
        from .shots import segment_video
        for s in segment_video(a.video, load_config(a.set, a.ball_color)):
            print(f"{'shot ' + str(s.shot_id) if s.shot_id >= 0 else 'transition':12s} frames {s.start}-{s.end - 1}")
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(a.pred, a.gt)
    elif a.cmd == "render":
        from .video import render
        render(a.video, a.pred, a.out, a.gt)


if __name__ == "__main__":
    main()
