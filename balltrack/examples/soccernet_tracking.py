"""Get test clips with ball ground truth from SoccerNet-Tracking.

SoccerNet-Tracking has 30-second 1080p clips (25 fps) in which every object,
including the ball, has a box in every frame. This script

  1. downloads a few sequences from the public train or test archive on Hugging
     Face, reading only the files it needs (no need to download the whole zip),
  2. converts each one to  <name>.mp4  and  <name>_gt.csv  in balltrack's format.

Frames without a ball box are labelled "not_visible".

    pip install huggingface_hub
    python examples/soccernet_tracking.py soccernet --count 3
    python -m balltrack track soccernet/SNMOT-060.mp4
    python -m balltrack evaluate out/SNMOT-060_ball.csv soccernet/SNMOT-060_gt.csv

Already have the sequences unzipped?  Convert them without downloading:
    python examples/soccernet_tracking.py soccernet --from-folder path/to/train
"""
import argparse
import configparser
import csv
import os
import zipfile

import cv2


def convert(seq_dir, out_dir):
    """SNMOT-xxx folder (img1/, gt/gt.txt, gameinfo.ini) -> mp4 + ground-truth CSV."""
    name = os.path.basename(os.path.normpath(seq_dir))
    info = configparser.ConfigParser()
    info.read(os.path.join(seq_dir, "gameinfo.ini"))
    tracks = info["Sequence"] if "Sequence" in info else {}
    ball_ids = {int(k.split("_")[-1]) for k, v in tracks.items()
                if k.lower().startswith("trackletid_") and v.strip().lower().startswith("ball")}
    if not ball_ids:
        print(f"{name}: no ball track, skipped")
        return

    ball = {}                       # frame (1-based) -> (x, y, diameter)
    for row in csv.reader(open(os.path.join(seq_dir, "gt", "gt.txt"))):
        frame, track, x, y, w, h = int(row[0]), int(row[1]), *map(float, row[2:6])
        if track in ball_ids:
            ball[frame] = (x + w / 2, y + h / 2, (w + h) / 2)

    images = sorted(os.listdir(os.path.join(seq_dir, "img1")))
    seq = configparser.ConfigParser()
    seq.read(os.path.join(seq_dir, "seqinfo.ini"))
    fps = float(seq["Sequence"].get("frameRate", 25)) if "Sequence" in seq else 25.0
    first = cv2.imread(os.path.join(seq_dir, "img1", images[0]))
    writer = cv2.VideoWriter(os.path.join(out_dir, name + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (first.shape[1], first.shape[0]))
    for img in images:
        writer.write(cv2.imread(os.path.join(seq_dir, "img1", img)))
    writer.release()

    with open(os.path.join(out_dir, name + "_gt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "status", "x", "y", "diameter"])
        for i in range(len(images)):
            if i + 1 in ball:
                x, y, d = ball[i + 1]
                w.writerow([i, "visible", f"{x:.1f}", f"{y:.1f}", f"{d:.1f}"])
            else:
                w.writerow([i, "not_visible", "", "", ""])
    print(f"{name}: {len(images)} frames, ball in {len(ball)} -> {out_dir}")


def download(out_dir, count, split):
    """Extract `count` sequences from SoccerNet's public tracking archive."""
    from huggingface_hub import HfFileSystem
    path = f"datasets/SoccerNet/SN-Tracking-2023/{split}.zip"
    with HfFileSystem().open(path, "rb", block_size=2 ** 22) as remote, zipfile.ZipFile(remote) as z:
        names = z.namelist()
        seqs = sorted({n.split("/")[-3] for n in names if n.endswith("gt/gt.txt")})[:count]
        for seq in seqs:
            files = [n for n in names if f"/{seq}/" in n and not n.endswith("/")]
            print(f"downloading {seq} ({len(files)} files) ...")
            for n in files:
                target = os.path.join(out_dir, "raw", seq, n.split(f"/{seq}/", 1)[1])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(z.read(n))
    return [os.path.join(out_dir, "raw", s) for s in seqs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--count", type=int, default=3, help="number of sequences to download")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--from-folder", help="convert already-unzipped SNMOT-* folders instead")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.from_folder:
        seqs = sorted(os.path.join(a.from_folder, d) for d in os.listdir(a.from_folder)
                      if os.path.isdir(os.path.join(a.from_folder, d, "img1")))
    else:
        seqs = download(a.out_dir, a.count, a.split)
    for s in seqs:
        convert(s, a.out_dir)


if __name__ == "__main__":
    main()
