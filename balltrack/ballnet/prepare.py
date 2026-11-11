"""Download SoccerNet-Tracking and convert it to training arrays.

    python -m ballnet.prepare E:\\soccernet                     # train split, 1280x720
    python -m ballnet.prepare E:\\soccernet --size 640x360      # smaller frames
    python -m ballnet.prepare E:\\soccernet --split test        # also the test split

1. Downloads <split>.zip from the public Hugging Face repo SoccerNet/SN-Tracking-2023
   into the folder you give (one large file, resumable; skipped if it is already there).
2. Reads every SNMOT-xxx sequence straight from the zip (no extraction) and writes
       prepared_<W>x<H>/SNMOT-xxx/frames.npy   uint8 (T, H, W, 3), BGR, frames scaled to WxH
       prepared_<W>x<H>/SNMOT-xxx/ball.npy     float32 (T, 4): x, y, diameter, visible (WxH pixels)
   frames.npy is ~2 GB per sequence at 1280x720 (~0.5 GB at 640x360); training reads
   it as a memory map, so it never has to fit in RAM. Sequences already prepared
   are skipped, so the command can be re-run after an interruption.

Pick the size of the clip you will test on, so the ball has about the same size in
training and testing. Width and height must be multiples of 8.
"""
import argparse
import configparser
import os
import zipfile
from collections import defaultdict

import cv2
import numpy as np

REPO = "SoccerNet/SN-Tracking-2023"


def download(folder, split):
    path = os.path.join(folder, f"{split}.zip")
    if os.path.exists(path):
        print(f"using {path}")
        return path
    from huggingface_hub import hf_hub_download
    print(f"downloading {split}.zip from {REPO} to {folder} (several GB, resumable) ...")
    return hf_hub_download(REPO, f"{split}.zip", repo_type="dataset", local_dir=folder)


def sequences(z):
    """{name: {"images": [...], "gt": member, "info": member}} for every SNMOT-* sequence in the zip."""
    seqs = defaultdict(lambda: {"images": []})
    for n in z.namelist():
        parts = n.replace("\\", "/").split("/")
        name = next((p for p in parts if p.startswith("SNMOT")), None)
        if name is None or n.endswith("/"):
            continue
        if "img1" in parts and n.lower().endswith((".jpg", ".png")):
            seqs[name]["images"].append(n)
        elif n.endswith("gt/gt.txt"):
            seqs[name]["gt"] = n
        elif n.endswith("gameinfo.ini"):
            seqs[name]["info"] = n
    return seqs


def ball_ids(z, info_member):
    cp = configparser.ConfigParser()
    cp.read_string(z.read(info_member).decode("utf-8", "replace"))
    sec = cp["Sequence"] if "Sequence" in cp else {}
    return {int(k.split("_")[-1]) for k, v in sec.items()
            if k.lower().startswith("trackletid_") and v.strip().lower().startswith("ball")}


def prepare_sequence(z, name, seq, out_root, W, H):
    out = os.path.join(out_root, name)
    if os.path.exists(os.path.join(out, "ball.npy")):
        return "already prepared"
    if "gt" not in seq or "info" not in seq:
        return "no ground truth, skipped"
    ids = ball_ids(z, seq["info"])
    if not ids:
        return "no ball track, skipped"
    images = sorted(seq["images"])
    first = cv2.imdecode(np.frombuffer(z.read(images[0]), np.uint8), cv2.IMREAD_COLOR)
    sx, sy = W / first.shape[1], H / first.shape[0]

    ball = np.zeros((len(images), 4), np.float32)          # x, y, diameter, visible
    for line in z.read(seq["gt"]).decode().splitlines():
        row = line.split(",")
        if len(row) < 6:
            continue
        frame, track, x, y, w, h = int(row[0]), int(float(row[1])), *map(float, row[2:6])
        if track in ids and 1 <= frame <= len(images):
            ball[frame - 1] = ((x + w / 2) * sx, (y + h / 2) * sy, (w * sx + h * sy) / 2, 1.0)

    os.makedirs(out, exist_ok=True)
    frames = np.lib.format.open_memmap(os.path.join(out, "frames.npy"), mode="w+", dtype=np.uint8,
                                       shape=(len(images), H, W, 3))
    for i, member in enumerate(images):
        img = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
        frames[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    frames.flush()
    del frames
    np.save(os.path.join(out, "ball.npy"), ball)           # written last: marks the sequence as done
    return f"{len(images)} frames, ball in {int(ball[:, 3].sum())}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="where the zip goes; output in <folder>/prepared_<W>x<H>")
    ap.add_argument("--size", default="1280x720", help="frame size WxH for training (default 1280x720)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--zip", help="use this already-downloaded zip instead of downloading")
    a = ap.parse_args()
    os.makedirs(a.folder, exist_ok=True)
    zpath = a.zip or download(a.folder, a.split)
    W, H = (int(v) for v in a.size.lower().split("x"))
    if W % 8 or H % 8:
        raise SystemExit("--size: width and height must be multiples of 8")
    out_root = os.path.join(a.folder, f"prepared_{W}x{H}")
    with zipfile.ZipFile(zpath) as z:
        seqs = sequences(z)
        print(f"{len(seqs)} sequences in {os.path.basename(zpath)}")
        for i, name in enumerate(sorted(seqs)):
            print(f"[{i + 1}/{len(seqs)}] {name}: {prepare_sequence(z, name, seqs[name], out_root, W, H)}", flush=True)
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
