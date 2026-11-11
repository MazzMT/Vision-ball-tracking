"""Train BallNet on prepared SoccerNet-Tracking sequences.

    python -m ballnet.train E:\\soccernet\\prepared_1280x720 --out E:\\soccernet\\ballnet.pt

The last `--val-clips` sequences (by name) are held out for validation and never
trained on. After every epoch the model is scored on them: a frame counts as
correct if the predicted ball is within 2 ball diameters of the true ball. The
confidence threshold that gives the best F1 is stored in the checkpoint together
with the weights, and the best epoch is kept.

Needs an NVIDIA GPU with CUDA (it runs on the CPU too, but far too slowly).
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Clips, clip_dirs
from .model import BallNet, decode, focal_loss

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def validate(model, loader, device):
    """Per-frame confidence, error (in ball diameters) and visibility over the validation set."""
    model.eval()
    conf, err, vis = [], [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for x, _, ball in loader:
            prob = torch.sigmoid(model(x.to(device, non_blocking=True))[0].float()).cpu().numpy()
            for p, b in zip(prob.reshape(-1, *prob.shape[-2:]), ball.reshape(-1, 4).numpy()):
                c, px, py = decode(p)
                conf.append(c)
                vis.append(b[3] > 0)
                err.append(np.hypot(px - b[0], py - b[1]) / max(b[2], 1.0) if b[3] > 0 else np.inf)
    conf, err, vis = np.array(conf), np.array(err), np.array(vis)
    best = (0.0, 0.5, 0.0, 0.0)
    for t in THRESHOLDS:
        pred = conf >= t
        tp = np.sum(pred & (err <= 2))
        p = tp / max(pred.sum(), 1)
        r = tp / max(vis.sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best[0]:
            best = (f1, float(t), p, r)
    found = vis & (err <= 2)
    median = float(np.median(err[found])) if found.any() else float("nan")
    return best, median


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="folder with prepared sequences (from ballnet.prepare)")
    ap.add_argument("--out", default="ballnet.pt", help="checkpoint to write (best epoch)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2, help="sequences per batch (2 fits 720p in 12 GB; lower it if out of memory)")
    ap.add_argument("--seq-len", type=int, default=8, help="frames per training sequence")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-clips", type=int, default=5, help="sequences held out for validation")
    ap.add_argument("--workers", type=int, default=4, help="data-loading processes")
    ap.add_argument("--no-pretrained", action="store_true", help="do not start from ImageNet weights")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found, training on the CPU will be very slow.\n"
              "Check with:  python -c \"import torch; print(torch.cuda.is_available())\"")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dirs = clip_dirs(a.data)
    if len(dirs) <= a.val_clips:
        raise SystemExit(f"found {len(dirs)} prepared sequences in {a.data}; need more than --val-clips={a.val_clips}")
    train_dirs, val_dirs = dirs[:-a.val_clips], dirs[-a.val_clips:]
    size = np.load(os.path.join(dirs[0], "frames.npy"), mmap_mode="r").shape[2:0:-1]     # (W, H)
    print(f"frame size {size[0]}x{size[1]}")
    print(f"train: {len(train_dirs)} sequences, validation: {', '.join(os.path.basename(d) for d in val_dirs)}")

    loader_args = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=device.type == "cuda",
                       persistent_workers=a.workers > 0)
    train = DataLoader(Clips(train_dirs, a.seq_len, step=4, train=True), shuffle=True, drop_last=True, **loader_args)
    val = DataLoader(Clips(val_dirs, a.seq_len, step=a.seq_len, train=False), shuffle=False, **loader_args)
    print(f"{len(train)} training batches per epoch, {len(val)} validation batches")

    model = BallNet(pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0

    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for i, (x, target, _) in enumerate(train, 1):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = model(x)
            loss = focal_loss(logits, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if i % 50 == 0 or i == len(train):
                rate = i / (time.time() - t0)
                print(f"\repoch {epoch}/{a.epochs}  batch {i}/{len(train)}  loss {total / i:.3f}  "
                      f"{rate:.1f} batches/s  ETA {(len(train) - i) / rate / 60:.0f} min", end="", flush=True)
        (f1, thr, p, r), median = validate(model, val, device)
        print(f"\nepoch {epoch}: train loss {total / len(train):.3f} | validation F1 {f1:.3f} "
              f"(precision {p:.3f}, recall {r:.3f}) at threshold {thr} | median error {median:.2f} diameters "
              f"| {(time.time() - t0) / 60:.1f} min")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(dict(model=model.state_dict(), threshold=thr, seq_len=a.seq_len, size=size, epoch=epoch,
                            val_f1=f1, val_precision=p, val_recall=r, val_clips=val_dirs), a.out)
            print(f"  saved {a.out} (best so far)")
    print(f"done. best validation F1 {best_f1:.3f}; checkpoint: {a.out}")


if __name__ == "__main__":
    main()
