"""Training samples: L consecutive frames of one sequence plus target heatmaps."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import STRIDE, to_input


def clip_dirs(root):
    """Prepared sequences (folders with frames.npy + ball.npy), sorted by name."""
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "ball.npy")))


def heatmaps(ball, h, w, sigma=1.0):
    """ball: (L, 4) x, y, diameter, visible in input pixels -> (L, 1, h, w) targets.
    A Gaussian around the ball's cell, exactly 1 at that cell; all zeros without a ball."""
    out = np.zeros((len(ball), 1, h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for t, (x, y, _, vis) in enumerate(ball):
        if vis < 0.5:
            continue
        u, v = x / STRIDE - 0.5, y / STRIDE - 0.5               # cell coordinates
        cx, cy = int(round(u)), int(round(v))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        out[t, 0] = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))
        out[t, 0, cy, cx] = 1.0
    return out


class Clips(Dataset):
    """Every window of `seq_len` frames, starting every `step` frames, in the given sequences.
    With train=True: random horizontal flip and brightness/contrast jitter."""

    def __init__(self, dirs, seq_len=8, step=4, train=True):
        self.dirs, self.seq_len, self.train = dirs, seq_len, train
        self.items = []
        for d in dirs:
            n = len(np.load(os.path.join(d, "ball.npy")))
            self.items += [(d, s) for s in range(0, n - seq_len + 1, step)]
        self.cache = {}                     # memory maps, opened lazily in each worker

    def __len__(self):
        return len(self.items)

    def _arrays(self, d):
        if d not in self.cache:
            self.cache[d] = (np.load(os.path.join(d, "frames.npy"), mmap_mode="r"),
                             np.load(os.path.join(d, "ball.npy")))
        return self.cache[d]

    def __getitem__(self, i):
        d, s = self.items[i]
        frames, ball = self._arrays(d)
        x = np.array(frames[s:s + self.seq_len])
        b = ball[s:s + self.seq_len].copy()
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
                b[:, 0] = np.where(b[:, 3] > 0, x.shape[2] - b[:, 0], 0)
            gain, bias = np.random.uniform(0.8, 1.2), np.random.uniform(-20, 20)
            x = np.clip(x.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        h, w = x.shape[1] // STRIDE, x.shape[2] // STRIDE
        return to_input(x), torch.from_numpy(heatmaps(b, h, w)), torch.from_numpy(b)
