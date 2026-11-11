"""BallNet: CNN encoder -> ConvLSTM -> heatmap decoder.

    frame (3 x H x W), e.g. 3 x 720 x 1280
      -> ResNet-18 stem + layer1 + layer2 (ImageNet weights)   128 x H/8 x W/8
      -> ConvLSTM, memory over the previous frames             64 x H/8 x W/8
      -> decoder with skip connections from 1/4 and 1/2        32 x H/2 x W/2
      -> 1 x H/2 x W/2 heatmap: chance that the ball's centre is in each cell
    The network is fully convolutional, so it works at any size that is a multiple of 8.

The ConvLSTM is an LSTM whose hidden state is a feature map instead of a vector,
so it remembers *where* the ball was and how it moved. A plain nn.LSTM on a
pooled feature vector would lose the position of a ball only a few pixels wide.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STRIDE = 2                                            # heatmap cell = 2 x 2 input pixels
MEAN = np.array([0.485, 0.456, 0.406], np.float32)    # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, ch, k=3):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=k // 2)

    def forward(self, x, state=None):
        if state is None:
            zeros = x.new_zeros(x.shape[0], self.ch, *x.shape[2:])
            state = (zeros, zeros)
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # 1/2, 64 channels
        self.pool = r.maxpool
        self.layer1 = r.layer1                                  # 1/4, 64
        self.layer2 = r.layer2                                  # 1/8, 128
        self.lstm = ConvLSTMCell(128, 64)
        self.up1 = conv_block(64 + 64, 64)                      # 1/4
        self.up2 = conv_block(64 + 64, 32)                      # 1/2
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.head.bias, -4.6)                 # start at p = 0.01 (focal-loss practice)

    def forward(self, x, state=None):
        """x: (B, L, 3, H, W). Returns heatmap logits (B, L, 1, H/2, W/2) and the LSTM state."""
        B, L = x.shape[:2]
        f = x.flatten(0, 1)
        s2 = self.stem(f)
        s4 = self.layer1(self.pool(s2))
        s8 = self.layer2(s4)
        s8 = s8.view(B, L, *s8.shape[1:])
        hs = []
        for t in range(L):                                      # the recurrence over time
            h, state = self.lstm(s8[:, t], state)
            hs.append(h)
        h = torch.stack(hs, 1).flatten(0, 1)
        y = self.up1(torch.cat([F.interpolate(h, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], 1))
        y = self.up2(torch.cat([F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        out = self.head(y)
        return out.view(B, L, 1, *out.shape[-2:]), state


def focal_loss(logits, target):
    """CenterNet focal loss: the target is 1 at the ball's cell and a Gaussian
    around it; cells near the ball are penalised less for firing. Normalised by
    the number of balls, so the many empty cells do not swamp the loss."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * (1 - pos)
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def to_input(frames_bgr):
    """uint8 BGR frames (L, H, W, 3) -> normalised float tensor (L, 3, H, W)."""
    rgb = frames_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(0, 3, 1, 2).copy())


def decode(prob):
    """prob: heatmap (h, w) as numpy. Returns (confidence, x, y) in input pixels;
    the position is refined with the weighted centre of the 3x3 cells around the peak."""
    i = int(prob.argmax())
    cy, cx = divmod(i, prob.shape[1])
    y0, x0 = max(cy - 1, 0), max(cx - 1, 0)
    win = prob[y0:cy + 2, x0:cx + 2]
    yy, xx = np.mgrid[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
    s = win.sum()
    fx, fy = (float((win * xx).sum() / s), float((win * yy).sum() / s)) if s > 0 else (cx, cy)
    return float(prob[cy, cx]), (fx + 0.5) * STRIDE, (fy + 0.5) * STRIDE
