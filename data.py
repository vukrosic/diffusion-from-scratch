"""The data. Two kinds of "picture":
- a dot: a picture with only 2 numbers (x, y). Real dots lie on a drawing of a face.
- a digit: a picture with 64 numbers, 8 x 8 pixels, handwritten (UCI Optical Recognition of Handwritten Digits,
  Alpaydin and Kaynak 1998, the 1,797 digits that ship with scikit-learn)."""
import gzip
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# the face: (centre x, centre y, radius, first angle, last angle) in degrees
PARTS = [(0, 0, 0.9, 0, 360), (-0.32, 0.3, 0.1, 0, 360), (0.32, 0.3, 0.1, 0, 360), (0, 0, 0.5, 210, 330)]
NEAR = 0.05                                    # a dot this close to a line of the face counts as "on the face"


def _curve(u):
    """u: numbers in 0..1. Returns a point on the face for each, spread evenly along its lines."""
    length = torch.tensor([r * math.radians(b - a) for _, _, r, a, b in PARTS])
    edge = torch.cumsum(length, 0) / length.sum()
    part = torch.bucketize(u, edge).clamp(max=len(PARTS) - 1)
    start = torch.cat([torch.zeros(1), edge])[part]
    along = (u - start) / (edge[part] - start)
    p = torch.tensor(PARTS)[part]
    angle = torch.deg2rad(p[:, 3] + along * (p[:, 4] - p[:, 3]))
    return torch.stack([p[:, 0] + p[:, 2] * angle.cos(), p[:, 1] + p[:, 2] * angle.sin()], 1)


def face(n, gen):
    """n real dots."""
    return _curve(torch.rand(n, generator=gen)) + 0.01 * torch.randn(n, 2, generator=gen)


_LINE = _curve(torch.linspace(0, 1, 4000))


def on_face(x):
    """Share of the dots that lie on the face."""
    return (torch.cdist(x, _LINE).min(1).values < NEAR).float().mean().item()


def digits(n_test=500, seed=0):
    """Returns train pictures, train labels, held-out pictures, held-out labels. Pixels run from -1 (paper) to 1 (ink)."""
    raw = np.loadtxt(gzip.open(os.path.join(HERE, 'digits.csv.gz'), 'rt'), delimiter=',')
    order = np.random.default_rng(seed).permutation(len(raw))
    x = torch.tensor(raw[order, :64] / 8.0 - 1, dtype=torch.float32)
    y = torch.tensor(raw[order, 64], dtype=torch.long)
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test]


def show(pictures, levels=' .:*#'):
    """Print 8 x 8 pictures side by side with text characters."""
    pictures = torch.as_tensor(pictures).reshape(-1, 8, 8)
    for row in range(8):
        print('   '.join(''.join(levels[min(4, int(max(v.item() + 1, 0) / 2 * 5))] * 2 for v in p[row]) for p in pictures))


if __name__ == '__main__':
    gen = torch.Generator().manual_seed(0)
    print('real dots on the face: %.1f%%' % (100 * on_face(face(2000, gen))))
    print('random dots on the face: %.1f%%' % (100 * on_face(torch.randn(2000, 2, generator=gen))))
    x, y, _, _ = digits()
    print(len(x), 'training digits; the first three are', y[:3].tolist())
    show(x[:3])
