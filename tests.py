"""Checks. Run first by run_all.sh."""
import torch

from data import digits, face, on_face
from diffusion import BETA, KEEP, T, Denoiser, clock, noisy, sample_fast

gen = torch.Generator().manual_seed(0)

# the schedule: nothing mixed in at step 0, almost nothing of the picture left at the last step
assert BETA[0] == 0 and KEEP[0] == 1 and KEEP[T] < 1e-4
assert torch.all(KEEP[1:] < KEEP[:-1])

# the jump keeps the size of the data: a picture of spread 1 stays at spread 1 at every step
x0 = torch.randn(50000, 2, generator=gen)
for t in (1, 300, 1000):
    assert abs(noisy(x0, torch.full((len(x0),), t), torch.randn(x0.shape, generator=gen)).std() - 1) < 0.02

# with zero noise mixed in, the jump is only a shrink
assert torch.allclose(noisy(x0, torch.full((len(x0),), 250), torch.zeros_like(x0)), KEEP[250].sqrt() * x0)

# real dots are on the face, noise mostly is not
assert on_face(face(2000, gen)) > 0.99 and on_face(torch.randn(2000, 2, generator=gen)) < 0.2

# the digits: 1,797 pictures of 64 pixels in -1..1, split 1,297 / 500
x, y, hx, hy = digits()
assert x.shape == (1297, 64) and hx.shape == (500, 64) and x.min() == -1 and x.max() == 1 and set(y.tolist()) == set(range(10))

# the clock gives a different reading for every step; the network answers with one number per input number
assert clock(torch.arange(1, T + 1)).unique(dim=0).shape[0] == T
net = Denoiser(64, classes=10)
assert net(torch.randn(5, 64), torch.full((5,), 7), torch.tensor([0, 1, 2, 3, 10])).shape == (5, 64)

# the same seed gives the same pictures
a = sample_fast(net.eval(), 4, 5, torch.Generator().manual_seed(1))
b = sample_fast(net, 4, 5, torch.Generator().manual_seed(1))
assert torch.equal(a, b)
print('all checks pass')
