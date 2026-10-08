"""Step 1. Destroy a picture slowly: 1,000 small steps, each mixes in a little noise."""
import torch

from data import face, on_face
from diffusion import BETA, T, save

gen = torch.Generator().manual_seed(1)
x = face(2000, gen)
print('step     noise mixed in at this step   dots still on the face')
print('%4d     %26s   %6.1f%%' % (0, '-', 100 * on_face(x)))
rows = [[0, 0.0, on_face(x)]]
for t in range(1, T + 1):
    x = (1 - BETA[t]).sqrt() * x + BETA[t].sqrt() * torch.randn(x.shape, generator=gen)   # shrink a little, add a little noise
    if t in (1, 50, 100, 250, 500, 1000):
        print('%4d     %25.2f%%   %6.1f%%' % (t, 100 * BETA[t], 100 * on_face(x)))
        rows.append([t, BETA[t].item(), on_face(x)])
print()
noise = torch.randn(2000, 2, generator=gen)
print('after 1,000 steps: centre (%.2f, %.2f), spread %.2f' % (x[:, 0].mean(), x[:, 1].mean(), x.std()))
print('plain noise:       centre (%.2f, %.2f), spread %.2f, on the face %.1f%%' % (
    noise[:, 0].mean(), noise[:, 1].mean(), noise.std(), 100 * on_face(noise)))
save('step1', rows=rows, end=[x[:, 0].mean().item(), x[:, 1].mean().item(), x.std().item()],
     noise=[noise[:, 0].mean().item(), noise[:, 1].mean().item(), noise.std().item(), on_face(noise)])
