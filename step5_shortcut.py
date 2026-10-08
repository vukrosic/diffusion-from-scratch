"""Step 5. Try to skip the walk. One big step fails; a few big steps work."""
import torch

from data import face, on_face
from diffusion import KEEP, T, Denoiser, guess, sample, sample_fast, save, trained

model, _ = trained('dots', Denoiser(2, waves=4), face(20000, torch.Generator().manual_seed(3)), 8000)
gen = torch.Generator().manual_seed(5)

x = torch.randn(2000, 2, generator=gen)
one = (x - (1 - KEEP[T]).sqrt() * guess(model, x, T)) / KEEP[T].sqrt()        # take out all the noise the model sees, at once
print('one step, no clamp:   %5.1f%% on the face, average distance from the centre %.2f' % (100 * on_face(one), one.norm(dim=1).mean()))
print()
print('steps   dots on the face   runs of the network')
rows = []
for steps in (1, 2, 5, 10, 20, 50, 100, 250, 500):
    x = sample_fast(model, 2000, steps, gen)
    print('%5d   %15.1f%%   %19d' % (steps, 100 * on_face(x), steps))
    rows.append([steps, on_face(x)])
x = sample(model, 2000, gen)
print('%5d   %15.1f%%   %19d   (the full walk of step 4)' % (T, 100 * on_face(x), T))
save('step5', one=[on_face(one), one.norm(dim=1).mean().item()], rows=rows + [[T, on_face(x)]])
