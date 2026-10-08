"""Step 4. Make new dots: start from noise and walk the 1,000 steps back."""
import torch

from data import face, on_face
from diffusion import Denoiser, sample, save, trained

model, _ = trained('dots', Denoiser(2, waves=4), face(20000, torch.Generator().manual_seed(3)), 8000)
gen = torch.Generator().manual_seed(4)
watch = (1000, 750, 500, 250, 100, 50, 20, 10, 5, 0)
x, seen = sample(model, 2000, gen, watch=watch)
print('steps left   dots on the face')
for t in watch:
    print('%10d   %6.1f%%' % (t, 100 * on_face(seen[t])))
print()
print('2,000 new dots, none of them copied: %.1f%% on the face' % (100 * on_face(x)))
real = face(20000, torch.Generator().manual_seed(3))
print('distance from a new dot to the nearest training dot: %.4f on average' % torch.cdist(x, real).min(1).values.mean())
save('step4', rows=[[t, on_face(seen[t])] for t in watch], near=torch.cdist(x, real).min(1).values.mean().item())
