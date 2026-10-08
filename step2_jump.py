"""Step 2. The shortcut: one line of arithmetic lands on any step, without walking there."""
import torch

from data import face, on_face
from diffusion import BETA, KEEP, noisy, save

print('picture left after t steps (KEEP):  ' + '   '.join('t=%d: %.4f' % (t, KEEP[t]) for t in (1, 100, 250, 500, 1000)))
print()
gen = torch.Generator().manual_seed(2)
x0 = face(20000, gen)
print('step   walked there (t small steps)        jumped there (1 line)')
print('       spread x  spread y  on face         spread x  spread y  on face')
walk, rows = x0.clone(), []
for t in range(1, 501):
    walk = (1 - BETA[t]).sqrt() * walk + BETA[t].sqrt() * torch.randn(walk.shape, generator=gen)
    if t in (100, 250, 500):
        jump = noisy(x0, torch.full((len(x0),), t), torch.randn(x0.shape, generator=gen))
        print('%4d   %8.3f  %8.3f  %6.1f%%         %8.3f  %8.3f  %6.1f%%' % (
            t, walk[:, 0].std(), walk[:, 1].std(), 100 * on_face(walk), jump[:, 0].std(), jump[:, 1].std(), 100 * on_face(jump)))
        rows.append([t, walk[:, 0].std().item(), walk[:, 1].std().item(), on_face(walk), jump[:, 0].std().item(), jump[:, 1].std().item(), on_face(jump)])
save('step2', keep={t: KEEP[t].item() for t in (1, 100, 250, 500, 1000)}, rows=rows)
