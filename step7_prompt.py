"""Step 7. Ask for a digit. The network also gets a label (the prompt), and a dial says how hard to follow it."""
import os

import torch

from data import digits, show
from diffusion import HERE, Denoiser, sample, save, trained
from judge import JUNK, judge, verdict

x, y, held_x, held_y = digits()
j = judge()
ask = torch.arange(10).repeat_interleave(200)                      # 200 requests for each digit


def report(model, dial):
    new = sample(model, len(ask), torch.Generator().manual_seed(7), label=ask, guide=dial)
    seen = verdict(j, new)
    spread = torch.stack([new[ask == d].std(0).mean() for d in range(10)]).mean()
    return new, (100 * (seen == ask).float().mean().item(), 100 * (seen == JUNK).float().mean().item(), spread.item(), 100 * (new.abs() > 1.05).float().mean().item())


print('A. every training digit has the right label')
model, _ = trained('prompted', Denoiser(64, classes=10), x, 8000, labels=y)
clean = {}
for dial, name in ((0.0, 'prompt ignored (dial 0)'), (1.0, 'prompt used (dial 1)')):
    new, r = report(model, dial)
    clean[dial] = r
    torch.save(new, os.path.join(HERE, 'runs', 'clean-dial%g.pt' % dial))
    print('  %-24s the judge sees the digit I asked for: %5.1f%%' % (name, r[0]))
print('  asked for 0 1 2 3 4 5 6 7, dial 1:')
show(new[torch.arange(8) * 200])

print()
gen = torch.Generator().manual_seed(70)
wrong = torch.rand(len(y), generator=gen) < 0.5
sloppy = torch.where(wrong, torch.randint(10, (len(y),), generator=gen), y)      # half the labels are replaced by a random one
print('B. sloppy labels: only %.1f%% of the training labels are still right' % (100 * (sloppy == y).float().mean()))
model, _ = trained('sloppy', Denoiser(64, classes=10), x, 8000, labels=sloppy)
print('  real digits: spread inside one class %.3f' % torch.stack([x[y == d].std(0).mean() for d in range(10)]).mean())
print('  dial   asked digit seen   not a digit   spread inside one class   pixels past full ink or full paper')
rows = []
for dial in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
    new, r = report(model, dial)
    rows.append((dial,) + r)
    torch.save(new, os.path.join(HERE, 'runs', 'sloppy-dial%g.pt' % dial))
    print('  %4.1f   %15.1f%%   %10.1f%%   %23.3f   %33.1f%%' % ((dial,) + r))
    if dial in (1.0, 3.0, 8.0):
        print('  asked for 0 1 2 3 4 5 6 7, dial %g:' % dial)
        show(new[torch.arange(8) * 200])
save('step7', clean=[clean[0.0][0], clean[1.0][0]], labels_right=(sloppy == y).float().mean().item(),
     real_spread=torch.stack([x[y == d].std(0).mean() for d in range(10)]).mean().item(), rows=rows)
