"""Step 8. Work in a smaller room. First squeeze each picture into 8 numbers, then run the diffusion on those 8."""
import os
import time

import torch
from torch import nn

from data import digits, show
from diffusion import HERE, Denoiser, sample, save, trained
from judge import judge, passes, verdict

SMALL = 8
x, y, held_x, held_y = digits()
j = judge()

# the squeezer: 64 numbers -> 8 numbers -> 64 numbers. Trained only to give back the picture it was shown.
squeeze = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, SMALL))
unsqueeze = nn.Sequential(nn.Linear(SMALL, 128), nn.SiLU(), nn.Linear(128, 64))
path = os.path.join(HERE, 'runs', 'squeezer.pt')
if not os.path.exists(path):
    torch.manual_seed(0)
    for layer in list(squeeze) + list(unsqueeze):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(list(squeeze.parameters()) + list(unsqueeze.parameters()), 2e-3)
    for step in range(4000):
        i = torch.randint(len(x), (256,), generator=gen)
        small = squeeze(x[i])
        loss = ((unsqueeze(small) - x[i]) ** 2).mean() + 1e-3 * (small ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save([squeeze.state_dict(), unsqueeze.state_dict()], path)
a, b = torch.load(path)
squeeze.load_state_dict(a)
unsqueeze.load_state_dict(b)

with torch.no_grad():
    back = unsqueeze(squeeze(held_x))
    round_trip = [((back - held_x) ** 2).mean().sqrt().item(), passes(j, back), (verdict(j, back) == held_y).float().mean().item()]
    print('real unseen digits squeezed to %d numbers and opened again:' % SMALL)
    print('  pixel error %.3f; the judge accepts %.1f%% and names %.1f%% correctly (before squeezing: %.1f%% and %.1f%%)' % (
        ((back - held_x) ** 2).mean().sqrt(), 100 * passes(j, back), 100 * (verdict(j, back) == held_y).float().mean(),
        100 * passes(j, held_x), 100 * (verdict(j, held_x) == held_y).float().mean()))
    small = squeeze(x)
    centre, size = small.mean(0), small.std(0)
    small = (small - centre) / size                                # same scale as the noise

print()
print('network           numbers per picture   weights   accepted by the judge   seconds for 2,000 pictures')
rows = []
for name, dim, data, hidden in (('pixels, big', 64, x, 256), ('pixels, small', 64, x, 64), ('small room', SMALL, small, 64)):
    model = Denoiser(dim, hidden=hidden)
    model, _ = trained('room-%s-%d' % (dim, hidden), model, data, 8000)
    start = time.time()
    new = sample(model, 2000, torch.Generator().manual_seed(8))
    took = time.time() - start
    if dim == SMALL:
        with torch.no_grad():
            new = unsqueeze(new * size + centre)
        keep = new
    print('%-16s  %19d   %7d   %20.1f%%   %25.1f' % (name, dim, sum(p.numel() for p in model.parameters()), 100 * passes(j, new), took))
    rows.append([name, dim, sum(p.numel() for p in model.parameters()), passes(j, new), took])
print()
print('eight digits made in the small room (the judge reads them as %s):' % verdict(j, keep[:8]).tolist())
show(keep[:8])
seen_as = verdict(j, keep)
print('which digits come out: ' + '  '.join('%d: %.1f%%' % (d, 100 * (seen_as == d).float().mean()) for d in range(10)))
print('distance to the nearest training digit: %.2f on average' % torch.cdist(keep, x).min(1).values.mean())
print()
print('the same idea at full size: a 512 x 512 colour picture is %s numbers; Stable Diffusion works on 64 x 64 x 4 = %s, %d times fewer' % (
    format(512 * 512 * 3, ','), format(64 * 64 * 4, ','), 512 * 512 * 3 // (64 * 64 * 4)))
torch.save(keep, os.path.join(HERE, 'runs', 'room-digits.pt'))
save('step8', round_trip=round_trip, rows=rows, which=[(seen_as == d).float().mean().item() for d in range(10)],
     near=torch.cdist(keep, x).min(1).values.mean().item())
