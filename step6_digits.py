"""Step 6. The same code on real pictures: handwritten digits, 8 x 8 pixels, 64 numbers each."""
import torch

from data import digits, show
from diffusion import Denoiser, sample, sample_fast, save, trained
from judge import judge, passes, verdict

x, y, held_x, held_y = digits()
model = Denoiser(64)
print('%d training digits, %d weights in the network' % (len(x), sum(p.numel() for p in model.parameters())))
model, curve = trained('digits', model, x, 8000)
print('error of the noise guess: %.3f at the start, %.3f at the end' % (curve[0], sum(curve[-10:]) / 10))

j = judge()
gen = torch.Generator().manual_seed(6)
print()
print('the judge on real digits it never saw: %.1f%% accepted, %.1f%% named correctly' % (
    100 * passes(j, held_x), 100 * (verdict(j, held_x) == held_y).float().mean()))
print('the judge on plain noise: %.1f%% accepted' % (100 * passes(j, torch.randn(2000, 64, generator=gen))))

new, seen = sample(model, 2000, gen, watch=(1000, 500, 250, 100, 50, 0))
print()
print('steps left   accepted by the judge')
for t in (1000, 500, 250, 100, 50, 0):
    print('%10d   %6.1f%%' % (t, 100 * passes(j, seen[t])))
print()
print('eight new digits (the judge reads them as %s):' % verdict(j, new[:8]).tolist())
show(new[:8])
print()
seen_as = verdict(j, new)
print('which digits come out: ' + '  '.join('%d: %.1f%%' % (d, 100 * (seen_as == d).float().mean()) for d in range(10)))

# did it copy? compare each new digit with its nearest training digit
near_new = torch.cdist(new, x).min(1)
near_real = torch.cdist(held_x, x).min(1)
print()
print('distance to the nearest training digit: new digits %.2f on average, real unseen digits %.2f' % (
    near_new.values.mean(), near_real.values.mean()))
print('new digits closer than 1.0 to a training digit: %.1f%%   (real unseen digits: %.1f%%)' % (
    100 * (near_new.values < 1.0).float().mean(), 100 * (near_real.values < 1.0).float().mean()))
print()
print('steps   accepted by the judge')
fast = [[steps, passes(j, sample_fast(model, 2000, steps, gen))] for steps in (1, 5, 10, 20, 50, 100)]
for steps, share in fast:
    print('%5d   %6.1f%%' % (steps, 100 * share))
save('step6', train=len(x), weights=sum(p.numel() for p in model.parameters()), error=[curve[0], sum(curve[-10:]) / 10],
     judge_real=[passes(j, held_x), (verdict(j, held_x) == held_y).float().mean().item()],
     walk=[[t, passes(j, seen[t])] for t in (1000, 500, 250, 100, 50, 0)], accepted=passes(j, new),
     which=[(seen_as == d).float().mean().item() for d in range(10)],
     near=[near_new.values.mean().item(), near_real.values.mean().item()],
     close=[(near_new.values < 1.0).float().mean().item(), (near_real.values < 1.0).float().mean().item()], fast=fast)
