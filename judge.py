"""The judge: a separate small network that has never seen a generated picture.
It answers 0..9 for a digit, or 10 for "this is not a digit". It learned "not a digit" from real digits with a lot of noise mixed in."""
import os

import torch
from torch import nn

from data import digits
from diffusion import HERE, T, noisy

JUNK = 10


def judge():
    model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 11))
    path = os.path.join(HERE, 'runs', 'judge.pt')
    if not os.path.exists(path):
        os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
        torch.manual_seed(0)
        for layer in model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        x, y, _, _ = digits()
        gen = torch.Generator().manual_seed(0)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        for step in range(3000):
            i = torch.randint(len(x), (256,), generator=gen)
            clean = x[i] + 0.1 * torch.randn(256, 64, generator=gen)                       # real digits, slightly shaken
            t = torch.randint(100, T + 1, (256,), generator=gen)
            junk = noisy(x[i], t, torch.randn(256, 64, generator=gen))                     # digits drowned in noise
            loss = nn.functional.cross_entropy(model(torch.cat([clean, junk])), torch.cat([y[i], torch.full((256,), JUNK)]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path))
    return model.eval()


@torch.no_grad()
def verdict(model, x):
    """For each picture: the digit the judge sees, or 10."""
    return model(x).argmax(1)


def passes(model, x):
    """Share of the pictures the judge accepts as a digit."""
    return (verdict(model, x) != JUNK).float().mean().item()


if __name__ == '__main__':
    j = judge()
    _, _, x, y = digits()
    gen = torch.Generator().manual_seed(0)
    print('held-out real digits: %.1f%% accepted as a digit, %.1f%% named correctly' % (
        100 * passes(j, x), 100 * (verdict(j, x) == y).float().mean()))
    print('plain noise: %.1f%% accepted' % (100 * passes(j, torch.randn(2000, 64, generator=gen))))
    for t in (10, 25, 50, 100, 200, 500):
        print('real digits at noise step %3d: %.1f%% accepted' % (t, 100 * passes(j, noisy(x, torch.full((len(x),), t), torch.randn(x.shape, generator=gen)))))
