"""The shared parts: the noise schedule, the small network, training and sampling."""
import json
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))

T = 1000                                                           # number of noise steps
BETA = torch.cat([torch.zeros(1), torch.linspace(1e-4, 0.02, T)])  # BETA[t]: the share of noise that step t mixes in
KEEP = torch.cumprod(1 - BETA, 0)                                  # KEEP[t]: how much of the picture is left after t steps


def noisy(x0, t, eps):
    """Jump from the clean pictures x0 straight to step t. eps is the noise to mix in."""
    k = KEEP[t].unsqueeze(1)
    return k.sqrt() * x0 + (1 - k).sqrt() * eps


def clock(t):
    """Turn the step number into 32 numbers the network can read (waves of different lengths)."""
    speed = torch.exp(-math.log(10000) * torch.arange(16) / 16)
    angle = t.float().unsqueeze(1) * speed
    return torch.cat([angle.sin(), angle.cos()], 1)


class Block(nn.Module):
    """One layer of thinking. It is told the step number every time."""

    def __init__(self, hidden):
        super().__init__()
        self.a, self.b, self.step = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden), nn.Linear(32, hidden)

    def forward(self, h, when):
        return h + self.b(F.silu(self.a(F.silu(h)) + self.step(when)))


class Denoiser(nn.Module):
    """Sees a noisy picture and the step number. Answers with its guess of the noise in it.
    waves > 0: each number of the picture is also given as waves of different lengths, which helps with fine detail.
    classes > 0: it also sees a label, the prompt. The last label means "no prompt"."""

    def __init__(self, dim, hidden=256, blocks=3, waves=0, classes=0):
        super().__init__()
        self.dim, self.waves, self.classes = dim, waves, classes
        self.picture = nn.Linear(dim * (1 + 2 * waves), hidden)
        self.prompt = nn.Embedding(classes + 1, hidden) if classes else None
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(blocks)])
        self.out = nn.Linear(hidden, dim)

    def forward(self, x, t, label=None):
        if self.waves:
            angle = torch.cat([x * math.pi * 2 ** k for k in range(self.waves)], 1)
            x = torch.cat([x, angle.sin(), angle.cos()], 1)
        h, when = self.picture(x), clock(t)
        if self.prompt is not None:
            h = h + self.prompt(label if label is not None else torch.full((len(x),), self.classes))
        for block in self.blocks:
            h = block(h, when)
        return self.out(F.silu(h))


def train(model, data, steps, labels=None, drop=0.15, batch=512, lr=1e-3, seed=0):
    """Mix noise into real pictures and ask the model which noise it was. Returns the loss every 50 steps."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    slow = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve = []
    for step in range(steps):
        i = torch.randint(len(data), (batch,), generator=gen)
        t = torch.randint(1, T + 1, (batch,), generator=gen)
        eps = torch.randn(batch, data.shape[1], generator=gen)
        label = None
        if labels is not None:                                     # sometimes hide the prompt
            label = labels[i].clone()
            label[torch.rand(batch, generator=gen) < drop] = model.classes
        loss = ((model(noisy(data[i], t, eps), t, label) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        slow.step()
        if step % 50 == 0:
            curve.append(round(loss.item(), 4))
    return curve


def trained(name, model, data, steps, **more):
    """Train once and keep the result in runs/. A second call loads it."""
    path = os.path.join(HERE, 'runs', name)
    if not os.path.exists(path + '.pt'):
        os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
        torch.manual_seed(more.get('seed', 0))
        for layer in model.modules():                              # same starting weights on every run
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        curve = train(model, data, steps, **more) if steps else []
        torch.save(model.state_dict(), path + '.pt')
        with open(path + '.json', 'w') as f:
            json.dump(curve, f)
    model.load_state_dict(torch.load(path + '.pt'))
    with open(path + '.json') as f:
        return model.eval(), json.load(f)


def save(name, **numbers):
    """Keep a step's numbers in runs/<name>.json."""
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    with open(os.path.join(HERE, 'runs', name + '.json'), 'w') as f:
        json.dump(numbers, f)


@torch.no_grad()
def guess(model, x, t, label=None, guide=1.0):
    """The model's guess of the noise. With a prompt: the guess without it, pushed `guide` times towards the guess with it."""
    step = torch.full((len(x),), int(t))
    if label is None:
        return model(x, step)
    without = model(x, step)
    return without + guide * (model(x, step, label) - without)


@torch.no_grad()
def sample(model, n, gen, label=None, guide=1.0, watch=()):
    """Start from noise and take all T steps back (Ho et al. 2020). `watch`: step numbers at which to keep a copy."""
    x = torch.randn(n, model.dim, generator=gen)
    seen = {T: x.clone()} if T in watch else {}
    for t in range(T, 0, -1):
        eps = guess(model, x, t, label, guide)
        x = (x - BETA[t] / (1 - KEEP[t]).sqrt() * eps) / (1 - BETA[t]).sqrt()   # take out this step's share of noise
        if t > 1:
            x = x + BETA[t].sqrt() * torch.randn(x.shape, generator=gen)        # and put a little fresh noise back
        if t - 1 in watch:
            seen[t - 1] = x.clone()
    return (x, seen) if watch else x


@torch.no_grad()
def sample_fast(model, n, steps, gen, label=None, guide=1.0):
    """Take only `steps` big steps back (Song et al. 2020, DDIM). No fresh noise is added on the way."""
    x = torch.randn(n, model.dim, generator=gen)
    stops = torch.linspace(T, 0, steps + 1).round().long().tolist()
    for t, s in zip(stops[:-1], stops[1:]):
        eps = guess(model, x, t, label, guide)
        clean = ((x - (1 - KEEP[t]).sqrt() * eps) / KEEP[t].sqrt()).clamp(-1, 1)   # the clean picture this guess implies
        eps = (x - KEEP[t].sqrt() * clean) / (1 - KEEP[t]).sqrt()
        x = KEEP[s].sqrt() * clean + (1 - KEEP[s]).sqrt() * eps                    # noise it again, but only to step s
    return x
