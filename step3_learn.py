"""Step 3. Teach a small network one job: look at a noisy dot and say which noise is in it."""
import torch

from data import face, on_face
from diffusion import Denoiser, T, noisy, sample, save, trained

gen = torch.Generator().manual_seed(3)
data = face(20000, gen)
model = Denoiser(2, waves=4)
print('weights in the network:', sum(p.numel() for p in model.parameters()))
print()
print('training steps   error of the noise guess   new dots that land on the face')
stages, by_noise = [], []
for steps in (0, 100, 500, 2000, 8000):
    model, curve = trained('dots' if steps == 8000 else 'dots-%d' % steps, Denoiser(2, waves=4), data, steps)
    with torch.no_grad():
        t = torch.randint(1, T + 1, (5000,), generator=gen)
        eps = torch.randn(5000, 2, generator=gen)
        error = ((model(noisy(data[:5000], t, eps), t) - eps) ** 2).mean()
    new = sample(model, 2000, torch.Generator().manual_seed(4))
    print('%14d   %24.3f   %29.1f%%' % (steps, error, 100 * on_face(new)))
    stages.append([steps, error.item(), on_face(new)])
print('(an error of 1.0 is what you get by always answering "no noise")')
print()
print('error of the finished network, by how noisy the dot is:')
test = face(5000, gen)
with torch.no_grad():
    for t in (1, 10, 50, 100, 250, 500, 1000):
        step = torch.full((len(test),), t)
        eps = torch.randn(test.shape, generator=gen)
        by_noise.append([t, ((model(noisy(test, step, eps), step) - eps) ** 2).mean().item()])
        print('  step %4d   %.3f' % tuple(by_noise[-1]))
save('step3', weights=sum(p.numel() for p in model.parameters()), stages=stages, by_noise=by_noise, curve=curve)
