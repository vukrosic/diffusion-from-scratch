# Diffusion from scratch

Eight short Python programs that build an image generator of the Stable Diffusion kind, one idea per program: destroy a picture with noise, train a network to guess the noise, walk back from pure noise to a new picture, skip steps, steer with a prompt, and move the whole method into a smaller space. Everything runs on a laptop CPU.

This is not Stable Diffusion. The pictures here are dots (2 numbers) and handwritten digits (64 numbers, 8 x 8 pixels), the "prompt" is one of ten labels, and the networks are small fully connected ones. The method is the same; the scale is about a million times smaller.

Video: the link is in the description of the film this repo belongs to.

## The question

A diffusion model was never shown the picture it makes. So where does the picture come from? The programs answer it with one running score: what share of the made pictures pass as real.

## Run it

```bash
pip install torch numpy
bash run_all.sh
```

Python 3.9 or newer. The first run trains every model and stores it in `runs/`; later runs reuse them. `demo-output.txt` is one saved run, and every number below comes from it. The first run takes 25 to 30 minutes on a laptop CPU. A second run from a clean copy gave the same numbers; only the seconds column of step 8 changes with the machine.

## The practice task

- **Dots.** A real "picture" is a point on a line drawing of a face. A made dot passes if it lands on the drawing. Plain noise lands there 8.6 % of the time.
- **Digits.** 1,297 handwritten digits for training, 500 held back (UCI Optical Recognition of Handwritten Digits, Alpaydin and Kaynak 1998; `digits.csv.gz`). A separate small network, the judge, was trained to name a digit or answer "not a digit". On real digits it never saw it accepts 99.0 % and names 97.4 % correctly; on plain noise it accepts 0.0 %.

| file | idea | result of the saved run |
| --- | --- | --- |
| `step1_noise.py` | add a little noise 1,000 times | dots on the face: 100.0 % → 10.6 %, the same as plain noise |
| `step2_jump.py` | jump to any step with one line instead of walking | step 250: spread 0.797 walked, 0.808 jumped |
| `step3_learn.py` | train a network to guess the noise that was added | new dots on the face: 0.0 % untrained → 98.8 % after 8,000 steps |
| `step4_sample.py` | start from noise and remove the guessed noise 1,000 times | 98.8 % on the face; nearest training dot 0.0033 away on average |
| `step5_shortcut.py` | take fewer, bigger steps | 1 step 2.1 %, 20 steps 51.7 %, 50 steps 90.1 %, 250 steps 97.9 % |
| `step6_digits.py` | the same code on 64-number pictures | 99.8 % accepted by the judge; nearest training digit 2.45 away (real unseen digits: 2.12) |
| `step7_prompt.py` | give the network a label, and a dial for how hard to follow it | right labels: 10.5 % → 99.4 %. Half the labels spoiled: 47.2 % at dial 1, 88.5 % at dial 3, 7.2 % at dial 8 |
| `step8_small_room.py` | squeeze digits to 8 numbers and do the noise work there | 32,392 weights in the small space: 100.0 %. 39,616 weights on pixels: 91.4 %. 453,184 weights on pixels: 99.8 % |

Shared files: `data.py` (the face and the digits), `diffusion.py` (noise table, network, training, the two samplers), `judge.py` (the digit judge), `tests.py` (checks that run first).

## Two things the runs showed that I did not expect

1. **The prompt dial has a narrow good range.** With half of the training labels spoiled, raising the dial from 1 to 3 lifts "the judge sees the digit I asked for" from 47.2 % to 88.5 %. At 4 it falls to 77.6 %, and at 8 it is 7.2 %, with 92.8 % of the outputs no longer digits and 44.2 % of the pixels pushed past full ink or blank paper.
2. **A smaller network won in the small space.** The 32,392-weight network that works on 8 numbers beat the 453,184-weight network that works on 64 pixels (100.0 % against 99.8 %), and a pixel network of about the same size as the small one reached only 91.4 %.

## What this does not show

- Real pictures. 64 squares in grey are far from 512 x 512 in colour.
- A real prompt. Ten labels are not text, and there is no text encoder here.
- An independent judge. The judge is my own small network and it forgives blur; it is not a standard image-quality score.
- A natural labelling problem. The spoiled labels in step 7 are my own setup, made to give the dial something to fix.
- Scale. The model card of Stable Diffusion v1-4 estimates 150,000 hours on A100 graphics cards; these programs train in minutes.

## Sources for the real model

- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models", arXiv 2006.11239 (the noise table, noise guessing, the walk back).
- Song, Meng, Ermon, "Denoising Diffusion Implicit Models", arXiv 2010.02502 (fewer steps).
- Ho, Salimans, "Classifier-Free Diffusion Guidance", arXiv 2207.12598 (the dial).
- Rombach, Blattmann, Lorenz, Esser, Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models", arXiv 2112.10752 (the smaller space).
- Model card `CompVis/stable-diffusion-v1-4` on Hugging Face (factor-8 autoencoder with 4 channels, guidance scales, the five folders, the training-time estimate).

## An experiment to try

Step 7 spoils half of the labels and the dial brings the asked digit back to 88.5 %. Spoil nine out of ten: in `step7_prompt.py` change `< 0.5` to `< 0.9` in the line that picks the wrong labels, and rename the cache key `'sloppy'` so the model trains again. Then sweep the dial. Is there still a setting that brings the digits back? I have not run it.

MIT license.
