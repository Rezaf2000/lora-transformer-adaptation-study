# LoRA from scratch

![CI](https://github.com/delcenjo/lora-from-scratch/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An annotated implementation of LoRA -- Hu et al., ["LoRA: Low-Rank Adaptation
of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021) -- built
from the paper's equations in plain PyTorch, plus a real experiment: pretrain
a tiny character-level GPT on Shakespeare, then adapt it to Don Quijote five
different ways (full fine-tuning, LoRA at three ranks, and head-only
fine-tuning) and compare what each one actually costs and forgets.

The code in `src/lora/lora.py` is the point of this repository. It is a
single file, fewer than 250 lines including comments, and the comments are
not decoration: every line of arithmetic is tied back to the equation in the
paper that motivates it, including the two decisions that are easy to gloss
over -- why B starts at zero, and what merging weights actually buys you.

## The idea

Fine-tuning a large pretrained weight matrix W0 updates every one of its
entries. The paper's starting observation is that this update, ΔW, tends to
have low "intrinsic rank": most of what changes during fine-tuning can be
captured by a much smaller matrix than W0 itself. So instead of learning a
full ΔW (same shape as W0, d x k parameters), LoRA factorizes it as the
product of two skinny matrices and freezes W0 entirely:

```
ΔW = B A,       B in R^(d x r),  A in R^(r x k),  r << min(d, k)

h  = W0 x + (alpha / r) B A x
```

Only A and B are trained; W0 never receives a gradient. A is initialized
with small random values, B is initialized to all zeros. That single choice
is why LoRA can be attached to a working model without breaking it: at step
zero, B A = 0 regardless of what A contains, so the adapted model starts as
an exact copy of the pretrained one, and training moves it away from that
point gradually. The `alpha / r` scaling lets you change the rank without
having to retune the learning rate every time -- it keeps the update's
magnitude at initialization comparable across different values of r.

At inference time, the two branches can be folded into one:
`W' = W0 + (alpha / r) B A`. A merged LoRALinear runs exactly one matmul,
the same cost as the plain layer it replaced -- the low-rank structure is a
training-time and storage-time trick, not an inference-time one.

`src/lora/lora.py` implements this as `LoRALinear`, a module that wraps an
existing `nn.Linear`, plus `apply_lora()` to attach it to every attention
projection in a model by name, `mark_only_lora_as_trainable()` to freeze
everything else, and `merge()` / `unmerge()` / an `enabled` switch to move
between the two-branch and folded forms.

## The experiment

`scripts/run_experiment.py` runs the whole thing end to end:

1. **Pretrain** a small GPT (2 layers, 2 heads, 128-d embeddings, 128-token
   context, character-level tokenizer) on the tiny Shakespeare corpus.
2. **Adapt** that exact pretrained checkpoint to Don Quijote -- a different
   language, a different register, a visibly different character set -- five
   separate ways, all given the same number of steps and the same data:
   - **full**: every parameter trainable, the ordinary fine-tune.
   - **LoRA, r = 1 / 4 / 16**: only the attention projections (the fused
     qkv layer and the output projection) get an adapter; everything else
     stays frozen.
   - **head**: only the final output layer is trainable; the entire
     transformer body, attention included, stays exactly as pretrained.
3. **Measure**, for every method: how many parameters were actually
   trainable, the final validation loss on Quijote, and -- the part a normal
   fine-tuning writeup tends to skip -- the validation loss back on
   Shakespeare *after* adapting to Quijote. That last column is catastrophic
   forgetting made concrete.
4. For the LoRA runs specifically, also evaluate Shakespeare with the
   adapters switched off. Since LoRA never writes into the frozen base
   weights, that number should reproduce the pretrained model's original
   Shakespeare loss almost exactly, no matter how much the adapters moved
   during Quijote training.

## Results

Model: 2 layers, 2 heads, 128-d embeddings, 128-token context, 438,016
parameters, character-level vocabulary of 97 symbols (the union of both
corpora). Pretrained on Shakespeare for 2,500 steps, then adapted to Quijote
for 800 steps per method, batch size 64. Total wall time for the whole
experiment (pretraining plus all five adaptations): 20.9 minutes on a
laptop CPU.

| method | trainable params | % of total | val loss B (Quijote) | val loss A after adapting | val loss A, adapter off |
| --- | --- | --- | --- | --- | --- |
| pretrained (no adaptation) | 0 | 0.00% | -- | 1.9074 | -- |
| full fine-tune | 438,016 | 100.00% | **1.7567** | 3.5686 | -- |
| LoRA r=1 | 1,536 | 0.35% | 2.5490 | 2.5163 | 1.9060 |
| LoRA r=4 | 6,144 | 1.38% | 2.3996 | 2.7474 | 1.9142 |
| LoRA r=16 | 24,576 | 5.31% | 2.2862 | 2.8979 | 1.9094 |
| head only | 12,416 | 2.83% | 2.2377 | 2.6867 | -- |

What this actually shows, at this scale:

- **Full fine-tuning wins on raw adaptation quality** (1.7567 val loss on
  Quijote, clearly the best) but pays for it with the worst forgetting by
  far: Shakespeare's loss goes from 1.9074 to 3.5686, worse than an
  untrained model would score on English text it had never seen. Every one
  of the 438K parameters was free to move, and 800 steps on a very
  different corpus moved plenty of them away from what pretraining had
  found.
- **LoRA's adaptation quality scales with rank, roughly as expected**:
  r=1 barely moves off the pretrained model (2.5490), r=4 and r=16 get
  closer to the full fine-tune and head-only numbers (2.3996, 2.2862) while
  training 1.38% and 5.31% of the parameters respectively.
- **The forgetting number for LoRA "with adapters on" is confusing on its
  own** -- it gets *worse* as rank grows (2.5163 to 2.8979), which looks at
  first like more capacity means more forgetting. It does not: LoRA's
  frozen base weights are bit-for-bit untouched in every one of these runs.
  What is moving is the adapter itself, which was trained purely on
  Quijote and, mixed into the forward pass unconditionally, actively pulls
  predictions toward Spanish even when the input is Shakespeare. A bigger
  adapter pulls harder.
- **The number that actually matters is the last column.** With the
  adapters switched off, every LoRA run reproduces the pretrained
  Shakespeare loss to within 0.007 (1.9060, 1.9142, 1.9094 against a
  baseline of 1.9074) -- differences small enough to be batch-sampling
  noise in the validation estimate, not model drift. That is the concrete
  version of "LoRA does not forget": it is not that the adapted model
  happens to remember the old task, it is that the old model is still
  sitting there, completely unedited, one boolean flag away. Full
  fine-tuning and head-only training have no equivalent flag -- their
  original weights are gone the moment training starts.
- **Head-only training is a surprisingly strong, cheap baseline here**:
  2.83% of the parameters reach the best Quijote loss of all the
  restricted methods (2.2377, edging out LoRA r=16), because with only two
  transformer layers most of what separates "predict English" from
  "predict Spanish" is which characters the output layer scores highly,
  not how attention moves information around. That is a property of this
  toy model's shallowness more than of head-only fine-tuning in general;
  see Limitations below.

Full metrics (per-step training/validation curves, generation samples) are
in [`results/results.json`](results/results.json), produced by an actual run
of `scripts/run_experiment.py`, not hand-edited.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

python scripts/download_data.py     # tiny Shakespeare + Don Quijote (public domain)
python scripts/run_experiment.py    # pretrain + all five adaptation methods, ~20-30 min on CPU
pytest                              # unit tests for the LoRA layer and the model
jupyter nbconvert --to notebook --execute notebook.ipynb   # re-run the walkthrough
```

`run_experiment.py` writes `checkpoints/pretrained.pt` and
`results/results.json`; the notebook reads the latter, it does not retrain
anything itself, so it stays fast to open and re-execute.

## Tests

`tests/test_lora.py` checks the claims made in the comments, not just that
the code runs: B = 0 at initialization reproduces the frozen base layer's
output exactly, gradients flow only into A and B, a merged layer computes
bit-for-bit the same thing as the unmerged two-branch forward, and disabling
the adapter ignores whatever A and B learned. `tests/test_model.py` checks
the underlying GPT: output shapes, that it is genuinely causal (a later
token cannot change an earlier position's logits), and the parameter count.

## Limitations

This is a deliberately tiny setup -- a few hundred thousand parameters,
two short corpora, a CPU budget of well under an hour -- built to make the
*mechanism* of LoRA checkable end to end rather than to produce numbers that
generalize. A few things that would change at real scale:

- The paper adapts models with billions of parameters; the gap between
  "0.35% of parameters trainable" and "full fine-tune" narrows or widens
  differently depending on how much spare capacity the base model has.
  Whether rank 4 beats rank 1 here says very little about rank 8 vs 16 on
  a 7B model.
- Both corpora are small enough that even the "full" fine-tune is not
  fighting overfitting the way a production fine-tune would; the
  catastrophic-forgetting numbers should be read as directional (full
  and head-only both move away from Shakespeare, LoRA-with-adapters-off
  does not), not as a precise forgetting rate.
- LoRA is only applied to the attention projections here, matching the
  paper's main experiments; the feed-forward layers were left out of scope
  to keep the parameter-count comparison simple, not because adapting them
  would be wrong.

## Notes and non-obvious decisions

- **Shared vocabulary, built from both corpora up front.** The character
  vocabulary is the union of Shakespeare's and Quijote's characters (97
  distinct characters total), built before pretraining even starts. This
  is what lets the exact same pretrained embedding table and output head
  be reused during adaptation without resizing anything -- the Quijote-only
  characters (ñ, accented vowels, ¿, ¡, ...) are present in the vocabulary
  from the start, just essentially untrained until the model sees them.
- **alpha = r for every LoRA run**, so the `alpha / r` scaling factor is
  always 1 regardless of rank. This isolates rank as the only variable
  being compared; letting alpha vary with rank (a common choice in
  practice) would have folded a second hyperparameter into what is meant
  to be a clean rank comparison.
- **LoRA targets the fused qkv projection and the output projection**,
  not four separate q/k/v/o matrices. The paper's own ablation (Table 5)
  compares adapting different subsets of the four projections; this
  model fuses q, k and v into one linear layer for speed, so "qkv" here
  plays the role of adapting Wq, Wk and Wv together.
- **The `enabled` flag is separate from `merge()`.** Merging physically
  edits `base.weight`, which is the right thing to do for a deployed,
  single-purpose model. `enabled` never touches the base weight at all --
  it exists specifically to make the forgetting experiment possible: proof
  that the pretrained model is sitting underneath, untouched, is only
  convincing if you can show it recovers to the *same* number by flipping
  a flag rather than reloading a checkpoint from disk.
