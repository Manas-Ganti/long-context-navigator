# Lessons from the failed runs

Every run here that went wrong, what it actually was, the fix, and the general
rule to carry to the next project. Ordered as they happened. Each one cost at
least a GPU allocation; several looked like a *model* problem and were not.

Format for each: **Symptom** (what the log said) → **What it actually was** →
**Fix** → **How to spot it elsewhere**.

---

## 0. The generator marked evidence chunks by their size (caught locally, before any GPU)

**Symptom.** The surface-confound audit on the first generator reported
`n_entries → evidence  AUC 0.60`: chunks with more entries were more likely to
hold a required fact. Everything else was at 0.50.

**What it actually was.** Size-biased sampling. I picked the chain's entities
uniformly *over entities*, then asked which chunk each one landed in. A chunk
with 7 entries is 7/4 times as likely to contain a uniformly chosen entity as
a chunk with 4 — so "big chunk" quietly meant "probably evidence". A policy
could learn to read big chunks first and be right more often without
understanding anything.

**Fix.** Choose the *chunk* uniformly first, then an entity inside it. Also
give every chunk the same number of near-miss twins, so "has a twin pair"
marks nothing either.

**How to spot it elsewhere.** Whenever you pick "a random X" and then look at
a property of the container X lives in, the container's size is now
correlated with being picked. Audit every group-level feature (length, count,
position, header) against the label with a rank AUC on a few hundred
instances; anything off 0.50 is a shortcut waiting to be learned. And plant a
deliberate bias once to prove the audit can see it (`tests/test_confound.py`).

---

## 1. vLLM died after loading the model: "Could not find nvcc"

**Symptom.** Both first GPU jobs loaded 15 GB of weights, captured CUDA
graphs, then aborted in warm-up:
`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`.

**What it actually was.** This vLLM build routes top-k/top-p sampling through
FlashInfer, which compiles its kernel *on first use* and needs the CUDA
compiler. The compute nodes have the CUDA driver but no toolkit. Nothing
checks for this at import time; it fails after the expensive part.

**Fix.** `VLLM_USE_FLASHINFER_SAMPLER=0` in `arc_env.sh` (falls back to the
PyTorch sampler). Also probe for a CUDA module and export `CUDA_HOME` if one
exists.

**How to spot it elsewhere.** Anything that "JIT-compiles" (FlashInfer,
DeepSpeed fused ops, Triton kernels, `torch.compile` with custom ops) needs a
compiler on the *compute* node, and it will fail late. Run
`command -v nvcc` inside a job before trusting a new inference/training stack,
and prefer the pure-PyTorch code path or prebuilt wheels on clusters that
hide the toolkit.

---

## 2. The "full-document" baseline said the task was too hard (0.295)

**Symptom.** The three pre-training baselines came back no-read 0.000,
single-chunk 0.010, full-document **0.295** — and the spec requires the
full-document number to be *high*, or the environment is invalid.

**What it actually was.** My baseline prompt handicapped the model: "answer
with the value only", 24 output tokens. A 7B asked to follow a 3–5-step chain
past superseded values and near-duplicate names *in one forward pass with no
reasoning* is being tested on something harder than the environment ever asks
of it (one lookup per step, a THOUGHT line each time). The mechanical reader
scored 1.000, so the task was fine; the ceiling measurement was wrong.

**Fix.** A second regime that lets the model reason and then write
`ANSWER: <value>` (512 tokens). Full-document went to **0.795**. Both numbers
are reported.

**How to spot it elsewhere.** A ceiling/oracle baseline must give the model at
least the affordances the policy gets (reasoning tokens, the same format,
the same instructions). If the "upper bound" is lower than what you later see
from the trained policy, the upper bound was measured under a handicap.

---

## 3. The untrained model scored 0 in the environment — because my parser rejected 87% of its actions

**Symptom.** Base model through the environment: accuracy 0.000, 45% of
episodes never issued a `READ`, 50% ran out of steps.

**What it actually was.** `7006 of 8077` actions were parse errors. The model
wrote perfectly clear commands with a copied map line after the id —
`READ 20 :: Project Register · Coral Curlew – …`, `READ [35] Personnel …` —
and my parser refused anything after the id. The score was 0 for a reason
that had nothing to do with navigation.

**Fix.** Parse the leading ids and ignore trailing text (`actions.py`). Raise
the generation cap 160 → 256 so long THOUGHT lines don't truncate the ACTION
line. Re-run: zero-read episodes 45% → 1.5%, budget exhaustion 50% → 4%.

**How to spot it elsewhere.** Before concluding "the model can't do X", count
the *interface* failures: parse errors, truncations, rejected formats. Print
twenty raw responses. An accuracy of exactly 0.000 is almost always the
harness, not the model.

---

## 4. After the parser fix, still 0.000 — and this time it was real

**Symptom.** Same eval after the fix: accuracy 0.000, ceiling violations 57%,
234 wrong answers.

**What it actually was.** A capability gap, not a bug — and provable
mechanically: the model reached the answer chunk in **1 of 600** episodes.
It reads the first entry and answers with what it sees (159 of 234 answers
were the lead's *name*, i.e. the hop-0 bridge). It does not follow the chain,
and it reads a second chunk without compressing first.

**Fix.** None needed — this is the "before training" row. It is what the
teacher trajectories demonstrate (read → write the bridge into a summary →
read the next register → … → answer the *attribute*).

**How to spot it elsewhere.** When a score is 0, ask a mechanical question
that separates "can't operate the interface" from "can't do the task": did
the policy ever *reach* the information it needed? Here that is one line over
the trajectory file. Only after that do you decide whether it's the harness,
the model, or the data.

---

## 5. SFT: strong in distribution, collapse out of distribution (a finding, not a bug)

**Symptom.** SFT policy: 2–3 hops **0.630**, 4–5 hops **0.050**. Ceiling
violations 0. Compression used in 98% of episodes.

**What it actually was.** The audit explains it: fact retention in summaries
0.98 in distribution → 0.60 at 4–5 hops; 59% of episodes re-read a chunk they
had already compressed (the summary lost the name, so they went back); 52%
ran out of steps in that loop. SFT taught a compression *routine* for the
lengths it was shown; at more hops the routine drops facts.

**Fix.** Nothing to fix in the pipeline — it's the result the OOD split exists
to measure, and the thing GRPO is then tested on. Recorded in the README.

**How to spot it elsewhere.** Don't stop at the accuracy split. Have
mechanical probes that say *how* the OOD failures happen (here: retention,
re-reads, budget). A collapse with a mechanism is a finding; a collapse
without one is just a number.

---

## 6. GRPO: KL was 2.0 at step 0 — the policy was being pulled toward the *base* model

**Symptom.** First calibration run: `kl=2.101` on step 0, ~2.0 every step.
KL should be ≈0 on step 0 because the policy *is* the reference.

**What it actually was.** The "reference" was the model with the LoRA adapter
disabled — but I was training the SFT adapter itself, so "adapter disabled"
meant the *untrained base*. KL(SFT ‖ base) is large by design. With
`kl_beta = 0.04` that is a constant per-token push back toward forgetting
SFT, on every step, for the whole run.

**Fix.** Merge the SFT adapter into the weights once, save that model, and
train a *fresh* LoRA on top. Now "adapter disabled" = SFT policy, and step 0
reads `kl=0.000`.

**How to spot it elsewhere.** Know exactly what tensor your KL/reference term
compares against, and check KL on the first step: if it isn't ~0, the
reference isn't the policy you started from. With PEFT, "disable adapter"
means the base *underneath the adapter* — which is only your SFT model if SFT
was merged in.

---

## 7. GRPO: KL stayed at 0.000 for ten steps — the policy wasn't moving

**Symptom.** After the merge fix: `kl=0.000` at steps 0–9, accuracy noisy but
flat.

**What it actually was.** A fresh LoRA starts at exactly zero (its B matrix is
zero-initialised). With Adam each weight moves about one learning rate per
step, and the rate was `1e-6` — a full-fine-tune rate. 150 steps at 1e-6 is
~40× less total movement than the SFT run that produced the policy (630 steps
at 1e-5). The run would have ended as "SFT with noise" and told us nothing.

**Fix.** LoRA learning rate `1e-5`. KL lifted off within ~10 steps and sat
around 0.01–0.1 by step 60.

**How to spot it elsewhere.** For a fresh LoRA, watch KL (or any
distance-from-start measure) over the first 10–20 steps. Exactly 0.000 means
the learning rate is too small to matter for the adapter's parameter scale.
Rule of thumb: LoRA rates are ~10× full-fine-tune rates.

---

## 8. GRPO crashed with SIGABRT — and the 4 GPUs had never been sharing gradients

**Symptom.** Rank 3 died with `SIGABRT` (NCCL watchdog); the others were
SIGTERM'd. Earlier steps looked fine.

**What it actually was.** Two things.
(a) I ran the forward on the *unwrapped* model, not through the DDP wrapper,
so DDP's gradient all-reduce never fired. Four ranks trained four independent
adapters on a quarter of the data each; only rank 0's was saved. Three GPUs
were wasted, silently.
(b) The only collective was a save barrier every 10 steps. Ranks draw
different instances, so their steps take 3–10 min each; over 10 steps they
drifted more than the 30-minute NCCL timeout apart, and the rank waiting at
the barrier aborted.

**Fix.** Don't use DDP at all (it needs equal micro-batch counts per rank,
and ours differ every step because episode length is the policy's choice).
Average the LoRA gradients across ranks by hand each step — one flat
all-reduce — and seed the LoRA init identically on all ranks so averaging is
meaningful. That is also a per-step sync, so drift is bounded by one step.

**How to spot it elsewhere.** Verify gradient sync directly: after one step,
the trainable weights must be bit-identical on every rank. If your ranks do
variable amounts of work, DDP's "every backward is an all-reduce" assumption
is broken; do the reduction explicitly. And any collective that ranks reach
at very different times (barriers, saves) needs a timeout longer than the
worst-case drift.

---

## 9. GRPO made the policy faster and worse — the loss was averaged over steps, not episodes

**Symptom.** Over 60 steps: accuracy 0.58 → 0.37, ceiling violations
0.02 → 0.20, COMPRESS use 1.0 → 0.85, episodes shorter, budget exhaustion
→ 0. KL only 0.1, so the policy hadn't run away — it was being *steered*
somewhere bad.

**What it actually was.** Length bias in the objective. Every step of an
episode is one training sample and all share the episode's advantage; I
averaged the loss over *samples*. So an 18-step budget-exhausted failure
contributed 18 "do less of this" samples, a 2-step ceiling violation only 2,
a correct 6-step episode 6 "do more". Summed over thousands of episodes the
dominant message was "long episodes are punished hardest, short ones barely"
— and the policy obliged: stop compressing, read into the ceiling, answer
early. Not reward hacking (a ceiling violation is the worst reward); a
mis-weighted gradient.

**Fix.** Weight each step by `1 / (steps in its episode)` and divide by the
number of episodes: every episode gets exactly one vote, split across its
moves. That is GRPO's own sequence-level mean; I had written the
token-level shortcut.

**How to spot it elsewhere.** Ask what the *unit of averaging* in your loss
is. If samples come in variable-length groups (tokens per sequence, steps per
episode, turns per dialogue) and you average over the leaves, long groups
dominate. Check the tell-tale drift: the policy changes the *length* of what
it produces before it changes the *quality*. Length bias in RL objectives is
a known trap (see the DAPO/Dr. GRPO discussions); the safe default is one
weight per group.

---

## 11. Judging a run by its training curve instead of held-out evaluation

**Symptom.** Rollout accuracy in the GRPO log swung between 0.27 and 0.85
step to step. It was tempting to read decade averages (0.66 → 0.67 → 0.60 →
0.64) as a trend.

**What it actually was.** Nothing — noise. Rollout accuracy is 128 sampled
episodes per step at temperature 0.8, drawn from fresh random *training*
instances (≤3 hops). Binomial noise alone is ±0.04, instance-difficulty
variance adds more, and the quantity of interest (4–5-hop generalisation) is
not in that number at all, because the policy never trains on those.

**Fix.** Save an adapter snapshot every N steps and evaluate snapshots on the
fixed held-out set. Three 1-GPU evals gave a straight answer (flat ID, flat
OOD) where 65 steps of training curve gave an argument. Steer during a run by
the *mechanism* metrics instead — ceiling violations, compression usage, fact
retention, invalid actions — which sit near their bounds and move far outside
noise when something is wrong.

**How to spot it elsewhere.** Ask of any training-curve number: how many
samples, on which distribution, and is it the quantity the project is about?
If the answer is "few, train, no", it is a health check, not a result. Budget
for snapshot evaluation from the start — it is cheap next to the training run
and it is the only thing that answers the question.

---

## The pattern across all of them

- **Zero or suspiciously round numbers are harness problems until proven
  otherwise** (3, 4). Look at raw outputs and count interface failures
  before diagnosing the model.
- **Check the invariant on step 0** (6, 7). KL ≈ 0, weights identical across
  ranks, the first loss finite. Every one of these is a five-second check and
  each saved or would have saved a 20-hour run.
- **Late failures are the expensive ones** (1, 8). Anything that compiles,
  connects or synchronises lazily fails after the weights are loaded; probe
  for it early.
- **The direction of drift names the bug** (9). Shorter before better →
  length bias. Toward the base model → wrong reference. No drift at all →
  learning rate.
- **Baselines can be wrong in both directions** (2). An upper bound measured
  under a handicap is not an upper bound.
- **When it's a real capability gap, prove it mechanically** (4, 5) — "reached
  the answer chunk in 1/600", "retention 0.98 → 0.60" — and then it's a
  finding, not a bug.
