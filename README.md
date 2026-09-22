# Long-Context Navigation under a Hard Context Ceiling

A verifiable RL environment in which an agent answers multi-hop questions over documents far
larger than the context it is allowed to hold. It decides what to read, what to keep, what to
compress and what to discard, under a **hard context ceiling that terminates the episode if
exceeded** — a constraint to satisfy, not a cost to trade off.

Most long-context work measures whether a model can retrieve a fact from a long document. This
environment asks a different question: when the document does not fit and the agent must choose
what to keep, does it learn a *compression policy* that generalises to deeper compositions than
it was trained on — or does it learn to find one lucky chunk and guess? The baselines and the
audit exist so that the answer is trustworthy in either direction.

**Status.** Generator, validator, environment, reward, baselines, audit, teacher, SFT and GRPO
trainers, evaluation and the ARC launchers are built and tested (`pytest`: 100 tests, no GPU).
Everything model-free has been run and is reported, and so have the **LLM baselines and the
untrained base model through the environment** (Qwen2.5-7B-Instruct, one A100). **SFT has been run
and evaluated** (in-distribution 0.630, OOD 0.050 — see [Results so far](#results-so-far)); GRPO
is pending. Nothing in this file is a
placeholder; every number came from a command listed next to it.

---

## The task

An instance is a ~33k-token document (up to 200k, configurable) made of six alphabetised
**registers** — Project Register, Personnel Directory, Department Ledger, Facilities Register,
Regional Offices, Vendor Roster — plus irrelevant noise sections (minutes, logs, bulletins).
Each register entry names the next entity in a chain and carries attributes:

```
Amber Falcon — Lead: Dana Ruiz [current since 2023; previously Omar Lind from 2019 to 2023].
               Charge code: PJ-4471. Cost centre: CC-1200. Status: active. Notes: ...
```

A question with `n_hops = k` requires following `k` entries across `k` different sections:

> *What is the annual budget of the unit of the lead of project Amber Falcon?*
> project → **Lead** (person) → **Unit** (department) → **Annual budget**

The question names only the anchor and the attribute. Every intermediate ("bridge") entity must
be discovered by reading, and — because the working-context ceiling (600 tokens) is smaller
than two chunks (320–480 tokens) — must be *carried across reads as a written summary*.

Distractors, all present in every instance: **near-miss entities** (`Amber Falcon II`, sorted
next to the real one, so usually in the same chunk, with different links), **superseded
statements** (a previous lead / previous value in two different phrasings), and **near-miss
values** (every attribute value inside a chunk is clustered around one centre, so the answer
chunk holds ~6 values within a few percent of each other).

### What the generator records per instance

Exact evidence locations (chunk index and token offset) for every hop; every distractor together
with the *wrong answer a policy would reach by following it*; all candidate values of the
answer's type (defines chance); the minimum number of reads (`= n_hops`) and the minimum number of
steps under the ceiling (`min_step_schedule`: one READ per hop, one ANSWER, one COMPRESS whenever
the next chunk does not fit beside what is held — provably optimal in step count); whether a
DROP-only strategy is possible (`compression_required`); and the seed, so `longctx generate`
reproduces the file byte for byte.

## The environment

The policy is **stateless between steps**. Every step it sees exactly: the question, the document
map (one line per register chunk with its alphabetical range and token cost; noise sections
collapsed), the memory ledger (what was read, dropped, compressed, and the result of the last
action), and the working context. Nothing of its previous reasoning is carried forward. A summary
written by `COMPRESS` is therefore the only way to persist a bridge entity, which is what makes
"compression policy" a well-defined object of study rather than a no-op.

| action | effect |
|---|---|
| `READ <id>` | loads a chunk; its recorded token count is charged |
| `COMPRESS <ids> :: <summary>` | replaces held chunks/summaries with the policy's own text (truncated at `max_summary_tokens`, counted with the same tokenizer) |
| `DROP <ids>` | frees the items fully; re-reading a dropped chunk is allowed and costs a step |
| `ANSWER <value>` | terminal |

Termination: `ANSWER`, step budget exhausted (`6 + 4·n_hops`), or **any action that would push
usage above the ceiling — the episode ends as a failure**. Token accounting is exact by
construction (`used = Σ held items`) and pinned by `tests/test_env.py`.

## Reward (`reward.py`, pure, no model)

```
ceiling exceeded        -> r_fail   = -0.5
step budget exhausted   -> r_fail   = -0.5
correct                 -> max(0.5, 1 - beta · steps_used / min_steps)     beta = 0.05
wrong                   -> r_wrong  = 0.0
ANSWER unknown          -> r_abstain = 0.1   (only when reward.allow_abstain: true)
```

Correctness is normalised exact match (alphanumerics only), with one mechanical leniency for
numeric truths: `USD 4,885,866` matches `4,885,866`; `4,885,866 or 4,885,870` does not. The
efficiency term is normalised by the instance's own minimum, so deep instances are not punished
for being deep. All values live in `configs/env.yaml`.

Why not `R = accuracy − α·peak_context`: that objective has a degenerate optimum at both ends of
α and a narrow band in between where the intended behaviour appears; tuning to find it is tuning
to manufacture a result. Here compression is a constraint and there is no α.

---

## Validity: generation-time checks and their rejection rates

Every instance passes `validate_instance.validate` before it is written (`validate_instance.py`
is pure and reads the answer back **from the text** with a mechanical parser, not from generator
state):

1. every required fact is present in the chunk it claims to be in;
2. the chain is solvable from the text, along exactly the recorded chunks;
3. required facts are in distinct chunks ≥ `min_separation` (1200) tokens apart;
4. no distractor path — superseded value or near-miss entity at any hop — reaches the answer;
5. the answer value occurs exactly once as a field value in the corpus;
6. lexical shortcut: for `n_hops ≥ 2` the anchor never appears in the answer chunk and the
   answer chunk is never the question's top-overlap chunk;
7. the instance is solvable under the ceiling (every evidence chunk fits; a summary + the
   next chunk fits) and the recorded `min_steps` matches the schedule.

`longctx generate --out data/v1` (2000 / 300 / 300 / 150 instances):

| split | n_hops | doc tokens | attempts | rejected | rejection rate | reasons (an attempt may carry several) |
|---|---|---|---|---|---|---|
| train | 2, 3 | 33.3k | 2068 | 68 | **3.3%** | separation 43, distractor path reaches answer 25, near-miss path 16, superseded path 9 |
| id_test | 2, 3 | 33.3k | 315 | 15 | **4.8%** | distractor 8, separation 7, near-miss 4, superseded 4 |
| ood_test | 4, 5 | 33.3k | 356 | 56 | **15.7%** | distractor 43, near-miss 29, separation 18, superseded 14 |
| scale_test | 3 | 32k / 64k / 128k | 158 | 8 | **5.1%** | distractor 7, near-miss 4, superseded 3, separation 2 |

The rejection rate is a finding: it rises from 3% to 16% going from 2–3 to 4–5 hops because a
longer chain gives a wrong path more chances to converge on the true final entity. Those
instances are exactly the ones where a policy could be right for the wrong reason, and they
are removed rather than counted. All accepted instances have `compression_required = True`.

### Surface-confound audit (population level)

Whether chunk position, length, entry count or section header predicts evidence is a
statement about a *distribution*, not one instance: rejecting single instances where "the
answer happens to be in the longest chunk" would make *longest* anti-predictive, which is
itself a surface signal a policy could learn. So position / length / header are audited across
each split with rank AUCs and a two-fold-CV logistic regression over all gated features, and
`longctx generate` **exits non-zero** if any AUC leaves 0.5 ± 0.06. The audit is run within
the answer's own section (the only pool where a shortcut would matter, and where the header no
longer separates anything) and over all register chunks.

| pool · split | pos_global | pos_in_section | length | n_entries | overlap | logreg (gated) |
|---|---|---|---|---|---|---|
| answer chunk within its section · train (n=2000) | 0.500 | 0.495 | 0.504 | 0.497 | 0.500 | 0.491 |
| answer chunk within its section · id_test | 0.500 | 0.513 | 0.490 | 0.485 | 0.500 | 0.481 |
| answer chunk within its section · ood_test | 0.502 | 0.519 | 0.494 | 0.508 | 0.504 | 0.488 |
| answer chunk within its section · scale_test | 0.495 | 0.511 | 0.542 | 0.496 | 0.502 | 0.478 |
| evidence among all register chunks · train | 0.496 | 0.502 | 0.504 | 0.499 | **0.804** | 0.494 |

The one number above chance — question overlap → evidence, over all register chunks — is the
hop-0 chunk: it contains the anchor named in the question. That is navigation, not a confound
(it is what the question is *for*), and it is reported but not gated; overlap → *answer chunk*
is 0.50 on every split. The audit found a real confound during development: the first version
chose chain entities uniformly over entities, which made larger chunks more likely to be
evidence (n_entries AUC 0.60). The fix — group into chunks first, give every chunk the same
number of near-miss twins, then draw the evidence chunk uniformly over chunks — is what the
generator does now. `tests/test_confound.py` plants a length bias and checks the audit catches it.

---

## Baselines (before any training)

`longctx baselines --instances data/v1/id_test.jsonl data/v1/ood_test.jsonl` — mechanical mode
computes exact expectations with no model, over the 600 held-out instances:

| baseline | what it tests | accuracy | by n_hops (2 / 3 / 4 / 5) |
|---|---|---|---|
| **no-read** (fixed guess) | priors | **0.000** | 0 / 0 / 0 / 0 |
| no-read chance level (1 / candidate values in the corpus) | | 0.017 | 0.017 each |
| **single-chunk oracle** (answer chunk given; guess among its values) | single-hop shortcut | **0.183** | 0.193 / 0.184 / 0.179 / 0.176 |
| single-chunk by question overlap (does it even contain the answer?) | lexical shortcut | 0.000 | 0 / 0 / 0 / 0 |
| **full-document oracle** (mechanical chain reader, no ceiling) | solvable at all | **1.000** | 1 / 1 / 1 / 1 |

Reference policies through the environment (`longctx evaluate --policy ...`, same 600 instances):

| policy | accuracy | ID (2–3) | OOD (4–5) | steps / min | ceiling violations | COMPRESS use | fact retention |
|---|---|---|---|---|---|---|---|
| privileged oracle navigator (noise 0.15) | 1.000 | 1.000 | 1.000 | 1.19 | 0.000 | 1.00 | 1.000 |
| random reader (3 random reads, guess) | 0.003 | 0.000 | 0.007 | 0.95 | 0.000 | 0.00 | — |
| no-read | 0.000 | 0.000 | 0.000 | 0.16 | 0.000 | 0.00 | — |

The noise-free oracle solves every instance in exactly `min_steps` (`tests/test_baselines.py`),
which is the check that the environment, the recorded evidence and the per-instance minimum
agree.

### LLM baselines — Qwen2.5-7B-Instruct, untrained, greedy (`JOB=baselines`, 1×A100)

Same 600 held-out instances, two prompt regimes: *answer-only* (24 output tokens, the value
and nothing else) and *reasoning* (think step by step, then `ANSWER: <value>`, 512 tokens).

| condition | answer-only | reasoning | reasoning by n_hops (2 / 3 / 4 / 5) | abstains (reasoning) |
|---|---|---|---|---|
| **no-read** | **0.000** | **0.000** | 0 / 0 / 0 / 0 | 82% |
| **single-chunk oracle** | **0.010** | **0.000** | 0 / 0 / 0 / 0 | 98% |
| **full-document oracle** | **0.295** | **0.795** | 0.944 / 0.931 / 0.761 / 0.532 | 0% |

All three gates hold. The answer-only full-document number (0.295; 0.63 at 2 hops, ~0.2 beyond)
is what one-shot composition costs a 7B: asked for the value alone it cannot follow a 3-hop
chain through superseded values and near-miss twins in a single forward pass. Given room to
reason it can (0.795), which is the fair ceiling and the one the trained policy is measured
against. Two things to carry forward: 16% of its reasoning full-document answers are a known
distractor value, so the traps bite a model that *is* reading; and accuracy falls with depth even
with every evidence chunk in view — 4–5 hops are harder for the model itself, not only OOD for
the policy.

### The untrained model inside the environment (`JOB=eval`, 1×A100)

| policy | accuracy | ceiling violations | budget exhausted | reads / min reads | COMPRESS use | fact retention |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct, no training | **0.000** | 0.570 | 0.040 | 0.36 | 0.14 | 0.868 |

It reached the answer chunk in **1 of 600** episodes. It reads the project entry and stops: 159
of its 234 answers are the lead's *name* (the hop-0 bridge), the rest the project's own code or
"not provided"; the other 57% of episodes die reading a second chunk without compressing first.
The first run of this eval scored the same 0.000 for a different reason — 7,006 of 8,077 actions
were rejected by a parser that refused `READ 20 :: <copied map line>`. That was interface
strictness, not navigation failure, and the parser now ignores text after the ids
(`tests/test_actions.py`). The number above is from the re-run. This is the "before" row: the
model can operate the interface but does not follow the chain, does not compress on purpose, and
never composes — exactly what the teacher trajectories demonstrate.

---

## Reward-hacking audit

`longctx audit` runs five mechanical probes over any trajectory file. On the reference
policies above (600 episodes each):

| probe | oracle navigator | random reader |
|---|---|---|
| 1. guess without reading — accuracy given zero `READ`s | 0 zero-read episodes | 0 zero-read episodes |
| 2. position bias — accuracy spread across answer-position quintiles | 0.000 | 0.009 |
| 3. compression as no-op — do summaries retain the required fact? | 1763 COMPRESS actions; retention 1.000; degenerate 0.000; 11.8 tokens/summary | never compresses |
| 4. drop-and-reread loops — episodes with a re-read; reads / min reads | 0.013; 1.19 | 0.000; 0.95 |
| 5. distractor capture — wrong answers matching a known distractor | no wrong answers | 2.3% of 598 wrong answers (9 near-miss value, 5 near-miss entity) |

Probe 3 is the one that decides the research question for a trained policy: a policy that
compresses to free tokens without writing the bridge entity down will score high on `COMPRESS`
use and low on fact retention, and its accuracy will sit near the single-chunk baseline.

---

## Cold start

`longctx teacher --instances data/v1/train.jsonl --out data/v1/sft.jsonl`: two trajectories per
training instance from the oracle navigator (the second with 15% detour noise), kept only if
correct, within 1.5× `min_steps`, and **leak-free** — every name and value the teacher wrote
appears in that step's observation, and every read of an evidence chunk happened while its key
entity was visible in the question or the working context (the map's range labels do not
count). Privilege chooses among justifiable actions; it never reveals anything.

| sampled | rejected: incorrect | too long | leak | kept | keep rate | SFT rows (one per step) |
|---|---|---|---|---|---|---|
| 4000 | 0 | 168 | 0 | 3832 | 95.8% | 20 068 |

Because the policy is stateless, each step is an independent (prompt, completion) pair: SFT is
plain `transformers.Trainer` + LoRA with the loss on the assistant tokens, no multi-turn masking.

## Training

* **SFT** (`train_sft.py`): Qwen2.5-7B-Instruct + LoRA r=32 on the rows above.
* **GRPO** (`train_grpo.py`): a compact loop rather than TRL's `GRPOTrainer`. One episode is
  *several* independent (prompt, completion) pairs sharing one return; GRPOTrainer's contract is
  one completion per fixed prompt, and packing a multi-step episode into that shape would
  recompute logprobs under a context the policy never conditioned on. The loop: B instances ×
  G episodes per rank (lockstep-batched HF generate, per-step token ids recorded);
  `A = (R − mean_group) / std_group` inherited by every step of the episode; clipped surrogate
  + `β·KL` to the adapter-disabled reference; DDP or DeepSpeed ZeRO-2 through `accelerate`.
  Only `n_hops ≤ train_max_hops = 3` is ever trained on. Curriculum (`--curriculum`) starts at
  2 hops and escalates when the windowed success rate passes a threshold.
* `vLLM` serves evaluation and the LLM baselines (LoRA adapters loaded through vLLM's LoRA
  support); GRPO rollouts use HF generate in-process so the update sees the same weights.

## Results so far

### After SFT (`JOB=eval ADAPTER=checkpoints/v1/sft-… TAG=sft`, 600 held-out instances)

SFT: LoRA r=32 on the 20 068 teacher rows, 2 epochs, effective batch 64, 4×A100, 2.8 h; loss
2.22 → 0.016.

| policy | ID (2–3 hops) | OOD (4–5 hops) | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| untrained base, in environment | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **SFT** | **0.630** | **0.050** | 0.730 | 0.557 | 0.082 | 0.014 |
| single-chunk baseline (mechanical) | 0.187 | 0.178 | 0.193 | 0.184 | 0.179 | 0.176 |
| full-document, reasoning (no ceiling) | 0.936 | 0.653 | 0.944 | 0.931 | 0.761 | 0.532 |

| SFT policy | ID | OOD |
|---|---|---|
| ceiling violations | 0.000 | 0.017 |
| steps / instance minimum | 1.88 | 2.39 |
| COMPRESS used (episodes) | 0.98 | 1.00 |
| fact retention in summaries | 0.982 | 0.599 |

Audit (all 600): zero-read episodes 0; accuracy spread across answer-position quintiles 0.047;
3654 `COMPRESS` actions, 0.2% degenerate, 9.5 tokens each; **59% of episodes re-read a chunk
they had already compressed**, reads / min reads 2.12; 52% of episodes exhausted the step
budget; of 78 wrong answers 15% are a known distractor (5 near-miss value, 4 near-miss entity,
3 superseded link).

Reading: SFT learned the hard constraint outright (ceiling violations 0 in distribution) and
the compression routine — but as a routine for the lengths it was shown. At 4–5 hops the
summaries stop carrying every bridge (retention 0.98 → 0.60), the policy goes back to re-read
what it lost, and runs out of steps. Against the ceiling: the same model with every evidence
chunk in view reaches 0.93 at 3 hops; navigating under 600 tokens it reaches 0.56. The
composition-generalisation question is now whether GRPO — trained only on ≤3 hops — closes
either gap.

### GRPO, first full run (LoRA lr 1e-5): the policy did not move

Held-out evaluation of adapter snapshots against the SFT policy they started
from (600 instances, sampled decoding, same audit):

| policy | ID (2–3) | OOD (4–5) | ceiling viol. ID / OOD | retention ID / OOD | re-read episodes |
|---|---|---|---|---|---|
| SFT (= GRPO step 0) | 0.630 | 0.050 | 0.000 / 0.017 | 0.982 / 0.599 | 0.588 |
| GRPO step 20 | 0.653 | 0.077 | 0.017 / 0.060 | 0.973 / 0.651 | 0.585 |
| GRPO step 40 | 0.600 | 0.053 | 0.060 / 0.193 | 0.968 / 0.736 | 0.415 |
| GRPO step 60 | 0.623 | 0.060 | 0.017 / 0.100 | 0.965 / 0.627 | 0.512 |

Every accuracy difference is inside ~2 standard errors (0.028 ID, 0.014 OOD at
n=300) and non-monotone, so this run changed nothing measurable — as its
training log predicted: `kl` reached only 0.007 in 65 steps, because splitting
each episode's loss across its ~6 steps (the length-bias fix, below) cut the
gradient about 6× and the learning rate was not raised with it. The run is kept
as a **control**: it confirms the evaluation path tracks the adapter, and it is
the baseline the higher-rate run is compared against.

Two things the run did establish. The length-bias fix holds: over 65 training
steps ceiling violations stayed at 0.01–0.05 and `COMPRESS` usage at ~1.0,
where the previous objective drove them to 0.20 and 0.85. And OOD ceiling
violations rose above the SFT policy's 0.017 at every snapshot — training on
≤3 hops may cost ceiling discipline at 4–5, which is worth watching once the
policy actually moves.

## Evaluation

`longctx evaluate` reports, on held-out generated instances: accuracy by `n_hops`; in-distribution
(2–3) vs OOD (4–5); accuracy by document size (`scale_test`: 32k / 64k / 128k); steps used /
per-instance minimum; `COMPRESS` utilisation and fact retention; ceiling-violation rate — with the
three baselines alongside. The OOD split is the composition-generalisation test: a collapse there
means the policy learned patterns rather than composition, and that would be a legitimate finding.

---

## Running locally (no GPU)

```bash
pip install -e ".[test]"
pytest                                                     # 100 tests
longctx generate --out data/v1                             # ~2 min; exits 1 on a confound
longctx baselines --instances data/v1/id_test.jsonl data/v1/ood_test.jsonl
longctx teacher  --instances data/v1/train.jsonl --out data/v1/sft.jsonl
longctx evaluate --policy oracle --noise 0.15 --instances data/v1/id_test.jsonl data/v1/ood_test.jsonl --out results/eval_oracle
longctx audit    --instances data/v1/id_test.jsonl data/v1/ood_test.jsonl --trajectories results/eval_oracle.trajectories.jsonl
longctx grpo --dry-run --instances data/v1/train.jsonl     # wiring check with the mechanical oracle
```

## Running on ARC

Cluster facts and traps live in [`arc_quickref.md`](arc_quickref.md) and
[`arc_runbook.md`](arc_runbook.md); the launchers carry those lessons (absolute `CONDA_ENV`,
`$PY` never bare `python`, explicit `--mem`, `*_short` QOS, NCCL interface probing, mail on
`END,FAIL`). One conda env suffices here — nothing depends on TRL.

```bash
# once
bash scripts/arc_setup_env.sh                              # -> ~/miniconda3/envs/lcn
HF_HOME=/home/$USER/hf_cache ~/miniconda3/envs/lcn/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
longctx generate --out data/v1 && longctx teacher --instances data/v1/train.jsonl --out data/v1/sft.jsonl

# LLM baselines FIRST (1 GPU)
HF_HOME=/home/$USER/hf_cache CONDA_ENV=/home/$USER/miniconda3/envs/lcn JOB=baselines \
sbatch --account=ece-6524-spring2026 --partition=a100_normal_q --qos=tc_a100_normal_short \
       --gres=gpu:a100:1 --mem=96G --time=02:00:00 --mail-user=$USER@vt.edu scripts/arc_infer.slurm
JOB=eval ... scripts/arc_infer.slurm                       # base model through the environment

# then the chain: sft -> grpo -> eval(sft) / eval(grpo) / scale(grpo)
SBATCH_ACCOUNT=ece-6524-spring2026 MAIL_USER=$USER@vt.edu BASELINES_OK=1 DRY_RUN=1 ./scripts/train_all.sh
SBATCH_ACCOUNT=ece-6524-spring2026 MAIL_USER=$USER@vt.edu BASELINES_OK=1 ./scripts/train_all.sh
```

Calibrate GRPO wall-clock with `... scripts/arc_grpo.slurm --max-steps 5` before a long
allocation; the launcher saves the adapter every `save_every` steps and resumes with `RESUME=`.

## What went wrong along the way

Every failed run, its actual cause, the fix and the transferable rule is in
[`docs/LESSONS.md`](docs/LESSONS.md) — nine incidents from a JIT-compiled sampler
with no compiler on the node to a GRPO objective that rewarded short episodes.

## Repository

```
configs/        env.yaml (ceiling, budget, reward) · generator.yaml (hops, sizes, distractors, splits) · train.yaml · deepspeed_*.json
src/longctx/
  corpus.py             entity types, entry grammar, parser (generator writes it, validator reads it)
  generate.py           task generator + per-instance minimum step computation
  validate_instance.py  PURE: solvability, separation, distractor paths, lexical shortcut, confound audit
  env.py / actions.py / prompts.py   observation, action grammar, ceiling enforcement, termination
  reward.py             PURE: mechanical scoring
  policies.py           oracle navigator (teacher), random reader, no-read
  rollout.py            lockstep batched episodes -> trajectories
  baselines.py          no-read / single-chunk / full-document, mechanical and LLM
  audit.py              the five reward-hacking probes
  teacher.py            rejection sampling + leak filter -> SFT rows
  llm.py / common.py    HF + vLLM backends, LoRA/dist plumbing
  train_sft.py / train_grpo.py / evaluate.py / cli.py
tests/          100 tests, no GPU: reward branches, action grammar, exact token accounting,
                generator invariants, confound audit (incl. a planted bias), baselines, audit, leak filter
scripts/        arc_env.sh · arc_infer.slurm · arc_sft.slurm · arc_grpo.slurm · train_all.sh · arc_setup_env.sh
results/        baselines_mechanical.* · eval_{oracle,random,no-read}.* · audit_*.* · generation_report_v1.json · teacher_stats_v1.json
```

Dependencies: `numpy`, `pydantic`, `pyyaml` (+ `pytest`); training adds `torch`, `transformers`,
`peft`, `accelerate`, `deepspeed`, `vllm`, `wandb`.
