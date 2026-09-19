# CLAUDE.md

## Project: Long-Context Navigation Environment (context management under a hard ceiling)

A verifiable RL environment in which an agent answers multi-hop questions over documents far
larger than the context it is allowed to hold. It must decide what to read, what to keep, what
to compress, and what to discard — under a **hard context ceiling that terminates the episode
if exceeded**, not a soft penalty it can trade against.

The research question: does an agent trained under a context constraint learn a genuine
*compression policy*, or does it learn to find one lucky chunk and guess?

---

## HOW TO WORK ON THIS PROJECT (read first)

- **Scaffold the whole codebase in this session**: environment, task generator, verifier,
  baselines, eval harness, training scripts. Runnable end to end.
- Build in this order: task generator -> solvability checks -> baselines -> environment ->
  reward -> training. **Baselines before training.** If the baselines are built afterwards
  they get tuned until the trained policy looks good.
- **Never report a number that did not come from a real run.** No placeholder metrics in
  the README.

---

## THE INVIOLABLE RULES

**1. The task must be unsolvable without multi-hop retrieval, and this is proven, not assumed.**

Before any training, three baselines must be run. The environment is only valid if all three
fail:

| baseline | what it tests | must score |
|---|---|---|
| **no-read** (answer from the question alone) | priors / guessability | at chance |
| **single-chunk** (best single chunk in context, oracle-selected) | single-hop shortcut | well below ceiling |
| **full-document oracle** (all evidence, no ceiling) | task is solvable at all | high |

If the single-chunk baseline scores well, the task does not require navigation and the
environment is broken. Fix the generator, do not proceed to training. This is the same
ceiling/floor separation discipline as the flagship's 0.930 / 0.591 probe.

**2. A hard ceiling, not a soft penalty.**

Do NOT use `R = accuracy - alpha * peak_context`. That formulation has a degenerate optimum:
with a large alpha the policy reads nothing and guesses; with a small alpha it reads everything
and the compression actions are never exercised. There is a narrow alpha band where the
intended behaviour appears, and tuning to find it is tuning to manufacture a result.

Instead: **exceeding the context ceiling terminates the episode with the failure reward.**
Compression becomes a constraint to satisfy rather than a cost to trade off, and the alpha
tuning problem disappears entirely.

---

## Task Generation (build this first)

Tasks are **generated, not hand-written**, so their properties are known by construction.

A task is a document corpus plus a question whose answer requires composing facts from
`k` separate locations, where `k` is a generator parameter.

Generator parameters:
- `n_hops` — how many facts must be combined (this is the difficulty dial)
- `doc_tokens` — total corpus size (target 32k–200k+)
- `min_separation` — minimum token distance between required facts, set so that **no two
  required facts can co-occur in one context window**. This is what makes compression
  necessary rather than optional.
- `n_distractors` — plausible-but-wrong facts, including near-miss values and superseded
  earlier statements
- `noise_ratio` — irrelevant filler

**The generator must know and record, per instance:**
- the exact locations of all required facts
- the minimum number of `READ_CHUNK` calls needed (the **efficiency ceiling**)
- the ground-truth answer

**Generator validity checks (automatic, run on every instance):**
- all required facts are present and mutually reachable
- required facts are separated by at least `min_separation`
- no distractor produces the correct answer by a plausible wrong path
- **no surface feature predicts the answer** — run a trivial classifier on chunk position,
  chunk length, lexical overlap with the question, and section headers. If any predicts the
  answer above chance, the instance is rejected and regenerated.

That last check is the confound audit moved to generation time. Document the rejection rate in
the README — it is a finding in itself.

Source material: synthetic corpora with controlled structure. Optionally validate the trained
policy on a real long-context benchmark afterwards, but train on generated data so ground
truth is exact.

---

## Environment

### Observation
- the question
- the **working context**: chunks currently held, plus any compressed summaries
- a memory ledger: what has been read, what was dropped, current token usage against ceiling
- a document map (section headers / chunk indices), so navigation is informed rather than blind

### Action space
| action | effect |
|---|---|
| `READ_CHUNK(idx)` | loads chunk `idx` into working context |
| `COMPRESS(idx_list)` | replaces held chunks with a model-written summary; frees tokens, loses detail irreversibly |
| `DROP(idx_list)` | removes chunks from working context; frees tokens fully |
| `ANSWER(response)` | terminal |

Reading a dropped chunk again is allowed but costs a step — so `DROP` is a genuine bet, not a
free action.

### Termination
- `ANSWER` emitted, or
- step budget exhausted, or
- **context ceiling exceeded -> episode fails**

---

## Reward (verifiable, mechanical, no LLM judge)

```
if context_ceiling_exceeded:   R = R_fail
elif step_budget_exhausted:    R = R_fail
elif answer_correct:           R = 1.0 - beta * (steps_used / min_steps_for_instance)
else:                          R = 0.0    (or small negative)
```

Notes:
- Correctness is checked by **exact match / normalised match against generated ground truth**.
  No judge model anywhere in the reward path.
- The efficiency term is normalised **per instance** by that instance's known minimum step
  count, so hard instances are not punished for being hard. This is only possible because the
  generator computes the minimum.
- `beta` is small. Correctness dominates; efficiency is a tiebreaker, not a competing objective.
- Optionally support abstention (`ANSWER("unknown")`) scored above a confident wrong answer, to
  connect to calibration. Keep this behind a config flag and report both settings.

---

## Reward-Hacking Audit (required, not optional)

The environment ships with an explicit audit suite. Each of these is a hypothesis about how the
policy might be cheating; each has a mechanical test:

1. **Guess-without-reading** — measure answer accuracy conditioned on zero `READ_CHUNK` calls
2. **Position bias** — accuracy by required-fact position; a policy exploiting "answers are
   usually early" shows up here
3. **Compression as no-op** — check whether `COMPRESS` outputs retain the required facts or are
   degenerate strings that merely free tokens
4. **Drop-and-reread looping** — detect cycles that game a per-step signal
5. **Distractor capture** — rate at which the final answer matches a known distractor

Findings go in the README whether or not they are flattering. The audit existing and being
honest is more valuable than a high score.

---

## Training Pipeline

- **Cold start**: rejection-sampled trajectories from a privileged teacher that can see fact
  locations, filtered for leak-freedom, then SFT.
- **GRPO** with LoRA on a 7B-class model (Qwen2.5-7B or similar).
- vLLM for rollout generation; DeepSpeed ZeRO for training; SLURM job chaining.
- Reuse the scaffolding from the existing RLVR environments — this project should not rebuild
  training infrastructure from scratch.

**Curriculum (optional, flag-gated):** start at `n_hops=2`, escalate as success rate passes a
threshold. Because difficulty is a generator parameter, this is nearly free.

---

## Evaluation

Report, on **held-out generated instances**:
- accuracy by `n_hops` (the composition-generalisation curve)
- **train on `n_hops <= 3`, evaluate at 4 and 5** — the OOD split. A collapse here means the
  policy learned patterns rather than composition, and that is a legitimate finding.
- accuracy by `doc_tokens`, to show scaling behaviour
- **step efficiency**: steps used / instance minimum
- compression utilisation: how often `COMPRESS` is used, and whether it preserves required facts
- ceiling-violation rate (how often the policy fails by overrunning context)
- all three baselines alongside the trained policy, always

---

## Repository Structure

```
longctx-nav/
|- README.md                    # design, BASELINE TABLE, real results, audit findings
|- pyproject.toml
|- configs/
|  |- env.yaml                  # ceiling, step budget, beta, chunk size
|  |- generator.yaml            # n_hops, doc_tokens, min_separation, distractors
|  |- train.yaml
|- src/longctx/
|  |- generate.py               # task generator + per-instance minimum step computation
|  |- validate_instance.py      # PURE. solvability + separation + surface-confound checks
|  |- env.py                    # observation, actions, ceiling enforcement, termination
|  |- reward.py                 # PURE. mechanical scoring. no LLM.
|  |- baselines.py              # no-read, single-chunk, full-document oracle
|  |- audit.py                  # the five reward-hacking probes
|  |- train_sft.py / train_grpo.py
|  |- evaluate.py
|  |- cli.py
|- tests/
|  |- test_reward.py            # every branch; ceiling violation scores as failure
|  |- test_generator.py         # separation enforced; unsolvable instances rejected
|  |- test_confound.py          # surface-feature classifier stays at chance on accepted set
|  |- test_env.py               # ceiling accounting exact; DROP/COMPRESS token math correct
|  |- test_baselines.py         # oracle baseline achieves high score on generated set
```

---

## Conventions

- `reward.py` and `validate_instance.py` are **pure and fully unit-tested**, with no model
  dependency. These are the modules a reviewer will read first.
- Token accounting must be exact and tested. If the ceiling can be miscounted, the central
  constraint is meaningless.
- Everything seeded; generator seed recorded with every instance so any result is reproducible.
- Config-driven; no magic numbers.
- Ask before adding dependencies beyond: `torch`, `transformers`, `trl`, `peft`, `vllm`,
  `deepspeed`, `pydantic`, `pyyaml`, `numpy`, `pytest`.

---

## Definition of Done

1. `generate` produces a validated task set, with the confound-rejection rate reported.
2. `baselines` shows no-read at chance, single-chunk low, full-document oracle high — **printed
   in the README before any trained result**.
3. `train` runs the SFT -> GRPO pipeline on the cluster.
4. `evaluate` produces the accuracy-by-hops curve, the in-distribution vs OOD comparison, step
   efficiency against per-instance minimums, and ceiling-violation rate.
5. `audit` runs all five reward-hacking probes, with findings in the README.
6. `pytest` passes without a GPU.

---

## Framing (keep this in the README)

Most long-context work measures whether a model can retrieve a fact from a long document. This
environment asks a different question: when the document does not fit and the agent must choose
what to keep, does it learn a compression policy that generalises to deeper compositions than
it was trained on? The baselines and the audit exist so that the answer is trustworthy in
either direction.