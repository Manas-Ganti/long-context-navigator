# The whole system, and where scope can be cut

Written to make a scoping decision from facts rather than memory. Costs are
measured from runs that actually happened on VT ARC (A100-80).

---

## 1. The environment

One episode = one question over one document the agent cannot hold.

**What the policy sees, every step** (`prompts.py`, ~1.5–3k tokens):

| part | content |
|---|---|
| question | the multi-hop question; names only the anchor and the attribute |
| context meter | `used / 600 tokens`, and that exceeding it ends the episode |
| step meter | steps used of `6 + 4·n_hops` |
| document map | one line per chunk: id, header, token cost (~85 lines for a 32k doc) |
| memory ledger | chunks read, dropped, compressed; the last action's result |
| working context | the chunks and summaries currently held, verbatim |

**The policy is stateless between steps.** No reasoning carries over. The only
way to move a fact from step *t* to step *t+1* is to write it into a
`COMPRESS` summary. This is the design decision the whole project rests on: it
makes "does it learn a compression policy" a question about an artefact the
environment can read, not an inference about hidden state.

**Actions** (`actions.py` — parser tolerates trailing text, repeated ids, a
`THOUGHT:` line first):

| action | effect |
|---|---|
| `READ <id>` | load a chunk; charged its exact recorded token count |
| `COMPRESS <ids> :: <text>` | replace held items with the policy's own words, truncated at 120 tokens and counted with the same tokenizer |
| `DROP <ids>` | free items entirely; re-reading later is allowed and costs a step |
| `ANSWER <value>` | terminal |

**Termination:** `ANSWER`; or step budget exhausted; or **any action that would
push usage above 600 tokens** — checked *before* the action applies, so the
offending chunk never loads.

**Accounting:** `used == sum(tokens of held items)` at every step, asserted by
tests. Chunks are 320–480 tokens, so two never fit; a maximal summary plus the
largest chunk does.

---

## 2. The reward (`reward.py` — pure, no model, 88 lines)

```
ceiling exceeded       -> -0.5
step budget exhausted  -> -0.5
no answer emitted      -> -0.5
correct                -> max(0.5, 1 - 0.05 · steps_used / min_steps)
wrong                  -> 0.0
ANSWER unknown         ->  0.1     (only when reward.allow_abstain)
```

* **Correctness** is normalised exact match (alphanumerics only) against
  generated ground truth, plus dataset aliases where a substrate supplies them,
  plus one mechanical leniency: a numeric truth matches a prediction containing
  exactly one equal digit run. No judge model anywhere.
* **Efficiency** is normalised by *that instance's* minimum step count, which
  the generator computes — so deep instances are not punished for being deep.
  `beta` is 0.05: correctness dominates, efficiency is a tiebreaker.
* **No `accuracy − α·context` term.** The ceiling is a constraint, not a price,
  so there is no α to tune and no degenerate optimum at either end of it.

---

## 3. The pipeline

| # | stage | what it produces | cost | status |
|---|---|---|---|---|
| 1 | `generate` | 2,750 instances, 4 splits, confound audit gating the write | 2 min CPU | ✅ done |
| 2 | `baselines --mode mechanical` | exact expectations, no model | seconds | ✅ done |
| 3 | `baselines --mode llm` | the three conditions, answer-only + reasoning | 25 min · 1 GPU | ✅ done |
| 4 | `evaluate --policy llm` (base) | untrained model in the environment | 25 min · 1 GPU | ✅ done |
| 5 | `teacher` | 3,832 leak-filtered trajectories → 20,068 SFT rows | 3 min CPU | ✅ done |
| 6 | `train_sft` | LoRA adapter | 2.8 h · 4 GPU (11 GPU-h) | ✅ done |
| 7 | `evaluate` (SFT) + `audit` + `diagnose` | the headline result and its mechanism | 30 min · 1 GPU | ✅ done |
| 8 | `train_grpo` | GRPO adapter | 10 min/step · 4 GPU (~40 GPU-h spent) | ⚠️ no improvement |
| 9 | `evaluate --instances scale_test` | accuracy by document size | 40 min · 1 GPU | ❌ **never run** |

Two side branches: the **MuSiQue substrate** (rejected — ceiling 0.374,
single-chunk at 33% of it) and **`inspect`**, which renders any split as a
readable Markdown document.

---

## 4. Where the code is

5,729 lines. Grouped by what the project's claim depends on:

**Load-bearing — the claim fails without these (1,650 lines)**
`env.py` `reward.py` `actions.py` `prompts.py` `tokens.py` `schema.py`
`generate.py` `corpus.py` `validate_instance.py` `baselines.py` `audit.py`
`rollout.py` `policies.py`

**Evidence that it works (700 lines)**
`teacher.py` `train_sft.py` `evaluate.py` `diagnose.py` `llm.py` `common.py`

**Negative results, kept as documentation (954 lines)**
`train_grpo.py` (420) · `substrates/musique.py` (534) — 17% of the codebase,
neither producing a positive result

**Tools (538 lines)** `cli.py` `inspect_doc.py`
**Tests (748 lines)** 117 tests, no GPU

---

## 5. Cut options

### A. Stop now — 0 GPU-hours
Everything above except stage 9 exists. The write-up is honest and complete
apart from one Definition-of-Done item.

### B. Minimal completion — 2 GPU-hours  ← *recommended floor*
Stage 9 (the document-size curve, 32k/64k/128k) plus a greedy SFT eval so the
headline table is apples-to-apples. Completes every Definition-of-Done item
except "GRPO improved something", which becomes a documented negative result.

### C. One targeted experiment — +4 GPU-hours  ← *recommended*
The diagnostic found that **44% of the SFT policy's reads land on a chunk that
does not hold the entity sought**, at every depth, and identified a cause: the
privileged teacher *asserts* a chunk index ("the map places it in chunk 27")
without deriving it, so SFT learns to assert an index it cannot compute. The
fix — having the teacher show the alphabetical comparison — is implemented and
tested but **has never been trained on**. Regenerate the teacher rows (3 min
CPU), retrain (3 h · 4 GPU), evaluate and diagnose (30 min · 1 GPU).

Clear hypothesis, pre-registered measurement (off-target read rate), and a
useful result in either direction:
* precision improves → demonstrating a decision's *derivation* teaches the
  skill where demonstrating its *outcome* teaches assertion;
* precision unchanged → a 7B cannot do alphabetical range comparison, and the
  synthetic corpus charges a navigation tax orthogonal to compression — which
  is itself a finding about environment design.

### D. Another shot at RL — +12 to +20 GPU-hours  ← *not recommended*
Expert iteration (stable by construction: sample, keep the correct ones,
re-SFT) or GRPO with a 4× larger batch. Both plausible; neither targets
anything we have *diagnosed*, which is what makes C better value.

**Recommendation: B + C, about 6 GPU-hours, then stop.** That yields two
training results, one of them testing a mechanism the instrumentation
identified, plus the complete baseline/audit/diagnostic story and two honest
negative results.
