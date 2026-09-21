# Resume bullets — Long-Context Navigation RL Environment

**Long-Context Navigation Environment** · Python, PyTorch, PEFT, vLLM, SLURM · [github.com/Manas-Ganti/long-context-navigator](https://github.com/Manas-Ganti/long-context-navigator)

- **Designed and built a verifiable multi-hop RL environment with a hard context ceiling** —
  the agent navigates 33k–200k-token synthetic documents with a 600-token working context
  via `READ / COMPRESS / DROP / ANSWER`, where exceeding the ceiling terminates the episode
  rather than incurring a tunable penalty; made the policy stateless between steps so a
  written summary is the *only* cross-step memory, turning "does it learn a compression
  policy" into a mechanically measurable question (exact token accounting, per-instance
  minimum-step computation, 100 CPU-only tests).

- **Built the task generator and validity pipeline so every property is known by
  construction**: entity-chain questions over alphabetised registers with near-miss twins,
  superseded statements and clustered near-miss values; a pure validator that re-derives the
  answer from the text, enforces fact separation, and proves no distractor path reaches the
  answer; a population-level surface-confound audit (rank AUC + logistic regression on
  position / length / entry count / header) that gates generation and caught a size-biased
  sampling confound (AUC 0.60 → 0.50) before any GPU was spent; rejection rates reported as a
  finding (3% at 2–3 hops → 16% at 4–5).

- **Established the baselines and a five-probe reward-hacking audit before training, then
  ran the SFT → GRPO pipeline on VT ARC (A100s)**: no-read 0.000 / single-chunk 0.000 /
  full-document 0.795 for the untrained 7B, and 0.000 for the same model inside the
  environment; a leak-filtered privileged teacher (20k per-step SFT rows, 0 leaks) lifted
  in-distribution accuracy to 0.630 with zero ceiling violations while the audit exposed the
  OOD mechanism (summary fact retention 0.98 → 0.60, 59% re-read loops at 4–5 hops); wrote a
  custom episode-weighted GRPO loop after diagnosing three objective bugs (wrong KL reference,
  length-biased gradient, unsynchronised ranks) from their training-curve signatures.

---

*Shorter variant (if space is tight):*

- Built a verifiable multi-hop QA RL environment where a **hard 600-token context ceiling
  terminates the episode** and the policy is stateless between steps, so `COMPRESS`
  summaries are the only memory — making compression policy a measurable object rather
  than a soft penalty to tune.
- Wrote the generator + pure validator so evidence locations, minimum steps and every
  distractor's wrong answer are known by construction; a surface-confound audit gates
  generation and caught a size-biased sampling shortcut before training.
- Ran baselines (no-read 0.000, single-chunk 0.000, full-document 0.795, untrained-in-env
  0.000), a leak-filtered teacher → SFT (ID 0.630, OOD 0.050 with the failure mechanism
  audited), and a custom GRPO loop on A100s, documenting each failed run's cause and fix.
