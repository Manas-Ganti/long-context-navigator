"""The three baselines that decide whether the environment is valid at all.

    no-read              answer from the question alone      -> must be at chance
    single-chunk oracle  the answer-bearing chunk, no chain  -> must be well below the oracle
    full-document oracle all evidence, no ceiling            -> must be high

Two modes. `mechanical` computes exact expectations with no model: it is what
the construction guarantees and it runs on a laptop. `llm` puts a real model
in each condition (llm.py) and is the number that goes in the README next to
the trained policy.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .corpus import parse_chunk_entries
from .reward import answer_matches, normalize_answer
from .schema import Instance
from .validate_instance import question_overlap, solve_from_text


def answer_values_in_chunk(inst: Instance, idx: int) -> list[str]:
    vals = []
    for p in parse_chunk_entries(inst.chunks[idx].text):
        if inst.answer_type in p.fields:
            vals.append(p.fields[inst.answer_type])
    return vals


def mechanical_baselines(instances: list[Instance], seed: int = 0) -> dict:
    rng = random.Random(seed)
    by_hops: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        b = by_hops[inst.n_hops]
        # no-read: nothing to condition on. Chance = 1 / |values of the answer type in the corpus|.
        b["no_read_fixed_guess"].append(float(answer_matches("unknown", inst.answer)))
        b["no_read_chance"].append(1.0 / max(len(inst.candidate_answers), 1))
        # single chunk, oracle-selected: the chunk that holds the answer, guess among its values
        vals = answer_values_in_chunk(inst, inst.answer_chunk)
        b["single_chunk_oracle_expected"].append(1.0 / max(len(vals), 1))
        b["single_chunk_oracle_sampled"].append(float(rng.choice(vals) == inst.answer))
        # single chunk chosen by lexical overlap with the question: does it even contain the answer?
        top = max(range(len(inst.chunks)), key=lambda i: question_overlap(inst.question, inst.chunks[i]))
        b["single_chunk_top_overlap_contains_answer"].append(float(inst.answer in answer_values_in_chunk(inst, top)))
        # full document: the mechanical reader follows the chain from the text
        ans, _ = solve_from_text(inst)
        b["full_document_oracle"].append(float(ans == inst.answer))
    out = {"n": len(instances), "by_hops": {}, "overall": {}}
    keys = set(k for b in by_hops.values() for k in b)
    for h in sorted(by_hops):
        out["by_hops"][h] = {k: sum(v) / len(v) for k, v in by_hops[h].items()} | {"n": len(by_hops[h]["full_document_oracle"])}
    for k in keys:
        allv = [x for b in by_hops.values() for x in b[k]]
        out["overall"][k] = sum(allv) / len(allv)
    return out


# --------------------------------------------------------------------------- #
# LLM conditions
# --------------------------------------------------------------------------- #
BASELINE_SYSTEM = ("Answer the question with the value only — no explanation. "
                   "If the passage gives a superseded value, use the current one. "
                   "If you cannot determine the answer, answer exactly: unknown")

# Same conditions, but the model may reason first. A 7B asked to compose a
# 3-5 hop chain with superseded values in one forward pass and 24 tokens is
# measuring one-shot composition, not whether it can read the evidence; the
# environment gives it one lookup per step, so this is the fairer ceiling.
BASELINE_SYSTEM_REASONING = (
    "Answer the question using only the passage(s). Work step by step: identify each entity in the "
    "chain in turn, quoting the relevant field. Entries can carry superseded values "
    "('previously X', 'X from 2019 to 2023, then Y') — use the CURRENT value. Entities with similar "
    "names are different entities. Finish with a final line of the form\nANSWER: <value>\n"
    "If the answer cannot be determined from the passages, finish with ANSWER: unknown")


def baseline_messages(inst: Instance, condition: str, rng: random.Random, reasoning: bool = False) -> list[dict]:
    if condition == "no-read":
        user = f"QUESTION: {inst.question}\nANSWER:"
    elif condition == "single-chunk":
        user = f"PASSAGE:\n{inst.chunks[inst.answer_chunk].text}\n\nQUESTION: {inst.question}\nANSWER:"
    elif condition == "full-document":
        idxs = list(inst.evidence_chunks)
        rng.shuffle(idxs)                      # order carries no hint
        passages = "\n\n".join(inst.chunks[i].text for i in idxs)
        user = f"PASSAGES:\n{passages}\n\nQUESTION: {inst.question}\nANSWER:"
    else:
        raise ValueError(condition)
    if reasoning:
        user = user[: -len("ANSWER:")].rstrip()
    return [{"role": "system", "content": BASELINE_SYSTEM_REASONING if reasoning else BASELINE_SYSTEM},
            {"role": "user", "content": user}]


def extract_answer(text: str) -> str:
    t = text.strip()
    if "ANSWER:" in t.upper():
        t = t[t.upper().rfind("ANSWER:") + 7:]
    return t.strip().splitlines()[0].strip() if t.strip() else ""


def llm_baselines(instances: list[Instance], backend, conditions=("no-read", "single-chunk", "full-document"),
                  batch_size: int = 16, seed: int = 0, max_new_tokens: int = 24, reasoning: bool = False) -> dict:
    """Run each condition through `backend.generate(messages_batch, ...)`.
    With `reasoning`, the model may think first (max_new_tokens should be ~512)."""
    rng = random.Random(seed)
    out = {"n": len(instances), "reasoning": reasoning, "max_new_tokens": max_new_tokens, "conditions": {}}
    for cond in conditions:
        rows = []
        for start in range(0, len(instances), batch_size):
            batch = instances[start:start + batch_size]
            msgs = [baseline_messages(i, cond, rng, reasoning) for i in batch]
            texts = backend.generate(msgs, max_new_tokens=max_new_tokens, temperature=0.0, sample=False)
            for inst, txt in zip(batch, texts):
                pred = extract_answer(txt)
                rows.append({"instance_id": inst.id, "n_hops": inst.n_hops, "pred": pred,
                             "truth": inst.answer,
                             "correct": answer_matches(pred, inst.answer, inst.answer_aliases),
                             "abstained": normalize_answer(pred) == "unknown",
                             "matches_distractor": any(normalize_answer(pred) == normalize_answer(v)
                                                       for d in inst.distractors for v in (d.value, d.wrong_answer) if v),
                             "raw": txt if reasoning else None})
        by_hops = defaultdict(list)
        for r in rows:
            by_hops[r["n_hops"]].append(r["correct"])
        out["conditions"][cond] = {
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "abstain_rate": sum(r["abstained"] for r in rows) / len(rows),
            "distractor_rate": sum(r["matches_distractor"] for r in rows) / len(rows),
            "by_hops": {h: sum(v) / len(v) for h, v in sorted(by_hops.items())},
            "rows": rows,
        }
    return out


def baseline_table(mech: dict, llm: dict | None = None) -> str:
    lines = ["| baseline | mode | accuracy | by n_hops |", "|---|---|---|---|"]
    o = mech["overall"]
    def hops(key):
        return ", ".join(f"{h}: {v[key]:.3f}" for h, v in mech["by_hops"].items())
    lines.append(f"| no-read (fixed guess) | mechanical | {o['no_read_fixed_guess']:.3f} | {hops('no_read_fixed_guess')} |")
    lines.append(f"| no-read chance level (1/candidates) | mechanical | {o['no_read_chance']:.3f} | {hops('no_read_chance')} |")
    lines.append(f"| single-chunk oracle (expected, guess within chunk) | mechanical | {o['single_chunk_oracle_expected']:.3f} | {hops('single_chunk_oracle_expected')} |")
    lines.append(f"| single-chunk by question overlap (contains answer?) | mechanical | {o['single_chunk_top_overlap_contains_answer']:.3f} | {hops('single_chunk_top_overlap_contains_answer')} |")
    lines.append(f"| full-document oracle (mechanical reader) | mechanical | {o['full_document_oracle']:.3f} | {hops('full_document_oracle')} |")
    if llm:
        mode = "LLM, reasoning" if llm.get("reasoning") else "LLM, answer-only"
        for cond, r in llm["conditions"].items():
            bh = ", ".join(f"{h}: {v:.3f}" for h, v in r["by_hops"].items())
            lines.append(f"| {cond} (abstain {r['abstain_rate']:.2f}, distractor {r['distractor_rate']:.2f}) "
                         f"| {mode} | {r['accuracy']:.3f} | {bh} |")
    return "\n".join(lines)
