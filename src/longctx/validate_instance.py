"""PURE instance validation. No model, no randomness, no I/O.

Two layers:

* `validate(instance, ...)` — per-instance checks. A non-empty list of reasons
  means the instance is rejected and the generator tries a fresh seed.
    - every required fact is present in the chunk it claims to be in
    - the chain is solvable FROM THE TEXT by a mechanical reader
    - required facts sit in distinct chunks, >= min_separation tokens apart
    - no distractor path (superseded value, near-miss entity) reaches the answer
    - the answer value occurs exactly once as a field value in the corpus
    - lexical shortcut: for n_hops >= 2 the anchor entity never appears in the
      answer chunk, and the answer chunk is not the question's top-overlap chunk
    - the instance is solvable under the ceiling (every evidence chunk fits,
      and a summary plus the next chunk fits)

* `confound_audit(instances, ...)` — population checks. Whether chunk position,
  chunk length or the section header predicts evidence is a statement about a
  DISTRIBUTION of instances, not about one instance: rejecting single instances
  where "the answer happens to be in the longest chunk" would make "longest"
  anti-predictive, which is itself a surface signal. So position / length /
  header are audited across a batch with rank AUCs and a small logistic
  regression, and the generator run FAILS if any AUC leaves 0.5 +/- tolerance.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .config import EnvConfig, GeneratorConfig
from .corpus import TYPE_SPECS, chain_types, parse_chunk_entries
from .schema import Chunk, Instance

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")
STOPWORDS = frozenset("""
the of a an is what which who to in on at by for from and or with that this
current since previously then from until notes status
""".split())


# --------------------------------------------------------------------------- #
# Mechanical chain solver (reads the text, as a reader would)
# --------------------------------------------------------------------------- #
def _find_entry(inst: Instance, section: str, name: str):
    """Locate (chunk_idx, ParsedEntry) for `name` in `section`. Scans every chunk
    of that section — this is an oracle for solvability, not a policy."""
    for ch in inst.chunks:
        if ch.section != section:
            continue
        for p in parse_chunk_entries(ch.text):
            if p.name == name:
                return ch.idx, p
    return None, None


def solve_from_text(inst: Instance, branch: dict[int, str] | None = None) -> tuple[str | None, list[int]]:
    """Follow the chain by parsing entries. `branch` maps hop -> "prev" (take the
    superseded value at that hop) or an entity name (start that hop from a
    near-miss entity instead). Returns (answer_or_None, chunk_path)."""
    branch = branch or {}
    types = chain_types(inst.n_hops)
    anchor = inst.evidence[0].entity
    entity = anchor
    path: list[int] = []
    for hop, t in enumerate(types):
        spec = TYPE_SPECS[t]
        if branch.get(hop) not in (None, "prev"):
            entity = branch[hop]
        cidx, entry = _find_entry(inst, spec.section, entity)
        if entry is None:
            return None, path
        path.append(cidx)
        last = hop == len(types) - 1
        field = inst.answer_type if last else spec.link_field
        if branch.get(hop) == "prev":
            val = entry.previous.get(field)
            if val is None:
                return None, path
        else:
            val = entry.fields.get(field)
        if val is None:
            return None, path
        if last:
            return val, path
        entity = val
    return None, path


# --------------------------------------------------------------------------- #
# Per-instance checks
# --------------------------------------------------------------------------- #
def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS}


def question_overlap(question: str, chunk: Chunk) -> float:
    q = content_words(question)
    if not q:
        return 0.0
    return len(q & content_words(chunk.text)) / len(q)


def min_step_schedule(chunk_tokens: list[int], ceiling: int, summary_tokens: int) -> tuple[int, bool, bool]:
    """(min_steps, compression_required, solvable) for a forced read order.

    One READ per chunk, one ANSWER, plus one COMPRESS-everything whenever the
    next chunk would not fit beside what is held. COMPRESS-all is the maximal
    single freeing action, and freeing earlier than needed never lowers the
    count, so this greedy schedule is optimal in step count.

    compression_required is True when a DROP-only strategy cannot work: some
    consecutive pair of evidence chunks does not fit together, so the bridge
    entity from chunk i-1 must survive as a summary while chunk i is read.
    """
    used = 0
    steps = 0
    for i, c in enumerate(chunk_tokens):
        if c > ceiling:
            return 0, False, False
        if used + c > ceiling:
            steps += 1                     # COMPRESS everything held
            used = summary_tokens
            if used + c > ceiling:
                return 0, False, False
        steps += 1                         # READ
        used += c
    steps += 1                             # ANSWER
    comp_required = any(chunk_tokens[i - 1] + chunk_tokens[i] > ceiling
                        for i in range(1, len(chunk_tokens)))
    return steps, comp_required, True


def validate(inst: Instance, gen: GeneratorConfig, env: EnvConfig) -> list[str]:
    reasons: list[str] = []
    n = len(inst.chunks)

    # 1. facts present where recorded
    for ev in inst.evidence:
        if not (0 <= ev.chunk_idx < n):
            reasons.append(f"evidence_missing:hop{ev.hop}")
            continue
        text = inst.chunks[ev.chunk_idx].text
        if ev.entity not in text or ev.value not in text:
            reasons.append(f"evidence_missing:hop{ev.hop}")

    # 2. solvable from text, along the recorded chunks
    ans, path = solve_from_text(inst)
    if ans != inst.answer:
        reasons.append("unreachable")
    elif path != inst.evidence_chunks:
        reasons.append("evidence_path_mismatch")

    # 3. distinct chunks + separation
    idxs = inst.evidence_chunks
    if len(set(idxs)) != len(idxs):
        reasons.append("co_located")
    offs = [e.token_offset for e in inst.evidence]
    for i in range(len(offs)):
        for j in range(i + 1, len(offs)):
            if abs(offs[i] - offs[j]) < gen.min_separation:
                reasons.append("separation")
                break

    # 4. distractor paths never reach the answer
    for d in inst.distractors:
        if d.wrong_answer == inst.answer or d.value == inst.answer:
            reasons.append(f"distractor_reaches_answer:{d.kind}")
    for hop in range(inst.n_hops):
        a, _ = solve_from_text(inst, {hop: "prev"})
        if a == inst.answer:
            reasons.append(f"superseded_path_reaches_answer:hop{hop}")
        for d in inst.distractors:
            if d.kind == "near_miss_entity" and d.hop == hop:
                a, _ = solve_from_text(inst, {hop: d.entity})
                if a == inst.answer:
                    reasons.append(f"near_miss_path_reaches_answer:hop{hop}")

    # 5. answer value is unique as a field value
    occurrences = 0
    for ch in inst.chunks:
        for p in parse_chunk_entries(ch.text):
            occurrences += sum(1 for v in p.fields.values() if v == inst.answer)
            occurrences += sum(1 for v in p.previous.values() if v == inst.answer)
    if occurrences != 1:
        reasons.append(f"answer_not_unique:{occurrences}")

    # 6. lexical shortcut
    if inst.n_hops >= 2:
        anchor = inst.evidence[0].entity
        ans_chunk = inst.chunks[inst.answer_chunk]
        if anchor in ans_chunk.text:
            reasons.append("anchor_in_answer_chunk")
        overlaps = [question_overlap(inst.question, c) for c in inst.chunks]
        best = max(overlaps)
        if overlaps[inst.answer_chunk] >= best and overlaps.count(best) == 1:
            reasons.append("answer_chunk_is_top_overlap")

    # 7. solvable under the ceiling
    steps, comp, ok = min_step_schedule(
        [inst.chunks[i].tokens for i in idxs], env.context_ceiling, env.assumed_summary_tokens)
    if not ok:
        reasons.append("unsolvable_under_ceiling")
    elif steps != inst.min_steps or comp != inst.compression_required:
        reasons.append("min_steps_mismatch")

    return reasons


# --------------------------------------------------------------------------- #
# Population-level surface-confound audit
# --------------------------------------------------------------------------- #
def surface_features(inst: Instance) -> list[dict]:
    """One row per REGISTER chunk. Noise chunks are excluded on purpose: the map
    labels them, so 'not noise' is navigation, not a confound."""
    rows = []
    n = len(inst.chunks)
    by_section: dict[str, list[int]] = defaultdict(list)
    for ch in inst.chunks:
        if ch.kind == "register":
            by_section[ch.section].append(ch.idx)
    ev = set(inst.evidence_chunks)
    ans_section = inst.chunks[inst.answer_chunk].section
    for sec, idxs in by_section.items():
        m = len(idxs)
        for rank, idx in enumerate(idxs):
            ch = inst.chunks[idx]
            rows.append({
                "instance": inst.id,
                "section": sec,
                "pos_global": idx / max(n - 1, 1),
                "pos_in_section": rank / max(m - 1, 1),
                "length": ch.tokens,
                "n_entries": len(ch.entity_names),
                "overlap": question_overlap(inst.question, ch),
                "is_evidence": int(idx in ev),
                "is_answer": int(idx == inst.answer_chunk),
                "in_answer_section": int(sec == ans_section),
            })
    return rows


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUC with tie handling. 0.5 if one class is empty."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _logreg_cv_auc(X: np.ndarray, y: np.ndarray, folds: int = 2, epochs: int = 300, lr: float = 0.1) -> float:
    """A deliberately trivial classifier: standardized features, logistic
    regression by gradient descent, k-fold CV, pooled AUC."""
    n = len(y)
    if n < 10 or y.sum() == 0 or y.sum() == n:
        return 0.5
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    idx = np.arange(n)
    scores = np.zeros(n)
    for f in range(folds):
        test = idx % folds == f
        train = ~test
        w = np.zeros(X.shape[1])
        b = 0.0
        Xt, yt = Xs[train], y[train]
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Xt @ w + b)))
            g = p - yt
            w -= lr * (Xt.T @ g) / len(yt)
            b -= lr * g.mean()
        scores[test] = Xs[test] @ w + b
    return rank_auc(scores, y)


FEATURES = ("pos_global", "pos_in_section", "length", "n_entries", "overlap")
GATED = ("pos_global", "pos_in_section", "length", "n_entries")


def confound_audit(instances: list[Instance], tolerance: float) -> dict:
    """AUC of each single surface feature, and of a joint logistic regression,
    for predicting (a) evidence chunks and (b) the answer chunk, computed within
    the relevant section pool. Returns a report with a pass/fail verdict."""
    rows = [r for inst in instances for r in surface_features(inst)]
    report: dict = {"n_instances": len(instances), "n_chunks": len(rows), "auc": {}, "violations": []}
    if not rows:
        report["passed"] = True
        return report

    def block(rows_: list[dict], label: str) -> dict:
        y = np.array([r[label] for r in rows_], dtype=float)
        out = {}
        for f in FEATURES:
            out[f] = rank_auc(np.array([r[f] for r in rows_], dtype=float), y)
        X = np.array([[r[f] for f in GATED] for r in rows_], dtype=float)
        out["logreg_gated_features"] = _logreg_cv_auc(X, y)
        out["n_pos"] = int(y.sum())
        return out

    # Evidence among all register chunks (sections shuffled, so position/length
    # carry nothing; overlap legitimately flags the anchor's hop-0 chunk).
    report["auc"]["evidence_all_register"] = block(rows, "is_evidence")
    # The answer chunk among the chunks of ITS section: the only pool where a
    # shortcut would matter, and where the header no longer separates anything.
    ans_rows = [r for r in rows if r["in_answer_section"]]
    report["auc"]["answer_within_section"] = block(ans_rows, "is_answer")
    # Multi-hop instances only: the answer chunk must be lexically invisible.
    hops_of = {i.id: i.n_hops for i in instances}
    multi = [r for r in rows if r["in_answer_section"] and hops_of[r["instance"]] >= 2]
    if multi:
        report["auc"]["answer_within_section_multihop"] = block(multi, "is_answer")

    for pool, aucs in report["auc"].items():
        for f in GATED + ("logreg_gated_features",):
            if abs(aucs[f] - 0.5) > tolerance:
                report["violations"].append(f"{pool}:{f}={aucs[f]:.3f}")
        if pool == "answer_within_section_multihop" and abs(aucs["overlap"] - 0.5) > tolerance:
            report["violations"].append(f"{pool}:overlap={aucs['overlap']:.3f}")
    report["tolerance"] = tolerance
    report["passed"] = not report["violations"]
    return report
