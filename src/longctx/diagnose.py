"""Failure-mode accounting over trajectory files.

The audit asks "is the policy cheating". This asks "where do the steps go" —
which failure dominates, and whether the wasted steps are SEARCH (reading a
chunk that does not hold the entity being looked for) or MEMORY (compressing
away a fact and having to fetch it again).

A READ is scored against what the policy needed at that moment: the chain's
next unresolved hop. A read is
  * on-target   — the chunk holds that hop's entity
  * re-read     — a chunk already read earlier in the episode
  * off-target  — neither (a navigation miss)
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .schema import Instance


def _outcome(t: dict) -> str:
    if t["ceiling_exceeded"]:
        return "ceiling_violation"
    if t["correct"]:
        return "correct"
    if t["budget_exhausted"] or not t["answered"]:
        return "budget_exhausted"
    return "wrong_answer"


def classify_reads(t: dict, inst: Instance) -> dict:
    """Walk the episode, tracking which hop the policy still needs."""
    ev_chunk = {e.chunk_idx: e.hop for e in inst.evidence}
    resolved = 0                       # hops whose value the policy has seen
    seen: set[int] = set()
    counts = Counter()
    for s in t["steps"]:
        if s.get("kind") != "READ" or not s.get("ok"):
            continue
        idx = int(s["ids"][0])
        if idx in seen:
            counts["re_read"] += 1
        elif ev_chunk.get(idx) == resolved:
            counts["on_target"] += 1
            resolved += 1
        elif idx in ev_chunk:
            counts["out_of_order"] += 1    # an evidence chunk, but not the one needed yet
        else:
            counts["off_target"] += 1
        seen.add(idx)
    counts["total"] = sum(counts[k] for k in ("on_target", "re_read", "out_of_order", "off_target"))
    counts["hops_resolved"] = resolved
    return counts


def diagnose(trajs: list[dict], instances: list[Instance]) -> dict:
    insts = {i.id: i for i in instances}
    trajs = [t for t in trajs if t["instance_id"] in insts]
    by_split: dict[str, dict] = defaultdict(lambda: {"outcomes": Counter(), "reads": Counter(),
                                                     "n": 0, "steps": 0, "hops_needed": 0})
    for t in trajs:
        inst = insts[t["instance_id"]]
        split = "ood" if inst.n_hops > 3 else "id"
        for key in (split, "all"):
            b = by_split[key]
            b["n"] += 1
            b["steps"] += t["steps_used"]
            b["hops_needed"] += inst.n_hops
            b["outcomes"][_outcome(t)] += 1
            b["reads"].update(classify_reads(t, inst))
    out = {}
    for key, b in by_split.items():
        n, r = max(b["n"], 1), b["reads"]
        total_reads = max(r["total"], 1)
        out[key] = {
            "episodes": b["n"],
            "outcomes": {k: v / n for k, v in b["outcomes"].items()},
            "reads_per_episode": r["total"] / n,
            "hops_resolved_per_episode": r["hops_resolved"] / n,
            "hops_needed_per_episode": b["hops_needed"] / n,
            "read_precision_on_target": r["on_target"] / total_reads,
            "read_off_target": r["off_target"] / total_reads,
            "read_re_read": r["re_read"] / total_reads,
            "read_out_of_order": r["out_of_order"] / total_reads,
            "wasted_read_steps_per_episode": (r["off_target"] + r["re_read"] + r["out_of_order"]) / n,
            "steps_per_episode": b["steps"] / n,
        }
    return out


def markdown(rep: dict) -> str:
    keys = [k for k in ("all", "id", "ood") if k in rep]
    rows = [
        ("episodes", lambda b: f"{b['episodes']}"),
        ("correct", lambda b: f"{b['outcomes'].get('correct', 0):.3f}"),
        ("budget exhausted", lambda b: f"{b['outcomes'].get('budget_exhausted', 0):.3f}"),
        ("wrong answer", lambda b: f"{b['outcomes'].get('wrong_answer', 0):.3f}"),
        ("ceiling violation", lambda b: f"{b['outcomes'].get('ceiling_violation', 0):.3f}"),
        ("hops resolved / needed", lambda b: f"{b['hops_resolved_per_episode']:.2f} / {b['hops_needed_per_episode']:.2f}"),
        ("reads per episode", lambda b: f"{b['reads_per_episode']:.2f}"),
        ("READ on target", lambda b: f"{b['read_precision_on_target']:.3f}"),
        ("READ off target (navigation miss)", lambda b: f"{b['read_off_target']:.3f}"),
        ("READ re-read (memory/loop)", lambda b: f"{b['read_re_read']:.3f}"),
        ("READ out of order", lambda b: f"{b['read_out_of_order']:.3f}"),
        ("wasted read steps / episode", lambda b: f"{b['wasted_read_steps_per_episode']:.2f}"),
    ]
    lines = ["| metric | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(rep[k]) for k in keys) + " |")
    return "\n".join(lines)
