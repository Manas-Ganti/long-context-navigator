"""Evaluation on held-out generated instances.

Reports accuracy by n_hops (in-distribution vs OOD relative to the training
hop cap), accuracy by document size, step efficiency against the per-instance
minimum, compression utilisation and fact retention, and the ceiling-violation
rate. Writes trajectories (for the audit) and a metrics JSON + markdown.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .audit import probe_compression_noop
from .config import EnvConfig
from .rollout import Trajectory
from .schema import Instance


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize(trajs: list[Trajectory], instances: list[Instance], train_max_hops: int) -> dict:
    insts = {i.id: i for i in instances}
    tl = [t.to_dict() for t in trajs]

    def block(ts: list[dict]) -> dict:
        if not ts:
            return {"n": 0}
        comp = probe_compression_noop(ts, insts)
        return {
            "n": len(ts),
            "accuracy": _mean(t["correct"] for t in ts),
            "reward": _mean(t["reward"] for t in ts),
            "ceiling_violation_rate": _mean(t["ceiling_exceeded"] for t in ts),
            "budget_exhausted_rate": _mean(t["budget_exhausted"] for t in ts),
            "answered_rate": _mean(t["answered"] for t in ts),
            "steps_over_min": _mean(t["steps_used"] / t["min_steps"] for t in ts),
            "steps_over_min_when_correct": _mean(t["steps_used"] / t["min_steps"] for t in ts if t["correct"]),
            "reads_over_min": _mean(t["n_reads"] / max(t["min_reads"], 1) for t in ts),
            "compress_usage": comp["episodes_using_compress_rate"],
            "fact_retention": comp["fact_retention_rate"],
            "degenerate_summary_rate": comp["degenerate_summary_rate"],
            "invalid_action_rate": _mean(t["n_invalid"] / max(t["steps_used"], 1) for t in ts),
        }

    by_hops = defaultdict(list)
    by_doc = defaultdict(list)
    for t in tl:
        by_hops[t["n_hops"]].append(t)
        by_doc[insts[t["instance_id"]].doc_tokens_target].append(t)
    return {
        "policy": trajs[0].policy if trajs else None,
        "train_max_hops": train_max_hops,
        "overall": block(tl),
        "in_distribution": block([t for t in tl if t["n_hops"] <= train_max_hops]),
        "ood": block([t for t in tl if t["n_hops"] > train_max_hops]),
        "by_hops": {h: block(v) for h, v in sorted(by_hops.items())},
        "by_doc_tokens": {d: block(v) for d, v in sorted(by_doc.items())},
    }


def markdown(rep: dict) -> str:
    f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    cols = ["n", "accuracy", "ceiling_violation_rate", "steps_over_min", "compress_usage", "fact_retention"]
    lines = [f"policy: `{rep['policy']}`  (trained on n_hops <= {rep['train_max_hops']})", "",
             "| split | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name in ("overall", "in_distribution", "ood"):
        b = rep[name]
        lines.append(f"| {name} | " + " | ".join(f(b.get(c)) for c in cols) + " |")
    lines += ["", "| n_hops | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for h, b in rep["by_hops"].items():
        lines.append(f"| {h}{' (OOD)' if int(h) > rep['train_max_hops'] else ''} | " + " | ".join(f(b.get(c)) for c in cols) + " |")
    if len(rep["by_doc_tokens"]) > 1:
        lines += ["", "| doc_tokens | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for d, b in rep["by_doc_tokens"].items():
            lines.append(f"| {d} | " + " | ".join(f(b.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def write_report(rep: dict, trajs: list[Trajectory], out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(rep, f, indent=1)
    with open(f"{out_prefix}.md", "w") as f:
        f.write(markdown(rep) + "\n")
    with open(f"{out_prefix}.trajectories.jsonl", "w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict()) + "\n")
