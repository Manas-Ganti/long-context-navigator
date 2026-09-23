"""Render a dataset as one readable Markdown document.

Three layers, because a full dump of every instance is ~350 MB of text and
nobody reads that:

  1. summary — what the split contains and how it was built;
  2. worked examples — a handful rendered completely: question, the chain hop
     by hop, the document map, the evidence chunks verbatim, the distractors;
  3. an index of EVERY instance — one row each with its question, answer,
     hop count, evidence locations and size.

`--full` additionally renders every chunk of every instance; it is offered
because it is occasionally what you want, and warned about because of the size.
"""

from __future__ import annotations

from pathlib import Path

from .env import NavigationEnv
from .schema import Instance


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


def render_instance(inst: Instance, env_cfg, full: bool = False) -> str:
    env = NavigationEnv(env_cfg)
    obs = env.reset(inst)
    ev_chunks = inst.evidence_chunks
    out = [f"### `{inst.id}`  ·  {inst.n_hops} hops  ·  {inst.doc_tokens:,} tokens  "
           f"·  {len(inst.chunks)} chunks", "",
           f"**Question** {inst.question}", "",
           f"**Answer** `{inst.answer}`" +
           (f"  (also accepted: {', '.join('`%s`' % a for a in inst.answer_aliases)})"
            if inst.answer_aliases else ""), "",
           f"**Minimum** {inst.min_reads} reads, {inst.min_steps} steps; "
           f"compression required: {inst.compression_required}; "
           f"step budget {inst.env.get('step_budget')}; ceiling {inst.env.get('context_ceiling')} tokens", "",
           "**The chain**", "",
           "| hop | chunk | token offset | entry | what it gives |", "|---|---|---|---|---|"]
    for ev in inst.evidence:
        out.append(f"| {ev.hop} | {ev.chunk_idx} | {ev.token_offset:,} | {ev.entity} | "
                   f"{ev.field}: **{ev.value}** |")
    gaps = [abs(a.token_offset - b.token_offset) for i, a in enumerate(inst.evidence)
            for b in inst.evidence[i + 1:]]
    if gaps:
        out += ["", f"Closest pair of required facts: {min(gaps):,} tokens apart "
                    f"(ceiling is {inst.env.get('context_ceiling')}, so they can never be held together)."]

    if inst.distractors:
        out += ["", "**Distractors**", "",
                "| kind | hop | chunk | entry | value | answer it would produce |", "|---|---|---|---|---|---|"]
        for d in inst.distractors[:14]:
            out.append(f"| {d.kind} | {d.hop if d.hop >= 0 else '—'} | "
                       f"{d.chunk_idx if d.chunk_idx >= 0 else '—'} | {d.entity} | `{d.value}` | "
                       f"{'`%s`' % d.wrong_answer if d.wrong_answer else '—'} |")
        if len(inst.distractors) > 14:
            out.append(f"| … {len(inst.distractors) - 14} more | | | | | |")

    out += ["", "**Document map (what the agent sees)**", "", _fence("\n".join(obs.map_lines))]
    shown = range(len(inst.chunks)) if full else ev_chunks
    label = "Every chunk" if full else "The evidence chunks, verbatim"
    out += ["", f"**{label}**", ""]
    for idx in shown:
        c = inst.chunks[idx]
        tag = ""
        if idx in ev_chunks:
            tag = f"  ← hop {ev_chunks.index(idx)}"
        out += [f"*chunk {idx} · {c.tokens} tokens{tag}*", "", _fence(c.text), ""]
    return "\n".join(out)


def summary(instances: list[Instance], stats: dict | None, confound: dict | None) -> str:
    from collections import Counter
    hops = Counter(i.n_hops for i in instances)
    sizes = [i.doc_tokens for i in instances]
    chunk_tokens = [c.tokens for i in instances for c in i.chunks]
    subs = Counter(i.substrate for i in instances)
    out = ["## Summary", "",
           "| | |", "|---|---|",
           f"| instances | {len(instances)} |",
           f"| substrate | {', '.join(f'{k} ({v})' for k, v in subs.items())} |",
           f"| hops | {', '.join(f'{k}: {v}' for k, v in sorted(hops.items()))} |",
           f"| document tokens | min {min(sizes):,} · mean {sum(sizes) // len(sizes):,} · max {max(sizes):,} |",
           f"| chunks per document | {sum(len(i.chunks) for i in instances) // len(instances)} |",
           f"| chunk tokens | min {min(chunk_tokens)} · mean {sum(chunk_tokens) // len(chunk_tokens)} "
           f"· max {max(chunk_tokens)} |",
           f"| minimum steps | mean {sum(i.min_steps for i in instances) / len(instances):.2f} |",
           f"| compression required | {sum(i.compression_required for i in instances) / len(instances):.1%} |",
           f"| context ceiling | {instances[0].env.get('context_ceiling')} tokens |",
           ]
    if stats:
        out += ["", "### How it was built", "",
                _fence("\n".join(f"{k}: {v}" for k, v in stats.items() if k != "confound"), "yaml")]
    if confound and confound.get("auc"):
        out += ["", "### Surface-confound audit",
                "", "Does any surface feature predict where the evidence is? "
                    f"All gated AUCs must sit within 0.5 ± {confound.get('tolerance', 0.06)}.", "",
                "| pool | " + " | ".join(next(iter(confound["auc"].values())).keys()) + " |",
                "|---|" + "---|" * len(next(iter(confound["auc"].values())))]
        for pool, aucs in confound["auc"].items():
            out.append(f"| {pool} | " + " | ".join(
                f"{v:.3f}" if isinstance(v, float) else str(v) for v in aucs.values()) + " |")
        verdict = "PASSED" if confound.get("passed") else f"FAILED {confound.get('violations')}"
        out += ["", f"**{verdict}**"]
    return "\n".join(out)


def index_table(instances: list[Instance]) -> str:
    out = ["## Every instance", "",
           "| id | hops | doc tokens | evidence chunks | min steps | question | answer |",
           "|---|---|---|---|---|---|---|"]
    for i in instances:
        q = i.question if len(i.question) <= 110 else i.question[:107] + "…"
        out.append(f"| `{i.id}` | {i.n_hops} | {i.doc_tokens:,} | "
                   f"{', '.join(map(str, i.evidence_chunks))} | {i.min_steps} | {q} | `{i.answer}` |")
    return "\n".join(out)


def build_document(instances: list[Instance], env_cfg, *, title: str, examples: int = 3,
                   stats: dict | None = None, confound: dict | None = None,
                   full: bool = False) -> str:
    parts = [f"# {title}", "",
             f"{len(instances)} instances. Generated by `longctx inspect`; every number here is "
             "read from the instance files themselves.", "",
             summary(instances, stats, confound), ""]
    if examples:
        parts += ["## Worked examples", "",
                  "Rendered in full: the question, the chain of facts it requires, where each one "
                  "sits, the map the agent navigates with, and the evidence chunks as the agent "
                  "would load them.", ""]
        step = max(len(instances) // max(examples, 1), 1)
        for inst in instances[::step][:examples]:
            parts += [render_instance(inst, env_cfg, full=False), ""]
    if full:
        parts += ["## All instances, in full", ""]
        for inst in instances:
            parts += [render_instance(inst, env_cfg, full=True), ""]
    parts += [index_table(instances), ""]
    return "\n".join(parts)


def write_document(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
