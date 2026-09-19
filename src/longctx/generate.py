"""Task generator.

An instance is a document made of alphabetised REGISTERS (project, personnel,
department, facility, region, vendor) plus irrelevant NOISE sections, and a
question whose answer requires following a chain of `n_hops` entries:

    project X  --Lead-->  person  --Unit-->  department  --Site--> ...  --> <attribute>

Each entry names the next entity; only the last entry carries the answer. The
question mentions only the anchor (project X) and the attribute, so every
intermediate ("bridge") entity must be discovered by reading, and — because
the registers are separate sections and the working-context ceiling is smaller
than two chunks — carried across reads as a written summary.

Everything a policy might exploit is deliberately made uninformative:
  * sections are laid out in a random order, so evidence position is uniform;
  * every chunk is padded to a target length drawn from one distribution;
  * superseded statements and near-miss twin entities are inserted at the same
    rate for evidence and non-evidence entities, so their presence marks nothing;
  * attribute values are clustered WITHIN EVERY chunk (near-miss values), not
    just in the answer chunk.

The generator records, per instance, the exact evidence locations, all
distractors with the wrong answer each would produce, the minimum number of
reads and steps, and the seed. `validate_instance.validate` re-derives the
answer from the text and rejects anything that does not hold.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .config import EnvConfig, GeneratorConfig
from .corpus import (ANIMALS, COLORS, DEPT_ADJ, DEPT_NOUN, FACILITY_KIND, FILLER_GROUP,
                     FILLER_ITEM, FILLER_PLACE, FILLER_TEMPLATES, FILLER_WHEN, FIRST, LAST,
                     NOISE_SECTIONS, PLACE, TERRAIN, TYPE_SPECS, VENDOR_SUFFIX, AttrSpec,
                     Entity, chain_types, compose_question, format_value, range_label,
                     render_entry, value_number)
from .schema import Chunk, Distractor, Evidence, Instance
from .tokens import Tokenizer, get_tokenizer
from .validate_instance import min_step_schedule, validate


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def _name_space(t: str) -> list[str]:
    if t == "project":
        return [f"{c} {a}" for c in COLORS for a in ANIMALS]
    if t == "person":
        return [f"{f} {l}" for f in FIRST for l in LAST]
    if t == "department":
        return [f"{a} {n}" for a in DEPT_ADJ for n in DEPT_NOUN]
    if t == "facility":
        return [f"{p} {k}" for p in PLACE for k in FACILITY_KIND]
    if t == "region":
        return [f"{a} {t_}" for a in DEPT_ADJ for t_ in TERRAIN]
    if t == "vendor":
        return [f"{l} {s}" for l in LAST for s in VENDOR_SUFFIX]
    raise KeyError(t)


def filler_sentence(rng: random.Random) -> str:
    return rng.choice(FILLER_TEMPLATES).format(
        group=rng.choice(FILLER_GROUP), item=rng.choice(FILLER_ITEM),
        when=rng.choice(FILLER_WHEN), place=rng.choice(FILLER_PLACE))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass
class _Layout:
    chunks: list[Chunk]
    entity_chunk: dict[tuple[str, str], int]           # (type, name) -> chunk idx
    entity_offset: dict[tuple[str, str], int]          # (type, name) -> token offset


class InstanceGenerator:
    def __init__(self, gen: GeneratorConfig, env: EnvConfig, tokenizer: Tokenizer | None = None):
        self.gen = gen
        self.env = env
        self.tok = tokenizer or get_tokenizer(env.tokenizer)

    # ---- public --------------------------------------------------------- #
    def generate(self, seed: int, n_hops: int, doc_tokens: int, split: str = "") -> tuple[Instance | None, list[str]]:
        """One attempt. Returns (instance, []) or (None, rejection reasons).
        The instance is validated before it is returned."""
        try:
            inst = self._build(seed, n_hops, doc_tokens, split)
        except _Reject as r:
            return None, [str(r)]
        reasons = validate(inst, self.gen, self.env)
        if reasons:
            return None, reasons
        return inst, []

    def generate_valid(self, seed: int, n_hops: int, doc_tokens: int, split: str = "") -> tuple[Instance, Counter]:
        """Retry with derived seeds until an instance passes validation.
        Returns the instance and a Counter of rejection reasons seen on the way."""
        rejections: Counter = Counter()
        for attempt in range(self.gen.max_regeneration_attempts):
            inst, reasons = self.generate(seed * 1000 + attempt, n_hops, doc_tokens, split)
            if inst is not None:
                inst.generator["attempts"] = attempt + 1
                return inst, rejections
            for r in reasons:
                rejections[r.split(":")[0]] += 1
        raise RuntimeError(f"no valid instance after {self.gen.max_regeneration_attempts} attempts "
                           f"(seed={seed}, n_hops={n_hops}); rejections={dict(rejections)}")

    # ---- internals ------------------------------------------------------ #
    def _build(self, seed: int, n_hops: int, doc_tokens: int, split: str) -> Instance:
        rng = random.Random(seed)
        gen, env = self.gen, self.env
        if n_hops < 1:
            raise ValueError("n_hops must be >= 1")

        # -- sizes ------------------------------------------------------- #
        avg_chunk = (gen.chunk_min_tokens + gen.chunk_max_tokens) / 2
        total_chunks = max(int(round(doc_tokens / avg_chunk)), 2 * len(TYPE_SPECS) + 2)
        noise_chunks = int(round(gen.noise_ratio * total_chunks))
        register_chunks = total_chunks - noise_chunks
        per_type_chunks = max(2, register_chunks // len(TYPE_SPECS))
        avg_entries = (gen.entries_per_chunk_min + gen.entries_per_chunk_max) / 2
        per_type_entities = int(per_type_chunks * avg_entries)

        # -- pools ------------------------------------------------------- #
        pools: dict[str, list[Entity]] = {}
        taken: set[str] = set()
        for t, spec in TYPE_SPECS.items():
            space = _name_space(t)
            if per_type_entities > len(space):
                raise _Reject(f"name_space_exhausted:{t}")
            names = rng.sample(space, per_type_entities)
            pools[t] = [Entity(type=t, name=nm, link="", attrs={}, status=rng.choice(spec.status_values))
                        for nm in names]
            taken.update(names)

        # -- group each register into chunks BEFORE choosing the chain, and give
        #    EVERY chunk the same number of near-miss twins. The evidence chunk is
        #    then drawn uniformly over chunks, so no chunk-level property
        #    (entry count, having a twin pair, ...) can mark it. -------------- #
        groups: dict[str, list[list[Entity]]] = {}
        twin_of: dict[tuple[str, str], list[Entity]] = defaultdict(list)
        twinned: dict[str, list[list[Entity]]] = {}
        for t, spec in TYPE_SPECS.items():
            ents = sorted(pools[t], key=lambda e: e.name)
            n_tw = min(gen.n_distractors, len(spec.near_miss_suffixes))
            k_max = max(gen.entries_per_chunk_min, gen.entries_per_chunk_max - n_tw)
            groups[t], twinned[t] = [], []
            i = 0
            while i < len(ents):
                k = rng.randint(gen.entries_per_chunk_min, k_max)
                group = ents[i:i + k]
                i += k
                bases = rng.sample(group, min(n_tw, len(group)))
                for base in bases:
                    for suffix in rng.sample(spec.near_miss_suffixes, 1):
                        nm = base.name + suffix
                        if nm in taken:
                            continue
                        taken.add(nm)
                        tw = Entity(type=t, name=nm, link="", attrs={}, status=rng.choice(spec.status_values),
                                    is_near_miss_of=base.name)
                        pools[t].append(tw)
                        twin_of[(t, base.name)].append(tw)
                        group.append(tw)
                group.sort(key=lambda e: e.name)
                groups[t].append(group)
                twinned[t].append([b for b in bases if twin_of[(t, b.name)]])

        # -- the chain: chunk uniform over the type's chunks, entity uniform over
        #    that chunk's twinned entities (a type revisited by a long chain gets a
        #    fresh chunk) ------------------------------------------------------ #
        types = chain_types(n_hops)
        chain: list[Entity] = []
        used_groups: set[tuple[str, int]] = set()
        for t in types:
            options = [gi for gi in range(len(groups[t])) if (t, gi) not in used_groups and twinned[t][gi]]
            if not options:
                raise _Reject(f"no_free_chunk:{t}")
            gi = rng.choice(options)
            used_groups.add((t, gi))
            chain.append(rng.choice(twinned[t][gi]))
        answer_spec: AttrSpec = rng.choice(TYPE_SPECS[types[-1]].attrs)

        # -- links: random everywhere, chain overrides ---------------------- #
        for t, spec in TYPE_SPECS.items():
            nxt = pools[spec.next_type]
            for e in pools[t]:
                e.link = rng.choice(nxt).name
        for i in range(len(chain) - 1):
            chain[i].link = chain[i + 1].name
        # twins of chain entities must not point at the true next entity
        for i in range(len(chain) - 1):
            for tw in twin_of[(chain[i].type, chain[i].name)]:
                while tw.link == chain[i + 1].name:
                    tw.link = rng.choice(pools[types[i + 1]]).name

        # -- superseded statements (same rate for every entity) ------------ #
        for t, spec in TYPE_SPECS.items():
            nxt = pools[spec.next_type]
            for e in pools[t]:
                if rng.random() < gen.superseded_prob:
                    prev = rng.choice(nxt).name
                    while prev == e.link:
                        prev = rng.choice(nxt).name
                    e.prev_link = prev

        # -- render chunks: values clustered per chunk, padded to a target -- #
        used_values: dict[tuple[str, str], set[int]] = defaultdict(set)
        register_sections: dict[str, list[Chunk]] = {}
        entry_lines: dict[tuple[str, str], str] = {}
        for t, spec in TYPE_SPECS.items():
            chunks_here: list[Chunk] = []
            for group in groups[t]:
                for a in spec.attrs:
                    center = rng.randint(a.lo + a.near_delta, a.hi - a.near_delta)
                    for e in group:
                        e.attrs[a.name] = self._unique_near(rng, a, center, used_values[(t, a.name)])
                        if rng.random() < gen.superseded_prob * 0.5:
                            e.prev_attrs[a.name] = self._unique_near(rng, a, center, used_values[(t, a.name)])
                target = rng.randint(gen.chunk_min_tokens, gen.chunk_max_tokens - 24)
                header = f"{spec.section} · {range_label(group[0].name, group[-1].name)}"
                style_years = {}
                for e in group:
                    style, y1 = rng.randint(0, 1), rng.randint(2015, 2020)
                    style_years[e.name] = (style, y1, y1 + rng.randint(1, 4))
                lines = [render_entry(e, *style_years[e.name]) for e in group]
                total = self.tok.count(header) + sum(self.tok.count(l) + 1 for l in lines)
                if total > gen.chunk_max_tokens:
                    raise _Reject("chunk_overflow")
                while total < target:
                    e = rng.choice(group)
                    s = filler_sentence(rng)
                    cost = self.tok.count(s) + (2 if not e.notes else 0)
                    if total + cost > gen.chunk_max_tokens:
                        break
                    e.notes.append(s)
                    total += cost
                lines = [render_entry(e, *style_years[e.name]) for e in group]
                text = header + "\n" + "\n".join(lines)
                ntok = self.tok.count(text)
                if ntok > gen.chunk_max_tokens:
                    raise _Reject("chunk_overflow")
                for e, line in zip(group, lines):
                    entry_lines[(t, e.name)] = line
                chunks_here.append(Chunk(idx=-1, section=spec.section, header=header, kind="register",
                                         text=text, tokens=ntok, entity_names=[e.name for e in group]))
            register_sections[spec.section] = chunks_here

        # -- noise sections ------------------------------------------------ #
        noise_sections: dict[str, list[Chunk]] = {}
        if noise_chunks > 0:
            n_sections = max(1, min(len(NOISE_SECTIONS), noise_chunks // 4))
            names = rng.sample(NOISE_SECTIONS, n_sections)
            sizes = [noise_chunks // n_sections] * n_sections
            for j in range(noise_chunks - sum(sizes)):
                sizes[j] += 1
            for nm, sz in zip(names, sizes):
                cl = []
                for j in range(sz):
                    header = f"{nm} · entry {j + 1}"
                    target = rng.randint(gen.chunk_min_tokens, gen.chunk_max_tokens - 24)
                    lines, total = [], self.tok.count(header)
                    while total < target:
                        s = f"{rng.randint(2015, 2024)} — " + " ".join(filler_sentence(rng) for _ in range(rng.randint(1, 3)))
                        if total + self.tok.count(s) + 1 > gen.chunk_max_tokens:
                            break
                        lines.append(s)
                        total += self.tok.count(s) + 1
                    text = header + "\n" + "\n".join(lines)
                    cl.append(Chunk(idx=-1, section=nm, header=header, kind="noise", text=text,
                                    tokens=self.tok.count(text)))
                noise_sections[nm] = cl

        # -- layout: shuffle sections, number chunks, compute offsets ------- #
        sections = list(register_sections.items()) + list(noise_sections.items())
        rng.shuffle(sections)
        chunks: list[Chunk] = []
        entity_chunk: dict[tuple[str, str], int] = {}
        entity_offset: dict[tuple[str, str], int] = {}
        offset = 0
        for sec, cl in sections:
            for ch in cl:
                ch.idx = len(chunks)
                chunks.append(ch)
                if ch.kind == "register":
                    t = next(t for t, s in TYPE_SPECS.items() if s.section == sec)
                    inner = self.tok.count(ch.header) + 1
                    for nm in ch.entity_names:
                        entity_chunk[(t, nm)] = ch.idx
                        entity_offset[(t, nm)] = offset + inner
                        inner += self.tok.count(entry_lines[(t, nm)]) + 1
                offset += ch.tokens

        # -- evidence ------------------------------------------------------ #
        answer = chain[-1].attrs[answer_spec.name]
        evidence = []
        for hop, e in enumerate(chain):
            last = hop == len(chain) - 1
            evidence.append(Evidence(
                hop=hop, chunk_idx=entity_chunk[(e.type, e.name)], entity_type=e.type, entity=e.name,
                field=answer_spec.name if last else TYPE_SPECS[e.type].link_field,
                value=answer if last else e.link, token_offset=entity_offset[(e.type, e.name)]))

        # -- distractors (with the wrong answer each path produces) --------- #
        by_name = {(e.type, e.name): e for t in pools for e in pools[t]}

        def follow(start_type: str, start_name: str, from_hop: int) -> str | None:
            e = by_name.get((start_type, start_name))
            for hop in range(from_hop, len(types)):
                if e is None:
                    return None
                if hop == len(types) - 1:
                    return e.attrs.get(answer_spec.name)
                e = by_name.get((types[hop + 1], e.link))
            return None

        distractors = []
        for hop, e in enumerate(chain):
            last = hop == len(chain) - 1
            cidx = entity_chunk[(e.type, e.name)]
            for tw in twin_of[(e.type, e.name)]:
                distractors.append(Distractor(
                    kind="near_miss_entity", hop=hop, chunk_idx=entity_chunk[(tw.type, tw.name)],
                    entity=tw.name, value=tw.attrs[answer_spec.name] if last else tw.link,
                    wrong_answer=follow(tw.type, tw.name, hop)))
            if not last and e.prev_link:
                distractors.append(Distractor(
                    kind="superseded_link", hop=hop, chunk_idx=cidx, entity=e.name, value=e.prev_link,
                    wrong_answer=follow(types[hop + 1], e.prev_link, hop + 1)))
            if last and answer_spec.name in e.prev_attrs:
                distractors.append(Distractor(
                    kind="superseded_value", hop=hop, chunk_idx=cidx, entity=e.name,
                    value=e.prev_attrs[answer_spec.name], wrong_answer=e.prev_attrs[answer_spec.name]))
            if last:
                for other in chunks[cidx].entity_names:
                    if other != e.name:
                        oe = by_name[(e.type, other)]
                        distractors.append(Distractor(
                            kind="near_miss_value", hop=hop, chunk_idx=cidx, entity=other,
                            value=oe.attrs[answer_spec.name], wrong_answer=oe.attrs[answer_spec.name]))

        # -- minimum steps under the ceiling ------------------------------- #
        ev_tokens = [chunks[ev.chunk_idx].tokens for ev in evidence]
        min_steps, comp_required, ok = min_step_schedule(ev_tokens, env.context_ceiling, env.assumed_summary_tokens)
        if not ok:
            raise _Reject("unsolvable_under_ceiling")

        candidates = sorted(e.attrs[answer_spec.name] for e in pools[types[-1]])
        return Instance(
            id=f"{split or 'x'}-{seed}", seed=seed, split=split, n_hops=n_hops,
            doc_tokens_target=doc_tokens, doc_tokens=offset,
            question=compose_question(n_hops, chain[0].name, answer_spec.name),
            answer=answer, answer_type=answer_spec.name, answer_entity_type=types[-1],
            chunks=chunks, evidence=evidence, distractors=distractors,
            candidate_answers=candidates, min_reads=len(evidence), min_steps=min_steps,
            compression_required=comp_required,
            generator=gen.model_dump(exclude={"splits"}) | {"tokenizer": self.tok.name},
            env={"context_ceiling": env.context_ceiling, "assumed_summary_tokens": env.assumed_summary_tokens,
                 "step_budget": env.step_budget(n_hops)},
        )

    @staticmethod
    def _unique_near(rng: random.Random, a: AttrSpec, center: int, used: set[int]) -> str:
        for _ in range(200):
            n = center + rng.randint(-a.near_delta, a.near_delta)
            if a.lo <= n <= a.hi and n not in used:
                used.add(n)
                return format_value(a, n)
        for _ in range(2000):
            n = rng.randint(a.lo, a.hi)
            if n not in used:
                used.add(n)
                return format_value(a, n)
        raise _Reject("value_space_exhausted")


class _Reject(Exception):
    pass


# --------------------------------------------------------------------------- #
# Batch generation with bookkeeping
# --------------------------------------------------------------------------- #
def generate_split(gen: GeneratorConfig, env: EnvConfig, split: str, tokenizer: Tokenizer | None = None,
                   limit: int | None = None, progress=None) -> tuple[list[Instance], dict]:
    """Generate one named split from the config. Returns instances and a stats
    dict with the rejection counts per reason (the confound-rejection rate)."""
    cfg = gen.splits[split]
    g = InstanceGenerator(gen, env, tokenizer)
    n = cfg.n if limit is None else min(cfg.n, limit)
    out: list[Instance] = []
    rejections: Counter = Counter()
    attempts = 0
    for i in range(n):
        seed = cfg.seed_base + i
        rng = random.Random(seed)
        n_hops = rng.choice(cfg.n_hops)
        doc_tokens = rng.choice(cfg.doc_tokens)
        inst, rej = g.generate_valid(seed, n_hops, doc_tokens, split)
        rejections.update(rej)
        attempts += inst.generator["attempts"]
        out.append(inst)
        if progress and (i + 1) % 25 == 0:
            progress(i + 1, n)
    stats = {
        "split": split, "n": len(out), "attempts": attempts,
        "rejected_attempts": attempts - len(out),
        "rejection_rate": ((attempts - len(out)) / attempts) if attempts else 0.0,
        "rejections_by_reason": dict(rejections),   # an attempt may carry several reasons
        "n_hops": dict(Counter(i.n_hops for i in out)),
        "doc_tokens_mean": sum(i.doc_tokens for i in out) / max(len(out), 1),
        "compression_required_rate": sum(i.compression_required for i in out) / max(len(out), 1),
        "min_steps_mean": sum(i.min_steps for i in out) / max(len(out), 1),
    }
    return out, stats
