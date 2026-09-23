"""MuSiQue substrate: real Wikipedia paragraphs laid out for this environment.

MuSiQue composes multi-hop questions from single-hop ones and annotates, per
hop, the sub-question, its answer and the paragraph that supports it. It was
built so that removing one hop makes the question unanswerable — the property
our single-chunk baseline exists to test — which is why it suits this
environment better than a retrieval benchmark.

Layout, mirroring the synthetic generator:

  * each supporting paragraph seeds its own chunk, so no two required facts can
    be read together;
  * every chunk is padded to the configured token size with irrelevant
    paragraphs — this row's own non-supporting ones first (the dataset's
    curated distractors), then paragraphs drawn from other rows;
  * filler-only chunks are added until the document reaches `doc_tokens`, and
    ALL chunks are listed identically in the map (collapsing filler the way the
    synthetic substrate collapses noise sections would mark the evidence);
  * chunk order is shuffled and rejected if two required facts land closer than
    `min_separation` tokens — rejection sampling, so placement stays uniform
    conditional on validity.

Recorded per instance: evidence location per hop, the intermediate hop answers
as known distractors (answering with one is the "stopped early" failure), the
dataset's own distractor titles, `min_steps` from the shared schedule, and the
answer with its aliases.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from ..config import EnvConfig, GeneratorConfig
from ..reward import normalize_answer
from ..schema import Chunk, Distractor, Evidence, Instance
from ..tokens import Tokenizer, get_tokenizer
from ..validate_instance import min_step_schedule

DEFAULT_HF_ID = "dgslibisey/MuSiQue"


class Skip(Exception):
    """This row cannot be laid out; try the next one."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_rows(split: str, hf_id: str = DEFAULT_HF_ID, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset  # optional dependency: pip install datasets

    ds = load_dataset(hf_id, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]


def usable(row: dict) -> bool:
    if not row.get("answerable", True) or not row.get("answer"):
        return False
    hops = row.get("question_decomposition") or []
    if not hops:
        return False
    by_idx = {p["idx"] for p in row["paragraphs"]}
    return all(h.get("paragraph_support_idx") in by_idx for h in hops)


def filler_pool(rows: list[dict]) -> list[dict]:
    """Every non-supporting paragraph in the corpus, deduplicated by title+text."""
    seen, pool = set(), []
    for r in rows:
        for p in r["paragraphs"]:
            if p.get("is_supporting"):
                continue
            key = (p["title"], p["paragraph_text"][:120])
            if key in seen:
                continue
            seen.add(key)
            pool.append({"title": p["title"], "text": p["paragraph_text"]})
    return pool


class TopicalPool:
    """Filler paragraphs, retrievable by similarity to a question.

    Padding a 2k-token MuSiQue instance out to 32k with paragraphs drawn at
    random from unrelated questions leaves the evidence as the only text that
    shares vocabulary with the question: measured question-overlap located the
    answer chunk with AUC 0.811, which is the single-chunk shortcut the
    environment exists to exclude. Padding with the MOST question-similar
    paragraphs available instead makes overlap uninformative by construction.

    An inverted index over content words, built once; retrieval is a counter
    over the question's words.
    """

    def __init__(self, paragraphs: list[dict], max_postings: int = 4000):
        self.paras = paragraphs
        self.index: dict[str, list[int]] = {}
        for i, p in enumerate(paragraphs):
            for w in _content_words(p["title"] + " " + p["text"]):
                postings = self.index.setdefault(w, [])
                if len(postings) < max_postings:
                    postings.append(i)

    def similar(self, question: str, k: int = 600) -> list[dict]:
        counts: Counter = Counter()
        for w in _content_words(question):
            for i in self.index.get(w, ()):
                counts[i] += 1
        return [self.paras[i] for i, _ in counts.most_common(k)]


_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOP = frozenset("""the of a an is was were are be been being what which who whom whose when where
why how to in on at by for from and or with that this these those it its他 his her their they them
he she we you i as but if then than there here also not no yes do does did done has have had
""".split())


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP}


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def render_chunk(paras: list[dict]) -> tuple[str, str]:
    """(header, body). The header lists the titles — this is the map's
    navigation cue, and the reason navigation here is semantic rather than a
    string-ordering puzzle."""
    header = "; ".join(p["title"] for p in paras)
    body = "\n\n".join(f"## {p['title']}\n{p['text']}" for p in paras)
    return header, header + "\n" + body


class MusiqueBuilder:
    def __init__(self, gen: GeneratorConfig, env: EnvConfig, pool: list[dict] | TopicalPool,
                 tokenizer: Tokenizer | None = None):
        self.gen = gen
        self.env = env
        self.topical = pool if isinstance(pool, TopicalPool) else TopicalPool(pool)
        self.pool = self.topical.paras
        self.fallbacks = 0          # rows whose topical retrieval had to be topped up
        self.tok = tokenizer or get_tokenizer(env.tokenizer)

    def size(self, paras: list[dict]) -> int:
        """Exact rendered size. Estimating it and rendering separately is how a
        chunk ends up over the ceiling after the fact."""
        return self.tok.count(render_chunk(paras)[1])

    def _group(self, items: list[tuple[dict, bool]], rng: random.Random) -> list[list[tuple[dict, bool]]] | None:
        """Group a shuffled paragraph list into chunks — ONE procedure for every
        chunk, so nothing about how a chunk was built can mark it.

        Two earlier layouts failed the generation-time audit for exactly this
        reason: evidence chunks were seeded with a supporting paragraph and
        padded from the row's own (shorter) distractors while filler chunks were
        built from the pool, which showed up as n_entries AUC 0.67 and length
        0.56. The only constraint kept here is that two supporting paragraphs
        never share a chunk; everything else is identical.
        """
        chunks, cur = [], []
        target = rng.randint(self.gen.chunk_min_tokens, self.gen.chunk_max_tokens)
        for para, sup in items:
            if sup and any(s for _, s in cur):
                return None                       # collision; caller reshuffles
            grown = self.size([p for p, _ in cur] + [para])
            if grown > self.gen.chunk_max_tokens:
                if self.size([p for p, _ in cur]) >= self.gen.chunk_min_tokens:
                    chunks.append(cur)
                    cur, target = [], rng.randint(self.gen.chunk_min_tokens, self.gen.chunk_max_tokens)
                    if sup or self.size([para]) <= self.gen.chunk_max_tokens:
                        cur = [(para, sup)]
                    continue
                if sup:
                    return None                   # cannot place this hop here
                continue                          # drop an unplaceable filler
            cur.append((para, sup))
            if grown >= target:
                chunks.append(cur)
                cur, target = [], rng.randint(self.gen.chunk_min_tokens, self.gen.chunk_max_tokens)
        if cur and self.size([p for p, _ in cur]) >= self.gen.chunk_min_tokens:
            chunks.append(cur)
        elif any(s for _, s in cur):
            return None                           # a hop landed in an undersized tail
        return chunks

    @staticmethod
    def _mentions(para: dict, needles: set[str]) -> bool:
        body = normalize_answer(para["title"] + " " + para["text"])
        return any(w in body for w in needles)

    def build(self, row: dict, doc_tokens: int, seed: int, split: str) -> Instance:
        rng = random.Random(seed)
        if not usable(row):
            raise Skip("unanswerable or missing hop support")
        hops = row["question_decomposition"]
        n_hops = len(hops)
        paras = {p["idx"]: p for p in row["paragraphs"]}
        answer = row["answer"]
        aliases = [a for a in (row.get("answer_aliases") or []) if a]
        support_idx = [h["paragraph_support_idx"] for h in hops]
        if len(set(support_idx)) != n_hops:
            raise Skip("two hops share a supporting paragraph")

        # Nothing outside the final supporting paragraph may state the answer,
        # and nothing outside hop i's paragraph may state hop i's answer.
        needles = {w for w in ({normalize_answer(answer)} |
                               {normalize_answer(h["answer"]) for h in hops}) if len(w) >= 4}

        own = [{"title": p["title"], "text": p["paragraph_text"]}
               for p in row["paragraphs"] if not p.get("is_supporting")]
        own = [p for p in own if not self._mentions(p, needles)]
        question_text = row["question"] + " " + " ".join(h["question"] for h in hops)
        fill = [p for p in self.topical.similar(question_text, k=1200)
                if not self._mentions(p, needles)]
        if len(fill) < 400:
            # Short or unusual questions retrieve few neighbours; top up at
            # random rather than skip the row. Reported per split so a substrate
            # that falls back constantly is visible rather than silent.
            seen = {(p["title"], p["text"][:60]) for p in fill}
            extra = [p for p in self.pool
                     if (p["title"], p["text"][:60]) not in seen and not self._mentions(p, needles)]
            rng.shuffle(extra)
            self.fallbacks += 1
            fill = fill + extra[: 400 - len(fill)]
        if len(fill) < 40:
            raise Skip("too little filler available")

        # One paragraph list: the hops, this row's own distractors, and enough
        # question-similar filler to reach the target size. Everything below
        # treats them identically.
        items: list[tuple[dict, bool]] = [({"title": paras[i]["title"],
                                            "text": paras[i]["paragraph_text"]}, True)
                                          for i in support_idx]
        items += [(p, False) for p in own]
        est = sum(self.size([p]) for p, _ in items)
        j = 0
        while est < doc_tokens and j < len(fill):
            items.append((fill[j], False))
            est += self.size([fill[j]])
            j += 1
        if est < doc_tokens * 0.8:
            raise Skip("not enough filler to reach the target document size")

        for _ in range(40):
            rng.shuffle(items)
            grouped = self._group(items, rng)
            if grouped is None:
                continue
            if sum(1 for c in grouped for _, sup in c if sup) != n_hops:
                continue
            sizes = [self.size([p for p, _ in c]) for c in grouped]
            offsets, off = [], 0
            for n in sizes:
                offsets.append(off)
                off += n
            pos_of_evidence = []
            ok = True
            for hop, i in enumerate(support_idx):
                title, text = paras[i]["title"], paras[i]["paragraph_text"]
                found = [k for k, c in enumerate(grouped)
                         if any(sup and p["title"] == title and p["text"] == text for p, sup in c)]
                if len(found) != 1:
                    ok = False
                    break
                pos_of_evidence.append(found[0])
            if not ok:
                continue
            ev_off = [offsets[k] for k in pos_of_evidence]
            if all(abs(a - b) >= self.gen.min_separation
                   for x, a in enumerate(ev_off) for b in ev_off[x + 1:]):
                break
        else:
            raise Skip("cannot lay out with the required separation")

        chunks = []
        for pos, group in enumerate(grouped):
            group_paras = [p for p, _ in group]
            header, text = render_chunk(group_paras)
            chunks.append(Chunk(idx=pos, section="Documents", header=header, kind="register",
                                text=text, tokens=sizes[pos],
                                entity_names=[p["title"] for p in group_paras]))
        evidence = []
        for hop, h in enumerate(hops):
            pos = pos_of_evidence[hop]
            evidence.append(Evidence(hop=hop, chunk_idx=pos, entity_type="paragraph",
                                     entity=paras[h["paragraph_support_idx"]]["title"],
                                     field=h["question"], value=h["answer"],
                                     token_offset=offsets[pos]))

        distractors = []
        for hop, h in enumerate(hops[:-1]):
            distractors.append(Distractor(kind="intermediate_answer", hop=hop,
                                          chunk_idx=pos_of_evidence[hop],
                                          entity=paras[h["paragraph_support_idx"]]["title"],
                                          value=h["answer"], wrong_answer=h["answer"]))
        for p in row["paragraphs"]:
            if not p.get("is_supporting"):
                distractors.append(Distractor(kind="dataset_distractor", hop=-1, chunk_idx=-1,
                                              entity=p["title"], value=p["title"], wrong_answer=None))

        ev_tokens = [chunks[k].tokens for k in pos_of_evidence]
        min_steps, comp_required, solvable = min_step_schedule(
            ev_tokens, self.env.context_ceiling, self.env.assumed_summary_tokens)
        if not solvable:
            raise Skip("unsolvable under the ceiling")

        return Instance(
            id=f"{split}-{row['id']}", seed=seed, split=split, n_hops=n_hops,
            doc_tokens_target=doc_tokens, doc_tokens=off, question=row["question"],
            answer=answer, answer_aliases=aliases, answer_type="answer",
            answer_entity_type="entity", substrate="musique",
            chunks=chunks, evidence=evidence, distractors=distractors,
            candidate_answers=[], min_reads=n_hops, min_steps=min_steps,
            compression_required=comp_required,
            generator=self.gen.model_dump(exclude={"splits"}) | {"tokenizer": self.tok.name,
                                                                "source": DEFAULT_HF_ID},
            env={"context_ceiling": self.env.context_ceiling,
                 "assumed_summary_tokens": self.env.assumed_summary_tokens,
                 "step_budget": self.env.step_budget(n_hops)},
        )


# --------------------------------------------------------------------------- #
# Validation (what survives without a mechanical parser)
# --------------------------------------------------------------------------- #
def validate_real(inst: Instance, gen: GeneratorConfig, env: EnvConfig) -> list[str]:
    """Checks that do not depend on being able to re-derive the answer from the
    text. Solvability is the dataset's annotation, not a proof — stated plainly
    because it is the one guarantee a real substrate cannot give."""
    reasons = []
    n = len(inst.chunks)
    for ev in inst.evidence:
        if not (0 <= ev.chunk_idx < n) or ev.entity not in inst.chunks[ev.chunk_idx].header:
            reasons.append(f"evidence_missing:hop{ev.hop}")
    idxs = inst.evidence_chunks
    if len(set(idxs)) != len(idxs):
        reasons.append("co_located")
    offs = [e.token_offset for e in inst.evidence]
    if any(abs(offs[i] - offs[j]) < gen.min_separation
           for i in range(len(offs)) for j in range(i + 1, len(offs))):
        reasons.append("separation")
    # the answer must be present where the last hop is, and nowhere else
    norm_answer = normalize_answer(inst.answer)
    holding = [c.idx for c in inst.chunks if norm_answer and norm_answer in normalize_answer(c.text)]
    if inst.answer_chunk not in holding:
        reasons.append("answer_not_in_final_chunk")
    elif len(holding) > 1:
        reasons.append(f"answer_also_elsewhere:{len(holding)}")
    for c in inst.chunks:
        if c.tokens > gen.chunk_max_tokens:
            reasons.append("chunk_overflow")
            break
    steps, comp, ok = min_step_schedule([inst.chunks[i].tokens for i in idxs],
                                        env.context_ceiling, env.assumed_summary_tokens)
    if not ok:
        reasons.append("unsolvable_under_ceiling")
    elif steps != inst.min_steps or comp != inst.compression_required:
        reasons.append("min_steps_mismatch")
    return reasons


def build_split(rows: list[dict], gen: GeneratorConfig, env: EnvConfig, *, split: str,
                n: int, doc_tokens: int, seed_base: int = 0, tokenizer: Tokenizer | None = None,
                pool: TopicalPool | None = None, progress=None) -> tuple[list[Instance], dict]:
    pool = pool or TopicalPool(filler_pool(rows))
    builder = MusiqueBuilder(gen, env, pool, tokenizer)
    out: list[Instance] = []
    skipped: Counter = Counter()
    rejected: Counter = Counter()
    consumed = 0
    for row in rows:
        if len(out) >= n:
            break
        consumed += 1
        try:
            inst = builder.build(row, doc_tokens, seed_base + consumed, split)
        except Skip as e:
            skipped[str(e)] += 1
            continue
        bad = validate_real(inst, gen, env)
        if bad:
            for b in bad:
                rejected[b.split(":")[0]] += 1
            continue
        out.append(inst)
        if progress and len(out) % 25 == 0:
            progress(len(out), n)
    stats = {
        "split": split, "substrate": "musique", "n": len(out), "rows_consumed": consumed,
        "topical_fallbacks": builder.fallbacks,
        "skipped": dict(skipped), "rejected_by_validation": dict(rejected),
        "rejection_rate": sum(rejected.values()) / max(consumed, 1),
        "n_hops": dict(Counter(x.n_hops for x in out)),
        "doc_tokens_mean": sum(x.doc_tokens for x in out) / max(len(out), 1),
        "compression_required_rate": sum(x.compression_required for x in out) / max(len(out), 1),
        "min_steps_mean": sum(x.min_steps for x in out) / max(len(out), 1),
    }
    return out, stats
