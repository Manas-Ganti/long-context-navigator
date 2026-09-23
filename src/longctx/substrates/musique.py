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
        self.tok = tokenizer or get_tokenizer(env.tokenizer)

    def size(self, paras: list[dict]) -> int:
        """Exact rendered size. Estimating it and rendering separately is how a
        chunk ends up over the ceiling after the fact."""
        return self.tok.count(render_chunk(paras)[1])

    def _fill(self, seed: list[dict], fillers: list[dict], rng: random.Random,
              prefer: list[dict] | None = None) -> list[dict] | None:
        """Grow `seed` until the rendered chunk lands inside the configured size
        band. `prefer` is consumed first — that is where this row's OWN
        distractor paragraphs go.

        Why it matters: MuSiQue ships ~17 hard distractors selected FOR this
        question. A first version mixed them into a pool of ~100k paragraphs
        from other questions and sampled uniformly, so every chunk but the
        evidence ones was off-topic and question-overlap located the answer
        chunk with AUC 0.811 — the single-chunk shortcut this environment
        exists to exclude. On-topic padding is what makes overlap uninformative.
        """
        paras = list(seed)
        total = self.size(paras)
        if total > self.gen.chunk_max_tokens:
            return None
        queue = list(prefer or [])
        rng.shuffle(queue)
        target = rng.randint(self.gen.chunk_min_tokens, self.gen.chunk_max_tokens)
        for _ in range(200):
            if total >= target:
                break
            cand = queue.pop(0) if queue else rng.choice(fillers)
            grown = self.size(paras + [cand])
            if grown > self.gen.chunk_max_tokens:
                if queue:
                    continue
                break
            paras.append(cand)
            total = grown
        if not (self.gen.chunk_min_tokens <= total <= self.gen.chunk_max_tokens):
            return None
        rng.shuffle(paras)          # the supporting paragraph is not always first
        return paras

    def build(self, row: dict, doc_tokens: int, seed: int, split: str) -> Instance:
        rng = random.Random(seed)
        if not usable(row):
            raise Skip("unanswerable or missing hop support")
        hops = row["question_decomposition"]
        n_hops = len(hops)
        paras = {p["idx"]: p for p in row["paragraphs"]}
        answer = row["answer"]
        aliases = [a for a in (row.get("answer_aliases") or []) if a]

        own = [{"title": p["title"], "text": p["paragraph_text"]}
               for p in row["paragraphs"] if not p.get("is_supporting")]
        if len(self.pool) < 50:
            raise Skip("filler pool too small")
        # Filler is drawn from the paragraphs most similar to THIS question, so
        # that question-overlap does not mark the evidence (see TopicalPool).
        fill = self.topical.similar(row["question"] + " " +
                                    " ".join(h["question"] for h in hops))
        if len(fill) < 60:
            fill = self.pool

        # -- share this row's own (on-topic) distractors between the evidence
        #    chunks and an equal number of decoy chunks, so "reads like the
        #    question" marks a chunk as on-topic, never as evidence ---------- #
        support_idx = [h["paragraph_support_idx"] for h in hops]
        if len(set(support_idx)) != n_hops:
            raise Skip("two hops share a supporting paragraph")
        n_decoy = max(n_hops, 2)
        rng.shuffle(own)
        groups: list[list[dict]] = [[] for _ in range(n_hops + n_decoy)]
        for j, p in enumerate(own):
            groups[j % len(groups)].append(p)
        if min(len(g) for g in groups) == 0:
            raise Skip("too few on-topic distractors to pad with")

        chunks_paras: list[list[dict]] = []          # index 0..n_hops-1 are the evidence chunks
        for hop, i in enumerate(support_idx):
            sp = paras[i]
            got = self._fill([{"title": sp["title"], "text": sp["paragraph_text"]}],
                             fill, rng, prefer=groups[hop])
            if got is None:
                raise Skip("supporting paragraph does not fit a chunk")
            chunks_paras.append(got)
        decoys = []
        for d in range(n_decoy):
            g = groups[n_hops + d]
            got = self._fill([g[0]], fill, rng, prefer=g[1:])
            if got is not None:
                decoys.append(got)
        if len(decoys) < min(n_decoy, 2):
            raise Skip("could not build on-topic decoy chunks")
        chunks_paras.extend(decoys)

        # -- the rest is off-topic filler, up to the target document size ---- #
        avg = (self.gen.chunk_min_tokens + self.gen.chunk_max_tokens) / 2
        want = max(int(round(doc_tokens / avg)), n_hops + n_decoy + 4)
        guard = 0
        while len(chunks_paras) < want and guard < want * 20:
            guard += 1
            got = self._fill([rng.choice(fill)], fill, rng)
            if got is not None:
                chunks_paras.append(got)
        if len(chunks_paras) < want * 0.8:
            raise Skip("could not reach the target document size")

        # -- no filler chunk may contain the answer or an intermediate answer,
        #    which a borrowed paragraph can do by coincidence ---------------- #
        wanted = {w for w in ({normalize_answer(answer)} |
                              {normalize_answer(h["answer"]) for h in hops}) if len(w) >= 4}
        keep = list(range(n_hops))          # evidence chunks are never dropped
        for j in range(n_hops, len(chunks_paras)):
            body = normalize_answer(" ".join(p["text"] for p in chunks_paras[j]))
            if not any(w in body for w in wanted):
                keep.append(j)
        chunks_paras = [chunks_paras[j] for j in keep]
        if len(chunks_paras) < want * 0.8:
            raise Skip("too few filler chunks survived the answer-leak filter")

        # -- shuffle until the separation constraint holds. Rejection sampling,
        #    so placement stays uniform conditional on validity -------------- #
        sizes_by_chunk = [self.size(c) for c in chunks_paras]
        for _ in range(80):
            order = list(range(len(chunks_paras)))
            rng.shuffle(order)
            position = {src: pos for pos, src in enumerate(order)}
            offsets, off = [], 0
            for src in order:
                offsets.append(off)
                off += sizes_by_chunk[src]
            ev_offsets = [offsets[position[h]] for h in range(n_hops)]
            if all(abs(a - b) >= self.gen.min_separation
                   for i, a in enumerate(ev_offsets) for b in ev_offsets[i + 1:]):
                break
        else:
            raise Skip("cannot separate the required facts")
        pos_of_evidence = [position[h] for h in range(n_hops)]

        chunks, evidence = [], []
        for pos, src in enumerate(order):
            header, text = render_chunk(chunks_paras[src])
            chunks.append(Chunk(idx=pos, section="Documents", header=header, kind="register",
                                text=text, tokens=sizes_by_chunk[src],
                                entity_names=[p["title"] for p in chunks_paras[src]]))
        for hop, h in enumerate(hops):
            pos = pos_of_evidence[hop]
            sp = paras[h["paragraph_support_idx"]]
            evidence.append(Evidence(hop=hop, chunk_idx=pos, entity_type="paragraph",
                                     entity=sp["title"], field=h["question"], value=h["answer"],
                                     token_offset=offsets[pos]))

        # -- distractors ------------------------------------------------------ #
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

        ev_tokens = [chunks[p].tokens for p in pos_of_evidence]
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
        "skipped": dict(skipped), "rejected_by_validation": dict(rejected),
        "rejection_rate": sum(rejected.values()) / max(consumed, 1),
        "n_hops": dict(Counter(x.n_hops for x in out)),
        "doc_tokens_mean": sum(x.doc_tokens for x in out) / max(len(out), 1),
        "compression_required_rate": sum(x.compression_required for x in out) / max(len(out), 1),
        "min_steps_mean": sum(x.min_steps for x in out) / max(len(out), 1),
    }
    return out, stats
