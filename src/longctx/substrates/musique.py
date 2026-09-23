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

import math
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
        self.df: Counter = Counter()
        for i, p in enumerate(paragraphs):
            for w in _content_words(p["title"] + " " + p["text"]):
                self.df[w] += 1
                postings = self.index.setdefault(w, [])
                if len(postings) < max_postings:
                    postings.append(i)
        self.n = max(len(paragraphs), 1)

    def idf(self, word: str) -> float:
        return math.log(self.n / (1 + self.df.get(word, 0)))

    def similar(self, question: str, k: int = 600) -> list[dict]:
        """Rank filler by IDF-weighted overlap with the question.

        Counting raw matches lets common words dominate, and the words that
        actually make an evidence paragraph distinctive are the rare ones — the
        question's entity names. Weighting by inverse document frequency picks
        filler that shares those, which is what drives question-overlap down
        (AUC 0.628 with raw counts).
        """
        scores: dict[int, float] = {}
        for w in _content_words(question):
            weight = self.idf(w)
            if weight <= 0:
                continue
            for i in self.index.get(w, ()):
                scores[i] = scores.get(i, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [self.paras[i] for i, _ in ranked]


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
                 tokenizer: Tokenizer | None = None,
                 para_tokens: tuple[int, int] = (40, 260)):
        self.gen = gen
        self.env = env
        # Filler paragraphs are length-matched to THIS ROW's supporting ones.
        #
        # MuSiQue's supporting paragraphs are shorter than the pool average, so
        # a chunk holding one needed MORE paragraphs to reach the size band:
        # n_entries AUC 0.587, and with more paragraphs came more distinct
        # vocabulary and a residual question-overlap signal (0.628). A single
        # global band does not fix it — it truncates both distributions but
        # leaves the supporting paragraphs at the short end of it (n_entries
        # rose to 0.73). Matching per document does: every paragraph in a
        # document is then drawn from the same narrow length window, so how
        # many fit in a chunk cannot depend on which kind it is.
        #
        # `para_tokens` stays as an outer sanity bound: a supporting paragraph
        # outside it could not share a chunk with anything.
        self.para_tokens = para_tokens
        self.band_slack = 20          # tokens either side of the row's own range
        self.topical = pool if isinstance(pool, TopicalPool) else TopicalPool(pool)
        self.pool = self.topical.paras
        self.fallbacks = 0          # rows whose topical retrieval had to be topped up
        self.tok = tokenizer or get_tokenizer(env.tokenizer)
        self._psize: dict[tuple[str, str], int] = {}

    def in_band(self, para: dict, band: tuple[int, int] | None = None) -> bool:
        lo, hi = band or self.para_tokens
        return lo <= self.psize(para) <= hi

    def row_band(self, sup_paras: list[dict]) -> tuple[int, int]:
        """The length window every paragraph in this document must fall in."""
        lens = [self.psize(p) for p in sup_paras]
        return (max(min(lens) - self.band_slack, 10), max(lens) + self.band_slack)

    def size(self, paras: list[dict]) -> int:
        """Exact rendered size. Estimating it and rendering separately is how a
        chunk ends up over the ceiling after the fact."""
        return self.tok.count(render_chunk(paras)[1])

    def psize(self, para: dict) -> int:
        """Cached per-paragraph cost, including its share of the header and the
        '## ' markers. Layout tries thousands of arrangements per document, so
        re-rendering a chunk for every trial placement is what made an earlier
        version quadratic; totals are tracked from these and the exact rendered
        size is verified once per accepted chunk."""
        key = (para["title"], para["text"])
        n = self._psize.get(key)
        if n is None:
            n = self.tok.count(para["title"]) * 2 + self.tok.count(para["text"]) + 4
            self._psize[key] = n
        return n

    def _group(self, items: list[tuple[dict, bool]], k: int,
               rng: random.Random) -> list[list[tuple[dict, bool]]] | None:
        """Assign paragraphs to chunks of exactly `k`, least-loaded chunk first.

        Three surface features could otherwise mark the evidence, and each was
        measured doing so on real data:

        * how a chunk was BUILT — evidence chunks were once seeded with a
          supporting paragraph and padded differently (n_entries AUC 0.587);
          here every chunk is built by this one procedure;
        * how MANY paragraphs it holds — a fill-to-target rule needed more of
          them around MuSiQue's shorter supporting paragraphs; `k` is fixed, so
          the count is constant by construction;
        * how LONG it is — with `k` fixed, a chunk holding a short supporting
          paragraph was simply shorter (length AUC 0.182). Placing each
          paragraph in the currently lightest chunk equalises totals, so size
          stops depending on contents.

        Supporting paragraphs are placed first, into distinct chunks. Returns
        None when the result falls outside the chunk band, and the caller
        reshuffles.
        """
        n_chunks = len(items) // k
        if n_chunks < len(items) and any(sup for _, sup in items[n_chunks * k:]):
            pass                                   # tail is discarded below; hops are placed first
        if n_chunks < 4:
            return None
        chunks: list[list[tuple[dict, bool]]] = [[] for _ in range(n_chunks)]
        sizes = [0] * n_chunks
        # Longest-processing-time first: largest paragraphs placed while every
        # chunk is still empty, smallest placed last into whichever chunk is
        # lightest — which is what actually equalises totals. Least-loaded over
        # a random order does not: the deficit left by a short supporting
        # paragraph survives it (length AUC stayed at 0.18).
        ordered = sorted(items, key=lambda it: -self.psize(it[0]))
        placed = 0
        for para, sup in ordered:
            if placed >= n_chunks * k:
                break
            cand = [i for i in range(n_chunks)
                    if len(chunks[i]) < k and not (sup and any(s for _, s in chunks[i]))]
            if not cand:
                return None
            lightest = min(sizes[i] for i in cand)
            pick = rng.choice([i for i in cand if sizes[i] == lightest])
            chunks[pick].append((para, sup))
            sizes[pick] += self.psize(para)
            placed += 1
        for c in chunks:
            rng.shuffle(c)                         # size order carries no information inside a chunk
        if any(len(c) != k for c in chunks):
            return None
        if any(not (self.gen.chunk_min_tokens <= self.size([p for p, _ in c]) <= self.gen.chunk_max_tokens)
               for c in chunks):
            return None
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

        # Only the FINAL answer has to be unique in the document. Excluding every
        # paragraph that mentions an intermediate answer as well removed the most
        # question-similar filler there is, which is what kept question-overlap
        # at 0.617; and a filler paragraph mentioning a bridge entity is exactly
        # the right kind of distractor — it names the entity without carrying the
        # link to the next hop, so finding the bridge stays a reading task.
        needles = {w for w in {normalize_answer(answer)} if len(w) >= 4}

        sup_paras = [{"title": paras[i]["title"], "text": paras[i]["paragraph_text"]}
                     for i in support_idx]
        if not all(self.in_band(p) for p in sup_paras):
            raise Skip("supporting paragraph outside the sanity bound")
        band = self.row_band(sup_paras)

        own = [{"title": p["title"], "text": p["paragraph_text"]}
               for p in row["paragraphs"] if not p.get("is_supporting")]
        own = [p for p in own if self.in_band(p, band) and not self._mentions(p, needles)]
        question_text = row["question"] + " " + " ".join(h["question"] for h in hops)
        fill = [p for p in self.topical.similar(question_text, k=6000)
                if self.in_band(p, band) and not self._mentions(p, needles)]
        if len(fill) < 400:
            # Short or unusual questions retrieve few neighbours; top up at
            # random rather than skip the row. Reported per split so a substrate
            # that falls back constantly is visible rather than silent.
            seen = {(p["title"], p["text"][:60]) for p in fill}
            extra = [p for p in self.pool
                     if (p["title"], p["text"][:60]) not in seen
                     and self.in_band(p, band) and not self._mentions(p, needles)]
            rng.shuffle(extra)
            self.fallbacks += 1
            fill = fill + extra[: 400 - len(fill)]
        if len(fill) < 40:
            raise Skip("too little filler available")

        # One paragraph list: the hops, this row's own distractors, and enough
        # question-similar filler to reach the target size. Everything below
        # treats them identically.
        items: list[tuple[dict, bool]] = [(p, True) for p in sup_paras]
        items += [(p, False) for p in own]
        est = sum(self.psize(p) for p, _ in items)
        j = 0
        while est < doc_tokens and j < len(fill):
            items.append((fill[j], False))
            est += self.psize(fill[j])
            j += 1
        if est < doc_tokens * 0.8:
            raise Skip("not enough filler to reach the target document size")

        # Paragraphs per chunk, from the document's own median paragraph length.
        median = sorted(self.psize(p) for p, _ in items)[len(items) // 2]
        k = max(2, round(((self.gen.chunk_min_tokens + self.gen.chunk_max_tokens) / 2) / max(median, 1)))
        if not (self.gen.chunk_min_tokens <= k * median <= self.gen.chunk_max_tokens):
            raise Skip("no paragraph count fits the chunk band for this document")

        # Assemble once (the expensive part), then search chunk ORDERS for one
        # that satisfies the separation constraint — reordering is free, and
        # redoing the whole assignment for every attempt was what made most
        # rows fail to lay out at all.
        grouped = None
        for _ in range(8):
            rng.shuffle(items)
            grouped = self._group(items, k, rng)
            if grouped is not None and sum(1 for c in grouped for _, sup in c if sup) == n_hops:
                break
            grouped = None
        if grouped is None:
            raise Skip("cannot assign paragraphs to chunks in this size band")

        by_chunk = {}
        for ci, c in enumerate(grouped):
            for p, sup in c:
                if sup:
                    by_chunk[(p["title"], p["text"])] = ci
        support_keys = [(paras[i]["title"], paras[i]["paragraph_text"]) for i in support_idx]
        if any(key not in by_chunk for key in support_keys):
            raise Skip("a hop was dropped during assignment")
        chunk_sizes = [self.size([p for p, _ in c]) for c in grouped]

        for _ in range(400):
            order = list(range(len(grouped)))
            rng.shuffle(order)
            place = {src: pos for pos, src in enumerate(order)}
            offsets, off = [], 0
            for src in order:
                offsets.append(off)
                off += chunk_sizes[src]
            ev_off = [offsets[place[by_chunk[key]]] for key in support_keys]
            if all(abs(a - b) >= self.gen.min_separation
                   for x, a in enumerate(ev_off) for b in ev_off[x + 1:]):
                break
        else:
            raise Skip("cannot lay out with the required separation")
        grouped = [grouped[src] for src in order]
        sizes = [chunk_sizes[src] for src in order]
        pos_of_evidence = [place[by_chunk[key]] for key in support_keys]

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
                pool: TopicalPool | None = None, para_tokens: tuple[int, int] = (40, 260),
                progress=None) -> tuple[list[Instance], dict]:
    pool = pool or TopicalPool(filler_pool(rows))
    builder = MusiqueBuilder(gen, env, pool, tokenizer, para_tokens=para_tokens)
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
        "topical_fallbacks": builder.fallbacks, "para_tokens": list(para_tokens),
        "skipped": dict(skipped), "rejected_by_validation": dict(rejected),
        "rejection_rate": sum(rejected.values()) / max(consumed, 1),
        "n_hops": dict(Counter(x.n_hops for x in out)),
        "doc_tokens_mean": sum(x.doc_tokens for x in out) / max(len(out), 1),
        "compression_required_rate": sum(x.compression_required for x in out) / max(len(out), 1),
        "min_steps_mean": sum(x.min_steps for x in out) / max(len(out), 1),
    }
    return out, stats
