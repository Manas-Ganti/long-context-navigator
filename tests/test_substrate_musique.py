"""The MuSiQue layout, exercised on synthetic rows in MuSiQue's schema.

Downloading the real dataset is not a unit test: these check the LAYOUT
invariants the environment depends on — one supporting paragraph per chunk,
separation, exact sizes, no answer leaking into filler, correct min_steps.
"""

import random

import pytest

from longctx.config import EnvConfig, GeneratorConfig
from longctx.env import NavigationEnv
from longctx.substrates.musique import (MusiqueBuilder, Skip, build_split, filler_pool,
                                        usable, validate_real)

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
         "pi rho sigma tau upsilon phi chi psi omega").split()


def _para(rng, idx, title, supporting=False, contains=None):
    """Length does not depend on whether the paragraph is supporting — the fact
    replaces filler words rather than being appended. A fixture where supporting
    paragraphs are systematically longer makes the per-document length band
    exclude the whole pool, which is a property of the fixture, not the code."""
    n = rng.randint(80, 110)
    words = [rng.choice(WORDS) for _ in range(n)]
    if contains:
        words[n // 2:n // 2 + 4] = ["the", "record", "states", contains]
    return {"idx": idx, "title": title, "paragraph_text": " ".join(words),
            "is_supporting": supporting}


def make_row(seed: int, n_hops: int = 2, n_paras: int = 20) -> dict:
    rng = random.Random(seed)
    answers = [f"Entity{seed}x{h}" for h in range(n_hops)]
    paras, hops = [], []
    support_positions = rng.sample(range(n_paras), n_hops)
    for i in range(n_paras):
        if i in support_positions:
            h = support_positions.index(i)
            paras.append(_para(rng, i, f"Topic{seed}_{h}", True, answers[h]))
        else:
            paras.append(_para(rng, i, f"Filler{seed}_{i}"))
    for h in range(n_hops):
        hops.append({"id": h, "question": f"sub-question {h}?", "answer": answers[h],
                     "paragraph_support_idx": support_positions[h]})
    return {"id": f"{n_hops}hop__{seed}", "paragraphs": paras, "question": f"question {seed}?",
            "question_decomposition": hops, "answer": answers[-1],
            "answer_aliases": [answers[-1].lower()], "answerable": True}


@pytest.fixture(scope="module")
def rows():
    return [make_row(s, n_hops=2 + s % 3) for s in range(120)]


@pytest.fixture(scope="module")
def built(rows, gen_cfg, env_cfg):
    insts, stats = build_split(rows, gen_cfg, env_cfg, split="t", n=40, doc_tokens=32000)
    return insts, stats


def test_builds_valid_instances(built, gen_cfg, env_cfg):
    insts, stats = built
    assert len(insts) >= 25, stats
    for inst in insts:
        assert validate_real(inst, gen_cfg, env_cfg) == []
        assert inst.substrate == "musique"


def test_one_supporting_paragraph_per_chunk(built):
    insts, _ = built
    for inst in insts:
        assert len(set(inst.evidence_chunks)) == inst.n_hops
        for ev in inst.evidence:
            assert ev.entity in inst.chunks[ev.chunk_idx].header


def test_chunk_sizes_exact_and_within_band(built, gen_cfg):
    from longctx.tokens import WhitespaceTokenizer
    tok = WhitespaceTokenizer()
    insts, _ = built
    for inst in insts:
        for c in inst.chunks:
            assert c.tokens == tok.count(c.text)
            assert gen_cfg.chunk_min_tokens <= c.tokens <= gen_cfg.chunk_max_tokens


def test_separation_enforced(built, gen_cfg):
    insts, _ = built
    for inst in insts:
        offs = [e.token_offset for e in inst.evidence]
        for i in range(len(offs)):
            for j in range(i + 1, len(offs)):
                assert abs(offs[i] - offs[j]) >= gen_cfg.min_separation


def test_answer_appears_only_in_the_final_evidence_chunk(built):
    from longctx.reward import normalize_answer
    insts, _ = built
    for inst in insts:
        holding = [c.idx for c in inst.chunks if normalize_answer(inst.answer) in normalize_answer(c.text)]
        assert holding == [inst.answer_chunk]


def test_intermediate_answers_recorded_as_distractors(built):
    insts, _ = built
    for inst in insts:
        inter = [d for d in inst.distractors if d.kind == "intermediate_answer"]
        assert len(inter) == inst.n_hops - 1
        for d in inter:
            assert d.wrong_answer == d.value != inst.answer


def test_compression_required_and_min_steps(built, env_cfg):
    from longctx.validate_instance import min_step_schedule
    insts, _ = built
    for inst in insts:
        toks = [inst.chunks[i].tokens for i in inst.evidence_chunks]
        steps, comp, ok = min_step_schedule(toks, env_cfg.context_ceiling, env_cfg.assumed_summary_tokens)
        assert ok and steps == inst.min_steps and comp == inst.compression_required
        assert inst.compression_required


def test_environment_runs_a_musique_instance(built, env_cfg):
    insts, _ = built
    inst = insts[0]
    env = NavigationEnv(env_cfg)
    obs = env.reset(inst)
    assert inst.evidence[0].entity in "\n".join(obs.map_lines)      # titles are the navigation cue
    a, b = inst.evidence[0].chunk_idx, inst.evidence[1].chunk_idx
    env.step(f"READ {a}")
    obs, r, done, info = env.step(f"READ {b}")
    assert done and info["ceiling_exceeded"]


def test_aliases_are_accepted_by_the_environment(built, env_cfg):
    insts, _ = built
    inst = next(i for i in insts if i.answer_aliases)
    env = NavigationEnv(env_cfg)
    env.reset(inst)
    _, r, done, info = env.step(f"ANSWER {inst.answer_aliases[0]}")
    assert done and info["correct"] and r > 0


def test_unusable_rows_are_skipped(rows, gen_cfg, env_cfg):
    bad = dict(rows[0])
    bad["answerable"] = False
    assert not usable(bad)
    builder = MusiqueBuilder(gen_cfg, env_cfg, filler_pool(rows))
    with pytest.raises(Skip):
        builder.build(bad, 32000, 0, "t")
