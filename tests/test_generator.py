import copy

import pytest

from longctx.config import EnvConfig, GeneratorConfig
from longctx.corpus import chain_types
from longctx.generate import InstanceGenerator
from longctx.schema import Instance
from longctx.validate_instance import min_step_schedule, solve_from_text, validate


def test_deterministic_under_seed(generator):
    a, _ = generator.generate_valid(77, 3, 32000, "t")
    b, _ = generator.generate_valid(77, 3, 32000, "t")
    assert a.model_dump() == b.model_dump()


def test_different_seeds_differ(generator):
    a, _ = generator.generate_valid(77, 3, 32000, "t")
    b, _ = generator.generate_valid(78, 3, 32000, "t")
    assert a.question != b.question or a.answer != b.answer


@pytest.mark.parametrize("n_hops", [1, 2, 3, 4, 5, 6, 7])
def test_hops_and_chain_types(generator, n_hops):
    inst, _ = generator.generate_valid(500 + n_hops, n_hops, 32000, "t")
    assert inst.n_hops == n_hops and len(inst.evidence) == n_hops
    assert [e.entity_type for e in inst.evidence] == chain_types(n_hops)
    assert inst.min_reads == n_hops
    assert validate(inst, generator.gen, generator.env) == []


def test_every_accepted_instance_validates(instances, gen_cfg, env_cfg):
    for inst in instances:
        assert validate(inst, gen_cfg, env_cfg) == []


def test_facts_in_distinct_chunks_and_separated(instances, gen_cfg):
    for inst in instances:
        idxs = inst.evidence_chunks
        assert len(set(idxs)) == len(idxs)
        offs = [e.token_offset for e in inst.evidence]
        for i in range(len(offs)):
            for j in range(i + 1, len(offs)):
                assert abs(offs[i] - offs[j]) >= gen_cfg.min_separation


def test_separation_violation_rejected(inst3, gen_cfg, env_cfg):
    strict = GeneratorConfig(**(gen_cfg.model_dump() | {"min_separation": 10 ** 9}))
    assert "separation" in validate(inst3, strict, env_cfg)


def test_unsolvable_instance_rejected(inst3, gen_cfg, env_cfg):
    broken = Instance(**inst3.model_dump())
    ev = broken.evidence[1]
    ch = broken.chunks[ev.chunk_idx]
    ch.text = ch.text.replace(ev.value, "REDACTED")
    reasons = validate(broken, gen_cfg, env_cfg)
    assert any(r.startswith("evidence_missing") for r in reasons) and "unreachable" in reasons


def test_answer_reachable_only_along_recorded_path(instances):
    for inst in instances:
        ans, path = solve_from_text(inst)
        assert ans == inst.answer and path == inst.evidence_chunks


def test_distractor_paths_never_reach_answer(instances):
    for inst in instances:
        for hop in range(inst.n_hops):
            a, _ = solve_from_text(inst, {hop: "prev"})
            assert a != inst.answer
            for d in inst.distractors:
                if d.kind == "near_miss_entity" and d.hop == hop:
                    a, _ = solve_from_text(inst, {hop: d.entity})
                    assert a != inst.answer
        assert all(d.wrong_answer != inst.answer for d in inst.distractors)


def test_distractors_present_and_typed(instances):
    for inst in instances:
        kinds = {d.kind for d in inst.distractors}
        assert "near_miss_entity" in kinds and "near_miss_value" in kinds
        for d in inst.distractors:
            assert d.entity in inst.chunks[d.chunk_idx].text


def test_answer_chunk_lexically_invisible(instances):
    for inst in instances:
        if inst.n_hops >= 2:
            assert inst.evidence[0].entity not in inst.chunks[inst.answer_chunk].text


def test_chunk_sizes_within_config(instances, gen_cfg):
    for inst in instances:
        for c in inst.chunks:
            assert gen_cfg.chunk_min_tokens - 40 <= c.tokens <= gen_cfg.chunk_max_tokens


def test_doc_tokens_scale(generator):
    small, _ = generator.generate_valid(9, 3, 32000, "t")
    big, _ = generator.generate_valid(9, 3, 128000, "t")
    assert 0.85 * 32000 <= small.doc_tokens <= 1.15 * 32000
    assert 0.85 * 128000 <= big.doc_tokens <= 1.15 * 128000
    assert len(big.chunks) > 3 * len(small.chunks)


def test_noise_ratio_respected(instances, gen_cfg):
    for inst in instances:
        noise = sum(c.kind == "noise" for c in inst.chunks) / len(inst.chunks)
        assert abs(noise - gen_cfg.noise_ratio) < 0.06


def test_compression_required_under_default_ceiling(instances):
    assert all(i.compression_required for i in instances)


def test_min_step_schedule():
    # three chunks that never fit pairwise under a 600 ceiling: R C R C R A = 6
    assert min_step_schedule([400, 400, 400], 600, 32) == (6, True, True)
    # two chunks that fit together: R R A = 3, no compression needed
    assert min_step_schedule([200, 200], 600, 32) == (3, False, True)
    # mixed: R R C R A = 5, compression required (400+300 > 600)
    assert min_step_schedule([200, 300, 400], 600, 32) == (5, True, True)
    # a chunk larger than the ceiling is unsolvable
    assert min_step_schedule([700], 600, 32)[2] is False
    assert min_step_schedule([100], 600, 32) == (2, False, True)


def test_min_steps_recorded_matches_schedule(instances, env_cfg):
    for inst in instances:
        toks = [inst.chunks[i].tokens for i in inst.evidence_chunks]
        steps, comp, ok = min_step_schedule(toks, env_cfg.context_ceiling, env_cfg.assumed_summary_tokens)
        assert ok and steps == inst.min_steps and comp == inst.compression_required


def test_candidate_answers_include_truth(instances):
    for inst in instances:
        assert inst.answer in inst.candidate_answers and len(inst.candidate_answers) > 20


def test_rejection_bookkeeping(generator):
    inst, rej = generator.generate_valid(31, 4, 32000, "t")
    assert inst.generator["attempts"] >= 1
    assert sum(rej.values()) == inst.generator["attempts"] - 1


def test_seed_recorded(instances):
    for inst in instances:
        assert inst.seed == int(inst.id.split("-")[-1])
