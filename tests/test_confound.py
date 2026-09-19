import numpy as np

from longctx.validate_instance import confound_audit, rank_auc, surface_features


def test_rank_auc_basics():
    assert rank_auc(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1])) == 1.0
    assert rank_auc(np.array([0.9, 0.1, 0.8, 0.2]), np.array([0, 1, 0, 1])) == 0.0
    assert rank_auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 1, 0, 1])) == 0.5
    assert rank_auc(np.array([1.0, 2.0]), np.array([0, 0])) == 0.5


def test_surface_features_exclude_noise(inst3):
    rows = surface_features(inst3)
    assert all(r["section"] != "noise" for r in rows)
    assert sum(r["is_answer"] for r in rows) == 1
    assert sum(r["is_evidence"] for r in rows) == inst3.n_hops


def test_surface_classifier_at_chance_on_accepted_set(generator, gen_cfg):
    insts = [generator.generate_valid(7000 + s, 2 + s % 4, 32000, "c")[0] for s in range(160)]
    rep = confound_audit(insts, gen_cfg.confound_tolerance)
    assert rep["passed"], rep["violations"]
    ans = rep["auc"]["answer_within_section_multihop"]
    for f in ("pos_global", "pos_in_section", "length", "n_entries", "overlap"):
        assert abs(ans[f] - 0.5) <= gen_cfg.confound_tolerance, (f, ans[f])


def test_confound_audit_catches_a_planted_bias(instances, gen_cfg):
    """Sanity: if the answer chunk were systematically the longest in its
    section, the audit must fail."""
    import copy
    biased = []
    for inst in instances:
        b = copy.deepcopy(inst)
        sec = b.chunks[b.answer_chunk].section
        top = max(c.tokens for c in b.chunks if c.section == sec)
        b.chunks[b.answer_chunk].tokens = top + 50
        biased.append(b)
    rep = confound_audit(biased, gen_cfg.confound_tolerance)
    assert not rep["passed"]
    assert any("length" in v for v in rep["violations"])
