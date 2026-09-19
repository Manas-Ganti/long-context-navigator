from longctx.audit import run_audit
from longctx.baselines import mechanical_baselines
from longctx.policies import NoReadPolicy, OracleNavigator, RandomReader
from longctx.rollout import run_episodes
from longctx.teacher import leak_check, sample_teacher_trajectories, to_sft_rows


def test_mechanical_baselines_separate(instances):
    m = mechanical_baselines(instances)["overall"]
    assert m["no_read_fixed_guess"] == 0.0
    assert m["no_read_chance"] < 0.05
    assert m["single_chunk_oracle_expected"] < 0.35
    assert m["single_chunk_top_overlap_contains_answer"] == 0.0
    assert m["full_document_oracle"] == 1.0


def test_oracle_navigator_solves_everything_at_minimum(env_cfg, instances):
    trs = run_episodes(env_cfg, OracleNavigator(), instances)
    assert all(t.correct for t in trs)
    assert all(t.steps_used == t.min_steps for t in trs)
    assert all(t.n_reads == t.min_reads for t in trs)
    assert not any(t.ceiling_exceeded for t in trs)
    assert all(t.n_compress == t.n_hops - 1 for t in trs)   # ceiling < 2 chunks -> compress between reads


def test_guessers_fail(env_cfg, instances):
    for pol in (RandomReader(seed=1), NoReadPolicy()):
        trs = run_episodes(env_cfg, pol, instances)
        assert sum(t.correct for t in trs) / len(trs) < 0.1


def test_noisy_oracle_still_correct(env_cfg, instances):
    trs = run_episodes(env_cfg, OracleNavigator(noise_prob=0.3, seed=2), instances)
    assert all(t.correct for t in trs)


def test_audit_on_oracle_trajectories(env_cfg, instances):
    trs = run_episodes(env_cfg, OracleNavigator(noise_prob=0.3, seed=2), instances)
    rep = run_audit([t.to_dict() for t in trs], instances)
    assert rep["compression_noop"]["fact_retention_rate"] == 1.0
    assert rep["compression_noop"]["degenerate_summary_rate"] == 0.0
    assert rep["guess_without_reading"]["episodes_with_zero_reads"] == 0


def test_audit_detects_degenerate_compression_and_distractor_capture(env_cfg, instances):
    inst = instances[0]
    d = next(d for d in inst.distractors if d.wrong_answer)
    traj = {
        "instance_id": inst.id, "n_reads": 1, "correct": False, "answered": True, "ceiling_exceeded": False,
        "budget_exhausted": False, "answer": d.wrong_answer, "reads": [inst.evidence[0].chunk_idx] * 3,
        "min_reads": inst.min_reads,
        "steps": [{"kind": "COMPRESS", "ok": True, "summary": "x", "summary_sources": [str(inst.evidence[0].chunk_idx)]}],
    }
    rep = run_audit([traj], instances)
    assert rep["compression_noop"]["fact_retention_rate"] == 0.0
    assert rep["compression_noop"]["degenerate_summary_rate"] == 1.0
    assert rep["distractor_capture"]["distractor_capture_rate"] == 1.0
    assert rep["drop_reread_loops"]["episodes_with_reread_rate"] == 1.0


def test_teacher_keeps_leak_free_correct_trajectories(env_cfg, instances):
    kept, stats = sample_teacher_trajectories(env_cfg, instances[:12], per_instance=2, noise_prob=0.3)
    assert stats["rejected_leak"] == 0 and stats["rejected_incorrect"] == 0
    assert stats["kept"] >= 12
    rows = to_sft_rows(kept)
    assert rows and rows[0]["messages"][2]["role"] == "assistant"
    assert "ACTION:" in rows[0]["messages"][2]["content"]


def test_leak_filter_catches_privileged_write(env_cfg, instances):
    inst = instances[0]
    kept, _ = sample_teacher_trajectories(env_cfg, [inst], per_instance=1)
    t = kept[0]
    # the oracle's first step must not mention the answer, which is not visible yet
    t.steps[0]["response"] = f"THOUGHT: the answer is {inst.answer}.\nACTION: READ {inst.evidence[0].chunk_idx}"
    assert leak_check(t, inst)
    # reading the answer chunk before its key entity is visible is a leak
    kept, _ = sample_teacher_trajectories(env_cfg, [inst], per_instance=1)
    t = kept[0]
    t.steps[0]["kind"], t.steps[0]["ids"], t.steps[0]["ok"] = "READ", [str(inst.answer_chunk)], True
    if inst.n_hops >= 2:
        assert leak_check(t, inst)
