import pytest

from longctx.config import EnvConfig
from longctx.env import NavigationEnv
from longctx.tokens import WhitespaceTokenizer


def make_env(env_cfg, **over):
    return NavigationEnv(EnvConfig(**(env_cfg.model_dump() | over)))


def test_read_accounting_exact(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    obs, r, done, info = env.step(f"ACTION: READ {idx}")
    assert not done and r == 0.0
    assert obs.used == inst3.chunks[idx].tokens == info["used_tokens"]
    assert obs.read == [idx] and obs.steps_used == 1


def test_second_chunk_exceeds_ceiling_terminates_with_failure(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    a, b = inst3.evidence[0].chunk_idx, inst3.evidence[1].chunk_idx
    assert inst3.chunks[a].tokens + inst3.chunks[b].tokens > env_cfg.context_ceiling
    env.step(f"READ {a}")
    obs, r, done, info = env.step(f"READ {b}")
    assert done and info["ceiling_exceeded"] and r == env_cfg.reward.r_fail
    # the offending chunk was NOT loaded
    assert obs.used == inst3.chunks[a].tokens


def test_compress_drop_token_math(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    tok = WhitespaceTokenizer()
    idx = inst3.evidence[0].chunk_idx
    env.step(f"READ {idx}")
    summary = "Amber Falcon: Lead = Dana Ruiz"
    obs, _, _, _ = env.step(f"COMPRESS {idx} :: {summary}")
    assert obs.used == tok.count(summary)
    assert [h.id for h in obs.held] == ["S1"] and obs.held[0].sources == [str(idx)]
    assert obs.dropped == [idx]
    obs, _, _, _ = env.step("DROP S1")
    assert obs.used == 0 and obs.held == []


def test_summary_truncated_to_max_summary_tokens(env_cfg, inst3):
    env = make_env(env_cfg, max_summary_tokens=5)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    env.step(f"READ {idx}")
    obs, _, done, _ = env.step(f"ACTION: COMPRESS {idx} :: " + inst3.chunks[idx].text)
    assert not done and obs.used == 5


def test_compress_cannot_smuggle_chunk_past_ceiling(env_cfg, inst3):
    """A 'summary' that is a copy of the chunk is truncated first, so usage
    can only go down; but a summary that still does not fit is a violation."""
    env = make_env(env_cfg, max_summary_tokens=10_000, context_ceiling=env_cfg.context_ceiling)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    env.step(f"READ {idx}")
    big = " ".join(["word"] * (env_cfg.context_ceiling + 1))
    obs, r, done, info = env.step(f"COMPRESS {idx} :: {big}")
    assert done and info["ceiling_exceeded"] and r == env_cfg.reward.r_fail


def test_invalid_action_costs_a_step_without_state_change(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    obs, r, done, _ = env.step("ACTION: JUMP 3")
    assert obs.steps_used == 1 and obs.used == 0 and not done
    assert "ERROR" in obs.last_action_result


def test_drop_or_compress_unheld_is_error(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    obs, _, _, info = env.step("DROP 3")
    assert "not held" in obs.last_action_result and obs.steps_used == 1
    obs, _, _, _ = env.step("COMPRESS 3 :: x")
    assert "not held" in obs.last_action_result


def test_reread_after_drop_allowed_and_logged(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    env.step(f"READ {idx}")
    env.step(f"DROP {idx}")
    obs, _, done, info = env.step(f"READ {idx}")
    assert not done and obs.read == [idx, idx] and obs.used == inst3.chunks[idx].tokens


def test_reading_held_chunk_is_error(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    env.step(f"READ {idx}")
    obs, _, _, _ = env.step(f"READ {idx}")
    assert "already held" in obs.last_action_result and obs.used == inst3.chunks[idx].tokens


def test_step_budget_exhaustion_is_failure(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    budget = env_cfg.step_budget(inst3.n_hops)
    done = False
    for _ in range(budget):
        assert not done
        _, r, done, info = env.step("ACTION: JUMP")
    assert done and info["budget_exhausted"] and r == env_cfg.reward.r_fail


def test_answer_on_last_step_is_not_exhaustion(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    budget = env_cfg.step_budget(inst3.n_hops)
    for _ in range(budget - 1):
        env.step("ACTION: JUMP")
    _, r, done, info = env.step(f"ANSWER {inst3.answer}")
    assert done and info["correct"] and not info["budget_exhausted"] and r > 0


def test_correct_answer_reward_uses_instance_minimum(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    _, r, done, info = env.step(f"ANSWER {inst3.answer}")
    assert done and info["correct"]
    assert r == pytest.approx(1.0 - env_cfg.reward.beta * (1 / inst3.min_steps))


def test_used_tokens_always_sum_of_held(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    for a in [f"READ {idx}", f"COMPRESS {idx} :: a b c", "READ 0", "DROP 0", "DROP S1"]:
        obs, _, done, _ = env.step(a)
        assert obs.used == sum(h.tokens for h in obs.held)
        if done:
            break


def test_step_after_done_raises(env_cfg, inst3):
    env = make_env(env_cfg)
    env.reset(inst3)
    env.step("ANSWER x")
    with pytest.raises(RuntimeError):
        env.step("READ 0")


def test_map_covers_every_chunk(env_cfg, inst3):
    env = make_env(env_cfg)
    obs = env.reset(inst3)
    text = "\n".join(obs.map_lines)
    for c in inst3.chunks:
        if c.kind == "register":
            assert f"[{c.idx}] {c.header}" in text


def test_duplicate_ids_do_not_crash(env_cfg, inst3):
    """'DROP 75, 75' raised KeyError inside a GRPO rollout and killed the rank."""
    env = make_env(env_cfg)
    env.reset(inst3)
    idx = inst3.evidence[0].chunk_idx
    env.step(f"READ {idx}")
    obs, _, done, _ = env.step(f"ACTION: COMPRESS {idx}, {idx} :: fact")
    assert not done and [h.id for h in obs.held] == ["S1"] and obs.held[0].sources == [str(idx)]
    env.step("READ 0")
    obs, _, done, _ = env.step("DROP 0 0 S1 S1")
    assert not done and obs.held == [] and obs.used == 0
