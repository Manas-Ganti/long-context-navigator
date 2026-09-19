import pytest

from longctx.config import RewardConfig
from longctx.reward import EpisodeOutcome, answer_matches, compute_reward, normalize_answer


def outcome(**kw):
    base = dict(ceiling_exceeded=False, budget_exhausted=False, answered=True, answer="4,885,866",
                ground_truth="4,885,866", steps_used=6, min_steps=6)
    base.update(kw)
    return EpisodeOutcome(**base)


CFG = RewardConfig(beta=0.05, r_correct_min=0.5, r_fail=-0.5, r_wrong=0.0)


def test_ceiling_violation_is_failure_even_if_answer_present():
    assert compute_reward(outcome(ceiling_exceeded=True), CFG) == CFG.r_fail


def test_budget_exhausted_is_failure():
    assert compute_reward(outcome(budget_exhausted=True, answered=False, answer=None), CFG) == CFG.r_fail


def test_no_answer_is_failure():
    assert compute_reward(outcome(answered=False, answer=None), CFG) == CFG.r_fail


def test_wrong_answer():
    assert compute_reward(outcome(answer="4,885,870"), CFG) == CFG.r_wrong


def test_correct_at_minimum_steps():
    assert compute_reward(outcome(), CFG) == pytest.approx(1.0 - 0.05)


def test_correct_efficiency_normalised_per_instance():
    assert compute_reward(outcome(steps_used=12, min_steps=6), CFG) == pytest.approx(1.0 - 0.10)
    assert compute_reward(outcome(steps_used=12, min_steps=12), CFG) == pytest.approx(1.0 - 0.05)


def test_correct_reward_floor():
    cfg = RewardConfig(beta=0.5, r_correct_min=0.5)
    assert compute_reward(outcome(steps_used=30, min_steps=6), cfg) == 0.5


def test_failure_below_wrong():
    assert CFG.r_fail < CFG.r_wrong < compute_reward(outcome(), CFG)


def test_abstain_disabled_scores_as_wrong():
    assert compute_reward(outcome(answer="unknown"), CFG) == CFG.r_wrong


def test_abstain_enabled_scores_between_wrong_and_correct():
    cfg = RewardConfig(allow_abstain=True, r_abstain=0.1)
    r = compute_reward(outcome(answer="unknown"), cfg)
    assert cfg.r_wrong < r < compute_reward(outcome(), cfg)
    assert r == 0.1


def test_abstain_config_validation():
    with pytest.raises(ValueError):
        RewardConfig(allow_abstain=True, r_abstain=-0.1)


def test_min_steps_zero_raises():
    with pytest.raises(ValueError):
        compute_reward(outcome(min_steps=0), CFG)


@pytest.mark.parametrize("pred,truth,ok", [
    ("4,885,866", "4,885,866", True),
    ("$4,885,866", "4,885,866", True),
    ("USD 4885866", "4,885,866", True),
    ("4885866.", "4,885,866", True),
    ("4,885,866 or 4,885,870", "4,885,866", False),
    ("4,885,870", "4,885,866", False),
    ("pj-4471", "PJ-4471", True),
    ("PJ 4471", "PJ-4471", True),
    ("4471", "PJ-4471", False),
    ("Dana  Ruiz", "Dana Ruiz", True),
    ("", "PJ-4471", False),
    (None, "PJ-4471", False),
])
def test_answer_matching(pred, truth, ok):
    assert answer_matches(pred, truth) is ok


def test_normalize():
    assert normalize_answer(" $4,885,866 ") == "4885866"
    assert normalize_answer(None) == ""
