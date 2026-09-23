"""PURE reward. Mechanical scoring against generated ground truth. No model.

    if ceiling exceeded:        R = r_fail
    elif step budget exhausted: R = r_fail
    elif abstained (if enabled): R = r_abstain
    elif answer correct:        R = max(r_correct_min, 1 - beta * steps_used / min_steps)
    else:                       R = r_wrong

The efficiency term is normalised by the instance's own minimum step count,
which the generator computed, so hard instances are not punished for being hard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .config import RewardConfig

_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGIT_RUN = re.compile(r"\d+")


@dataclass(frozen=True)
class EpisodeOutcome:
    ceiling_exceeded: bool
    budget_exhausted: bool
    answered: bool
    answer: str | None
    ground_truth: str
    steps_used: int
    min_steps: int
    answer_aliases: tuple[str, ...] = ()


def normalize_answer(text: str | None) -> str:
    """Lowercase, keep alphanumerics only. '$4,885,866' -> '4885866';
    'PJ-4471' -> 'pj4471'; ' Dana Ruiz ' -> 'danaruiz'."""
    if text is None:
        return ""
    return _ALNUM.sub("", text.strip().lower())


def answer_matches(pred: str | None, truth: str, aliases: Sequence[str] = ()) -> bool:
    """Normalised exact match against the truth or any accepted alias, plus one
    mechanical leniency for numeric truths: a prediction containing exactly one
    digit run (commas stripped) equal to the truth's digits matches — so
    'USD 4,885,866' counts, '4,885,866 or 4,885,870' does not.

    Aliases come from the substrate, not from a model: the synthetic generator
    supplies none (its answers are exact codes and values), while a real
    dataset ships the accepted surface forms of the same entity. No judge is
    involved either way."""
    p = normalize_answer(pred)
    if not p:
        return False
    for candidate in (truth, *aliases):
        t = normalize_answer(candidate)
        if not t:
            continue
        if p == t:
            return True
        if t.isdigit():
            runs = _DIGIT_RUN.findall((pred or "").replace(",", ""))
            if len(runs) == 1 and runs[0] == t:
                return True
    return False


def is_abstention(pred: str | None, cfg: RewardConfig) -> bool:
    return cfg.allow_abstain and normalize_answer(pred) == normalize_answer(cfg.abstain_token)


def compute_reward(o: EpisodeOutcome, cfg: RewardConfig) -> float:
    if o.ceiling_exceeded:
        return cfg.r_fail
    if o.budget_exhausted:
        return cfg.r_fail
    if not o.answered:
        return cfg.r_fail
    if is_abstention(o.answer, cfg):
        return cfg.r_abstain
    if answer_matches(o.answer, o.ground_truth, o.answer_aliases):
        if o.min_steps <= 0:
            raise ValueError("min_steps must be positive")
        return max(cfg.r_correct_min, 1.0 - cfg.beta * (o.steps_used / o.min_steps))
    return cfg.r_wrong
