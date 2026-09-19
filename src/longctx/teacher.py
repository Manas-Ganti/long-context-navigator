"""Cold start: rejection-sampled trajectories from the privileged oracle,
filtered for leak-freedom, exported as per-step SFT rows.

A teacher trajectory is kept only if
  * it is correct and within `max_step_ratio` x the instance minimum, and
  * every step passes the leak filter: everything the teacher WROTE (names,
    values in THOUGHT and summaries) appears in that step's observation, and
    every READ of an evidence chunk was made while the entity it holds was
    visible (in the question or in the working context). Privilege may choose
    among justifiable actions; it may not reveal anything.
"""

from __future__ import annotations

import random
import re

from .config import EnvConfig
from .policies import OracleNavigator
from .prompts import SYSTEM_PROMPT
from .rollout import Trajectory, run_episodes
from .schema import Instance

_VALUE_LIKE = re.compile(r"\b(?:[A-Z]{2}-)?\d[\d,]{2,}\b")


def without_map(prompt: str) -> str:
    a, b = prompt.find("DOCUMENT MAP"), prompt.find("MEMORY LEDGER")
    return prompt[:a] + prompt[b:] if 0 <= a < b else prompt


def leak_check(traj: Trajectory, inst: Instance) -> list[str]:
    problems = []
    all_names = {nm for c in inst.chunks for nm in c.entity_names}
    ev_by_chunk = {ev.chunk_idx: ev for ev in inst.evidence}
    for s in traj.steps:
        prompt, resp = s.get("prompt", ""), s.get("response", "")
        # names / values written must be visible
        for nm in all_names:
            if nm in resp and nm not in prompt:
                problems.append(f"step {s['step']}: wrote unseen name {nm!r}")
        for v in _VALUE_LIKE.findall(resp):
            if v not in prompt:
                problems.append(f"step {s['step']}: wrote unseen value {v!r}")
        # reading an evidence chunk requires its key entity to be visible in the
        # question or the working context — the map's range labels do not count
        if s.get("kind") == "READ" and s.get("ok"):
            idx = int(s["ids"][0])
            if idx in ev_by_chunk and ev_by_chunk[idx].entity not in without_map(prompt):
                problems.append(f"step {s['step']}: read chunk {idx} without seeing {ev_by_chunk[idx].entity!r}")
    return problems


def sample_teacher_trajectories(env_cfg: EnvConfig, instances: list[Instance], *, per_instance: int = 2,
                                noise_prob: float = 0.15, max_step_ratio: float = 1.5, seed: int = 0,
                                progress=None) -> tuple[list[Trajectory], dict]:
    kept: list[Trajectory] = []
    stats = {"sampled": 0, "rejected_incorrect": 0, "rejected_too_long": 0, "rejected_leak": 0,
             "leak_examples": []}
    for k in range(per_instance):
        pol = OracleNavigator(noise_prob=noise_prob if k > 0 else 0.0, seed=seed + k)
        trajs = run_episodes(env_cfg, pol, instances, record_prompts=True, progress=progress)
        for t, inst in zip(trajs, instances):
            stats["sampled"] += 1
            if not t.correct:
                stats["rejected_incorrect"] += 1
                continue
            if t.steps_used > max_step_ratio * t.min_steps:
                stats["rejected_too_long"] += 1
                continue
            leaks = leak_check(t, inst)
            if leaks:
                stats["rejected_leak"] += 1
                if len(stats["leak_examples"]) < 5:
                    stats["leak_examples"].append(leaks[0])
                continue
            kept.append(t)
    stats["kept"] = len(kept)
    stats["keep_rate"] = len(kept) / max(stats["sampled"], 1)
    return kept, stats


def to_sft_rows(trajs: list[Trajectory]) -> list[dict]:
    rows = []
    for t in trajs:
        for s in t.steps:
            rows.append({"instance_id": t.instance_id, "step": s["step"], "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s["response"]},
            ]})
    return rows
