"""Drive episodes through a policy, many at a time, and record trajectories.

Because the policy is stateless between steps, a batch of episodes can be
advanced in lockstep: one batched generate per round, every active episode
gets one action. That is what makes vLLM/HF batching pay off for eval and
for GRPO rollouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import EnvConfig
from .env import NavigationEnv
from .schema import Instance
from .tokens import Tokenizer


@dataclass
class Trajectory:
    instance_id: str
    split: str
    n_hops: int
    doc_tokens: int
    min_steps: int
    min_reads: int
    compression_required: bool
    policy: str
    answer: str | None
    ground_truth: str
    correct: bool
    reward: float
    answered: bool
    ceiling_exceeded: bool
    budget_exhausted: bool
    steps_used: int
    n_reads: int
    n_compress: int
    n_drop: int
    n_invalid: int
    reads: list[int]
    evidence_chunks: list[int]
    steps: list[dict] = field(default_factory=list)   # StepRecord dicts (+ prompt/response if recorded)

    def to_dict(self) -> dict:
        return asdict(self)


def run_episodes(env_cfg: EnvConfig, policy, instances: list[Instance], *, batch_size: int = 16,
                 record_prompts: bool = False, tokenizer: Tokenizer | None = None,
                 progress=None) -> list[Trajectory]:
    """Run every instance once. Returns trajectories in the input order."""
    results: list[Trajectory | None] = [None] * len(instances)
    queue = list(range(len(instances)))
    active: list[tuple[int, NavigationEnv, list[dict]]] = []
    done_count = 0

    def fill():
        while queue and len(active) < batch_size:
            i = queue.pop(0)
            env = NavigationEnv(env_cfg, tokenizer)
            env.reset(instances[i])
            active.append((i, env, []))

    fill()
    while active:
        obs_list = [env.observe() for _, env, _ in active]
        insts = [instances[i] for i, _, _ in active]
        responses = policy.act_batch(obs_list, insts)
        # a policy may expose per-response extras (token ids for GRPO)
        extras = getattr(policy, "last_step_extras", None)
        if not (isinstance(extras, list) and len(extras) == len(responses)):
            extras = [None] * len(responses)
        still = []
        for (i, env, log), obs, resp, extra in zip(active, obs_list, responses, extras):
            _, reward, done, info = env.step(resp)
            if record_prompts or extra:
                entry = {"prompt": obs.to_text(), "response": resp} if record_prompts else {"response": resp}
                if extra:
                    entry.update(extra)
                log.append(entry)
            if done:
                results[i] = _trajectory(instances[i], env, policy, reward, info, log)
                done_count += 1
                if progress:
                    progress(done_count, len(instances))
            else:
                still.append((i, env, log))
        active = still
        fill()
    return [r for r in results if r is not None]


def _trajectory(inst: Instance, env: NavigationEnv, policy, reward: float, info: dict, log: list[dict]) -> Trajectory:
    steps = [asdict(r) for r in env.records]
    for s, l in zip(steps, log):
        s.update(l)
    return Trajectory(
        instance_id=inst.id, split=inst.split, n_hops=inst.n_hops, doc_tokens=inst.doc_tokens,
        min_steps=inst.min_steps, min_reads=inst.min_reads, compression_required=inst.compression_required,
        policy=getattr(policy, "name", type(policy).__name__), answer=info["answer"], ground_truth=inst.answer,
        correct=info["correct"], reward=reward, answered=info["answered"],
        ceiling_exceeded=info["ceiling_exceeded"], budget_exhausted=info["budget_exhausted"],
        steps_used=info["steps_used"], n_reads=info["n_reads"], n_compress=info["n_compress"],
        n_drop=info["n_drop"], n_invalid=info["n_invalid"], reads=info["reads"],
        evidence_chunks=inst.evidence_chunks, steps=steps)
