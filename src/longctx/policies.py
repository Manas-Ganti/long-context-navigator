"""Policies. A policy maps a batch of observations to a batch of response
texts. Mechanical policies live here (no model); LLM backends are in llm.py.

`OracleNavigator` is privileged — it knows the evidence locations — and is
used for (a) validating the environment end to end, (b) the efficiency
ceiling, and (c) cold-start trajectories for SFT. Everything it WRITES is
derived from the observation (see teacher.py for the leak filter), only its
CHOICES are informed by privilege.
"""

from __future__ import annotations

import random
import re
from typing import Protocol

from .corpus import TYPE_SPECS, chain_types, parse_chunk_entries
from .env import Observation
from .schema import Instance


class Policy(Protocol):
    name: str
    privileged: bool

    def act_batch(self, obs_list: list[Observation], insts: list[Instance]) -> list[str]: ...


class MechanicalPolicy:
    name = "mechanical"
    privileged = False

    def act(self, obs: Observation, inst: Instance) -> str:
        raise NotImplementedError

    def act_batch(self, obs_list, insts):
        return [self.act(o, i) for o, i in zip(obs_list, insts)]


# --------------------------------------------------------------------------- #
# Helpers shared by the oracle and the leak filter
# --------------------------------------------------------------------------- #
def visible_text(obs: Observation) -> str:
    parts = [obs.question, "\n".join(obs.map_lines)] + [h.text for h in obs.held]
    return "\n".join(parts)


def known_hops(obs: Observation, inst: Instance) -> list[bool]:
    """Which chain values are recoverable from the working context: hop h is
    known if its chunk is held or a summary literally contains its value."""
    held_chunks = {h.id for h in obs.held if h.kind == "chunk"}
    summaries = " ".join(h.text for h in obs.held if h.kind == "summary")
    out = []
    for ev in inst.evidence:
        out.append(str(ev.chunk_idx) in held_chunks or ev.value in summaries)
    return out


def fact_line(inst: Instance, hop: int) -> str:
    ev = inst.evidence[hop]
    return f"{ev.entity}: {ev.field} = {ev.value}"


class OracleNavigator(MechanicalPolicy):
    """Optimal-in-steps schedule: READ the next needed chunk; when it does not
    fit, COMPRESS everything held into the known facts; ANSWER once the last
    entry is held. With `noise_prob` > 0 it occasionally takes a valid but
    suboptimal action (a wrong-but-plausible read in the right section, a
    DROP of a redundant item), which gives SFT some recovery behaviour."""

    name = "oracle"
    privileged = True

    def __init__(self, noise_prob: float = 0.0, seed: int = 0):
        self.noise_prob = noise_prob
        self.rng = random.Random(seed)

    def act(self, obs: Observation, inst: Instance) -> str:
        known = known_hops(obs, inst)
        n = inst.n_hops
        held_ids = [h.id for h in obs.held]
        free = obs.ceiling - obs.used

        if all(known) and any(h.kind == "chunk" and h.id == str(inst.answer_chunk) for h in obs.held):
            ev = inst.evidence[-1]
            return f"THOUGHT: The entry for {ev.entity} gives {ev.field}: {ev.value}.\nACTION: ANSWER {ev.value}"
        if all(known):  # answer only in a summary — still answerable
            ev = inst.evidence[-1]
            return f"THOUGHT: My summary records {ev.field} of {ev.entity} as {ev.value}.\nACTION: ANSWER {ev.value}"

        # the first hop whose value is unknown; its entity name is known (hop-1) or in the question
        j = known.index(False)
        ev = inst.evidence[j]
        target = inst.chunks[ev.chunk_idx]
        redundant = [h.id for h in obs.held if h.kind == "chunk" and int(h.id) not in inst.evidence_chunks[:j]
                     and int(h.id) != ev.chunk_idx] + [h.id for h in obs.held if h.kind == "chunk"
                                                       and int(h.id) in inst.evidence_chunks[:max(j - 1, 0)]]

        # optional noise: a plausible detour
        if self.noise_prob and self.rng.random() < self.noise_prob:
            same_section = [c for c in inst.chunks if c.section == target.section and c.idx != target.idx
                            and str(c.idx) not in held_ids and c.tokens <= free]
            if redundant and self.rng.random() < 0.5:
                d = redundant[0]
                return f"THOUGHT: Chunk {d} is no longer needed; free it.\nACTION: DROP {d}"
            if same_section:
                c = self.rng.choice(same_section)
                return (f"THOUGHT: I need {ev.entity} in {target.section}; chunk {c.idx} ({c.header}) may cover it.\n"
                        f"ACTION: READ {c.idx}")

        if target.tokens <= free:
            return (f"THOUGHT: I need the {ev.field.lower()} of {ev.entity}; the map places it in chunk "
                    f"{target.idx} ({target.header}), which fits.\nACTION: READ {target.idx}")

        # does not fit: compress everything held into the facts known so far
        facts = "; ".join(fact_line(inst, h) for h in range(j) if known[h])
        if not facts:
            # nothing worth keeping (e.g. only detour chunks are held): drop them
            return f"THOUGHT: Nothing held is useful; free it all.\nACTION: DROP {', '.join(held_ids)}"
        return (f"THOUGHT: Chunk {target.idx} ({target.tokens} tokens) does not fit beside what I hold; "
                f"I keep only the facts I need.\nACTION: COMPRESS {', '.join(held_ids)} :: {facts}")


class RandomReader(MechanicalPolicy):
    """A guesser that does not understand the task: reads random register
    chunks (freeing everything when the next one would not fit), and after a
    few reads answers with a random value of the answer's type found in what it
    holds. A reference for 'find a lucky chunk and guess'."""

    name = "random-reader"
    privileged = False

    def __init__(self, reads: int = 3, seed: int = 0):
        self.reads = reads
        self.rng = random.Random(seed)

    def act(self, obs: Observation, inst: Instance) -> str:
        n_read = len(obs.read)
        if n_read >= self.reads or obs.steps_used >= obs.step_budget - 2:
            values = []
            for h in obs.held:
                if h.kind == "chunk":
                    for p in parse_chunk_entries(h.text):
                        if inst.answer_type in p.fields:
                            values.append(p.fields[inst.answer_type])
            guess = self.rng.choice(values) if values else "unknown"
            return f"THOUGHT: guessing.\nACTION: ANSWER {guess}"
        unread = [c for c in inst.chunks if c.kind == "register" and c.idx not in obs.read]
        c = self.rng.choice(unread)
        if c.tokens > obs.ceiling - obs.used and obs.held:
            return f"THOUGHT: make room.\nACTION: DROP {', '.join(h.id for h in obs.held)}"
        return f"THOUGHT: read something.\nACTION: READ {c.idx}"


class NoReadPolicy(MechanicalPolicy):
    """Answers immediately without reading anything."""

    name = "no-read"
    privileged = False

    def __init__(self, guess: str = "unknown"):
        self.guess = guess

    def act(self, obs: Observation, inst: Instance) -> str:
        return f"THOUGHT: no reading.\nACTION: ANSWER {self.guess}"
