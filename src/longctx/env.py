"""The navigation environment.

Observation: question, document map, memory ledger, working context.
Actions:     READ, COMPRESS, DROP, ANSWER (see actions.py).
Termination: ANSWER, step budget exhausted, or the context ceiling exceeded —
             the last two score reward.r_fail.

Token accounting is exact: chunk costs are the counts recorded by the
generator (same tokenizer), summaries are counted on arrival, and
`used_tokens` is always the sum over held items. Tests pin this down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ANSWER, COMPRESS, DROP, READ, Action, ParseError, parse_action
from .config import EnvConfig
from .prompts import build_messages, render_observation
from .reward import EpisodeOutcome, answer_matches, compute_reward
from .schema import Instance
from .tokens import Tokenizer, get_tokenizer


@dataclass
class HeldItem:
    id: str                 # "12" for a chunk, "S3" for a summary
    kind: str               # "chunk" | "summary"
    tokens: int
    text: str
    sources: list[str] = field(default_factory=list)   # summary: what it replaced


@dataclass
class Observation:
    question: str
    ceiling: int
    used: int
    steps_used: int
    step_budget: int
    map_lines: list[str]
    read: list[int]
    dropped: list[int]
    compressed: list[str]
    last_action_result: str | None
    held: list[HeldItem]

    def to_text(self) -> str:
        return render_observation(self)

    def to_messages(self) -> list[dict]:
        return build_messages(self)


@dataclass
class StepRecord:
    step: int
    action_text: str
    parsed: str | None
    kind: str | None
    ok: bool
    error: str | None
    used_after: int
    ids: list[str] = field(default_factory=list)
    summary: str | None = None
    summary_sources: list[str] = field(default_factory=list)


class NavigationEnv:
    def __init__(self, cfg: EnvConfig, tokenizer: Tokenizer | None = None):
        self.cfg = cfg
        self.tok = tokenizer or get_tokenizer(cfg.tokenizer)
        self.inst: Instance | None = None

    # ------------------------------------------------------------------ #
    def reset(self, inst: Instance) -> Observation:
        self.inst = inst
        self.held: dict[str, HeldItem] = {}
        self.steps_used = 0
        self.step_budget = self.cfg.step_budget(inst.n_hops)
        self.read_log: list[int] = []
        self.dropped: list[int] = []
        self.compressed: list[str] = []
        self.n_summaries = 0
        self.done = False
        self.ceiling_exceeded = False
        self.budget_exhausted = False
        self.answer: str | None = None
        self.last_result: str | None = None
        self.records: list[StepRecord] = []
        self._map_lines = self._build_map(inst)
        return self.observe()

    @property
    def used_tokens(self) -> int:
        return sum(h.tokens for h in self.held.values())

    def observe(self) -> Observation:
        return Observation(
            question=self.inst.question, ceiling=self.cfg.context_ceiling, used=self.used_tokens,
            steps_used=self.steps_used, step_budget=self.step_budget, map_lines=self._map_lines,
            read=list(self.read_log), dropped=list(self.dropped), compressed=list(self.compressed),
            last_action_result=self.last_result, held=list(self.held.values()))

    # ------------------------------------------------------------------ #
    def step(self, action_text: str) -> tuple[Observation, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        self.steps_used += 1
        rec = StepRecord(step=self.steps_used, action_text=action_text, parsed=None, kind=None,
                         ok=False, error=None, used_after=self.used_tokens)
        try:
            action = parse_action(action_text)
            rec.parsed, rec.kind, rec.ids = str(action), action.kind, list(action.ids)
            self._apply(action, rec)
            rec.ok = rec.error is None
        except ParseError as e:
            rec.error = f"invalid action: {e}"
        rec.used_after = self.used_tokens
        self.last_result = (rec.parsed or action_text.strip().splitlines()[-1][:60] if action_text.strip() else "(empty)") \
            + (" — ERROR: " + rec.error if rec.error else " (ok)")
        self.records.append(rec)

        if not self.done and self.steps_used >= self.step_budget:
            self.done = True
            self.budget_exhausted = True
        reward = self._final_reward() if self.done else 0.0
        return self.observe(), reward, self.done, self.info()

    def _apply(self, a: Action, rec: StepRecord) -> None:
        ceiling = self.cfg.context_ceiling
        if a.kind == READ:
            idx = int(a.ids[0])
            if not 0 <= idx < len(self.inst.chunks):
                rec.error = f"no chunk {idx}"
                return
            if a.ids[0] in self.held:
                rec.error = f"chunk {idx} is already held"
                return
            ch = self.inst.chunks[idx]
            if self.used_tokens + ch.tokens > ceiling:
                self.ceiling_exceeded = True
                self.done = True
                rec.error = f"ceiling exceeded: {self.used_tokens} + {ch.tokens} > {ceiling}"
                return
            self.held[a.ids[0]] = HeldItem(id=a.ids[0], kind="chunk", tokens=ch.tokens, text=ch.text)
            self.read_log.append(idx)
            return

        if a.kind in (DROP, COMPRESS):
            missing = [i for i in a.ids if i not in self.held]
            if missing:
                rec.error = f"not held: {', '.join(missing)}"
                return
            if a.kind == DROP:
                for i in a.ids:
                    item = self.held.pop(i)
                    if item.kind == "chunk":
                        self.dropped.append(int(i))
                return
            summary = self.tok.truncate(a.text, self.cfg.max_summary_tokens)
            s_tokens = self.tok.count(summary)
            freed = sum(self.held[i].tokens for i in a.ids)
            if self.used_tokens - freed + s_tokens > ceiling:
                self.ceiling_exceeded = True
                self.done = True
                rec.error = (f"ceiling exceeded: {self.used_tokens} - {freed} + {s_tokens} > {ceiling}")
                return
            for i in a.ids:
                item = self.held.pop(i)
                if item.kind == "chunk":
                    self.dropped.append(int(i))
            self.n_summaries += 1
            sid = f"S{self.n_summaries}"
            self.held[sid] = HeldItem(id=sid, kind="summary", tokens=s_tokens, text=summary, sources=list(a.ids))
            self.compressed.append(f"{sid}<-[{', '.join(a.ids)}]")
            rec.summary, rec.summary_sources = summary, list(a.ids)
            return

        # ANSWER
        self.answer = a.text
        self.done = True

    # ------------------------------------------------------------------ #
    def outcome(self) -> EpisodeOutcome:
        return EpisodeOutcome(
            ceiling_exceeded=self.ceiling_exceeded, budget_exhausted=self.budget_exhausted,
            answered=self.answer is not None, answer=self.answer, ground_truth=self.inst.answer,
            steps_used=self.steps_used, min_steps=self.inst.min_steps)

    def _final_reward(self) -> float:
        return compute_reward(self.outcome(), self.cfg.reward)

    def info(self) -> dict:
        o = self.outcome()
        return {
            "done": self.done, "steps_used": self.steps_used, "used_tokens": self.used_tokens,
            "ceiling_exceeded": o.ceiling_exceeded, "budget_exhausted": o.budget_exhausted,
            "answered": o.answered, "answer": o.answer,
            "correct": bool(o.answered and not o.ceiling_exceeded and answer_matches(o.answer, o.ground_truth)),
            "reads": list(self.read_log), "n_reads": len(self.read_log),
            "n_compress": self.n_summaries, "n_drop": sum(1 for r in self.records if r.kind == DROP and r.ok),
            "n_errors": sum(1 for r in self.records if r.error),
            "n_invalid": sum(1 for r in self.records if r.kind is None),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_map(inst: Instance) -> list[str]:
        """Register chunks one line each; noise sections collapsed to a range."""
        lines = []
        i = 0
        chunks = inst.chunks
        while i < len(chunks):
            ch = chunks[i]
            if ch.kind == "noise":
                j = i
                while j + 1 < len(chunks) and chunks[j + 1].kind == "noise" and chunks[j + 1].section == ch.section:
                    j += 1
                mean = sum(c.tokens for c in chunks[i:j + 1]) // (j - i + 1)
                lines.append(f"  [{i}–{j}] {ch.section} ({j - i + 1} chunks, ~{mean} tokens each)")
                i = j + 1
            else:
                lines.append(f"  [{ch.idx}] {ch.header} · {ch.tokens} tokens")
                i += 1
        return lines
