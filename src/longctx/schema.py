"""Data model for a generated task instance. Everything the verifier, the
baselines and the audit need is recorded at generation time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    idx: int
    section: str                 # e.g. "Project Register"
    header: str                  # e.g. "Project Register · Fal–Hen"
    kind: str                    # "register" | "noise"
    text: str
    tokens: int
    entity_names: list[str] = Field(default_factory=list)  # entries in this chunk


class Evidence(BaseModel):
    hop: int                     # 0-based position in the chain
    chunk_idx: int
    entity_type: str
    entity: str
    field: str                   # link field (-> next entity) or the answer attribute
    value: str                   # next entity's name, or the answer
    token_offset: int            # token position of the entry in the concatenated document


class Distractor(BaseModel):
    kind: str                    # "superseded_link" | "superseded_value" | "near_miss_entity" | "near_miss_value"
    hop: int
    chunk_idx: int
    entity: str
    value: str                   # the wrong value the distractor offers
    wrong_answer: str | None = None  # the answer a policy would reach by following it


class Instance(BaseModel):
    id: str
    seed: int
    split: str = ""
    n_hops: int
    doc_tokens_target: int
    doc_tokens: int
    question: str
    answer: str
    answer_type: str             # attribute name, e.g. "annual budget"
    answer_entity_type: str
    chunks: list[Chunk]
    evidence: list[Evidence]
    distractors: list[Distractor]
    candidate_answers: list[str]  # every value of answer_type in the corpus
    min_reads: int
    min_steps: int
    compression_required: bool
    generator: dict = Field(default_factory=dict)
    env: dict = Field(default_factory=dict)

    @property
    def evidence_chunks(self) -> list[int]:
        return [e.chunk_idx for e in self.evidence]

    @property
    def answer_chunk(self) -> int:
        return self.evidence[-1].chunk_idx

    def chunk(self, idx: int) -> Chunk:
        return self.chunks[idx]


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r.model_dump() if isinstance(r, BaseModel) else r) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_instances(path: str | Path) -> list[Instance]:
    return [Instance(**row) for row in read_jsonl(path)]
