"""Typed configs. Every number the environment, generator or reward uses is
declared here and loaded from configs/*.yaml — nothing is hardcoded in code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class RewardConfig(BaseModel):
    beta: float = 0.05
    r_correct_min: float = 0.5
    r_fail: float = -0.5
    r_wrong: float = 0.0
    allow_abstain: bool = False
    r_abstain: float = 0.1
    abstain_token: str = "unknown"

    @model_validator(mode="after")
    def _sane(self) -> "RewardConfig":
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if self.allow_abstain and not (self.r_wrong < self.r_abstain < 1.0):
            raise ValueError("r_abstain must sit strictly between r_wrong and 1.0")
        return self


class EnvConfig(BaseModel):
    tokenizer: str = "whitespace"
    context_ceiling: int = 600
    step_budget_base: int = 6
    step_budget_per_hop: int = 4
    assumed_summary_tokens: int = 32
    max_summary_tokens: int = 120
    reward: RewardConfig = Field(default_factory=RewardConfig)

    def step_budget(self, n_hops: int) -> int:
        return self.step_budget_base + self.step_budget_per_hop * n_hops


class SplitConfig(BaseModel):
    n: int
    seed_base: int
    n_hops: list[int]
    doc_tokens: list[int]


class GeneratorConfig(BaseModel):
    chunk_min_tokens: int = 320
    chunk_max_tokens: int = 480
    entries_per_chunk_min: int = 4
    entries_per_chunk_max: int = 7
    min_separation: int = 1200
    n_distractors: int = 1
    superseded_prob: float = 0.6
    near_miss_value_spread: float = 0.05
    noise_ratio: float = 0.3
    confound_tolerance: float = 0.06
    max_regeneration_attempts: int = 50
    splits: dict[str, SplitConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sane(self) -> "GeneratorConfig":
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise ValueError("chunk_min_tokens must be < chunk_max_tokens")
        if not 0 <= self.noise_ratio < 1:
            raise ValueError("noise_ratio must be in [0, 1)")
        return self


class LoraConfig(BaseModel):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target: str = "all-linear"


class GenerationConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95


class TeacherConfig(BaseModel):
    per_instance: int = 2
    noise_prob: float = 0.15
    max_step_ratio: float = 1.5


class SFTConfig(BaseModel):
    epochs: float = 2
    learning_rate: float = 1e-5
    per_device_batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    max_seq_len: int = 4096


class CurriculumConfig(BaseModel):
    enabled: bool = False
    start_hops: int = 2
    escalate_at_success: float = 0.6
    window: int = 20


class GRPOConfig(BaseModel):
    group_size: int = 8
    instances_per_step: int = 4
    learning_rate: float = 3e-5
    kl_beta: float = 0.04
    clip_eps: float = 0.2
    max_steps: int = 400
    save_every: int = 50
    train_max_hops: int = 3
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)


class TrainConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    data_dir: str = "data/v1"
    checkpoint_dir: str = "checkpoints/v1"
    wandb_project: str = "longctx-nav"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)


def _load_yaml(path: str | os.PathLike | None, default_name: str) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_DIR / default_name
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_env_config(path: str | None = None) -> EnvConfig:
    return EnvConfig(**_load_yaml(path, "env.yaml"))


def load_generator_config(path: str | None = None) -> GeneratorConfig:
    return GeneratorConfig(**_load_yaml(path, "generator.yaml"))


def load_train_config(path: str | None = None) -> TrainConfig:
    return TrainConfig(**_load_yaml(path, "train.yaml"))


def check_configs_consistent(env: EnvConfig, gen: GeneratorConfig) -> list[str]:
    """Cross-config invariants the environment's central claim depends on.
    Returned as warnings so a caller can decide whether to abort."""
    problems = []
    if gen.min_separation < env.context_ceiling:
        problems.append(
            f"min_separation ({gen.min_separation}) < context_ceiling ({env.context_ceiling}): "
            "two required facts could sit inside one window")
    if gen.chunk_max_tokens > env.context_ceiling:
        problems.append(
            f"chunk_max_tokens ({gen.chunk_max_tokens}) > context_ceiling ({env.context_ceiling}): "
            "some chunks can never be read")
    if env.max_summary_tokens + gen.chunk_max_tokens > env.context_ceiling:
        problems.append(
            f"max_summary_tokens + chunk_max_tokens ({env.max_summary_tokens + gen.chunk_max_tokens}) "
            f"> context_ceiling ({env.context_ceiling}): a maximal summary plus the largest chunk "
            "cannot coexist, so some instances may be unsolvable under compression")
    return problems
