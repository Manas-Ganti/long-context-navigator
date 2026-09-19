"""Shared training plumbing: distributed info, model/LoRA loading, W&B, path
conventions. Adapted from the sibling RLVR project's training/common.py —
the parts that survived contact with ARC (absolute-path conda env, rank-0
logging, config-kwarg guarding across transformers versions)."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT


# --------------------------------------------------------------------------- #
# Paths (relative to the repo unless absolute)
# --------------------------------------------------------------------------- #
def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def model_tag(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].lower()


def record_run(stage: str, note: str, log_dir: str | os.PathLike) -> None:
    if not is_main():
        return
    d = resolve_path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "runs.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "note": note,
                            "slurm_job": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename}) + "\n")


# --------------------------------------------------------------------------- #
# Distributed
# --------------------------------------------------------------------------- #
@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(*names: str, default: int = 0) -> int:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip():
            try:
                return int(v)
            except ValueError:
                pass
    return default


def dist_info() -> DistInfo:
    return DistInfo(rank=_env_int("RANK", "SLURM_PROCID", default=0),
                    local_rank=_env_int("LOCAL_RANK", "SLURM_LOCALID", default=0),
                    world_size=_env_int("WORLD_SIZE", "SLURM_NTASKS", default=1) or 1)


def is_main() -> bool:
    return dist_info().is_main


def rank0_print(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs, flush=True)


def shard(seq: list, info: DistInfo | None = None) -> list:
    info = info or dist_info()
    return seq[info.rank::info.world_size]


# --------------------------------------------------------------------------- #
# Config-kwarg guard (transformers 5.x renamed TrainingArguments fields)
# --------------------------------------------------------------------------- #
def supported_config_kwargs(cls, desired: dict) -> dict:
    """Keep only kwargs `cls` accepts, remap unambiguous renames, and say so
    loudly — a silently dropped knob changes training behaviour."""
    if dataclasses.is_dataclass(cls):
        accepted = {f.name for f in dataclasses.fields(cls)}
    else:
        try:
            accepted = set(inspect.signature(cls.__init__).parameters)
        except (TypeError, ValueError):
            return dict(desired)
    out, dropped, remapped = {}, [], []
    for k, v in desired.items():
        if k in accepted:
            out[k] = v
            continue
        stem = k.replace("_", "")
        cand = [a for a in accepted if a not in desired and stem in a.replace("_", "")]
        if len(cand) == 1:
            out[cand[0]] = v
            remapped.append(f"{k} -> {cand[0]}")
        else:
            dropped.append(k)
    name = getattr(cls, "__name__", str(cls))
    if remapped:
        rank0_print(f"[{name}] remapped for this library version: {', '.join(remapped)}")
    if dropped:
        rank0_print(f"[{name}] WARNING dropped, not accepted by this version: {', '.join(dropped)}")
    return out


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return f"cuda:{dist_info().local_rank}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"          # batched generate with a decoder-only model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(model_name: str, adapter: str | None = None, *, trainable: bool = False,
                device: str | None = None, gradient_checkpointing: bool = False):
    """Causal LM (+ optional LoRA adapter). Under a DeepSpeed ZeRO-3 launch the
    trainer owns placement; otherwise the model is moved to `device`."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if (device or resolve_device()).startswith("cuda") else torch.float32
    kwargs = {}
    try:
        kwargs["attn_implementation"] = "flash_attention_2"
        import flash_attn  # noqa: F401
    except ImportError:
        kwargs["attn_implementation"] = "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if not _zero3_active():
        model.to(device or resolve_device())
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train() if trainable else model.eval()
    return model


def _zero3_active() -> bool:
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        return bool(is_deepspeed_zero3_enabled())
    except Exception:
        return False


def lora_config(r: int, alpha: int, dropout: float, target: str):
    from peft import LoraConfig

    modules = {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }[target]
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=modules, task_type="CAUSAL_LM")


# --------------------------------------------------------------------------- #
# W&B
# --------------------------------------------------------------------------- #
def wandb_init(project: str, stage: str, config: dict):
    if not is_main():
        return None
    try:
        import wandb

        return wandb.init(project=project, name=f"{stage}-{time.strftime('%m%d-%H%M')}", config=config,
                          settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "60"))))
    except Exception as e:  # never let logging take down a run
        print(f"[wandb] disabled: {e}", flush=True)
        return None


def wandb_log(run, metrics: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:
        pass
