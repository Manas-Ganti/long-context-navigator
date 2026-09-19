"""Stage 1 — SFT on leak-filtered teacher trajectories.

Because the policy is stateless between steps, every step is an independent
(prompt, completion) pair: no multi-turn masking, no replay. Plain
transformers Trainer + PEFT LoRA; loss on the assistant tokens only.

    python -m longctx.train_sft --sft-data data/v1/sft.jsonl --out checkpoints/v1/sft
    torchrun --nproc_per_node 8 -m longctx.train_sft ... --deepspeed configs/deepspeed_zero2.json
"""

from __future__ import annotations

import argparse
import json
import os

from . import common
from .config import load_train_config
from .schema import read_jsonl


def build_example(tok, messages: list[dict], max_seq_len: int) -> dict | None:
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + messages[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_seq_len:
        return None
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}


class ListDataset:
    """Minimal map-style dataset so Trainer needs nothing beyond torch."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        def pad(key, val):
            return torch.tensor([b[key] + [val] * (n - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", self.pad_id), "labels": pad("labels", -100),
                "attention_mask": pad("attention_mask", 0)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--sft-data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--per-device-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="tokenise and report; no model")
    args = ap.parse_args(argv)

    cfg = load_train_config(args.config)
    model_name = args.model or cfg.model
    sft_data = args.sft_data or f"{cfg.data_dir}/sft.jsonl"
    out = args.out or f"{cfg.checkpoint_dir}/sft-{common.model_tag(model_name)}"
    s = cfg.sft
    epochs = args.epochs or s.epochs
    lr = args.learning_rate or s.learning_rate
    bs = args.per_device_batch_size or s.per_device_batch_size
    accum = args.grad_accum or s.grad_accum
    max_len = args.max_seq_len or s.max_seq_len

    rows = list(read_jsonl(common.resolve_path(sft_data)))
    if args.limit:
        rows = rows[: args.limit]
    common.rank0_print(f"SFT rows: {len(rows)} from {sft_data}")
    common.record_run("sft", f"model={model_name} rows={len(rows)} out={out}", os.path.join(cfg.data_dir, "logs"))

    tok = common.load_tokenizer(model_name)
    examples = [e for r in rows if (e := build_example(tok, r["messages"], max_len))]
    common.rank0_print(f"tokenised: {len(examples)} kept (<= {max_len} tokens), "
                       f"{len(rows) - len(examples)} dropped for length; "
                       f"max len {max(len(e['input_ids']) for e in examples) if examples else 0}")
    if args.dry_run:
        return

    import torch
    from transformers import Trainer, TrainingArguments

    dist = common.dist_info()
    wandb_run = common.wandb_init(cfg.wandb_project, "sft", {"model": model_name, "rows": len(examples),
                                                             "lr": lr, "epochs": epochs, "bs": bs, "accum": accum})
    targs = TrainingArguments(**common.supported_config_kwargs(TrainingArguments, dict(
        output_dir=out, per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        num_train_epochs=epochs, max_steps=args.max_steps, learning_rate=lr, warmup_ratio=s.warmup_ratio,
        lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed, ddp_find_unused_parameters=False, logging_steps=5, save_steps=200,
        save_total_limit=2, seed=args.seed, report_to=["wandb"] if wandb_run else [],
        remove_unused_columns=False, dataloader_num_workers=2,
    )))
    model = common.load_policy(model_name, trainable=True)
    from peft import get_peft_model

    model = get_peft_model(model, common.lora_config(cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target))
    if dist.is_main:
        model.print_trainable_parameters()
    trainer = Trainer(model=model, args=targs, train_dataset=ListDataset(examples),
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    trainer.save_model(out)
    if dist.is_main:
        tok.save_pretrained(out)
        with open(os.path.join(out, "sft_meta.json"), "w") as f:
            json.dump({"model": model_name, "rows": len(examples), "sft_data": sft_data}, f)
    common.rank0_print(f"saved SFT adapter to {out}")


if __name__ == "__main__":
    main()
