"""Stage 2 — GRPO against the environment's verifiable reward.

Why not TRL's GRPOTrainer: the policy is stateless between steps, so one
episode is SEVERAL independent (prompt, completion) pairs that share one
return. GRPOTrainer's contract is one completion per prompt with the prompt
fixed in the dataset; forcing a multi-step episode into that shape (prompt =
first prompt, completion = all steps concatenated) would recompute logprobs
under a context the policy never conditioned on. So this is a compact GRPO
loop over per-step samples:

    for each optimiser step:
        sample B instances (rank-local); roll out G episodes per instance
        (lockstep batched HF generate, per-step token ids recorded)
        A_episode = (R - mean_group) / (std_group + eps); every step of the
        episode inherits its episode's advantage
        loss = -E[min(ratio * A, clip(ratio) * A)] + beta * KL(policy || reference)
        reference = the same model with the LoRA adapter disabled
        every EPISODE weighs the same: a step's loss is scaled by 1/(steps in
        its episode). The first full run averaged over steps instead, so an
        18-step budget-exhausted failure carried 9x the gradient of a 2-step
        ceiling violation and the policy learned to be short rather than
        right: over 60 steps accuracy fell 0.58 -> 0.37 while ceiling
        violations rose 0.02 -> 0.20 and COMPRESS use fell. (GRPO's sequence-
        level mean is exactly this episode-level weighting.)

The reference must be the SFT policy, not the base model. So the SFT adapter
is MERGED into the weights first and a fresh LoRA is trained on top: with the
adapter disabled the model is exactly the SFT policy and KL starts at ~0. The
first calibration run trained the SFT adapter directly, and KL(policy || base)
read 2.0 nats/token at step 0 — a constant pull toward forgetting SFT. The
merged model is saved once (rank 0) so vLLM can evaluate GRPO adapters on it.

Parallelism: one process per GPU (torchrun). Each rank rolls out its own
instances (HF generate, same weights), computes the loss on its own per-step
samples, and the LoRA gradients are averaged across ranks by hand before the
optimiser step. Not DDP: DDP all-reduces inside every backward and needs
equal micro-batch counts on every rank, and ours differ each step (the
number of steps an episode takes is the policy's choice). A first version
ran the forward outside the DDP wrapper, so no all-reduce ever fired — four
ranks trained four independent adapters and only rank 0's was saved — and the
only collective was a barrier every 10 steps, which the ranks reached tens of
minutes apart until the NCCL watchdog aborted one of them. The manual
all-reduce is a per-step sync, so drift is bounded by one step.
Everything the policy is trained on has n_hops <= --train-max-hops; deeper
instances are held out.

    python -m longctx.train_grpo --instances data/v1/train.jsonl --adapter checkpoints/v1/sft-...
    torchrun --nproc_per_node 8 -m longctx.train_grpo ... --deepspeed configs/deepspeed_zero2.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque

from . import common
from .audit import probe_compression_noop
from .config import load_env_config, load_train_config
from .llm import HFBackend, LLMPolicy
from .rollout import run_episodes
from .schema import load_instances


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    m = sum(rewards) / len(rewards)
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    sd = var ** 0.5
    if sd < eps:
        return [0.0] * len(rewards)
    return [(r - m) / (sd + eps) for r in rewards]


def make_samples(trajs, instances_per_group: int, group_size: int) -> tuple[list[dict], dict]:
    """Per-step samples with episode-level advantages. `trajs` is ordered as
    [inst0 x G, inst1 x G, ...]."""
    samples = []
    usable_groups = 0
    for g in range(instances_per_group):
        group = trajs[g * group_size:(g + 1) * group_size]
        adv = group_advantages([t.reward for t in group])
        usable_groups += any(a != 0.0 for a in adv)
        for t, a in zip(group, adv):
            if a == 0.0:
                continue
            steps = [s for s in t.steps if s.get("completion_ids")]
            for s in steps:
                samples.append({"prompt_ids": s["prompt_ids"], "completion_ids": s["completion_ids"],
                                "advantage": a, "weight": 1.0 / len(steps)})
    return samples, {"usable_groups": usable_groups / max(instances_per_group, 1),
                     "n_episodes": round(sum(s["weight"] for s in samples))}


def sequence_logprobs(model, batch, pad_id: int):
    """Sum-free per-token logprobs of the completion positions. Returns
    (logprobs [B, T-1], mask [B, T-1]) where mask selects completion tokens."""
    import torch

    input_ids, attn, comp_mask = batch["input_ids"], batch["attention_mask"], batch["completion_mask"]
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return lp, comp_mask[:, 1:].float()


def collate(samples: list[dict], pad_id: int, device):
    import torch

    seqs = [s["prompt_ids"] + s["completion_ids"] for s in samples]
    n = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), n), dtype=torch.long)
    cmask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, (s, seq) in enumerate(zip(samples, seqs)):
        ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
        cmask[i, len(s["prompt_ids"]):len(seq)] = 1
    adv = torch.tensor([s["advantage"] for s in samples], dtype=torch.float)
    w = torch.tensor([s.get("weight", 1.0) for s in samples], dtype=torch.float)
    return {"input_ids": ids.to(device), "attention_mask": attn.to(device), "completion_mask": cmask.to(device),
            "advantage": adv.to(device), "weight": w.to(device)}


def sync_grads(params) -> None:
    """Average gradients across ranks as one flat all-reduce. A param with no
    grad on this rank (no samples) contributes zeros."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    off = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[off:off + n].view_as(p)
        off += n


def all_reduce_sums(values: dict, device) -> dict:
    """Sum a dict of floats across ranks (metrics are logged for the whole
    data-parallel batch, not one rank's slice)."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return dict(values)
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return dict(zip(keys, t.tolist()))


def grpo_loss(model, batch, *, clip_eps: float, kl_beta: float):
    """One micro-batch. ratio is 1 on the first (and only) pass over a rollout
    batch, so the clipped surrogate reduces to policy gradient; the clip is
    kept so --inner-epochs > 1 stays correct."""
    import torch

    lp, mask = sequence_logprobs(model, batch, None)
    with torch.no_grad():
        with model.disable_adapter():
            ref_lp, _ = sequence_logprobs(model, batch, None)
    old_lp = lp.detach()
    ratio = torch.exp(lp - old_lp)
    adv = batch["advantage"].unsqueeze(1)
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    kl = torch.exp(ref_lp - lp) - (ref_lp - lp) - 1
    per_tok = -surr + kl_beta * kl
    seq_loss = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
    kl_mean = ((kl * mask).sum() / mask.sum().clamp(min=1)).item()
    # episode-weighted SUM; the caller divides by the number of episodes
    return (seq_loss * batch["weight"]).sum(), kl_mean


def merge_sft_adapter(model_name: str, adapter: str, merged_dir, accelerator, tok) -> str:
    """Merge the SFT LoRA into the base weights and save once (rank 0); every
    rank then loads the merged model as its base. Idempotent: an existing
    merged_dir with a config.json is reused."""
    import torch

    marker = os.path.join(merged_dir, "config.json")
    if accelerator.is_main_process and not os.path.exists(marker):
        common.rank0_print(f"merging {adapter} into {model_name} -> {merged_dir}")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        with open(os.path.join(merged_dir, "merged_from.json"), "w") as f:
            json.dump({"base": model_name, "adapter": adapter}, f)
        del merged, base
    accelerator.wait_for_everyone()
    if not os.path.exists(marker):
        raise RuntimeError(f"merged model not found at {merged_dir}")
    return str(merged_dir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--env-config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="SFT LoRA adapter; merged into the weights, then a fresh LoRA is trained")
    ap.add_argument("--merged-dir", default=None, help="where the merged SFT model is saved/loaded (default <checkpoint_dir>/sft-merged-<tag>)")
    ap.add_argument("--no-merge", action="store_true", help="old behaviour: keep training the SFT adapter (reference = base model)")
    ap.add_argument("--instances", default=None, help="training instances jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--instances-per-step", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--kl-beta", type=float, default=None)
    ap.add_argument("--clip-eps", type=float, default=None)
    ap.add_argument("--train-max-hops", type=int, default=None)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=2, help="sequences per forward pass")
    ap.add_argument("--rollout-batch", type=int, default=32, help="episodes advanced per generate call")
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--resume", default=None, help="adapter dir to resume from (+ state.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="wire everything with the mechanical oracle, no model")
    args = ap.parse_args(argv)

    cfg = load_train_config(args.config)
    env_cfg = load_env_config(args.env_config)
    g = cfg.grpo
    model_name = args.model or cfg.model
    instances_path = args.instances or f"{cfg.data_dir}/train.jsonl"
    out = args.out or f"{cfg.checkpoint_dir}/grpo-{common.model_tag(model_name)}"
    max_steps = args.max_steps or g.max_steps
    G = args.group_size or g.group_size
    B = args.instances_per_step or g.instances_per_step
    lr = args.learning_rate or g.learning_rate
    kl_beta = g.kl_beta if args.kl_beta is None else args.kl_beta
    clip_eps = args.clip_eps or g.clip_eps
    train_max_hops = args.train_max_hops or g.train_max_hops
    save_every = args.save_every or g.save_every
    curriculum = g.curriculum.model_copy(update={"enabled": args.curriculum or g.curriculum.enabled})

    dist = common.dist_info()
    rng = random.Random(args.seed + dist.rank)
    all_insts = [i for i in load_instances(common.resolve_path(instances_path)) if i.n_hops <= train_max_hops]
    common.rank0_print(f"train instances: {len(all_insts)} (n_hops <= {train_max_hops}) from {instances_path}")
    common.record_run("grpo", f"model={model_name} adapter={args.adapter} out={out} G={G} B={B}",
                      os.path.join(cfg.data_dir, "logs"))

    if args.dry_run:
        from .policies import OracleNavigator

        picks = rng.sample(all_insts, B)
        trajs = run_episodes(env_cfg, OracleNavigator(noise_prob=0.5, seed=1), [i for i in picks for _ in range(G)])
        samples, info = make_samples(trajs, B, G)
        print(f"dry run: {len(trajs)} episodes, usable_groups={info['usable_groups']:.2f}, "
              f"reward mean={sum(t.reward for t in trajs) / len(trajs):.3f} (token samples need a model: {len(samples)})")
        return

    from datetime import timedelta

    import torch
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    # Only used for process-group init and the device; the model is NOT wrapped
    # (see the module docstring). Generous timeout: a step can take 10 minutes.
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))])
    device = accelerator.device
    tok = common.load_tokenizer(model_name)
    from peft import get_peft_model

    torch.manual_seed(args.seed)      # identical LoRA init on every rank — averaged grads need one start point

    merged_dir = common.resolve_path(args.merged_dir or f"{cfg.checkpoint_dir}/sft-merged-{common.model_tag(model_name)}")
    if args.adapter and not args.no_merge:
        base_name = merge_sft_adapter(model_name, args.adapter, merged_dir, accelerator, tok)
        model = common.load_policy(base_name, adapter=args.resume, trainable=True, device=str(device),
                                   gradient_checkpointing=True)
        if not args.resume:
            model = get_peft_model(model, common.lora_config(cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target))
    else:
        model = common.load_policy(model_name, adapter=args.resume or args.adapter, trainable=True, device=str(device),
                                   gradient_checkpointing=True)
        if not (args.adapter or args.resume):
            model = get_peft_model(model, common.lora_config(cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target))
    if dist.is_main:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    unwrapped = model
    torch.manual_seed(args.seed + dist.rank)   # rank-distinct sampling from here on

    start_step = 0
    if args.resume and os.path.exists(os.path.join(args.resume, "state.json")):
        start_step = json.load(open(os.path.join(args.resume, "state.json")))["step"]
        common.rank0_print(f"resuming at step {start_step}")

    wandb_run = common.wandb_init(cfg.wandb_project, "grpo", {
        "model": model_name, "adapter": args.adapter, "G": G, "B": B, "lr": lr, "kl_beta": kl_beta,
        "clip_eps": clip_eps, "train_max_hops": train_max_hops, "curriculum": curriculum.model_dump(),
        "world_size": dist.world_size, "ceiling": env_cfg.context_ceiling})

    backend = HFBackend(unwrapped, tok)
    policy = LLMPolicy(backend, max_new_tokens=cfg.generation.max_new_tokens,
                       temperature=cfg.generation.temperature, top_p=cfg.generation.top_p, name="grpo-policy")
    cur_hops = curriculum.start_hops if curriculum.enabled else train_max_hops
    success_window: deque = deque(maxlen=curriculum.window)
    t0 = time.time()

    for step in range(start_step, max_steps):
        pool = [i for i in all_insts if i.n_hops <= cur_hops]
        picks = rng.sample(pool, min(B, len(pool)))
        episodes = [i for i in picks for _ in range(G)]

        # ---- rollouts (no grad, eval mode, cache on) ------------------- #
        unwrapped.eval()
        unwrapped.config.use_cache = True
        with torch.no_grad():
            trajs = run_episodes(env_cfg, policy, episodes, batch_size=args.rollout_batch)
        unwrapped.config.use_cache = False
        unwrapped.train()

        samples, ginfo = make_samples(trajs, len(picks), G)
        rng.shuffle(samples)

        # ---- update: local gradient, then averaged across ranks --------- #
        optimizer.zero_grad(set_to_none=True)
        n_episodes = max(ginfo["n_episodes"], 1)
        kls, losses = [], []
        for m in range(0, len(samples), args.micro_batch):
            batch = collate(samples[m:m + args.micro_batch], tok.pad_token_id, device)
            loss, kl = grpo_loss(unwrapped, batch, clip_eps=clip_eps, kl_beta=kl_beta)
            (loss / n_episodes).backward()      # mean over episodes, not over steps
            losses.append(loss.item())          # summed over micro-batches below
            kls.append(kl)
        sync_grads(params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        # ---- metrics over the whole data-parallel batch ------------------ #
        comp = probe_compression_noop([t.to_dict() for t in trajs], {i.id: i for i in picks})
        sums = all_reduce_sums({
            "n": len(trajs), "reward": sum(t.reward for t in trajs), "correct": sum(t.correct for t in trajs),
            "ceiling": sum(t.ceiling_exceeded for t in trajs), "budget": sum(t.budget_exhausted for t in trajs),
            "steps_ratio": sum(t.steps_used / t.min_steps for t in trajs),
            "invalid": sum(t.n_invalid for t in trajs), "steps": sum(t.steps_used for t in trajs),
            "compress_eps": comp["episodes_using_compress_rate"] * len(trajs),
            "facts_kept": (comp["fact_retention_rate"] or 0.0) * comp["evidence_facts_compressed"],
            "facts": comp["evidence_facts_compressed"],
            "groups": len(picks), "usable": ginfo["usable_groups"] * len(picks), "samples": len(samples),
            "loss": sum(losses) / n_episodes, "kl": sum(kls), "micro": len(losses), "ranks": 1,
        }, device)
        n = max(sums["n"], 1)
        acc = sums["correct"] / n
        success_window.append(acc)
        metrics = {
            "rollout/reward": sums["reward"] / n, "rollout/accuracy": acc,
            "rollout/ceiling_violation": sums["ceiling"] / n, "rollout/budget_exhausted": sums["budget"] / n,
            "rollout/steps_over_min": sums["steps_ratio"] / n,
            "rollout/invalid_actions": sums["invalid"] / max(sums["steps"], 1),
            "rollout/compress_usage": sums["compress_eps"] / n,
            "rollout/fact_retention": sums["facts_kept"] / max(sums["facts"], 1),
            "train/usable_groups": sums["usable"] / max(sums["groups"], 1), "train/samples": sums["samples"],
            "train/loss": sums["loss"] / max(sums["ranks"], 1), "train/kl": sums["kl"] / max(sums["micro"], 1),
            "train/cur_hops": cur_hops, "time/elapsed_min": (time.time() - t0) / 60,
        }
        common.rank0_print(f"[step {step:>4}] " + " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in metrics.items()))
        common.wandb_log(wandb_run, metrics, step=step)

        if curriculum.enabled and cur_hops < train_max_hops and len(success_window) == curriculum.window \
                and sum(success_window) / len(success_window) >= curriculum.escalate_at_success:
            cur_hops += 1
            success_window.clear()
            common.rank0_print(f"curriculum: escalating to n_hops <= {cur_hops}")

        if (step + 1) % save_every == 0 or step + 1 == max_steps:
            accelerator.wait_for_everyone()
            if dist.is_main:
                unwrapped.save_pretrained(out)
                tok.save_pretrained(out)
                with open(os.path.join(out, "state.json"), "w") as f:
                    json.dump({"step": step + 1, "cur_hops": cur_hops}, f)
                # keep a snapshot too (160 MB each) so intermediate policies can be evaluated
                unwrapped.save_pretrained(os.path.join(out, "steps", f"step-{step + 1}"))
                common.rank0_print(f"saved adapter to {out} at step {step + 1}")
    common.rank0_print("done")


if __name__ == "__main__":
    main()
