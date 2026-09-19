"""`longctx` command line: generate -> validate -> baselines -> teacher -> sft -> grpo -> evaluate -> audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .audit import audit_markdown, run_audit
from .baselines import baseline_table, llm_baselines, mechanical_baselines
from .common import resolve_path
from .config import (check_configs_consistent, load_env_config, load_generator_config, load_train_config)
from .schema import load_instances, read_jsonl, write_jsonl


def _progress(prefix):
    t0 = time.time()

    def p(i, n):
        print(f"\r{prefix} {i}/{n} ({time.time() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
        if i == n:
            print(file=sys.stderr)
    return p


def _load_many(paths: list[str]):
    out = []
    for p in paths:
        out.extend(load_instances(resolve_path(p)))
    return out


def _policy(args, env_cfg):
    from .policies import NoReadPolicy, OracleNavigator, RandomReader

    if args.policy == "oracle":
        return OracleNavigator(noise_prob=getattr(args, "noise", 0.0), seed=args.seed)
    if args.policy == "random":
        return RandomReader(seed=args.seed)
    if args.policy == "no-read":
        return NoReadPolicy()
    if args.policy == "llm":
        from .llm import LLMPolicy, build_backend

        tcfg = load_train_config(args.config)
        backend = build_backend(args.backend, args.model or tcfg.model, args.adapter,
                                tensor_parallel_size=args.tp, seed=args.seed)
        return LLMPolicy(backend, max_new_tokens=tcfg.generation.max_new_tokens, sample=not args.greedy,
                         temperature=tcfg.generation.temperature, top_p=tcfg.generation.top_p,
                         name=f"llm-{args.backend}:{(args.adapter or args.model or tcfg.model).rstrip('/').split('/')[-1]}")
    raise SystemExit(f"unknown policy {args.policy}")


# --------------------------------------------------------------------------- #
def cmd_generate(args):
    from .generate import generate_split
    from .validate_instance import confound_audit

    gen, env = load_generator_config(args.generator_config), load_env_config(args.env_config)
    for w in check_configs_consistent(env, gen):
        print(f"CONFIG WARNING: {w}", file=sys.stderr)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = args.splits or list(gen.splits)
    report = {"generator": gen.model_dump(), "env": env.model_dump(), "splits": {}, "confound": {}}
    ok = True
    for split in splits:
        insts, stats = generate_split(gen, env, split, limit=args.limit, progress=_progress(f"[{split}]"))
        write_jsonl(out / f"{split}.jsonl", insts)
        aud = confound_audit(insts, gen.confound_tolerance)
        report["splits"][split], report["confound"][split] = stats, aud
        ok &= aud["passed"]
        print(f"{split}: {stats['n']} instances, {stats['attempts']} attempts, rejection rate "
              f"{stats['rejection_rate']:.3f} {stats['rejections_by_reason']}; confound audit "
              f"{'PASSED' if aud['passed'] else 'FAILED ' + str(aud['violations'])}")
    with open(out / "generation_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out}/generation_report.json")
    if not ok:
        print("SURFACE CONFOUND DETECTED — fix the generator before training", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args):
    from .validate_instance import confound_audit, validate

    gen, env = load_generator_config(args.generator_config), load_env_config(args.env_config)
    insts = _load_many(args.instances)
    bad = {i.id: r for i in insts if (r := validate(i, gen, env))}
    aud = confound_audit(insts, gen.confound_tolerance)
    print(f"{len(insts)} instances, {len(bad)} invalid; confound audit {'PASSED' if aud['passed'] else 'FAILED'}")
    print(json.dumps({k: v for k, v in aud["auc"].items()}, indent=1))
    if bad:
        print(json.dumps(dict(list(bad.items())[:10]), indent=1))
        sys.exit(1)
    if not aud["passed"]:
        print(aud["violations"])
        sys.exit(1)


def cmd_baselines(args):
    insts = _load_many(args.instances)
    mech = mechanical_baselines(insts, seed=args.seed)
    llm = None
    if args.mode == "llm":
        from .llm import build_backend

        tcfg = load_train_config(args.config)
        backend = build_backend(args.backend, args.model or tcfg.model, args.adapter, tensor_parallel_size=args.tp)
        llm = llm_baselines(insts, backend, batch_size=args.batch_size, seed=args.seed,
                            reasoning=args.reasoning, max_new_tokens=512 if args.reasoning else 24)
    table = baseline_table(mech, llm)
    print(table)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mechanical": mech, "llm": llm, "n": len(insts), "instances": args.instances,
                   "model": args.model if args.mode == "llm" else None}, f, indent=1)
    with open(out.with_suffix(".md"), "w") as f:
        f.write(table + "\n")
    print(f"wrote {out}")


def cmd_teacher(args):
    from .teacher import sample_teacher_trajectories, to_sft_rows

    env = load_env_config(args.env_config)
    tcfg = load_train_config(args.config)
    insts = _load_many(args.instances)
    if args.limit:
        insts = insts[: args.limit]
    kept, stats = sample_teacher_trajectories(
        env, insts, per_instance=args.per_instance or tcfg.teacher.per_instance,
        noise_prob=tcfg.teacher.noise_prob, max_step_ratio=tcfg.teacher.max_step_ratio, seed=args.seed,
        progress=_progress("[teacher]"))
    rows = to_sft_rows(kept)
    n = write_jsonl(resolve_path(args.out), rows)
    stats["sft_rows"] = n
    with open(resolve_path(args.out).with_suffix(".stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    print(f"wrote {n} SFT rows to {args.out}")


def cmd_evaluate(args):
    from .evaluate import summarize, write_report
    from .rollout import run_episodes

    env = load_env_config(args.env_config)
    insts = _load_many(args.instances)
    if args.limit:
        insts = insts[: args.limit]
    policy = _policy(args, env)
    trajs = run_episodes(env, policy, insts, batch_size=args.batch_size, record_prompts=args.record_prompts,
                         progress=_progress("[eval]"))
    rep = summarize(trajs, insts, args.train_max_hops or load_train_config(args.config).grpo.train_max_hops)
    from .evaluate import markdown

    print(markdown(rep))
    write_report(rep, trajs, resolve_path(args.out))
    print(f"wrote {args.out}.json / .md / .trajectories.jsonl")


def cmd_audit(args):
    insts = _load_many(args.instances)
    trajs = list(read_jsonl(resolve_path(args.trajectories)))
    rep = run_audit(trajs, insts)
    print(audit_markdown(rep))
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    with open(out.with_suffix(".md"), "w") as f:
        f.write(audit_markdown(rep) + "\n")
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="longctx")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_cfg(p):
        p.add_argument("--generator-config", default=None)
        p.add_argument("--env-config", default=None)
        p.add_argument("--config", default=None, help="train.yaml")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("generate", help="generate + validate all splits, run the confound audit")
    common_cfg(p)
    p.add_argument("--out", default="data/v1")
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="cap instances per split (smoke runs)")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("validate", help="re-validate instance files and re-run the confound audit")
    common_cfg(p)
    p.add_argument("--instances", nargs="+", required=True)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("baselines", help="no-read / single-chunk / full-document")
    common_cfg(p)
    p.add_argument("--instances", nargs="+", required=True)
    p.add_argument("--mode", choices=["mechanical", "llm"], default="mechanical")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--reasoning", action="store_true", help="let the model think before ANSWER: (512 tokens)")
    p.add_argument("--out", default="results/baselines.json")
    p.set_defaults(fn=cmd_baselines)

    p = sub.add_parser("teacher", help="cold-start trajectories from the privileged oracle -> SFT rows")
    common_cfg(p)
    p.add_argument("--instances", nargs="+", required=True)
    p.add_argument("--out", default="data/v1/sft.jsonl")
    p.add_argument("--per-instance", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_teacher)

    p = sub.add_parser("evaluate", help="run a policy over held-out instances")
    common_cfg(p)
    p.add_argument("--instances", nargs="+", required=True)
    p.add_argument("--policy", choices=["oracle", "random", "no-read", "llm"], default="llm")
    p.add_argument("--noise", type=float, default=0.0, help="oracle noise")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--train-max-hops", type=int, default=None)
    p.add_argument("--record-prompts", action="store_true")
    p.add_argument("--out", default="results/eval")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("audit", help="the five reward-hacking probes over a trajectory file")
    common_cfg(p)
    p.add_argument("--instances", nargs="+", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--out", default="results/audit.json")
    p.set_defaults(fn=cmd_audit)

    for name in ("sft", "grpo"):
        sub.add_parser(name, help=f"run train_{name} (all remaining args pass through; see --help there)")

    argv = sys.argv[1:] if argv is None else list(argv)
    # sft / grpo forward everything to their own parsers
    if argv and argv[0] in ("sft", "grpo"):
        from .train_grpo import main as grpo_main
        from .train_sft import main as sft_main

        return (sft_main if argv[0] == "sft" else grpo_main)(argv[1:])
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
