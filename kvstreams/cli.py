"""Command line entry points.

kvstreams validate   # train/inference mismatch for different replay strategies
kvstreams bench      # KV-streams vs re-prefill rollout cost
kvstreams train      # SFT warm-up + GRPO with compaction
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from .benchmark import format_table, run_benchmark
from .config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from .envs import build_env
from .factory import build_stack
from .training import RLConfig, RLTrainer, SFTTrainer, measure_mismatch
from .utils.checkpoint import save_checkpoint
from .utils.logging import JsonlLogger
from .utils.seed import set_seed

VALIDATION_CASES = [
    # label, engine, block_size, pad_to_block, commit_partial_blocks, replay mode
    ("KV-streams replay (vLLM-style, bs=16, padded)", "kvstream", 16, True, False, "kvstream"),
    ("KV-streams replay (SGLang-style, bs=1)", "kvstream", 1, False, True, "kvstream"),
    (
        "KV-streams replay, no padding + partial-block recompute",
        "kvstream",
        16,
        False,
        False,
        "kvstream",
    ),
    ("naive trainer: full rollout, no compaction", "kvstream", 16, True, False, "full"),
    ("naive trainer: rebuild compacted windows", "kvstream", 16, True, False, "windows"),
    ("re-prefill engine + window trainer", "reprefill", 16, True, True, "windows"),
]


def _policy_kwargs(args) -> dict:
    if args.policy in ("keep_recent", "markovian", "summary"):
        return {"keep_last": args.keep_last}
    if args.policy == "token_window":
        return {"window": args.window}
    return {}


def cmd_validate(args) -> int:
    set_seed(args.seed)
    base = build_stack(
        model_cfg=ModelConfig(init_std=args.init_std), dtype=args.dtype, seed=args.seed
    )
    print(f"model params: {base.model.num_parameters():,}  dtype: {args.dtype}")
    print("| case | rollouts | compactions | KL(k3) | exact KL | max abs dlogp |")
    print("|---|---:|---:|---:|---:|---:|")
    rows = []
    for label, engine, bs, pad, commit, mode in VALIDATION_CASES:
        stack = build_stack(
            engine=engine,
            policy=args.policy,
            policy_kwargs=_policy_kwargs(args),
            model=base.model,
            cache_cfg=CacheConfig(block_size=bs, pad_to_block=pad, commit_partial_blocks=commit),
            rollout_cfg=RolloutConfig(context_budget=args.budget, record_full_logprobs=True),
            sampling=SamplingConfig(max_new_tokens=args.max_new_tokens),
            dtype=args.dtype,
        )
        gen = torch.Generator().manual_seed(args.seed)
        rollouts = [
            stack.engine.rollout(build_env(args.env, seed=args.seed + i), generator=gen)
            for i in range(args.rollouts)
        ]
        rep = measure_mismatch(stack.model, rollouts, mode)
        comp = sum(r.stats.compactions for r in rollouts) / len(rollouts)
        rows.append({"case": label, **rep.to_dict(), "compactions": comp})
        exact = f"{rep.exact_kl:.2e}" if rep.exact_kl is not None else "-"
        print(
            f"| {label} | {len(rollouts)} | {comp:.1f} | {rep.kl_k3:.2e} | {exact} "
            f"| {rep.max_abs_logprob_diff:.2e} |"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


def cmd_bench(args) -> int:
    set_seed(args.seed)
    results = run_benchmark(
        budgets=args.budgets,
        num_rollouts=args.rollouts,
        env=args.env,
        env_kwargs={"num_distractors": args.turns, "filler_len": args.filler_len}
        if args.env == "recall"
        else {"num_steps": args.turns},
        policy=args.policy,
        policy_kwargs=_policy_kwargs(args),
        model_cfg=ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=8,
            n_kv_heads=4,
            d_ff=3 * args.d_model,
        ),
        block_size=args.block_size,
        seed=args.seed,
        device=args.device,
    )
    print(format_table(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    return 0


def cmd_train(args) -> int:
    set_seed(args.seed)
    model_cfg = ModelConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=4, n_kv_heads=2, d_ff=4 * args.d_model
    )
    stack = build_stack(
        engine=args.engine,
        policy=args.policy,
        policy_kwargs=_policy_kwargs(args),
        model_cfg=model_cfg,
        cache_cfg=CacheConfig(block_size=args.block_size),
        rollout_cfg=RolloutConfig(context_budget=args.budget),
        sampling=SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature),
        device=args.device,
        seed=args.seed,
    )
    print(
        f"model params: {stack.model.num_parameters():,}  engine: {args.engine}  policy: {stack.engine.policy}"
    )
    env_kwargs = {}
    if args.env == "recall":
        env_kwargs = {"num_distractors": args.turns}
    elif args.env == "chainsum":
        env_kwargs = {"num_steps": args.turns}

    def env_factory(seed: int):
        return build_env(args.env, seed=seed, **env_kwargs)

    logger = JsonlLogger(args.log)
    if args.sft_steps:
        sft = SFTTrainer(
            stack.model,
            stack.engine,
            env_factory,
            lr=args.sft_lr,
            batch_size=args.sft_batch,
            seed=args.seed,
        )
        for i, loss in enumerate(sft.train(args.sft_steps, log_every=args.log_every)):
            logger.log({"phase": "sft", "step": i, "loss": loss})
    if args.rl_steps:
        cfg = RLConfig(
            steps=args.rl_steps,
            tasks_per_step=args.tasks_per_step,
            group_size=args.group_size,
            lr=args.lr,
            seed=args.seed,
            log_every=args.log_every,
        )
        trainer = RLTrainer(stack.model, stack.engine, env_factory, cfg)
        for m in trainer.train():
            logger.log({"phase": "rl", **m})
    if args.save:
        save_checkpoint(args.save, stack.model, args=vars(args))
        print(f"saved checkpoint to {args.save}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kvstreams", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, env="recall", policy="keep_recent"):
        sp.add_argument("--env", default=env, choices=["recall", "chainsum"])
        sp.add_argument(
            "--policy",
            default=policy,
            choices=["keep_recent", "markovian", "summary", "token_window"],
        )
        sp.add_argument("--keep-last", type=int, default=2)
        sp.add_argument("--window", type=int, default=128)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--json", default=None, help="write results as JSON")

    v = sub.add_parser("validate", help="measure train/inference mismatch")
    common(v)
    v.add_argument("--rollouts", type=int, default=8)
    v.add_argument("--budget", type=int, default=160)
    v.add_argument("--max-new-tokens", type=int, default=12)
    v.add_argument("--init-std", type=float, default=0.1)
    v.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    v.set_defaults(func=cmd_validate)

    b = sub.add_parser("bench", help="compare KV-streams with re-prefill")
    common(b)
    b.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    b.add_argument("--rollouts", type=int, default=4)
    b.add_argument("--turns", type=int, default=24)
    b.add_argument("--filler-len", type=int, default=64)
    b.add_argument("--block-size", type=int, default=16)
    b.add_argument("--d-model", type=int, default=256)
    b.add_argument("--n-layers", type=int, default=4)
    b.add_argument("--device", default="cpu")
    b.set_defaults(func=cmd_bench)

    t = sub.add_parser("train", help="SFT warm-up + GRPO with compaction")
    common(t)
    t.add_argument("--engine", default="kvstream", choices=["kvstream", "reprefill"])
    t.add_argument("--budget", type=int, default=160)
    t.add_argument("--turns", type=int, default=4)
    t.add_argument("--block-size", type=int, default=16)
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--n-layers", type=int, default=2)
    t.add_argument("--max-new-tokens", type=int, default=8)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--sft-steps", type=int, default=200)
    t.add_argument("--sft-lr", type=float, default=3e-3)
    t.add_argument("--sft-batch", type=int, default=8)
    t.add_argument("--rl-steps", type=int, default=50)
    t.add_argument("--tasks-per-step", type=int, default=4)
    t.add_argument("--group-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--log", default=None, help="JSONL metrics path")
    t.add_argument("--save", default=None, help="checkpoint path")
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
