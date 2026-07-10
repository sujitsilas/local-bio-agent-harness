"""bioagent CLI.

  bioagent bench <dataset> [--sweep ...]   benchmark the agent's cell-type calling
  bioagent doctor                          check local setup (model endpoint, uv, seeds)
"""
from __future__ import annotations

import argparse
import sys

from bioagent.config import load_config
from bioagent.context import AgentContext


def _doctor(args) -> int:
    cfg = load_config()
    print(f"hardware profile : {cfg.hardware_profile}")
    print(f"provider         : {cfg.provider} (cloud_llm={'on' if cfg.allow_cloud_llm else 'off'})")
    for name, spec in cfg.models.items():
        print(f"  {name:11s}: {spec.model} @ {spec.endpoint}")
    import shutil

    print(f"uv               : {shutil.which('uv') or 'NOT FOUND'}")
    print(f"seeds dir        : {cfg.path('seeds_dir')}")
    try:
        from bioagent.llm.factory import build_provider

        r = build_provider(cfg).chat([{"role": "user", "content": "reply OK"}])
        print(f"model probe      : ok ({r.content[:40]!r})")
    except Exception as e:
        print(f"model probe      : FAILED — {type(e).__name__}: {e}")
        print("                   (start LM Studio / mlx_lm.server and load the primary model)")
    return 0


def _bench(args) -> int:
    from bioagent.eval.benchmark import default_sweep, render_table, run_sweep
    from bioagent.eval.calling import Ablation
    from bioagent.eval.datasets import available, resolve

    if args.dataset == "list":
        print("public datasets:", ", ".join(available()))
        return 0
    ctx = AgentContext.build()
    spec = resolve(args.dataset, ref_col=args.reference_col, embedding=args.embedding,
                   normalized=args.normalized)
    ablations = default_sweep() if args.sweep else [
        Ablation(grounding=not args.no_grounding, thinking=not args.no_thinking,
                 reuse_signatures=not args.no_reuse, critic=args.critic,
                 model_profile=args.model_profile, resolution=args.resolution)]
    out_dir = ctx.config.path("runs_dir") / "benchmarks" / spec.name
    print(f"benchmarking cell-type calling on '{spec.name}' (ref={spec.ref_col}); "
          f"{len(ablations)} config(s)…\n")
    rows = run_sweep(ctx, spec, ablations, out_dir=out_dir)
    table = render_table(spec.name, rows)
    print("\n" + table)
    (out_dir / f"{spec.name}_table.md").write_text(table)
    print(f"\nsaved: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bioagent")
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("doctor", help="check local setup")
    dp.set_defaults(fn=_doctor)

    bp = sub.add_parser("bench", help="benchmark the agent's cell-type calling (metrics + ablations)")
    bp.add_argument("dataset", help="public dataset name (pbmc68k|pbmc3k), a local .h5ad, or 'list'")
    bp.add_argument("--reference-col", help="obs column with trusted labels (required for local .h5ad)")
    bp.add_argument("--embedding", help="obsm key to cluster on (local .h5ad; else PCA computed)")
    bp.add_argument("--normalized", action="store_true", help="X is already log-normalized (local .h5ad)")
    bp.add_argument("--resolution", type=float, default=1.0, help="leiden resolution")
    bp.add_argument("--sweep", action="store_true", help="run the full ablation sweep (5 configs)")
    bp.add_argument("--no-grounding", action="store_true", help="disable NCBI literature grounding")
    bp.add_argument("--no-thinking", action="store_true", help="disable reasoning (much faster)")
    bp.add_argument("--no-reuse", action="store_true", help="disable self-discovered-signature reuse")
    bp.add_argument("--critic", action="store_true", help="enable the critic pass")
    bp.add_argument("--model-profile", default="primary")
    bp.set_defaults(fn=_bench)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
