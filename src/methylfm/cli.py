from __future__ import annotations

import argparse
import sys

# Heavy imports (torch, accelerate, ...) are done inside each command so the
# offline CpG-embedding commands also run in the JAX extraction environment,
# which deliberately has no torch.


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Thin wrapper over `methylfm.training.pretrain`'s --smoke path -- this
    is the SAME canonical training path `accelerate launch -m
    methylfm.training.pretrain --smoke` would run, just invoked without the
    `accelerate launch` wrapper for a quick single-process dev loop."""
    from methylfm.training.pretrain import SyntheticMethylationDataset, _smoke_config, run as pretrain_run

    config, vocab_size = _smoke_config(seed=args.seed, max_steps=args.steps)
    dataset = SyntheticMethylationDataset(num_samples=32, vocab_size=vocab_size, seed=config["seed"])
    run_name = f"smoke-s{config['seed']}"
    print(f"[smoke-test] running {args.steps} steps ...")
    result = pretrain_run(config, dataset, run_name=run_name, wandb_enabled=args.wandb)

    import torch

    if not torch.isfinite(torch.tensor(result.last_loss)):
        print(f"[smoke-test] FAILED: non-finite final loss {result.last_loss}", file=sys.stderr)
        return 1

    print(f"[smoke-test] OK: {result.global_step} steps completed, final loss={result.last_loss:.4f}")
    print("[smoke-test] NOTE: this exercises the pipeline with a synthetic CpG encoder and")
    print("[smoke-test] synthetic data only -- it validates engineering correctness, not the")
    print("[smoke-test] scientific model (the real CpG locus encoder is still unresolved).")
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from methylfm.training.pretrain import main as pretrain_main

    pretrain_argv = ["--config", args.config]
    if args.resume:
        pretrain_argv += ["--resume", args.resume]
    if args.run_name:
        pretrain_argv += ["--run-name", args.run_name]
    pretrain_main(pretrain_argv)
    return 0


def cmd_audit_data(args: argparse.Namespace) -> int:
    from methylfm.data.audit import render_report, run_audit
    from methylfm.training.pretrain import _cpgcorpus_config_from_dict
    from methylfm.utils.config import load_config

    data_cfg = load_config(args.config)
    corpus_config = _cpgcorpus_config_from_dict(data_cfg)
    report = run_audit(corpus_config)
    print(render_report(report))
    return 0


def cmd_audit_cpg_coordinates(args: argparse.Namespace) -> int:
    import json

    from methylfm.cpg_embeddings.audit import run_coordinate_audit
    from methylfm.cpg_embeddings.config import load_cpg_encoder_config

    cfg = load_cpg_encoder_config(args.config)
    report = run_coordinate_audit(cfg)
    keys = [
        "reference_genome", "total_cpg_ids", "valid_cpg_ids", "invalid_coordinates", "non_cpg_probes",
        "reference_mismatches", "reference_mismatch_fraction", "duplicate_genomic_loci",
        "unique_valid_loci", "missing_chromosomes", "ids_by_status", "passed", "problems",
    ]
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(f"full report: {cfg['output_dir']}/coordinate_audit.json")
    return 0 if report["passed"] else 1


def cmd_extract_cpg_embeddings(args: argparse.Namespace) -> int:
    from methylfm.cpg_embeddings.config import load_cpg_encoder_config
    from methylfm.cpg_embeddings.extract import run_extraction

    return run_extraction(load_cpg_encoder_config(args.config), benchmark_only=args.benchmark)


def cmd_validate_cpg_embeddings(args: argparse.Namespace) -> int:
    from methylfm.cpg_embeddings.config import load_cpg_encoder_config
    from methylfm.cpg_embeddings.validate import run_validation

    return run_validation(load_cpg_encoder_config(args.config), model_checks=not args.skip_model_checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="methylfm")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pretrain = sub.add_parser("pretrain", help="Run LeJEPA pretraining on CpGCorpus (via methylfm.training.pretrain)")
    p_pretrain.add_argument("--config", required=True)
    p_pretrain.add_argument("--resume", default=None)
    p_pretrain.add_argument("--run-name", default=None)
    p_pretrain.set_defaults(func=cmd_pretrain)

    p_audit = sub.add_parser("audit-data", help="Audit the CpGCorpus data layer")
    p_audit.add_argument("--config", required=True)
    p_audit.set_defaults(func=cmd_audit_data)

    p_cc = sub.add_parser("audit-cpg-coordinates", help="Audit CpG coordinates against the reference FASTA")
    p_cc.add_argument("--config", required=True)
    p_cc.set_defaults(func=cmd_audit_cpg_coordinates)

    p_ex = sub.add_parser(
        "extract-cpg-embeddings", help="Extract frozen AlphaGenome CpG embeddings (needs the JAX environment + GPU)"
    )
    p_ex.add_argument("--config", required=True)
    p_ex.add_argument("--benchmark", action="store_true", help="time a few windows, estimate total cost, write nothing")
    p_ex.set_defaults(func=cmd_extract_cpg_embeddings)

    p_val = sub.add_parser("validate-cpg-embeddings", help="Validate a CpG embedding store")
    p_val.add_argument("--config", required=True)
    p_val.add_argument("--skip-model-checks", action="store_true", help="only checks that need no AlphaGenome/JAX")
    p_val.set_defaults(func=cmd_validate_cpg_embeddings)

    p_smoke = sub.add_parser("smoke-test", help="Run a short synthetic end-to-end smoke pretraining")
    p_smoke.add_argument("--steps", type=int, default=10)
    p_smoke.add_argument("--seed", type=int, default=1337)
    p_smoke.add_argument("--wandb", action="store_true", help="Enable W&B logging (default: disabled)")
    p_smoke.set_defaults(func=cmd_smoke_test)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
