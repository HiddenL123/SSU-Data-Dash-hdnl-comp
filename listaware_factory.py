"""Run the copied list-aware AutoML factory (off-market parity).

Examples:
  uv run --with optuna python listaware_factory.py run --generations 20 --threads 16
  uv run --with optuna python listaware_factory.py status
  uv run --with optuna python listaware_factory.py audit --generations 20 --threads 16
  uv run --with optuna python listaware_factory.py finalize --generations 20 --threads 16
  uv run --with optuna python listaware_factory.py export --checkpoint-archive model.zip
  uv run --with optuna python listaware_factory.py restore --checkpoint-archive model.zip

This does NOT modify or replace run.py. Both tracks have separate manifest-
locked cache directories and fitted-model storage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.automl_portable import export_checkpoint, import_checkpoint
from models.listaware_ensemble_automl_factory import (
    FactorySettings, FactoryRun,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=("run", "audit", "finalize", "status", "export", "restore")
    )
    p.add_argument("--output-dir", default="outputs/listaware_ensemble_factory_v1")
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--candidates-per-generation", type=int, default=3)
    p.add_argument("--optuna-trials", type=int, default=5)
    p.add_argument("--trees", type=int, default=700)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default="listaware_factory_selected")
    p.add_argument("--checkpoint-archive", type=Path)
    args = p.parse_args(argv)
    root = Path(args.output_dir)
    if args.command == "status":
        print("List-aware factory:", root.resolve())
        for filename in ("manifest.json", "progress.json", "summary.json",
                         "roster.json", "shadow_audit_lock.json"):
            file = root / filename
            if file.exists():
                details = json.loads(file.read_text(encoding="utf-8"))
                if filename == "roster.json":
                    details = {
                        "processed": len(details["processed"]),
                        "admitted": [m["candidate"]["name"] for m in details["members"]],
                    }
                print(filename, details)
        return 0
    if args.command in ("export", "restore"):
        if not args.checkpoint_archive:
            p.error(f"{args.command} requires --checkpoint-archive")
        if args.command == "export":
            print("Checkpoint:", export_checkpoint(root, args.checkpoint_archive))
        else:
            print("Restored:", import_checkpoint(args.checkpoint_archive, root))
        return 0
    cfg = FactorySettings(
        output_dir=args.output_dir,
        generations=args.generations,
        candidates_per_generation=args.candidates_per_generation,
        optuna_trials=args.optuna_trials,
        trees=args.trees, seed=args.seed,
    )
    factory = FactoryRun(
        cfg, runtime_threads=args.threads,
        checkpoint_archive=args.checkpoint_archive,
    )
    if args.command == "run":
        result, _, _ = factory.run()
        print("Selection metrics:", result["selection_metrics"])
        print("Selected member weights:", result["effective_member_weights"])
        print("To open frozen audit, run the separate 'audit' command.")
    elif args.command == "audit":
        result, _ = factory.audit()
        print("Audit scores:", result["shadow_audit_metrics"])
    else:
        print("Final model bundle:", factory.finalize(model_name=args.model_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
