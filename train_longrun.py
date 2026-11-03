"""Long-running, resumable, track-isolated SSU Data Dash ensemble search.

Use a NEW output root. Existing audited experiments are intentionally not
modified. This runner uses the existing AutoML factories and their manifest,
fold caches, Optuna SQLite studies, atomic roster updates, and optional ZIP
checkpoints; it never changes model-selection/audit boundaries.

Examples (PowerShell):
  uv run --with optuna python train_longrun.py run --track both
  uv run python train_longrun.py status --track both
  uv run --with optuna python train_longrun.py run --track list_aware --generations 20 --no-auto-audit
  uv run --with optuna python train_longrun.py audit --track list_aware
  uv run --with optuna python train_longrun.py finalize --track list_aware
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TRACKS = ("off_market", "list_aware")
DEFAULT_OUTPUT_ROOT = "outputs/dual_track_longrun_v2"


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "audit", "finalize"))
    parser.add_argument("--track", choices=(*TRACKS, "both"), default="both")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Independent experiment root; never point at an audited run")
    parser.add_argument("--generations", type=int, default=12,
                        help="12 generations per track; may increase BEFORE audit to continue")
    parser.add_argument("--candidates-per-generation", type=int, default=3,
                        help="Immutable per experiment after first run")
    parser.add_argument("--optuna-trials", type=int, default=8,
                        help="Immutable per experiment after first run")
    parser.add_argument("--trees", type=int, default=700,
                        help="Full-fidelity tree budget, immutable after first run")
    parser.add_argument("--max-members", type=int, default=8,
                        help="Upper bound on accepted ensemble members")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads per model (may change on resume)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Optional OUTSIDE the experiment: also export ZIP and .sha256 after each training checkpoint (more disk I/O)")
    parser.add_argument("--export-dir", default="colab_checkpoints/dual_track_longrun_v2",
                        help="Where to automatically export one portable ZIP + .sha256 per track after run/audit")
    parser.add_argument("--no-auto-audit", action="store_true",
                        help="Run selection and export only; leave audit unopened so you can extend generations later")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration and checkpoint state without loading data or training")
    return parser


def settings_for(track, args):
    """Build full-strength settings without weakening train/validation folds."""
    output = (Path(args.output_root) / track).as_posix()
    if track == "off_market":
        from models.automl_profiles import release_settings
        from models.ensemble_automl_factory import FactorySettings
        settings = release_settings(output_dir=output, generations=args.generations)
    elif track == "list_aware":
        from dataclasses import asdict
        from models.listaware_ensemble_automl_factory import FactorySettings
        settings = asdict(FactorySettings())
        settings.update(output_dir=output, generations=args.generations)
    else:
        raise ValueError(f"Unknown competition track: {track}")
    settings.update(
        candidates_per_generation=args.candidates_per_generation,
        optuna_trials=args.optuna_trials,
        trees=args.trees,
        max_members=args.max_members,
        seed=args.seed,
    )
    return FactorySettings(**settings)


def checkpoint_for(track, args):
    if args.checkpoint_dir is None:
        return None
    return Path(args.checkpoint_dir) / f"{track}_checkpoint.zip"


def export_completed_checkpoint(track, args, experiment_dir):
    """Export a portable, checksum-protected snapshot after selection/audit.

    The archive is distinct from local live checkpoints. It excludes raw
    listings and fitted model.joblib; final fitting remains a separate step.
    """
    from models.automl_portable import export_checkpoint

    destination = Path(args.export_dir) / f"{track}_checkpoint.zip"
    source = Path(experiment_dir).resolve()
    target = destination.resolve()
    if target == source or source in target.parents:
        raise ValueError(
            "--export-dir must be outside the active experiment directory"
        )
    created = export_checkpoint(source, target)
    checksum = created.with_name(created.name + ".sha256")
    if not created.is_file() or not checksum.is_file():
        raise RuntimeError(f"Incomplete checkpoint export: {created}")
    print(f"[{track}] SEND: {created}", flush=True)
    print(f"[{track}] SEND: {checksum}", flush=True)
    return created, checksum


def show_state(track, args):
    from dataclasses import asdict
    settings = settings_for(track, args)
    root = Path(settings.output_dir)
    print(f"\n[{track}] experiment: {root}", flush=True)
    print(f"  input DATA_DIR: {os.environ.get('DATA_DIR', '(project data/; then HF cache)')}")
    print(f"  budget: generations={settings.generations}, candidates/generation="
          f"{settings.candidates_per_generation}, Optuna trials={settings.optuna_trials}, "
          f"trees={settings.trees}, max members={settings.max_members}")
    print(f"  threads: {args.threads}")
    print(f"  optional ongoing checkpoint: {checkpoint_for(track, args) or '(disabled; local resume still enabled)'}")
    print(f"  automatic ZIP export: {Path(args.export_dir) / (track + '_checkpoint.zip')}")
    print(f"  audit after run: {'disabled (selection only)' if args.no_auto_audit else 'automatic'}")
    manifest_file = root / "manifest.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        print(f"  existing data fingerprint: {manifest.get('data_sha256')}")
        # The manifest intentionally omits generations to allow safe growth
        # before audit. Other settings must be kept identical on a resume.
        frozen = manifest.get("settings", {})
        actual = asdict(settings)
        for k, v in frozen.items():
            if k == "generations":
                continue
            if actual.get(k) != v and not (
                isinstance(actual.get(k), tuple) and list(actual[k]) == v
            ):
                print(f"  WARNING: checkpoint setting {k} differs: {v!r} != {actual.get(k)!r}")
    else:
        print("  new experiment; no manifest found")
    for filename in ("progress.json", "roster.json", "shadow_audit_lock.json"):
        path = root / filename
        if not path.is_file():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        if filename == "progress.json":
            print(f"  progress: {saved}")
        elif filename == "roster.json":
            print(f"  processed candidates: {len(saved.get('processed', []))}; "
                  f"accepted members: {len(saved.get('members', []))}")
        else:
            print(f"  AUDIT OPENED ({saved.get('status')}): selection is locked")


def train_one(track, args):
    settings = settings_for(track, args)
    checkpoint = checkpoint_for(track, args)
    if args.command == "run":
        if track == "off_market":
            from models.ensemble_automl_factory import FactoryRun
        else:
            from models.listaware_ensemble_automl_factory import FactoryRun
        factory = FactoryRun(settings, runtime_threads=args.threads,
                             checkpoint_archive=checkpoint)
        outcome, _, _ = factory.run()
        print(f"[{track}] selection metrics: {outcome['selection_metrics']}", flush=True)
        print(f"[{track}] selected weights: {outcome['effective_member_weights']}", flush=True)
        if args.no_auto_audit:
            print(f"[{track}] audit NOT opened; generations can still be extended.", flush=True)
        else:
            # Only audit a completed selection. The factory locks the frozen
            # roster before exposing audit predictions; never extend afterward.
            report, _ = factory.audit()
            print(f"[{track}] frozen audit: {report['shadow_audit_metrics']}", flush=True)
        export_completed_checkpoint(track, args, settings.output_dir)
    elif args.command == "audit":
        if track == "off_market":
            from models.ensemble_automl_factory import FactoryRun
        else:
            from models.listaware_ensemble_automl_factory import FactoryRun
        factory = FactoryRun(settings, runtime_threads=args.threads,
                             checkpoint_archive=checkpoint)
        report, _ = factory.audit()
        print(f"[{track}] frozen audit: {report['shadow_audit_metrics']}", flush=True)
        export_completed_checkpoint(track, args, settings.output_dir)
    elif args.command == "finalize":
        if track == "off_market":
            from models.ensemble_release_candidate import finalize_release
            model_name = "off_market_longrun_v2"
            finalize_release(settings, threads=args.threads, model_name=model_name)
        else:
            from models.listaware_ensemble_automl_factory import FactoryRun
            model_name = "list_aware_longrun_v2"
            factory = FactoryRun(settings, runtime_threads=args.threads)
            factory.finalize(model_name=model_name)
        print(f"[{track}] finalized in saved_models/{track}/{model_name}", flush=True)
    else:
        raise ValueError(args.command)


def main(argv=None):
    args = parser_for_cli().parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    tracks = TRACKS if args.track == "both" else (args.track,)
    if args.command == "status" or args.dry_run:
        for track in tracks:
            show_state(track, args)
        return 0
    if args.track == "both":
        # Fresh interpreter per track so large fitted providers are released
        # before starting the next search / audit / full-history refit.
        for track in tracks:
            command = [
                sys.executable, str(Path(__file__).resolve()), args.command,
                "--track", track,
                "--output-root", args.output_root,
                "--generations", str(args.generations),
                "--candidates-per-generation", str(args.candidates_per_generation),
                "--optuna-trials", str(args.optuna_trials),
                "--trees", str(args.trees),
                "--max-members", str(args.max_members),
                "--seed", str(args.seed),
                "--threads", str(args.threads),
            ]
            if args.checkpoint_dir is not None:
                command.extend(["--checkpoint-dir", args.checkpoint_dir])
            command.extend(["--export-dir", args.export_dir])
            if args.no_auto_audit:
                command.append("--no-auto-audit")
            print(f"\n=== {args.command.upper()} {track} ===", flush=True)
            subprocess.run(command, check=True)
    else:
        show_state(args.track, args)
        train_one(args.track, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
