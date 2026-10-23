"""One CLI for daytime Colab and overnight local release-candidate training.

Both execution profiles run the SAME 700-tree/full-history experiment.
Only CPU thread allocation and optional external checkpoint backup differ.

Examples:
  python ensemble_rc.py run --profile overnight_local
  python ensemble_rc.py run --profile day_cloud --checkpoint-archive /content/drive/MyDrive/automl/checkpoint.zip
  python ensemble_rc.py status
  python ensemble_rc.py audit --profile overnight_local
  python ensemble_rc.py benchmark --profile overnight_local
  python ensemble_rc.py finalize --profile overnight_local
  python ensemble_rc.py predict
  python ensemble_rc.py export --checkpoint-archive D:/automl/checkpoint.zip
  python ensemble_rc.py restore --checkpoint-archive D:/automl/checkpoint.zip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.automl_profiles import (
    WORKER_PROFILES, get_profile, release_settings,
)
from models.automl_portable import export_checkpoint, import_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("run", "audit", "finalize", "predict",
                            "status", "export", "restore", "benchmark", "diagnose"),
    )
    parser.add_argument(
        "--profile", choices=tuple(WORKER_PROFILES),
        default="overnight_local",
    )
    parser.add_argument(
        "--output-dir", default="outputs/ensemble_release_candidate_full",
        help="Same experiment path across runs; on Colab restore to this local VM path",
    )
    parser.add_argument(
        "--checkpoint-archive", default=None,
        help="Destination/source portable ZIP, e.g. Google Drive; NOT a live SQLite directory",
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Increase before shadow audit; other experiment parameters are frozen",
    )
    parser.add_argument(
        "--use-checkpoint-settings", action="store_true",
        help="Explicitly reuse frozen settings from restored manifest; "
             "does NOT bypass input-data fingerprint checks",
    )
    parser.add_argument(
        "--model-name", default="off_market_release_candidate_full",
    )
    parser.add_argument(
        "--allow-audit-regression", action="store_true",
        help="Explicit experimental fallback; do not describe as audit-approved",
    )
    parser.add_argument(
        "--submission-file", default="submissions/submission_off_market_rc.csv",
    )
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    root = Path(args.output_dir)
    if args.command == "diagnose":
        from models.automl_manifest import diagnose
        result = diagnose(
            root,
            generations=args.generations if args.generations is not None else 2,
            use_saved_settings=args.use_checkpoint_settings,
        )
        return 0 if not result["differences"] else 1
    if args.command == "status":
        print("Experiment:", root.resolve())
        for filename in (
            "manifest.json", "progress.json", "roster.json",
            "summary.json", "shadow_audit_lock.json", "release_manifest.json",
        ):
            path = root / filename
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if filename == "manifest.json":
                    payload = {"trees": payload["settings"]["trees"],
                               "train_cap": payload["settings"]["train_cap"],
                               "validation_cap": payload["settings"]["validation_cap"]}
                elif filename == "roster.json":
                    payload = {
                        "accepted": [m["candidate"]["name"] for m in payload["members"]],
                        "processed_count": len(payload["processed"]),
                    }
                print(filename, payload)
        return 0
    if args.command in ("export", "restore") and not args.checkpoint_archive:
        parser.error(f"{args.command} requires --checkpoint-archive")
    if args.command == "export":
        print("Exported portable checkpoint:",
              export_checkpoint(root, args.checkpoint_archive))
        return 0
    if args.command == "restore":
        print("Restored to local experiment:",
              import_checkpoint(args.checkpoint_archive, root))
        return 0
    if args.command == "predict":
        from ensemble_release_candidate import predict_release
        print("Submission candidate:",
              predict_release(
                  model_name=args.model_name,
                  output_file=args.submission_file,
              ))
        return 0

    if args.use_checkpoint_settings:
        from models.automl_manifest import saved_factory_settings
        settings_obj = saved_factory_settings(
            root,
            generations=args.generations if args.generations is not None else 2,
        )
        settings = settings_obj.record()
        if Path(settings["output_dir"]).as_posix() != root.as_posix():
            parser.error(
                "The saved checkpoint's output_dir differs from --output-dir. "
                "Use the original experiment path or diagnose the manifest; "
                "checkpoint settings/data cannot be silently rewritten."
            )
        print("Using frozen Colab checkpoint settings; data hash still verified.")
    else:
        settings = release_settings(
            output_dir=args.output_dir, generations=args.generations
        )
    print(
        f"Profile={profile.name} | CPU threads={profile.xgb_threads} | "
        f"trees={settings['trees']} | full_history=True | "
        f"factory={root}",
        flush=True,
    )
    # Automatic export each completed fold/trial/member. On a Colab VM this
    # MUST be a persistent Drive/other durable path: /content is ephemeral.
    if args.command == "run":
        if profile.kind == "cloud" and not args.checkpoint_archive:
            parser.error(
                "day_cloud requires --checkpoint-archive on persistent "
                "storage, e.g. a mounted Google Drive ZIP"
            )
        from ensemble_automl_factory import run_automl_factory
        outcome, _, _ = run_automl_factory(
            **settings, runtime_threads=profile.xgb_threads,
            checkpoint_archive=args.checkpoint_archive,
        )
        print("Ensemble selection metrics:", outcome["selection_metrics"])
        print("Member weights:", outcome["effective_member_weights"])
        print("Audit remains closed. Use the separate 'audit' command once frozen.")
        return 0
    if args.command == "audit":
        from ensemble_automl_factory import run_automl_shadow_audit
        summary, _ = run_automl_shadow_audit(
            **settings, runtime_threads=profile.xgb_threads,
            checkpoint_archive=args.checkpoint_archive,
        )
        print("Frozen audit metrics:", summary["shadow_audit_metrics"])
        print("First-member reference:", summary["audit_first_reference_metrics"])
        return 0
    if args.command == "benchmark":
        from ensemble_automl_factory import FactorySettings
        from ensemble_release_candidate import benchmark_release
        report, _ = benchmark_release(
            FactorySettings(**settings),
            threads=profile.xgb_threads,
            checkpoint_archive=args.checkpoint_archive,
        )
        print("Full historical six-month benchmark:", report["ensemble"])
        print("First reference model:", report["first_reference"])
        print("Prior evaluations inspected this period: not an untouched holdout.")
        return 0
    if args.command == "finalize":
        from ensemble_automl_factory import FactorySettings
        from ensemble_release_candidate import finalize_release
        finalize_release(
            FactorySettings(**settings), threads=profile.xgb_threads,
            model_name=args.model_name,
            allow_audit_regression=args.allow_audit_regression,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
