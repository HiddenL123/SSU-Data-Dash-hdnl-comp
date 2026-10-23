"""Resumable, chronological, diversity-aware XGBoost ensemble AutoML (v1).

Run from ensemble_starter.ipynb (opt-in). Requires optional 'optuna'.
No model ever fits on the candidate's validation labels. Separate folds:
  tune: earlier months for Optuna hyperparameters
  select: later months for constrained ensemble admission and weights
  audit: still-later report ONLY, after membership and weights are frozen

Even the audit months may have been inspected in previous research. Audit is
shielded from THIS algorithm's selection loop; it is NOT guaranteed untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import data
from models.automl_checkpoint import (
    exclusive_factory_lock, write_json, write_csv,
)
from models.automl_search import (
    Candidate, choose_admission, generate_candidates, log_blend,
    metrics, objective, updated_weights,
)
from models.ensemble_bagging import (
    BagSpec, build_text_bank, sample_training_rows, select_text_terms,
)
from models.ensemble_snapshot import BaggedMember, FeatureBaggingEnsemble
from models.features import make_features
from models.text_features import TextAugmentedFeatures
from save_model import model_info, save_model
from spacetime_fast import XGB_TUNED, _cap, _make_estimator

VERSION = 1


@dataclass(frozen=True)
class FactorySettings:
    tuning_months: tuple[str, ...] = ("2025-03", "2025-04")
    selection_months: tuple[str, ...] = ("2025-05", "2025-06")
    audit_months: tuple[str, ...] = ("2025-07", "2025-08")
    train_start: str = "2022-01-01"
    train_cap: int = 5000
    validation_cap: int = 250
    trees: int = 220
    max_extra_terms: int = 1000
    min_df: int = 10
    candidates_per_generation: int = 2
    generations: int = 1
    optuna_trials: int = 2
    minimum_ensemble_gain_pp: float = 0.02
    max_mape_regression_pp: float = 0.03
    max_members: int = 8
    seed: int = 42
    output_dir: str = "outputs/ensemble_automl_factory_v1"
    save_accepted_snapshots: bool = False
    screen_max_mdape: float = 22.0  # crude catastrophic-model prefilter only

    def __post_init__(self):
        months = [
            tuple(pd.Period(m, freq="M") for m in block)
            for block in (self.tuning_months, self.selection_months, self.audit_months)
        ]
        if any(not block or len(set(block)) != len(block) or
               list(block) != sorted(block) for block in months):
            raise ValueError("Each fold group must contain distinct sorted months")
        if not months[0][-1] < months[1][0] or not months[1][-1] < months[2][0]:
            raise ValueError("Must satisfy tuning months < selection months < audit months")
        if months[2][-1] >= pd.Period("2025-12", freq="M"):
            raise ValueError("AutoML development and audit months must precede Dec 2025")
        if pd.Timestamp(self.train_start) >= months[0][0].start_time:
            raise ValueError("Training start is not before earliest tuning month")
        if min(
            self.train_cap, self.validation_cap, self.trees,
            self.max_extra_terms, self.min_df, self.candidates_per_generation,
            self.generations, self.optuna_trials, self.max_members,
        ) < 1:
            raise ValueError("All budgets and sampling limits must be positive")
        if (self.minimum_ensemble_gain_pp < 0 or
            self.max_mape_regression_pp < 0 or
            self.screen_max_mdape <= 0):
            raise ValueError("Admission constraints must be nonnegative")

    def record(self):
        return asdict(self)


def _text_provider(provider):
    if isinstance(provider, TextAugmentedFeatures):
        return provider
    for attr in ("base_provider", "provider"):
        item = getattr(provider, attr, None)
        if item is not None:
            result = _text_provider(item)
            if result is not None:
                return result
    return None


def _feature_name(candidate):
    suffix = "__st_" + "+".join(candidate.addons) if candidate.addons else ""
    return "Xpt_spatiotemporal__full_binary" + suffix


class FactoryRun:
    def __init__(self, settings=FactorySettings(), *, listings=None):
        self.cfg = settings
        self.root = Path(settings.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sales = data.load("listings") if listings is None else listings.copy()
        self.sales["date_listed"] = pd.to_datetime(self.sales.date_listed)
        self.sales["date_sold"] = pd.to_datetime(self.sales.date_sold)
        self.sale_months = self.sales.date_sold.dt.to_period("M")
        if self.sales.ml_num.astype(str).duplicated().any():
            raise ValueError("Monthly inputs require unique MLS identifiers")
        # Guard ALL existing predictions/studies against data/setting drift.
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(self.sales, index=True)
            .to_numpy(dtype="uint64").tobytes()
        ).hexdigest()
        settings_record = self.cfg.record()
        # Generations may be increased to continue the same frozen experiment.
        # ALL other tuning/selection/audit/data settings must remain identical.
        settings_record.pop("generations")
        setup = {"version": VERSION, "settings": settings_record,
                 "data_sha256": fingerprint, "xgb_parent": XGB_TUNED}
        setup = json.loads(json.dumps(setup, sort_keys=True))
        path = self.root / "manifest.json"
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != setup:
            raise ValueError("Factory settings/data changed: use a NEW output directory")
        if not path.is_file():
            write_json(path, setup)
        self.fold_features = {}
        self.truth_cache = {}
        events_file = self.root / "events.json"
        self.events = json.loads(events_file.read_text(encoding="utf-8")) if events_file.is_file() else []

    def _cohorts(self, month):
        period = pd.Period(month, freq="M")
        training = self.sales.loc[
            (self.sales.date_sold >= pd.Timestamp(self.cfg.train_start)) &
            (self.sales.date_sold < period.start_time)
        ]
        train = _cap(training, self.cfg.train_cap, self.cfg.seed)
        held = _cap(
            self.sales.loc[self.sale_months == period],
            self.cfg.validation_cap, self.cfg.seed + 11,
        )
        if train.empty or held.empty:
            raise ValueError(f"Empty chronological train/holdout in {period}")
        return train, held

    def _prepare_fold(self, candidate, month):
        key = (candidate.name, str(month))
        if key in self.fold_features:
            return self.fold_features[key]
        train, held = self._cohorts(month)
        bank = build_text_bank(
            train, max_extra_terms=self.cfg.max_extra_terms,
            min_df=self.cfg.min_df,
        )
        bag = BagSpec(
            name=candidate.name, seed=candidate.seed,
            text_mode=candidate.text_mode, n_text=candidate.n_text,
            selected_weight=candidate.selected_weight,
            row_mode=candidate.row_mode,
            row_fraction=candidate.row_fraction,
        )
        terms = select_text_terms(bank, bag)
        provider = make_features(
            _feature_name(candidate), model="xgboost", track="off_market"
        )
        text = _text_provider(provider)
        if text is None:
            raise RuntimeError("Expected training-fitted binary BoW provider")
        text.terms = tuple(bank)
        X = provider.fit_transform(train)
        hidden = data.hide_answers(held, "off_market")
        H = provider.transform(hidden)
        if not X.columns.equals(H.columns):
            raise ValueError("Training/holdout features not aligned")
        term_columns = text.text_feature_names
        inverse = {term: col for col, term in term_columns.items()}
        columns = [
            *(col for col in X.columns if col not in term_columns),
            *(inverse[term] for term in terms),
        ]
        row_ids = sample_training_rows(len(train), bag)
        target = np.log(train.sold_price.to_numpy(dtype=float))
        offset = float(np.median(target))
        fold = {
            "month": str(month),
            "X": X.iloc[row_ids][columns],
            "y": target[row_ids] - offset,
            "H": H[columns],
            "offset": offset,
            "held": held,
            "truth": held.sold_price.to_numpy(dtype=float),
            "ids": held.ml_num.astype(str).to_numpy(),
            "provider": provider,
            "cols": tuple(columns),
            "terms": tuple(terms),
            "rows": len(row_ids),
            "unique_rows": len(np.unique(row_ids)),
            "total_history": len(train),
        }
        # Record the exact per-fold vocabulary and feature schema. A
        # candidate's legal training vocabulary can vary across months.
        # Never materialize validation labels into this feature manifest.
        row_digest = hashlib.sha256(
            np.asarray(row_ids, dtype=np.int64).tobytes()
        ).hexdigest()
        provenance = {
            "candidate": candidate.record(),
            "month": str(month),
            "training_cutoff_exclusive": str(
                pd.Period(month, freq="M").start_time.date()
            ),
            "total_training_rows": len(train),
            "estimator_training_rows": len(row_ids),
            "unique_estimator_rows": len(np.unique(row_ids)),
            "row_index_sha256": row_digest,
            "heldout_homes": len(held),
            "feature_provider": _feature_name(candidate),
            "text_terms": list(terms),
            "selected_columns": list(columns),
        }
        details_path = (
            self.root / "members" / candidate.name / "feature_manifests" /
            f"{month}.json"
        )
        if details_path.is_file() and json.loads(
            details_path.read_text(encoding="utf-8")
        ) != json.loads(json.dumps(provenance)):
            raise ValueError(
                "Feature bag / column manifest changed: "
                f"{details_path}"
            )
        details_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(details_path, provenance)
        self.fold_features[key] = fold
        return fold

    def _predict_fold(self, candidate, month, params, *, fold_name, cache=True):
        # Important: an interrupted Optuna run may later find a better trial.
        # Include the FULL parameter set in the cache key, not just candidate ID.
        param_id = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        dest = (
            self.root / "members" / candidate.name /
            fold_name / param_id / f"{month}.csv"
        )
        if cache and dest.is_file():
            # Verify against the deterministic cohort WITHOUT rebuilding
            # the expensive training-fitted 1000-term feature matrix.
            _, held = self._cohorts(month)
            ids = held.ml_num.astype(str).to_numpy()
            truth = held.sold_price.to_numpy(dtype=float)
            df = pd.read_csv(dest, dtype={"ml_num": str})
            if (len(df) == len(ids) and
                np.array_equal(df.ml_num.to_numpy(), ids) and
                np.array_equal(
                    df.sold_price.to_numpy(dtype=float), truth
                ) and np.isfinite(df.predicted_price).all() and
                df.predicted_price.gt(0).all()):
                return df.predicted_price.to_numpy(dtype=float)
            raise ValueError(f"Stale/misaligned member cache: {dest}")

        fold = self._prepare_fold(candidate, month)
        model = _make_estimator(params, self.cfg.trees)
        model.set_params(random_state=candidate.seed)
        model.fit(fold["X"], fold["y"])
        predicted = np.exp(
            fold["offset"] + np.asarray(model.predict(fold["H"]), dtype=float)
        )
        metrics(fold["truth"], predicted)  # validate positive finite
        if cache:
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_csv(dest, pd.DataFrame({
                "ml_num": fold["ids"], "sold_price": fold["truth"],
                "predicted_price": predicted,
            }))
        return predicted

    def _assembled(self, candidate, months, params, *, stage):
        predictions, truth, keys = [], [], []
        for month in months:
            # Reading a cached prediction does not require rebuilding the
            # feature provider; cohort sampling is deterministic.
            _, held = self._cohorts(month)
            predictions.append(self._predict_fold(
                candidate, month, params, fold_name=stage, cache=True
            ))
            truth.append(held.sold_price.to_numpy(dtype=float))
            keys.extend((str(month), ident) for ident in held.ml_num.astype(str))
        p, y = np.concatenate(predictions), np.concatenate(truth)
        return pd.DataFrame({
            "month": [m for m, _ in keys],
            "ml_num": [name for _, name in keys],
            "sold_price": y, "predicted_price": p,
        }).sort_values(["month", "ml_num"]).reset_index(drop=True)

    @staticmethod
    def _aligned(current, candidate):
        if current is None:
            return candidate
        if not current[["month", "ml_num"]].equals(
            candidate[["month", "ml_num"]]
        ) or not np.array_equal(
            current.sold_price.to_numpy(), candidate.sold_price.to_numpy()
        ):
            raise ValueError("Ensemble candidates use different homes/actual prices")
        return candidate

    def _tune(self, candidate):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Install optuna (pip install optuna) to run the AutoML factory"
            ) from exc
        folder = self.root / "optuna"
        folder.mkdir(exist_ok=True)
        storage = f"sqlite:///{(folder / 'studies.db').resolve().as_posix()}"
        study = optuna.create_study(
            study_name=f"factory_v{VERSION}_{candidate.name}",
            storage=storage, direction="minimize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.cfg.seed + candidate.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        )
        if not study.trials:
            study.enqueue_trial({
                "max_depth": XGB_TUNED["max_depth"],
                "min_child_weight": XGB_TUNED["min_child_weight"],
                "learning_rate": XGB_TUNED["learning_rate"],
                "reg_lambda": XGB_TUNED["reg_lambda"],
                "subsample": XGB_TUNED["subsample"],
                "colsample_bytree": XGB_TUNED["colsample_bytree"],
            })
        def trial_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 35.0
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", .03, .16, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1.0, 50.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", .75, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", .70, 1.0
                ),
            }
            scores = []
            for month in self.cfg.tuning_months:
                # Every trial gets identical chronological folds; the fitted
                # feature context is reused but XGBoost is always fresh.
                fold = self._prepare_fold(candidate, month)
                model = _make_estimator(
                    {**XGB_TUNED, **params}, self.cfg.trees
                )
                model.set_params(random_state=candidate.seed)
                model.fit(fold["X"], fold["y"])
                prices = np.exp(
                    fold["offset"] + np.asarray(model.predict(fold["H"]), float)
                )
                scores.append(objective(fold["truth"], prices))
                trial.report(float(np.mean(scores)), step=len(scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        # After Ctrl+C or an OS/process kill, Optuna can retain RUNNING
        # trials in SQLite. Single-writer lock guarantees none are active
        # here; reclassify orphan trials as interrupted and retry their
        # budget instead of counting them as completed.
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(
                    trial.number, state=optuna.trial.TrialState.FAIL
                )
                print(
                    f"  Recovering interrupted Optuna trial #{trial.number} "
                    f"for {candidate.name}",
                    flush=True,
                )
        finished = sum(
            trial.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            ) for trial in study.trials
        )
        remaining = max(0, self.cfg.optuna_trials - finished)
        if remaining:
            study.optimize(trial_objective, n_trials=remaining, n_jobs=1)
        complete = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not complete:
            raise RuntimeError(f"No successful Optuna trial for {candidate.name}")
        params = {**XGB_TUNED, **study.best_params}
        return params, float(study.best_value)

    def _save_member(self, candidate, params):
        """Save refitted individual model at fixed selection cutoff; never audit."""
        name = f"factory_v{VERSION}_{candidate.name}"
        try:
            existing = model_info(name, track="off_market")
        except FileNotFoundError:
            existing = None
        if existing is not None:
            meta = existing.get("metadata", {})
            if (json.dumps(meta.get("candidate"), sort_keys=True) !=
                json.dumps(candidate.record(), sort_keys=True) or
                meta.get("parameters") != params):
                raise ValueError("Saved model name already belongs to different configuration")
            return
        cutoff = self.cfg.audit_months[0]
        fold = self._prepare_fold(candidate, cutoff)
        # IMPORTANT: fold training history strictly precedes audit first month.
        model = _make_estimator(params, self.cfg.trees)
        model.set_params(random_state=candidate.seed)
        model.fit(fold["X"], fold["y"])
        bundle = FeatureBaggingEnsemble(
            fold["provider"],
            {candidate.name: BaggedMember(
                name=candidate.name, estimator=model,
                selected_columns=fold["cols"],
                target_offset=fold["offset"],
                terms=fold["terms"], settings=candidate.record()
            )}
        )
        save_model(bundle, name, track="off_market", metadata={
            "candidate": candidate.record(), "parameters": params,
            "target": "log(sold_price)", "frozen_as_of": str(cutoff),
            "source_factory_dir": str(self.root), "model_type": "AutoMLCandidate",
            "warning": "Development-period model; not full 700-tree production fit",
        })
        print(f"  Saved fitted member: saved_models/off_market/{name}", flush=True)

    def _commit_candidate(self, roster, event):
        """One atomic transaction for a candidate's decision and event.

        roster.json is authoritative. If interrupted before its os.replace,
        candidate is retried; if interrupted afterward, the decision is
        replayed and events.json is repaired on restart.
        """
        name = event["candidate"]["name"]
        if name in roster["processed"]:
            raise ValueError(f"Candidate already committed: {name}")
        roster["processed"].append(name)
        roster["history"].append(event)
        write_json(self.root / "roster.json", roster)
        self.events = list(roster["history"])
        write_json(self.root / "events.json", self.events)

    def run(self):
        with exclusive_factory_lock(self.root):
            return self._run_locked()

    def _run_locked(self):
        """Run bounded generations and report final fixed-weight shadow audit."""
        roster_path = self.root / "roster.json"
        if roster_path.exists():
            roster = json.loads(roster_path.read_text(encoding="utf-8"))
            # Reconstruct current model selection by deterministic replays of
            # saved configs and cached selection predictions, never new draws.
            if roster.get("version") != VERSION:
                raise ValueError("Old roster version")
        else:
            roster = {
                "version": VERSION, "members": [], "processed": [],
                "history": [],
            }
            write_json(roster_path, roster)
        # Old v1 rosters stored events in a separate file. Migrate their
        # existing decisions only when known, then keep roster history
        # authoritative. Crash in the legacy gap cannot be fully recovered.
        if "history" not in roster:
            historic = [
                event for event in self.events
                if event.get("candidate", {}).get("name") in roster["processed"]
            ]
            roster["history"] = historic
            write_json(roster_path, roster)
        self.events = list(roster["history"])
        # Repair a crash between the atomic roster commit and rebuilding
        # events.json. Never reconstruct roster from the weaker events view.
        write_json(self.root / "events.json", self.events)
        configs, weights, current = {}, {}, None
        # Replay previously admitted members using their frozen parameters,
        # without refitting Optuna. Reuse cached member predictions.
        for member in roster["members"]:
            candidate = Candidate(**member["candidate"])
            configs[candidate.name] = member
            pred = self._assembled(
                candidate, self.cfg.selection_months, member["parameters"],
                stage="selection",
            )
            if current is not None:
                self._aligned(current, pred)
                pred = pred.copy()
                pred["predicted_price"] = log_blend(
                    current.predicted_price, pred.predicted_price,
                    member["admitted_alpha"],
                )
            current = pred
            weights = updated_weights(
                weights, candidate.name, member["admitted_alpha"]
            )
        # If stopped after accepting a candidate but before serializing its
        # snapshot, rebuild that missing artifact on restart. Completed
        # members are preserved; only missing models are refitted.
        if self.cfg.save_accepted_snapshots:
            for member in roster["members"]:
                self._save_member(
                    Candidate(**member["candidate"]),
                    member["parameters"],
                )
        processed = set(roster["processed"])
        audit_lock = self.root / "shadow_audit_lock.json"
        if audit_lock.is_file():
            remaining = [
                candidate.name
                for generation in range(self.cfg.generations)
                for candidate in generate_candidates(
                    generation, self.cfg.candidates_per_generation, self.cfg.seed
                ) if candidate.name not in processed
            ]
            if remaining:
                raise RuntimeError(
                    "Shadow audit has already been opened; do not extend "
                    "ensemble selection against its known results. "
                    "Start an independent new output directory."
                )
        for generation in range(self.cfg.generations):
            for candidate in generate_candidates(
                generation, self.cfg.candidates_per_generation, self.cfg.seed
            ):
                if candidate.name in processed:
                    continue
                if len(weights) >= self.cfg.max_members:
                    self._commit_candidate(roster, {
                        "generation": generation, "candidate": candidate.record(),
                        "admitted": False, "reason": "ensemble_member_cap_reached",
                    })
                    continue
                print(f"\n[{generation+1}/{self.cfg.generations}] "
                      f"Trying {candidate.name}: {candidate.record()}", flush=True)
                t0 = perf_counter()
                # Stage A: cheap untuned screen on FIRST earlier tuning fold.
                baseline_params = dict(XGB_TUNED)
                from_time = self._predict_fold(
                    candidate, self.cfg.tuning_months[0], baseline_params,
                    fold_name="screen",
                )
                _, held = self._cohorts(self.cfg.tuning_months[0])
                screen = metrics(
                    held.sold_price.to_numpy(dtype=float), from_time
                )
                # Cheap catastrophic-model rejection before any Optuna calls.
                # This is a predefined absolute cutoff, NOT adaptively fitted
                # to an outer validation or audit result.
                if screen["mdape"] > self.cfg.screen_max_mdape:
                    self._commit_candidate(roster, {
                        "generation": generation,
                        "candidate": candidate.record(),
                        "screen": screen, "admitted": False,
                        "reason": "failed_predeclared_screen_cutoff",
                    })
                    self.fold_features.clear()
                    print("  Skipped Optuna: poor cheap screen", flush=True)
                    continue
                # Stage B: strict earlier-fold Optuna tuning of survivors.
                params, tuned_score = self._tune(candidate)
                prediction = self._assembled(
                    candidate, self.cfg.selection_months, params,
                    stage="selection",
                )
                self._aligned(current, prediction)
                current_prediction = (
                    current.predicted_price.to_numpy() if current is not None else None
                )
                decision = choose_admission(
                    prediction.sold_price.to_numpy(),
                    prediction.predicted_price.to_numpy(),
                    current_prediction,
                    min_gain=self.cfg.minimum_ensemble_gain_pp,
                    max_mape_regression=self.cfg.max_mape_regression_pp,
                )
                # Diversity is descriptive. Correlation alone is not a
                # rejection criterion; marginal paired ensemble gain is.
                error_corr = None
                if current_prediction is not None:
                    truth_log = np.log(prediction.sold_price.to_numpy(dtype=float))
                    old_errors = truth_log - np.log(current_prediction)
                    new_errors = truth_log - np.log(
                        prediction.predicted_price.to_numpy(dtype=float)
                    )
                    if np.std(old_errors) > 0 and np.std(new_errors) > 0:
                        error_corr = float(
                            np.corrcoef(old_errors, new_errors)[0, 1]
                        )
                accept = bool(decision["accept"] and len(weights) < self.cfg.max_members)
                record = {
                    "generation": generation, "candidate": candidate.record(),
                    "params": params, "tuning_score": tuned_score,
                    "screen": screen,
                    "admitted": accept,
                    "weight": decision["weight"] if accept else 0.,
                    "marginal_gain": decision["gain"],
                    "solo": decision["solo"],
                    "error_corr_with_ensemble": error_corr,
                    "updated_metrics": decision["updated"] if accept else
                        (metrics(current.sold_price, current.predicted_price)
                         if current is not None else {}),
                    "elapsed_seconds": perf_counter() - t0,
                }
                if accept:
                    alpha = decision["weight"]
                    if current is None:
                        current = prediction.copy()
                    else:
                        current = current.copy()
                        current["predicted_price"] = log_blend(
                            current.predicted_price, prediction.predicted_price, alpha
                        )
                    weights = updated_weights(weights, candidate.name, alpha)
                    roster["members"].append({
                        "candidate": candidate.record(), "parameters": params,
                        "admitted_alpha": alpha,
                    })
                # Atomic roster commit BEFORE potentially slow model snapshot
                # serialization. On restart, a missing snapshot is recovered
                # from this already-committed member/configuration.
                self._commit_candidate(roster, record)
                if accept and self.cfg.save_accepted_snapshots:
                    self._save_member(candidate, params)
                self.fold_features.clear()  # avoid accumulating 1k-BoW matrices
                print(f"  {'ADMIT' if accept else 'reject'} "
                      f"gain={decision['gain']} ensemble size={len(weights)}",
                      flush=True)
        if not roster["members"]:
            raise RuntimeError("No accepted model; reference candidate missing")
        selection_score = metrics(
            current.sold_price.to_numpy(), current.predicted_price.to_numpy()
        )
        write_csv(self.root / "selection_ensemble_predictions.csv", current)
        selection_monthly = {
            month: metrics(
                current.loc[current.month == month, "sold_price"].to_numpy(),
                current.loc[current.month == month, "predicted_price"].to_numpy(),
            ) for month in self.cfg.selection_months
        }
        outcome = {
            "selection_metrics": selection_score,
            "selection_monthly_metrics": selection_monthly,
            "effective_member_weights": weights,
            "members": [entry["candidate"]["name"] for entry in roster["members"]],
            "shadow_audit_status": (
                "locked" if audit_lock.is_file() else "unopened"
            ),
        }
        summary_path = self.root / "summary.json"
        # Do not destroy a previous audit summary during harmless resume.
        if not audit_lock.is_file():
            write_json(summary_path, outcome)
        print("\nAutoML ensemble selection:", selection_score, flush=True)
        print(
            "Shadow audit is LOCKED." if audit_lock.is_file() else
            "Shadow audit NOT opened; explicitly enable its separate cell.",
            flush=True,
        )
        return outcome, pd.DataFrame(self.events), current

    def audit(self):
        with exclusive_factory_lock(self.root):
            return self._audit_locked()

    def _audit_locked(self):
        """Open a one-time shadow diagnostic AFTER roster/weights are frozen.

        Audit metrics never influence candidate acceptance. Once opened, the
        factory refuses to add new generations in this same output directory,
        preventing retrospective audit-score optimization via iterative runs.
        """
        roster_path = self.root / "roster.json"
        if not roster_path.is_file():
            raise FileNotFoundError("First complete the AutoML selection run")
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        if not roster.get("members"):
            raise ValueError("No accepted ensemble members to audit")
        required = {
            candidate.name
            for generation in range(self.cfg.generations)
            for candidate in generate_candidates(
                generation, self.cfg.candidates_per_generation, self.cfg.seed
            )
        }
        if not required.issubset(set(roster["processed"])):
            raise RuntimeError(
                "Factory selection has not finished all requested generations"
            )
        checksum = hashlib.sha256(
            json.dumps(roster, sort_keys=True).encode("utf-8")
        ).hexdigest()
        lock_path = self.root / "shadow_audit_lock.json"
        if lock_path.is_file():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if lock.get("roster_sha256") != checksum:
                raise RuntimeError(
                    "Selected model roster changed after the shadow audit"
                )
            if lock.get("status", "complete") == "complete":
                summary = json.loads(
                    (self.root / "summary.json").read_text(encoding="utf-8")
                )
                saved = pd.read_csv(
                    self.root / "shadow_audit_predictions.csv",
                    dtype={"ml_num": str},
                )
                return summary, saved
            if lock.get("status") != "pending":
                raise ValueError("Unknown audit checkpoint status")
            print("Resuming interrupted shadow audit...", flush=True)
        else:
            # Audit lock MUST be written before any audit calculation or
            # output. Otherwise a crash after revealing audit scores but
            # before locking would allow continued selection on that result.
            write_json(lock_path, {
                "roster_sha256": checksum,
                "members": [
                    m["candidate"]["name"] for m in roster["members"]
                ],
                "audit_months": list(self.cfg.audit_months),
                "status": "pending",
            })
        weights = {}
        streams = []
        for member in roster["members"]:
            candidate = Candidate(**member["candidate"])
            weights = updated_weights(
                weights, candidate.name, member["admitted_alpha"]
            )
            streams.append((
                candidate.name,
                self._assembled(
                    candidate, self.cfg.audit_months,
                    member["parameters"], stage="audit"
                ),
            ))
            self.fold_features.clear()
        reference = streams[0][1]
        for _, other in streams[1:]:
            self._aligned(reference, other)
        true = reference.sold_price.to_numpy(dtype=float)
        values = np.exp(sum(
            weights[name] * np.log(frame.predicted_price.to_numpy(dtype=float))
            for name, frame in streams
        ))
        audit_score = metrics(true, values)
        audit_reference = metrics(
            true, reference.predicted_price.to_numpy(dtype=float)
        )
        results = reference[["month", "ml_num", "sold_price"]].copy()
        results["predicted_price"] = values
        selection = json.loads(
            (self.root / "summary.json").read_text(encoding="utf-8")
        )
        audit_monthly = {
            month: metrics(
                results.loc[results.month == month, "sold_price"].to_numpy(),
                results.loc[results.month == month, "predicted_price"].to_numpy(),
            ) for month in self.cfg.audit_months
        }
        selection.update({
            "shadow_audit_status": "locked",
            "shadow_audit_metrics": audit_score,
            "shadow_audit_monthly_metrics": audit_monthly,
            "audit_first_reference_metrics": audit_reference,
            "warning": (
                "Shadow audit consulted ONCE after factory selection. "
                "Historically these months may have been examined elsewhere."
            ),
        })
        # Pending audit lock is already durable. Independently atomic
        # artifacts can be regenerated from cached predictions after a
        # crash; final status is committed LAST.
        write_csv(self.root / "shadow_audit_predictions.csv", results)
        write_json(self.root / "summary.json", selection)
        write_json(lock_path, {
            "roster_sha256": checksum,
            "members": selection["members"],
            "audit_months": list(self.cfg.audit_months),
            "status": "complete",
        })
        print("Frozen shadow audit:", audit_score, flush=True)
        return selection, results

def run_automl_factory(*, listings=None, **kwargs):
    """Notebook entry point. Named kwargs are FactorySettings constructor args."""
    return FactoryRun(FactorySettings(**kwargs), listings=listings).run()



def run_automl_shadow_audit(*, listings=None, **kwargs):
    """Opt-in one-time audit of a previously frozen factory selection."""
    return FactoryRun(FactorySettings(**kwargs), listings=listings).audit()
