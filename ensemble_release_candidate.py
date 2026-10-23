"""Full-history off-market refit and saved submission-ready ensemble.

The historical chronological factory decides membership/weights. This module
does NOT retune after reading shadow-audit outcomes. It refits each frozen
member on all eligible known sales, and saves fitted individual transformers
and the weighted deployable ensemble in saved_models/off_market/.

List-aware is a separate track and remains unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from ensemble_automl_factory import (
    FactoryRun, FactorySettings, _feature_name, _text_provider,
)
from models.automl_checkpoint import exclusive_factory_lock, write_json
from models.automl_search import Candidate, updated_weights
from models.ensemble_bagging import (
    BagSpec, build_text_bank, sample_training_rows, select_text_terms,
)
from models.ensemble_snapshot import BaggedMember, FeatureBaggingEnsemble
from models.features import make_features
from models.final_ensemble import WeightedOffMarketEnsemble
from save_model import load_model, model_info, save_model


def _fingerprint(roster, manifest):
    raw = json.dumps({"roster": roster, "manifest": manifest}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fit_member(train, candidate, params, cfg, *, threads):
    bank = build_text_bank(
        train, max_extra_terms=cfg.max_extra_terms, min_df=cfg.min_df
    )
    spec = BagSpec(
        name=candidate.name, seed=candidate.seed,
        text_mode=candidate.text_mode, n_text=candidate.n_text,
        selected_weight=candidate.selected_weight,
        row_mode=candidate.row_mode, row_fraction=candidate.row_fraction,
    )
    terms = select_text_terms(bank, spec)
    provider = make_features(
        _feature_name(candidate), model="xgboost", track="off_market"
    )
    text = _text_provider(provider)
    if text is None:
        raise RuntimeError("Expected full-binary training-fitted text features")
    text.terms = tuple(bank)
    X = provider.fit_transform(train)
    mapping = text.text_feature_names  # {column: term}
    inverted = {term: col for col, term in mapping.items()}
    selected_columns = (
        [col for col in X.columns if col not in mapping] +
        [inverted[term] for term in terms]
    )
    rows = sample_training_rows(len(train), spec)
    y = np.log(train.sold_price.to_numpy(dtype=float))
    offset = float(np.median(y))
    from spacetime_fast import _make_estimator
    model = _make_estimator(params, cfg.trees)
    model.set_params(n_jobs=int(threads), random_state=candidate.seed)
    model.fit(X.iloc[rows][selected_columns], y[rows] - offset)
    bundle = FeatureBaggingEnsemble(
        provider,
        {candidate.name: BaggedMember(
            name=candidate.name, estimator=model,
            selected_columns=tuple(selected_columns),
            target_offset=offset, terms=tuple(terms), settings=candidate.record(),
        )},
        metadata={
            "track": "off_market", "target": "log(sold_price)",
            "trained_sales": len(train),
            "fitting_rows": len(rows),
            "unique_fitting_rows": len(np.unique(rows)),
            "feature_family": _feature_name(candidate),
        },
    )
    sample_hidden = data.hide_answers(train.iloc[:min(len(train), 3)], "off_market")
    check = bundle.predict_member(sample_hidden, candidate.name)
    if len(check) != len(sample_hidden) or not np.isfinite(check).all() or (check <= 0).any():
        raise ValueError(f"Invalid fitted inference model: {candidate.name}")
    return bundle


def finalize_release(
    settings: FactorySettings, *, threads: int = 4,
    model_name: str = "off_market_release_candidate_full",
    allow_audit_regression: bool = False, listings=None,
):
    """Save one recoverable fitted artifact per accepted member, then ensemble.

    Gate compares the factory's already-scored chronological shadow audit
    with its first member under IDENTICAL full-fidelity audit homes. It does
    not claim independence from any previously consulted development data.
    """
    if threads < 1 or settings.trees < 700:
        raise ValueError("Release requires >=700 trees and positive CPU threads")
    factory = FactoryRun(settings, listings=listings, runtime_threads=threads)
    with exclusive_factory_lock(factory.root):
        roster_path = factory.root / "roster.json"
        audit_lock = factory.root / "shadow_audit_lock.json"
        summary_path = factory.root / "summary.json"
        if not all(p.is_file() for p in (roster_path, audit_lock, summary_path)):
            raise RuntimeError("First run selection and the explicitly sealed shadow audit")
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        lock = json.loads(audit_lock.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        roster_digest = hashlib.sha256(json.dumps(roster, sort_keys=True).encode()).hexdigest()
        if lock.get("status") != "complete" or lock.get("roster_sha256") != roster_digest:
            raise ValueError("Audit not complete for the frozen roster")
        if summary.get("shadow_audit_status") != "locked":
            raise ValueError("Missing completed shadow-audit report")
        audit = summary["shadow_audit_metrics"]
        reference = summary["audit_first_reference_metrics"]
        passes = (
            audit["mdape"] <= reference["mdape"] and
            audit["mape"] <= reference["mape"] + 0.05 and
            audit["p90_ape"] <= reference["p90_ape"] + 0.5
        )
        if not passes and not allow_audit_regression:
            raise ValueError(
                "Frozen ensemble has not passed the audit non-regression gate. "
                "Inspect summary.json or explicitly set allow_audit_regression=True "
                "to save it as an experimental fallback, not an approved winner."
            )
        manifest = json.loads(
            (factory.root / "manifest.json").read_text(encoding="utf-8")
        )
        provenance = _fingerprint(roster, manifest)
        if not roster.get("members"):
            raise ValueError("No admitted candidates")
        train = factory.sales.loc[
            factory.sales.date_sold >= pd.Timestamp(settings.train_start)
        ]
        if len(train) < 1:
            raise ValueError("No eligible historical sales for final refit")
        prices = train.sold_price.to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Invalid training sale prices")
        member_bundles = {}
        weights = {}
        for entry in roster["members"]:
            candidate = Candidate(**entry["candidate"])
            name = candidate.name
            params = dict(entry["parameters"])
            weights = updated_weights(weights, name, entry["admitted_alpha"])
            artifact_name = f"{model_name}__{name}"
            metadata = {
                "track": "off_market", "source_factory": str(factory.root),
                "frozen_roster_sha256": provenance,
                "candidate": candidate.record(), "parameters": params,
                "trees": settings.trees, "trained_sales": len(train),
                "full_history": True, "kind": "final_member",
            }
            try:
                info = model_info(artifact_name, track="off_market")
            except FileNotFoundError:
                info = None
            if info is not None:
                saved = info["metadata"]
                if (saved.get("frozen_roster_sha256") != provenance or
                    saved.get("candidate") != candidate.record() or
                    saved.get("parameters") != params):
                    raise ValueError(
                        f"Existing saved model {artifact_name} belongs to another run"
                    )
                bundle = load_model(artifact_name, track="off_market")
            else:
                print(f"Refitting full-history model {name} on {len(train):,} homes...",
                      flush=True)
                bundle = _fit_member(train, candidate, params, settings, threads=threads)
                save_model(bundle, artifact_name, track="off_market", metadata=metadata)
            member_bundles[name] = bundle
        release = WeightedOffMarketEnsemble(
            member_bundles, weights, provenance={
                "source_factory": str(factory.root),
                "frozen_roster_sha256": provenance,
                "audit": audit, "audit_first_reference": reference,
                "audit_non_regression_gate_passed": passes,
                "target": "off_market", "trees": settings.trees,
            },
        )
        try:
            existing = model_info(model_name, track="off_market")
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing["metadata"].get("frozen_roster_sha256") != provenance:
                raise ValueError("Final ensemble name already belongs to another run")
        else:
            save_model(release, model_name, track="off_market", metadata={
                "track": "off_market", "kind": "final_weighted_ensemble",
                "frozen_roster_sha256": provenance, "weights": weights,
                "audit_non_regression_gate_passed": passes,
                "full_history": True, "trees": settings.trees,
            })
        write_json(factory.root / "release_manifest.json", {
            "model_name": model_name,
            "member_names": list(member_bundles),
            "weights": weights,
            "audit_non_regression_gate_passed": passes,
            "frozen_roster_sha256": provenance,
            "trained_sales": len(train),
            "trees": settings.trees,
            "status": "ready_to_predict" if passes else "experimental_fallback",
        })
        print(f"Fitted off-market ensemble: saved_models/off_market/{model_name}", flush=True)
        return release


def predict_release(
    *, model_name="off_market_release_candidate_full",
    output_file="submissions/submission_off_market_rc.csv",
    homes=None,
):
    """Write an OFF-MARKET-only submission candidate without overwriting baseline."""
    model = load_model(model_name, track="off_market")
    homes = data.load("off_market") if homes is None else homes
    if "ml_num" not in homes or homes.ml_num.astype(str).duplicated().any():
        raise ValueError("Test homes require unique ml_num")
    if len(homes) == 0:
        raise ValueError("No off-market test homes")
    hidden = data.hide_answers(homes, "off_market")
    prices = np.asarray(model.predict(hidden), dtype=float)
    if prices.shape != (len(homes),) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Nonfinite/invalid ensemble prices")
    out = Path(output_file)
    if out.exists():
        raise FileExistsError(f"Will not overwrite an existing submission: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ml_num": homes.ml_num.to_numpy(),
        "predicted_price": np.round(prices).astype(np.int64),
    }).to_csv(out, index=False)
    return out
