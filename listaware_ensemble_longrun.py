"""List-aware full-strength ensemble selection + one-time chronological audit.

Consumes ONLY the previously screened and full-strength confirmed individual
model pool (feature.listaware_pool_screen). Member weights are picked from
Jun--Jul 2025 confirmation predictions, and then frozen. Later Aug--Oct
expanding-origin forecasts are separately fit/scored in audit mode.

No tuning on audit labels. Existing audit months have been inspected elsewhere
in this repository, so the shadow audit is NOT an untouched test.
This is the first multi-family *selection* runner, not a list-aware Optuna
or BoW-bagging factory; those can be added after the friend-run baseline.

  python listaware_ensemble_longrun.py select
  python listaware_ensemble_longrun.py audit
  python listaware_ensemble_longrun.py finalize --model-name selected_longrun

Defaults expect outputs/listaware_pool_confirmation_curated from the
feature.listaware_pool_screen screen + confirm notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from listaware_shotgun import _dataset_signature, _fit_predict, _frame, _verified_cache
from models.automl_checkpoint import exclusive_factory_lock, write_csv, write_json
from models.automl_search import choose_admission, log_blend, metrics, updated_weights
from models.listaware_shotgun_specs import Candidate
from spacetime_fast import _cap

KEY = ["month", "ml_num"]
AUDIT_MONTHS = ("2025-08", "2025-09", "2025-10")
VERSION = 1
DEFAULT_POOL = Path("outputs/listaware_pool_confirmation_curated")
DEFAULT_ROOT = Path("outputs/listaware_ensemble_longrun")
CUTOFF = pd.Timestamp("2026-06-01")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pool(pool_dir):
    root = Path(pool_dir)
    manpath = root / "manifest.json"
    predpath = root / "predictions.csv"
    if not (manpath.is_file() and predpath.is_file()):
        raise FileNotFoundError(
            "A complete full-strength list-aware confirmation run is required: "
            f"{root}. Run the pool screen and confirmation first."
        )
    spec = _json(manpath)
    if (spec.get("track") != "list_aware"
        or spec.get("stage") != "confirm"
        or spec.get("mode") != "expanding"
        or spec.get("full") is not True
        or tuple(spec.get("months", ())) != ("2025-06", "2025-07")
        or spec.get("tree_cap") is not None
        or spec.get("train_cap") != 200000
        or spec.get("validation_cap") != 100000):
        raise ValueError("Ensemble source is not a verified full-strength pool confirmation")
    if "cat_mae" not in spec["candidates"]:
        raise ValueError("Confirmed pool lacks the CatBoost MAE incumbent")
    df = pd.read_csv(predpath, dtype={"ml_num":str})
    df["month"] = df.month.astype(str)
    if not set(["arm", "variant", "sold_price", "predicted_price", *KEY]).issubset(df):
        raise ValueError("Missing confirmed pool prediction columns")
    if df.duplicated(KEY+["variant"]).any():
        raise ValueError("Duplicate variant/house in confirmed predictions")
    if set(df.variant) != set(spec["candidates"]) or set(df.month) != set(spec["months"]):
        raise ValueError("Confirmed pool predictions incomplete or stale")
    reference = df.loc[df.variant.eq("cat_mae")].set_index(KEY).sort_index()
    if reference.empty or not reference.index.is_unique:
        raise ValueError("Unusable reference cohort")
    y = reference.sold_price.to_numpy(float)
    streams = {}
    for name, rows in df.groupby("variant"):
        aligned = rows.set_index(KEY).sort_index()
        p = aligned.predicted_price.to_numpy(float)
        if (not aligned.index.equals(reference.index)
            or not np.array_equal(aligned.sold_price.to_numpy(float), y)
            or not np.isfinite(p).all() or (p <= 0).any()):
            raise ValueError(f"Unmatched or invalid predictions: {name}")
        streams[name] = p
    return spec, reference, streams, {
        "pool_manifest_sha256": _sha(manpath),
        "pool_predictions_sha256": _sha(predpath),
        "data_sha256": spec["data_sha256"],
    }


def select(*, pool_dir=DEFAULT_POOL, output_dir=DEFAULT_ROOT,
           max_members=8, min_gain=0.01, max_mape_regression=0.03):
    if not 1 <= max_members <= 30 or min_gain < 0 or max_mape_regression < 0:
        raise ValueError("Invalid admission settings")
    src, ref, streams, signature = _pool(pool_dir)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    settings = {"version": VERSION, "stage": "list_aware_ensemble_selection",
                "max_members": max_members, "min_gain": min_gain,
                "max_mape_regression": max_mape_regression, **signature}
    with exclusive_factory_lock(root):
        man = root / "manifest.json"
        if man.is_file() and _json(man) != settings:
            raise ValueError("Ensemble settings/input changed. Use another output dir")
        if not man.is_file():
            old = [p for p in root.iterdir() if p.name != ".run.lock"]
            if old:
                raise ValueError("Output not empty, no valid manifest")
            write_json(man, settings)
        roster_file = root / "roster.json"
        if roster_file.is_file():
            roster = _json(roster_file)
            if not (root / "selection_predictions.csv").is_file():
                raise ValueError("Selection roster exists but predictions are absent")
            print("Selection roster already frozen; reusing", roster_file)
            return roster
        y = ref.sold_price.to_numpy(float)
        weights = {"cat_mae":1.0}
        current = streams["cat_mae"].copy()
        history = [{"name":"cat_mae","weight":1.0,"reason":"fixed_reference"}]
        remaining = set(streams) - {"cat_mae"}
        while remaining and len(weights) < max_members:
            feasible = []
            for name in sorted(remaining):
                decision = choose_admission(
                    y, streams[name], current, min_gain=min_gain,
                    max_mape_regression=max_mape_regression
                )
                if decision["accept"]:
                    feasible.append((float(decision["gain"]), name, decision))
            if not feasible:
                break
            _, name, decision = max(feasible, key=lambda x:(x[0],x[1]))
            alpha = float(decision["weight"])
            current = log_blend(current, streams[name], alpha)
            weights = updated_weights(weights, name, alpha)
            remaining.remove(name)
            history.append({"name":name,"weight_at_admission":alpha,
                            "marginal_gain_pp":decision["gain"],
                            "standalone":decision["solo"],
                            "updated":decision["updated"]})
            print(f"Added {name} (marginal objective gain={decision['gain']:.5f} pp)")
        # Rooster frozen before any later audit labels are accessed.
        roster = {"version":VERSION, "weights":weights,
                  "admission_order":history,
                  "member_specs":{name:src["candidate_specs"][name]
                                  for name in weights},
                  "selection_metrics":metrics(y,current),
                  "selection_months":src["months"],
                  **signature}
        snapshot = ref.reset_index()[KEY+["sold_price"]].copy()
        snapshot["predicted_price"] = current
        # Selection decisions are immutable once saved.
        write_csv(root / "selection_predictions.csv", snapshot)
        write_json(roster_file, roster)
        print("Selected weights:", weights)
        print("Selection metrics:", roster["selection_metrics"])
        return roster


def _historical_sales(expected_hash):
    sales=data.load("listings").copy()
    sales["date_sold"]=pd.to_datetime(sales.date_sold)
    sales["date_listed"]=pd.to_datetime(sales.date_listed)
    if _dataset_signature(sales)!=expected_hash:
        raise ValueError("Historical data differ from selected pool source")
    if sales.ml_num.astype(str).duplicated().any():
        raise ValueError("Duplicate MLS numbers in historical sales")
    return sales


def _locked_roster(root):
    roster_path=root/"roster.json"
    if not roster_path.is_file():
        raise FileNotFoundError("First run list-aware ensemble selection")
    roster=_json(roster_path)
    identity=_sha(roster_path)
    return roster,identity


def audit(*, output_dir=DEFAULT_ROOT, threads=8, seed=42):
    """Open one-way locked Aug--Oct expanding-origin model evaluation."""
    if threads<1: raise ValueError("threads must be positive")
    root=Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_factory_lock(root):
        roster,digest=_locked_roster(root)
        if _json(root/"manifest.json")["data_sha256"]!=roster["data_sha256"]:
            raise ValueError("Roster/source hash mismatch")
        lockfile=root/"audit_lock.json"
        if lockfile.is_file():
            saved=_json(lockfile)
            if saved["roster_sha256"]!=digest or saved["months"]!=list(AUDIT_MONTHS):
                raise ValueError("Audit lock differs from frozen selection")
            if saved.get("status")=="complete":
                print("Audit already completed; leaving results unchanged")
                return _json(root/"audit_summary.json")
            if saved.get("status")!="pending":
                raise ValueError("Unknown audit state")
        else:
            write_json(lockfile, {"roster_sha256":digest,
                                  "months":list(AUDIT_MONTHS),"status":"pending"})
        sales=_historical_sales(roster["data_sha256"])
        month_index=sales.date_sold.dt.to_period("M")
        frames=[]
        for month in AUDIT_MONTHS:
            period=pd.Period(month,freq="M")
            train=_cap(sales.loc[sales.date_sold<period.start_time],200000,seed)
            held=_cap(sales.loc[month_index==period],100000,seed+11)
            if train.empty or held.empty: raise ValueError(f"Empty audit cohort {month}")
            hidden=data.hide_answers(held,"list_aware")
            if "sold_price" in hidden or "date_sold" in hidden or "list_price" not in hidden:
                raise ValueError("Audited prediction received forbidden columns")
            for name,serialized in roster["member_specs"].items():
                spec=Candidate(**serialized)
                path=root/"audit"/month/f"{name}.csv"
                cached=_verified_cache(path,held,period,"model",name)
                if cached is None:
                    print(f"Audit {month}/{name}: full-strength model fit",flush=True)
                    p=_fit_predict(spec,train,hidden,tree_cap=None,
                                   threads=threads,features_cache={})
                    cached=_frame(held,period,"model",name,p)
                    write_csv(path,cached)
                frames.append(cached)
        full=pd.concat(frames,ignore_index=True)
        anchor=full.loc[full.variant.eq("cat_mae")].set_index(KEY).sort_index()
        true=anchor.sold_price.to_numpy(float)
        all_members={}
        for name,group in full.groupby("variant"):
            checked=group.set_index(KEY).sort_index()
            if (not checked.index.equals(anchor.index)
                or not np.array_equal(checked.sold_price.to_numpy(float),true)):
                raise ValueError(f"Audit member {name} has mismatched homes")
            all_members[name]=checked.predicted_price.to_numpy(float)
        weighted=np.exp(sum(
            w*np.log(all_members[name]) for name,w in roster["weights"].items()
        ))
        scored=anchor.reset_index()[KEY+["sold_price"]].copy()
        scored["predicted_price"]=weighted
        summary={"version":VERSION,"roster_sha256":digest,
                 "audit_months":list(AUDIT_MONTHS),
                 "ensemble":metrics(true,weighted),
                 "cat_mae":metrics(true,all_members["cat_mae"]),
                 "weights":roster["weights"],
                 "monthly":{},"warning":"Previously inspected periods; shadow audit only"}
        for month in AUDIT_MONTHS:
            rows=scored.month.eq(month).to_numpy()
            summary["monthly"][month]=metrics(
                true[rows],weighted[rows]
            )
        write_csv(root/"audit_predictions.csv",scored)
        write_json(root/"audit_summary.json",summary)
        write_json(lockfile, {"roster_sha256":digest,
                              "months":list(AUDIT_MONTHS),"status":"complete"})
        print("Frozen list-aware audit:",summary["ensemble"])
        return summary


def finalize(*, output_dir=DEFAULT_ROOT, model_name="listaware_longrun_selected",
             threads=8):
    """Fit accepted models on all pre-Jun-2026 labeled sales; save model bundle.

    This does NOT change run.py or activate the bundle in the submission CLI.
    """
    if threads<1: raise ValueError("threads must be positive")
    root=Path(output_dir)
    with exclusive_factory_lock(root):
        roster,digest=_locked_roster(root)
        lock=root/"audit_lock.json"
        if not lock.is_file() or _json(lock).get("status")!="complete":
            raise RuntimeError("Complete the frozen audit before final full-history fit")
        if _json(lock)["roster_sha256"]!=digest:
            raise ValueError("Roster changed after audit")
        from models.listaware_selected_ensemble import ListAwareSelectedEnsemble
        from models.save_model import save_model
        sales=_historical_sales(roster["data_sha256"])
        train=sales.loc[sales.date_sold<CUTOFF].copy()
        if train.empty: raise ValueError("No legally eligible historical training data")
        model=ListAwareSelectedEnsemble(
            {name:Candidate(**serialized)
             for name,serialized in roster["member_specs"].items()},
            roster["weights"],threads=threads
        ).fit(train)
        location=save_model(model,model_name,track="list_aware",
                            metadata={"cutoff_exclusive":str(CUTOFF.date()),
                                      "training_homes":len(train),
                                      "roster_sha256":digest,
                                      "selection_source":str(root)})
        print("Saved list-aware ensemble:",location)
        return location


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("select","audit","finalize"))
    p.add_argument("--pool-dir",type=Path,default=DEFAULT_POOL)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--max-members",type=int,default=8)
    p.add_argument("--min-gain",type=float,default=0.01)
    p.add_argument("--max-mape-regression",type=float,default=0.03)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model-name",default="listaware_longrun_selected")
    args=p.parse_args(argv)
    if args.command=="select":
        return select(pool_dir=args.pool_dir,output_dir=args.output_dir,
                      max_members=args.max_members,min_gain=args.min_gain,
                      max_mape_regression=args.max_mape_regression)
    if args.command=="audit":
        return audit(output_dir=args.output_dir,threads=args.threads,seed=args.seed)
    return finalize(output_dir=args.output_dir,model_name=args.model_name,
                    threads=args.threads)


if __name__=="__main__":
    main()
