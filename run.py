"""Run the whole pipeline. From this folder:

    python run.py validate   test the ORIGINAL baselines chronologically
    python run.py report     score those predictions and draw a figure
    python run.py predict    load both finalized ensembles and write submissions
    python run.py check      check the submission files before you hand them in

On Windows, type py instead of python if python is not found.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models.model as models
from check_submission import check_all

ROOT = Path(__file__).parent
TRACKS = ["off_market", "list_aware"]
# Walk-forward validation: each of these months is predicted by models trained only
# on the sales before that month, the same way the real test works.
VALIDATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


def validate():
    listings = data.load("listings")
    results = []
    for month in VALIDATION_MONTHS:
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        for track in TRACKS:
            model = models.make_model(track).fit(train)
            predicted = model.predict(data.hide_answers(homes, track))
            results.append(pd.DataFrame({
                "ml_num": homes["ml_num"], "track": track, "month": str(month),
                "house_type_name": homes["house_type_name"], "sold_price": homes["sold_price"],
                "predicted_price": np.round(predicted)}))
        print(f"{month}: trained on {len(train):,} earlier sales, predicted {len(homes):,} homes")
    path = ROOT / "outputs" / "validation_predictions.csv"
    path.parent.mkdir(exist_ok=True)
    pd.concat(results).to_csv(path, index=False)
    print(f"Wrote {path.relative_to(ROOT)}")


def bootstrap_ci(values, statistic, repeats=1000):
    """95% interval: recompute the statistic on 1,000 resamples of the homes
    (drawn with replacement) and keep the middle 95% of the results."""
    rng = np.random.default_rng(0)
    results = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(x), 2) for x in np.percentile(results, [2.5, 97.5])]


def metrics(predictions):
    """Scores for one track. Errors are in percent of the sold price."""
    signed = 100 * (predictions["predicted_price"] - predictions["sold_price"]) / predictions["sold_price"]
    absolute = signed.abs().to_numpy()
    return {
        "n": len(predictions),
        "mape": round(float(np.mean(absolute)), 2),          # mean absolute % error
        "mape_95ci": bootstrap_ci(absolute, np.mean),
        "mdape": round(float(np.median(absolute)), 2),       # median absolute % error
        "mdape_95ci": bootstrap_ci(absolute, np.median),
        "median_signed_error": round(float(np.median(signed)), 2),  # above 0: predicts too high
    }


def report():
    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    predictions = pd.read_csv(path)
    scores = {track: metrics(predictions[predictions["track"] == track]) for track in TRACKS}

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "validation_report.json").write_text(json.dumps(scores, indent=2))
    lines = [
        "# Validation report", "",
        f"Walk-forward validation, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}: each month was "
        "predicted by models trained only on the sales before it.",
        "Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.", "",
        "| Track | Homes | MAPE | MdAPE | Median signed error |", "|---|---|---|---|---|"]
    for track, s in scores.items():
        lines.append(f"| {track} | {s['n']:,} | {s['mape']:.2f}% ({s['mape_95ci'][0]:.2f} to {s['mape_95ci'][1]:.2f}) "
                     f"| {s['mdape']:.2f}% ({s['mdape_95ci'][0]:.2f} to {s['mdape_95ci'][1]:.2f}) "
                     f"| {s['median_signed_error']:+.2f}% |")
    lines += [
        "", "MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.",
        "A median signed error above 0 means the model tends to predict too high.",
        "The intervals treat homes as independent. Homes sold in the same month share the same",
        "market swings, so the real uncertainty is somewhat wider.", "",
        "![Error by price range](../figures/error_by_price_range.png)", ""]
    (ROOT / "reports" / "validation_report.md").write_text("\n".join(lines))
    write_latex_table(scores)
    error_by_price_range(predictions)
    print("\n".join(lines))
    print("Wrote reports/validation_report.md, reports/validation_report.json, reports/results_table.tex "
          "and figures/error_by_price_range.png")


def write_latex_table(scores):
    """The same scores as a LaTeX table, for report/report.tex to include."""
    rows = []
    for track, s in scores.items():
        name = track.replace("_", "-")
        rows.append(f"{name} & {s['n']:,} & {s['mdape']:.2f} ({s['mdape_95ci'][0]:.2f}--{s['mdape_95ci'][1]:.2f}) "
                    f"& {s['mape']:.2f} ({s['mape_95ci'][0]:.2f}--{s['mape_95ci'][1]:.2f}) "
                    f"& {s['median_signed_error']:+.2f} \\\\")
    table = "\n".join([
        "% Written by `python run.py report`. Do not edit by hand: rerun the command instead.",
        "\\begin{tabular}{lrccc}", "\\hline",
        "Track & Homes & MdAPE \\% (95\\% CI) & MAPE \\% (95\\% CI) & Median signed error \\% \\\\", "\\hline",
        *rows, "\\hline", "\\end{tabular}"])
    (ROOT / "reports" / "results_table.tex").write_text(table + "\n")


def error_by_price_range(predictions):
    """Figure: how big the errors are, from the cheapest tenth of homes to the dearest.

    Want more figures? Copy this function and group the errors by something else:
    house_type_name, month, or community (add it to the predictions file in validate()).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"off_market": "#2a78d6", "list_aware": "#eb6834"}
    for track in TRACKS:
        rows = predictions[predictions["track"] == track]
        error = 100 * (rows["predicted_price"] - rows["sold_price"]).abs() / rows["sold_price"]
        decile = pd.qcut(rows["sold_price"], 10, labels=False) + 1   # 1 = cheapest tenth
        by_decile = error.groupby(decile)
        middle = by_decile.median()
        ax.plot(middle.index, middle.values, marker="o", linewidth=2, color=colours[track], label=track)
        ax.fill_between(middle.index, by_decile.quantile(0.25), by_decile.quantile(0.75),
                        color=colours[track], alpha=0.15, linewidth=0)
    prices = rows["sold_price"].groupby(decile).median()
    ax.set_xticks(prices.index, [f"{d}\n${p / 1e6:.2f}M" for d, p in prices.items()])
    ax.set_xlabel("Sold-price decile (1 = cheapest tenth of homes), with its median price")
    ax.set_ylabel("Absolute % error\n(line: median, band: middle 50%)")
    ax.set_title(f"Validation error by price range, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "error_by_price_range.png", dpi=150)
    plt.close(fig)


def predict():
    """Write both submissions using trusted, fitted models (no retraining)."""
    import os
    import tempfile
    from models.ensemble_serving import EnsembleServingInterface
    from check_submission import check_file

    serving = EnsembleServingInterface(
        off_market_name=os.environ.get(
            "OFF_MARKET_MODEL_NAME", "off_market_release_candidate_full"
        ),
        list_aware_name=os.environ.get(
            "LIST_AWARE_MODEL_NAME", "list_aware_ensemble_v1"
        ),
        save_dir=os.environ.get("SAVED_MODELS_DIR") or None,
    )
    serving.load()  # Fail before writing if either snapshot is unavailable.
    prepared, expected_ids = {}, {}
    for track in TRACKS:
        homes = data.load(track)
        if homes.empty or "ml_num" not in homes.columns:
            raise ValueError(f"{track}: expected nonempty homes with ml_num")
        if homes["ml_num"].isna().any():
            raise ValueError(f"{track}: missing ml_num")
        ids = homes["ml_num"].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"{track}: duplicate ml_num")
        hidden = data.hide_answers(homes, track)
        predictions = serving.predict(track, hidden)
        rounded = np.rint(predictions)
        if (not np.isfinite(rounded).all()
                or (rounded <= 0).any()
                or (rounded > np.iinfo(np.int64).max).any()):
            raise ValueError(f"{track}: invalid rounded predicted prices")
        prepared[track] = pd.DataFrame({
            "ml_num": ids.to_numpy(),
            "predicted_price": rounded.astype(np.int64),
        })
        expected_ids[track] = ids

    output = ROOT / "submissions"
    output.mkdir(parents=True, exist_ok=True)
    # Validate BOTH staged CSVs before replacing either final output.
    with tempfile.TemporaryDirectory(prefix=".submission_", dir=output) as temp_dir:
        temp = Path(temp_dir)
        for track in TRACKS:
            path = temp / f"submission_{track}.csv"
            prepared[track].to_csv(path, index=False)
            issues = check_file(path, expected_ids[track])
            if issues:
                raise ValueError(f"{track}: {'; '.join(issues)}")
        for track in TRACKS:
            path = output / f"submission_{track}.csv"
            (temp / path.name).replace(path)
            print(f"Wrote {path.relative_to(ROOT)} ({len(prepared[track]):,} homes)")


def check():
    return check_all(ROOT / "submissions")


COMMANDS = {"validate": validate, "report": report, "predict": predict, "check": check}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    result = COMMANDS[sys.argv[1]]()
    sys.exit(1 if result is False else 0)
