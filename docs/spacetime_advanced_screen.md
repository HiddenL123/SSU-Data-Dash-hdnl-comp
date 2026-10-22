# Advanced spatiotemporal feature screening

This branch adds **opt-in** feature families without changing the production
XGBoost baseline. The existing `SpaceTimeAddon` registry in
`models/spacetime_screen_features.py` exposes:

- `adaptive_comps`: switches between three chronological comparable-sale
  neighborhoods (0.75, 3, 9 km bandwidths) using eligible sale counts and
  effective sample size.
- `hedonic_index`: expanding, out-of-time, regularized hedonic residuals,
  aggregated across prior 3/6 months and local market segments. This is an
  experimental hedonic-adjusted market *signal*, not an official repeat-sales
  housing-price index.
- `geo_context`: distances to training-fold municipality/city centroids,
  nearest other municipality centroid **proxy** (not a real boundary), and
  trailing-12-month spatial sale counts at 1 and 5 km.
- `property_spacetime`: property-size, bedroom, bathroom and type interactions
  with geographic coordinates and listing month.
- `kalman_market`: causal local-level/trend Kalman filters for log price per
  square foot, with local and city states plus uncertainty.
- `advanced_combined`: all five advanced families.

The existing `market`, `multiscale`, `time_adjusted_comp`,
`spacetime_rbf`, `spacetime_splines`, `combined`,
`comp_type_quota`, and `comp_both` experiments remain available.

## Install and run tests

From the repo root:

```powershell
git fetch origin
git switch feature/spacetime-advanced-screen
python -m pytest -q tests/test_spacetime_advanced_features.py
```

## Stage A: low-budget shotgun screen

Run with the existing runner on an **earlier development month**. Keep
`--residuals` with no variant arguments to disable residual experiments.
The command below runs the baseline plus the prior feature candidates and
the six new variants on the same held-out properties.

```powershell
python spacetime_shotgun.py --validation-month 2025-08 --train-start 2022-01-01 --train-cap 10000 --validation-cap 1000 --screen-trees 220 --features baseline comp_type_quota comp_both market multiscale time_adjusted_comp spacetime_rbf spacetime_splines adaptive_comps hedonic_index geo_context property_spacetime kalman_market advanced_combined --residuals --output-dir outputs/spacetime_advanced_shotgun_aug
```

Inspect `outputs/spacetime_advanced_shotgun_aug/summary.csv` and
`failures.json`. **Do not** choose a winner from a single month; select
3–5 candidates for the next stage by MdAPE, MAPE and execution cost.

## Stage B: chronological confirmation on three earlier months

Replace `adaptive_comps hedonic_index geo_context` below with the winners
from Stage A:

```powershell
python spacetime_three_month.py --months 2025-08 2025-09 2025-10 --train-start 2022-01-01 --train-cap 40000 --trees 500 --validation-cap 1000000 --features baseline adaptive_comps hedonic_index geo_context --residuals --output-dir outputs/spacetime_advanced_three_month
```

This runner resumes completed months with `--no-resume` to override
its cache. Compare `pooled_summary.csv`, `monthly_summary.csv`,
and the property-type breakdown. Check whether effects persist across months
and types, not just the pooled score.

## Stage C: full-history model validation

Only after selecting features on prior development folds:

1. Register the winning feature variant in `models/features.make_features`
   so `OffMarketModel` can use it without experimental runner code.
2. Rerun the same expanding-month or fixed-origin validation as the saved
   700-tree XGBoost baseline, with the same splits/hyperparameters.
3. Use paired comparisons on identical homes and preserve an independent
   final test wherever possible. Dec 2025–May 2026 has already been
   consulted repeatedly and is **not** an untouched test.

## Leakage rules

- Labels used to form historical features must have `date_sold` in months
  **strictly before** the current home's `date_listed` month.
- All fitted providers see only the outer training fold.
- The hedonic index fits its own coefficients on earlier sale months before
  constructing each month's residual signals.
- The Kalman filter is one-sided: no smoothing backward with future sales.
- Geographic training-fold centroids use coordinates only, not sale prices.
- Production baseline source files and serialized models are unchanged.

## Test coverage and limits

`tests/test_spacetime_advanced_features.py` checks the registry, index and
numeric schema, transform without post-sale columns, and invariance of
historical features to changes in future sold prices. These tests do not
replace running the actual dataset end-to-end. The new feature families,
especially adaptive comparables and the combined configuration, may be
expensive on the full training history.
