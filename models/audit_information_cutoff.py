"""Temporal leakage audit for ComparableSalesFeatures in SSU housing predictions.

From your project root:
    from models.audit_information_cutoff import audit_fold
    audit_fold(train, validation, cutoff="2026-05-01")

Or run this file directly to execute synthetic chronological tests:
    python models/audit_information_cutoff.py

The audit checks comparable features only. Your full validation function must
also fit CatBoost and other preprocessing on training data for each fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

try:
    from features.spatiotemporal_features import ComparableSalesFeatures
except ImportError:
    from spatiotemporal_features import ComparableSalesFeatures


DEFAULTS = dict(candidates=128, neighbors=32, radius_km=4.0,
                half_life_days=180, max_age_days=730, min_comps=2)


def _build(history: pd.DataFrame, params: dict) -> ComparableSalesFeatures:
    return ComparableSalesFeatures(**params).fit(history)


def _compare(a: pd.DataFrame, b: pd.DataFrame, label: str):
    assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-6,
                       check_dtype=True, check_like=False,
                       obj=label)


def audit_fold(train: pd.DataFrame, validation: pd.DataFrame,
               cutoff: str | pd.Timestamp, sample_size: int = 40,
               **feature_params) -> None:
    """Run chronological integrity tests against one expanding validation fold.

    Args:
        train: exactly the sales passed to the model's fit() in this fold.
        validation: sales being predicted; should include sold_price/date_sold
            for retrospective validation only.
        cutoff: start of the held-out sale month, e.g. '2026-05-01'.
        sample_size: number of held-out homes to inspect for metamorphic tests.
        **feature_params: use the SAME parameters as your real experiment.
    """
    params = {**DEFAULTS, **feature_params}
    cutoff = pd.Timestamp(cutoff).normalize()
    assert cutoff.day == 1, 'Expected first day of held-out sale month'
    assert len(train) and len(validation), 'Nonempty train and validation required'
    sold_train = pd.to_datetime(train['date_sold'], errors='raise')
    assert bool((sold_train < cutoff).all()), (
        'LEAK: training fold contains sale on/after evaluation cutoff')
    assert bool((pd.to_datetime(validation['date_sold'], errors='raise') >= cutoff).all()), (
        'Validation fold contains a sale before the evaluation cutoff')
    listed = pd.to_datetime(validation['date_listed'], errors='raise')
    sold_val = pd.to_datetime(validation['date_sold'], errors='raise')
    assert bool((listed <= sold_val).all()), (
        'Some properties have date_listed after date_sold; inspect date semantics')

    # A distinct prediction pipeline must be created per fold. Verify this
    # comparable history does not itself contain sales on/after the fold cutoff.
    model = _build(train, params)
    last_history_sale = pd.to_datetime(
        int(model._history['sold_day'].max()), unit='D', origin='unix')
    assert last_history_sale < cutoff, (
        'LEAK: comparable history extends into evaluation period')

    # Instrument the actual neighbor search; every candidate supplied to it
    # must already have sold STRICTLY before the first day of listing month.
    observed = [0]
    original = model._one
    def checked_one(coord, area, property_type, cutoff_day, hist, tree):
        if len(hist):
            assert bool((hist['sold_day'].to_numpy() < cutoff_day).all()), (
                'LEAK: neighbor pool contains sale during/after listing month')
        observed[0] += 1
        return original(coord, area, property_type, cutoff_day, hist, tree)
    model._one = checked_one

    query = validation.sample(n=min(sample_size, len(validation)), random_state=42)
    safe = model.transform(query)
    assert observed[0] > 0, 'No valid sample homes reached neighbor search'
    # A target home must also be unable to see its own future sold price when
    # its features are constructed as part of the training matrix.
    train_sample = train.sample(n=min(10, len(train)), random_state=17)
    model.transform(train_sample)

    # Adversarial test: deliberately add ALL held-out sales to the reference
    # history, with absurdly inflated labels. Query features must be identical
    # because these labels are later than each queried listing-month cutoff.
    val_poison = validation.copy()
    val_poison['sold_price'] = pd.to_numeric(
        val_poison['sold_price'], errors='raise') * 1_000
    contaminated_history = pd.concat([train, val_poison], ignore_index=True)
    contaminated = _build(contaminated_history, params).transform(query)
    _compare(safe, contaminated, 'held-out-sale-label injection')

    # The feature result must not depend on other rows in the prediction batch.
    one = model.transform(query.iloc[[0]])
    _compare(safe.iloc[[0]], one, 'prediction-batch independence')

    print(f'PASS: {len(train):,} fitting sales sold before {cutoff.date()}')
    print(f'PASS: {observed[0]} neighbor searches with strictly past sale dates')
    print('PASS: injecting future validation prices does not change features')
    print('PASS: single-home and batch predictions generate identical features')
    print('NOTE: audit only covers comparable features; not other fit stages.')


def _synthetic_tests():
    s = pd.DataFrame({
        'date_listed': ['2025-01-04', '2025-02-01', '2025-02-02',
                        '2025-03-08', '2025-04-01', '2025-04-10'],
        'date_sold': ['2025-01-20', '2025-02-15', '2025-02-28',
                      '2025-03-19', '2025-04-21', '2025-04-22'],
        'sold_price': [500_000, 550_000, 570_000, 620_000, 680_000, 710_000],
        'latitude': [43.700, 43.701, 43.7005, 43.7003, 43.7009, 43.7011],
        'longitude': [-79.40, -79.4005, -79.4008, -79.401,
                      -79.3995, -79.4001],
        'house_area': ['1000', '1050', '1080', '1120', '1140', '1180'],
        'house_type_name': ['Detached'] * 6,
    })
    train, valid = s.iloc[:4].copy(), s.iloc[4:].copy()
    audit_fold(train, valid, cutoff='2025-04-01', sample_size=2,
               min_comps=1, candidates=4, neighbors=3)

    # Explicitly test an earlier home's training features don't change when
    # a subsequent house's outcome is altered by a factor of 1000.
    old = _build(s, dict(DEFAULTS, min_comps=1)).transform(s.iloc[[1]])
    altered = s.copy()
    altered.loc[altered.index >= 2, 'sold_price'] *= 1_000
    new = _build(altered, dict(DEFAULTS, min_comps=1)).transform(s.iloc[[1]])
    _compare(old, new, 'future sale label perturbation on training row')
    print('PASS: perturbing later training sale prices cannot change earlier rows')


if __name__ == '__main__':
    _synthetic_tests()
