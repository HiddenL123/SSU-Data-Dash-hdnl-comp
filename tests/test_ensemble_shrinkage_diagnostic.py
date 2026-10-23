"""Unit tests for prediction-only residual shrinkage vs blending diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest

from ensemble_shrinkage_diagnostic import run_shrinkage_vs_blending


def example_predictions():
    # Mixed residual signs, varying base estimates, and two chronological months
    # exercise alignment, paired comparisons and per-month bootstrap.
    base = np.array([100, 200, 350, 450, 120, 220, 320, 420.], float)
    actual = np.array([99, 230, 330, 490, 110, 260, 285, 455.], float)
    g_eff = np.array([.10, -.08, .05, -.10, .03, -.07, .04, -.04])
    s_eff = np.array([-.04, -.02, .10, -.03, -.02, -.07, .08, .07])
    st_eff = np.array([.08, -.12, .02, .03, .09, -.05, .01, -.09])
    streams = {
        "baseline": base,
        "xgb_tensor_gam": base * np.exp(g_eff),
        "xgb_spatial_matern_tuned": base * np.exp(s_eff),
        "xgb_spacetime_matern_tuned": base * np.exp(st_eff),
    }
    return pd.DataFrame([
        {
            "month": "2025-09" if i < 4 else "2025-10",
            "ml_num": f"h{i:02d}",
            "variant": variant, "sold_price": actual[i],
            "predicted_price": float(values[i]),
        }
        for variant, values in streams.items()
        for i in range(len(base))
    ])


def _variant(df, name):
    return df[df.variant == name].sort_values(
        ["month", "ml_num"]
    ).predicted_price.to_numpy()


def test_shrinkage_vs_blend_algebra_and_fixed_50_50(tmp_path):
    source = example_predictions()
    pooled, monthly, paired, corr, created = run_shrinkage_vs_blending(
        source, bootstrap_repeats=100, seed=12,
        output_dir=tmp_path,
    )
    baseline = _variant(source, "baseline")
    gam = _variant(source, "xgb_tensor_gam")
    spatial = _variant(source, "xgb_spatial_matern_tuned")
    spacetime = _variant(source, "xgb_spacetime_matern_tuned")
    # Each shrinkage is interpolation between baseline and independently
    # corrected prediction ON THE LOG scale.
    np.testing.assert_allclose(
        _variant(created, "gam_half_strength"),
        np.sqrt(baseline * gam),
    )
    np.testing.assert_allclose(
        _variant(created, "spacetime_half_strength"),
        np.sqrt(baseline * spacetime),
    )
    np.testing.assert_allclose(
        _variant(created, "spatial_half_strength"),
        np.sqrt(baseline * spatial),
    )
    # Reproduces the existing 50/50 geometric ensemble exactly.
    np.testing.assert_allclose(
        _variant(created, "gam_spacetime_50_50"),
        np.sqrt(gam * spacetime),
    )
    np.testing.assert_allclose(
        _variant(created, "gam_spatial_50_50"),
        np.sqrt(gam * spatial),
    )
    np.testing.assert_allclose(
        _variant(created, "gam_spacetime_blend_half_strength"),
        np.sqrt(baseline * np.sqrt(gam * spacetime)),
    )
    assert set(paired.reference).issuperset({
        "gam_half_strength",
        "spacetime_half_strength",
        "spatial_half_strength",
    })
    assert len(paired) == 12
    assert len(corr) == 2
    assert sorted(monthly.month.unique().tolist()) == ["2025-09", "2025-10"]
    assert paired.mdape_ci_low.notna().all()
    assert paired.mape_ci_high.notna().all()
    assert (tmp_path / "pooled.csv").exists()
    assert (tmp_path / "monthly.csv").exists()
    assert (tmp_path / "pairwise.csv").exists()
    assert (tmp_path / "predictions.csv").exists()
    setup = json.loads((tmp_path / "setup.json").read_text())
    assert setup["homes"] == 8
    assert setup["shrinkage"] == .5
    # Results must be unchanged by input row ordering.
    again = run_shrinkage_vs_blending(
        source.sample(frac=1, random_state=4), bootstrap_repeats=100,
        seed=12, output_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(pooled, again[0])
    pd.testing.assert_frame_equal(paired, again[2])


def test_reject_misaligned_or_inconsistent_input():
    source = example_predictions()
    kwargs = {"bootstrap_repeats": 0}
    missing = source[~(
        (source.variant == "xgb_tensor_gam") &
        (source.ml_num == "h01")
    )]
    with pytest.raises(ValueError, match="same homes"):
        run_shrinkage_vs_blending(missing, **kwargs)

    inconsistent = source.copy()
    inconsistent.loc[
        inconsistent.variant == "xgb_tensor_gam", "sold_price"
    ] += 100
    with pytest.raises(ValueError, match="Disagreement"):
        run_shrinkage_vs_blending(inconsistent, **kwargs)

    with pytest.raises(ValueError, match="Duplicate"):
        run_shrinkage_vs_blending(
            pd.concat([source, source.iloc[[0]]]), **kwargs
        )

    negative = source.copy()
    negative.loc[0, "predicted_price"] = -1
    with pytest.raises(ValueError, match="positive"):
        run_shrinkage_vs_blending(negative, **kwargs)

    with pytest.raises(ValueError, match="Missing required"):
        run_shrinkage_vs_blending(
            source[source.variant != "baseline"], **kwargs
        )
    with pytest.raises(ValueError, match="shrinkage"):
        run_shrinkage_vs_blending(source, shrinkage=1, **kwargs)


def test_report_directory_refuses_changed_source_or_settings(tmp_path):
    source = example_predictions()
    run_shrinkage_vs_blending(
        source, output_dir=tmp_path, bootstrap_repeats=0
    )
    with pytest.raises(ValueError, match="different"):
        run_shrinkage_vs_blending(
            source, output_dir=tmp_path, bootstrap_repeats=50
        )
    altered = source.copy()
    altered.loc[0, "predicted_price"] *= 1.1
    with pytest.raises(ValueError, match="different"):
        run_shrinkage_vs_blending(
            altered, output_dir=tmp_path, bootstrap_repeats=0
        )
