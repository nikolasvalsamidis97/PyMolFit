from types import SimpleNamespace

import numpy as np
import pytest

from pymolfit._wavelength_validation import (
    build_wavelength_validation_plan,
    evaluate_wavelength_validation,
    score_wavelength_validation_fold,
)


def _part(size=1000, centers=(80, 220, 360, 500, 640, 780, 920)):
    pixels = np.arange(size, dtype=float)
    transmission = np.ones(size)
    for center in centers:
        transmission -= 0.4 * np.exp(-0.5 * ((pixels - center) / 3.0) ** 2)
    return SimpleNamespace(
        spectrum=SimpleNamespace(
            wavelength=1 + pixels * 1e-5,
            flux=np.full(size, 100.0),
            uncertainty=np.full(size, 2.0),
            valid=np.ones(size, dtype=bool),
        ),
        transmission=transmission,
        fit_mask=np.ones(size, dtype=bool),
        model_flux=np.full(size, 99.0),
        continuum=np.full(size, 120.0),
        success=True,
    )


def _plan(part, **kwargs):
    return build_wavelength_validation_plan(
        [part], continuum_order=2, lsf_sigma_pixels=1.0, **kwargs
    )


def test_plan_withholds_whole_cores_and_guards_without_training_leakage():
    part = _part()
    folds, details = _plan(part, wavelength_guard_pixels=12)
    assert len(folds) == 3
    assert details["reason"] == "ready"
    assert details["guard_pixels"] == 12
    for fold in folds:
        train, score = fold["train_masks"][0], fold["score_masks"][0]
        assert train.dtype == np.bool_ and score.dtype == np.bool_
        assert not np.any(train & score)
        assert np.count_nonzero(train) >= 32
        for block in fold["blocks"]:
            lo, hi = block["withhold_start"], block["withhold_stop"]
            core_lo, core_hi = block["core_start"], block["core_stop"]
            assert not np.any(train[lo:hi])
            assert np.all(score[core_lo:core_hi])
            assert not np.any(score[lo:core_lo])
            assert not np.any(score[core_hi:hi])
            assert core_lo - lo >= 12 and hi - core_hi >= 12
            assert core_lo <= block["center_index"] <= core_hi
            assert np.count_nonzero(train[:lo]) >= 4
            assert np.count_nonzero(train[hi:]) >= 4
    again, _ = _plan(part, wavelength_guard_pixels=12)
    for first, second in zip(folds, again, strict=True):
        np.testing.assert_array_equal(first["score_masks"][0], second["score_masks"][0])
        assert first["score_signature"] == second["score_signature"]


def test_masks_respect_invalid_excluded_opaque_pixels_and_wavelength_gaps():
    part = _part()
    part.spectrum.valid[355:365] = False
    part.fit_mask[75:85] = False
    part.spectrum.uncertainty[700] = 0.0
    part.transmission[215:225] = 0.02
    # A gap close to a feature must not become an interpolated training bridge.
    part.spectrum.wavelength[502:] += 0.1
    folds, details = _plan(part)
    assert len(folds) == 3
    for fold in folds:
        train, score = fold["train_masks"][0], fold["score_masks"][0]
        assert not np.any(train[355:365]) and not np.any(score[355:365])
        assert not np.any(train[75:85]) and not np.any(score[75:85])
        assert not train[700] and not score[700]
        assert not np.any(score[part.transmission < 0.1])
        for block in fold["blocks"]:
            assert not block["withhold_start"] < 502 < block["withhold_stop"]
    assert details["block_count"] >= 3


def test_insufficient_features_or_training_support_has_explanatory_fallback():
    folds, details = _plan(_part(size=70, centers=(18, 35, 52)), wavelength_guard_pixels=12)
    assert folds == []
    assert details["reason"] == "insufficient_usable_absorption_features"
    part = _part()
    part.spectrum.uncertainty = None
    folds, details = _plan(part)
    assert folds == [] and details["reason"] == "missing_original_uncertainty"


def test_plan_keeps_training_support_in_every_affected_segment():
    parts = [_part(size=150, centers=(35, 75, 115)), _part()]
    parts[0].fit_mask[:20] = False
    parts[0].fit_mask[130:] = False
    folds, _ = build_wavelength_validation_plan(
        parts, continuum_order=4, lsf_sigma_pixels=2.0, wavelength_guard_pixels=15
    )
    assert len(folds) == 3
    for fold in folds:
        for train, score in zip(fold["train_masks"], fold["score_masks"], strict=True):
            assert not np.any(train & score)
            assert np.count_nonzero(train) >= 40


def test_deep_features_require_moderate_support_on_both_wings():
    part = _part()
    pixels = np.arange(len(part.transmission))
    part.transmission = np.ones_like(part.transmission)
    for center in (80, 220, 360, 500, 640, 780, 920):
        part.transmission -= 0.97 * np.exp(-0.5 * ((pixels - center) / 4.0) ** 2)
    folds, details = _plan(part, wavelength_guard_pixels=12)
    assert len(folds) == 3
    for block in details["blocks"]:
        center = block["center_index"]
        assert part.transmission[center] < 0.1
        moderate = part.fit_mask & (part.transmission >= 0.1) & (part.transmission <= 0.95)
        assert np.count_nonzero(moderate[block["core_start"] : center]) >= 4
        assert np.count_nonzero(moderate[center + 1 : block["core_stop"]]) >= 4
    for fold in folds:
        assert not np.any(fold["score_masks"][0] & (part.transmission < 0.1))
    # Unavailable moderate wings cannot lend support to otherwise opaque cores.
    part.fit_mask[part.transmission >= 0.1] = False
    folds, details = _plan(part)
    assert folds == []
    assert details["reason"] == "insufficient_usable_absorption_features"


def test_guard_cannot_be_clipped_at_invalid_run_boundary():
    part = _part()
    # Feature at 360 remains valid, but its right guard would cross this gap.
    part.spectrum.valid[375:380] = False
    folds, details = _plan(part, wavelength_guard_pixels=12)
    assert len(folds) == 3
    assert 360 not in [block["center_index"] for block in details["blocks"]]


def test_score_uses_original_errors_and_fixed_pixels():
    part = _part()
    folds, _ = _plan(part)
    part.fit_mask = folds[0]["train_masks"][0].copy()
    scored = score_wavelength_validation_fold([part], folds[0])
    assert scored["valid"]
    assert scored["rms"] == pytest.approx(0.5)
    assert scored["median_absolute_residual"] == pytest.approx(0.5)
    assert scored["weighted_rss"] == pytest.approx(scored["point_count"] * 0.25)
    # Lowering a candidate's predicted transmission cannot remove score pixels.
    part.transmission[:] = 0.0
    assert score_wavelength_validation_fold([part], folds[0]) == scored
    part.spectrum.uncertainty[:] *= 2
    bad = score_wavelength_validation_fold([part], folds[0])
    assert not bad["valid"]
    assert bad["reason"] == "input_grid_flux_or_uncertainty_changed"


@pytest.mark.parametrize(
    "failure", ["nonfinite", "negative_continuum", "failed_fit", "changed_grid"]
)
def test_score_rejects_invalid_candidates_without_dropping_pixels(failure):
    part = _part()
    folds, _ = _plan(part)
    part.fit_mask = folds[0]["train_masks"][0].copy()
    index = np.flatnonzero(folds[0]["score_masks"][0])[0]
    if failure == "nonfinite":
        part.model_flux[index] = np.nan
    elif failure == "negative_continuum":
        part.continuum[index] = -1
    elif failure == "failed_fit":
        part.success = False
    else:
        part.spectrum.wavelength[index] += 1e-6
    result = score_wavelength_validation_fold([part], folds[0])
    assert not result["valid"]
    assert result["weighted_rss"] is None


def test_score_rejects_fit_that_consumed_withheld_pixels():
    part = _part()
    folds, _ = _plan(part)
    result = score_wavelength_validation_fold([part], folds[0])
    assert not result["valid"]
    assert result["reason"] == "training_mask_mismatch"


def _metrics(rms, median=None, *, valid=True, signature="same"):
    return {
        "valid": valid,
        "point_count": 20,
        "weighted_rss": rms * rms * 20,
        "rms": rms,
        "median_absolute_residual": rms if median is None else median,
        "score_signature": signature,
    }


def test_acceptance_requires_repeatable_predictive_improvement():
    reference = [_metrics(1), _metrics(1), _metrics(1)]
    candidate = [_metrics(0.8), _metrics(0.9), _metrics(0.85)]
    result = evaluate_wavelength_validation(reference, candidate)
    assert result["accepted"]
    assert result["relative_rms_improvement"] > 0.1
    assert result["reason"] == "supported_by_guarded_prediction"


def test_acceptance_rejects_small_gains_median_tradeoff_or_unpaired_failure():
    reference = [_metrics(1), _metrics(1), _metrics(1)]
    assert not evaluate_wavelength_validation(reference, [_metrics(0.98)] * 3)["accepted"]
    tradeoff = evaluate_wavelength_validation(reference, [_metrics(0.8, median=1.1)] * 3)
    assert tradeoff["reason"] == "median_absolute_residual_tradeoff"
    failed = [_metrics(0.8), _metrics(0.8), _metrics(0.8, valid=False)]
    assert not evaluate_wavelength_validation(reference, failed)["accepted"]
    changed = [_metrics(0.8), _metrics(0.8), _metrics(0.8, signature="different")]
    assert evaluate_wavelength_validation(reference, changed)["reason"] == "score_pixels_differ"
    # One enormous win must not outvote two deteriorating folds.
    reference = [_metrics(10), _metrics(1), _metrics(1)]
    candidate = [_metrics(1), _metrics(1.1), _metrics(1.1)]
    assert (
        evaluate_wavelength_validation(reference, candidate)["reason"]
        == "rms_improvement_not_repeated_across_folds"
    )
