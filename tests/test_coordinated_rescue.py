from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pymolfit.workflow as workflow
from pymolfit import _coordinated_rescue as rescue
from pymolfit.spectrum import Spectrum
from tests.test_paired_continuum_rescue import exercise


@pytest.fixture
def scenario():
    n = 400
    spectrum = Spectrum(
        np.linspace(5.14, 5.15, n),
        np.full(n, 10.0),
        uncertainty=np.full(n, 2.0),
        wavelength_unit="micron",
    )
    transmission = np.full(n, 0.7)
    transmission[::3] = 0.99
    config = workflow.FitConfig(
        continuum_order=1,
        fit_wavelength_shift=True,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=(-12.0, 12.0),
        lsf_sigma_pixels=1.2,
        fit_lsf_sigma=True,
        initial_species_scales={"H2O": 5.0},
    )
    start = replace(config, initial_species_scales=None, lsf_sigma_pixels=1.0)

    def result(residual=1.0):
        return SimpleNamespace(
            spectrum=spectrum,
            fit_mask=spectrum.valid.copy(),
            continuum=np.full(n, 12.0),
            model_flux=np.full(n, 10.0 - 2.0 * residual),
            transmission=transmission.copy(),
            wavelength_coefficients=np.array([-0.2, 0.01, 0.1]),
            success=True,
            parameter_bound_status={},
            nfev=8,
        )

    before = result()
    deferred = {
        "continuum_order": 1,
        "wavelength_model": "quadratic_pixel_trend",
        "wavelength_shift_bounds": [-12.0, 12.0],
    }
    return before, config, start, result, deferred


def test_accepted_pair_keeps_exact_full_fit_result_and_config(scenario):
    before, config, start, make, deferred = scenario
    candidate = make(0.8)
    seen = []

    def fit(c):
        seen.append(c)
        return candidate

    after, chosen, details = rescue.apply_rescued_pair(
        before, config, deferred, run_fit=fit, starting_config=start
    )
    assert details["accepted"]
    assert after is candidate and chosen is seen[0]
    assert chosen.continuum_order == 1 and chosen.fit_wavelength_polynomial
    assert chosen.wavelength_polynomial_order == 2
    assert chosen.initial_species_scales is None
    assert chosen.initial_wavelength_shift == 0.0
    assert chosen.lsf_sigma_pixels == 1.0
    assert chosen.fit_ranges == config.fit_ranges
    assert chosen.exclude_ranges == config.exclude_ranges


@pytest.mark.parametrize(
    "failure",
    [
        "negative_continuum",
        "negative_transmission",
        "failed_fit",
        "changed_errors",
        "changed_mask",
        "full_regression",
        "transparent_regression",
        "bounds",
    ],
)
def test_rejected_pair_returns_bit_identical_native_objects(scenario, failure):
    before, config, start, make, deferred = scenario
    originals = {
        name: getattr(before, name).copy()
        for name in ("model_flux", "continuum", "transmission", "fit_mask")
    }
    candidate = make(0.8)
    if failure == "negative_continuum":
        candidate.continuum[100] = -1.0
    elif failure == "negative_transmission":
        candidate.transmission[100] = -1.0
    elif failure == "failed_fit":
        candidate.success = False
    elif failure == "changed_errors":
        candidate.spectrum = replace(before.spectrum, uncertainty=np.ones(400))
    elif failure == "changed_mask":
        candidate.fit_mask[100] = False
    elif failure == "full_regression":
        candidate.model_flux[:] = 7.0
    elif failure == "transparent_regression":
        candidate.model_flux[::3] = 7.0
    elif failure == "bounds":
        candidate.wavelength_coefficients[0] = 12.0
    after, chosen, details = rescue.apply_rescued_pair(
        before, config, deferred, run_fit=lambda c: candidate, starting_config=start
    )
    assert not details["accepted"]
    assert after is before and chosen is config
    for name, value in originals.items():
        np.testing.assert_array_equal(getattr(after, name), value)


def test_bounds_never_changed_even_if_pair_requests_it(scenario):
    before, config, start, _, deferred = scenario
    deferred["wavelength_shift_bounds"] = [-24.0, 24.0]
    after, chosen, details = rescue.apply_rescued_pair(
        before,
        config,
        deferred,
        run_fit=lambda c: pytest.fail("must not fit"),
        starting_config=start,
    )
    assert after is before and chosen is config
    assert details["reason"] == "deferred_bounds_differ_from_baseline"


def test_missing_transparent_band_stays_explicitly_unevaluated(scenario):
    before, config, start, make, deferred = scenario
    before.transmission[:] = 0.7
    _after, _chosen, details = rescue.apply_rescued_pair(
        before, config, deferred, run_fit=lambda c: make(0.8), starting_config=start
    )
    assert details["accepted"]
    assert details["full_comparison"]["scores"]["transparent"] == {"pixels": 0, "evaluated": False}


def native_selection_fixture(invalid_degrees):
    """Expected native decisions captured from the pre-rescue selector.

    These are independent baseline expectations, including the original failed
    folds. They must remain unchanged when a rescued pair is deferred for later.
    """

    def scores(coefficient_count):
        quadratic = coefficient_count == 3
        result = []
        for index in range(3):
            valid = not (invalid_degrees and index > 0)
            result.append(
                {
                    "valid": valid,
                    "reason": "scored" if valid else "nonpositive_continuum",
                    "weighted_rss": (3.1999999999999984 if quadratic else 5.0) if valid else None,
                    "point_count": 5 if valid else 0,
                    "rms": (0.7999999999999998 if quadratic else 1.0) if valid else None,
                    "median_absolute_residual": (0.7999999999999998 if quadratic else 1.0)
                    if valid
                    else None,
                    "score_signature": f"frozen_fold_{index}",
                    "wavelength_bound_status": {},
                    "selected_continuum_order": 2,
                    "continuum_refinement": {"selected_order": 2},
                    "training_cost": 0.6400000000000001 if quadratic else 1.0,
                    "nfev": 10,
                    "lsf_sigma_pixels": 1.0,
                    "coefficients_pixels": [0.0] * coefficient_count,
                    "fit_success": True,
                }
            )
        return result

    comparisons = []
    for coefficient_count, family in ((2, "linear_pixel_trend"), (3, "quadratic_pixel_trend")):
        quadratic = coefficient_count == 3
        comparison = {
            "accepted": quadratic and not bool(invalid_degrees),
            "relative_rms_improvement": None
            if invalid_degrees
            else (0.2000000000000003 if quadratic else 0.0),
            "fold_improvements": []
            if invalid_degrees
            else [0.20000000000000018 if quadratic else 0.0] * 3,
            "reason": "three_valid_paired_folds_required"
            if invalid_degrees
            else (
                "supported_by_guarded_prediction"
                if quadratic
                else "aggregate_rms_improvement_below_five_percent"
            ),
            "from_model": "constant_pixel_shift",
            "to_model": family,
            "bounds_pixels": [-12.0, 12.0],
            "fold_scores": scores(coefficient_count),
        }
        if not invalid_degrees:
            comparison.update(
                {
                    "reference_mean_fold_median_absolute_residual": 1.0,
                    "candidate_mean_fold_median_absolute_residual": 0.7999999999999998
                    if quadratic
                    else 1.0,
                }
            )
        comparisons.append(comparison)
    selected = "constant_pixel_shift" if invalid_degrees else "quadratic_pixel_trend"
    config = workflow.FitConfig(
        continuum_order=2,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=(-12.0, 12.0),
        fit_wavelength_shift=bool(invalid_degrees),
        fit_wavelength_polynomial=not bool(invalid_degrees),
        wavelength_polynomial_order=1 if invalid_degrees else 2,
    )
    return (
        selected,
        config,
        {
            "reference_model": "constant_pixel_shift",
            "reference_fold_scores": scores(1),
            "comparisons": comparisons,
            "status": "completed",
            "selected_model": selected,
        },
    )


@pytest.mark.parametrize("invalid_degrees", [(), (2,)])
def test_original_wavelength_decision_and_diagnostics_are_preserved(monkeypatch, invalid_degrees):
    selected, details, _, _ = exercise(monkeypatch, invalid_degrees=invalid_degrees)
    name, config, expected_details = native_selection_fixture(invalid_degrees)
    assert selected[0] == name
    assert selected[1] == config
    for key, value in expected_details.items():
        assert details[key] == value
    if invalid_degrees:
        assert details["deferred_continuum_wavelength_pair"]["continuum_order"] == 1
    else:
        assert "deferred_continuum_wavelength_pair" not in details


def test_workflow_applies_pair_after_all_continuum_refinement_before_quality_mask():
    source = Path(workflow.__file__).read_text()
    late = source.index(
        "selected_result, fit_config, coordinated_details = coordinated_model_selection("
    )
    assert (
        source.index("selected_result, fit_config, validation = guarded_continuum_refinement(")
        < late
    )
    assert (
        source.index("selected_result, quality_mask_resolution = mask_uncertain_correction(") > late
    )
    tail = source[late : source.index("    quality_mask_resolution = {", late)]
    assert "choose_continuum_order(" not in tail


def test_zero_reference_residual_fails_closed_with_serializable_diagnostics(scenario):
    import json

    before, config, start, make, deferred = scenario
    before.model_flux[:] = before.spectrum.flux
    result, returned, details = rescue.apply_rescued_pair(
        before, config, deferred, run_fit=lambda c: make(0.8), starting_config=start
    )
    assert result is before and returned is config
    assert not details["accepted"]
    json.dumps(details, allow_nan=False)
