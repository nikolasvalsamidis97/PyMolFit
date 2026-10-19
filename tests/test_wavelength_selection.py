from dataclasses import replace

import numpy as np
import pytest

import pymolfit.workflow as workflow
from pymolfit import LineList, correct_arrays
from pymolfit.fit import _shift_basis
from pymolfit.model import optical_depth_basis, transmission_from_basis


def _spectrum(coefficients, *, uncertainty=0.2, noise=0.0005):
    wavelength = np.linspace(1.0, 1.01, 1500)
    x = np.linspace(-1.0, 1.0, wavelength.size)
    lines = LineList(
        wavelength=np.linspace(wavelength[150], wavelength[-150], 7),
        strength=np.full(7, 3.0e-5),
        sigma=np.full(7, 2.0e-5),
        gamma=np.full(7, 3.0e-6),
        species=np.full(7, "H2O"),
    )
    species, basis = optical_depth_basis(wavelength, lines)
    shift = np.polynomial.polynomial.polyval(x, coefficients) * np.gradient(wavelength)
    flux = transmission_from_basis(
        species, _shift_basis(wavelength, basis, shift), species_scales={"H2O": 1.2}
    )
    flux += np.random.default_rng(22).normal(0.0, noise, wavelength.size)
    return wavelength, flux, np.full(wavelength.size, uncertainty), lines


def _correct(data, **changes):
    wavelength, flux, uncertainty, lines = data
    options = {
        "continuum_order": 0,
        "automatic_fit_regions": False,
        "high_resolution_grid": False,
        "lsf_sigma_pixels": 0.0,
        "fit_lsf_sigma": False,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "fit_lsf_lorentz_fwhm": False,
        "lsf_variable_width": False,
        "segment_size": 0.01,
        **changes,
    }
    return correct_arrays(wavelength, flux, uncertainty=uncertainty, line_list=lines, **options)


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([0.25, 1.1], "linear_pixel_trend"),
        ([0.4, 0.8, 1.5], "quadratic_pixel_trend"),
        ([0.25, 4.5], "linear_pixel_trend"),
    ],
)
def test_guarded_validation_recovers_distortion_rejected_by_fixed_noise_bic(coefficients, expected):
    # Deliberately large supplied errors reproduce the conservative BIC failure;
    # they must remain unchanged rather than being rescaled to force a selection.
    data = _spectrum(coefficients)
    result = _correct(data)
    alignment = result.provenance["wavelength_alignment"]

    assert result.success
    assert not any(comparison["accepted"] for comparison in alignment["comparisons"])
    assert alignment["selected_model"] == expected
    assert alignment["selection_reason"] == "guarded_block_validation_improvement"
    assert alignment["blocked_validation"]["plan"]["fold_count"] == 3
    np.testing.assert_allclose(result.wavelength_coefficients, coefficients, atol=0.02)
    assert np.sqrt(np.mean((result.model_flux - data[1]) ** 2)) < 0.0006
    np.testing.assert_array_equal(result.spectrum.wavelength, data[0])
    np.testing.assert_array_equal(result.spectrum.flux, data[1])
    np.testing.assert_array_equal(result.spectrum.uncertainty, data[2])
    if max(np.abs(coefficients)) > 3.0:
        assert alignment["bounds"] == [-12.0, 12.0]
        assert any(
            record["model"] == expected for record in alignment["validation_bound_expansions"]
        )


@pytest.mark.parametrize("bounds", [None, (-3.0, 3.0)])
def test_explicit_default_sized_bounds_are_never_automatically_expanded(bounds):
    result = _correct(_spectrum([7.0], uncertainty=0.002), wavelength_shift_bounds=bounds)
    alignment = result.provenance["wavelength_alignment"]

    assert result.success
    assert alignment["coefficient_unit"] == "pixel"
    if bounds is None:
        assert alignment["wavelength_shift_bound_expansion"]["triggered"]
        assert alignment["bounds"] == [-12.0, 12.0]
        np.testing.assert_allclose(result.wavelength_coefficients, [7.0], atol=0.02)
    else:
        assert not alignment["wavelength_shift_bound_expansion"]["triggered"]
        assert alignment["validation_bound_expansions"] == []
        assert alignment["bounds"] == [-3.0, 3.0]
        assert np.all(np.abs(result.wavelength_coefficients) <= 3.0)


def test_aligned_noisy_spectrum_keeps_its_input_wavelength_calibration():
    result = _correct(_spectrum([0.0], uncertainty=0.005, noise=0.005))
    assert result.success
    assert result.provenance["wavelength_alignment"]["selected_model"] == "none"
    np.testing.assert_array_equal(result.wavelength_coefficients, [0.0])


@pytest.mark.parametrize(
    "options",
    [
        {"fit_wavelength_shift": False},
        {"fit_wavelength_shift": True, "wavelength_shift_bounds": (-0.0001, 0.0001)},
        {
            "fit_wavelength_shift": "auto",
            "fit_wavelength_polynomial": True,
            "wavelength_polynomial_order": 1,
            "wavelength_shift_bounds": (-0.0001, 0.0001),
        },
    ],
)
def test_explicit_wavelength_controls_bypass_automatic_selection(monkeypatch, options):
    def unexpected_selection(*args, **kwargs):
        pytest.fail("An explicit wavelength control entered automatic model selection")

    monkeypatch.setattr(workflow, "_select_wavelength_model_from_pilots", unexpected_selection)
    result = _correct(_spectrum([0.8], uncertainty=0.002), **options)
    assert result.success
    assert "blocked_validation" not in result.provenance["wavelength_alignment"]


@pytest.mark.parametrize("failure", ["insufficient", "exception"])
def test_unavailable_validation_retains_the_original_penalized_selection(monkeypatch, failure):
    def unavailable_plan(*args, **kwargs):
        if failure == "exception":
            raise RuntimeError("validation unavailable in this regression test")
        return [], {"reason": "insufficient_usable_absorption_features"}

    monkeypatch.setattr(workflow, "build_wavelength_validation_plan", unavailable_plan)
    result = _correct(_spectrum([0.25, 1.1]))
    alignment = result.provenance["wavelength_alignment"]

    assert result.success
    assert alignment["selected_model"] == "none"
    assert alignment["selection_reason"] == "no_supported_wavelength_correction"
    assert alignment["blocked_validation"]["status"] == (
        "validation_error" if failure == "exception" else "insufficient_validation_blocks"
    )
    np.testing.assert_array_equal(result.wavelength_coefficients, [0.0])


@pytest.mark.parametrize("auto_segment", [False, True])
def test_native_grid_validation_preserves_single_continuum_geometry(auto_segment):
    # A processing size smaller than the order must not introduce independent
    # validation continua when the final native-grid fit uses one continuum.
    result = _correct(_spectrum([0.25, 1.1]), segment_size=0.003, auto_segment=auto_segment)
    alignment = result.provenance["wavelength_alignment"]
    assert result.success
    assert alignment["blocked_validation"]["processing_segment_pixel_counts"] == [1500]
    assert "segmentation" not in result.provenance
    assert alignment["selected_model"] == "linear_pixel_trend"
    np.testing.assert_allclose(result.wavelength_coefficients, [0.25, 1.1], atol=0.02)


def test_automatic_continuum_refinement_uses_only_each_folds_training_mask(monkeypatch):
    data = _spectrum([0.25, 1.1])
    original_validation = workflow._validate_wavelength_candidates
    original_fit = workflow.fit_telluric_segments
    original_choose = workflow.choose_continuum_order
    in_validation = False
    previous_fit = None
    fold_choices = []
    refit_checks = []
    expected_refit = None

    def validation(*args, **kwargs):
        nonlocal in_validation
        assert kwargs["automatic_continuum_order"] is True
        in_validation = True
        try:
            return original_validation(*args, **kwargs)
        finally:
            in_validation = False

    def fit(*args, **kwargs):
        nonlocal previous_fit, expected_refit
        if in_validation and expected_refit is not None:
            before, masks, order = expected_refit
            assert kwargs["config"].continuum_order == order
            actual_masks = kwargs.get("fit_masks")
            assert (actual_masks is None) == (masks is None)
            if masks is not None:
                for actual, expected in zip(actual_masks, masks, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            assert kwargs["config"].initial_species_scales == before.species_scales
            assert kwargs["config"].lsf_sigma_pixels == before.lsf_sigma_pixels
            refit_checks.append(masks is not None)
            expected_refit = None
        result = original_fit(*args, **kwargs)
        if in_validation:
            previous_fit = (result, kwargs.get("fit_masks"))
        return result

    def choose(samples, **kwargs):
        nonlocal expected_refit
        if in_validation:
            assert previous_fit is not None
            fitted, masks = previous_fit
            for sample, part in zip(samples, fitted.segment_results, strict=True):
                np.testing.assert_array_equal(sample[0], part.spectrum.wavelength)
                np.testing.assert_array_equal(sample[1], part.spectrum.flux)
                np.testing.assert_array_equal(sample[2], part.spectrum.uncertainty)
                np.testing.assert_array_equal(sample[3], part.transmission)
                np.testing.assert_array_equal(sample[4], part.fit_mask)
                np.testing.assert_array_equal(sample[2], data[2])
            if masks is not None:
                for sample, mask in zip(samples, masks, strict=True):
                    np.testing.assert_array_equal(sample[4], mask)
                assert any(np.any(~mask) for mask in masks)
                fold_choices.append(samples)
        order, details = original_choose(samples, **kwargs)
        if in_validation and order != kwargs["initial_order"]:
            expected_refit = (*previous_fit, order)
        return order, details

    monkeypatch.setattr(workflow, "_validate_wavelength_candidates", validation)
    monkeypatch.setattr(workflow, "fit_telluric_segments", fit)
    monkeypatch.setattr(workflow, "choose_continuum_order", choose)
    result = _correct(data, continuum_order="auto")
    assert result.success
    assert len(fold_choices) >= 6
    assert any(refit_checks)
    alignment = result.provenance["wavelength_alignment"]
    assert alignment["selected_model"] == "linear_pixel_trend"
    np.testing.assert_allclose(result.wavelength_coefficients, [0.25, 1.1], atol=0.02)
    scores = list(alignment["blocked_validation"]["reference_fold_scores"])
    scores += [
        score
        for comparison in alignment["blocked_validation"]["comparisons"]
        for score in comparison.get("fold_scores", [])
    ]
    assert all(score["continuum_refinement"]["requested"] == "auto" for score in scores)
    assert any(score["selected_continuum_order"] == 1 for score in scores)
    assert all(np.isfinite(score["training_cost"]) for score in scores)
    assert all(score["fit_success"] and score["nfev"] > 0 for score in scores)


def test_explicit_continuum_degree_bypasses_validation_refinement(monkeypatch):
    def unexpected_refinement(*args, **kwargs):
        pytest.fail("Explicit continuum order entered automatic degree selection")

    monkeypatch.setattr(workflow, "choose_continuum_order", unexpected_refinement)
    result = _correct(_spectrum([0.25, 1.1]), continuum_order=2)
    assert result.success
    validation = result.provenance["wavelength_alignment"]["blocked_validation"]
    assert validation["status"] == "completed"
    assert all(
        score["selected_continuum_order"] == 2 for score in validation["reference_fold_scores"]
    )


def test_automatic_continuum_refinement_reaches_each_physical_group():
    first = _spectrum([0.25, 1.1])
    second = _spectrum([0.4, 0.8, 1.5])
    second_lines = replace(second[3], wavelength=second[3].wavelength + 0.02)
    lines = LineList(
        **{
            field: np.concatenate([getattr(first[3], field), getattr(second_lines, field)])
            for field in ("wavelength", "strength", "sigma", "gamma", "species")
        }
    )
    data = (
        np.concatenate([first[0], second[0] + 0.02]),
        np.concatenate([first[1], second[1]]),
        np.concatenate([first[2], second[2]]),
        lines,
    )
    result = _correct(data, continuum_order="auto", group_id=np.repeat([10, 20], 1500))
    groups = result.provenance["wavelength_alignment"]["groups"]
    assert result.success
    assert [group["polynomial_order"] for group in groups] == [1, 2]
    for group in groups:
        assert group["validation_refines_continuum_order"]
        refinement = group["blocked_validation"]["reference_continuum_refinement"]
        assert refinement["requested"] == "auto"
        assert refinement["selected_order"] == 1


@pytest.mark.parametrize("failed", [False, True])
def test_unusable_continuum_refinement_retains_the_initial_fold_fit(monkeypatch, failed):
    original_validation = workflow._validate_wavelength_candidates
    original_fit = workflow.fit_telluric_segments
    original_choose = workflow.choose_continuum_order
    in_validation = False

    def validation(*args, **kwargs):
        nonlocal in_validation
        in_validation = True
        try:
            return original_validation(*args, **kwargs)
        finally:
            in_validation = False

    def choose(samples, **kwargs):
        if in_validation:
            return 1, {"requested": "auto", "selected_order": 1}
        return original_choose(samples, **kwargs)

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        if in_validation and kwargs["config"].continuum_order == 1:
            if failed:
                return replace(result, success=False)
            return replace(
                result,
                segment_results=tuple(
                    replace(part, continuum=-np.abs(part.continuum))
                    for part in result.segment_results
                ),
            )
        return result

    monkeypatch.setattr(workflow, "_validate_wavelength_candidates", validation)
    monkeypatch.setattr(workflow, "fit_telluric_segments", fit)
    monkeypatch.setattr(workflow, "choose_continuum_order", choose)
    result = _correct(_spectrum([0.25, 1.1]), continuum_order="auto")
    assert result.success
    validation_details = result.provenance["wavelength_alignment"]["blocked_validation"]
    assert validation_details["status"] == "completed"
    for score in validation_details["reference_fold_scores"]:
        assert score["valid"]
        assert score["selected_continuum_order"] == 2
        assert "fallback" in score["continuum_refinement"]
