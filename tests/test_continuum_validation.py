"""Focused guarded-plan, acceptance, independent-start and leakage checks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import pymolfit.workflow as workflow
from pymolfit import FitConfig, LineList, Spectrum, correct_arrays, fit_telluric_segments
from pymolfit._automatic import (
    _separated_continuum_centers,
    continuum_guarded_plan,
    guarded_continuum_refinement,
    promising_continuum_degree,
)


def _setup():
    wave = np.linspace(2.3, 2.33, 900)
    spectrum = Spectrum(wave, np.full(900, 10.0), uncertainty=np.ones(900))
    config = FitConfig(
        continuum_order=1,
        lsf_sigma_pixels=2.0,
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.0, 3.0),
        initial_species_scales={"H2O": 9.0},
    )
    reference = SimpleNamespace(
        spectrum=spectrum,
        model_flux=np.full(900, 9.0),
        continuum=np.full(900, 10.0),
        transmission=np.full(900, 0.9),
        fit_mask=np.ones(900, bool),
        success=True,
        nfev=1,
        provenance={
            "segmentation": {
                "boundaries_micron": [[float(wave[i]), float(wave[i + 299])] for i in (0, 300, 600)]
            }
        },
    )
    resolution = {
        "candidates": {
            1: {"validation_cost": 100.0, "score": 110.0, "segments": 3},
            2: {"validation_cost": 80.0, "score": 120.0, "segments": 3},
        }
    }
    start = replace(
        config, initial_species_scales=None, lsf_sigma_pixels=1.0, initial_wavelength_shift=0.0
    )
    return reference, config, resolution, start


def _fake_fitter(reference, *, bad_fold=None, leak=False, full_regression=False):
    calls = []

    def run(config):
        calls.append(config)
        heldout = bool(config.exclude_ranges)
        fit_mask = reference.fit_mask.copy()
        if not leak:
            for lo, hi in config.exclude_ranges or ():
                fit_mask &= ~(
                    (reference.spectrum.wavelength >= lo) & (reference.spectrum.wavelength <= hi)
                )
        residual = 0.8 if config.continuum_order == 2 else 1.0
        if config.continuum_order == 2 and (
            (bad_fold is not None and len(calls) == 2 * (bad_fold + 1))
            or (full_regression and not heldout)
        ):
            residual = 1.2
        return SimpleNamespace(
            **{
                **vars(reference),
                "fit_mask": fit_mask,
                "model_flux": reference.spectrum.flux - residual,
            }
        )

    return run, calls


def test_screen_requires_promising_higher_degree_and_finite_family():
    _, _, resolution, _ = _setup()
    assert promising_continuum_degree(resolution, 1) == 2
    assert promising_continuum_degree(resolution, 2) is None
    resolution["candidates"][2]["score"] = None
    assert promising_continuum_degree(resolution, 1) is None


def test_plan_has_three_disjoint_guarded_cores_and_enlarged_motion_support():
    reference, config, _, _ = _setup()
    config = replace(
        config,
        fit_wavelength_polynomial=True,
        wavelength_polynomial_order=2,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=(-1.0, 1.0),
    )
    folds, plan = continuum_guarded_plan(reference, config, 2)
    assert len(folds) == 3
    assert plan["guard_pixels_per_side"] >= plan["maximum_kernel_half_support_pixels"] + 6
    assert all(f["score_mask"].sum() == 48 for f in folds)
    assert all(not np.any(f["score_mask"] & ~f["guard_mask"]) for f in folds)
    assert not np.any(folds[0]["score_mask"] & folds[1]["score_mask"])


def test_accepted_candidate_refits_all_folds_with_independent_starts():
    reference, config, resolution, start = _setup()
    run, calls = _fake_fitter(reference)
    result, selected, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert details["accepted"] and selected.continuum_order == 2 and result is not reference
    assert len(calls) == 7
    assert all(c.initial_species_scales is None and c.lsf_sigma_pixels == 1.0 for c in calls[:6])
    assert all(c.exclude_ranges for c in calls[:6])
    assert all(calls[i].exclude_ranges == calls[i + 1].exclude_ranges for i in (0, 2, 4))
    assert details["pooled_rms_improvement_percent"] == pytest.approx(20.0)


def test_separated_centers_adapt_to_feasible_geometry_without_clustering():
    assert _separated_continuum_centers(range(100, 260), 300) == (100, 150, 239)
    assert _separated_continuum_centers(range(100, 130), 300) is None


def test_invalid_guard_pixels_are_excluded_and_never_scored():
    reference, config, _, _ = _setup()
    valid = reference.spectrum.valid.copy()
    valid[[45, 60, 345, 360, 645, 660]] = False
    reference.spectrum = replace(reference.spectrum, mask=valid)
    reference.fit_mask = valid.copy()
    folds, details = continuum_guarded_plan(reference, config, 2)
    assert details["reason"] == "ready" and len(folds) == 3
    assert not any(np.any(f["score_mask"] & ~valid) for f in folds)
    assert any(np.any(f["guard_mask"] & ~valid) for f in folds)
    assert all(block["original_valid_core_pixels"] >= 8 for f in folds for block in f["blocks"])


def test_center_selection_does_not_use_flux_or_fitted_residuals():
    reference, config, _, _ = _setup()
    first, _ = continuum_guarded_plan(reference, config, 2)
    reference.spectrum = replace(reference.spectrum, flux=reference.spectrum.flux * 1e12)
    reference.model_flux = np.full(900, -1e12)
    second, _ = continuum_guarded_plan(reference, config, 2)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a["score_mask"], b["score_mask"])
        np.testing.assert_array_equal(a["guard_mask"], b["guard_mask"])


@pytest.mark.parametrize("failure", ["fold", "leak", "full"])
def test_regression_or_guard_leak_retains_exact_reference(failure):
    reference, config, resolution, start = _setup()
    run, _ = _fake_fitter(
        reference,
        bad_fold=0 if failure == "fold" else None,
        leak=failure == "leak",
        full_regression=failure == "full",
    )
    result, selected, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert result is reference and selected is config and not details["accepted"]


def test_opaque_or_invalid_geometry_fails_before_any_refit():
    reference, config, resolution, start = _setup()
    reference.transmission[:] = 0.01
    run, calls = _fake_fitter(reference)
    result, _, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert result is reference and not calls
    assert details["reason"] == "insufficient_validation_support"


def test_full_pipeline_lowlevel_exclusions_ignore_guarded_flux_perturbation():
    # The production run_fit path calls this same segmented fitter with
    # config.exclude_ranges. Coarse LSF/wavelength starts must also ignore them.
    wave = np.linspace(2.3, 2.33, 300)
    line_list = LineList(
        wavelength=np.array([2.308, 2.323]),
        strength=np.array([0.001, 0.0015]),
        sigma=np.array([0.0001, 0.0001]),
        gamma=np.zeros(2),
        species=np.array(["H2O", "H2O"]),
    )
    flux = 10 + 0.1 * np.sin(np.arange(300) / 40)
    exclude = ((float(wave[120]), float(wave[159])),)
    config = FitConfig(
        continuum_order=2,
        solve_continuum_linear=True,
        loss="soft_l1",
        exclude_ranges=exclude,
        initialize_lsf_sigma_grid=True,
        fit_lsf_sigma=True,
        lsf_sigma_pixels=1.0,
        lsf_sigma_bounds=(0.0, 3.0),
        fit_wavelength_shift=True,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=(-2.0, 2.0),
        estimate_uncertainties=False,
    )
    original = Spectrum(wave, flux, uncertainty=np.full(300, 0.1))
    changed_flux = flux.copy()
    changed_flux[120:160] = np.resize([1e12, -1e12], 40)
    changed = replace(original, flux=changed_flux)
    a = fit_telluric_segments((original,), line_list=line_list, config=config)
    b = fit_telluric_segments((changed,), line_list=line_list, config=config)
    np.testing.assert_array_equal(a.segment_results[0].fit_mask, b.segment_results[0].fit_mask)
    assert not np.any(a.segment_results[0].fit_mask[120:160])
    np.testing.assert_array_equal(a.segment_results[0].model_flux, b.segment_results[0].model_flux)
    assert a.species_scales == b.species_scales


@pytest.mark.parametrize(
    "failure", ["fit", "flux", "errors", "valid", "model", "continuum", "transmission"]
)
def test_invalid_prediction_or_changed_original_data_retains_reference(failure):
    reference, config, resolution, start = _setup()
    ordinary_fit, calls = _fake_fitter(reference)

    def run(trial_config):
        result = ordinary_fit(trial_config)
        if trial_config.continuum_order != 2:
            return result
        if failure == "fit":
            result.success = False
        elif failure in ("flux", "errors", "valid"):
            field = {"flux": "flux", "errors": "uncertainty", "valid": "mask"}[failure]
            values = getattr(result.spectrum, "valid" if field == "mask" else field).copy()
            values[0] = False if field == "mask" else values[0] * 2
            result.spectrum = replace(result.spectrum, **{field: values})
        else:
            field = "model_flux" if failure == "model" else failure
            values = getattr(result, field).copy()
            # This pixel is outside all scored cores. Quality must cover it too.
            values[0] = -1 if failure == "continuum" else np.nan
            setattr(result, field, values)
        return result

    result, selected, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert result is reference and selected is config
    assert not details["accepted"] and details["reason"] == "validation_error"
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["small_gain", "median_regression"])
def test_pooled_acceptance_requires_both_rms_threshold_and_median(failure):
    reference, config, resolution, start = _setup()
    ordinary_fit, calls = _fake_fitter(reference)

    def run(trial_config):
        result = ordinary_fit(trial_config)
        if trial_config.continuum_order == 2:
            residual = (
                np.full(900, 0.96)
                if failure == "small_gain"
                else np.where(np.arange(900) % 5 < 3, 1.01, 0.0)
            )
            result.model_flux = result.spectrum.flux - residual
        return result

    result, selected, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert result is reference and selected is config
    assert details["reason"] == "guarded_prediction_threshold_not_met"
    assert len(calls) == 6  # No full-data candidate refit after rejection.
    if failure == "median_regression":
        assert details["pooled_rms_improvement_percent"] > 5
        assert details["pooled_median_absolute_after"] > details["pooled_median_absolute_before"]


@pytest.mark.parametrize("error", [ValueError, RuntimeError, np.linalg.LinAlgError])
def test_refit_exception_retains_exact_reference_and_records_reason(error):
    reference, config, resolution, start = _setup()

    def run(_config):
        raise error("unusable validation fit")

    result, selected, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert result is reference and selected is config
    assert details["reason"] == "validation_error"
    assert details["error"] == "unusable validation fit"


@pytest.mark.parametrize(
    "overrides",
    [
        {"lsf_variable_width": True},
        {"lsf_lorentz_fwhm_pixels": 1.0},
        {"lsf_box_width_pixels": 1.0},
        {"fit_segment_wavelength_shifts": True},
        {"fit_wavelength_shift": True, "wavelength_shift_unit": "micron"},
    ],
)
def test_unsupported_geometry_never_launches_refits(overrides):
    reference, config, resolution, start = _setup()
    config = replace(config, **overrides)
    run, calls = _fake_fitter(reference)
    result, selected, details = guarded_continuum_refinement(
        reference, config, resolution, run_fit=run, starting_config=start
    )
    assert result is reference and selected is config and not calls
    assert details["reason"] == "insufficient_validation_support"


def _quiet_options():
    return {
        "line_list": LineList.empty_hitran(),
        "physical": False,
        "lsf_sigma_pixels": 0.0,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "lsf_variable_width": False,
        "fit_wavelength_shift": False,
        "high_resolution_grid": False,
        "auto_segment": False,
        "automatic_fit_regions": False,
    }


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_explicit_continuum_degree_bypasses_guarded_refinement(monkeypatch, degree):
    def unexpected(*args, **kwargs):
        pytest.fail("Explicit continuum degree must not launch automatic validation")

    monkeypatch.setattr(workflow, "guarded_continuum_refinement", unexpected)
    wave = np.linspace(1.0, 1.01, 300)
    result = correct_arrays(
        wave,
        np.ones(300),
        uncertainty=np.full(300, 0.02),
        continuum_order=degree,
        f_scale=1.0,
        **_quiet_options(),
    )
    assert result.success
    assert result.provenance["automatic_continuum"]["selected_order"] == degree
    assert "guarded_degree_validation" not in result.provenance["automatic_continuum"]


def test_automatic_workflow_records_guarded_decision_and_preserves_errors(monkeypatch):
    calls = []
    real_refinement = workflow.guarded_continuum_refinement

    def record(*args, **kwargs):
        calls.append(kwargs["starting_config"])
        return real_refinement(*args, **kwargs)

    monkeypatch.setattr(workflow, "guarded_continuum_refinement", record)
    wave = np.linspace(1.0, 1.01, 300)
    flux = 1 + 0.1 * np.linspace(-1, 1, 300)
    errors = np.linspace(0.01, 0.02, 300)
    result = correct_arrays(wave, flux, uncertainty=errors, **_quiet_options())
    assert result.success and len(calls) == 1
    details = result.provenance["automatic_continuum"]["guarded_degree_validation"]
    assert details["reason"] == "no_promising_rejected_degree"
    assert not details["accepted"]
    assert calls[0].initial_species_scales is None
    np.testing.assert_array_equal(result.spectrum.uncertainty, errors)


def test_fixed_automatic_width_does_not_reuse_full_data_estimate_in_folds(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A fixed automatic width estimate cannot seed validation folds")

    monkeypatch.setattr(workflow, "guarded_continuum_refinement", unexpected)
    options = {**_quiet_options(), "lsf_sigma_pixels": "auto", "fit_lsf_sigma": False}
    wave = np.linspace(1.0, 1.01, 300)
    result = correct_arrays(wave, np.ones(300), uncertainty=np.full(300, 0.02), **options)
    assert result.success
    assert "guarded_degree_validation" not in result.provenance["automatic_continuum"]
