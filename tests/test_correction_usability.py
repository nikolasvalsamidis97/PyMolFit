"""Numerical usability must not rewrite optimizer status or corrected samples."""

import json
from dataclasses import replace

import numpy as np
import pytest

from pymolfit import Spectrum
from pymolfit.diagnostics import (
    correction_usability_diagnostics,
    fit_quality_diagnostics,
    format_fit_summary,
)
from pymolfit.fit import TelluricFitResult
from pymolfit.spectrum import correct_spectrum


def _make_result():
    wave = np.linspace(1.5, 1.51, 100)
    transmission = np.full(100, 0.8)
    spectrum = Spectrum(wave, transmission, uncertainty=np.full(100, 0.01))
    return TelluricFitResult(
        spectrum=spectrum,
        corrected=correct_spectrum(spectrum, transmission),
        transmission=transmission,
        continuum=np.ones(100),
        model_flux=transmission.copy(),
        species_scales={"H2O": 1.0},
        wavelength_shift=0.0,
        wavelength_coefficients=np.array([0.0]),
        lsf_sigma_pixels=1.0,
        lsf_box_width_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_wavelength_exponent=0.0,
        continuum_coefficients=np.array([1.0]),
        metrics={},
        success=True,
        message="converged",
        cost=0.0,
        nfev=1,
        fit_mask=np.ones(100, dtype=bool),
        provenance={"fit_config": {"fields": {"min_transmission": 0.01}}},
    )


def _snapshot(result):
    arrays = {}
    for prefix, obj, names in [
        (
            "input",
            result.spectrum,
            ("wavelength", "flux", "uncertainty", "mask", "group_id"),
        ),
        (
            "corrected",
            result.corrected,
            ("wavelength", "flux", "uncertainty", "mask", "group_id"),
        ),
        ("model", result, ("model_flux", "continuum", "transmission", "fit_mask")),
    ]:
        for key in names:
            value = getattr(obj, key)
            arrays[prefix + "." + key] = None if value is None else np.array(value, copy=True)
    return arrays


def test_healthy_status_report_and_nonmutation():
    result = _make_result()
    before = _snapshot(result)
    quality = fit_quality_diagnostics(result)
    status = quality["correction_usability"]
    assert status["status"] == "usable" and status["numerically_usable"]
    assert status["retained_fraction"] == 1 and status["reasons"] == []
    assert "success: yes" in format_fit_summary(result)
    assert (
        "numerical correction status: usable (100/100 valid samples retained)"
        in format_fit_summary(result)
    )
    assert result.success is True
    after = _snapshot(result)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    json.dumps(status, allow_nan=False)


def test_negative_continuum_outside_fit_mask_still_invalid_model():
    result = _make_result()
    continuum = np.ones(100)
    continuum[50:60] = -1
    mask = np.ones(100, dtype=bool)
    mask[50:60] = False
    result = replace(result, continuum=continuum, fit_mask=mask)
    status = correction_usability_diagnostics(result)
    assert result.success is True and status["status"] == "invalid_model"
    assert status["nonpositive_continuum_pixels"] == 10
    assert status["nonpositive_continuum_fit_pixels"] == 0
    assert status["nonpositive_continuum_retained_pixels"] == 10


def test_input_invalid_bad_continuum_not_used_as_false_alarm():
    result = _make_result()
    mask = np.ones(100, dtype=bool)
    mask[:5] = False
    continuum = np.ones(100)
    continuum[:5] = np.nan
    result = replace(result, spectrum=replace(result.spectrum, mask=mask), continuum=continuum)
    status = correction_usability_diagnostics(result)
    assert status["status"] == "usable" and status["input_valid_pixels"] == 95
    assert status["nonfinite_continuum_pixels"] == 0


def test_optimizer_success_empty_opaque_output_is_explicit():
    result = _make_result()
    transmission = np.zeros(100)
    result = replace(
        result,
        transmission=transmission,
        model_flux=transmission,
        corrected=correct_spectrum(result.spectrum, transmission),
    )
    status = correction_usability_diagnostics(result)
    assert result.success is True and status["optimizer_success"] is True
    assert status["status"] == "empty_correction" and not status["numerically_usable"]
    assert status["pixels_below_transmission_floor"] == 100
    assert "all_valid_samples_below_transmission_floor" in status["reasons"]
    assert status["negative_transmission_pixels"] == 0


def test_empty_output_with_negative_continuum_keeps_both_reasons():
    result = _make_result()
    mask = np.zeros(100, dtype=bool)
    result = replace(
        result, continuum=-np.ones(100), corrected=replace(result.corrected, mask=mask)
    )
    status = correction_usability_diagnostics(result)
    assert status["status"] == "empty_correction"
    assert "nonpositive_continuum" in status["reasons"]
    assert "no_retained_corrected_samples" in status["reasons"]
    assert "all_valid_samples_below_transmission_floor" not in status["reasons"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("continuum", 0.0, "nonpositive_continuum"),
        ("continuum", np.inf, "nonfinite_continuum"),
        ("model_flux", np.nan, "nonfinite_model"),
        ("transmission", np.nan, "nonfinite_transmission"),
        ("transmission", -0.1, "negative_transmission"),
    ],
)
def test_invalid_model_arrays(field, value, reason):
    result = _make_result()
    values = np.array(getattr(result, field), copy=True)
    values[20] = value
    result = replace(result, **{field: values})
    status = correction_usability_diagnostics(result)
    assert status["status"] == "invalid_model" and reason in status["reasons"]
    assert result.success is True


def test_partial_transmission_mask_is_usable_without_arbitrary_fraction_threshold():
    result = _make_result()
    transmission = result.transmission.copy()
    transmission[:70] = 0.001
    result = replace(
        result,
        transmission=transmission,
        model_flux=transmission,
        corrected=correct_spectrum(result.spectrum, transmission),
    )
    status = correction_usability_diagnostics(result)
    assert status["status"] == "usable" and status["retained_corrected_pixels"] == 30
    assert status["pixels_below_transmission_floor"] == 70
    assert status["retained_fraction"] == 0.3


def test_optimizer_failed_remains_failed_despite_finite_correction():
    result = replace(_make_result(), success=False, message="maximum evaluations")
    status = correction_usability_diagnostics(result)
    assert status["status"] == "optimizer_not_converged"
    assert not result.success and not status["optimizer_success"]


def test_all_invalid_input_and_missing_floor_metadata_json_safe():
    result = _make_result()
    result = replace(
        result,
        spectrum=replace(result.spectrum, mask=np.zeros(100, dtype=bool)),
        provenance={},
    )
    status = correction_usability_diagnostics(result)
    assert status["status"] == "no_valid_input" and status["retained_fraction"] is None
    assert (
        status["minimum_transmission"] is None and status["pixels_below_transmission_floor"] is None
    )
    json.dumps(status, allow_nan=False)


def test_no_floor_metadata_does_not_invent_opacity_diagnosis():
    result = _make_result()
    result = replace(
        result,
        provenance={},
        corrected=replace(result.corrected, mask=np.zeros(100, dtype=bool)),
    )
    status = correction_usability_diagnostics(result)
    assert status["status"] == "empty_correction"
    assert status["reasons"] == ["no_retained_corrected_samples"]


def test_noisy_negative_observed_flux_is_not_itself_invalid_model():
    result = _make_result()
    flux = result.spectrum.flux.copy()
    flux[:30] = -0.02
    result = replace(result, spectrum=replace(result.spectrum, flux=flux))
    assert correction_usability_diagnostics(result)["status"] == "usable"
