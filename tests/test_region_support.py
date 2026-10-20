from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pymolfit import LineList, correct_arrays, workflow
from pymolfit._region_support import local_support_ranges, refine_region_support
from pymolfit.fit import _windows_mask
from pymolfit.model import optical_depth_basis, transmission_from_basis
from tests.test_continuum_validation import _setup


def support_case():
    reference, config, _, start = _setup()
    wave = reference.spectrum.wavelength
    reference.provenance["segmentation"]["boundaries_micron"] = [(wave[0], wave[-1])]
    catalogue = tuple((wave[i - 25], wave[i + 24]) for i in (180, 450, 720))
    ranges = local_support_ranges(
        (reference.spectrum,), catalogue, None, continuum_order=config.continuum_order
    )
    calls = []

    def run(trial):
        calls.append(trial)
        mask = reference.spectrum.valid & _windows_mask(
            wave, trial.fit_ranges, trial.exclude_ranges
        )
        residual = 0.7 if trial.fit_ranges == ranges else 1.0
        return SimpleNamespace(
            **{
                **vars(reference),
                "fit_mask": mask,
                "model_flux": reference.spectrum.flux - residual,
            }
        )

    def refine(fit=run, **kwargs):
        return refine_region_support(
            reference,
            config,
            (reference.spectrum,),
            kwargs.get("catalogue", catalogue),
            run_fit=fit,
            starting_config=start,
        )

    return reference, config, catalogue, ranges, calls, run, refine


def test_reduced_support_requires_paired_guarded_improvement():
    reference, config, catalogue, ranges, calls, _, refine = support_case()
    result, selected, details = refine()
    assert details["accepted"]
    assert selected.fit_ranges == ranges
    assert selected.continuum_order == config.continuum_order
    assert result.fit_mask.sum() == 225
    assert len(calls) == 7
    assert selected is calls[-1]
    assert all(c.initial_species_scales is None for c in calls)
    original_windows = _windows_mask(reference.spectrum.wavelength, catalogue, None)
    for fold in details["plan"]["folds"]:
        for block in fold["blocks"]:
            lo, hi = block["core_indices"]
            assert original_windows[lo : hi + 1].sum() >= 8
    assert details["validation"]["pooled_rms_improvement_fraction"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "failure",
    ["negative_continuum", "nonconvergence", "worse_prediction", "changed_mask", "new_bound"],
)
def test_failed_reduced_support_keeps_original_result_and_config(failure):
    reference, config, _, ranges, _, run, refine = support_case()

    def bad(trial):
        result = run(trial)
        if trial.fit_ranges == ranges:
            if failure == "negative_continuum":
                result.continuum = np.full(result.continuum.size, -1.0)
            elif failure == "nonconvergence":
                result.success = False
            elif failure == "worse_prediction":
                result.model_flux = reference.spectrum.flux - 1.2
            elif failure == "new_bound":
                result.parameter_bound_status = {"lsf_sigma_pixels": "upper"}
            else:
                result.fit_mask = np.ones(result.fit_mask.size, bool)
        return result

    result, selected, details = refine(bad)
    assert result is reference and selected is config
    assert not details["accepted"]


def test_no_separated_validation_support_retains_broad_fit_without_refitting():
    reference, config, _, _, calls, _, refine = support_case()
    wave = reference.spectrum.wavelength
    result, selected, details = refine(catalogue=((wave[430], wave[469]),))
    assert result is reference and selected is config
    assert details["reason"] == "insufficient_guarded_support"
    assert not calls


@pytest.mark.parametrize("failure", ["catalogue_regression", "empty_correction"])
def test_validation_gain_cannot_override_a_bad_final_refit(failure):
    reference, config, _, ranges, calls, run, refine = support_case()

    def bad_final(trial):
        result = run(trial)
        if trial.fit_ranges == ranges and not trial.exclude_ranges:
            if failure == "catalogue_regression":
                result.model_flux = reference.spectrum.flux - 1.1
            else:
                result.transmission = np.zeros(reference.transmission.size)
        return result

    result, selected, details = refine(bad_final)
    assert len(calls) == 7
    assert details["validation"]["accepted"]
    assert not details["accepted"]
    assert details["reason"] == "catalogue_or_physical_regression"
    assert result is reference and selected is config


def test_local_support_respects_exclusions_and_preserves_unseeded_chunks():
    reference, config, catalogue, _, _, _, _ = support_case()
    spectrum = reference.spectrum
    wave = spectrum.wavelength
    exclusions = ((wave[140], wave[160]),)
    ranges = local_support_ranges((spectrum,), catalogue, exclusions, continuum_order=1)
    selected = _windows_mask(wave, ranges, exclusions)
    assert not selected[140:161].any()
    assert selected.sum() == int(np.ceil(0.25 * (900 - 21)))
    unseeded = replace(spectrum, wavelength=wave + 1)
    ranges = local_support_ranges(
        (spectrum, unseeded), catalogue, exclusions, continuum_order=config.continuum_order
    )
    assert np.all(_windows_mask(unseeded.wavelength, ranges, exclusions))


def test_uncertainty_mode_is_kept_for_the_final_fit_but_not_validation_folds():
    reference, config, catalogue, _, calls, run, _ = support_case()
    config = replace(config, estimate_uncertainties=True)
    result, selected, details = refine_region_support(
        reference, config, (reference.spectrum,), catalogue, run_fit=run, starting_config=config
    )
    assert result is not reference and selected.estimate_uncertainties
    assert details["accepted"]
    assert not any(c.estimate_uncertainties for c in calls[:-1])
    assert calls[-1].estimate_uncertainties


def test_real_fits_recover_molecular_scale_despite_distant_stellar_absorption(monkeypatch):
    wave = np.linspace(1.0, 1.009, 900)
    lines = LineList(
        wavelength=np.array([1.0018, 1.0045, 1.0072]),
        strength=np.full(3, 1e-4),
        sigma=np.full(3, 3e-5),
        gamma=np.full(3, 3e-6),
        species=np.array(["H2O"] * 3),
    )
    species, basis = optical_depth_basis(wave, lines)
    flux = (1 + 0.1 * np.linspace(-1, 1, wave.size)) * transmission_from_basis(
        species, basis, species_scales={"H2O": 1.2}
    )
    flux *= 1 - 0.4 * np.maximum(0, 1 - ((wave - 1.0032) / 0.00045) ** 2) ** 2
    options = {
        "uncertainty": np.full(wave.size, 0.01),
        "line_list": lines,
        "physical": False,
        "continuum_order": 1,
        "lsf_sigma_pixels": 0.0,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "lsf_variable_width": False,
        "fit_wavelength_shift": False,
        "high_resolution_grid": False,
        "auto_segment": False,
        "estimate_uncertainties": False,
    }
    with monkeypatch.context() as patch:
        patch.setattr(
            workflow,
            "refine_region_support",
            lambda result, config, *a, **kw: (result, config, {"accepted": False}),
        )
        broad = correct_arrays(wave, flux, **options)
    reduced = correct_arrays(wave, flux, **options)
    assert reduced.provenance["fit_region_selection"]["support_refinement"]["accepted"]
    assert broad.fit_mask.sum() == 900
    assert reduced.fit_mask.sum() == 225
    assert reduced.species_scales["H2O"] == pytest.approx(1.2, abs=1e-7)
    assert abs(reduced.species_scales["H2O"] - 1.2) < abs(broad.species_scales["H2O"] - 1.2)
    np.testing.assert_array_equal(reduced.spectrum.flux, broad.spectrum.flux)
    np.testing.assert_array_equal(reduced.spectrum.uncertainty, broad.spectrum.uncertainty)
    assert reduced.provenance["validation_region_policy"] == "nonexcluded"
    manual = correct_arrays(wave, flux, **options, fit_ranges=((wave[120], wave[750]),))
    assert manual.provenance["validation_region_policy"] == "fitting"
    assert "support_refinement" not in manual.provenance["fit_region_selection"]
