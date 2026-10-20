import json
from types import SimpleNamespace

import numpy as np
import pytest

from pymolfit import LineList, Spectrum, correct_arrays, load_fit_product
from pymolfit._automatic import (
    automatic_regions,
    automatic_robust_scale,
    choose_continuum_order,
    estimate_missing_uncertainty,
    mask_uncertain_correction,
    sampling_indices,
)
from pymolfit.fit import FitConfig, _shift_basis, fit_telluric_segments
from pymolfit.model import optical_depth_basis, transmission_from_basis
from pymolfit.regions import RegionSelection


def quiet_options():
    return {
        "line_list": LineList.empty_hitran(),
        "physical": False,
        "lsf_sigma_pixels": 0.0,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "lsf_variable_width": False,
        "fit_wavelength_shift": False,
        "high_resolution_grid": False,
        "auto_segment": False,
    }


def test_noise_estimate_preserves_supplied_errors():
    s = Spectrum(np.linspace(1.0, 1.01, 100), np.ones(100), uncertainty=np.full(100, 0.03))
    estimated, details = estimate_missing_uncertainty(s)
    assert estimated is s
    assert details == {"source": "supplied", "rescaled": False}


def test_noise_estimate_respects_overlapping_orders_gaps_masks_and_irregular_sampling():
    rng = np.random.default_rng(73)
    wave = np.cumsum(rng.uniform(0.5e-5, 1.5e-5, 2000)) + 1.0
    flux = 1.0 + 0.1 * wave + rng.normal(0.0, 0.01, wave.size)
    s = Spectrum(
        np.concatenate((wave, wave)),
        np.concatenate((flux, flux * 4)),
        group_id=np.repeat([11, 23], wave.size),
    )
    estimated, details = estimate_missing_uncertainty(s)
    assert np.median(estimated.uncertainty[:2000]) == pytest.approx(0.01, rel=0.2)
    assert np.median(estimated.uncertainty[2000:]) == pytest.approx(0.04, rel=0.2)
    np.testing.assert_array_equal(estimated.flux, s.flux)
    assert s.uncertainty is None
    assert len(sampling_indices(s)) == 2
    json.dumps(details, allow_nan=False)


def test_noise_estimate_handles_constant_invalid_and_short_sections():
    s = Spectrum(np.arange(10.0) + 1.0, np.ones(10), mask=np.arange(10) != 3)
    estimated, details = estimate_missing_uncertainty(s)
    assert np.all(estimated.uncertainty > 0)
    np.testing.assert_array_equal(estimated.valid, s.valid)
    assert details["blocks"][0]["source"] == "fractional_fallback"


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_continuum_order_validation_recovers_supported_curvature(degree):
    rng = np.random.default_rng(8)
    x = np.linspace(-1, 1, 2000)
    coefficients = np.array([1.0, 0.1, 0.3, -0.25, 0.3])[: degree + 1]
    flux = np.polynomial.polynomial.polyval(x, coefficients) + rng.normal(0.0, 0.003, x.size)
    order, details = choose_continuum_order(
        [(x, flux, np.full(x.size, 0.003), np.ones(x.size), np.ones(x.size, bool))],
        loss="soft_l1",
        f_scale=1.0,
    )
    assert order == degree
    json.dumps(details, allow_nan=False)


def test_continuum_order_retains_quadratic_with_insufficient_information():
    order, details = choose_continuum_order([], loss="soft_l1", f_scale=1.0)
    assert order == 2
    assert details["candidates"][2]["segments"] == 0


def test_region_windows_are_distributed_and_preserve_exclusions_and_gaps():
    w = np.concatenate((np.linspace(1.0, 1.01, 1000), np.linspace(1.1, 1.11, 1000)))
    lines = LineList(
        wavelength=np.array([1.001, 1.009, 1.105, 1.109]),
        strength=np.array([1.0, 0.5, 1.0, 0.2]),
        species=np.array(["H2O"] * 4),
        sigma=np.full(4, 1.0e-5),
        gamma=np.full(4, 1.0e-6),
    )
    windows, details = automatic_regions(
        Spectrum(w, np.ones(w.size)), lines, 2.0, ((1.104, 1.106),)
    )
    assert len(windows) == 3
    for lower, upper in windows:
        assert upper < 1.01 or lower > 1.1
        assert not lower <= 1.105 <= upper
    assert details["selected_line_count"] == 3


def test_explicit_controls_and_manual_windows_override_policies():
    w = np.linspace(1.0, 1.01, 500)
    r = correct_arrays(
        w,
        1.0 + 0.1 * (w - 1.0),
        uncertainty=np.full(w.size, 0.02),
        continuum_order=1,
        f_scale=1.7,
        min_transmission=0.1,
        fit_ranges=((1.002, 1.008),),
        **quiet_options(),
    )
    assert r.success
    assert r.provenance["automatic_continuum"]["selected_order"] == 1
    assert r.provenance["robust_scale"]["selected_scale"] == 1.7
    assert r.provenance["correction_mask"]["source"] == "fixed_threshold"
    assert r.provenance["fit_region_selection"]["source"] == "explicit"
    np.testing.assert_array_equal(r.fit_mask, (w >= 1.002) & (w <= 1.008))


def test_default_correction_selects_continuum_and_serializes_provenance(tmp_path):
    w = np.linspace(1.0, 1.01, 1000)
    x = np.linspace(-1.0, 1.0, w.size)
    flux = 1.0 + 0.1 * x + 0.3 * x * x + np.random.default_rng(4).normal(0.0, 0.005, w.size)
    with pytest.warns(RuntimeWarning, match="No usable automatic"):
        r = correct_arrays(w, flux, **quiet_options())
    assert r.success
    assert r.provenance["automatic_continuum"]["selected_order"] == 2
    assert r.provenance["fit_config"]["fields"]["loss"] == "soft_l1"
    assert r.provenance["input_uncertainty"]["source"] == "estimated"
    assert r.provenance["correction_mask"] == {
        "source": "fixed_threshold",
        "minimum_transmission": 0.01,
    }
    json.dumps(r.provenance)
    for name in ("automatic_continuum", "input_uncertainty", "robust_scale", "correction_mask"):
        json.dumps(r.provenance[name], allow_nan=False)
    r.write(tmp_path / "fit.ecsv")


def test_exclusion_only_file_disables_automatic_windows(tmp_path):
    path = tmp_path / "regions.ecsv"
    RegionSelection(exclude_ranges=((1.004, 1.006),)).write(path)
    w = np.linspace(1.0, 1.01, 500)
    result = correct_arrays(
        w, np.ones(w.size), region_file=path, continuum_order=0, **quiet_options()
    )
    assert result.provenance["fit_region_selection"]["source"] == "all_valid_pixels"
    np.testing.assert_array_equal(result.fit_mask, (w < 1.004) | (w > 1.006))


@pytest.mark.parametrize(
    "option,value",
    [
        ("continuum_order", -1),
        ("f_scale", 0),
        ("min_transmission", 1.0),
        ("max_corrected_relative_uncertainty", 0.0),
    ],
)
def test_invalid_automatic_overrides_rejected(option, value):
    w = np.linspace(1.0, 1.01, 100)
    with pytest.raises(ValueError):
        correct_arrays(w, np.ones(w.size), **quiet_options(), **{option: value})


def test_automatic_noise_mask_only_masks_uncertain_correction():
    w = np.linspace(1.0, 1.01, 5)
    transmission = np.array([1.0, 0.8, 0.3, 0.02, 0.005])
    spectrum = Spectrum(w, transmission, uncertainty=np.full(5, 0.01))
    corrected = Spectrum(w, np.ones(5), uncertainty=0.01 / transmission, mask=transmission >= 0.01)
    result = SimpleNamespace(
        spectrum=spectrum, corrected=corrected, transmission=transmission, continuum=np.ones(5)
    )
    # Use a real dataclass so the output uses the same immutable replacement path.
    from dataclasses import make_dataclass

    Result = make_dataclass("Result", [(name, object) for name in vars(result)])
    result = Result(**vars(result))
    updated, details = mask_uncertain_correction(result, relative_limit=0.1)
    np.testing.assert_array_equal(updated.corrected.valid, [True, True, True, False, False])
    np.testing.assert_array_equal(updated.spectrum.flux, spectrum.flux)
    assert details["additional_masked_pixels"] == 1
    assert np.isfinite(result.corrected.flux[3])


def test_default_retains_uncertain_divisions_without_changing_the_fit(tmp_path):
    wave = np.linspace(1.0, 1.002, 401)
    lines = LineList(
        wavelength=np.array([1.001]),
        strength=np.array([5.0e-4]),
        sigma=np.array([3.0e-5]),
        gamma=np.array([3.0e-6]),
        species=np.array(["H2O"]),
    )
    species, basis = optical_depth_basis(wave, lines)
    flux = transmission_from_basis(species, basis, species_scales={"H2O": 1.0})
    flux[0], flux[1], flux[2] = 0.0, -0.001, np.nan
    valid = np.ones(wave.size, bool)
    valid[3] = False
    options = {
        **quiet_options(),
        "line_list": lines,
        "uncertainty": np.full(wave.size, 0.05),
        "mask": valid,
        "automatic_fit_regions": False,
        "continuum_order": 0,
        "fixed_species_scales": {"H2O": 1.0},
        "loss": "linear",
        "f_scale": 1.0,
    }
    default = correct_arrays(wave, flux, **options)
    strict = correct_arrays(wave, flux, min_transmission="auto", **options)
    keep = default.spectrum.valid & (default.transmission >= 0.01)
    assert np.any(default.transmission < 0.01)
    np.testing.assert_array_equal(default.corrected.valid, keep)
    assert default.corrected.valid[0] and default.corrected.valid[1]
    assert not default.corrected.valid[2] and not default.corrected.valid[3]
    recovered = default.corrected.valid & ~strict.corrected.valid
    assert recovered.any()
    assert np.all(default.corrected.uncertainty[recovered] > 0.1 * default.continuum[recovered])
    np.testing.assert_allclose(
        default.corrected.flux[keep], flux[keep] / default.transmission[keep]
    )
    np.testing.assert_allclose(
        default.corrected.uncertainty[keep], 0.05 / default.transmission[keep]
    )
    for name in ("transmission", "continuum", "model_flux", "fit_mask"):
        np.testing.assert_array_equal(getattr(default, name), getattr(strict, name))
    assert default.species_scales == strict.species_scales
    assert strict.provenance["correction_mask"]["source"] == "propagated_uncertainty"
    default.write(tmp_path / "default.ecsv")
    loaded = load_fit_product(tmp_path / "default.ecsv")
    np.testing.assert_array_equal(loaded.corrected.valid, default.corrected.valid)
    np.testing.assert_allclose(loaded.corrected.uncertainty, default.corrected.uncertainty)


def test_robust_scale_is_bounded():
    rng = np.random.default_rng(2)
    s = Spectrum(np.arange(500.0), rng.normal(0.0, 10.0, 500), uncertainty=np.ones(500))
    result = SimpleNamespace(
        spectrum=s, fit_mask=None, transmission=np.ones(500), model_flux=np.zeros(500)
    )
    assert automatic_robust_scale(result) == 2.0


@pytest.mark.parametrize("sparse", [True, False])
def test_mixed_wavelength_degrees_fit_only_supported_parameters(sparse):
    spectra, lines = [], []
    for group in range(3):
        w = np.linspace(1.0 + group * 0.02, 1.01 + group * 0.02, 700)
        centers = np.linspace(w[70], w[-70], 5)
        local = LineList(
            wavelength=centers,
            strength=np.full(5, 1.0e-5),
            sigma=np.full(5, 1.2e-5),
            gamma=np.full(5, 3.0e-6),
            species=np.array(["H2O"] * 5),
        )
        species, basis = optical_depth_basis(w, local)
        x = np.linspace(-1.0, 1.0, w.size)
        shift = 0.0 if group == 0 else (0.8 if group == 1 else 0.5 + 0.7 * x + 0.9 * x * x)
        flux = transmission_from_basis(
            species, _shift_basis(w, basis, shift * np.gradient(w)), species_scales={"H2O": 1.2}
        )
        spectra.append(Spectrum(w, flux, uncertainty=np.full(w.size, 0.002)))
        lines.append(local)
    combined = LineList(
        **{
            field: np.concatenate([getattr(line, field) for line in lines])
            for field in ("wavelength", "strength", "sigma", "gamma", "species")
        }
    )
    fit = fit_telluric_segments(
        spectra,
        line_list=combined,
        wavelength_group_ids=(10, 20, 30),
        config=FitConfig(
            continuum_order=0,
            solve_continuum_linear=True,
            fit_segment_wavelength_polynomial=True,
            segment_wavelength_polynomial_order=2,
            wavelength_shift_unit="pixel",
            wavelength_shift_bounds=(-3.0, 3.0),
            wavelength_group_polynomial_orders={10: -1, 20: 0, 30: 2},
            use_jacobian_sparsity=sparse,
        ),
    )
    assert fit.success
    names = [name for name in fit.parameter_names if name.startswith("wavelength_group")]
    assert len(names) == 4
    assert not any("group:10:" in name for name in names)
    np.testing.assert_allclose(fit.wavelength_group_coefficients[20], [0.8, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(fit.wavelength_group_coefficients[30], [0.5, 0.7, 0.9], atol=0.02)


def test_automatic_wavelength_models_select_quadratic_only_where_supported():
    wave_parts, flux_parts, catalogs = [], [], []
    for index in range(2):
        w = np.linspace(1.0 + index * 0.02, 1.01 + index * 0.02, 1500)
        x = np.linspace(-1.0, 1.0, w.size)
        lines = LineList(
            wavelength=np.linspace(w[150], w[-150], 5),
            strength=np.full(5, 1.0e-5),
            sigma=np.full(5, 1.2e-5),
            gamma=np.full(5, 3.0e-6),
            species=np.full(5, "H2O"),
        )
        species, basis = optical_depth_basis(w, lines)
        shift = np.full(w.size, 0.8) if index == 0 else 0.5 + 0.8 * x + 1.5 * x * x
        flux = transmission_from_basis(
            species, _shift_basis(w, basis, shift * np.gradient(w)), species_scales={"H2O": 1.2}
        )
        wave_parts.append(w)
        flux_parts.append(flux)
        catalogs.append(lines)
    catalog = LineList(
        **{
            field: np.concatenate([getattr(line, field) for line in catalogs])
            for field in ("wavelength", "strength", "sigma", "gamma", "species")
        }
    )
    result = correct_arrays(
        np.concatenate(wave_parts),
        np.concatenate(flux_parts),
        uncertainty=np.full(3000, 0.001),
        group_id=np.repeat([10, 20], 1500),
        line_list=catalog,
        continuum_order=0,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_variable_width=False,
        high_resolution_grid=False,
        segment_size=0.004,
    )
    groups = result.provenance["wavelength_alignment"]["groups"]
    assert result.success
    assert [group["polynomial_order"] for group in groups] == [0, 2]
    np.testing.assert_allclose(result.model_flux, result.spectrum.flux, atol=0.002)
