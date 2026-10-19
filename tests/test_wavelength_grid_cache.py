from dataclasses import replace

import numpy as np
import pytest

import pymolfit.fit as fit_module
from pymolfit import FitConfig, Spectrum, fit_telluric_segments
from pymolfit.fit import _high_resolution_margin_pixels, _opacity_cache_key
from tests.helpers import demo_lines


@pytest.mark.parametrize("per_group", [False, True])
def test_polynomial_grid_margin_covers_sum_of_all_coefficient_bounds(per_group):
    wavelength = np.linspace(1.0, 1.01, 100)
    constant = FitConfig(
        high_resolution_grid=True,
        high_resolution_margin_pixels=0.0,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=(-12.0, 12.0),
        fit_wavelength_shift=True,
    )
    quadratic = replace(
        constant,
        fit_wavelength_shift=False,
        fit_wavelength_polynomial=not per_group,
        fit_segment_wavelength_polynomial=per_group,
        wavelength_polynomial_order=2,
        segment_wavelength_polynomial_order=2,
    )
    assert _high_resolution_margin_pixels(wavelength, quadratic) == pytest.approx(
        _high_resolution_margin_pixels(wavelength, constant) + 24.0
    )
    assert _high_resolution_margin_pixels(wavelength, quadratic) >= 36.0


def test_opacity_cache_keys_follow_effective_grid_margins():
    wavelength = np.linspace(2.31, 2.36, 240)
    lines = demo_lines()
    baseline = FitConfig(high_resolution_grid=True, wavelength_shift_unit="pixel")
    expanded = replace(
        baseline,
        fit_wavelength_polynomial=True,
        wavelength_polynomial_order=2,
        wavelength_shift_bounds=(-12.0, 12.0),
    )
    assert _opacity_cache_key(wavelength, lines, baseline) != _opacity_cache_key(
        wavelength, lines, expanded
    )
    assert _opacity_cache_key(wavelength, lines, baseline) == _opacity_cache_key(
        wavelength, lines, replace(baseline)
    )
    # Native-grid opacity is independent of the fitted wavelength offset.
    assert _opacity_cache_key(
        wavelength, lines, replace(baseline, high_resolution_grid=False)
    ) == _opacity_cache_key(wavelength, lines, replace(expanded, high_resolution_grid=False))


def test_expanded_wavelength_fit_matches_uncached_fit_after_baseline_cache(monkeypatch):
    wavelength = np.linspace(2.31, 2.36, 240)
    lines = demo_lines().select_species(("H2O",))
    spectrum = Spectrum(wavelength, np.ones(wavelength.size), uncertainty=np.full(240, 0.01))
    baseline = FitConfig(
        continuum_order=0,
        solve_continuum_linear=True,
        high_resolution_grid=True,
        wavelength_shift_unit="pixel",
        fixed_species_scales={"H2O": 1.0},
    )
    expanded = replace(
        baseline,
        fit_wavelength_polynomial=True,
        wavelength_polynomial_order=2,
        wavelength_shift_bounds=(-12.0, 12.0),
    )
    calls = 0
    original_basis = fit_module.optical_depth_basis

    def counted_basis(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_basis(*args, **kwargs)

    monkeypatch.setattr(fit_module, "optical_depth_basis", counted_basis)
    cache = {}
    options = {"spectra": (spectrum,), "line_list": lines}
    first = fit_telluric_segments(**options, config=baseline, _opacity_cache=cache)
    repeated = fit_telluric_segments(**options, config=baseline, _opacity_cache=cache)
    assert calls == 1
    np.testing.assert_array_equal(
        first.segment_results[0].model_flux, repeated.segment_results[0].model_flux
    )
    cached = fit_telluric_segments(**options, config=expanded, _opacity_cache=cache)
    assert calls == 2
    uncached = fit_telluric_segments(**options, config=expanded)
    assert calls == 3
    assert cached.success and uncached.success
    np.testing.assert_array_equal(
        cached.segment_results[0].model_flux, uncached.segment_results[0].model_flux
    )
    np.testing.assert_array_equal(
        cached.segment_results[0].transmission, uncached.segment_results[0].transmission
    )
