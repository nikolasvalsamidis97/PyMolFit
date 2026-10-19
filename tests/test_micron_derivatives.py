"""Wavelength derivatives resolve detector scales without changing pixel routes."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

import pymolfit.fit as fit_module
from pymolfit import LineList, ModelConfig, Spectrum, transmission_model


class _CapturedOptimizer(Exception):
    pass


@pytest.fixture
def line_spectrum():
    spacing = 1e-5
    wave = 4.08 + spacing * np.arange(480)
    lines = LineList(
        wavelength=wave[[70, 170, 310, 410]] + 0.21 * spacing,
        strength=np.array([0.8, 1.5, 1.0, 1.8]) * spacing,
        sigma=np.full(4, 0.7 * spacing),
        gamma=np.full(4, 0.12 * spacing),
        species=np.full(4, "H2O"),
    )
    truth_lines = replace(lines, wavelength=lines.wavelength + 0.37 * spacing)
    transmission = transmission_model(
        wave,
        truth_lines,
        ModelConfig(
            species_scales={"H2O": 1.3},
            lsf_sigma_pixels=0.9,
            high_resolution_grid=True,
        ),
    )
    continuum = 1.1 + 0.035 * np.linspace(-1, 1, wave.size)
    spectrum = Spectrum(wave, continuum * transmission, uncertainty=np.full(wave.size, 0.002))
    spectra = [
        replace(
            spectrum,
            wavelength=wave[start:stop],
            flux=spectrum.flux[start:stop],
            uncertainty=spectrum.uncertainty[start:stop],
        )
        for start, stop in ((0, 240), (240, 480))
    ]
    return spectrum, spectra, lines, spacing


def _run_fit(line_spectrum, route, unit, highres, **options):
    spectrum, spectra, lines, spacing = line_spectrum
    unit_scale = spacing if unit == "micron" else 1.0
    config = fit_module.FitConfig(
        continuum_order=1,
        solve_continuum_linear=True,
        initial_species_scales={"H2O": 1.15},
        fit_wavelength_shift=route != "grouped_two",
        fit_segment_wavelength_shifts=route == "grouped_two",
        wavelength_shift_unit=unit,
        initial_wavelength_shift=0.24 * unit_scale,
        wavelength_shift_bounds=(-3 * unit_scale, 3 * unit_scale),
        fit_lsf_sigma=True,
        lsf_sigma_pixels=0.82,
        lsf_sigma_bounds=(0.4, 2.0),
        initialize_wavelength_shift_grid=False,
        initialize_lsf_sigma_grid=False,
        staged_optimization=False,
        loss="soft_l1",
        f_scale=1,
        high_resolution_grid=highres,
        max_nfev=80,
        **options,
    )
    if route == "single":
        return fit_module.fit_tellurics(spectrum, line_list=lines, config=config)
    return fit_module.fit_telluric_segments(
        [spectrum] if route == "global_one" else spectra,
        line_list=lines,
        config=config,
        wavelength_group_ids=[0] if route == "global_one" else [0, 0],
    )


def _capture_optimizer(monkeypatch, line_spectrum, route, unit, highres):
    captured = {}

    def intercept(fun, x0, **kwargs):
        captured.update(fun=fun, x0=x0, kwargs=kwargs)
        raise _CapturedOptimizer

    with monkeypatch.context() as patch:
        patch.setattr(fit_module, "least_squares", intercept)
        with pytest.raises(_CapturedOptimizer):
            _run_fit(line_spectrum, route, unit, highres)
    assert captured["x0"].size == 3
    spacing = line_spectrum[-1]
    return captured, np.array([1.0, spacing if unit == "micron" else 1.0, 1.0])


@pytest.mark.parametrize("highres", [False, True])
def test_grouped_micron_residual_jacobian_resolves_wavelength_derivative(
    monkeypatch, line_spectrum, highres
):
    captured, scales = _capture_optimizer(
        monkeypatch, line_spectrum, "grouped_two", "micron", highres
    )
    fun, x = captured["fun"], captured["x0"]
    actual = captured["kwargs"]["jac"](x) * scales
    columns = []
    for index, scale in enumerate(scales):
        delta = np.zeros_like(x)
        delta[index] = 1e-4 * scale
        columns.append((fun(x + delta) - fun(x - delta)) / 2e-4)
    reference = np.column_stack(columns)
    relative = np.linalg.norm(actual - reference, axis=0) / np.linalg.norm(reference, axis=0)
    assert np.all(relative < 3e-5), relative


def test_grouped_micron_optimization_matches_pixel_solution(monkeypatch, line_spectrum):
    fits = []
    for unit in ("pixel", "micron"):
        captured, scales = _capture_optimizer(monkeypatch, line_spectrum, "grouped_two", unit, True)
        result = least_squares(captured["fun"], captured["x0"], **captured["kwargs"])
        assert result.success
        assert result.cost < 0.05
        fits.append(result.x / scales)
    np.testing.assert_allclose(fits[0], fits[1], rtol=0, atol=1e-6)


def test_characteristic_scales_and_bounds_preserve_physical_steps():
    x = np.array([0.0, 0.24, -0.24, 1.0, -1.0])
    lo, hi = np.full(5, -1.0), np.ones(5)
    scales = np.array([1e-5, 1e-5, 1.0, 1.0, 1.0])
    steps = fit_module._bounded_forward_steps(x, lo, hi, parameter_scales=scales)
    assert steps[0] == pytest.approx(np.cbrt(np.finfo(float).eps) * 1e-5)
    assert steps[3] < 0 and steps[4] > 0
    assert np.all((x + steps >= lo) & (x + steps <= hi))
    assert np.all(steps != 0)
    factor = np.array([1e5, 1e-5, 3.0, 5.0, 0.2])
    changed = fit_module._bounded_forward_steps(
        x * factor, lo * factor, hi * factor, parameter_scales=scales * factor
    )
    np.testing.assert_allclose(changed / factor, steps, rtol=1e-10)


def test_unscaled_forward_steps_keep_legacy_pixel_displacements():
    x = np.array([0.0, 0.24, -0.24, 1.0, -1.0, 30.0, -30.0])
    lo, hi = -30 * np.ones_like(x), 30 * np.ones_like(x)
    step = np.cbrt(np.finfo(float).eps) * np.where(x >= 0, 1.0, -1.0) * np.maximum(1, np.abs(x))
    step[-2:] *= -1  # Upper/lower bounds reverse the forward direction.
    expected = (x + step) - x
    np.testing.assert_array_equal(fit_module._bounded_forward_steps(x, lo, hi), expected)


@pytest.mark.parametrize("scales", [np.array([0.0, 1.0]), np.array([np.nan, 1.0]), np.ones(3)])
def test_invalid_characteristic_scales_are_rejected(scales):
    with pytest.raises(ValueError, match="parameter scales"):
        fit_module._bounded_forward_steps(
            np.zeros(2), -np.ones(2), np.ones(2), parameter_scales=scales
        )


def test_sparse_and_dense_derivatives_agree_with_mixed_parameter_units():
    physical_matrix = np.array(
        [
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 5.0, 0.0],
        ]
    )
    scales = np.array([1e-5, 2e-5, 1.0])
    matrix = physical_matrix / scales
    x = np.array([0.2e-5, -0.3e-5, 0.4])
    lo, hi = -3 * scales, 3 * scales

    def fun(p):
        return matrix @ p

    dense = fit_module._grouped_dense_jacobian(
        fun, csr_matrix(np.ones_like(matrix, dtype=bool)), lo, hi, parameter_scales=scales
    )(x)
    grouped = fit_module._grouped_dense_jacobian(
        fun, csr_matrix(matrix != 0), lo, hi, parameter_scales=scales
    )(x)
    np.testing.assert_allclose(dense, matrix, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(dense, grouped)


def test_propagated_uncertainty_is_invariant_to_parameter_units():
    x = np.array([0.2, 0.4])
    covariance = np.array([[0.01, 0.002], [0.002, 0.04]])
    matrix = np.array([[2.0, 1.0], [1.0, -3.0]])
    lo, hi = -np.ones(2), np.ones(2)
    expected = np.sqrt(np.diag(matrix @ covariance @ matrix.T))
    for factor in (np.ones(2), np.array([1e-5, 1.0])):
        actual = fit_module._finite_difference_output_uncertainty(
            lambda p, factor=factor: matrix @ (p / factor),
            x * factor,
            covariance * factor[:, None] * factor[None, :],
            lo * factor,
            hi * factor,
            np.arange(2),
            parameter_scales=factor,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9)


@pytest.mark.parametrize("highres", [False, True])
@pytest.mark.parametrize("route", ["single", "global_one", "global_two", "grouped_two"])
def test_pixel_fit_and_uncertainty_paths_keep_unscaled_steps(
    monkeypatch, line_spectrum, highres, route
):
    calls = {"jacobian": 0, "uncertainty": 0}
    original_jacobian = fit_module._grouped_dense_jacobian
    original_uncertainty = fit_module._finite_difference_output_uncertainty

    def jacobian(*args, **kwargs):
        calls["jacobian"] += 1
        assert kwargs.get("parameter_scales") is None
        return original_jacobian(*args, **kwargs)

    def uncertainty(*args, **kwargs):
        calls["uncertainty"] += 1
        assert kwargs.get("parameter_scales") is None
        return original_uncertainty(*args, **kwargs)

    monkeypatch.setattr(fit_module, "_grouped_dense_jacobian", jacobian)
    monkeypatch.setattr(fit_module, "_finite_difference_output_uncertainty", uncertainty)
    _run_fit(line_spectrum, route, "pixel", highres, estimate_uncertainties=True)
    assert calls["uncertainty"] > 0
    assert bool(calls["jacobian"]) == (route == "grouped_two")


@pytest.mark.parametrize("route", ["single", "global_one", "global_two"])
def test_single_and_global_micron_optimization_keeps_scipy_jacobian(
    monkeypatch, line_spectrum, route
):
    captured, _ = _capture_optimizer(monkeypatch, line_spectrum, route, "micron", True)
    assert captured["kwargs"].get("jac", "2-point") == "2-point"


@pytest.mark.parametrize("profile_continuum", [False, True])
@pytest.mark.parametrize("polynomial_order", [0, 2])
def test_each_wavelength_group_uses_its_detector_dispersion(
    monkeypatch, line_spectrum, profile_continuum, polynomial_order
):
    _, spectra, lines, spacing = line_spectrum
    second = spectra[1]
    spectra[1] = replace(
        second,
        wavelength=second.wavelength[0] + 2 * (second.wavelength - second.wavelength[0]),
    )
    captured = {}
    original_jacobian = fit_module._grouped_dense_jacobian

    def jacobian(*args, **kwargs):
        captured["scales"] = kwargs["parameter_scales"].copy()
        return original_jacobian(*args, **kwargs)

    def intercept(_fun, x0, **_kwargs):
        captured["parameters"] = x0
        raise _CapturedOptimizer

    monkeypatch.setattr(fit_module, "_grouped_dense_jacobian", jacobian)
    monkeypatch.setattr(fit_module, "least_squares", intercept)
    config = fit_module.FitConfig(
        continuum_order=1,
        solve_continuum_linear=profile_continuum,
        fit_wavelength_shift=False,
        fit_segment_wavelength_polynomial=True,
        segment_wavelength_polynomial_order=polynomial_order,
        wavelength_shift_unit="micron",
        wavelength_shift_bounds=(-3 * spacing, 3 * spacing),
        fit_lsf_sigma=True,
        lsf_sigma_pixels=0.82,
        lsf_sigma_bounds=(0.4, 2.0),
        initialize_wavelength_shift_grid=False,
        initialize_lsf_sigma_grid=False,
        staged_optimization=False,
    )
    with pytest.raises(_CapturedOptimizer):
        fit_module.fit_telluric_segments(
            spectra, line_list=lines, config=config, wavelength_group_ids=[7, 12]
        )
    # Species/continuum/LSF scales stay dimensionless. Each wavelength
    # coefficient follows its own detector dispersion, regardless of the
    # number of preceding continuum parameters or group coefficients.
    expected = np.concatenate(
        [
            np.ones(1 + (0 if profile_continuum else 4)),
            np.full(polynomial_order + 1, spacing),
            np.full(polynomial_order + 1, 2 * spacing),
            np.ones(1),
        ]
    )
    assert captured["parameters"].shape == expected.shape
    np.testing.assert_allclose(captured["scales"], expected, rtol=1e-9, atol=0)
