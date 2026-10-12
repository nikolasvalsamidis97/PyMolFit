import json

import numpy as np
import pytest
from astropy.table import Table

import pymolfit.fit as fit_module
from pymolfit import (
    FitConfig,
    LineList,
    ModelConfig,
    Spectrum,
    fit_telluric_segments,
    fit_tellurics,
    transmission_model,
)
from pymolfit.fit import _linearized_parameter_covariance
from tests.helpers import demo_lines


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("solve_continuum_linear", [False, True])
def test_restricted_fit_preserves_continuum_coordinate(segmented, solve_continuum_linear):
    wavelength = np.linspace(2.31, 2.36, 600)
    coordinate = np.linspace(-1.0, 1.0, wavelength.size)
    continuum = 1.2 + 0.2 * coordinate + 0.1 * coordinate**2
    line_list = demo_lines()
    transmission = transmission_model(wavelength, line_list, ModelConfig())
    spectrum = Spectrum(
        wavelength=wavelength,
        flux=continuum * transmission,
        uncertainty=np.full(wavelength.size, 0.001),
    )
    config = FitConfig(
        continuum_order=2,
        solve_continuum_linear=solve_continuum_linear,
        fixed_species_scales=dict.fromkeys(np.unique(line_list.species), 1.0),
        fit_ranges=((2.318, 2.342),),
    )
    if segmented:
        result = fit_telluric_segments(
            [spectrum], line_list=line_list, config=config
        ).segment_results[0]
    else:
        result = fit_tellurics(spectrum, line_list=line_list, config=config)

    assert result.success
    np.testing.assert_allclose(result.continuum, continuum, atol=1e-8)
    np.testing.assert_allclose(result.model_flux, spectrum.flux, atol=1e-8)
    np.testing.assert_allclose(result.transmission, transmission, atol=1e-10)


def test_multi_segment_opacity_cache_reuses_basis_without_changing_result(monkeypatch):
    wavelength = np.linspace(2.31, 2.36, 240)
    line_list = demo_lines().select_species(("H2O",))
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.25}),
    )
    spectra = (Spectrum(wavelength=wavelength, flux=flux),)
    config = FitConfig(continuum_order=0, solve_continuum_linear=True)

    calls = 0
    original = fit_module.optical_depth_basis

    def counted_basis(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit_module, "optical_depth_basis", counted_basis)
    opacity_cache = {}
    first = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=config,
        _opacity_cache=opacity_cache,
    )
    second = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=config,
        _opacity_cache=opacity_cache,
    )
    assert calls == 1

    uncached = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=config,
    )
    assert calls == 2
    first_segment = first.segment_results[0]
    second_segment = second.segment_results[0]
    uncached_segment = uncached.segment_results[0]
    np.testing.assert_array_equal(first_segment.transmission, second_segment.transmission)
    np.testing.assert_array_equal(first_segment.transmission, uncached_segment.transmission)
    np.testing.assert_array_equal(first_segment.corrected.flux, second_segment.corrected.flux)
    np.testing.assert_array_equal(first_segment.corrected.flux, uncached_segment.corrected.flux)


def test_fit_tellurics_improves_synthetic_spectrum():
    rng = np.random.default_rng(4)
    wavelength = np.linspace(2.31, 2.36, 900)
    line_list = demo_lines()
    true_transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(airmass=1.1, species_scales={"H2O": 2.0, "CO2": 0.6, "CH4": 1.3}),
    )
    continuum = 1.2 + 0.08 * (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    flux = continuum * true_transmission + rng.normal(0.0, 0.003, wavelength.size)
    spectrum = Spectrum(wavelength=wavelength, flux=flux, uncertainty=np.full_like(flux, 0.003))

    result = fit_tellurics(
        spectrum,
        line_list=line_list,
        config=FitConfig(airmass=1.1, continuum_order=1),
    )

    raw_scatter = np.nanstd(flux / continuum - 1.0)
    corrected_scatter = np.nanstd(result.corrected.flux / result.continuum - 1.0)
    assert result.success
    assert corrected_scatter < raw_scatter
    assert all(scale > 0 for scale in result.species_scales.values())


def test_fit_tellurics_accepts_fit_mask():
    wavelength = np.linspace(2.31, 2.36, 300)
    line_list = demo_lines()
    flux = transmission_model(wavelength, line_list, ModelConfig(species_scales={"H2O": 1.2}))
    spectrum = Spectrum(wavelength=wavelength, flux=flux)
    fit_mask = wavelength < 2.345

    result = fit_tellurics(spectrum, line_list=line_list, fit_mask=fit_mask)

    assert result.transmission.shape == wavelength.shape
    assert result.corrected.flux.shape == wavelength.shape
    np.testing.assert_array_equal(result.fit_mask, fit_mask)
    product = result.to_table()
    np.testing.assert_array_equal(product["fit_mask"], fit_mask)
    provenance = json.loads(product.meta["provenance_json"])
    assert provenance["line_source"] == "demo"
    assert provenance["line_count"] == line_list.wavelength.size
    assert provenance["selected_line_count"] <= provenance["line_count"]
    assert len(provenance["line_list_sha256"]) == 64
    assert len(provenance["fit_config_sha256"]) == 64


def test_fit_weights_downweight_a_biased_pixel():
    wavelength = np.linspace(1.0, 1.01, 801)
    line_list = LineList(
        wavelength=np.array([1.005]),
        strength=np.array([5.0e-4]),
        sigma=np.array([2.0e-5]),
        gamma=np.array([5.0e-6]),
        species=np.array(["H2O"]),
    )
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.0}),
    )
    biased = transmission.copy()
    center = int(np.argmin(np.abs(wavelength - 1.005)))
    biased[center + 5 : center + 14] *= 0.7
    spectrum = Spectrum(wavelength, biased)
    config = FitConfig(
        continuum_order=0,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
    )
    weights = np.ones_like(wavelength)
    weights[center + 5 : center + 14] = 0.01

    unweighted = fit_tellurics(spectrum, line_list=line_list, config=config)
    weighted = fit_tellurics(
        spectrum,
        line_list=line_list,
        config=config,
        fit_weights=weights,
    )

    assert abs(weighted.species_scales["H2O"] - 1.0) < abs(unweighted.species_scales["H2O"] - 1.0)
    np.testing.assert_allclose(weighted.fit_weights, weights)


def test_fit_ranges_do_not_limit_lines_applied_to_full_correction():
    wavelength = np.linspace(2.30, 2.37, 700)
    line_list = LineList(
        wavelength=np.array([2.315, 2.355]),
        strength=np.array([0.4, 0.3]),
        sigma=np.full(2, 3.0e-5),
        gamma=np.zeros(2),
        species=np.array(["H2O", "H2O"]),
    )
    expected = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.0}),
    )
    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=expected),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fit_ranges=((2.305, 2.325),),
            fixed_species_scales={"H2O": 1.0},
        ),
    )

    second_line = np.abs(wavelength - 2.355) < 2.0e-4
    assert np.nanmin(result.transmission[second_line]) < 0.99
    np.testing.assert_allclose(result.transmission, expected, rtol=0.0, atol=1.0e-12)


def test_species_outside_fit_ranges_is_applied_but_not_fitted():
    wavelength = np.linspace(2.30, 2.37, 700)
    line_list = LineList(
        wavelength=np.array([2.315, 2.355]),
        strength=np.array([0.4, 0.3]),
        sigma=np.full(2, 3.0e-5),
        gamma=np.zeros(2),
        species=np.array(["H2O", "CO2"]),
    )
    expected = transmission_model(wavelength, line_list, ModelConfig())
    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=expected),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fit_ranges=((2.305, 2.325),),
        ),
    )

    assert "log_scale:H2O" in result.parameter_names
    assert "log_scale:CO2" not in result.parameter_names
    assert result.species_scales["CO2"] == 1.0
    assert result.provenance["species_observability"]["automatically_fixed"] == {"CO2": 1.0}
    np.testing.assert_allclose(result.transmission, expected, rtol=0.0, atol=2.0e-9)


def test_fit_tellurics_reorders_fit_mask_with_unsorted_wavelength():
    wavelength = np.linspace(2.31, 2.36, 300)[::-1]
    line_list = demo_lines()
    flux = transmission_model(wavelength, line_list, ModelConfig())
    fit_mask = wavelength < 2.345

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        fit_mask=fit_mask,
    )

    assert np.all(np.diff(result.spectrum.wavelength) > 0)
    np.testing.assert_array_equal(
        result.fit_mask,
        result.spectrum.wavelength < 2.345,
    )


def test_fit_tellurics_estimates_and_propagates_local_uncertainties():
    rng = np.random.default_rng(19)
    wavelength = np.linspace(2.31, 2.36, 500)
    line_list = demo_lines().select_species(("H2O",))
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.4}),
    )
    uncertainty = np.full(wavelength.shape, 0.003)
    flux = transmission + rng.normal(0.0, uncertainty)

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux, uncertainty=uncertainty),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            species=("H2O",),
            estimate_uncertainties=True,
        ),
    )

    assert result.parameter_covariance is not None
    assert result.parameter_covariance.shape == (2, 2)
    assert result.covariance_rank == 2
    assert result.reduced_chi_square > 0
    assert result.species_scale_uncertainties["H2O"] > 0
    assert result.transmission_uncertainty is not None
    assert np.all(np.isfinite(result.transmission_uncertainty))
    assert result.corrected.uncertainty is not None
    valid = np.isfinite(result.corrected.uncertainty)
    assert np.all(
        result.corrected.uncertainty[valid] >= uncertainty[valid] / result.transmission[valid]
    )
    assert "transmission_uncertainty" in result.to_table().colnames
    assert "corrected_uncertainty" in result.to_table().colnames


def test_multi_segment_fit_estimates_and_propagates_shared_uncertainties(tmp_path):
    rng = np.random.default_rng(29)
    wavelength = np.linspace(2.31, 2.36, 350)
    line_list = demo_lines().select_species(("H2O",))
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.35}),
    )
    uncertainty = np.full(wavelength.shape, 0.004)
    spectra = tuple(
        Spectrum(
            wavelength=wavelength,
            flux=continuum * transmission + rng.normal(0.0, uncertainty),
            uncertainty=uncertainty,
        )
        for continuum in (0.9, 1.1)
    )

    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            species=("H2O",),
            continuum_order=0,
            estimate_uncertainties=True,
        ),
    )

    assert result.success
    assert result.parameter_covariance is not None
    assert result.parameter_covariance.shape == (3, 3)
    assert result.covariance_rank == 3
    assert result.species_scale_uncertainties["H2O"] > 0
    for segment in result.segment_results:
        assert segment.transmission_uncertainty is not None
        assert np.all(np.isfinite(segment.transmission_uncertainty))
        assert segment.corrected.uncertainty is not None
        product = tmp_path / f"segment_{len(list(tmp_path.iterdir()))}.ecsv"
        segment.write(product)
        table = Table.read(product)
        assert "input_mask" in table.colnames
        assert "fit_mask" in table.colnames
        assert "corrected_mask" in table.colnames
        assert json.loads(table.meta["provenance_json"])["schema_version"] == 1
        assert table.meta["covariance_full_rank"]
        assert table.meta["wavelength_medium"] == "vacuum"


def test_rank_deficient_covariance_is_not_reported_as_false_precision():
    jacobian = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    covariance, reduced_chi_square, rank = _linearized_parameter_covariance(
        jacobian,
        cost=1.0,
        n_residuals=3,
        n_parameters=2,
    )

    assert rank == 1
    assert reduced_chi_square == 2.0
    assert np.all(np.isnan(covariance))


def test_covariance_rank_is_invariant_to_parameter_units():
    jacobian = np.array(
        [
            [1.0e9, 1.0],
            [2.0e9, -1.0],
            [3.0e9, 0.5],
            [4.0e9, -0.5],
        ]
    )

    covariance, reduced_chi_square, rank = _linearized_parameter_covariance(
        jacobian,
        cost=1.0,
        n_residuals=4,
        n_parameters=2,
    )

    assert rank == 2
    assert reduced_chi_square == 1.0
    assert np.all(np.isfinite(covariance))
    assert covariance[0, 0] < covariance[1, 1]


def test_robust_fit_reports_raw_reduced_chi_square_with_profiled_continuum():
    wavelength = np.linspace(2.31, 2.36, 300)
    line_list = demo_lines().select_species(("H2O",))
    uncertainty = np.full(wavelength.shape, 0.003)
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.3}),
    )
    flux[150] += 0.1

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux, uncertainty=uncertainty),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            solve_continuum_linear=True,
            loss="soft_l1",
        ),
    )

    residual = (result.spectrum.flux - result.model_flux) / uncertainty
    effective_parameter_count = len(result.parameter_names) + 1
    expected = float(np.dot(residual, residual) / (residual.size - effective_parameter_count))
    assert result.reduced_chi_square == pytest.approx(expected)
