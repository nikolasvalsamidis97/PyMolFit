import numpy as np
import pytest
from astropy.table import Table

from genmolfit import (
    FitConfig,
    LineList,
    ModelConfig,
    Spectrum,
    fit_telluric_segments,
    fit_tellurics,
    transmission_model,
)
from genmolfit.diagnostics import correction_summary, residual_by_window
from genmolfit.atmosphere import AtmosphereProfile
from genmolfit.fit import _shift_basis
from genmolfit.model import optical_depth_basis, transmission_from_basis


def test_optimizer_tolerances_match_molecfit_defaults():
    config = FitConfig()
    assert config.ftol == 1.0e-10
    assert config.xtol == 1.0e-10
    assert config.gtol == 1.0e-10


def _shifted_demo_line_list(shift_micron):
    base = LineList.demo_near_ir()
    return LineList(
        wavelength=base.wavelength + shift_micron,
        strength=base.strength,
        sigma=base.sigma,
        gamma=base.gamma,
        species=base.species,
    )


def test_standard_midlatitude_profile_has_columns():
    profile = AtmosphereProfile.standard_midlatitude(airmass=1.3, n_layers=12)

    assert len(profile.layers) == 12
    assert profile.total_column_cm2("H2O") > 0
    assert profile.total_column_cm2("CO2") > 0
    assert profile.layers[0].pressure_atm > profile.layers[-1].pressure_atm


def test_pwv_scaling_sets_water_column():
    profile = AtmosphereProfile.standard_midlatitude(n_layers=8).with_pwv_mm(2.0)

    np.testing.assert_allclose(profile.total_column_cm2("H2O"), 2.0 * 3.34556e21)
    np.testing.assert_allclose(profile.total_vertical_column_cm2("H2O"), 2.0 * 3.34556e21)


def test_fit_recovers_wavelength_shift():
    wavelength = np.linspace(2.31, 2.36, 1000)
    true_shift = 1.2e-4
    shifted_lines = _shifted_demo_line_list(true_shift)
    flux = 1.05 * transmission_model(
        wavelength,
        shifted_lines,
        ModelConfig(species_scales={"H2O": 1.8, "CO2": 0.9, "CH4": 1.1}),
    )

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=LineList.demo_near_ir(),
        config=FitConfig(
            continuum_order=0,
            fit_wavelength_shift=True,
            wavelength_shift_bounds=(-3.0e-4, 3.0e-4),
        ),
    )

    assert result.success
    assert abs(result.wavelength_shift - true_shift) < 4.0e-5


def test_fit_lsf_sigma_and_writes_product_table(tmp_path):
    wavelength = np.linspace(2.32, 2.34, 2000)
    line_list = LineList(
        wavelength=np.array([2.326, 2.331, 2.336]),
        strength=np.array([0.004, 0.005, 0.004]),
        sigma=np.full(3, 2.0e-5),
        gamma=np.full(3, 1.0e-5),
        species=np.array(["H2O", "H2O", "H2O"]),
    )
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.0}, lsf_sigma_pixels=2.0),
    )

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(continuum_order=0, fit_lsf_sigma=True, lsf_sigma_bounds=(0.0, 5.0)),
    )
    out = tmp_path / "fit.ecsv"
    result.write(out)
    table = Table.read(out)

    assert result.success
    assert abs(result.lsf_sigma_pixels - 2.0) < 0.2
    assert "corrected_flux" in table.colnames
    assert result.metrics["rms_residual"] >= 0


def test_fit_lsf_box_width():
    wavelength = np.linspace(2.32, 2.34, 2000)
    line_list = LineList(
        wavelength=np.array([2.326, 2.331, 2.336]),
        strength=np.array([0.02, 0.024, 0.022]),
        sigma=np.full(3, 1.0e-5),
        gamma=np.full(3, 5.0e-6),
        species=np.array(["H2O", "H2O", "H2O"]),
    )
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.0}, lsf_box_width_pixels=1.8),
    )

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fixed_species_scales={"H2O": 1.0},
            lsf_box_width_pixels=1.2,
            fit_lsf_box_width=True,
            lsf_box_width_bounds=(1.0, 3.0),
        ),
    )

    assert result.success
    assert abs(result.lsf_box_width_pixels - 1.8) < 0.2


def test_fit_lsf_lorentz_fwhm():
    wavelength = np.linspace(2.32, 2.34, 2000)
    line_list = LineList(
        wavelength=np.array([2.326, 2.331, 2.336]),
        strength=np.array([0.01, 0.012, 0.011]),
        sigma=np.full(3, 1.5e-5),
        gamma=np.full(3, 8.0e-6),
        species=np.array(["H2O", "H2O", "H2O"]),
    )
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.0}, lsf_lorentz_fwhm_pixels=1.1),
    )

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fixed_species_scales={"H2O": 1.0},
            lsf_lorentz_fwhm_pixels=0.6,
            fit_lsf_lorentz_fwhm=True,
            lsf_lorentz_fwhm_bounds=(0.0, 3.0),
        ),
    )

    assert result.success
    assert abs(result.lsf_lorentz_fwhm_pixels - 1.1) < 0.2


def test_fit_respects_fixed_species_scale():
    wavelength = np.linspace(2.32, 2.34, 800)
    line_list = LineList(
        wavelength=np.array([2.331]),
        strength=np.array([0.006]),
        sigma=np.array([2.0e-5]),
        gamma=np.array([1.0e-5]),
        species=np.array(["H2O"]),
    )
    flux = 1.2 * transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 2.5}),
    )

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(continuum_order=0, fixed_species_scales={"H2O": 2.5}),
    )

    assert result.success
    assert result.species_scales["H2O"] == 2.5


def test_fit_telluric_segments_shares_species_scale_across_segments():
    line_list = LineList(
        wavelength=np.array([2.331, 2.351]),
        strength=np.array([0.006, 0.004]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "H2O"]),
    )
    spectra = []
    for center, continuum in [(2.331, 1.1), (2.351, 2.3)]:
        wavelength = np.linspace(center - 0.004, center + 0.004, 500)
        flux = continuum * transmission_model(
            wavelength,
            line_list,
            ModelConfig(species_scales={"H2O": 1.8}),
        )
        spectra.append(Spectrum(wavelength=wavelength, flux=flux))

    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(continuum_order=0),
    )

    assert result.success
    assert len(result.segment_results) == 2
    assert abs(result.species_scales["H2O"] - 1.8) < 0.2


def test_segment_jacobian_sparsity_preserves_multi_segment_solution():
    line_list = LineList(
        wavelength=np.array([2.331, 2.351]),
        strength=np.array([0.006, 0.004]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "H2O"]),
    )
    spectra = []
    for center, continuum in [(2.331, 0.9), (2.351, 1.1), (2.341, 1.3)]:
        wavelength = np.linspace(center - 0.004, center + 0.004, 220)
        flux = continuum * transmission_model(
            wavelength,
            line_list,
            ModelConfig(species_scales={"H2O": 1.4}),
        )
        spectra.append(Spectrum(wavelength=wavelength, flux=flux))

    dense = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(continuum_order=1, use_jacobian_sparsity=False),
    )
    sparse = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(continuum_order=1, use_jacobian_sparsity=True),
    )

    assert dense.success and sparse.success
    np.testing.assert_allclose(
        [sparse.species_scales[name] for name in sorted(sparse.species_scales)],
        [dense.species_scales[name] for name in sorted(dense.species_scales)],
        rtol=2.0e-5,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(sparse.cost, dense.cost, rtol=2.0e-8, atol=1.0e-10)


def test_fit_telluric_segments_accepts_continuum_priors():
    line_list = LineList(
        wavelength=np.array([2.331]),
        strength=np.array([0.006]),
        sigma=np.array([2.0e-5]),
        gamma=np.array([1.0e-5]),
        species=np.array(["H2O"]),
    )
    wavelength = np.linspace(2.327, 2.335, 500)
    prior = 1.2 + 0.05 * (wavelength - np.nanmean(wavelength)) / np.ptp(wavelength)
    flux = prior * transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.0}),
    )

    result = fit_telluric_segments(
        [Spectrum(wavelength=wavelength, flux=flux)],
        line_list=line_list,
        config=FitConfig(
            continuum_order=1,
            continuum_prior_weight=1.0,
            continuum_prior_fractional_sigma=0.02,
        ),
        continuum_priors=[prior],
    )

    np.testing.assert_allclose(result.segment_results[0].continuum, prior, rtol=0.03)


def test_fit_telluric_segments_can_solve_continuum_linearly():
    line_list = LineList(
        wavelength=np.array([2.331, 2.351]),
        strength=np.array([0.006, 0.004]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "H2O"]),
    )
    spectra = []
    for center, continuum_scale in [(2.331, 1.2), (2.351, 2.1)]:
        wavelength = np.linspace(center - 0.004, center + 0.004, 500)
        continuum = continuum_scale + 0.03 * (wavelength - np.nanmean(wavelength)) / np.ptp(wavelength)
        flux = continuum * transmission_model(
            wavelength,
            line_list,
            ModelConfig(species_scales={"H2O": 1.7}),
        )
        spectra.append(Spectrum(wavelength=wavelength, flux=flux, uncertainty=np.full_like(flux, 0.01)))

    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            continuum_order=1,
            solve_continuum_linear=True,
            loss="linear",
        ),
    )

    assert result.success
    assert abs(result.species_scales["H2O"] - 1.7) < 0.05


def test_fit_telluric_segments_can_fit_segment_wavelength_shifts():
    line_list = LineList(
        wavelength=np.array([2.331, 2.351]),
        strength=np.array([0.006, 0.004]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "H2O"]),
    )
    shifts = [8.0e-5, -7.0e-5]
    spectra = []
    for center, shift in zip([2.331, 2.351], shifts, strict=True):
        wavelength = np.linspace(center - 0.004, center + 0.004, 500)
        shifted_lines = LineList(
            wavelength=line_list.wavelength + shift,
            strength=line_list.strength,
            sigma=line_list.sigma,
            gamma=line_list.gamma,
            species=line_list.species,
        )
        flux = 1.2 * transmission_model(
            wavelength,
            shifted_lines,
            ModelConfig(species_scales={"H2O": 1.4}),
        )
        spectra.append(Spectrum(wavelength=wavelength, flux=flux))

    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fit_segment_wavelength_shifts=True,
            wavelength_shift_bounds=(-2.0e-4, 2.0e-4),
        ),
    )

    assert result.success
    recovered = [segment.wavelength_shift for segment in result.segment_results]
    np.testing.assert_allclose(recovered, shifts, atol=3.0e-5)


def test_fit_telluric_segments_can_fit_segment_wavelength_polynomial():
    line_list = LineList(
        wavelength=np.array([2.326, 2.331, 2.336]),
        strength=np.array([0.004, 0.006, 0.004]),
        sigma=np.full(3, 2.0e-5),
        gamma=np.full(3, 1.0e-5),
        species=np.array(["H2O", "H2O", "H2O"]),
    )
    wavelength = np.linspace(2.322, 2.340, 900)
    x = 2.0 * (wavelength - 0.5 * (wavelength[0] + wavelength[-1])) / (wavelength[-1] - wavelength[0])
    true_coefficients = np.array([2.0e-5, 8.0e-5])
    true_shift = true_coefficients[0] + true_coefficients[1] * x
    species_names, basis = optical_depth_basis(wavelength, line_list)
    flux = 1.15 * transmission_from_basis(
        species_names,
        _shift_basis(wavelength, basis, true_shift),
        species_scales={"H2O": 1.3},
    )

    result = fit_telluric_segments(
        [Spectrum(wavelength=wavelength, flux=flux)],
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fit_segment_wavelength_polynomial=True,
            segment_wavelength_polynomial_order=1,
            wavelength_shift_bounds=(-2.0e-4, 2.0e-4),
        ),
    )

    assert result.success
    np.testing.assert_allclose(
        result.segment_results[0].wavelength_coefficients,
        true_coefficients,
        atol=2.0e-5,
    )


def test_fit_telluric_segments_can_fit_shared_wavelength_polynomial():
    line_list = LineList(
        wavelength=np.array([2.302, 2.308, 2.312, 2.388, 2.392, 2.398]),
        strength=np.array([0.004, 0.007, 0.005, 0.006, 0.004, 0.007]),
        sigma=np.full(6, 1.5e-5),
        gamma=np.full(6, 8.0e-6),
        species=np.full(6, "H2O"),
    )
    spectra = []
    true_coefficients = np.array([1.5e-5, 5.0e-5])
    global_bounds = (2.298, 2.402)
    for lower, upper in ((2.298, 2.316), (2.384, 2.402)):
        wavelength = np.linspace(lower, upper, 700)
        x = 2.0 * (wavelength - np.mean(global_bounds)) / np.ptp(global_bounds)
        shift = true_coefficients[0] + true_coefficients[1] * x
        species_names, basis = optical_depth_basis(wavelength, line_list)
        flux = 1.1 * transmission_from_basis(
            species_names,
            _shift_basis(wavelength, basis, shift),
            species_scales={"H2O": 1.2},
        )
        spectra.append(Spectrum(wavelength=wavelength, flux=flux))

    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fit_wavelength_polynomial=True,
            wavelength_polynomial_order=1,
            wavelength_shift_bounds=(-1.0e-4, 1.0e-4),
        ),
    )

    assert result.success
    assert result.parameter_names.count("wavelength_coefficient:0") == 1
    assert result.parameter_names.count("wavelength_coefficient:1") == 1
    for segment in result.segment_results:
        np.testing.assert_allclose(
            segment.wavelength_coefficients,
            true_coefficients,
            atol=7.0e-6,
        )


def test_global_and_segment_wavelength_polynomials_are_mutually_exclusive():
    wavelength = np.linspace(2.31, 2.36, 300)
    line_list = LineList.demo_near_ir()

    with pytest.raises(ValueError, match="cannot be combined"):
        fit_telluric_segments(
            [Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))],
            line_list=line_list,
            config=FitConfig(
                fit_wavelength_polynomial=True,
                fit_segment_wavelength_polynomial=True,
            ),
        )


def test_fit_ranges_limit_model_fit():
    wavelength = np.linspace(2.31, 2.36, 700)
    line_list = LineList.demo_near_ir()
    flux = transmission_model(wavelength, line_list, ModelConfig(species_scales={"H2O": 1.5}))

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            fit_ranges=((2.317, 2.322), (2.330, 2.333)),
            exclude_ranges=((2.3205, 2.3215),),
        ),
    )

    assert result.success
    assert result.transmission.shape == wavelength.shape


def test_diagnostics_helpers_return_summary():
    wavelength = np.linspace(2.31, 2.36, 500)
    line_list = LineList.demo_near_ir()
    flux = transmission_model(wavelength, line_list, ModelConfig(species_scales={"H2O": 1.5}))
    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(continuum_order=0),
    )

    summary = correction_summary(result)
    windows = residual_by_window(result, ((2.318, 2.322),))

    assert 0 < summary["n_pixels"] <= wavelength.size
    assert 0 <= summary["median_transmission"] <= 1
    assert windows[0]["n_pixels"] > 0
