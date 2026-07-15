import numpy as np

from pymolfit import (
    AtmosphereLayer,
    AtmosphereProfile,
    LineList,
    ModelConfig,
    average_high_resolution_values,
    high_resolution_wavelength_grid,
    optical_depth_basis,
    prepare_piecewise_constant_rebin,
    prepare_sample_average_rebin,
    radiative_transfer_wavelength_grid,
    rebin_high_resolution_values,
    rebin_piecewise_constant_values,
    sample_high_resolution_values,
    transmission_from_high_resolution_basis,
    transmission_model,
)
from pymolfit.model import convolve_lsf


def test_transmission_model_shape_and_bounds():
    wavelength = np.linspace(2.31, 2.36, 500)
    transmission = transmission_model(
        wavelength,
        LineList.demo_near_ir(),
        ModelConfig(airmass=1.2, species_scales={"H2O": 1.5}),
    )

    assert transmission.shape == wavelength.shape
    assert np.all(np.isfinite(transmission))
    assert np.nanmin(transmission) >= 0
    assert np.nanmax(transmission) <= 1


def test_optical_depth_basis_has_species_rows():
    wavelength = np.linspace(2.31, 2.36, 100)
    species, basis = optical_depth_basis(wavelength, LineList.demo_near_ir())

    assert set(species) == {"CH4", "CO2", "H2O"}
    assert basis.shape == (3, wavelength.size)
    assert np.all(basis >= 0)


def test_composite_lsf_preserves_constant_and_smooths_delta():
    values = np.ones(101)
    convolved = convolve_lsf(values, box_width_pixels=1.2, lorentz_fwhm_pixels=1.5)

    np.testing.assert_allclose(convolved, values)

    delta = np.zeros(101)
    delta[50] = 1.0
    smoothed = convolve_lsf(delta, box_width_pixels=1.2, lorentz_fwhm_pixels=1.5)

    assert smoothed[50] < 1.0
    np.testing.assert_allclose(np.sum(smoothed), 1.0)


def test_variable_lsf_requires_wavelength_and_matches_constant_at_reference():
    values = np.zeros(101)
    values[50] = 1.0
    wavelength = np.full(values.shape, 3.4)

    with np.testing.assert_raises(ValueError):
        convolve_lsf(values, box_width_pixels=1.2, variable_width=True)

    constant = convolve_lsf(
        values,
        box_width_pixels=1.2,
        lorentz_fwhm_pixels=1.5,
        kernel_width_fwhm=3.0,
    )
    variable = convolve_lsf(
        values,
        box_width_pixels=1.2,
        lorentz_fwhm_pixels=1.5,
        wavelength_micron=wavelength,
        variable_width=True,
        reference_wavelength_micron=3.4,
        kernel_width_fwhm=3.0,
    )

    np.testing.assert_allclose(variable, constant)


def test_molecfit_voigt_lsf_mode_is_flux_conserving_and_distinct():
    delta = np.zeros(101)
    delta[50] = 1.0

    separate = convolve_lsf(
        delta,
        gaussian_sigma_pixels=0.8,
        lorentz_fwhm_pixels=0.7,
        kernel_width_fwhm=5.0,
    )
    molecfit_voigt = convolve_lsf(
        delta,
        gaussian_sigma_pixels=0.8,
        lorentz_fwhm_pixels=0.7,
        kernel_width_fwhm=5.0,
        molecfit_voigt=True,
    )

    np.testing.assert_allclose(np.sum(molecfit_voigt), 1.0)
    assert molecfit_voigt[50] < 1.0
    assert np.nanmax(np.abs(molecfit_voigt - separate)) > 1.0e-4


def test_high_resolution_rebin_preserves_smooth_values():
    observed = np.linspace(2.30, 2.32, 200)
    highres, pixels_per_observed = high_resolution_wavelength_grid(observed, oversampling=6.0)
    values = 0.9 + 0.02 * np.sin((highres - np.mean(highres)) / np.ptp(highres) * 2.0 * np.pi)

    rebinned = rebin_high_resolution_values(observed, highres, values)

    assert pixels_per_observed == 6.0
    assert rebinned.shape == observed.shape
    np.testing.assert_allclose(np.nanmedian(rebinned), np.nanmedian(values), atol=5.0e-4)


def test_high_resolution_center_sampling_matches_interpolation():
    observed = np.array([2.301, 2.305, 2.309])
    highres = np.array([2.300, 2.304, 2.308, 2.312])
    values = np.array([0.8, 0.9, 0.85, 0.95])

    sampled = sample_high_resolution_values(observed, highres, values)

    np.testing.assert_allclose(sampled, np.interp(observed, highres, values))


def test_high_resolution_sample_average_uses_bin_samples():
    observed = np.array([1.0, 2.0, 3.0])
    highres = np.array([0.75, 1.05, 1.40, 1.75, 2.25, 2.60, 3.20])
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])

    averaged = average_high_resolution_values(observed, highres, values)

    np.testing.assert_allclose(averaged, [7.0 / 3.0, 12.0, 48.0])


def test_molecfit_overlap_rebin_matches_source_bin_weighting():
    output = np.array([1.0, 2.0, 3.0])
    input_wavelength = np.array([0.75, 1.25, 2.25, 3.25])
    values = np.array([1.0, 3.0, 5.0, 7.0])

    rebinned = rebin_piecewise_constant_values(output, input_wavelength, values)

    # Output bins are [0.5,1.5], [1.5,2.5], [2.5,3.5]. Input bins are
    # [0.5,1.0], [1.0,1.75], [1.75,2.75], [2.75,3.75].
    np.testing.assert_allclose(rebinned, [2.0, 4.5, 6.5])


def test_precomputed_overlap_rebin_matches_direct_for_repeated_values():
    rng = np.random.default_rng(123)
    output = np.linspace(1.0, 3.0, 80) ** 1.01
    input_wavelength = np.linspace(0.95, 3.05, 503) ** 1.01
    plan = prepare_piecewise_constant_rebin(output, input_wavelength)

    for _ in range(4):
        values = rng.uniform(0.0, 1.0, input_wavelength.size)
        expected = rebin_piecewise_constant_values(output, input_wavelength, values)
        actual = plan.apply(values)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_precomputed_sample_average_matches_direct_for_repeated_values():
    rng = np.random.default_rng(321)
    model = np.linspace(1.0, 3.0, 80) ** 1.01
    native = np.linspace(0.95, 3.05, 5003) ** 1.01
    plan = prepare_sample_average_rebin(model, native)

    for _ in range(4):
        values = rng.uniform(0.0, 1.0, native.size)
        expected = average_high_resolution_values(model, native, values)
        actual = plan.apply(values)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_radiative_transfer_grid_uses_lblrtm_layer_spacing_and_is_bounded():
    model = np.linspace(0.758, 0.770, 700)
    atmosphere = AtmosphereProfile(
        (
            AtmosphereLayer(0.75, 285.0, 500.0, {"H2O": 2.0e-3, "O2": 0.2095}),
            AtmosphereLayer(0.01, 220.0, 2_000.0, {"H2O": 1.0e-5, "O2": 0.2095}),
        )
    )

    native, step = radiative_transfer_wavelength_grid(model, atmosphere)
    native_wavenumber = 1.0e4 / native

    assert np.all(np.diff(native) > 0)
    np.testing.assert_allclose(np.diff(native_wavenumber), -step, rtol=0.0, atol=2.0e-9)
    assert step < np.min(np.abs(np.diff(1.0e4 / model)))
    assert native.size < 2_000_000


def test_native_transmission_sampling_resolves_saturated_subpixel_line():
    observed = np.linspace(0.7590, 0.7610, 41)
    model, pixels_per_observed = high_resolution_wavelength_grid(observed, oversampling=5.0)
    native, _ = high_resolution_wavelength_grid(observed, oversampling=500.0)
    reference_grid, _ = high_resolution_wavelength_grid(observed, oversampling=2000.0)
    model_wavenumber = 1.0e4 / model
    model_step = float(np.median(np.abs(np.diff(model_wavenumber))))
    center = 0.5 * (model_wavenumber[model.size // 2] + model_wavenumber[model.size // 2 + 1])
    sigma = 0.035 * model_step

    def basis(wavelength):
        wavenumber = 1.0e4 / wavelength
        return (30.0 * np.exp(-0.5 * ((wavenumber - center) / sigma) ** 2))[None, :]

    detector_plan = prepare_piecewise_constant_rebin(observed, model)
    native_model = transmission_from_high_resolution_basis(
        observed,
        native,
        ("O2",),
        basis(native),
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="molecfit_overlap",
        rebin_plan=detector_plan,
        model_wavelength_micron=model,
        native_to_model_plan=prepare_sample_average_rebin(model, native),
    )
    reference = transmission_from_high_resolution_basis(
        observed,
        reference_grid,
        ("O2",),
        basis(reference_grid),
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="molecfit_overlap",
        rebin_plan=detector_plan,
        model_wavelength_micron=model,
        native_to_model_plan=prepare_sample_average_rebin(model, reference_grid),
    )
    coarse = transmission_from_high_resolution_basis(
        observed,
        model,
        ("O2",),
        basis(model),
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="molecfit_overlap",
        rebin_plan=detector_plan,
    )

    native_error = float(np.sqrt(np.mean((native_model - reference) ** 2)))
    coarse_error = float(np.sqrt(np.mean((coarse - reference) ** 2)))
    assert native_error < 0.08 * coarse_error


def test_native_two_stage_path_preserves_weak_resolved_line():
    observed = np.linspace(2.30, 2.32, 120)
    model, pixels_per_observed = high_resolution_wavelength_grid(observed, oversampling=5.0)
    native, _ = high_resolution_wavelength_grid(observed, oversampling=80.0)

    def basis(wavelength):
        tau = 0.02 * np.exp(-0.5 * ((wavelength - 2.31) / 4.0e-4) ** 2)
        return tau[None, :]

    detector_plan = prepare_piecewise_constant_rebin(observed, model)
    two_stage = transmission_from_high_resolution_basis(
        observed,
        native,
        ("H2O",),
        basis(native),
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="molecfit_overlap",
        rebin_plan=detector_plan,
        model_wavelength_micron=model,
        native_to_model_plan=prepare_sample_average_rebin(model, native),
    )
    model_only = transmission_from_high_resolution_basis(
        observed,
        model,
        ("H2O",),
        basis(model),
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="molecfit_overlap",
        rebin_plan=detector_plan,
    )

    np.testing.assert_allclose(two_stage, model_only, atol=1.0e-4)


def test_molecfit_high_resolution_mode_rebins_before_observed_pixel_lsf():
    observed = np.linspace(2.30, 2.32, 31)
    highres, pixels_per_observed = high_resolution_wavelength_grid(observed, oversampling=5.0)
    optical_depth = np.zeros((1, highres.size), dtype=float)
    optical_depth[0, highres.size // 2] = 2.0

    actual = transmission_from_high_resolution_basis(
        observed,
        highres,
        ("H2O",),
        optical_depth,
        highres_pixels_per_observed_pixel=pixels_per_observed,
        lsf_sigma_pixels=1.1,
        rebin_mode="molecfit_overlap",
    )

    raw = np.exp(-optical_depth[0])
    expected = convolve_lsf(
        rebin_piecewise_constant_values(observed, highres, raw),
        gaussian_sigma_pixels=1.1,
    )
    np.testing.assert_allclose(actual, expected)


def test_transmission_from_high_resolution_basis_matches_direct_for_resolved_line():
    observed = np.linspace(2.30, 2.32, 500)
    highres, pixels_per_observed = high_resolution_wavelength_grid(observed, oversampling=5.0)
    line_list = LineList(
        wavelength=np.array([2.31]),
        strength=np.array([8.0e-4]),
        sigma=np.array([1.5e-4]),
        gamma=np.array([4.0e-5]),
        species=np.array(["H2O"]),
    )
    species, basis_high = optical_depth_basis(highres, line_list)
    highres_transmission = transmission_from_high_resolution_basis(
        observed,
        highres,
        species,
        basis_high,
        species_scales={"H2O": 1.2},
        highres_pixels_per_observed_pixel=pixels_per_observed,
    )
    direct = transmission_model(
        observed,
        line_list,
        ModelConfig(species_scales={"H2O": 1.2}),
    )

    np.testing.assert_allclose(highres_transmission, direct, atol=8.0e-3)


def test_high_resolution_transmission_accepts_center_rebin_mode():
    observed = np.linspace(2.30, 2.32, 80)
    highres, pixels_per_observed = high_resolution_wavelength_grid(observed, oversampling=5.0)
    line_list = LineList(
        wavelength=np.array([2.31]),
        strength=np.array([1.0e-3]),
        sigma=np.array([1.0e-4]),
        gamma=np.array([2.0e-5]),
        species=np.array(["H2O"]),
    )
    species, basis_high = optical_depth_basis(highres, line_list)

    sampled = transmission_from_high_resolution_basis(
        observed,
        highres,
        species,
        basis_high,
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="center",
    )
    integrated = transmission_from_high_resolution_basis(
        observed,
        highres,
        species,
        basis_high,
        highres_pixels_per_observed_pixel=pixels_per_observed,
        rebin_mode="integrate",
    )

    assert sampled.shape == observed.shape
    assert integrated.shape == observed.shape
    assert np.nanmax(np.abs(sampled - integrated)) > 0.0


def test_transmission_model_high_resolution_mode_runs():
    wavelength = np.linspace(2.31, 2.36, 120)
    line_list = LineList.demo_near_ir()

    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(
            high_resolution_grid=True,
            high_resolution_oversampling=4.0,
            high_resolution_rebin_mode="center",
        ),
    )

    assert transmission.shape == wavelength.shape
    assert np.nanmin(transmission) >= 0.0
    assert np.nanmax(transmission) <= 1.0
