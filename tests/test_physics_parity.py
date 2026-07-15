import numpy as np
from scipy.integrate import trapezoid

from pymolfit.components import (
    _lblrtm_self_mixture_corrected_air_width,
    _lblrtm_self_mixture_corrected_pressure_shift,
)
from pymolfit.continuum import MTCKDH2OContinuum, radiation_term_cm, radiation_term_interval_cm
from pymolfit.linelist import LBLRTM_BROADENER_SPECIES
from pymolfit.model import (
    _fractional_box_kernel,
    _integrated_gaussian_kernel,
    _integrated_lorentz_kernel,
    _molecfit_voigt_approx_kernel,
)
from pymolfit.physics import (
    AMU_KG,
    BOLTZMANN_J_PER_K,
    LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
    SPEED_OF_LIGHT_M_PER_S,
    doppler_sigma_wavenumber,
    lblrtm_panel_accumulate_wavenumber,
    lblrtm_panel_interpolate_r1_r2_r3,
    lblrtm_panel_voigt_profile_wavenumber,
    lblrtm_tabulated_voigt_profile_offset,
    lblrtm_temperature_scaling_lower_energy,
    lblrtm_voigt_hwhm,
    line_strength_temperature,
    lorentz_hwhm_wavenumber,
    pressure_shift_wavenumber,
    voigt_profile_wavenumber,
)


def test_radiation_term_matches_lblrtm_radfn_formula():
    temperature = 296.0
    x = np.array([0.005, 1.0, 11.0])
    wavenumber = x * temperature / LBLRTM_SECOND_RADIATION_CONSTANT_CM_K

    term = radiation_term_cm(
        wavenumber,
        temperature,
        second_radiation_constant_cm_k=LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
    )

    expected = wavenumber.copy()
    expected[0] = 0.5 * x[0] * wavenumber[0]
    expected[1] = wavenumber[1] * (1.0 - np.exp(-x[1])) / (1.0 + np.exp(-x[1]))
    np.testing.assert_allclose(term, expected, rtol=1.0e-14)


def test_radiation_term_interval_matches_lblrtm_radfni_bookkeeping():
    temperature = 296.0
    dvi = 0.002
    for x in (0.005, 1.0, 11.0):
        vi = x * temperature / LBLRTM_SECOND_RADIATION_CONSTANT_CM_K
        radfni, vinew, rdel, rdlast, intervals = radiation_term_interval_cm(
            vi,
            dvi,
            temperature,
            second_radiation_constant_cm_k=LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
        )
        expected_start = radiation_term_cm(
            np.array([vi]),
            temperature,
            second_radiation_constant_cm_k=LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
        )[0]
        expected_end = radiation_term_cm(
            np.array([vinew]),
            temperature,
            second_radiation_constant_cm_k=LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
        )[0]

        np.testing.assert_allclose(radfni, expected_start, rtol=1.0e-14)
        np.testing.assert_allclose(rdlast, expected_end, rtol=1.0e-14)
        np.testing.assert_allclose(vinew, vi + intervals * dvi, rtol=1.0e-14)
        np.testing.assert_allclose(rdel, (expected_end - expected_start) / intervals, rtol=1.0e-14)


def test_radiation_term_interval_accepts_existing_rdlast_and_explicit_next_frequency():
    radfni, vinew, rdel, rdlast, intervals = radiation_term_interval_cm(
        1000.0,
        0.01,
        296.0,
        vinew_cm=-1001.23,
        rdlast_cm=12.5,
        second_radiation_constant_cm_k=LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
    )
    expected_end = radiation_term_cm(
        np.array([1001.23]),
        296.0,
        second_radiation_constant_cm_k=LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
    )[0]

    assert radfni == 12.5
    assert vinew == 1001.23
    assert intervals == int((1001.23 - 1000.0) / 0.01)
    np.testing.assert_allclose(rdlast, expected_end, rtol=1.0e-14)
    np.testing.assert_allclose(rdel, (expected_end - 12.5) / intervals, rtol=1.0e-14)


def test_hitran_line_strength_temperature_formula_is_explicit():
    strength_ref = np.array([2.0e-23])
    wavenumber = np.array([4320.0])
    lower_energy = np.array([100.0])
    temperature = 280.0
    reference_temperature = 296.0
    partition_ratio = np.array([174.58 / 150.0])
    c2 = LBLRTM_SECOND_RADIATION_CONSTANT_CM_K

    scaled = line_strength_temperature(
        strength_ref,
        wavenumber,
        lower_energy,
        temperature,
        reference_temperature_k=reference_temperature,
        partition_ratio=partition_ratio,
        second_radiation_constant_cm_k=c2,
    )

    expected = (
        strength_ref
        * partition_ratio
        * np.exp(-c2 * lower_energy * (1.0 / temperature - 1.0 / reference_temperature))
        * (1.0 - np.exp(-c2 * wavenumber / temperature))
        / (1.0 - np.exp(-c2 * wavenumber / reference_temperature))
    )
    np.testing.assert_allclose(scaled, expected, rtol=1.0e-14)


def test_lblrtm_lower_state_energy_unknown_epp_convention():
    lower_energy = np.array([-2.0, -1.0, -0.5, 30.0])

    effective, unknown = lblrtm_temperature_scaling_lower_energy(lower_energy)

    np.testing.assert_allclose(effective, np.array([2.0, -1.0, -0.5, 30.0]))
    np.testing.assert_array_equal(unknown, np.array([False, True, False, False]))


def test_lorentz_width_uses_hitran_temperature_exponent_convention():
    air_width = np.array([0.07, 0.05])
    self_width = np.array([0.30, 0.20])
    exponent = np.array([0.75, 0.50])
    pressure = 0.8
    temperature = 270.0
    absorber_fraction = np.array([0.01, 0.2])

    width = lorentz_hwhm_wavenumber(
        air_width,
        self_width,
        exponent,
        pressure,
        temperature,
        absorber_fraction=absorber_fraction,
        reference_temperature_k=296.0,
    )

    expected = (
        ((1.0 - absorber_fraction) * air_width + absorber_fraction * self_width)
        * pressure
        * (296.0 / temperature) ** exponent
    )
    np.testing.assert_allclose(width, expected, rtol=1.0e-14)


def test_lblrtm_o2_n2_air_width_self_mixture_correction():
    air_width = np.array([0.07, 0.08, 0.09])
    self_width = np.array([0.12, 0.20, 0.11])
    mol_id = np.array([7, 22, 1])

    corrected = _lblrtm_self_mixture_corrected_air_width(air_width, self_width, mol_id)

    expected = air_width.copy()
    expected[0] = (air_width[0] - 0.21 * self_width[0]) / (1.0 - 0.21)
    expected[1] = (air_width[1] - 0.79 * self_width[1]) / (1.0 - 0.79)
    expected[1] = max(expected[1], 0.0)
    np.testing.assert_allclose(corrected, expected, rtol=1.0e-14)


def test_lblrtm_o2_air_shift_self_mixture_correction():
    pressure_shift = np.array([-0.001, -0.002])
    mol_id = np.array([7, 1])
    flags = np.zeros((2, len(LBLRTM_BROADENER_SPECIES)), dtype=int)
    shifts = np.zeros_like(flags, dtype=float)
    o2_index = LBLRTM_BROADENER_SPECIES.index("O2")
    flags[0, o2_index] = 1
    shifts[0, o2_index] = -0.004

    corrected = _lblrtm_self_mixture_corrected_pressure_shift(pressure_shift, flags, shifts, mol_id)

    expected = pressure_shift.copy()
    expected[0] = (pressure_shift[0] - 0.21 * shifts[0, o2_index]) / (1.0 - 0.21)
    np.testing.assert_allclose(corrected, expected, rtol=1.0e-14)


def test_pressure_shift_supports_hitran_and_lblrtm_density_conventions():
    pressure_shift = np.array([-0.001, 0.002])
    pressure = 0.75
    temperature = 260.0

    hitran = pressure_shift_wavenumber(
        pressure_shift,
        pressure,
        temperature,
        reference_temperature_k=296.0,
        convention="hitran",
    )
    lblrtm = pressure_shift_wavenumber(
        pressure_shift,
        pressure,
        temperature,
        reference_temperature_k=296.0,
        convention="lblrtm_density",
    )

    np.testing.assert_allclose(hitran, pressure_shift * pressure)
    np.testing.assert_allclose(lblrtm, pressure_shift * pressure * 296.0 / temperature)


def test_doppler_sigma_matches_standard_thermal_formula():
    wavenumber = np.array([3000.0, 5000.0])
    temperature = 285.0
    mass_amu = np.array([18.010565, 31.989829])

    sigma = doppler_sigma_wavenumber(wavenumber, temperature, mass_amu)

    mass_kg = mass_amu * AMU_KG
    expected = wavenumber * np.sqrt(
        BOLTZMANN_J_PER_K * temperature / (mass_kg * SPEED_OF_LIGHT_M_PER_S**2)
    )
    np.testing.assert_allclose(sigma, expected, rtol=1.0e-14)


def test_voigt_profile_integrates_to_nearly_one_for_intermediate_grid():
    grid = np.linspace(999.0, 1001.0, 20001)
    profile = voigt_profile_wavenumber(
        grid,
        centers_cm=np.array([1000.0]),
        sigma_cm=np.array([0.02]),
        gamma_cm=np.array([0.0]),
    )[0]

    integral = trapezoid(profile, grid)
    np.testing.assert_allclose(integral, 1.0, rtol=2.0e-4)


def test_lblrtm_tabulated_voigt_profile_is_finite_and_cut_off():
    grid = np.linspace(-5.0, 5.0, 401)
    gamma = np.array([0.03])
    sigma = np.array([0.02])

    profile = lblrtm_tabulated_voigt_profile_offset(grid[None, :], gamma, sigma)[0]
    cutoff = 64.0 * lblrtm_voigt_hwhm(gamma, sigma)[0]

    assert np.all(np.isfinite(profile))
    assert np.all(profile >= 0.0)
    np.testing.assert_allclose(profile, profile[::-1], rtol=1.0e-12, atol=2.0e-13)
    assert profile[grid.size // 2] == np.max(profile)
    assert np.all(profile[np.abs(grid) >= cutoff] == 0.0)


def test_lblrtm_tabulated_voigt_profile_tracks_exact_profile_near_center():
    grid = np.linspace(-0.05, 0.05, 101)
    gamma = np.array([0.01])
    sigma = np.array([0.02])

    tabulated = lblrtm_tabulated_voigt_profile_offset(grid[None, :], gamma, sigma)[0]
    exact = voigt_profile_wavenumber(
        grid,
        centers_cm=np.array([0.0]),
        sigma_cm=sigma,
        gamma_cm=gamma,
    )[0]

    np.testing.assert_allclose(tabulated, exact, rtol=0.03, atol=0.0)


def test_lblrtm_panel_interpolation_matches_source_coefficients():
    r1 = np.zeros((1, 12))
    r2 = np.zeros((1, 3))
    r3 = np.zeros((1, 1))
    r2[0, 1] = 1.0

    combined = lblrtm_panel_interpolate_r1_r2_r3(r1, r2, r3)[0]

    expected = np.zeros(12)
    expected[1] = 35.0 / 128.0
    expected[2] = 9.0 / 16.0
    expected[3] = 105.0 / 128.0
    expected[4] = 1.0
    expected[5] = 105.0 / 128.0
    expected[6] = 9.0 / 16.0
    expected[7] = 35.0 / 128.0
    expected[9] = -7.0 / 128.0
    expected[10] = -1.0 / 16.0
    expected[11] = -5.0 / 128.0
    np.testing.assert_allclose(combined, expected, rtol=1.0e-14, atol=1.0e-14)


def test_lblrtm_panel_voigt_profile_is_finite_and_tracks_table_mode():
    grid = np.linspace(-0.2, 0.2, 401)
    gamma = np.array([0.01])
    sigma = np.array([0.02])

    panel = lblrtm_panel_voigt_profile_wavenumber(grid, np.array([0.0]), sigma, gamma)[0]
    table = lblrtm_tabulated_voigt_profile_offset(grid[None, :], gamma, sigma)[0]

    assert np.all(np.isfinite(panel))
    assert np.all(panel >= 0.0)
    np.testing.assert_allclose(panel, panel[::-1], rtol=0.02, atol=2.0e-12)
    np.testing.assert_allclose(panel, table, rtol=0.04, atol=2.0e-10)


def test_lblrtm_panel_voigt_profile_accepts_single_pixel_grid():
    grid = np.array([1000.0])
    centers = np.array([999.95, 1000.05])
    gamma = np.array([0.01, 0.02])
    sigma = np.array([0.02, 0.02])

    profile = lblrtm_panel_voigt_profile_wavenumber(grid, centers, sigma, gamma)

    assert profile.shape == (2, 1)
    assert np.all(np.isfinite(profile))
    assert np.all(profile >= 0.0)


def test_lblrtm_panel_accumulation_matches_weighted_line_profiles():
    grid = np.linspace(999.6, 1000.4, 801)
    centers = np.array([999.91, 1000.03, 1000.17])
    sigma = np.array([0.012, 0.018, 0.015])
    gamma = np.array([0.025, 0.010, 0.035])
    scale = np.array([0.7, 1.2, 0.4])
    groups = np.array([0, 1, 0])
    coupling = np.zeros(3)

    accumulated = lblrtm_panel_accumulate_wavenumber(
        grid,
        centers,
        sigma,
        gamma,
        scale,
        groups,
        2,
        profile_coupling=coupling,
    )
    profile = lblrtm_panel_voigt_profile_wavenumber(grid, centers, sigma, gamma)
    alfv = lblrtm_voigt_hwhm(gamma, sigma)
    profile *= 1.0 + coupling[:, None] * (grid[None, :] - centers[:, None]) / alfv[:, None]
    expected = np.vstack(
        [
            np.sum(profile[groups == group] * scale[groups == group, None], axis=0)
            for group in range(2)
        ]
    )

    np.testing.assert_allclose(accumulated, expected, rtol=2.0e-12, atol=2.0e-11)


def test_h2o_continuum_scaling_matches_lblrtm_contnm_terms_without_interpolation():
    wavenumber = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0])
    continuum = MTCKDH2OContinuum(
        wavenumber_cm=wavenumber,
        self_absco_ref=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) * 1.0e-24,
        foreign_absco_ref=np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) * 1.0e-25,
        foreign_closure_absco_ref=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) * 1.0e-25,
        self_temperature_exponent=np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]),
        reference_pressure_mbar=1013.0,
        reference_temperature_k=296.0,
    )
    target = np.array([110.0, 120.0, 130.0])
    pressure = 800.0
    temperature = 260.0
    h2o_vmr = 0.03

    self_coeff, foreign_coeff = continuum.absorption_coefficients(
        target,
        pressure_mbar=pressure,
        temperature_k=temperature,
        h2o_vmr=h2o_vmr,
        include_radiation_term=True,
    )

    density_ratio = (pressure / 1013.0) * (296.0 / temperature)
    radiation = radiation_term_cm(target, temperature)
    expected_self = (
        continuum.self_absco_ref[1:4]
        * (296.0 / temperature) ** continuum.self_temperature_exponent[1:4]
        * h2o_vmr
        * density_ratio
        * radiation
    )
    expected_foreign = continuum.foreign_absco_ref[1:4] * (1.0 - h2o_vmr) * density_ratio * radiation
    np.testing.assert_allclose(self_coeff, expected_self, rtol=1.0e-14)
    np.testing.assert_allclose(foreign_coeff, expected_foreign, rtol=1.0e-14)


def test_molecfit_synthetic_lsf_kernel_sizes_and_normalisation():
    box = _fractional_box_kernel(2.4)
    edge = np.fmod((2.4 - 1.0) / 2.0, 1.0) / 2.4
    center = 1.0 / 2.4
    np.testing.assert_allclose(box, [edge, center, edge], rtol=1.0e-14)
    np.testing.assert_allclose(np.sum(box), 1.0, rtol=1.0e-14)

    sigma = 0.9
    fwhm = sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel_width_fwhm = 5.0
    expected_gauss_size = int(2 * np.ceil(fwhm * kernel_width_fwhm / 2.0 - 0.5) + 1)
    gaussian = _integrated_gaussian_kernel(sigma, kernel_width_fwhm=kernel_width_fwhm)
    assert gaussian.size == expected_gauss_size
    np.testing.assert_allclose(np.sum(gaussian), 1.0, rtol=1.0e-14)

    lorentz_fwhm = 1.7
    expected_lorentz_size = int(2 * np.ceil(lorentz_fwhm * kernel_width_fwhm / 2.0 - 0.5) + 1)
    lorentz = _integrated_lorentz_kernel(lorentz_fwhm, kernel_width_fwhm=kernel_width_fwhm)
    assert lorentz.size == expected_lorentz_size
    np.testing.assert_allclose(np.sum(lorentz), 1.0, rtol=1.0e-14)


def test_molecfit_synthetic_voigt_approximation_kernel():
    gaussian_fwhm = 1.4
    lorentz_fwhm = 0.8
    kernel_width_fwhm = 5.0
    gamma = lorentz_fwhm / 2.0
    voigt_fwhm = gamma + np.sqrt(gamma * gamma + gaussian_fwhm * gaussian_fwhm)
    expected_size = int(2 * np.ceil(voigt_fwhm * kernel_width_fwhm / 2.0 - 0.5) + 1)

    kernel = _molecfit_voigt_approx_kernel(
        gaussian_fwhm,
        lorentz_fwhm,
        kernel_width_fwhm=kernel_width_fwhm,
    )

    assert kernel.size == expected_size
    np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1.0e-14)
    np.testing.assert_allclose(kernel, kernel[::-1], rtol=1.0e-14)
    assert kernel[kernel.size // 2] == np.max(kernel)


def test_lnfl_tape8_temperature_exponent_conversion_is_documented():
    hitran_temperature_exponent = 0.72

    lblrtm_tmpalf = 1.0 - hitran_temperature_exponent
    tape8_temperature_exponent = 1.0 - lblrtm_tmpalf

    np.testing.assert_allclose(tape8_temperature_exponent, hitran_temperature_exponent)
