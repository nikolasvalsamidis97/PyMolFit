from dataclasses import fields

import numpy as np
from scipy.special import wofz

from genmolfit.atmosphere import (
    AtmosphereLayer,
    AtmosphereProfile,
    BOLTZMANN_J_PER_K,
    CM_PER_M,
    PA_PER_ATM,
)
import genmolfit.components as components_impl
from genmolfit.components import (
    _line_wing_settings,
    _screen_and_accumulate_f4_chunk,
    hitran_line_optical_depth_basis,
)
from genmolfit.linelist import LineList
from genmolfit.physics import (
    AMU_KG,
    LBLRTM_AVRAT,
    LBLRTM_AVRAT_ZETA_GRID,
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_DPTFAC,
    LBLRTM_DEFAULT_DPTMIN,
    LBLRTM_DEFAULT_SAMPLE,
    LBLRTM_F4_BOUND_CM,
    LBLRTM_F4_GRID_RATIO,
    LBLRTM_SECOND_RADIATION_CONSTANT_CM_K,
    LBLRTM_VOIGT_TABLE_DOMAINS,
    LBLRTM_VOIGT_TABLE_POINTS,
    LBLRTM_VOIGT_DOMAIN_HWF3,
    SECOND_RADIATION_CONSTANT_CM_K,
    SPEED_OF_LIGHT_M_PER_S,
    _lblrtm_f4_coefficient_tables,
    _lblrtm_voigt_subfunction_tables,
    lblrtm_voigt_hwhm,
    lblrtm_radiation_term,
    lblrtm_f4_peak_factor,
    lblrtm_f4_profile_offset,
    lblrtm_layer_wavenumber_spacing_cm,
    lblrtm_layer_wavenumber_spacings_cm,
    lblrtm_merge_layer_wavenumber_spacings_cm,
    lblrtm_panel_accumulate_wavenumber,
    lblrtm_panel_interpolate_f4_wavenumber,
)


def _single_h2o_line(*, strength=2.0e-24, air_width=0.07, self_width=0.30):
    center_cm = 4320.0
    return LineList(
        wavelength=np.array([1.0e4 / center_cm]),
        strength=np.array([strength]),
        sigma=np.array([0.01]),
        gamma=np.array([0.02]),
        species=np.array(["H2O"]),
        wavenumber=np.array([center_cm]),
        mol_id=np.array([1]),
        iso_id=np.array([1]),
        air_width=np.array([air_width]),
        self_width=np.array([self_width]),
        lower_state_energy=np.array([100.0]),
        temperature_exponent=np.array([0.75]),
        pressure_shift=np.array([-0.001]),
        molecular_mass_amu=np.array([18.010565]),
        line_source="hitran_par",
    )


def test_panel_default_keeps_source_dynamic_window_unless_cutoff_is_explicit():
    _, default_cutoff, _, _ = _line_wing_settings(
        line_wing_mode="lblrtm_panel",
        line_cutoff_cm=None,
        subtract_cutoff_profile=False,
        line_taper_cm=0.0,
    )
    _, explicit_cutoff, _, _ = _line_wing_settings(
        line_wing_mode="lblrtm_panel",
        line_cutoff_cm=18.0,
        subtract_cutoff_profile=False,
        line_taper_cm=0.0,
    )

    assert default_cutoff is None
    assert explicit_cutoff == 18.0


def _one_layer_atmosphere():
    return AtmosphereProfile.single_layer(
        pressure_atm=0.72,
        temperature_k=278.0,
        path_length_m=1500.0,
        mixing_ratios={"H2O": 1.2e-5},
    )


def _manual_layer_quantities(line_list, atmosphere):
    layer = atmosphere.layers[0]
    reference_temperature = line_list.reference_temperature
    vmr = layer.mixing_ratios["H2O"]
    center = line_list.wavenumber[0] + line_list.pressure_shift[0] * layer.pressure_atm

    c2 = SECOND_RADIATION_CONSTANT_CM_K
    partition_ratio = (reference_temperature / layer.temperature_k) ** 1.5
    strength = (
        line_list.strength[0]
        * partition_ratio
        * np.exp(-c2 * line_list.lower_state_energy[0] * (1.0 / layer.temperature_k - 1.0 / reference_temperature))
        * (1.0 - np.exp(-c2 * line_list.wavenumber[0] / layer.temperature_k))
        / (1.0 - np.exp(-c2 * line_list.wavenumber[0] / reference_temperature))
    )
    mass_kg = line_list.molecular_mass_amu[0] * AMU_KG
    sigma = center * np.sqrt(
        BOLTZMANN_J_PER_K * layer.temperature_k / (mass_kg * SPEED_OF_LIGHT_M_PER_S**2)
    )
    gamma = (
        ((1.0 - vmr) * line_list.air_width[0] + vmr * line_list.self_width[0])
        * layer.pressure_atm
        * (reference_temperature / layer.temperature_k) ** line_list.temperature_exponent[0]
    )
    column = (
        layer.pressure_atm
        * PA_PER_ATM
        * layer.path_length_m
        / (BOLTZMANN_J_PER_K * layer.temperature_k * CM_PER_M**2)
        * vmr
    )
    return center, strength, sigma, gamma, column


def _source_panel_profile(wavenumber, center, sigma, gamma, *, effective_hwhm=None):
    """Independent scalar transcription of CNVFNV and PANEL for one line."""

    spacing = float(wavenumber[1] - wavenumber[0])
    pad = 32
    fine = wavenumber[0] + spacing * np.arange(-pad, wavenumber.size + pad, dtype=float)
    coarse2 = fine[0] + 4.0 * spacing * np.arange(int(np.ceil(fine.size / 4.0)) + 4)
    coarse3 = fine[0] + 16.0 * spacing * np.arange(int(np.ceil(fine.size / 16.0)) + 4)

    doppler_hwhm = sigma * np.sqrt(2.0 * np.log(2.0))
    zeta = gamma / (gamma + doppler_hwhm)
    avrat = np.interp(zeta, LBLRTM_AVRAT_ZETA_GRID, LBLRTM_AVRAT)
    voigt_hwhm = (
        avrat * (gamma + doppler_hwhm)
        if effective_hwhm is None
        else float(effective_hwhm)
    )
    zeta_position = 100.0 * zeta
    zeta_index = min(max(int(np.floor(zeta_position)), 0), 100)
    zeta_fraction = zeta_position - zeta_index
    tables = _lblrtm_voigt_subfunction_tables()

    def deposit(target, domain_index):
        domain = LBLRTM_VOIGT_TABLE_DOMAINS[domain_index]
        normalized = np.abs(target - center) / voigt_hwhm
        table_position = normalized * (LBLRTM_VOIGT_TABLE_POINTS - 1) / domain
        table_index = np.floor(table_position + 0.5).astype(int)
        keep = (table_index >= 0) & (table_index < LBLRTM_VOIGT_TABLE_POINTS)
        table_index = np.clip(table_index, 0, LBLRTM_VOIGT_TABLE_POINTS - 1)
        values = (
            tables[domain_index][zeta_index, table_index]
            + zeta_fraction
            * (
                tables[domain_index][zeta_index + 1, table_index]
                - tables[domain_index][zeta_index, table_index]
            )
        )
        return np.where(keep, values / voigt_hwhm, 0.0)

    def interpolate_four_to_one(coarse, size):
        padded = np.pad(coarse, (1, 2))
        fine_values = np.zeros(size, dtype=float)
        x00, x01, x02, x03 = -7.0 / 128.0, 105.0 / 128.0, 35.0 / 128.0, -5.0 / 128.0
        x10, x11 = -1.0 / 16.0, 9.0 / 16.0
        for output_index in range(size):
            base = min(output_index // 4, coarse.size - 1) + 1
            phase = output_index % 4
            pm1, p0, p1, p2 = padded[base - 1 : base + 3]
            if phase == 0:
                fine_values[output_index] = p0
            elif phase == 1:
                fine_values[output_index] = x00 * pm1 + x01 * p0 + x02 * p1 + x03 * p2
            elif phase == 2:
                fine_values[output_index] = x10 * (pm1 + p2) + x11 * (p0 + p1)
            else:
                fine_values[output_index] = x03 * pm1 + x02 * p0 + x01 * p1 + x00 * p2
        return fine_values

    def xint(source_grid, source_values, target_grid):
        result = np.zeros(target_grid.shape, dtype=float)
        spacing = source_grid[1] - source_grid[0]
        for index, target in enumerate(target_grid):
            position = (target - source_grid[0]) / spacing
            base = int(np.floor(position))
            if base < 1 or base + 2 >= source_grid.size:
                continue
            p = position - base
            c = (3.0 - 2.0 * p) * p**2
            b = 0.5 * p * (1.0 - p)
            b1 = b * (1.0 - p)
            b2 = b * p
            result[index] = (
                -source_values[base - 1] * b1
                + source_values[base] * (1.0 - c + b2)
                + source_values[base + 1] * (c + b1)
                - source_values[base + 2] * b2
            )
        return result

    r4_spacing = LBLRTM_F4_GRID_RATIO * spacing
    r4_grid = np.arange(
        wavenumber[0] - 2.0 * r4_spacing,
        wavenumber[-1] + 2.5 * r4_spacing,
        r4_spacing,
    )
    offset = r4_grid - center
    a3_table, b3_table = _lblrtm_f4_coefficient_tables()
    a3 = a3_table[zeta_index] + zeta_fraction * (a3_table[zeta_index + 1] - a3_table[zeta_index])
    b3 = b3_table[zeta_index] + zeta_fraction * (b3_table[zeta_index + 1] - b3_table[zeta_index])
    offset_sq = offset**2
    z_sq = offset_sq / voigt_hwhm**2
    z_bound_sq = LBLRTM_VOIGT_DOMAIN_HWF3**2
    f4_at_64 = a3 + b3 * z_bound_sq
    numerator = f4_at_64 / voigt_hwhm * (gamma**2 + voigt_hwhm**2 * z_bound_sq)
    boundary = numerator / (gamma**2 + LBLRTM_F4_BOUND_CM**2)
    r4 = np.where(
        z_sq <= z_bound_sq,
        (a3 + b3 * z_sq) / voigt_hwhm,
        numerator / (gamma**2 + offset_sq),
    ) - boundary
    r4 = np.where(np.abs(offset) <= LBLRTM_F4_BOUND_CM, r4, 0.0)

    r1 = deposit(fine, 0)
    r2 = deposit(coarse2, 1)
    r3 = deposit(coarse3, 2)
    r3 += xint(r4_grid, r4, coarse3)
    r2 += interpolate_four_to_one(r3, r2.size)
    r1 += interpolate_four_to_one(r2, r1.size)
    return r1[pad : pad + wavenumber.size]


def test_single_line_component_path_matches_audited_lblrtm_panel_formula():
    """Audit the complete HITRAN line-basis path for one isolated layer/line.

    This intentionally uses synthetic data, not a real spectrum.  It verifies
    that GenMolFit's combined line path still follows the source-derived
    bookkeeping: HITRAN pressure shift, HITRAN temperature intensity scaling,
    Doppler width, Lorentz width, column density, and the LBLRTM panel Voigt
    accumulator.
    """

    line_list = _single_h2o_line()
    atmosphere = _one_layer_atmosphere()
    wavenumber = np.linspace(4319.5, 4320.5, 301)
    wavelength = 1.0e4 / wavenumber

    names, basis = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        species=("H2O",),
        chunk_size=1,
        line_wing_mode="lblrtm_panel",
    )
    center, strength, sigma, gamma, column = _manual_layer_quantities(line_list, atmosphere)
    raw_alfv = float(lblrtm_voigt_hwhm(np.array([gamma]), np.array([sigma]))[0])
    spacing = float(np.median(np.diff(wavenumber)))
    layer_spacing = float(
        lblrtm_layer_wavenumber_spacing_cm(
            0.5 * (wavenumber[0] + wavenumber[-1]),
            atmosphere.layers[0].pressure_atm,
            atmosphere.layers[0].temperature_k,
            h2o_fraction=atmosphere.layers[0].mixing_ratios["H2O"],
        )
    )
    alfmax = 4.0 * LBLRTM_DEFAULT_SAMPLE * layer_spacing * 0.04 / LBLRTM_DEFAULT_ALFAL0
    bounded_alfv = np.clip(raw_alfv, layer_spacing, alfmax)
    line_scale = np.array([strength * column])
    r4_spacing = LBLRTM_F4_GRID_RATIO * spacing
    r4_grid = np.arange(
        np.min(wavenumber) - LBLRTM_F4_BOUND_CM - 2.0 * r4_spacing,
        np.max(wavenumber) + LBLRTM_F4_BOUND_CM + 3.0 * r4_spacing,
        r4_spacing,
    )
    total_r4 = np.zeros(r4_grid.size)
    grouped_r4 = np.zeros((1, r4_grid.size))
    accepted_scale = _screen_and_accumulate_f4_chunk(
        r4_grid=r4_grid,
        total_r4=total_r4,
        grouped_r4=grouped_r4,
        centers=np.array([center]),
        sigma=np.array([sigma]),
        gamma=np.array([gamma]),
        raw_alfv=np.array([raw_alfv]),
        line_scale=line_scale,
        group_index=np.array([0]),
        profile_coupling=np.array([0.0]),
        dptmin=LBLRTM_DEFAULT_DPTMIN
        / lblrtm_radiation_term(np.max(wavenumber), atmosphere.layers[0].temperature_k),
    )
    core = lblrtm_panel_accumulate_wavenumber(
        wavenumber,
        np.array([center]),
        np.array([sigma]),
        np.array([gamma]),
        accepted_scale,
        np.array([0]),
        1,
        profile_coupling=np.array([0.0]),
        effective_hwhm_cm=np.array([bounded_alfv]),
        include_f4=False,
    )
    f4 = lblrtm_panel_interpolate_f4_wavenumber(wavenumber, r4_grid, grouped_r4)
    expected = core[0] + f4[0]

    assert names == ("H2O",)
    np.testing.assert_allclose(basis[0], expected, rtol=2.0e-13, atol=2.0e-30)


def test_completed_f4_interpolation_matches_panel_f4_difference():
    wavenumber = np.linspace(4319.5, 4320.5, 301)
    center = np.array([4320.0])
    sigma = np.array([0.004])
    gamma = np.array([0.03])
    scale = np.array([2.3e-4])
    group = np.array([0])
    raw_alfv = lblrtm_voigt_hwhm(gamma, sigma)
    spacing = float(np.median(np.diff(wavenumber)))
    r4_spacing = LBLRTM_F4_GRID_RATIO * spacing
    r4_grid = np.arange(
        wavenumber[0] - 2.0 * r4_spacing,
        wavenumber[-1] + 2.5 * r4_spacing,
        r4_spacing,
    )
    r4 = lblrtm_f4_profile_offset(
        r4_grid[None, :] - center[:, None],
        gamma,
        sigma,
        effective_hwhm_cm=raw_alfv,
    ) * scale[:, None]

    completed = lblrtm_panel_interpolate_f4_wavenumber(
        wavenumber,
        r4_grid,
        r4,
    )
    with_f4 = lblrtm_panel_accumulate_wavenumber(
        wavenumber,
        center,
        sigma,
        gamma,
        scale,
        group,
        1,
        effective_hwhm_cm=raw_alfv,
        include_f4=True,
    )
    without_f4 = lblrtm_panel_accumulate_wavenumber(
        wavenumber,
        center,
        sigma,
        gamma,
        scale,
        group,
        1,
        effective_hwhm_cm=raw_alfv,
        include_f4=False,
    )

    np.testing.assert_allclose(completed, with_f4 - without_f4, rtol=3.0e-6, atol=2.0e-12)


def test_two_pass_f4_basis_is_chunk_size_invariant():
    line_list = _single_h2o_line()
    line_list = LineList(
        **{
            field.name: (
                np.repeat(value, 2, axis=0)
                if isinstance((value := getattr(line_list, field.name)), np.ndarray)
                else value
            )
            for field in fields(LineList)
        }
    )
    atmosphere = _one_layer_atmosphere()
    wavenumber = np.linspace(4319.5, 4320.5, 301)
    wavelength = 1.0e4 / wavenumber

    _, one_line_chunks = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        species=("H2O",),
        chunk_size=1,
        line_wing_mode="lblrtm_panel",
    )
    _, one_chunk = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        species=("H2O",),
        chunk_size=128,
        line_wing_mode="lblrtm_panel",
    )

    np.testing.assert_array_equal(one_line_chunks, one_chunk)


def test_cached_two_pass_line_state_is_identical_to_recomputation(monkeypatch):
    line_list = _single_h2o_line()
    atmosphere = AtmosphereProfile(
        layers=(
            AtmosphereLayer(
                pressure_atm=0.72,
                temperature_k=278.0,
                path_length_m=800.0,
                mixing_ratios={"H2O": 1.2e-5},
            ),
            AtmosphereLayer(
                pressure_atm=0.41,
                temperature_k=251.0,
                path_length_m=1600.0,
                mixing_ratios={"H2O": 7.5e-6},
            ),
        )
    )
    wavenumber = np.linspace(4319.5, 4320.5, 301)
    wavelength = 1.0e4 / wavenumber

    monkeypatch.setattr(components_impl, "_CACHE_LBLRTM_SCREENED_LINE_STATE", False)
    names_uncached, uncached = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        species=("H2O",),
        chunk_size=1,
        line_wing_mode="lblrtm_panel",
    )
    monkeypatch.setattr(components_impl, "_CACHE_LBLRTM_SCREENED_LINE_STATE", True)
    names_cached, cached = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        species=("H2O",),
        chunk_size=1,
        line_wing_mode="lblrtm_panel",
    )

    assert names_cached == names_uncached
    np.testing.assert_array_equal(cached, uncached)


def test_single_line_dynamic_cutoff_matches_lblrtm_source_formula():
    """Audit the dynamic LBLRTM line window without fitting any observation."""

    line_list = _single_h2o_line(air_width=0.001, self_width=0.001)
    atmosphere = _one_layer_atmosphere()
    wavenumber = np.linspace(4318.8, 4321.2, 241)
    wavelength = 1.0e4 / wavenumber

    _, basis = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        species=("H2O",),
        chunk_size=1,
        line_wing_mode="lblrtm_dynamic",
        lblrtm_sample=LBLRTM_DEFAULT_SAMPLE,
        lblrtm_alfal0=LBLRTM_DEFAULT_ALFAL0,
        lblrtm_hwf3=LBLRTM_VOIGT_DOMAIN_HWF3,
    )

    center, strength, sigma, gamma, column = _manual_layer_quantities(line_list, atmosphere)
    offset = wavenumber - center
    z = (offset + 1j * gamma) / (sigma * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

    doppler_hwhm = sigma * np.sqrt(2.0 * np.log(2.0))
    zeta = gamma / (gamma + doppler_hwhm)
    avrat = np.interp(zeta, LBLRTM_AVRAT_ZETA_GRID, LBLRTM_AVRAT)
    alfv = avrat * (gamma + doppler_hwhm)
    dv = float(
        lblrtm_layer_wavenumber_spacings_cm(
            0.5 * (wavenumber[0] + wavenumber[-1]),
            np.array([atmosphere.layers[0].pressure_atm]),
            np.array([atmosphere.layers[0].temperature_k]),
            h2o_fraction=np.array([atmosphere.layers[0].mixing_ratios["H2O"]]),
        )[0]
    )
    alfv = max(alfv, dv)
    alfmax = 4.0 * LBLRTM_DEFAULT_SAMPLE * dv * 0.04 / LBLRTM_DEFAULT_ALFAL0
    alfv = min(alfv, alfmax)
    cutoff = LBLRTM_VOIGT_DOMAIN_HWF3 * alfv

    expected = np.where(np.abs(offset) <= cutoff, profile, 0.0) * strength * column

    np.testing.assert_allclose(basis[0], expected, rtol=2.0e-13, atol=2.0e-30)
    assert np.all(basis[0, np.abs(offset) > cutoff] == 0.0)


def test_lblrtm_layer_spacing_merge_matches_captured_tape6_sequence():
    calculated = np.array(
        [
            0.008041,
            0.007292,
            0.006597,
            0.006041,
            0.005611,
            0.005295,
            0.005046,
            0.004810,
            0.004548,
            0.004280,
            0.004033,
            0.003804,
            0.003599,
            0.003409,
        ]
    )
    expected = np.array(
        [
            0.008040,
            0.008040,
            0.006700,
            0.006700,
            0.006700,
            0.005360,
            0.005360,
            0.005360,
            0.005360,
            0.004288,
            0.004288,
            0.004288,
            0.004288,
            0.0034304,
        ]
    )
    np.testing.assert_allclose(
        lblrtm_merge_layer_wavenumber_spacings_cm(calculated),
        expected,
        rtol=0.0,
        atol=5.0e-16,
    )


def test_lblrtm_source_constants_used_by_audit_are_explicit():
    assert LBLRTM_DEFAULT_SAMPLE == 4.0
    assert LBLRTM_DEFAULT_ALFAL0 == 0.04
    assert LBLRTM_VOIGT_DOMAIN_HWF3 == 64.0
    assert LBLRTM_F4_BOUND_CM == 25.0
    assert LBLRTM_F4_GRID_RATIO == 64
    assert LBLRTM_DEFAULT_DPTMIN == 2.0e-4
    assert LBLRTM_DEFAULT_DPTFAC == 1.0e-3
    np.testing.assert_allclose(LBLRTM_SECOND_RADIATION_CONSTANT_CM_K, 1.4387752)


def test_lblrtm_radiation_term_matches_all_source_branches():
    temperature = 300.0
    xkt = temperature / SECOND_RADIATION_CONSTANT_CM_K
    wavenumber = np.array([0.1, 1000.0, 3000.0])
    ratio = wavenumber / xkt
    expected = np.array(
        [
            0.5 * ratio[0] * wavenumber[0],
            wavenumber[1] * (1.0 - np.exp(-ratio[1])) / (1.0 + np.exp(-ratio[1])),
            wavenumber[2],
        ]
    )
    np.testing.assert_allclose(lblrtm_radiation_term(wavenumber, temperature), expected)


def test_lblrtm_dptmin_rejects_only_lines_below_source_threshold():
    atmosphere = _one_layer_atmosphere()
    wavenumber = np.linspace(4319.5, 4320.5, 101)
    wavelength = 1.0e4 / wavenumber

    _, rejected = hitran_line_optical_depth_basis(
        wavelength,
        _single_h2o_line(strength=1.0e-29),
        atmosphere,
        species=("H2O",),
        line_wing_mode="lblrtm_panel",
    )
    _, retained = hitran_line_optical_depth_basis(
        wavelength,
        _single_h2o_line(strength=1.0e-28),
        atmosphere,
        species=("H2O",),
        line_wing_mode="lblrtm_panel",
    )

    assert np.all(rejected == 0.0)
    assert np.any(retained > 0.0)


def test_f4_screening_is_source_ordered_and_uses_accumulated_r4():
    r4_grid = np.linspace(4319.0, 4321.0, 257)
    centers = np.array([4320.0, 4320.0])
    sigma = np.full(2, 0.01)
    gamma = np.full(2, 0.05)
    raw_alfv = lblrtm_voigt_hwhm(gamma, sigma)
    peak = lblrtm_f4_peak_factor(gamma, sigma)
    line_scale = np.array([1.0, 5.0e-4]) * raw_alfv / peak
    total = np.zeros(r4_grid.size)
    grouped = np.zeros((1, r4_grid.size))

    accepted = _screen_and_accumulate_f4_chunk(
        r4_grid=r4_grid,
        total_r4=total,
        grouped_r4=grouped,
        centers=centers,
        sigma=sigma,
        gamma=gamma,
        raw_alfv=raw_alfv,
        line_scale=line_scale,
        group_index=np.zeros(2, dtype=int),
        profile_coupling=np.zeros(2),
        dptmin=2.0e-4,
    )

    assert accepted[0] != 0.0
    assert accepted[1] == 0.0
    np.testing.assert_allclose(total, grouped[0])


def test_lblrtm_voigt_tables_retain_source_real_precision():
    assert all(table.dtype == np.float32 for table in _lblrtm_voigt_subfunction_tables())
