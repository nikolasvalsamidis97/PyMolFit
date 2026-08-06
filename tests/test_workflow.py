from dataclasses import replace
import inspect

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import pymolfit.workflow as workflow_module
from pymolfit import (
    AtmosphereProfile,
    LineList,
    ModelConfig,
    Observation,
    Spectrum,
    air_to_vacuum_wavelength,
    correct,
    correct_arrays,
    correct_file,
    save_corrected_txt,
    save_fit_product_ecsv,
    transmission_model,
    vacuum_to_air_wavelength,
)
from pymolfit.fit import FitConfig, fit_telluric_segments, fit_tellurics
from pymolfit.physics import SPEED_OF_LIGHT_M_PER_S
from pymolfit.workflow import (
    _barycentric_velocity_from_header_km_s,
    _estimate_lsf_sigma_from_resolving_power,
    _estimate_lsf_sigma_from_spectral_features,
    _make_atmosphere,
    _infer_wavelength_medium_from_header,
    _ranges_to_observatory_vacuum,
    _resolve_initial_wavelength_shift,
    _resolve_line_list,
    _split_spectrum,
    _spectrum_to_observatory_vacuum,
    _stitch_segment_results,
)
from pymolfit.fit import _shift_basis
from pymolfit.model import optical_depth_basis, transmission_from_basis


def _fixed_decimal(value, width, decimals):
    text = f"{value:.{decimals}f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return f"{text:>{width}}"[-width:]


def _hitran_row(*, mol_id=1, wavenumber=4320.0, strength=1.0e-24):
    row = (
        f"{mol_id:2d}"
        f"{1:1d}"
        f"{wavenumber:12.6f}"
        f"{strength:10.3E}"
        f"{1.0:10.3E}"
        f"{_fixed_decimal(0.07, 5, 4)}"
        f"{_fixed_decimal(0.30, 5, 4)}"
        f"{100.0:10.4f}"
        f"{0.75:4.2f}"
        f"{_fixed_decimal(-0.001, 8, 6)}"
    )
    return row + " " * (160 - len(row))


def test_physical_line_list_only_auto_enables_matching_molecule_continua(tmp_path):
    hitran_path = tmp_path / "o2.par"
    hitran_path.write_text(_hitran_row(mol_id=7) + "\n")
    center = 1.0e4 / 4320.0
    wavelength = np.linspace(center - 0.001, center + 0.001, 80)

    result = correct_arrays(
        wavelength,
        np.ones_like(wavelength),
        hitran_par=hitran_path,
        hitran_species=("O2",),
        mixing_ratios={"O2": 0.2095},
        allow_default_observatory=True,
        continuum_order=0,
    )

    assert result.success
    assert set(result.species_scales) == {"O2"}


def test_rayleigh_component_uses_fixed_physical_scale():
    wavelength = np.linspace(0.50, 0.51, 80)

    result = correct_arrays(
        wavelength,
        np.ones_like(wavelength),
        line_list=LineList.empty_hitran(),
        atmosphere=AtmosphereProfile.standard_midlatitude(),
        continuum_order=0,
        solve_continuum_linear=True,
        rayleigh=True,
        high_resolution_grid=False,
        auto_segment=False,
    )

    assert result.success
    assert result.species_scales == {"Rayleigh": 1.0}
    assert result.provenance["fit_quality"]["measured_group_count"] == 0


def test_correct_arrays_uses_demo_workflow():
    wavelength = np.linspace(2.31, 2.36, 400)
    line_list = LineList.demo_near_ir()
    flux = transmission_model(wavelength, line_list, ModelConfig(species_scales={"H2O": 1.3}))

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        solve_continuum_linear=True,
    )

    assert result.success
    assert result.corrected.flux.shape == wavelength.shape


def test_correct_arrays_preserves_overlapping_echelle_group_ids():
    first = np.linspace(1.500, 1.530, 120)
    second = np.linspace(1.510, 1.540, 120)
    wavelength = np.concatenate((first, second))
    group_id = np.concatenate(
        (np.full(first.size, "order-1"), np.full(second.size, "order-2"))
    )
    line_list = LineList(
        wavelength=np.array([1.520]),
        strength=np.array([0.01]),
        sigma=np.array([2.0e-5]),
        gamma=np.array([1.0e-5]),
        species=np.array(["H2O"]),
    )

    result = correct_arrays(
        wavelength,
        np.ones_like(wavelength),
        group_id=group_id,
        line_list=line_list,
        continuum_order=0,
        solve_continuum_linear=True,
        auto_segment=True,
        segment_size=0.01,
    )

    np.testing.assert_array_equal(result.spectrum.group_id, group_id)
    np.testing.assert_array_equal(result.corrected.group_id, group_id)


def test_multi_order_auto_alignment_uses_constant_shift_per_physical_group():
    wavelength = np.concatenate(
        (np.linspace(1.500, 1.506, 240), np.linspace(1.510, 1.516, 240))
    )
    group_id = np.repeat((10, 20), 240)
    line_list = LineList(
        wavelength=np.array([1.503, 1.513]),
        strength=np.array([0.006, 0.005]),
        sigma=np.full(2, 2.0e-5),
        gamma=np.full(2, 1.0e-5),
        species=np.array(["H2O", "H2O"]),
    )
    shifts = np.array([7.0e-5, -6.0e-5])
    shifted_lines = LineList(
        wavelength=line_list.wavelength + shifts,
        strength=line_list.strength,
        sigma=line_list.sigma,
        gamma=line_list.gamma,
        species=line_list.species,
    )
    flux = transmission_model(
        wavelength,
        shifted_lines,
        ModelConfig(species_scales={"H2O": 1.4}),
    )

    result = correct_arrays(
        wavelength,
        flux,
        group_id=group_id,
        line_list=line_list,
        continuum_order=0,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        high_resolution_grid=False,
        segment_size=0.02,
    )

    alignment = result.provenance["wavelength_alignment"]
    assert alignment["selected_model"] == "per_segment_constant"
    recovered = result.provenance["segmentation"]["wavelength_shifts_micron"]
    np.testing.assert_allclose(recovered, shifts, atol=3.0e-5)


def test_unified_correct_passes_array_group_ids():
    wavelength = np.concatenate(
        (np.linspace(1.500, 1.510, 40), np.linspace(1.505, 1.515, 40))
    )
    group_id = np.repeat((3, 7), 40)
    observation = Observation(wavelength_frame="observatory")

    result = correct(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        group_id=group_id,
        wavelength_medium="vacuum",
        observation=observation,
        line_list=LineList.empty_hitran(),
        physical=False,
        continuum_order=0,
        solve_continuum_linear=True,
        auto_segment=True,
        segment_size=0.01,
    )

    np.testing.assert_array_equal(result.spectrum.group_id, group_id)


def test_correct_arrays_automatically_uses_linear_continuum_for_linear_loss():
    wavelength = np.linspace(2.31, 2.36, 400)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.3}),
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=1,
        high_resolution_grid=False,
        auto_segment=False,
    )

    details = result.provenance["continuum_solver"]
    assert result.success
    assert details["requested"] == "auto"
    assert details["selected"] == "linear"
    assert details["fallback_used"] is False
    assert [attempt["solver"] for attempt in details["attempts"]] == ["linear"]


def test_correct_arrays_auto_continuum_falls_back_after_failed_linear_fit(
    monkeypatch,
):
    wavelength = np.linspace(2.31, 2.36, 400)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.3}),
    )
    real_fit = workflow_module.fit_tellurics
    attempted_configs = []

    def fail_linear_fit(spectrum, *, line_list, config):
        attempted_configs.append(config)
        result = real_fit(spectrum, line_list=line_list, config=config)
        if config.solve_continuum_linear:
            return replace(
                result,
                success=False,
                message="forced linear non-convergence",
            )
        return result

    monkeypatch.setattr(workflow_module, "fit_tellurics", fail_linear_fit)

    with pytest.warns(RuntimeWarning, match="retrying with nonlinear"):
        result = correct_arrays(
            wavelength,
            flux,
            line_list=line_list,
            continuum_order=1,
            high_resolution_grid=False,
            auto_segment=False,
        )

    details = result.provenance["continuum_solver"]
    assert result.success
    assert [config.solve_continuum_linear for config in attempted_configs] == [
        True,
        False,
    ]
    assert attempted_configs[0].max_nfev == 100
    assert attempted_configs[1].max_nfev is None
    assert details["selected"] == "nonlinear"
    assert details["fallback_used"] is True
    assert "forced linear non-convergence" in details["fallback_reason"]
    assert [attempt["solver"] for attempt in details["attempts"]] == [
        "linear",
        "nonlinear",
    ]


def test_correct_arrays_auto_continuum_profiles_robust_loss():
    wavelength = np.linspace(2.31, 2.36, 400)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.3}),
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=1,
        loss="soft_l1",
        high_resolution_grid=False,
        auto_segment=False,
    )

    details = result.provenance["continuum_solver"]
    assert result.success
    assert details["requested"] == "auto"
    assert details["selected"] == "linear"
    assert "iteratively reweighted" in details["selection_reason"]
    assert details["fallback_used"] is False
    assert "soft_l1" in details["selection_reason"]


def test_lsf_sigma_estimate_uses_resolving_power_and_spectrum_sampling():
    wavelength = np.linspace(0.500, 0.501, 1001)
    spectrum = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))

    estimate = _estimate_lsf_sigma_from_resolving_power(
        spectrum,
        {"SPEC_RES": 100_000.0},
    )

    assert estimate is not None
    assert estimate["source"] == "fits_resolving_power"
    assert estimate["header_keyword"] == "SPEC_RES"
    assert estimate["resolving_power"] == 100_000.0
    assert estimate["initial_sigma_pixels"] == pytest.approx(
        0.5005 / 100_000.0 / 1.0e-6 / 2.354820045,
        rel=2.0e-3,
    )


def test_lsf_sigma_estimate_uses_narrow_spectral_features_without_metadata():
    pixel = np.arange(600.0)
    wavelength = 0.600 + pixel * 1.0e-6
    flux = np.ones(pixel.size)
    for center in (100.0, 250.0, 430.0):
        flux -= 0.25 * np.exp(-0.5 * ((pixel - center) / 2.0) ** 2)

    estimate = _estimate_lsf_sigma_from_spectral_features(
        Spectrum(wavelength=wavelength, flux=flux),
        fit_ranges=None,
        exclude_ranges=None,
    )

    assert estimate is not None
    assert estimate["source"] == "spectrum_features"
    assert estimate["feature_count"] == 3
    assert estimate["initial_sigma_pixels"] == pytest.approx(2.0, abs=0.15)


def test_correct_arrays_automatically_estimates_and_refines_lsf_sigma():
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
        ModelConfig(
            species_scales={"H2O": 1.0},
            lsf_sigma_pixels=2.0,
        ),
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        auto_segment=False,
    )

    details = result.provenance["lsf_sigma"]
    assert result.success
    assert result.lsf_sigma_pixels == pytest.approx(2.0, abs=0.15)
    assert details["requested"] == "auto"
    assert details["fit_enabled"] is True
    assert details["source"] == "spectrum_features"
    assert details["coarse_search"]["candidate_count"] >= 7


def test_correct_arrays_keeps_explicit_lsf_sigma_fixed_by_default():
    wavelength = np.linspace(2.31, 2.36, 300)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    flux = transmission_model(wavelength, line_list)

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        lsf_sigma_pixels=1.25,
        continuum_order=0,
        auto_segment=False,
    )

    details = result.provenance["lsf_sigma"]
    assert result.lsf_sigma_pixels == pytest.approx(1.25)
    assert details["source"] == "user"
    assert details["fit_requested"] == "auto"
    assert details["fit_enabled"] is False


def test_correct_arrays_disables_auto_lsf_without_metadata_or_features():
    wavelength = np.linspace(2.31, 2.36, 300)

    result = correct_arrays(
        wavelength,
        np.ones_like(wavelength),
        line_list=LineList.demo_near_ir(),
        continuum_order=0,
        auto_segment=False,
    )

    details = result.provenance["lsf_sigma"]
    assert result.lsf_sigma_pixels == 0.0
    assert details["source"] == "disabled_no_lsf_information"
    assert details["fit_enabled"] is False


def _distributed_lsf_test_spectrum(lorentz_fwhm_pixels):
    rng = np.random.default_rng(42)
    wavelength = np.linspace(2.30, 2.35, 10_000)
    centers = np.array(
        [
            2.303,
            2.306,
            2.313,
            2.316,
            2.323,
            2.326,
            2.333,
            2.336,
            2.343,
            2.346,
        ]
    )
    line_list = LineList(
        wavelength=centers,
        strength=np.full(centers.size, 3.0e-6),
        sigma=np.full(centers.size, 5.0e-7),
        gamma=np.full(centers.size, 1.0e-7),
        species=np.full(centers.size, "H2O"),
    )
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(
            species_scales={"H2O": 1.0},
            lsf_sigma_pixels=1.4,
            lsf_lorentz_fwhm_pixels=lorentz_fwhm_pixels,
        ),
    )
    flux += rng.normal(0.0, 1.0e-4, wavelength.size)
    return wavelength, flux, line_list


def test_automatic_lsf_pilots_keep_gaussian_model_without_wings():
    wavelength, flux, line_list = _distributed_lsf_test_spectrum(0.0)

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=1.2,
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.2, 6.0),
    )

    details = result.provenance["lsf_lorentz"]
    assert result.success
    assert details["pilot_region_count"] >= 2
    assert details["selected_model"] == "gaussian"
    assert result.lsf_lorentz_fwhm_pixels == 0.0


def test_automatic_lsf_pilots_select_consistent_lorentzian_wings():
    wavelength, flux, line_list = _distributed_lsf_test_spectrum(3.0)

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=1.2,
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.2, 6.0),
    )

    details = result.provenance["lsf_lorentz"]
    assert result.success
    assert details["pilot_region_count"] >= 2
    assert details["selected_model"] == "gaussian_lorentz"
    assert details["bic_improvement"] >= 10.0
    assert details["improved_region_fraction"] >= 0.6
    assert result.lsf_lorentz_fwhm_pixels > 0.0


@pytest.mark.parametrize(
    ("true_exponent", "expected_model"),
    ((0.0, "constant"), (1.25, "power_law")),
)
def test_automatic_lsf_variable_width_selects_supported_model(
    true_exponent,
    expected_model,
):
    centers = np.array([1.0, 1.5, 2.0])
    line_list = LineList(
        wavelength=centers,
        strength=np.full(centers.size, 0.03),
        sigma=np.full(centers.size, 1.0e-5),
        gamma=np.full(centers.size, 5.0e-6),
        species=np.full(centers.size, "H2O"),
    )
    wavelength_parts = []
    flux_parts = []
    for center in centers:
        wavelength = np.linspace(center - 0.0015, center + 0.0015, 600)
        species_names, basis = optical_depth_basis(wavelength, line_list)
        flux = transmission_from_basis(
            species_names,
            basis,
            species_scales={"H2O": 1.0},
            lsf_sigma_pixels=2.5,
            wavelength_micron=wavelength,
            lsf_variable_width=true_exponent != 0.0,
            lsf_reference_wavelength_micron=1.5,
            lsf_wavelength_exponent=true_exponent,
        )
        wavelength_parts.append(wavelength)
        flux_parts.append(flux)
    wavelength = np.concatenate(wavelength_parts)
    flux = np.concatenate(flux_parts)

    result = correct_arrays(
        wavelength,
        flux,
        uncertainty=np.full(wavelength.size, 0.002),
        line_list=line_list,
        physical=False,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=2.5,
        fit_lsf_sigma=False,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
        lsf_variable_width="auto",
        fit_wavelength_shift=False,
        high_resolution_grid=False,
        auto_segment=False,
    )

    details = result.provenance["lsf_variable_width"]
    assert details["selected_model"] == expected_model
    assert details["reference_wavelength_micron"] == pytest.approx(1.5)
    if true_exponent == 0.0:
        assert result.lsf_wavelength_exponent == 0.0
        assert details["bic_improvement"] < 6.0
    else:
        assert details["bic_improvement"] >= 6.0
        assert details["improved_region_fraction"] >= 0.6
        assert result.lsf_wavelength_exponent == pytest.approx(
            true_exponent,
            abs=0.04,
        )


def test_correct_arrays_rejects_unknown_continuum_solver_mode():
    wavelength = np.linspace(2.31, 2.36, 100)

    with pytest.raises(
        ValueError,
        match="solve_continuum_linear must be 'auto', True, or False",
    ):
        correct_arrays(
            wavelength,
            np.ones_like(wavelength),
            line_list=LineList.demo_near_ir(),
            solve_continuum_linear="sometimes",
        )


def test_correct_arrays_rejects_unknown_lsf_variable_width_mode():
    wavelength = np.linspace(2.31, 2.36, 100)

    with pytest.raises(
        ValueError,
        match="lsf_variable_width must be 'auto', True, or False",
    ):
        correct_arrays(
            wavelength,
            np.ones_like(wavelength),
            line_list=LineList.demo_near_ir(),
            lsf_variable_width="sometimes",
        )


def test_correct_arrays_exposes_minimum_transmission_mask():
    wavelength = np.linspace(1.0, 1.01, 400)
    line_list = LineList(
        wavelength=np.array([1.005]),
        strength=np.array([0.02]),
        sigma=np.array([5.0e-5]),
        gamma=np.array([2.0e-5]),
        species=np.array(["H2O"]),
    )
    flux = transmission_model(wavelength, line_list)

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        solve_continuum_linear=True,
        min_transmission=0.5,
    )

    opaque = result.transmission < 0.5
    assert np.any(opaque)
    assert np.all(~result.corrected.valid[opaque])
    assert np.all(np.isnan(result.corrected.flux[opaque]))


def test_correct_arrays_fixes_species_below_default_observability_threshold():
    wavelength = np.linspace(2.31, 2.36, 500)
    line_list = LineList(
        wavelength=np.array([2.325, 2.345]),
        strength=np.array([0.01, 1.0e-14]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "O2"]),
    )
    flux = transmission_model(wavelength, line_list, ModelConfig())

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        lsf_sigma_pixels=0.0,
        fit_lsf_sigma=False,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
        lsf_variable_width=False,
        fit_wavelength_shift=False,
    )

    observability = result.provenance["species_observability"]
    assert observability["minimum_peak_optical_depth"] == pytest.approx(5.0e-3)
    assert observability["automatically_fixed"] == {"O2": 1.0}
    assert "log_scale:O2" not in result.parameter_names


def test_correct_arrays_allows_expert_observability_threshold_override():
    wavelength = np.linspace(2.31, 2.36, 500)
    line_list = LineList(
        wavelength=np.array([2.325, 2.345]),
        strength=np.array([0.01, 1.0e-14]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "O2"]),
    )
    flux = transmission_model(wavelength, line_list, ModelConfig())

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        lsf_sigma_pixels=0.0,
        fit_lsf_sigma=False,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
        lsf_variable_width=False,
        fit_wavelength_shift=False,
        minimum_species_peak_optical_depth=0.0,
    )

    observability = result.provenance["species_observability"]
    assert observability["minimum_peak_optical_depth"] == 0.0
    assert observability["automatically_fixed"] == {}
    assert "log_scale:O2" in result.parameter_names


def test_correct_arrays_accepts_native_radiative_transfer_controls():
    wavelength = np.linspace(2.31, 2.36, 120)
    line_list = LineList.demo_near_ir()

    result = correct_arrays(
        wavelength,
        np.ones_like(wavelength),
        line_list=line_list,
        continuum_order=0,
        radiative_transfer_grid="model",
        radiative_transfer_step_cm=0.002,
        radiative_transfer_max_points=10_000,
        lblrtm_avmass_amu=35.5,
    )

    assert result.success


def test_correct_file_automatically_segments_native_grid_and_stitches_output(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    input_path = tmp_path / "broad_spectrum.txt"
    output_path = tmp_path / "corrected.txt"
    hitran_path.write_text(_hitran_row() + "\n")
    center = 1.0e4 / 4320.0
    wavelength = np.linspace(center - 0.015, center + 0.015, 600)
    np.savetxt(input_path, np.column_stack([wavelength, np.ones_like(wavelength)]))

    with pytest.raises(ValueError, match="exceeding max_points"):
        correct_file(
            input_path,
            wavelength_medium="vacuum",
            hitran_par=hitran_path,
            hitran_species=("H2O",),
            mixing_ratios={"H2O": 1.0e-5},
            allow_default_observatory=True,
            continuum_order=0,
            solve_continuum_linear=True,
            radiative_transfer_max_points=20_000,
            auto_segment=False,
        )

    result = correct_file(
        input_path,
        output_path,
        wavelength_medium="vacuum",
        hitran_par=hitran_path,
        hitran_species=("H2O",),
        mixing_ratios={"H2O": 1.0e-5},
        allow_default_observatory=True,
        continuum_order=0,
        solve_continuum_linear=True,
        radiative_transfer_max_points=20_000,
        segment_size=0.01,
        fit_ranges=((center - 0.001, center + 0.001),),
    )

    assert result.success
    assert output_path.exists()
    segmentation = result.provenance["segmentation"]
    assert segmentation["segment_count"] >= 3
    assert all(
        upper - lower <= 0.01 + 1.0e-12
        for lower, upper in segmentation["boundaries_micron"]
    )
    assert result.spectrum.wavelength.size == wavelength.size
    assert result.corrected.wavelength.size == wavelength.size
    assert 0 < np.count_nonzero(result.fit_mask) < wavelength.size
    np.testing.assert_allclose(result.spectrum.wavelength, wavelength)
    assert np.loadtxt(output_path).shape[0] == wavelength.size


def test_segmented_physical_result_matches_unsegmented_result(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    hitran_path.write_text(_hitran_row() + "\n")
    center = 1.0e4 / 4320.0
    wavelength = np.linspace(center - 0.015, center + 0.015, 600)
    flux = np.ones_like(wavelength)
    options = {
        "hitran_par": hitran_path,
        "hitran_species": ("H2O",),
        "mixing_ratios": {"H2O": 1.0e-5},
        "allow_default_observatory": True,
        "continuum_order": 0,
        "solve_continuum_linear": True,
        "radiative_transfer_max_points": 200_000,
    }

    segmented = correct_arrays(
        wavelength,
        flux,
        segment_size=0.01,
        **options,
    )
    unsegmented = correct_arrays(
        wavelength,
        flux,
        auto_segment=False,
        **options,
    )

    assert segmented.provenance["segmentation"]["segment_count"] == 3
    np.testing.assert_allclose(
        segmented.transmission,
        unsegmented.transmission,
        rtol=0.0,
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        segmented.corrected.flux,
        unsegmented.corrected.flux,
        rtol=0.0,
        atol=5.0e-8,
    )


def test_segmented_overlap_convolution_matches_unsegmented_at_boundary(tmp_path):
    center = 1.0e4 / 4320.0
    lower = center - 0.015
    upper = center + 0.015
    line_wavelength = lower + (upper - lower) / 3.0
    hitran_path = tmp_path / "boundary_h2o.par"
    hitran_path.write_text(
        _hitran_row(
            wavenumber=1.0e4 / line_wavelength,
            strength=1.0e-22,
        )
        + "\n"
    )
    line_list = LineList.from_hitran_par(hitran_path, species=("H2O",))
    wavelength = np.linspace(lower, upper, 600)
    spectrum = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))
    config = FitConfig(
        continuum_order=0,
        solve_continuum_linear=True,
        fixed_species_scales={"H2O": 1.0},
        atmosphere=AtmosphereProfile.standard_midlatitude(),
        high_resolution_grid=True,
        high_resolution_rebin_mode="molecfit_overlap",
        radiative_transfer_grid="auto",
        line_wing_mode="lblrtm_panel",
        lsf_sigma_pixels=2.0,
    )

    unsegmented = fit_tellurics(spectrum, line_list=line_list, config=config)
    segments = _split_spectrum(spectrum, segment_size=0.01, minimum_points=2)
    segmented = fit_telluric_segments(
        segments,
        line_list=line_list,
        config=config,
    )
    stitched = np.concatenate(
        [result.transmission for result in segmented.segment_results]
    )

    np.testing.assert_allclose(
        stitched,
        unsegmented.transmission,
        rtol=0.0,
        atol=5.0e-6,
    )


def test_automatic_segmentation_splits_large_echelle_gaps():
    wavelength = np.concatenate(
        (
            np.linspace(1.500, 1.506, 120),
            np.linspace(1.508, 1.514, 120),
        )
    )
    spectrum = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))

    segments = _split_spectrum(spectrum, segment_size=0.01)

    assert len(segments) == 2
    assert segments[0].wavelength[-1] == pytest.approx(1.506)
    assert segments[1].wavelength[0] == pytest.approx(1.508)
    assert all(np.ptp(segment.wavelength) < 0.01 for segment in segments)


def test_automatic_segmentation_preserves_short_island_between_gaps():
    wavelength = np.concatenate(
        (
            np.linspace(1.500, 1.504, 40),
            np.array([1.506]),
            np.linspace(1.508, 1.512, 40),
        )
    )
    spectrum = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))

    segments = _split_spectrum(
        spectrum,
        segment_size=0.01,
        minimum_points=4,
    )

    stitched = np.concatenate([segment.wavelength for segment in segments])
    np.testing.assert_array_equal(stitched, wavelength)
    assert any(1.506 in segment.wavelength for segment in segments)


def test_stitching_keeps_overlapping_physical_groups_contiguous():
    first = np.linspace(1.500, 1.530, 120)
    second = np.linspace(1.510, 1.540, 120)
    spectrum = Spectrum(
        wavelength=np.concatenate((first, second)),
        flux=np.ones(first.size + second.size),
        group_id=np.concatenate(
            (np.full(first.size, 10), np.full(second.size, 20))
        ),
    )
    line_list = LineList(
        wavelength=np.array([1.520]),
        strength=np.array([0.01]),
        sigma=np.array([2.0e-5]),
        gamma=np.array([1.0e-5]),
        species=np.array(["H2O"]),
    )
    config = FitConfig(
        continuum_order=0,
        solve_continuum_linear=True,
        fixed_species_scales={"H2O": 1.0},
    )
    segments = _split_spectrum(spectrum, segment_size=0.01, minimum_points=2)
    multi = fit_telluric_segments(segments, line_list=line_list, config=config)

    stitched = _stitch_segment_results(multi, segment_size=0.01)

    assert stitched.spectrum.group_id is not None
    group_id = stitched.spectrum.group_id
    assert np.count_nonzero(np.diff(group_id)) == 1
    assert np.all(group_id[: first.size] == 10)
    assert np.all(group_id[first.size :] == 20)


def test_correct_arrays_exposes_independent_segment_wavelength_shifts():
    line_list = LineList(
        wavelength=np.array([1.503, 1.513]),
        strength=np.array([0.006, 0.005]),
        sigma=np.full(2, 2.0e-5),
        gamma=np.full(2, 1.0e-5),
        species=np.array(["H2O", "H2O"]),
    )
    shifts = np.array([7.0e-5, -6.0e-5])
    wavelength = np.concatenate(
        (
            np.linspace(1.500, 1.506, 240),
            np.linspace(1.510, 1.516, 240),
        )
    )
    shifted_line_list = LineList(
        wavelength=line_list.wavelength + shifts,
        strength=line_list.strength,
        sigma=line_list.sigma,
        gamma=line_list.gamma,
        species=line_list.species,
    )
    flux = transmission_model(
        wavelength,
        shifted_line_list,
        ModelConfig(species_scales={"H2O": 1.4}),
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        auto_segment=True,
        segment_size=0.02,
        fit_segment_wavelength_shifts=True,
        wavelength_shift_bounds=(-2.0e-4, 2.0e-4),
    )

    recovered = result.provenance["segmentation"]["wavelength_shifts_micron"]
    np.testing.assert_allclose(recovered, shifts, atol=3.0e-5)


def test_correct_arrays_exposes_global_wavelength_polynomial():
    wavelength = np.linspace(2.31, 2.36, 700)
    line_list = LineList.demo_near_ir()
    names, basis = optical_depth_basis(wavelength, line_list)
    x = 2.0 * (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    coefficients = np.array([1.0e-5, 4.0e-5])
    flux = transmission_from_basis(
        names,
        _shift_basis(wavelength, basis, coefficients[0] + coefficients[1] * x),
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        fit_wavelength_polynomial=True,
        wavelength_polynomial_order=1,
        wavelength_shift_bounds=(-1.0e-4, 1.0e-4),
    )

    assert result.success
    np.testing.assert_allclose(result.wavelength_coefficients, coefficients, atol=1.0e-5)


def test_correct_arrays_auto_selects_wavelength_dependent_pixel_shift():
    pixel = np.arange(6000, dtype=float)
    wavelength = 2.30 + 6.0e-6 * pixel + 3.0e-10 * pixel**2
    line_list = LineList(
        wavelength=np.array(
            [2.303, 2.308, 2.315, 2.322, 2.329, 2.337, 2.345]
        ),
        strength=np.array([0.006, 0.008, 0.005, 0.009, 0.007, 0.008, 0.006]),
        sigma=np.full(7, 1.2e-5),
        gamma=np.full(7, 6.0e-6),
        species=np.full(7, "H2O"),
    )
    x = 2.0 * (wavelength - np.mean((wavelength[0], wavelength[-1]))) / np.ptp(
        wavelength
    )
    true_coefficients = np.array([0.25, 1.15])
    species_names, basis = optical_depth_basis(wavelength, line_list)
    flux = transmission_from_basis(
        species_names,
        _shift_basis(
            wavelength,
            basis,
            (true_coefficients[0] + true_coefficients[1] * x)
            * np.gradient(wavelength),
        ),
        species_scales={"H2O": 1.2},
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        auto_segment=False,
        high_resolution_grid=False,
        lsf_sigma_pixels=0.0,
        fit_lsf_sigma=False,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
    )

    alignment = result.provenance["wavelength_alignment"]
    assert result.success
    assert alignment["selected_model"] == "linear_pixel_trend"
    assert alignment["coefficient_unit"] == "pixel"
    np.testing.assert_allclose(
        result.wavelength_coefficients,
        true_coefficients,
        atol=0.15,
    )


@pytest.mark.parametrize(
    ("true_pixel_shift", "expected_model"),
    [
        (0.0, "none"),
        (1.2, "constant_pixel_shift"),
        (-1.2, "constant_pixel_shift"),
    ],
)
def test_correct_arrays_auto_selects_simplest_supported_wavelength_model(
    true_pixel_shift,
    expected_model,
):
    pixel = np.arange(4000, dtype=float)
    wavelength = 2.30 + 9.0e-6 * pixel + 4.0e-10 * pixel**2
    line_list = LineList(
        wavelength=np.array([2.304, 2.312, 2.320, 2.328, 2.336]),
        strength=np.array([0.006, 0.008, 0.005, 0.009, 0.007]),
        sigma=np.full(5, 1.2e-5),
        gamma=np.full(5, 6.0e-6),
        species=np.full(5, "H2O"),
    )
    species_names, basis = optical_depth_basis(wavelength, line_list)
    flux = transmission_from_basis(
        species_names,
        _shift_basis(
            wavelength,
            basis,
            true_pixel_shift * np.gradient(wavelength),
        ),
        species_scales={"H2O": 1.2},
    )

    result = correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        auto_segment=False,
        high_resolution_grid=False,
        lsf_sigma_pixels=0.0,
        fit_lsf_sigma=False,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
    )

    alignment = result.provenance["wavelength_alignment"]
    assert alignment["selected_model"] == expected_model
    if true_pixel_shift:
        np.testing.assert_allclose(
            result.wavelength_coefficients,
            [true_pixel_shift],
            atol=0.12,
        )


def test_correct_file_writes_outputs(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 300)
    flux = transmission_model(wavelength, LineList.demo_near_ir(), ModelConfig())
    input_path = tmp_path / "spectrum.txt"
    output_path = tmp_path / "corrected.txt"
    product_path = tmp_path / "product.ecsv"
    np.savetxt(input_path, np.column_stack([wavelength, flux]))

    result = correct_file(
        input_path,
        output_path,
        wavelength_medium="vacuum",
        demo_line_list=True,
        continuum_order=0,
        product_path=product_path,
    )

    assert result.success
    assert output_path.exists()
    assert product_path.exists()


def test_standalone_result_writers_save_txt_and_ecsv(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 300)
    flux = transmission_model(wavelength, LineList.demo_near_ir(), ModelConfig())
    result = correct_arrays(
        wavelength,
        flux,
        line_list=LineList.demo_near_ir(),
        continuum_order=0,
    )
    txt_path = tmp_path / "corrected_spectrum.txt"
    ecsv_path = tmp_path / "fit_product.ecsv"

    assert save_corrected_txt(result, txt_path) == txt_path
    assert save_fit_product_ecsv(result, ecsv_path) == ecsv_path

    corrected = np.loadtxt(txt_path)
    product = Table.read(ecsv_path, format="ascii.ecsv")
    np.testing.assert_allclose(corrected[:, 0], result.corrected.wavelength)
    np.testing.assert_allclose(corrected[:, 1], result.corrected.flux)
    assert {
        "wavelength",
        "flux",
        "model_flux",
        "continuum",
        "transmission",
        "corrected_flux",
        "input_mask",
        "corrected_mask",
    } <= set(product.colnames)
    assert product.meta["fit_success"] is True


def test_correct_file_refuses_implicit_synthetic_line_data(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 20)
    input_path = tmp_path / "spectrum.txt"
    np.savetxt(input_path, np.column_stack([wavelength, np.ones_like(wavelength)]))

    with pytest.raises(ValueError, match="no molecular line data supplied"):
        correct_file(input_path, wavelength_medium="vacuum", aer_catalog=None)


def test_correct_file_with_hitran_nm_input(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    input_path = tmp_path / "spectrum_nm.txt"
    output_path = tmp_path / "corrected.txt"
    hitran_path.write_text(_hitran_row() + "\n")
    center_nm = (1.0e4 / 4320.0) * 1000.0
    wavelength_nm = np.linspace(center_nm - 1.0, center_nm + 1.0, 80)
    np.savetxt(input_path, np.column_stack([wavelength_nm, np.ones_like(wavelength_nm)]))

    result = correct_file(
        input_path,
        output_path,
        wavelength_unit="nm",
        wavelength_medium="vacuum",
        hitran_par=hitran_path,
        mixing_ratios={"H2O": 1.0e-5},
        allow_default_observatory=True,
        continuum_order=0,
    )

    assert result.success
    assert output_path.exists()


def test_hitran_selection_uses_dynamic_lblrtm_margin(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    hitran_path.write_text(_hitran_row() + "\n")
    wavenumber = np.linspace(4280.0, 4284.0, 5)
    spectrum = Spectrum(wavelength=1.0e4 / wavenumber, flux=np.ones(wavenumber.size))

    with pytest.raises(ValueError, match="no HITRAN lines"):
        _resolve_line_list(
            spectrum,
            line_list=None,
            line_list_path=None,
            hitran_par=hitran_path,
            hitran_species=None,
            hitran_min_strength=None,
            hitran_max_lines=None,
            line_cutoff_cm=None,
            line_wing_mode="full",
            lblrtm_sample=4.0,
            lblrtm_alfal0=0.04,
            lblrtm_hwf3=64.0,
        )
    dynamic_margin = _resolve_line_list(
        spectrum,
        line_list=None,
        line_list_path=None,
        hitran_par=hitran_path,
        hitran_species=None,
        hitran_min_strength=None,
        hitran_max_lines=None,
        line_cutoff_cm=None,
        line_wing_mode="lblrtm_dynamic",
        lblrtm_sample=4.0,
        lblrtm_alfal0=0.04,
        lblrtm_hwf3=64.0,
    )

    assert dynamic_margin.wavelength.size == 1


def test_correct_file_converts_air_wavelengths_before_hitran_fit(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    input_path = tmp_path / "spectrum_air_nm.txt"
    output_path = tmp_path / "corrected.txt"
    hitran_path.write_text(_hitran_row() + "\n")
    center_nm = (1.0e4 / 4320.0) * 1000.0
    wavelength_vacuum_nm = np.linspace(center_nm - 0.1, center_nm + 0.1, 80)
    wavelength_air_nm = vacuum_to_air_wavelength(wavelength_vacuum_nm, unit="nm")
    np.savetxt(input_path, np.column_stack([wavelength_air_nm, np.ones_like(wavelength_air_nm)]))

    result = correct_file(
        input_path,
        output_path,
        wavelength_unit="nm",
        wavelength_medium="air",
        hitran_par=hitran_path,
        mixing_ratios={"H2O": 1.0e-5},
        allow_default_observatory=True,
        continuum_order=0,
    )

    assert result.success
    assert result.spectrum.wavelength_medium == "vacuum"
    np.testing.assert_allclose(result.spectrum.wavelength, wavelength_vacuum_nm * 1.0e-3, rtol=0, atol=1e-10)
    assert output_path.exists()


def test_correct_file_uses_atmosphere_table(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    input_path = tmp_path / "spectrum.txt"
    output_path = tmp_path / "corrected.txt"
    atmosphere_path = tmp_path / "atmosphere.ecsv"
    hitran_path.write_text(_hitran_row() + "\n")
    center = 1.0e4 / 4320.0
    wavelength = np.linspace(center - 0.001, center + 0.001, 80)
    np.savetxt(input_path, np.column_stack([wavelength, np.ones_like(wavelength)]))

    atmosphere = Table()
    atmosphere["pressure_atm"] = [0.75]
    atmosphere["temperature_k"] = [280.0]
    atmosphere["path_length_m"] = [1200.0]
    atmosphere["mix_H2O"] = [1.0e-5]
    atmosphere.write(atmosphere_path, format="ascii.ecsv")

    result = correct_file(
        input_path,
        output_path,
        wavelength_medium="vacuum",
        hitran_par=hitran_path,
        atmosphere_table=atmosphere_path,
        continuum_order=0,
    )

    assert result.success
    assert output_path.exists()


def test_correct_file_stops_when_wavelength_medium_is_unknown(tmp_path):
    input_path = tmp_path / "spectrum.txt"
    wavelength = np.linspace(2.31, 2.36, 20)
    np.savetxt(input_path, np.column_stack([wavelength, np.ones_like(wavelength)]))

    with pytest.raises(
        ValueError,
        match="stopped the correction because wavelength_medium was not provided",
    ):
        correct_file(input_path, demo_line_list=True, continuum_order=0)


@pytest.mark.parametrize(
    ("ctype", "expected_medium"),
    (("AWAV", "air"), ("AWAV-TAB", "air"), ("WAVE", "vacuum"), ("WAVE-TAB", "vacuum")),
)
def test_fits_wcs_infers_wavelength_medium(ctype, expected_medium):
    assert _infer_wavelength_medium_from_header({"CTYPE1": ctype}) == expected_medium


def test_explicit_fits_metadata_infers_wavelength_medium():
    assert (
        _infer_wavelength_medium_from_header({"PYMOLFIT WAVE": "air micron"})
        == "air"
    )
    assert _infer_wavelength_medium_from_header({"VACUUM": True}) == "vacuum"
    assert _infer_wavelength_medium_from_header({"VACUUM": False}) == "air"
    assert _infer_wavelength_medium_from_header({"SPECSYS": "BARYCENT"}) is None


def test_espresso_wave_column_infers_vacuum_wavelength():
    header = {"INSTRUME": "ESPRESSO", "TTYPE1": "WAVE", "TUNIT1": "angstrom"}

    assert _infer_wavelength_medium_from_header(header) == "vacuum"


def test_espresso_wave_air_column_infers_air_wavelength():
    header = {"INSTRUME": "ESPRESSO", "TTYPE1": "WAVE_AIR", "TUNIT1": "angstrom"}

    assert _infer_wavelength_medium_from_header(header) == "air"


def test_generic_wave_table_column_does_not_assume_a_medium():
    header = {"INSTRUME": "OTHER", "TTYPE1": "WAVE", "TUNIT1": "angstrom"}

    assert _infer_wavelength_medium_from_header(header) is None


def test_conflicting_fits_wavelength_medium_is_rejected():
    with pytest.raises(ValueError, match="conflicting FITS metadata"):
        _infer_wavelength_medium_from_header(
            {"CTYPE1": "AWAV", "WAVEMED": "vacuum"}
        )


@pytest.mark.parametrize("ctype", ("AWAV", "WAVE"))
def test_correct_file_uses_fits_wavelength_medium_when_omitted(tmp_path, ctype):
    input_path = tmp_path / f"spectrum_{ctype.lower()}.fits"
    wavelength = np.linspace(2.31, 2.36, 300)
    flux = transmission_model(wavelength, LineList.demo_near_ir(), ModelConfig())
    primary = fits.PrimaryHDU()
    primary.header["CTYPE1"] = ctype
    spectrum_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="wavelength", format="D", unit="um", array=wavelength),
            fits.Column(name="flux", format="D", array=flux),
        ]
    )
    fits.HDUList([primary, spectrum_hdu]).writeto(input_path)

    result = correct_file(input_path, demo_line_list=True, continuum_order=0)

    expected = (
        air_to_vacuum_wavelength(wavelength)
        if ctype == "AWAV"
        else wavelength
    )
    np.testing.assert_allclose(result.spectrum.wavelength, expected)


def test_workflow_preslants_internal_atmosphere_once():
    atmosphere = _make_atmosphere(
        atmosphere_table=None,
        atmosphere_mode="single",
        atmosphere_header=None,
        mipas_profile="equ",
        gdas_profile=None,
        gdas_mode="average",
        gdas_cache_dir=None,
        gdas_download_timeout_s=15.0,
        observatory_latitude_deg=None,
        observatory_longitude_deg=None,
        observatory_altitude_m=None,
        allow_default_observatory=False,
        airmass=2.0,
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1200.0,
        pwv_mm=None,
        relative_humidity_percent=None,
        mixing_ratios={"H2O": 1.0e-5},
    )

    np.testing.assert_allclose(atmosphere.layers[0].path_length_m, 2400.0)
    np.testing.assert_allclose(atmosphere.layers[0].vertical_path_length_m, 1200.0)


def test_workflow_mipas_gdas_atmosphere_uses_header_metadata():
    header = {
        "MJD-OBS": 59581.2,
        "ESO TEL AIRM START": 1.1,
        "ESO TEL AIRM END": 1.3,
        "ESO TEL GEOELEV": 2635.0,
        "ESO TEL GEOLAT": -24.6,
        "ESO TEL GEOLON": -70.4,
        "ESO TEL AMBI PRES START": 743.0,
        "ESO TEL AMBI TEMP": 8.0,
        "ESO TEL AMBI RHUM": 25.0,
    }

    atmosphere = _make_atmosphere(
        atmosphere_table=None,
        atmosphere_mode="mipas_gdas",
        atmosphere_header=header,
        mipas_profile="equ",
        gdas_profile=None,
        gdas_mode="average",
        gdas_cache_dir=None,
        gdas_download_timeout_s=15.0,
        observatory_latitude_deg=None,
        observatory_longitude_deg=None,
        observatory_altitude_m=None,
        allow_default_observatory=False,
        airmass=1.0,
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1200.0,
        pwv_mm=None,
        relative_humidity_percent=None,
        mixing_ratios=None,
    )

    assert len(atmosphere.layers) > 40
    np.testing.assert_allclose(atmosphere.layers[0].pressure_atm, 743.0 / 1013.25, rtol=0.03)
    np.testing.assert_allclose(atmosphere.layers[0].temperature_k, 281.15, rtol=0.01)


def test_workflow_infers_barycentric_berv_initial_wavelength_shift():
    spectrum = Spectrum(wavelength=np.array([0.686, 0.688, 0.690]), flux=np.ones(3))
    header = {"SPECSYS": "BARYCENT", "ESO DRS BERV": -4.2}

    shift = _resolve_initial_wavelength_shift(spectrum, None, header)

    expected = np.nanmedian(spectrum.wavelength) * header["ESO DRS BERV"] / (SPEED_OF_LIGHT_M_PER_S / 1000.0)
    np.testing.assert_allclose(shift, expected)
    assert _resolve_initial_wavelength_shift(spectrum, 1.2e-5, header) == 1.2e-5
    assert _resolve_initial_wavelength_shift(spectrum, None, {"SPECSYS": "TOPOCENT"}) == 0.0


def test_workflow_reconstructs_missing_barycentric_velocity_from_fits_metadata():
    header = {
        "SPECSYS": "BARYCENT",
        "DATE-OBS": "2021-09-13T02:18:06.238",
        "RA": 311.29288,
        "DEC": -31.34092,
        "ESO TEL GEOLON": -70.7345,
        "ESO TEL GEOLAT": -29.2584,
        "ESO TEL GEOELEV": 2400.0,
    }

    velocity = _barycentric_velocity_from_header_km_s(header)

    assert velocity == pytest.approx(-20.72, abs=0.03)
    spectrum = Spectrum(wavelength=np.array([0.686, 0.688, 0.690]), flux=np.ones(3))
    expected = np.nanmedian(spectrum.wavelength) * velocity / (
        SPEED_OF_LIGHT_M_PER_S / 1000.0
    )
    np.testing.assert_allclose(
        _resolve_initial_wavelength_shift(spectrum, None, header),
        expected,
    )


def test_workflow_reconstructs_combined_spectrum_velocity_at_midpoint():
    header = {
        "SPECSYS": "BARYCENT",
        "DATE-OBS": "2024-08-26T00:04:44.603",
        "MJD-OBS": 60548.00329403,
        "MJD-END": 60548.3346547122,
        "NCOMBINE": 81,
        "RA": 311.291581,
        "DEC": -31.3434,
        "ESO TEL1 GEOLON": -70.4051,
        "ESO TEL1 GEOLAT": -24.6276,
        "ESO TEL1 GEOELEV": 2648.0,
    }

    velocity = _barycentric_velocity_from_header_km_s(header)

    assert velocity == pytest.approx(-13.77, abs=0.03)


def test_workflow_applies_molecfit_air_rv_order_before_vacuum_conversion():
    spectrum = Spectrum(
        wavelength=np.array([0.5889, 0.5890, 0.5891]),
        flux=np.ones(3),
        group_id=np.array([2, 2, 3]),
        wavelength_medium="air",
    )
    header = {"SPECSYS": "BARYCENT", "ESO DRS BERV": -7.5}

    converted = _spectrum_to_observatory_vacuum(spectrum, header)

    factor = (1.0 + 1.55e-8) * (1.0 + header["ESO DRS BERV"] / (SPEED_OF_LIGHT_M_PER_S / 1000.0))
    expected = air_to_vacuum_wavelength(spectrum.wavelength / factor)
    np.testing.assert_allclose(converted.wavelength, expected, rtol=0.0, atol=1.0e-15)
    assert converted.meta["observatory_frame_correction"] is True
    np.testing.assert_array_equal(converted.group_id, spectrum.group_id)
    assert _resolve_initial_wavelength_shift(converted, None, header) == 0.0

    ranges = ((0.58888, 0.58912), (0.58948, 0.58978))
    converted_ranges = _ranges_to_observatory_vacuum(ranges, "air", header)
    expected_ranges = air_to_vacuum_wavelength(np.asarray(ranges) / factor)
    np.testing.assert_allclose(converted_ranges, expected_ranges, rtol=0.0, atol=1.0e-15)


def test_workflow_converts_documented_heliocentric_product_to_observatory_frame():
    spectrum = Spectrum(
        wavelength=np.array([0.6860, 0.6870, 0.6880]),
        flux=np.ones(3),
        wavelength_medium="vacuum",
    )
    header = {
        "HELIOCNT": "Heliocentric correction applied.",
        "HELIOVEL": -21.2144,
    }

    converted = _spectrum_to_observatory_vacuum(spectrum, header)

    factor = (1.0 + 1.55e-8) * (
        1.0 + header["HELIOVEL"] / (SPEED_OF_LIGHT_M_PER_S / 1000.0)
    )
    np.testing.assert_allclose(converted.wavelength, spectrum.wavelength / factor)
    assert converted.meta["original_spectral_frame"] == "HELIOCENTRIC"
    assert converted.meta["observatory_frame_velocity_km_s"] == header["HELIOVEL"]


def test_unified_correct_rejects_ambiguous_input_routes():
    wavelength = np.linspace(2.31, 2.36, 40)
    flux = np.ones_like(wavelength)
    observation = Observation(wavelength_frame="observatory")

    with pytest.raises(ValueError, match="provide input_path"):
        correct()
    with pytest.raises(ValueError, match="not both"):
        correct(
            input_path="spectrum.fits",
            wavelength=wavelength,
            flux=flux,
            observation=observation,
            wavelength_medium="vacuum",
        )
    with pytest.raises(ValueError, match="both wavelength and flux"):
        correct(wavelength=wavelength, observation=observation)
    with pytest.raises(ValueError, match="requires observation"):
        correct(
            wavelength=wavelength,
            flux=flux,
            wavelength_medium="vacuum",
        )
    with pytest.raises(ValueError, match="wavelength_frame"):
        correct(
            wavelength=wavelength,
            flux=flux,
            wavelength_medium="vacuum",
            observation=Observation(),
        )
    with pytest.raises(ValueError, match="requires wavelength_medium"):
        correct(
            wavelength=wavelength,
            flux=flux,
            observation=observation,
        )


def test_unified_correct_declares_public_options_for_editor_completion():
    correct_parameters = inspect.signature(correct).parameters
    file_parameters = inspect.signature(correct_file).parameters
    array_parameters = inspect.signature(correct_arrays).parameters

    assert all(
        name in correct_parameters
        for name in file_parameters
        if name != "input_path"
    )
    assert all(name in correct_parameters for name in array_parameters)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in correct_parameters.values()
    )
    assert correct_parameters["continuum_order"].default == 1
    assert correct_parameters["continuum_order"].kind is inspect.Parameter.KEYWORD_ONLY


def test_unified_correct_file_and_array_routes_are_numerically_identical(tmp_path):
    input_path = tmp_path / "spectrum.txt"
    wavelength = np.linspace(2.31, 2.36, 500)
    line_list = LineList.demo_near_ir()
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.2, "O2": 0.8}),
    )
    np.savetxt(input_path, np.column_stack((wavelength, flux)))
    observation = Observation(
        time="2021-09-13T02:18:06.238",
        latitude_deg=-29.2584,
        longitude_deg=-70.7345,
        altitude_m=2400.0,
        airmass=1.2,
        resolving_power=100_000.0,
        wavelength_frame="barycentric",
        frame_velocity_km_s=-7.5,
        instrument="TEST",
    )
    options = {
        "line_list": line_list,
        "physical": False,
        "continuum_order": 0,
        "solve_continuum_linear": True,
        "lsf_sigma_pixels": 0.0,
        "fit_lsf_sigma": False,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "fit_lsf_lorentz_fwhm": False,
        "lsf_variable_width": False,
        "high_resolution_grid": False,
        "fit_wavelength_shift": False,
        "auto_segment": False,
    }

    file_result = correct(
        input_path=input_path,
        wavelength_medium="vacuum",
        observation=observation,
        **options,
    )
    array_result = correct(
        wavelength=wavelength,
        flux=flux,
        wavelength_medium="vacuum",
        observation=observation,
        **options,
    )

    np.testing.assert_allclose(
        file_result.spectrum.wavelength,
        array_result.spectrum.wavelength,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        file_result.model_flux,
        array_result.model_flux,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        file_result.transmission,
        array_result.transmission,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        file_result.corrected.flux,
        array_result.corrected.flux,
        rtol=0.0,
        atol=1.0e-12,
        equal_nan=True,
    )
    assert file_result.spectrum.meta["original_spectral_frame"] == "BARYCENTRIC"
    assert array_result.spectrum.meta["original_spectral_frame"] == "BARYCENTRIC"


def test_unified_correct_arrays_preserves_mask_and_writes_products(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 120)
    flux = np.ones_like(wavelength)
    mask = np.ones_like(wavelength, dtype=bool)
    mask[[3, 17]] = False
    output_path = tmp_path / "corrected.txt"
    product_path = tmp_path / "fit_product.ecsv"

    result = correct(
        wavelength=wavelength,
        flux=flux,
        mask=mask,
        wavelength_medium="vacuum",
        observation=Observation(wavelength_frame="observatory"),
        output_path=output_path,
        product_path=product_path,
        demo_line_list=True,
        physical=False,
        continuum_order=0,
        lsf_sigma_pixels=0.0,
        fit_lsf_sigma=False,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
        lsf_variable_width=False,
        high_resolution_grid=False,
        fit_wavelength_shift=False,
        auto_segment=False,
    )

    assert result.success
    np.testing.assert_array_equal(result.spectrum.mask, mask)
    assert output_path.exists()
    assert product_path.exists()
