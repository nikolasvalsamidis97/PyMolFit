from __future__ import annotations

import numpy as np
import pytest

from pymolfit import (
    ConfigurationError,
    LineList,
    Observation,
    Spectrum,
    TheoreticalSpectrum,
    correct,
    correct_arrays,
    load_region_file,
    select_telluric_regions,
)
from pymolfit.model import ModelConfig, transmission_model
from pymolfit.theoretical import SPEED_OF_LIGHT_KM_S
from pymolfit.diagnostics import format_fit_summary


def _write_template(path, wavelength, flux) -> None:
    rows = "\n".join(
        f"{wave:.8f} {value:.12e}"
        for wave, value in zip(wavelength, flux, strict=True)
    )
    path.write_text(
        "# BT-NextGen test spectrum\n"
        "# teff = 8000 K\n"
        "# column 1: WAVELENGTH (ANGSTROM)\n"
        "# column 2: FLUX (ERG/CM2/S/A)\n"
        + rows
        + "\n"
    )


def test_theoretical_spectrum_reads_svo_ascii_metadata(tmp_path) -> None:
    path = tmp_path / "svo.dat"
    _write_template(
        path,
        np.array([4999.0, 5000.0, 5001.0]),
        np.array([1.0e-8, 0.8e-8, 1.0e-8]),
    )

    template = TheoreticalSpectrum(path, radial_velocity_kms=20.0, vsini_kms=130.0)

    assert template.wavelength_unit == "angstrom"
    assert template.wavelength_medium == "air"
    assert template.metadata["teff"] == 8000.0
    np.testing.assert_allclose(template.wavelength, [4999.0, 5000.0, 5001.0])


def test_theoretical_mask_applies_velocity_and_broadening(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    rest_wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    physical_continuum = 2.0e-8 * (1.0 + 2.0e-4 * (rest_wavelength - 5_000.0))
    absorption = 1.0 - 0.5 * np.exp(-0.5 * ((rest_wavelength - 5_000.0) / 0.08) ** 2)
    _write_template(path, rest_wavelength, physical_continuum * absorption)

    radial_velocity = 30.0
    frame_velocity = 10.0
    radial_factor = np.sqrt(
        (1.0 + radial_velocity / SPEED_OF_LIGHT_KM_S)
        / (1.0 - radial_velocity / SPEED_OF_LIGHT_KM_S)
    )
    frame_factor = 1.0 + frame_velocity / SPEED_OF_LIGHT_KM_S
    expected_center = 0.5 * radial_factor / frame_factor
    wavelength = np.linspace(expected_center - 0.001, expected_center + 0.001, 2_001)
    observed = Spectrum(wavelength, np.ones_like(wavelength))
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=radial_velocity,
        vsini_kms=20.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        fit_velocity_offset=False,
    )

    result = template.build_mask(
        observed,
        frame_correction_factor=frame_factor,
        resolving_power=100_000.0,
    )

    center_index = int(np.argmin(np.abs(wavelength - expected_center)))
    assert result.mask[center_index]
    assert result.diagnostics["masked_pixel_count"] > 1
    assert result.diagnostics["resolving_power"] == 100_000.0
    assert result.selection.exclude_ranges


def test_automatic_mask_padding_tracks_line_broadening(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    flux = 1.0 - 0.5 * np.exp(-0.5 * ((wavelength - 5_000.0) / 0.08) ** 2)
    _write_template(path, wavelength, flux)
    observed_wavelength = np.linspace(0.499, 0.501, 2_001)
    observed = Spectrum(observed_wavelength, np.ones_like(observed_wavelength))
    common = {
        "path": path,
        "radial_velocity_kms": 0.0,
        "vsini_kms": 120.0,
        "wavelength_medium": "vacuum",
        "mask_depth": 0.05,
        "fit_velocity_offset": False,
    }

    automatic = TheoreticalSpectrum(**common).build_mask(
        observed,
        resolving_power=100_000.0,
    )
    fixed = TheoreticalSpectrum(**common, mask_padding_kms=5.0).build_mask(
        observed,
        resolving_power=100_000.0,
    )

    automatic_width = sum(
        upper - lower for lower, upper in automatic.selection.exclude_ranges
    )
    fixed_width = sum(upper - lower for lower, upper in fixed.selection.exclude_ranges)
    assert automatic.diagnostics["mask_padding_mode"] == "auto"
    assert automatic.diagnostics["mask_padding_kms"] == pytest.approx(60.0, rel=0.01)
    assert (
        automatic.diagnostics["masked_pixel_count"]
        > automatic.diagnostics["core_pixel_count"]
    )
    assert automatic_width > fixed_width


def test_selector_opens_with_editable_theoretical_exclusions(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    flux = 1.0 - 0.5 * np.exp(
        -0.5 * ((wavelength - 5_000.0) / 0.08) ** 2
    )
    _write_template(path, wavelength, flux)
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=12.0,
        vsini_kms=30.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        mask_padding_kms=10.0,
        fit_velocity_offset=False,
    )
    observed = Spectrum(
        np.linspace(4_995.0, 5_005.0, 2_001),
        np.ones(2_001),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )

    selector = select_telluric_regions(
        observed,
        theoretical_spectrum=template,
        enable_theoretical_controls=True,
        show_telluric_lines=False,
        show=False,
    )

    assert selector.selection.exclude_ranges
    assert selector.stellar_mask_result is not None
    assert selector.stellar_rv_box.text == "12"
    assert selector.stellar_vsini_box.text == "30"
    assert selector.stellar_padding_box.text == "10"
    selector.close()


def test_selector_updates_stellar_width_and_saves_combined_file(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    flux = 1.0 - 0.5 * np.exp(
        -0.5 * ((wavelength - 5_000.0) / 0.08) ** 2
    )
    _write_template(path, wavelength, flux)
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        mask_padding_kms=2.0,
        fit_velocity_offset=False,
    )
    observed = Spectrum(
        np.linspace(4_995.0, 5_005.0, 2_001),
        np.ones(2_001),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )
    destination = tmp_path / "combined_regions.ecsv"
    selector = select_telluric_regions(
        observed,
        theoretical_spectrum=template,
        enable_theoretical_controls=True,
        output_path=destination,
        show_telluric_lines=False,
        show=False,
    )
    selector.add_region(5_003.0, 5_004.0, kind="fit")
    initial_width = sum(
        upper - lower for lower, upper in selector.selection.exclude_ranges
    )

    selector.stellar_padding_box.set_val("40")
    updated = selector.update_stellar_mask()
    updated_width = sum(upper - lower for lower, upper in updated)
    written = selector.save()
    saved = load_region_file(written)

    assert selector.theoretical_spectrum.mask_padding_kms == 40.0
    assert updated_width > initial_width
    assert saved.fit_ranges == ((5_003.0, 5_004.0),)
    assert saved.exclude_ranges
    selector.close()


def test_correct_arrays_uses_stellar_template_only_as_fit_exclusion(tmp_path) -> None:
    template_path = tmp_path / "stellar.dat"
    template_wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    stellar_profile = 1.0 - 0.35 * np.exp(
        -0.5 * ((template_wavelength - 5_000.0) / 0.10) ** 2
    )
    _write_template(template_path, template_wavelength, stellar_profile)
    template = TheoreticalSpectrum(
        template_path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        fit_velocity_offset=False,
        confidence_weighted_masking=False,
    )

    wavelength = np.linspace(0.499, 0.501, 1_001)
    line_list = LineList(
        wavelength=np.array([0.50055]),
        strength=np.array([0.015]),
        sigma=np.array([1.5e-5]),
        gamma=np.array([5.0e-6]),
        species=np.array(["H2O"]),
    )
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.2}),
    )
    stellar_absorption = 1.0 - 0.30 * np.exp(-0.5 * ((wavelength - 0.5) / 1.0e-5) ** 2)
    product_path = tmp_path / "stellar_regions.ecsv"

    result = correct_arrays(
        wavelength,
        transmission * stellar_absorption,
        line_list=line_list,
        theoretical_spectrum=template,
        stellar_mask_path=product_path,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_sigma=False,
        fit_lsf_lorentz_fwhm=False,
        fit_wavelength_shift=False,
        high_resolution_grid=False,
        auto_segment=False,
    )

    stellar_index = int(np.argmin(np.abs(result.spectrum.wavelength - 0.5)))
    telluric_index = int(np.argmin(np.abs(result.spectrum.wavelength - 0.50055)))
    assert result.fit_mask is not None
    assert not result.fit_mask[stellar_index]
    assert np.isfinite(result.transmission[stellar_index])
    assert result.transmission[telluric_index] < 1.0
    assert "stellar_template" in result.provenance
    summary = format_fit_summary(result)
    assert "stellar template:" in summary
    assert "stellar exclusions:" in summary
    saved = load_region_file(product_path)
    assert saved.wavelength_unit == "micron"
    assert saved.exclude_ranges


def test_correct_arrays_uses_opt_in_stellar_confidence_weights_without_binary_exclusion(
    tmp_path,
) -> None:
    template_path = tmp_path / "stellar.dat"
    template_wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    stellar_profile = 1.0 - 0.4 * np.exp(
        -0.5 * ((template_wavelength - 5_000.0) / 0.10) ** 2
    )
    _write_template(template_path, template_wavelength, stellar_profile)
    template = TheoreticalSpectrum(
        template_path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        fit_velocity_offset=False,
        confidence_weighted_masking=True,
        confidence_weight_floor=0.05,
    )
    wavelength = np.linspace(0.499, 0.501, 1_001)
    line_list = LineList(
        wavelength=np.array([0.50055]),
        strength=np.array([0.015]),
        sigma=np.array([1.5e-5]),
        gamma=np.array([5.0e-6]),
        species=np.array(["H2O"]),
    )
    transmission = transmission_model(wavelength, line_list, ModelConfig())

    result = correct_arrays(
        wavelength,
        transmission,
        line_list=line_list,
        theoretical_spectrum=template,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_sigma=False,
        fit_lsf_lorentz_fwhm=False,
        fit_wavelength_shift=False,
        high_resolution_grid=False,
        auto_segment=False,
    )

    stellar_index = int(np.argmin(np.abs(result.spectrum.wavelength - 0.5)))
    assert result.fit_mask is not None and result.fit_mask[stellar_index]
    assert result.fit_weights is not None
    assert result.fit_weights[stellar_index] < 1.0
    np.testing.assert_allclose(
        result.to_table()["fit_weight"],
        result.fit_weights,
    )


def test_joint_stellar_model_is_opt_in_on_unified_correct_only(tmp_path) -> None:
    import inspect

    from pymolfit import correct_file

    wavelength = np.linspace(0.499, 0.501, 401)
    spectrum = Spectrum(wavelength, np.ones_like(wavelength))

    assert inspect.signature(correct).parameters["joint_stellar_model"].default is False
    assert "joint_stellar_model" not in inspect.signature(correct_arrays).parameters
    assert "joint_stellar_model" not in inspect.signature(correct_file).parameters
    with pytest.raises(ConfigurationError, match="requires theoretical_spectrum"):
        correct(
            spectrum=spectrum,
            line_list=LineList.empty_hitran(),
            joint_stellar_model=True,
            report=False,
        )


def test_unified_correct_runs_joint_stellar_model_for_array_input(tmp_path) -> None:
    template_path = tmp_path / "stellar.dat"
    template_wavelength = np.linspace(4_995.0, 5_005.0, 2_001)
    stellar_profile = 1.0 - 0.25 * np.exp(
        -0.5 * ((template_wavelength - 5_000.0) / 0.08) ** 2
    )
    _write_template(template_path, template_wavelength, stellar_profile)
    template = TheoreticalSpectrum(
        template_path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        fit_velocity_offset=False,
    )
    wavelength = np.linspace(0.4995, 0.5005, 801)
    line_list = LineList(
        wavelength=np.array([0.50012]),
        strength=np.array([2.0e-4]),
        sigma=np.array([8.0e-6]),
        gamma=np.array([2.0e-6]),
        species=np.array(["H2O"]),
    )
    atmosphere = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.1}),
    )
    star = np.interp(
        wavelength * 1.0e4,
        template_wavelength,
        stellar_profile,
    )

    result = correct(
        wavelength=wavelength,
        flux=star * atmosphere,
        wavelength_medium="vacuum",
        observation=Observation(wavelength_frame="observatory"),
        line_list=line_list,
        theoretical_spectrum=template,
        joint_stellar_model=True,
        allow_default_observatory=True,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_sigma=False,
        fit_lsf_lorentz_fwhm=False,
        fit_wavelength_shift=False,
        high_resolution_grid=False,
        auto_segment=False,
        report=False,
    )

    assert result.success
    assert result.stellar_model is not None
    assert result.provenance["stellar_template"]["joint_forward_model"] is True
    assert "stellar_model" in result.to_table().colnames


def test_unified_correct_runs_joint_stellar_model_for_file_input(tmp_path) -> None:
    template_path = tmp_path / "stellar.dat"
    template_wavelength = np.linspace(4_995.0, 5_005.0, 1_001)
    stellar_profile = 1.0 - 0.2 * np.exp(
        -0.5 * ((template_wavelength - 5_000.0) / 0.08) ** 2
    )
    _write_template(template_path, template_wavelength, stellar_profile)
    template = TheoreticalSpectrum(
        template_path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        fit_velocity_offset=False,
    )
    wavelength = np.linspace(0.4995, 0.5005, 401)
    line_list = LineList(
        wavelength=np.array([0.50012]),
        strength=np.array([2.0e-4]),
        sigma=np.array([8.0e-6]),
        gamma=np.array([2.0e-6]),
        species=np.array(["H2O"]),
    )
    atmosphere = transmission_model(wavelength, line_list, ModelConfig())
    star = np.interp(wavelength * 1.0e4, template_wavelength, stellar_profile)
    spectrum_path = tmp_path / "observed.txt"
    np.savetxt(spectrum_path, np.column_stack((wavelength, star * atmosphere)))

    result = correct(
        input_path=spectrum_path,
        wavelength_medium="vacuum",
        line_list=line_list,
        theoretical_spectrum=template,
        joint_stellar_model=True,
        allow_default_observatory=True,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_sigma=False,
        fit_lsf_lorentz_fwhm=False,
        fit_wavelength_shift=False,
        high_resolution_grid=False,
        auto_segment=False,
        report=False,
    )

    assert result.success
    assert result.stellar_model is not None
    assert result.provenance["stellar_template"]["joint_forward_model"] is True


def test_stellar_mask_can_be_exported_in_native_air_angstrom_coordinates(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    flux = 1.0 - 0.4 * np.exp(-0.5 * ((wavelength - 5_000.0) / 0.1) ** 2)
    _write_template(path, wavelength, flux)
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        fit_velocity_offset=False,
    )
    native = Spectrum(
        np.linspace(4_995.0, 5_005.0, 1_001),
        np.ones(1_001),
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )
    mask = template.build_mask(native.to_unit("micron").to_vacuum())

    selection = mask.selection_for_spectrum(native)

    assert selection.wavelength_unit == "angstrom"
    assert selection.wavelength_medium == "air"
    assert selection.exclude_ranges


def test_theoretical_mask_refines_small_residual_velocity(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    rest = np.linspace(4_995.0, 5_005.0, 2_001)
    model = 1.0 - 0.5 * np.exp(-0.5 * ((rest - 5_000.0) / 0.08) ** 2)
    _write_template(path, rest, model)
    residual_velocity = 6.0
    factor = np.sqrt(
        (1.0 + residual_velocity / SPEED_OF_LIGHT_KM_S)
        / (1.0 - residual_velocity / SPEED_OF_LIGHT_KM_S)
    )
    observed_wavelength = np.linspace(0.4995, 0.5005, 2_001)
    rest_at_observed = observed_wavelength / factor
    observed_flux = np.interp(rest_at_observed * 1.0e4, rest, model)
    observed = Spectrum(observed_wavelength, observed_flux)
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        velocity_search_kms=12.0,
    )

    result = template.build_mask(observed)

    assert result.diagnostics["residual_velocity_kms"] == pytest.approx(
        residual_velocity,
        abs=0.75,
    )


def test_confidence_weighting_preserves_pixels_and_downweights_features(
    tmp_path,
) -> None:
    path = tmp_path / "stellar.dat"
    wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    flux = 1.0 - 0.6 * np.exp(-0.5 * ((wavelength - 5_000.0) / 0.1) ** 2)
    _write_template(path, wavelength, flux)
    observed = Spectrum(
        np.linspace(0.499, 0.501, 2_001),
        np.ones(2_001),
    )
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        fit_velocity_offset=False,
        confidence_weighted_masking=True,
        confidence_weight_floor=0.1,
    )

    result = template.build_mask(observed, resolving_power=100_000.0)

    assert result.fit_weights is not None
    assert np.nanmin(result.fit_weights) == pytest.approx(0.1)
    assert np.nanmax(result.fit_weights) == pytest.approx(1.0)
    assert result.diagnostics["confidence_weighted_masking"] is True


def test_default_theoretical_masking_uses_binary_exclusions(tmp_path) -> None:
    path = tmp_path / "stellar.dat"
    wavelength = np.linspace(4_990.0, 5_010.0, 2_001)
    flux = 1.0 - 0.6 * np.exp(-0.5 * ((wavelength - 5_000.0) / 0.1) ** 2)
    _write_template(path, wavelength, flux)
    observed = Spectrum(
        np.linspace(0.499, 0.501, 2_001),
        np.ones(2_001),
    )
    template = TheoreticalSpectrum(
        path,
        radial_velocity_kms=0.0,
        vsini_kms=0.0,
        wavelength_medium="vacuum",
        mask_depth=0.05,
        fit_velocity_offset=False,
    )

    result = template.build_mask(observed, resolving_power=100_000.0)

    assert result.fit_weights is None
    assert result.selection.exclude_ranges
    assert result.diagnostics["confidence_weighted_masking"] is False
