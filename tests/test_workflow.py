import numpy as np
import pytest
from astropy.table import Table

from pymolfit import (
    LineList,
    ModelConfig,
    Spectrum,
    air_to_vacuum_wavelength,
    correct_arrays,
    correct_file,
    transmission_model,
    vacuum_to_air_wavelength,
)
from pymolfit.physics import SPEED_OF_LIGHT_M_PER_S
from pymolfit.workflow import (
    _barycentric_velocity_from_header_km_s,
    _make_atmosphere,
    _ranges_to_observatory_vacuum,
    _resolve_initial_wavelength_shift,
    _resolve_line_list,
    _split_spectrum,
    _spectrum_to_observatory_vacuum,
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


def _hitran_row(*, mol_id=1):
    row = (
        f"{mol_id:2d}"
        f"{1:1d}"
        f"{4320.0:12.6f}"
        f"{1.0e-24:10.3E}"
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
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        segmented.corrected.flux,
        unsegmented.corrected.flux,
        rtol=0.0,
        atol=2.0e-8,
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
        demo_line_list=True,
        continuum_order=0,
        product_path=product_path,
    )

    assert result.success
    assert output_path.exists()
    assert product_path.exists()


def test_correct_file_refuses_implicit_synthetic_line_data(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 20)
    input_path = tmp_path / "spectrum.txt"
    np.savetxt(input_path, np.column_stack([wavelength, np.ones_like(wavelength)]))

    with pytest.raises(ValueError, match="no molecular line data supplied"):
        correct_file(input_path, aer_catalog=None)


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
        hitran_par=hitran_path,
        atmosphere_table=atmosphere_path,
        continuum_order=0,
    )

    assert result.success
    assert output_path.exists()


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


def test_workflow_applies_molecfit_air_rv_order_before_vacuum_conversion():
    spectrum = Spectrum(
        wavelength=np.array([0.5889, 0.5890, 0.5891]),
        flux=np.ones(3),
        wavelength_medium="air",
    )
    header = {"SPECSYS": "BARYCENT", "ESO DRS BERV": -7.5}

    converted = _spectrum_to_observatory_vacuum(spectrum, header)

    factor = (1.0 + 1.55e-8) * (1.0 + header["ESO DRS BERV"] / (SPEED_OF_LIGHT_M_PER_S / 1000.0))
    expected = air_to_vacuum_wavelength(spectrum.wavelength / factor)
    np.testing.assert_allclose(converted.wavelength, expected, rtol=0.0, atol=1.0e-15)
    assert converted.meta["observatory_frame_correction"] is True
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
