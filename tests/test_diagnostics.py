import numpy as np

from pymolfit import (
    LineList,
    ModelConfig,
    correct_arrays,
    correct_file,
    format_fit_summary,
    transmission_model,
)


def _demo_result():
    wavelength = np.linspace(2.31, 2.36, 160)
    line_list = LineList.demo_near_ir()
    flux = transmission_model(wavelength, line_list, ModelConfig())
    return correct_arrays(
        wavelength,
        flux,
        line_list=line_list,
        continuum_order=0,
        auto_segment=False,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_wavelength_shift=False,
        fit_ranges=((2.31, 2.35),),
        exclude_ranges=((2.32, 2.321),),
    )


def test_format_fit_summary_reports_effective_configuration():
    result = _demo_result()
    report = format_fit_summary(result, input_path="spectrum.fits")

    assert report.startswith("PyMolFit effective fit configuration")
    assert "input: spectrum.fits" in report
    assert "wavelength frame: observatory-frame vacuum" in report
    assert "fit ranges: custom (1 intervals)" in report
    assert "fit ranges (observatory vacuum micron)" not in report
    assert "excluded ranges (observatory vacuum micron): custom (1 intervals)" in report
    assert "continuum solver: linear (requested=auto, fallback=no)" in report
    assert "Gaussian LSF sigma: 0 pixels (source=user, fitted=no" in report
    assert "wavelength alignment: none" in report
    assert "line wings: lblrtm_panel" in report
    assert "radiative-transfer airmass: 1" in report
    assert "success: yes" in report
    assert "residual alignment: median=" in report
    assert "continuum coefficients:" not in report
    assert "fit_quality" in result.provenance


def test_format_fit_summary_distinguishes_profile_and_multiplier_airmass():
    result = _demo_result()
    result.provenance["atmosphere_metadata"] = {"airmass": 1.23926543209877}

    report = format_fit_summary(result)

    assert (
        "observation airmass: 1.2392654 (incorporated into atmospheric layer path lengths)"
    ) in report
    assert "additional opacity airmass multiplier: 1" in report
    assert "airmass used by fit" not in report


def test_correct_file_always_prints_effective_configuration(tmp_path, capsys):
    wavelength = np.linspace(2.31, 2.36, 160)
    flux = transmission_model(wavelength, LineList.demo_near_ir(), ModelConfig())
    input_path = tmp_path / "spectrum.txt"
    np.savetxt(input_path, np.column_stack((wavelength, flux)))

    correct_file(
        input_path,
        wavelength_medium="vacuum",
        demo_line_list=True,
        continuum_order=0,
        auto_segment=False,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_wavelength_shift=False,
    )

    output = capsys.readouterr().out
    assert output.count("PyMolFit effective fit configuration") == 1
    assert f"input: {input_path}" in output
    assert "pixels: 160 total, 160 fitted" in output
    assert "species: CH4, CO2, H2O" in output
    assert "termination:" in output
