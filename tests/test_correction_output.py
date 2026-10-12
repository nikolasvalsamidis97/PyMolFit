import inspect

import numpy as np
import pytest
from astropy.table import Table

from pymolfit import (
    AdvancedOptions,
    Observation,
    Spectrum,
    correct,
    correct_file,
    load_fit_product,
    plot_fit,
    save_corrected_txt,
    workflow,
)
from pymolfit.cli import build_parser
from pymolfit.provenance import provenance_json
from tests.helpers import demo_lines


@pytest.mark.parametrize("route", ["spectrum", "file"])
def test_output_path_round_trips_full_result_for_every_input_route(tmp_path, route, monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_line_list", lambda *a, **kw: demo_lines())
    wavelength = np.linspace(2.31, 2.36, 120)
    flux = np.ones_like(wavelength)
    uncertainty = np.full_like(wavelength, 0.01)
    mask = np.ones(wavelength.size, dtype=bool)
    mask[7] = False
    spectrum = Spectrum(
        wavelength,
        flux,
        uncertainty=uncertainty,
        mask=mask,
        observation=Observation(wavelength_frame="observatory"),
    )
    if route == "spectrum":
        inputs = {"input_path": spectrum}
    else:
        from tests.helpers import write_spectrum_fits

        inputs = {"input_path": write_spectrum_fits(spectrum, tmp_path / "input.fits")}
    destination = tmp_path / "corrected" / "result.ecsv"

    result = correct(
        **inputs,
        wavelength_medium="vacuum",
        advanced=AdvancedOptions(
            lsf_sigma_pixels=0.0,
            continuum_order=0,
            fit_lsf_sigma=False,
            wavelength_order=None,
            lsf_lorentz_fwhm_pixels=0.0,
            fit_lsf_lorentz_fwhm=False,
            lsf_variable_width=False,
            segment_size=0.1,
        ),
        output_path=destination,
        report=False,
    )
    restored = load_fit_product(destination)
    np.testing.assert_allclose(restored.spectrum.flux, result.spectrum.flux, equal_nan=True)
    np.testing.assert_allclose(restored.corrected.flux, result.corrected.flux, equal_nan=True)
    np.testing.assert_allclose(
        restored.corrected.uncertainty, result.corrected.uncertainty, equal_nan=True
    )
    np.testing.assert_allclose(restored.transmission, result.transmission, equal_nan=True)
    np.testing.assert_array_equal(restored.spectrum.valid, result.spectrum.valid)
    np.testing.assert_array_equal(restored.corrected.valid, result.corrected.valid)
    np.testing.assert_array_equal(restored.fit_mask, result.fit_mask)
    assert restored.species_scales == result.species_scales
    assert provenance_json(restored.provenance) == provenance_json(result.provenance)

    compact = save_corrected_txt(result, tmp_path / "compact.txt")
    np.testing.assert_allclose(np.loadtxt(compact)[:, 1], result.corrected.flux, equal_nan=True)
    assert "corrected_flux" in Table.read(destination, format="ascii.ecsv").colnames

    import matplotlib.pyplot as plt

    figure = plot_fit(destination, show=False)
    assert len(figure.axes) == 2
    plt.close(figure)


@pytest.mark.parametrize("function", [correct, correct_file])
def test_correction_signature_has_one_output_destination(function):
    parameters = inspect.signature(function).parameters
    assert "output_path" in parameters
    if function is correct:
        assert not hasattr(AdvancedOptions(), "output_format")
        assert "output_format" not in parameters
    else:
        assert parameters["output_format"].default == "ascii.ecsv"
    assert "product_path" not in parameters
    assert "product_format" not in parameters


@pytest.mark.parametrize("function", [correct, correct_file])
@pytest.mark.parametrize("removed", ["product_path", "product_format"])
def test_removed_output_options_fail_before_reading_input(function, removed):
    with pytest.raises(TypeError, match=removed):
        function("missing.fits", **{removed: "old-option"})


def test_cli_has_one_output_destination_and_explicit_format():
    parser = build_parser()
    args = parser.parse_args(["fit", "input.fits", "result.ecsv"])
    assert args.output_format == "ascii.ecsv"
    assert not hasattr(args, "product")
    args = parser.parse_args(["fit", "input.fits", "result.fits", "--output-format", "fits"])
    assert args.output_format == "fits"
    with pytest.raises(SystemExit):
        parser.parse_args(["fit", "input.fits", "result.ecsv", "--product", "other.ecsv"])
