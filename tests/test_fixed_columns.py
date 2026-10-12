"""Public fixed-column controls retain absorption and do not alter defaults."""

import numpy as np
import pytest

from pymolfit import (
    AdvancedOptions,
    LineList,
    ModelConfig,
    Observation,
    Spectrum,
    correct,
    transmission_model,
    workflow,
)
from tests.helpers import write_spectrum_fits


@pytest.fixture
def case(monkeypatch):
    wave = np.linspace(2.31, 2.35, 900)
    lines = LineList(
        wavelength=np.array([2.322, 2.339]),
        strength=np.array([2e-5, 3e-5]),
        sigma=np.array([8e-5, 8e-5]),
        gamma=np.array([1e-5, 1e-5]),
        species=np.array(["H2O", "CO2"]),
    )
    flux = transmission_model(wave, lines, ModelConfig(species_scales={"H2O": 0.8, "CO2": 1.13}))
    observation = Observation(
        time="2022-04-05T00:02:35",
        latitude_deg=-24.6268,
        longitude_deg=-70.4045,
        altitude_m=2648,
        airmass=1,
        wavelength_frame="observatory",
    )
    monkeypatch.setattr(workflow, "_resolve_line_list", lambda *a, **kw: lines)
    options = {
        "continuum_order": 0,
        "solve_continuum_linear": True,
        "automatic_fit_regions": False,
        "lsf_sigma_pixels": 0,
        "fit_lsf_sigma": False,
        "lsf_lorentz_fwhm_pixels": 0,
        "fit_lsf_lorentz_fwhm": False,
        "lsf_variable_width": False,
        "wavelength_order": None,
        "segment_size": 0.1,
        "loss": "linear",
    }
    spectrum = Spectrum(wave, flux, wavelength_medium="vacuum", observation=observation)
    return spectrum, options


@pytest.mark.parametrize("route", ("spectrum", "file"))
def test_public_routes_fix_column_but_keep_its_absorption(case, route, tmp_path):
    spectrum, options = case
    wave, flux = spectrum.wavelength, spectrum.flux
    source = (
        spectrum if route == "spectrum" else write_spectrum_fits(spectrum, tmp_path / "input.fits")
    )
    result = correct(
        source,
        advanced=AdvancedOptions(**options, fixed_species_scales={"CO2": 1.13}),
        report=False,
    )
    assert result.success
    assert result.species_scales["CO2"] == 1.13
    assert "log_scale:CO2" not in result.parameter_names
    assert "log_scale:H2O" in result.parameter_names
    assert result.species_scales["H2O"] == pytest.approx(0.8, rel=1e-4)
    assert np.min(result.transmission[(wave > 2.338) & (wave < 2.340)]) < 0.95
    np.testing.assert_allclose(result.model_flux, flux, atol=2e-5)


@pytest.mark.parametrize(
    "scales", ({"CO2": 0}, {"CO2": -1}, {"CO2": np.nan}, {"CO2": np.inf}, {"TYPO": 1})
)
def test_invalid_fixed_columns_fail_explicitly(case, scales):
    spectrum, options = case
    with pytest.raises(ValueError, match=r"finite and positive|not present"):
        correct(
            spectrum,
            advanced=AdvancedOptions(**options, fixed_species_scales=scales),
            report=False,
        )


def test_omitting_fixed_columns_keeps_both_species_free(case):
    spectrum, options = case
    result = correct(spectrum, advanced=AdvancedOptions(**options), report=False)
    assert {"log_scale:H2O", "log_scale:CO2"}.issubset(result.parameter_names)
    assert result.species_scales["CO2"] == pytest.approx(1.13, rel=1e-4)


def test_fixed_columns_survive_segmentation(case):
    spectrum, options = case
    flux = spectrum.flux
    options.update(segment_size=0.01)
    result = correct(
        spectrum,
        advanced=AdvancedOptions(**options, fixed_species_scales={"CO2": 1.13}),
        report=False,
    )
    assert result.success
    assert result.species_scales["CO2"] == 1.13
    assert "log_scale:CO2" not in result.parameter_names
    assert result.species_scales["H2O"] == pytest.approx(0.8, rel=1e-4)
    np.testing.assert_allclose(result.model_flux, flux, atol=2e-5)


def test_all_columns_can_be_fixed(case):
    spectrum, options = case
    flux = spectrum.flux
    result = correct(
        spectrum,
        advanced=AdvancedOptions(**options, fixed_species_scales={"H2O": 0.8, "CO2": 1.13}),
        report=False,
    )
    assert result.success
    assert not any(name.startswith("log_scale:") for name in result.parameter_names)
    np.testing.assert_allclose(result.model_flux, flux, atol=2e-5)
