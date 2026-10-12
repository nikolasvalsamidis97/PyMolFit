import inspect
from dataclasses import replace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from pymolfit import (
    AdvancedOptions,
    ConfigurationError,
    Observation,
    Spectrum,
    correct,
    correct_arrays,
    load_fit_product,
    load_spectrum,
    workflow,
)
from pymolfit.regions import _selector_header
from tests.helpers import demo_lines


def test_data_and_observation_are_no_longer_correction_arguments():
    moved = {"wavelength", "flux", "uncertainty", "mask", "group_id", "observation"}
    assert not (moved & inspect.signature(correct).parameters.keys())
    assert moved <= inspect.signature(Spectrum).parameters.keys()


@pytest.fixture
def observation():
    return Observation(
        time="2022-04-05T04:12:00",
        latitude_deg=-24.627,
        longitude_deg=-70.404,
        altitude_m=2635,
        airmass=1.25,
        resolving_power=100_000,
        wavelength_frame="barycentric",
        frame_velocity_km_s=-7.5,
        pwv_mm=2.4,
        metadata={"ORIGIN": "test"},
    )


def test_observation_dict_round_trip(observation):
    assert Observation.from_dict(observation.to_dict()).to_header() == observation.to_header()


@pytest.mark.parametrize("operation", ["with_flux", "to_unit", "to_air", "sorted", "to_vacuum"])
def test_spectrum_operations_preserve_observing_information(observation, operation):
    spectrum = Spectrum(
        np.array([590.0, 589.0]), np.ones(2), wavelength_unit="nm", observation=observation
    )
    if operation == "with_flux":
        modified = spectrum.with_flux(np.full(2, 2.0))
    elif operation == "to_unit":
        modified = spectrum.to_unit("angstrom")
    elif operation == "to_vacuum":
        modified = spectrum.to_air().to_vacuum()
    else:
        modified = getattr(spectrum, operation)()
    assert modified.observation.to_dict() == observation.to_dict()
    assert modified.observation.pwv_mm == 2.4


def test_observation_type_and_replacement_are_explicit(observation):
    with pytest.raises(TypeError, match="Observation object"):
        Spectrum(np.arange(2), np.ones(2), observation={})
    spectrum = Spectrum(np.arange(2), np.ones(2), observation=observation)
    changed = replace(spectrum, observation=replace(observation, airmass=1.6))
    assert changed.observation.airmass == 1.6
    assert changed.meta["observation"]["AIRMASS"] == 1.6
    assert spectrum.observation.airmass == 1.25


def test_old_observation_metadata_is_still_accepted(observation):
    spectrum = Spectrum(np.arange(2), np.ones(2), meta={"observation": observation.to_header()})
    assert spectrum.observation.to_header() == observation.to_header()


def test_prepared_spectrum_requires_observing_metadata():
    spectrum = Spectrum(np.arange(2), np.ones(2))
    with pytest.raises(ConfigurationError, match="Spectrum has no observing metadata"):
        correct(spectrum)
    with pytest.raises(ValueError, match="explicit wavelength_frame"):
        correct(replace(spectrum, observation=Observation()))


def test_attached_observation_drives_fit_and_selector_without_source_file(monkeypatch, observation):
    spectrum = Spectrum(
        np.linspace(0.58, 0.59, 12),
        np.ones(12),
        observation=observation,
        meta={"source": "deleted.fits"},
    )
    captured = {}

    def no_header(*a, **kw):
        pytest.fail("attached observing metadata must not be replaced by the source header")

    def model(spectrum, **kwargs):
        captured.update(kwargs)
        return spectrum

    monkeypatch.setattr(workflow, "_load_fits_header_if_available", no_header)
    monkeypatch.setattr(workflow, "_correct_spectrum_workflow", model)
    monkeypatch.setattr(workflow, "_finalize_correction", lambda result, **kw: result)
    assert correct(spectrum) is spectrum
    assert captured["atmosphere_header"] == observation.to_header()
    assert captured["pwv_mm"] == observation.pwv_mm
    assert _selector_header(spectrum) == observation.to_header()


@pytest.fixture
def model_options(monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_line_list", lambda *a, **kw: demo_lines())
    return AdvancedOptions(
        segment_size=0.1,
        fit_lsf_lorentz_fwhm=False,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_variable_width=False,
        lsf_sigma_pixels=0.0,
        continuum_order=0,
        wavelength_order=None,
        fit_lsf_sigma=False,
    )


def test_array_spectrum_matches_old_array_fit_and_round_trips(tmp_path, observation, model_options):
    wave = np.linspace(2310.0, 2360.0, 120)
    flux = np.ones(wave.size)
    uncertainty = np.full(wave.size, 0.01)
    mask = np.ones(wave.size, dtype=bool)
    mask[5] = False
    spectrum = Spectrum(
        wave,
        flux,
        uncertainty=uncertainty,
        mask=mask,
        observation=observation,
        wavelength_unit="nm",
        wavelength_medium="vacuum",
    )
    expected = correct_arrays(
        wave,
        flux,
        uncertainty=uncertainty,
        mask=mask,
        observation=observation,
        wavelength_unit="nm",
        wavelength_medium="vacuum",
        physical=False,
        auto_segment=False,
        high_resolution_grid=False,
        fit_lsf_lorentz_fwhm=False,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_variable_width=False,
        continuum_order=0,
        fit_wavelength_shift=False,
        fit_lsf_sigma=False,
        line_list=demo_lines(),
        lsf_sigma_pixels=model_options.lsf_sigma_pixels,
    )
    destination = tmp_path / "result.ecsv"
    actual = correct(spectrum, advanced=model_options, output_path=destination, report=False)
    np.testing.assert_allclose(actual.spectrum.wavelength, expected.spectrum.wavelength)
    np.testing.assert_allclose(actual.transmission, expected.transmission)
    np.testing.assert_allclose(actual.corrected.flux, expected.corrected.flux, equal_nan=True)
    np.testing.assert_array_equal(actual.fit_mask, expected.fit_mask)
    restored = load_fit_product(destination)
    assert restored.spectrum.observation.to_dict() == actual.spectrum.observation.to_dict()
    assert restored.corrected.observation.to_dict() == actual.corrected.observation.to_dict()
    assert restored.spectrum.observation.wavelength_frame == "observatory"
    reconverted = workflow._spectrum_to_observatory_vacuum(
        restored.spectrum, restored.spectrum.observation.to_header()
    )
    np.testing.assert_array_equal(reconverted.wavelength, restored.spectrum.wavelength)


def test_loaded_fits_keeps_header_when_file_is_gone(tmp_path, model_options):
    path = tmp_path / "input.fits"
    table = Table({"wavelength": np.linspace(2310.0, 2360.0, 120), "flux": np.ones(120)})
    table["wavelength"].unit = "nm"
    primary = fits.PrimaryHDU()
    primary.header["SPECSYS"] = "BARYCENT"
    primary.header["BERV"] = -7.5
    primary.header["AIRMASS"] = 1.25
    primary.header["SPEC_RES"] = 100_000
    extension = fits.BinTableHDU(table)
    extension.header["TUCD1"] = "em.wl"
    extension.header["AIRMASS"] = 1.8
    fits.HDUList([primary, extension]).writeto(path)
    loaded = load_spectrum(path, save_header=False)
    assert loaded.observation.metadata["AIRMASS"] == 1.25
    assert loaded.observation.metadata["BERV"] == -7.5
    options = {
        "advanced": model_options,
        "report": False,
    }
    direct = correct(path, **options)
    path.unlink()
    prepared = correct(loaded, **options)
    np.testing.assert_array_equal(prepared.spectrum.wavelength, direct.spectrum.wavelength)
    np.testing.assert_allclose(prepared.transmission, direct.transmission)
    np.testing.assert_allclose(prepared.corrected.flux, direct.corrected.flux, equal_nan=True)
