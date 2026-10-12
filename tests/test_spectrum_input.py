import inspect
from typing import get_args, get_type_hints

import numpy as np
import pytest

from pymolfit import ConfigurationError, Observation, Spectrum, correct, workflow


def test_signature_uses_one_argument_for_file_or_spectrum():
    parameters = inspect.signature(correct).parameters
    assert "spectrum" not in parameters
    assert Spectrum in get_args(get_type_hints(correct)["input_path"])
    assert list(parameters)[:5] == [
        "input_path",
        "output_path",
        "theoretical_spectrum",
        "region_file",
        "show_plot",
    ]


@pytest.fixture
def spectrum():
    return Spectrum(
        np.linspace(588, 590, 12),
        np.ones(12),
        uncertainty=np.full(12, 0.01),
        mask=np.array([False, *([True] * 11)]),
        group_id=np.repeat([2, 3], 6),
        wavelength_unit="nm",
        wavelength_medium="air",
        meta={"instrument": "TEST", "airmass": 1.25, "name": "prepared"},
        observation=Observation(wavelength_frame="observatory", airmass=1.25),
    )


@pytest.mark.parametrize("mode", ["keyword", "positional"])
def test_object_reaches_fit_intact_without_reloading_flux(monkeypatch, spectrum, mode):
    received = {}
    result = object()

    def fit(value, **kwargs):
        received["input"] = value
        return result

    def no_file(*args, **kwargs):
        pytest.fail("a Spectrum input must not reload the spectrum from a file")

    monkeypatch.setattr(workflow, "load_spectrum", no_file)
    monkeypatch.setattr(workflow, "correct_file", no_file)
    monkeypatch.setattr(workflow, "_correct_spectrum_workflow", fit)
    monkeypatch.setattr(workflow, "_finalize_correction", lambda result, **kw: result)
    actual = correct(spectrum) if mode == "positional" else correct(input_path=spectrum)
    assert actual is result
    assert received["input"] is spectrum
    assert spectrum.wavelength_unit == "nm"
    assert spectrum.wavelength_medium == "air"
    np.testing.assert_array_equal(spectrum.group_id, np.repeat([2, 3], 6))
    np.testing.assert_allclose(spectrum.uncertainty, 0.01)
    assert not spectrum.valid[0]
    assert spectrum.meta["airmass"] == 1.25


@pytest.mark.parametrize("value", [123, [1, 2, 3], np.ones(5), {"flux": [1]}])
def test_invalid_input_type_fails_before_fitting(value):
    with pytest.raises(TypeError, match="file path or a Spectrum"):
        correct(input_path=value)


@pytest.mark.parametrize("as_path", [True, False])
def test_input_and_legacy_spectrum_cannot_be_combined(spectrum, as_path):
    with pytest.raises(TypeError, match="spectrum"):
        correct(input_path="missing.fits" if as_path else spectrum, spectrum=spectrum)


def test_legacy_spectrum_rejects_file_path():
    with pytest.raises(TypeError, match="spectrum"):
        correct(spectrum="missing.fits")


@pytest.mark.parametrize("key", ["wavelength", "flux", "uncertainty", "mask", "group_id"])
def test_object_and_separate_arrays_cannot_be_combined(spectrum, key):
    with pytest.raises(TypeError, match=key):
        correct(input_path=spectrum, **{key: np.ones(12)})


@pytest.mark.parametrize(
    "key, value",
    [
        ("input_format", "fits"),
        ("wavelength_col", 0),
        ("flux_col", 1),
        ("uncertainty_col", 2),
        ("image_index", 0),
    ],
)
def test_file_read_options_do_not_silently_apply_to_object(spectrum, key, value):
    with pytest.raises(ConfigurationError, match="only to file-path input"):
        correct(input_path=spectrum, **{key: value})


def test_conflicting_wavelength_medium_still_rejected(spectrum):
    with pytest.raises(ValueError, match=r"conflicts with spectrum\.wavelength_medium"):
        correct(input_path=spectrum, wavelength_medium="vacuum")
