import numpy as np
import pytest

from pymolfit import Spectrum, correct_spectrum
from pymolfit.provenance import _spectrum_summary


def test_spectrum_owns_input_arrays():
    wavelength = np.array([1.0, 1.1])
    flux = np.array([10.0, 8.0])
    uncertainty = np.array([0.5, 0.4])
    mask = np.array([True, False])
    group_id = np.array([1, 2])
    spectrum = Spectrum(wavelength, flux, uncertainty, mask, group_id)

    wavelength[0] = 2.0
    flux[0] = 20.0
    uncertainty[0] = 5.0
    mask[0] = False
    group_id[0] = 9

    np.testing.assert_array_equal(spectrum.wavelength, [1.0, 1.1])
    np.testing.assert_array_equal(spectrum.flux, [10.0, 8.0])
    np.testing.assert_array_equal(spectrum.uncertainty, [0.5, 0.4])
    np.testing.assert_array_equal(spectrum.mask, [True, False])
    np.testing.assert_array_equal(spectrum.group_id, [1, 2])


def test_to_unit_returns_same_object_when_unit_is_unchanged():
    spectrum = Spectrum(
        wavelength=np.array([1.0, 1.1]),
        flux=np.array([10.0, 8.0]),
        wavelength_unit="micron",
    )

    assert spectrum.to_unit("micron") is spectrum


def test_correct_spectrum_propagates_flux_and_transmission_uncertainty():
    spectrum = Spectrum(
        wavelength=np.array([1.0, 1.1]),
        flux=np.array([10.0, 8.0]),
        uncertainty=np.array([0.5, 0.4]),
    )
    transmission = np.array([0.8, 0.5])
    transmission_uncertainty = np.array([0.02, 0.03])

    corrected = correct_spectrum(
        spectrum,
        transmission,
        transmission_uncertainty=transmission_uncertainty,
    )

    expected = np.sqrt(
        (spectrum.uncertainty / transmission) ** 2
        + (spectrum.flux * transmission_uncertainty / transmission**2) ** 2
    )
    np.testing.assert_allclose(corrected.uncertainty, expected)
    assert corrected.meta["transmission_uncertainty_propagated"] is True


def test_correct_spectrum_preserves_physical_groups():
    spectrum = Spectrum(
        wavelength=np.array([1.0, 1.1]),
        flux=np.array([10.0, 8.0]),
        group_id=np.array([4, 7]),
    )

    corrected = correct_spectrum(spectrum, np.array([0.8, 0.5]))

    np.testing.assert_array_equal(corrected.group_id, spectrum.group_id)


def test_correct_spectrum_rejects_negative_transmission_uncertainty():
    spectrum = Spectrum(wavelength=np.array([1.0, 1.1]), flux=np.ones(2))

    with pytest.raises(ValueError, match="non-negative"):
        correct_spectrum(
            spectrum,
            np.ones(2),
            transmission_uncertainty=np.array([0.01, -0.01]),
        )


def test_correct_spectrum_masks_invalid_input_pixels():
    spectrum = Spectrum(
        wavelength=np.array([1.0, 1.1, 1.2]),
        flux=np.array([10.0, np.nan, 8.0]),
        uncertainty=np.array([0.5, 0.4, -1.0]),
    )

    corrected = correct_spectrum(spectrum, np.ones(3))

    np.testing.assert_array_equal(corrected.mask, [True, False, False])
    assert np.isfinite(corrected.flux[0])
    assert np.isnan(corrected.flux[1:]).all()


def test_spectrum_provenance_includes_physical_groups():
    common = {
        "wavelength": np.array([1.0, 1.1]),
        "flux": np.ones(2),
    }
    first = Spectrum(**common, group_id=np.array([1, 1]))
    second = Spectrum(**common, group_id=np.array([1, 2]))

    assert _spectrum_summary(first)["sha256"] != _spectrum_summary(second)["sha256"]


def test_spectrum_sorting_preserves_physical_groups():
    spectrum = Spectrum(
        wavelength=np.array([2.4, 2.1, 2.3, 2.2]),
        flux=np.arange(4.0),
        group_id=np.array([2, 1, 2, 1]),
    )

    ordered = spectrum.sorted()

    np.testing.assert_array_equal(ordered.wavelength, [2.1, 2.2, 2.3, 2.4])
    np.testing.assert_array_equal(ordered.group_id, [1, 1, 2, 2])
