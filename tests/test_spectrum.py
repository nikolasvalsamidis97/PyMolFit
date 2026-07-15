import numpy as np
import pytest

from genmolfit import Spectrum, correct_spectrum


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


def test_correct_spectrum_rejects_negative_transmission_uncertainty():
    spectrum = Spectrum(wavelength=np.array([1.0, 1.1]), flux=np.ones(2))

    with pytest.raises(ValueError, match="non-negative"):
        correct_spectrum(
            spectrum,
            np.ones(2),
            transmission_uncertainty=np.array([0.01, -0.01]),
        )
