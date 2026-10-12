"""Synthetic data for deterministic tests, never for scientific corrections."""

import numpy as np
from astropy.io import fits
from astropy.table import Table

from pymolfit import LineList


def demo_lines():
    wavelength = np.array([2.3183, 2.3210, 2.3265, 2.3312, 2.3377, 2.3420, 2.3485, 2.3540])
    return LineList(
        wavelength=wavelength,
        strength=np.array([0.018, 0.026, 0.014, 0.03, 0.022, 0.012, 0.018, 0.015]),
        sigma=np.full(wavelength.shape, 1.8e-4),
        gamma=np.full(wavelength.shape, 9.0e-5),
        species=np.array(["H2O", "H2O", "CO2", "H2O", "CH4", "CO2", "H2O", "CH4"]),
        line_source="demo",
    )


def write_spectrum_fits(spectrum, path):
    table = Table({"wavelength": spectrum.wavelength, "flux": spectrum.flux})
    table["wavelength"].unit = spectrum.wavelength_unit
    if spectrum.uncertainty is not None:
        table["uncertainty"] = spectrum.uncertainty
    table["mask"] = ~spectrum.valid
    metadata = spectrum.observation.to_header() if spectrum.observation else {}
    header = fits.Header(
        {f"HIERARCH {key}" if len(key) > 8 else key: value for key, value in metadata.items()}
    )
    header["WAVEMED"] = spectrum.wavelength_medium
    fits.HDUList([fits.PrimaryHDU(header=header), fits.BinTableHDU(table)]).writeto(path)
    return path
