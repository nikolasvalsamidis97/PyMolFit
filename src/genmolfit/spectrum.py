from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

WAVELENGTH_TO_MICRON = {
    "micron": 1.0,
    "microns": 1.0,
    "um": 1.0,
    "mum": 1.0,
    "nm": 1.0e-3,
    "nanometer": 1.0e-3,
    "nanometers": 1.0e-3,
    "angstrom": 1.0e-4,
    "angstroms": 1.0e-4,
    "aa": 1.0e-4,
    "a": 1.0e-4,
    "meter": 1.0e6,
    "meters": 1.0e6,
    "m": 1.0e6,
}

WAVELENGTH_MEDIUM_ALIASES = {
    "vacuum": "vacuum",
    "vac": "vacuum",
    "air": "air",
}


def wavelength_scale_to_micron(unit: str) -> float:
    key = unit.strip().lower().replace("angstrom", "angstrom")
    try:
        return WAVELENGTH_TO_MICRON[key]
    except KeyError as exc:
        raise ValueError(f"unsupported wavelength unit: {unit!r}") from exc


def normalize_wavelength_medium(medium: str) -> str:
    key = medium.strip().lower()
    try:
        return WAVELENGTH_MEDIUM_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported wavelength medium: {medium!r}") from exc


def standard_air_refractive_index(wavelength_micron: np.ndarray) -> np.ndarray:
    """Molecfit's Edlen (1966) refractive index for standard dry air."""

    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    sigma2 = (1.0 / wavelength_micron) ** 2
    return 1.0 + 1.0e-8 * (
        8342.13
        + 2_406_030.0 / (130.0 - sigma2)
        + 15_997.0 / (38.9 - sigma2)
    )


def air_to_vacuum_wavelength(wavelength: np.ndarray, *, unit: str = "micron") -> np.ndarray:
    """Convert standard-air wavelengths to vacuum wavelengths in the same unit."""

    scale = wavelength_scale_to_micron(unit)
    wavelength_micron = np.asarray(wavelength, dtype=float) * scale
    vacuum_micron = wavelength_micron * standard_air_refractive_index(wavelength_micron)
    return vacuum_micron / scale


def vacuum_to_air_wavelength(wavelength: np.ndarray, *, unit: str = "micron") -> np.ndarray:
    """Convert vacuum wavelengths to standard-air wavelengths in the same unit."""

    scale = wavelength_scale_to_micron(unit)
    vacuum_micron = np.asarray(wavelength, dtype=float) * scale
    air_micron = vacuum_micron / standard_air_refractive_index(vacuum_micron)
    for _ in range(3):
        air_micron = vacuum_micron / standard_air_refractive_index(air_micron)
    return air_micron / scale


@dataclass(frozen=True)
class Spectrum:
    """A generic one-dimensional spectrum.

    Wavelength is assumed to be in microns unless another unit is stated in
    ``wavelength_unit``. The arrays are copied into float NumPy arrays and must
    all have the same one-dimensional shape.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray | None = None
    mask: np.ndarray | None = None
    wavelength_unit: str = "micron"
    wavelength_medium: str = "vacuum"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        uncertainty = None if self.uncertainty is None else np.asarray(self.uncertainty, dtype=float)
        mask = None if self.mask is None else np.asarray(self.mask, dtype=bool)

        if wavelength.ndim != 1 or flux.ndim != 1:
            raise ValueError("wavelength and flux must be one-dimensional arrays")
        if wavelength.shape != flux.shape:
            raise ValueError("wavelength and flux must have the same shape")
        if uncertainty is not None and uncertainty.shape != flux.shape:
            raise ValueError("uncertainty must have the same shape as flux")
        if mask is not None and mask.shape != flux.shape:
            raise ValueError("mask must have the same shape as flux")

        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "wavelength_medium", normalize_wavelength_medium(self.wavelength_medium))

    @property
    def valid(self) -> np.ndarray:
        valid = np.isfinite(self.wavelength) & np.isfinite(self.flux)
        if self.uncertainty is not None:
            valid &= np.isfinite(self.uncertainty) & (self.uncertainty > 0)
        if self.mask is not None:
            valid &= self.mask
        return valid

    def with_flux(self, flux: np.ndarray, *, uncertainty: np.ndarray | None = None) -> "Spectrum":
        return Spectrum(
            wavelength=self.wavelength.copy(),
            flux=np.asarray(flux, dtype=float),
            uncertainty=self.uncertainty if uncertainty is None else uncertainty,
            mask=self.mask,
            wavelength_unit=self.wavelength_unit,
            wavelength_medium=self.wavelength_medium,
            meta=dict(self.meta),
        )

    def to_unit(self, wavelength_unit: str) -> "Spectrum":
        source = wavelength_scale_to_micron(self.wavelength_unit)
        target = wavelength_scale_to_micron(wavelength_unit)
        factor = source / target
        return Spectrum(
            wavelength=self.wavelength * factor,
            flux=self.flux.copy(),
            uncertainty=None if self.uncertainty is None else self.uncertainty.copy(),
            mask=None if self.mask is None else self.mask.copy(),
            wavelength_unit=wavelength_unit,
            wavelength_medium=self.wavelength_medium,
            meta={**dict(self.meta), "original_wavelength_unit": self.wavelength_unit},
        )

    def to_vacuum(self) -> "Spectrum":
        if self.wavelength_medium == "vacuum":
            return self
        return Spectrum(
            wavelength=air_to_vacuum_wavelength(self.wavelength, unit=self.wavelength_unit),
            flux=self.flux.copy(),
            uncertainty=None if self.uncertainty is None else self.uncertainty.copy(),
            mask=None if self.mask is None else self.mask.copy(),
            wavelength_unit=self.wavelength_unit,
            wavelength_medium="vacuum",
            meta={**dict(self.meta), "original_wavelength_medium": self.wavelength_medium},
        )

    def to_air(self) -> "Spectrum":
        if self.wavelength_medium == "air":
            return self
        return Spectrum(
            wavelength=vacuum_to_air_wavelength(self.wavelength, unit=self.wavelength_unit),
            flux=self.flux.copy(),
            uncertainty=None if self.uncertainty is None else self.uncertainty.copy(),
            mask=None if self.mask is None else self.mask.copy(),
            wavelength_unit=self.wavelength_unit,
            wavelength_medium="air",
            meta={**dict(self.meta), "original_wavelength_medium": self.wavelength_medium},
        )

    def sorted(self) -> "Spectrum":
        order = np.argsort(self.wavelength)
        uncertainty = None if self.uncertainty is None else self.uncertainty[order]
        mask = None if self.mask is None else self.mask[order]
        return Spectrum(
            wavelength=self.wavelength[order],
            flux=self.flux[order],
            uncertainty=uncertainty,
            mask=mask,
            wavelength_unit=self.wavelength_unit,
            wavelength_medium=self.wavelength_medium,
            meta=dict(self.meta),
        )


def correct_spectrum(
    spectrum: Spectrum,
    transmission: np.ndarray,
    *,
    transmission_uncertainty: np.ndarray | None = None,
    min_transmission: float = 0.03,
) -> Spectrum:
    """Divide a spectrum by a telluric transmission model.

    Very deep fitted absorption can amplify noise severely, so values below
    ``min_transmission`` are masked to NaN in the corrected flux.
    """

    transmission = np.asarray(transmission, dtype=float)
    if transmission.shape != spectrum.flux.shape:
        raise ValueError("transmission must have the same shape as the spectrum")
    if transmission_uncertainty is not None:
        transmission_uncertainty = np.asarray(transmission_uncertainty, dtype=float)
        if transmission_uncertainty.shape != spectrum.flux.shape:
            raise ValueError("transmission_uncertainty must have the same shape as the spectrum")
        if np.any(np.isfinite(transmission_uncertainty) & (transmission_uncertainty < 0)):
            raise ValueError("transmission_uncertainty must be non-negative")
    if not 0 < min_transmission < 1:
        raise ValueError("min_transmission must be between 0 and 1")

    safe = np.isfinite(transmission) & (transmission >= min_transmission)
    corrected_flux = np.full_like(spectrum.flux, np.nan, dtype=float)
    corrected_flux[safe] = spectrum.flux[safe] / transmission[safe]

    corrected_uncertainty = None
    if spectrum.uncertainty is not None or transmission_uncertainty is not None:
        corrected_uncertainty = np.full_like(spectrum.flux, np.nan, dtype=float)
        variance = np.zeros(np.count_nonzero(safe), dtype=float)
        if spectrum.uncertainty is not None:
            variance += (spectrum.uncertainty[safe] / transmission[safe]) ** 2
        if transmission_uncertainty is not None:
            variance += (
                spectrum.flux[safe]
                * transmission_uncertainty[safe]
                / transmission[safe] ** 2
            ) ** 2
        corrected_uncertainty[safe] = np.sqrt(variance)

    corrected_mask = safe if spectrum.mask is None else (spectrum.mask & safe)
    return Spectrum(
        wavelength=spectrum.wavelength.copy(),
        flux=corrected_flux,
        uncertainty=corrected_uncertainty,
        mask=corrected_mask,
        wavelength_unit=spectrum.wavelength_unit,
        wavelength_medium=spectrum.wavelength_medium,
        meta={
            **dict(spectrum.meta),
            "telluric_corrected": True,
            "transmission_uncertainty_propagated": transmission_uncertainty is not None,
        },
    )
