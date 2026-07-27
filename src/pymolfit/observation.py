from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

import numpy as np
from astropy.time import Time


WavelengthFrame = Literal[
    "observatory",
    "topocentric",
    "barycentric",
    "heliocentric",
]

_FRAME_ALIASES = {
    "observatory": "observatory",
    "observer": "observatory",
    "topocentric": "observatory",
    "topocent": "observatory",
    "barycentric": "barycentric",
    "barycent": "barycentric",
    "heliocentric": "heliocentric",
    "heliocen": "heliocentric",
}


@dataclass(frozen=True)
class Observation:
    """Metadata describing when, where, and how a spectrum was observed.

    Use this object with wavelength/flux arrays, which do not carry a FITS
    header. ``wavelength_frame`` describes the velocity frame already applied
    to the supplied wavelength array. Barycentric arrays need either
    ``frame_velocity_km_s`` or sufficient time, site, and target coordinates
    for PyMolFit to reconstruct that correction. Heliocentric arrays require
    an explicit ``frame_velocity_km_s``.

    Weather values are optional. When supplied, they refine the lower
    atmosphere used by the MIPAS/GDAS builder. ``metadata`` can carry
    additional FITS-like keywords needed by a specialized instrument.
    """

    time: Time | datetime | str | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    altitude_m: float | None = None
    airmass: float | None = None
    resolving_power: float | None = None
    wavelength_frame: WavelengthFrame | None = None
    frame_velocity_km_s: float | None = None
    target_ra_deg: float | None = None
    target_dec_deg: float | None = None
    pressure_hpa: float | None = None
    temperature_c: float | None = None
    relative_humidity_percent: float | None = None
    pwv_mm: float | None = None
    instrument: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame = self.wavelength_frame
        if frame is not None:
            key = str(frame).strip().lower()
            try:
                frame = _FRAME_ALIASES[key]
            except KeyError as exc:
                choices = ", ".join(sorted(set(_FRAME_ALIASES.values())))
                raise ValueError(
                    f"unsupported wavelength_frame {self.wavelength_frame!r}; "
                    f"choose {choices}"
                ) from exc
            object.__setattr__(self, "wavelength_frame", frame)

        if self.time is not None:
            try:
                parsed_time = Time(self.time, scale="utc")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "time must be an astropy Time, datetime, or ISO timestamp"
                ) from exc
            if not np.isfinite(parsed_time.mjd):
                raise ValueError("time must be finite")

        self._validate_optional_finite("latitude_deg", self.latitude_deg)
        self._validate_optional_finite("longitude_deg", self.longitude_deg)
        self._validate_optional_finite("altitude_m", self.altitude_m)
        self._validate_optional_positive("airmass", self.airmass)
        self._validate_optional_positive("resolving_power", self.resolving_power)
        self._validate_optional_finite(
            "frame_velocity_km_s",
            self.frame_velocity_km_s,
        )
        self._validate_optional_finite("target_ra_deg", self.target_ra_deg)
        self._validate_optional_finite("target_dec_deg", self.target_dec_deg)
        self._validate_optional_positive("pressure_hpa", self.pressure_hpa)
        self._validate_optional_finite("temperature_c", self.temperature_c)
        self._validate_optional_nonnegative("pwv_mm", self.pwv_mm)

        if self.latitude_deg is not None and not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be between -90 and 90")
        if self.target_ra_deg is not None and not 0.0 <= self.target_ra_deg < 360.0:
            raise ValueError("target_ra_deg must be in [0, 360)")
        if self.target_dec_deg is not None and not -90.0 <= self.target_dec_deg <= 90.0:
            raise ValueError("target_dec_deg must be between -90 and 90")
        if self.temperature_c is not None and self.temperature_c <= -273.15:
            raise ValueError("temperature_c must be above absolute zero")
        if (
            self.relative_humidity_percent is not None
            and (
                not np.isfinite(self.relative_humidity_percent)
                or not 0.0 <= self.relative_humidity_percent <= 100.0
            )
        ):
            raise ValueError(
                "relative_humidity_percent must be finite and between 0 and 100"
            )
        if frame == "observatory" and self.frame_velocity_km_s is not None:
            raise ValueError(
                "frame_velocity_km_s is not used for observatory-frame wavelengths"
            )
        if frame == "heliocentric" and self.frame_velocity_km_s is None:
            raise ValueError(
                "heliocentric wavelengths require frame_velocity_km_s"
            )
        if frame == "barycentric" and self.frame_velocity_km_s is None:
            missing = [
                name
                for name, value in (
                    ("time", self.time),
                    ("latitude_deg", self.latitude_deg),
                    ("longitude_deg", self.longitude_deg),
                    ("altitude_m", self.altitude_m),
                    ("target_ra_deg", self.target_ra_deg),
                    ("target_dec_deg", self.target_dec_deg),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "barycentric wavelengths without frame_velocity_km_s require "
                    + ", ".join(missing)
                )

        object.__setattr__(self, "metadata", dict(self.metadata))

    @staticmethod
    def _validate_optional_finite(name: str, value: float | None) -> None:
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    @classmethod
    def _validate_optional_positive(cls, name: str, value: float | None) -> None:
        cls._validate_optional_finite(name, value)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    @classmethod
    def _validate_optional_nonnegative(cls, name: str, value: float | None) -> None:
        cls._validate_optional_finite(name, value)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")

    def to_header(
        self,
        base_header: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return FITS-like metadata consumed by the existing science pipeline.

        Values explicitly set on the observation override ``base_header`` and
        the free-form ``metadata`` mapping.
        """

        header = {} if base_header is None else dict(base_header)
        header.update(dict(self.metadata))

        if self.time is not None:
            header["DATE-OBS"] = Time(self.time, scale="utc").utc.isot
        if self.latitude_deg is not None:
            header["ESO TEL GEOLAT"] = float(self.latitude_deg)
        if self.longitude_deg is not None:
            header["ESO TEL GEOLON"] = float(self.longitude_deg)
        if self.altitude_m is not None:
            header["ESO TEL GEOELEV"] = float(self.altitude_m)
        if self.airmass is not None:
            header["AIRMASS"] = float(self.airmass)
        if self.resolving_power is not None:
            header["SPEC_RES"] = float(self.resolving_power)
        if self.target_ra_deg is not None:
            header["RA"] = float(self.target_ra_deg)
        if self.target_dec_deg is not None:
            header["DEC"] = float(self.target_dec_deg)
        if self.pressure_hpa is not None:
            header["ESO TEL AMBI PRES START"] = float(self.pressure_hpa)
        if self.temperature_c is not None:
            header["ESO TEL AMBI TEMP"] = float(self.temperature_c)
        if self.relative_humidity_percent is not None:
            header["ESO TEL AMBI RHUM"] = float(self.relative_humidity_percent)
        if self.pwv_mm is not None:
            header["PWV"] = float(self.pwv_mm)
        if self.instrument is not None:
            header["INSTRUME"] = str(self.instrument)

        if self.wavelength_frame == "observatory":
            header["SPECSYS"] = "TOPOCENT"
        elif self.wavelength_frame == "barycentric":
            header["SPECSYS"] = "BARYCENT"
            if self.frame_velocity_km_s is not None:
                header["BERV"] = float(self.frame_velocity_km_s)
        elif self.wavelength_frame == "heliocentric":
            header["SPECSYS"] = "HELIOCEN"
            header["HELIOVEL"] = float(self.frame_velocity_km_s)

        return header
