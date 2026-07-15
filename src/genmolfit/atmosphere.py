from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Mapping

import numpy as np
from astropy.table import Table
from astropy.time import Time, TimeDelta

from .gdas import resolve_time_local_gdas_profile
from .physics import BOLTZMANN_J_PER_K

PA_PER_ATM = 101_325.0
HPA_PER_ATM = 1013.25
CM_PER_M = 100.0
H2O_MOLECULES_PER_PWV_MM_CM2 = 3.34556e21
EARTH_RADIUS_M = 6_371_230.0
MIPAS_GDAS_MERGE_FRACTION = 0.2
DEFAULT_OBSERVATORY_ALTITUDE_M = 2635.0
DEFAULT_OBSERVATORY_LATITUDE_DEG = -24.6276
DEFAULT_OBSERVATORY_LONGITUDE_DEG = -70.4051

_LATITUDE_HEADER_KEYS = (
    "ESO TEL GEOLAT",
    "HIERARCH ESO TEL GEOLAT",
    "LAT-OBS",
    "OBSGEO-B",
    "LATITUDE",
    "SITELAT",
)
_LONGITUDE_HEADER_KEYS = (
    "ESO TEL GEOLON",
    "HIERARCH ESO TEL GEOLON",
    "LONG-OBS",
    "OBSGEO-L",
    "LONGITUDE",
    "SITELONG",
)
_ALTITUDE_HEADER_KEYS = (
    "ESO TEL GEOELEV",
    "HIERARCH ESO TEL GEOELEV",
    "ALT-OBS",
    "OBSGEO-H",
    "ALTITUDE",
    "ELEVATIO",
    "SITEALT",
)


@dataclass(frozen=True)
class _ObservatorySite:
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


_OBSERVATORY_SITES = {
    "paranal": _ObservatorySite("Paranal", -24.6276, -70.4051, 2635.0),
    "la_silla": _ObservatorySite("La Silla", -29.2567, -70.7346, 2400.0),
    "keck": _ObservatorySite("W. M. Keck Observatory", 19.82658656, -155.4722, 4145.0),
}
_OBSERVATORY_ALIASES = {
    "PARANAL": "paranal",
    "ESO PARANAL": "paranal",
    "VLT": "paranal",
    "ESO VLT": "paranal",
    "ANTU": "paranal",
    "KUEYEN": "paranal",
    "MELIPAL": "paranal",
    "YEPUN": "paranal",
    "VISTA": "paranal",
    "LA SILLA": "la_silla",
    "ESO LA SILLA": "la_silla",
    "NTT": "la_silla",
    "ESO 3 6M": "la_silla",
    "MPG ESO 2 2M": "la_silla",
    "KECK": "keck",
    "KECK I": "keck",
    "KECK II": "keck",
    "KECK1": "keck",
    "KECK2": "keck",
    "WM KECK OBSERVATORY": "keck",
    "W M KECK OBSERVATORY": "keck",
}
DEFAULT_METEO_MIXING_HEIGHT_M = 5_000.0
_LAYER_QUADRATURE_NODES, _LAYER_QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)
MOLECFIT_FIXED_LOW_LEVELS_M = np.asarray(
    [
        0.0, 500.0, 1_000.0, 1_500.0, 2_000.0, 2_500.0, 3_000.0,
        3_500.0, 4_000.0, 4_500.0, 5_000.0, 5_500.0, 6_000.0,
        6_500.0, 7_000.0, 7_500.0, 8_000.0, 8_500.0, 9_000.0,
        9_500.0, 10_000.0, 11_000.0, 12_000.0, 13_000.0, 14_000.0,
        15_000.0, 16_000.0, 17_000.0, 18_000.0, 20_000.0, 22_000.0,
        24_000.0, 26_000.0,
    ],
    dtype=float,
)

DEFAULT_TELLURIC_MIXING_RATIOS = {
    "H2O": 2.0e-3,
    "CO2": 4.2e-4,
    "CH4": 1.9e-6,
    "O2": 2.095e-1,
    "O3": 1.0e-7,
    "CO": 1.0e-7,
    "N2O": 3.3e-7,
}

MIPAS_PROFILE_ALIASES = {
    "equ": "equ",
    "equatorial": "equ",
    "tropical_day": "equ",
    "std": "std",
    "standard": "std",
    "midlatitude": "std",
    "tro": "tro",
    "tropical": "tro",
}


@dataclass(frozen=True)
class AtmosphereLayer:
    pressure_atm: float
    temperature_k: float
    path_length_m: float
    mixing_ratios: Mapping[str, float] = field(default_factory=dict)
    vertical_path_length_m: float | None = None

    def __post_init__(self) -> None:
        if self.pressure_atm <= 0:
            raise ValueError("pressure_atm must be positive")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.path_length_m <= 0:
            raise ValueError("path_length_m must be positive")
        if self.vertical_path_length_m is not None and self.vertical_path_length_m <= 0:
            raise ValueError("vertical_path_length_m must be positive")
        for species, fraction in self.mixing_ratios.items():
            if fraction < 0:
                raise ValueError(f"mixing ratio for {species} must be non-negative")

    def column_density_cm2(self, species: str) -> float:
        return self._column_density_cm2(species, self.path_length_m)

    def vertical_column_density_cm2(self, species: str) -> float:
        vertical_path = self.path_length_m if self.vertical_path_length_m is None else self.vertical_path_length_m
        return self._column_density_cm2(species, vertical_path)

    def _column_density_cm2(self, species: str, path_length_m: float) -> float:
        mixing_ratio = self.mixing_ratios.get(species, 0.0)
        pressure_pa = self.pressure_atm * PA_PER_ATM
        number_density_m3 = pressure_pa / (BOLTZMANN_J_PER_K * self.temperature_k)
        return number_density_m3 * mixing_ratio * path_length_m / (CM_PER_M**2)


@dataclass(frozen=True)
class AtmosphereProfile:
    layers: tuple[AtmosphereLayer, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("AtmosphereProfile must contain at least one layer")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def total_column_cm2(self, species: str) -> float:
        return sum(layer.column_density_cm2(species) for layer in self.layers)

    def total_vertical_column_cm2(self, species: str) -> float:
        return sum(layer.vertical_column_density_cm2(species) for layer in self.layers)

    @property
    def species_names(self) -> tuple[str, ...]:
        species = set()
        for layer in self.layers:
            species.update(layer.mixing_ratios)
        return tuple(sorted(species))

    def to_table(self) -> Table:
        table = Table()
        table["pressure_atm"] = [layer.pressure_atm for layer in self.layers]
        table["temperature_k"] = [layer.temperature_k for layer in self.layers]
        table["path_length_m"] = [layer.path_length_m for layer in self.layers]
        table["vertical_path_length_m"] = [
            layer.path_length_m if layer.vertical_path_length_m is None else layer.vertical_path_length_m
            for layer in self.layers
        ]
        for species in self.species_names:
            table[f"mix_{species}"] = [layer.mixing_ratios.get(species, 0.0) for layer in self.layers]
        table.meta.update(dict(self.metadata))
        return table

    def write(self, path: str | Path, *, format: str = "ascii.ecsv") -> None:
        self.to_table().write(path, format=format, overwrite=True)

    def scaled_species(self, species_scales: Mapping[str, float]) -> "AtmosphereProfile":
        layers = []
        for layer in self.layers:
            mixing_ratios = dict(layer.mixing_ratios)
            for species, scale in species_scales.items():
                if scale < 0:
                    raise ValueError(f"scale for {species} must be non-negative")
                mixing_ratios[species] = mixing_ratios.get(species, 0.0) * scale
            layers.append(
                AtmosphereLayer(
                    pressure_atm=layer.pressure_atm,
                    temperature_k=layer.temperature_k,
                    path_length_m=layer.path_length_m,
                    mixing_ratios=mixing_ratios,
                    vertical_path_length_m=layer.vertical_path_length_m,
                )
            )
        return AtmosphereProfile(tuple(layers), metadata=self.metadata)

    def with_path_scale(self, scale: float) -> "AtmosphereProfile":
        """Return a profile with all layer path lengths multiplied by ``scale``."""

        if scale <= 0:
            raise ValueError("path scale must be positive")
        return AtmosphereProfile(
            tuple(
                AtmosphereLayer(
                    pressure_atm=layer.pressure_atm,
                    temperature_k=layer.temperature_k,
                    path_length_m=layer.path_length_m * scale,
                    mixing_ratios=dict(layer.mixing_ratios),
                    vertical_path_length_m=(
                        None if layer.vertical_path_length_m is None else layer.vertical_path_length_m * scale
                    ),
                )
                for layer in self.layers
            ),
            metadata=self.metadata,
        )

    def perturbed(
        self,
        *,
        pressure_scale: float = 1.0,
        temperature_offset_k: float = 0.0,
        path_scale: float = 1.0,
        species_scales: Mapping[str, float] | None = None,
        label: str | None = None,
    ) -> "AtmosphereProfile":
        """Return a traceable atmosphere variant for sensitivity analysis.

        The operation is deliberately mechanical: callers choose perturbations
        from instrument/site uncertainties or an atmospheric ensemble. No
        observation-specific coefficients are inferred here.
        """

        if not np.isfinite(pressure_scale) or pressure_scale <= 0:
            raise ValueError("pressure_scale must be positive and finite")
        if not np.isfinite(temperature_offset_k):
            raise ValueError("temperature_offset_k must be finite")
        if not np.isfinite(path_scale) or path_scale <= 0:
            raise ValueError("path_scale must be positive and finite")
        resolved_species_scales = {} if species_scales is None else dict(species_scales)
        for species, scale in resolved_species_scales.items():
            if not np.isfinite(scale) or scale < 0:
                raise ValueError(f"species scale for {species} must be non-negative and finite")

        layers = []
        for layer in self.layers:
            temperature = layer.temperature_k + temperature_offset_k
            if temperature <= 0:
                raise ValueError("temperature perturbation produces a non-positive layer temperature")
            ratios = dict(layer.mixing_ratios)
            for species, scale in resolved_species_scales.items():
                ratios[species] = ratios.get(species, 0.0) * scale
            layers.append(
                AtmosphereLayer(
                    pressure_atm=layer.pressure_atm * pressure_scale,
                    temperature_k=temperature,
                    path_length_m=layer.path_length_m * path_scale,
                    mixing_ratios=ratios,
                    vertical_path_length_m=(
                        None
                        if layer.vertical_path_length_m is None
                        else layer.vertical_path_length_m * path_scale
                    ),
                )
            )

        perturbation = {
            "pressure_scale": float(pressure_scale),
            "temperature_offset_k": float(temperature_offset_k),
            "path_scale": float(path_scale),
            "species_scales": resolved_species_scales,
            "label": None if label is None else str(label),
        }
        return AtmosphereProfile(
            tuple(layers),
            metadata={
                **dict(self.metadata),
                "systematic_perturbation": perturbation,
            },
        )

    def with_species_column(self, species: str, target_column_cm2: float) -> "AtmosphereProfile":
        if target_column_cm2 < 0:
            raise ValueError("target_column_cm2 must be non-negative")
        current = self.total_column_cm2(species)
        if current <= 0:
            raise ValueError(f"cannot scale {species}; current column is zero")
        return self.scaled_species({species: target_column_cm2 / current})

    def with_pwv_mm(self, pwv_mm: float) -> "AtmosphereProfile":
        """Return a profile scaled to a precipitable water vapor column."""

        if pwv_mm < 0:
            raise ValueError("pwv_mm must be non-negative")
        current = self.total_vertical_column_cm2("H2O")
        if current <= 0:
            raise ValueError("cannot scale H2O; current vertical column is zero")
        return self.scaled_species({"H2O": pwv_mm * H2O_MOLECULES_PER_PWV_MM_CM2 / current})

    @classmethod
    def single_layer(
        cls,
        *,
        pressure_atm: float = 0.75,
        temperature_k: float = 280.0,
        path_length_m: float = 8_000.0,
        airmass: float = 1.0,
        mixing_ratios: Mapping[str, float] | None = None,
    ) -> "AtmosphereProfile":
        if airmass <= 0:
            raise ValueError("airmass must be positive")
        return cls(
            layers=(
                AtmosphereLayer(
                    pressure_atm=pressure_atm,
                    temperature_k=temperature_k,
                    path_length_m=path_length_m * airmass,
                    mixing_ratios=DEFAULT_TELLURIC_MIXING_RATIOS if mixing_ratios is None else dict(mixing_ratios),
                    vertical_path_length_m=path_length_m,
                ),
            )
        )

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        pressure_col: str | None = None,
        temperature_col: str = "temperature_k",
        path_length_col: str | None = None,
        mixing_prefixes: tuple[str, ...] = ("mix_", "vmr_"),
        airmass: float = 1.0,
    ) -> "AtmosphereProfile":
        if airmass <= 0:
            raise ValueError("airmass must be positive")
        table = Table.read(path)
        pressure_name = _resolve_pressure_column(table, pressure_col)
        path_name = _resolve_path_length_column(table, path_length_col)
        if temperature_col not in table.colnames:
            raise ValueError(f"temperature column {temperature_col!r} not found")

        pressure_atm = _pressure_to_atm(table, pressure_name)
        temperature_k = np.asarray(table[temperature_col], dtype=float)
        vertical_path_name = _resolve_vertical_path_length_column(table)
        vertical_path_length_m = (
            np.asarray(table[vertical_path_name], dtype=float)
            if vertical_path_name
            else np.asarray(table[path_name], dtype=float)
        )
        path_length_m = np.asarray(table[path_name], dtype=float) * airmass
        mixing_cols = _mixing_columns(table, mixing_prefixes)
        if not mixing_cols:
            raise ValueError("atmosphere table must contain mixing-ratio columns like mix_H2O")

        layers = []
        for index in range(len(table)):
            layers.append(
                AtmosphereLayer(
                    pressure_atm=float(pressure_atm[index]),
                    temperature_k=float(temperature_k[index]),
                    path_length_m=float(path_length_m[index]),
                    mixing_ratios={
                        species: float(np.asarray(table[col], dtype=float)[index])
                        for col, species in mixing_cols.items()
                    },
                    vertical_path_length_m=float(vertical_path_length_m[index]),
                )
            )
        return cls(tuple(layers))

    @classmethod
    def standard_midlatitude(
        cls,
        *,
        airmass: float = 1.0,
        observatory_altitude_m: float = 2635.0,
        pressure_at_observatory_atm: float = 0.75,
        temperature_at_observatory_k: float = 280.0,
        top_altitude_m: float = 50_000.0,
        n_layers: int = 36,
        mixing_ratios: Mapping[str, float] | None = None,
        water_scale_height_m: float = 2_000.0,
        spherical_slant: bool = True,
        earth_radius_m: float = EARTH_RADIUS_M,
    ) -> "AtmosphereProfile":
        """Simple multi-layer atmosphere above an observatory.

        This is deliberately self-contained. It is not a replacement for GDAS or
        MIPAS profiles, but it gives the radiative-transfer model pressure,
        temperature, and absorber columns that vary with height.
        """

        if airmass <= 0:
            raise ValueError("airmass must be positive")
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")
        if top_altitude_m <= observatory_altitude_m:
            raise ValueError("top_altitude_m must be above the observatory")
        if earth_radius_m <= 0:
            raise ValueError("earth_radius_m must be positive")

        base_ratios = DEFAULT_TELLURIC_MIXING_RATIOS if mixing_ratios is None else dict(mixing_ratios)
        edges = np.linspace(observatory_altitude_m, top_altitude_m, n_layers + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        thickness = (
            _spherical_layer_path_lengths_m(
                edges,
                airmass=airmass,
                earth_radius_m=earth_radius_m,
            )
            if spherical_slant
            else np.diff(edges) * airmass
        )
        scale_height_m = 7_500.0
        lapse_rate_k_per_m = 0.0065
        layers = []

        vertical_thickness = np.diff(edges)
        for altitude_m, path_length_m, vertical_path_length_m in zip(
            centers,
            thickness,
            vertical_thickness,
            strict=True,
        ):
            height_above_site = altitude_m - observatory_altitude_m
            pressure_atm = pressure_at_observatory_atm * np.exp(-height_above_site / scale_height_m)
            temperature_k = max(
                216.65,
                temperature_at_observatory_k - lapse_rate_k_per_m * height_above_site,
            )
            ratios = dict(base_ratios)
            if "H2O" in ratios:
                ratios["H2O"] *= float(np.exp(-height_above_site / water_scale_height_m))
            if "O3" in ratios:
                ozone_peak = np.exp(-0.5 * ((altitude_m - 25_000.0) / 7_000.0) ** 2)
                ratios["O3"] *= 1.0 + 40.0 * ozone_peak

            layers.append(
                AtmosphereLayer(
                    pressure_atm=float(pressure_atm),
                    temperature_k=float(temperature_k),
                    path_length_m=float(path_length_m),
                    mixing_ratios=ratios,
                    vertical_path_length_m=float(vertical_path_length_m),
                )
            )

        return cls(tuple(layers))

    @classmethod
    def from_observatory_conditions(
        cls,
        *,
        airmass: float = 1.0,
        observatory_altitude_m: float = DEFAULT_OBSERVATORY_ALTITUDE_M,
        pressure_at_observatory_atm: float | None = None,
        temperature_at_observatory_k: float | None = None,
        pwv_mm: float | None = None,
        n_layers: int = 48,
        top_altitude_m: float = 80_000.0,
        mixing_ratios: Mapping[str, float] | None = None,
        spherical_slant: bool = True,
    ) -> "AtmosphereProfile":
        """Build a self-contained atmosphere from common observing metadata.

        This is the no-GDAS fallback intended for generic instruments. It uses
        an approximate standard atmosphere above the observatory and optional
        PWV scaling. It is not as data-rich as GDAS, but it gives the physics
        engine a layered pressure/temperature/composition profile without
        relying on external binaries or Molecfit products.
        """

        if pressure_at_observatory_atm is None:
            pressure_at_observatory_atm = _standard_pressure_at_altitude_atm(observatory_altitude_m)
        if temperature_at_observatory_k is None:
            temperature_at_observatory_k = _standard_temperature_at_altitude_k(observatory_altitude_m)
        profile = cls.standard_midlatitude(
            airmass=airmass,
            observatory_altitude_m=observatory_altitude_m,
            pressure_at_observatory_atm=pressure_at_observatory_atm,
            temperature_at_observatory_k=temperature_at_observatory_k,
            top_altitude_m=top_altitude_m,
            n_layers=n_layers,
            mixing_ratios=mixing_ratios,
            spherical_slant=spherical_slant,
        )
        return profile.with_pwv_mm(pwv_mm) if pwv_mm is not None else profile

    @classmethod
    def from_mipas_gdas(
        cls,
        *,
        observation_time: Time | datetime | str | float | None = None,
        latitude_deg: float | None = DEFAULT_OBSERVATORY_LATITUDE_DEG,
        longitude_deg: float | None = DEFAULT_OBSERVATORY_LONGITUDE_DEG,
        observatory_altitude_m: float = DEFAULT_OBSERVATORY_ALTITUDE_M,
        airmass: float = 1.0,
        mipas_profile: str = "equ",
        gdas_profile: str | Path | None = None,
        gdas_mode: str = "auto",
        gdas_cache_dir: str | Path | None = None,
        gdas_download_timeout_s: float = 15.0,
        pressure_at_observatory_atm: float | None = None,
        temperature_at_observatory_k: float | None = None,
        relative_humidity_percent: float | None = None,
        meteo_mixing_height_m: float = DEFAULT_METEO_MIXING_HEIGHT_M,
        pwv_mm: float | None = None,
        top_altitude_m: float = 120_000.0,
        mixing_ratios: Mapping[str, float] | None = None,
        spherical_slant: bool = True,
        refracted_slant: bool = True,
        reference_wavenumber_cm: float = 10_000.0,
        earth_radius_m: float = EARTH_RADIUS_M,
    ) -> "AtmosphereProfile":
        """Build a Molecfit-style MIPAS+GDAS atmospheric profile.

        The shipped MIPAS profile supplies the long-lived species and the
        stratosphere/mesosphere. The GDAS average profile for the observation
        month replaces pressure, temperature, and H2O in the lower atmosphere.
        Above the highest GDAS level, the GDAS/standard mismatch is blended out
        using Molecfit's relative-deviation merge rule.

        ``gdas_mode="auto"`` first tries an exact cached/downloaded ESO GDAS
        profile for the observation time and site. If that is unavailable, it
        falls back to the same six bundled two-month average GDAS profiles that
        Molecfit uses.
        """

        if airmass <= 0:
            raise ValueError("airmass must be positive")
        if top_altitude_m <= observatory_altitude_m:
            raise ValueError("top_altitude_m must be above the observatory")
        if meteo_mixing_height_m <= 0:
            raise ValueError("meteo_mixing_height_m must be positive")
        if earth_radius_m <= 0:
            raise ValueError("earth_radius_m must be positive")
        if reference_wavenumber_cm <= 0 or not np.isfinite(reference_wavenumber_cm):
            raise ValueError("reference_wavenumber_cm must be positive and finite")

        mipas = _load_mipas_profile(_resolve_mipas_profile_name(mipas_profile, latitude_deg))
        gdas = _load_gdas_profile(
            gdas_profile,
            observation_time,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            gdas_mode=gdas_mode,
            gdas_cache_dir=gdas_cache_dir,
            gdas_download_timeout_s=gdas_download_timeout_s,
        )
        level_heights_m = _molecfit_fixed_height_levels_m(
            observatory_altitude_m=observatory_altitude_m,
            top_altitude_m=top_altitude_m,
        )
        pressure_hpa, temperature_k, level_mixing_ratios = _merge_mipas_gdas_fixed_levels(
            level_heights_m,
            mipas=mipas,
            gdas=gdas,
        )

        pressure_hpa, temperature_k, level_mixing_ratios = _adapt_profile_to_local_meteo(
            level_heights_m,
            pressure_hpa,
            temperature_k,
            level_mixing_ratios,
            observatory_altitude_m=observatory_altitude_m,
            meteo_mixing_height_m=meteo_mixing_height_m,
            pressure_at_observatory_atm=pressure_at_observatory_atm,
            temperature_at_observatory_k=temperature_at_observatory_k,
            relative_humidity_percent=relative_humidity_percent,
        )
        if mixing_ratios is not None:
            for species, value in mixing_ratios.items():
                if value < 0:
                    raise ValueError(f"mixing ratio for {species} must be non-negative")
                level_mixing_ratios[str(species)] = np.full(level_heights_m.size, float(value))

        profile = cls(
            _layers_from_atmosphere_levels(
                level_heights_m,
                pressure_hpa,
                temperature_k,
                level_mixing_ratios,
                observatory_altitude_m=observatory_altitude_m,
                airmass=airmass,
                spherical_slant=spherical_slant,
                refracted_slant=refracted_slant,
                reference_wavenumber_cm=reference_wavenumber_cm,
                earth_radius_m=earth_radius_m,
            ),
            metadata={
                **dict(gdas.get("metadata", {})),
                "mipas_profile": _resolve_mipas_profile_name(mipas_profile, latitude_deg),
                "observation_time_utc": (
                    None
                    if _coerce_time(observation_time) is None
                    else _coerce_time(observation_time).utc.isot
                ),
                "latitude_deg": None if latitude_deg is None else float(latitude_deg),
                "longitude_deg": None if longitude_deg is None else float(longitude_deg),
                "observatory_altitude_m": float(observatory_altitude_m),
                "airmass": float(airmass),
                "pressure_at_observatory_atm": (
                    None
                    if pressure_at_observatory_atm is None
                    else float(pressure_at_observatory_atm)
                ),
                "temperature_at_observatory_k": (
                    None
                    if temperature_at_observatory_k is None
                    else float(temperature_at_observatory_k)
                ),
                "relative_humidity_percent": (
                    None
                    if relative_humidity_percent is None
                    else float(relative_humidity_percent)
                ),
                "reference_wavenumber_cm": float(reference_wavenumber_cm),
                "refracted_slant": bool(spherical_slant and refracted_slant),
            },
        )
        return profile.with_pwv_mm(pwv_mm) if pwv_mm is not None else profile

    @classmethod
    def from_fits_header_mipas_gdas(
        cls,
        header: Mapping[str, object],
        *,
        airmass: float | None = None,
        mipas_profile: str = "equ",
        gdas_profile: str | Path | None = None,
        gdas_mode: str = "auto",
        gdas_cache_dir: str | Path | None = None,
        gdas_download_timeout_s: float = 15.0,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        observatory_altitude_m: float | None = None,
        allow_default_observatory: bool = False,
        pressure_at_observatory_atm: float | None = None,
        temperature_at_observatory_k: float | None = None,
        relative_humidity_percent: float | None = None,
        pwv_mm: float | None = None,
        top_altitude_m: float = 120_000.0,
        mixing_ratios: Mapping[str, float] | None = None,
        spherical_slant: bool = True,
        refracted_slant: bool = True,
        reference_wavenumber_cm: float = 10_000.0,
    ) -> "AtmosphereProfile":
        """Build a MIPAS+GDAS atmosphere from FITS observing metadata."""

        pressure = pressure_at_observatory_atm
        if pressure is None:
            pressure_hpa = _header_float(
                header,
                (
                    "ESO TEL AMBI PRES START",
                    "HIERARCH ESO TEL AMBI PRES START",
                    "PRESSURE",
                    "PRES",
                    "AMBIPRES",
                    "WXPRESS",
                ),
                np.nan,
            )
            pressure = None if not np.isfinite(pressure_hpa) or pressure_hpa <= 0 else pressure_hpa / HPA_PER_ATM

        temperature = temperature_at_observatory_k
        if temperature is None:
            temperature_c = _header_float(
                header,
                (
                    "ESO TEL AMBI TEMP",
                    "HIERARCH ESO TEL AMBI TEMP",
                    "AIRTEMP",
                    "TEMP",
                    "WXOUTTMP",
                    "WXDOMTMP",
                ),
                np.nan,
            )
            temperature = None if not np.isfinite(temperature_c) else temperature_c + 273.15

        humidity = relative_humidity_percent
        if humidity is None:
            humidity_value = _header_float(
                header,
                (
                    "ESO TEL AMBI RHUM",
                    "HIERARCH ESO TEL AMBI RHUM",
                    "RELHUM",
                    "RELH",
                    "HUMIDITY",
                    "WXOUTHUM",
                    "WXDOMHUM",
                ),
                np.nan,
            )
            humidity = None if not np.isfinite(humidity_value) else humidity_value

        site, site_source = _header_observatory_site(header)
        latitude, longitude, altitude, coordinate_source = _resolve_header_observatory_geometry(
            header,
            site=site,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            observatory_altitude_m=observatory_altitude_m,
            allow_default_observatory=allow_default_observatory,
        )
        profile = cls.from_mipas_gdas(
            observation_time=_header_gdas_observation_time(header),
            latitude_deg=latitude,
            longitude_deg=longitude,
            observatory_altitude_m=altitude,
            airmass=_header_airmass(header) if airmass is None else float(airmass),
            mipas_profile=mipas_profile,
            gdas_profile=gdas_profile,
            gdas_mode=gdas_mode,
            gdas_cache_dir=gdas_cache_dir,
            gdas_download_timeout_s=gdas_download_timeout_s,
            pressure_at_observatory_atm=pressure,
            temperature_at_observatory_k=temperature,
            relative_humidity_percent=humidity,
            pwv_mm=pwv_mm,
            top_altitude_m=top_altitude_m,
            mixing_ratios=mixing_ratios,
            spherical_slant=spherical_slant,
            refracted_slant=refracted_slant,
            reference_wavenumber_cm=reference_wavenumber_cm,
        )
        metadata = dict(profile.metadata)
        metadata.update(
            {
                "observatory_site": None if site is None else site.name,
                "observatory_site_source": site_source,
                "observatory_coordinate_source": coordinate_source,
                "default_observatory_allowed": bool(allow_default_observatory),
            }
        )
        return cls(profile.layers, metadata=metadata)

    @classmethod
    def from_fits_header(
        cls,
        header: Mapping[str, object],
        *,
        airmass: float | None = None,
        observatory_altitude_m: float | None = None,
        allow_default_observatory: bool = False,
        pressure_at_observatory_atm: float | None = None,
        temperature_at_observatory_k: float | None = None,
        pwv_mm: float | None = None,
        n_layers: int = 48,
        top_altitude_m: float = 80_000.0,
        mixing_ratios: Mapping[str, float] | None = None,
        spherical_slant: bool = True,
    ) -> "AtmosphereProfile":
        """Build a standard atmosphere from a FITS-like header mapping."""

        resolved_airmass = (
            _header_airmass(header)
            if airmass is None
            else float(airmass)
        )
        site, site_source = _header_observatory_site(header)
        altitude, altitude_source = _resolve_header_observatory_altitude(
            header,
            site=site,
            observatory_altitude_m=observatory_altitude_m,
            allow_default_observatory=allow_default_observatory,
        )
        pressure = pressure_at_observatory_atm
        if pressure is None:
            pressure_hpa = _header_float(
                header,
                (
                    "ESO TEL AMBI PRES START",
                    "HIERARCH ESO TEL AMBI PRES START",
                    "PRESSURE",
                    "PRES",
                    "AMBIPRES",
                    "WXPRESS",
                ),
                np.nan,
            )
            pressure = None if not np.isfinite(pressure_hpa) or pressure_hpa <= 0 else pressure_hpa / 1013.25
        temperature = temperature_at_observatory_k
        if temperature is None:
            temperature_c = _header_float(
                header,
                (
                    "ESO TEL AMBI TEMP",
                    "HIERARCH ESO TEL AMBI TEMP",
                    "AIRTEMP",
                    "TEMP",
                    "WXOUTTMP",
                    "WXDOMTMP",
                ),
                np.nan,
            )
            temperature = None if not np.isfinite(temperature_c) else temperature_c + 273.15
        profile = cls.from_observatory_conditions(
            airmass=resolved_airmass,
            observatory_altitude_m=altitude,
            pressure_at_observatory_atm=pressure,
            temperature_at_observatory_k=temperature,
            pwv_mm=pwv_mm,
            n_layers=n_layers,
            top_altitude_m=top_altitude_m,
            mixing_ratios=mixing_ratios,
            spherical_slant=spherical_slant,
        )
        metadata = {
            "observatory_site": None if site is None else site.name,
            "observatory_site_source": site_source,
            "observatory_coordinate_source": altitude_source,
            "default_observatory_allowed": bool(allow_default_observatory),
            "observatory_altitude_m": altitude,
            "airmass": resolved_airmass,
            "pressure_at_observatory_atm": pressure,
            "temperature_at_observatory_k": temperature,
        }
        return cls(profile.layers, metadata=metadata)


def _resolve_mipas_profile_name(profile: str, latitude_deg: float | None) -> str:
    del latitude_deg  # Molecfit's default is equ.fits unless the user overrides it.
    key = str(profile).strip().lower()
    if key == "auto":
        return "equ"
    try:
        return MIPAS_PROFILE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted({"equ", "std", "tro", "auto"}))
        raise ValueError(f"unknown MIPAS profile {profile!r}; expected one of {valid}") from exc


def _package_data_path(*parts: str):
    return resources.files("genmolfit").joinpath("data", *parts)


def _load_mipas_profile(profile_name: str) -> dict[str, object]:
    resource = _package_data_path("profiles", "mipas", f"{profile_name}.fits")
    with resources.as_file(resource) as path:
        table = Table.read(path, hdu=1)
    columns = {name.upper(): name for name in table.colnames}
    required = ("HGT", "PRE", "TEM")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"MIPAS profile {profile_name!r} is missing columns: {', '.join(missing)}")
    height_m = np.asarray(table[columns["HGT"]], dtype=float) * 1000.0
    pressure_hpa = np.asarray(table[columns["PRE"]], dtype=float)
    temperature_k = np.asarray(table[columns["TEM"]], dtype=float)
    order = np.argsort(height_m)
    species = {}
    for upper, original in columns.items():
        if upper in required:
            continue
        values = np.asarray(table[original], dtype=float)[order]
        species[upper] = np.clip(values, 0.0, None) * 1.0e-6
    return {
        "height_m": height_m[order],
        "pressure_hpa": pressure_hpa[order],
        "temperature_k": temperature_k[order],
        "mixing_ratios": species,
    }


def _load_gdas_profile(
    gdas_profile: str | Path | None,
    observation_time: Time | datetime | str | float | None,
    *,
    latitude_deg: float | None,
    longitude_deg: float | None,
    gdas_mode: str,
    gdas_cache_dir: str | Path | None,
    gdas_download_timeout_s: float,
) -> dict[str, object]:
    if gdas_profile is not None and str(gdas_profile).strip().lower() not in {"auto", "none"}:
        table = Table.read(gdas_profile, hdu=1)
        metadata = {
            "gdas_source": "user",
            "gdas_profile": str(Path(gdas_profile).expanduser()),
        }
    else:
        resolved = resolve_time_local_gdas_profile(
            observation_time=observation_time,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            mode=gdas_mode,
            cache_dir=gdas_cache_dir,
            timeout_s=gdas_download_timeout_s,
        )
        if resolved is not None:
            table = Table.read(resolved.path, hdu=1)
            metadata = {
                "gdas_source": "eso_time_local",
                "gdas_resolution": resolved.source,
                "gdas_before": resolved.before_member,
                "gdas_after": resolved.after_member,
                "gdas_profile": str(resolved.path),
            }
        else:
            table = _load_average_gdas_profile(observation_time)
            metadata = {
                "gdas_source": "average",
                "gdas_average_index": _gdas_average_index_for_month(
                    _observation_month(observation_time)
                ),
            }

    columns = {name.lower(): name for name in table.colnames}
    required = ("press", "height", "temp", "relhum")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"GDAS profile is missing columns: {', '.join(missing)}")
    height_m = np.asarray(table[columns["height"]], dtype=float) * 1000.0
    pressure_hpa = np.asarray(table[columns["press"]], dtype=float)
    temperature_k = np.asarray(table[columns["temp"]], dtype=float)
    relative_humidity_percent = np.asarray(table[columns["relhum"]], dtype=float)
    order = np.argsort(height_m)
    height_m = height_m[order]
    pressure_hpa = pressure_hpa[order]
    temperature_k = temperature_k[order]
    relative_humidity_percent = relative_humidity_percent[order]
    h2o_ppmv = _relative_humidity_to_ppmv(
        temperature_k,
        pressure_hpa,
        relative_humidity_percent,
    )
    return {
        "height_m": height_m,
        "pressure_hpa": pressure_hpa,
        "temperature_k": temperature_k,
        "h2o_mixing_ratio": h2o_ppmv * 1.0e-6,
        "metadata": metadata,
    }


def _load_average_gdas_profile(
    observation_time: Time | datetime | str | float | None,
) -> Table:
        index = _gdas_average_index_for_month(_observation_month(observation_time))
        resource = _package_data_path("profiles", "gdas", f"GDAS_t0_s{index}.fits")
        with resources.as_file(resource) as path:
            return Table.read(path, hdu=1)


def _observation_month(observation_time: Time | datetime | str | float | None) -> int:
    time = _coerce_time(observation_time)
    if time is None:
        return 1
    return int(time.datetime.month)


def _coerce_time(value: Time | datetime | str | float | None) -> Time | None:
    if value is None:
        return None
    if isinstance(value, Time):
        return value
    if isinstance(value, datetime):
        return Time(value)
    if isinstance(value, str):
        try:
            return Time(value, scale="utc")
        except Exception:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 10_000.0:
        return Time(numeric, format="mjd", scale="utc")
    if 1900.0 < numeric < 2500.0:
        year = int(np.floor(numeric))
        fraction = numeric - year
        start = Time(f"{year:04d}-01-01T00:00:00", scale="utc")
        stop = Time(f"{year + 1:04d}-01-01T00:00:00", scale="utc")
        return start + fraction * (stop - start)
    return None


def _gdas_average_index_for_month(month: int) -> int:
    month = int(month)
    if month < 1 or month > 12:
        raise ValueError("month must be in the range 1..12")
    return (month % 12) // 2 + 1


def _molecfit_fixed_height_levels_m(
    *,
    observatory_altitude_m: float,
    top_altitude_m: float = 120_000.0,
) -> np.ndarray:
    """Return the fixed level grid used by Molecfit when ``LAYERS=TRUE``.

    Molecfit retains levels down to one kilometre below the observing site,
    then requires at least ``54 - i0`` levels and spaces the upper levels
    uniformly from 26 km to 120 km. Keeping the below-site levels is important:
    the local meteorology correction is applied before LBLRTM starts the ray at
    the observatory altitude.
    """

    altitude = float(observatory_altitude_m)
    top = float(top_altitude_m)
    if not np.isfinite(altitude):
        raise ValueError("observatory_altitude_m must be finite")
    if top <= max(altitude, MOLECFIT_FIXED_LOW_LEVELS_M[-1]):
        raise ValueError("top_altitude_m must exceed both the observatory and 26 km")

    candidates = np.nonzero(MOLECFIT_FIXED_LOW_LEVELS_M > altitude - 1_000.0)[0]
    i0 = int(candidates[0]) if candidates.size else MOLECFIT_FIXED_LOW_LEVELS_M.size - 1
    low = MOLECFIT_FIXED_LOW_LEVELS_M[i0:]
    n_rows = max(2, 54 - i0)
    n_upper = n_rows - low.size
    if n_upper <= 0:
        return low.copy()
    step = (top - low[-1]) / float(n_upper)
    upper = low[-1] + step * np.arange(1, n_upper + 1, dtype=float)
    return np.concatenate((low, upper))


def _merge_mipas_gdas_fixed_levels(
    altitude_m: np.ndarray,
    *,
    mipas: Mapping[str, object],
    gdas: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Port Molecfit's fixed-grid GDAS/MIPAS merge.

    GDAS supplies pressure, temperature and H2O below 20 km. The four overlap
    levels at 20, 22, 24 and 26 km contain respectively 80, 60, 40 and 20 per
    cent GDAS, with MIPAS supplying the remainder. Other molecules always come
    from MIPAS.
    """

    altitude = np.asarray(altitude_m, dtype=float)
    mipas_height = np.asarray(mipas["height_m"], dtype=float)
    gdas_height = np.asarray(gdas["height_m"], dtype=float)

    mipas_pressure = _interp_linear_clipped(altitude, mipas_height, np.asarray(mipas["pressure_hpa"], dtype=float))
    mipas_temperature = _interp_linear_clipped(
        altitude,
        mipas_height,
        np.asarray(mipas["temperature_k"], dtype=float),
    )
    gdas_pressure = _interp_linear_clipped(altitude, gdas_height, np.asarray(gdas["pressure_hpa"], dtype=float))
    gdas_temperature = _interp_linear_clipped(
        altitude,
        gdas_height,
        np.asarray(gdas["temperature_k"], dtype=float),
    )
    gdas_h2o = _interp_linear_clipped(
        altitude,
        gdas_height,
        np.asarray(gdas["h2o_mixing_ratio"], dtype=float),
    )

    standard_fraction = np.clip((altitude - 18_000.0) / 10_000.0, 0.0, 1.0)
    standard_fraction = np.where(altitude < 20_000.0, 0.0, standard_fraction)
    standard_fraction = np.where(altitude > 26_000.0, 1.0, standard_fraction)
    pressure = gdas_pressure * (1.0 - standard_fraction) + mipas_pressure * standard_fraction
    temperature = gdas_temperature * (1.0 - standard_fraction) + mipas_temperature * standard_fraction

    mixing_ratios: dict[str, np.ndarray] = {
        species: _interp_linear_clipped(altitude, mipas_height, np.asarray(values, dtype=float))
        for species, values in dict(mipas["mixing_ratios"]).items()
    }
    mipas_h2o = mixing_ratios.get("H2O", np.zeros_like(altitude))
    mixing_ratios["H2O"] = np.clip(
        gdas_h2o * (1.0 - standard_fraction) + mipas_h2o * standard_fraction,
        0.0,
        None,
    )
    return (
        np.clip(pressure, np.finfo(float).tiny, None),
        np.clip(temperature, np.finfo(float).tiny, None),
        mixing_ratios,
    )


def _layers_from_atmosphere_levels(
    altitude_m: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    mixing_ratios: Mapping[str, np.ndarray],
    *,
    observatory_altitude_m: float,
    airmass: float,
    spherical_slant: bool,
    refracted_slant: bool = True,
    reference_wavenumber_cm: float = 10_000.0,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> tuple[AtmosphereLayer, ...]:
    """Convert Molecfit/LBLRTM level values into GenMolFit path layers."""

    altitude = np.asarray(altitude_m, dtype=float)
    pressure = np.asarray(pressure_hpa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    site = float(observatory_altitude_m)
    above = altitude > site
    edges = np.concatenate(([site], altitude[above]))
    if edges.size < 2:
        raise ValueError("atmosphere levels do not extend above the observatory")

    level_pressure = np.concatenate(([_interp_log_clipped_scalar(site, altitude, pressure)], pressure[above]))
    level_temperature = np.concatenate(
        ([_interp_log_clipped_scalar(site, altitude, temperature)], temperature[above])
    )
    level_ratios = {}
    for species, values in mixing_ratios.items():
        array = np.asarray(values, dtype=float)
        site_value = _interp_log_clipped_scalar(site, altitude, np.maximum(array, np.finfo(float).tiny))
        level_ratios[species] = np.concatenate(([site_value], array[above]))

    # LBLRTM's ALAYER exponentially interpolates pressure, total density,
    # and molecular number density between atmospheric levels. It then forms
    # density-weighted PBAR/TBAR values. Integrating on altitude nodes lets the
    # same quadrature include LBLRTM's refracted spherical ray through
    # n(r) r sin(theta) = constant.
    vertical = np.diff(edges)
    fraction = 0.5 * (_LAYER_QUADRATURE_NODES[None, :] + 1.0)
    altitude_nodes = edges[:-1, None] + vertical[:, None] * fraction
    pressure_nodes = _exponential_level_interpolation(level_pressure, fraction)
    total_density_levels = level_pressure * 100.0 / (BOLTZMANN_J_PER_K * level_temperature)
    total_density_nodes = _exponential_level_interpolation(total_density_levels, fraction)
    temperature_nodes = pressure_nodes * 100.0 / (BOLTZMANN_J_PER_K * total_density_nodes)
    h2o_levels = np.asarray(level_ratios.get("H2O", np.zeros_like(level_pressure)), dtype=float)
    h2o_nodes = _exponential_level_interpolation(
        np.maximum(h2o_levels, np.finfo(float).tiny),
        fraction,
    )
    path_factor = _lblrtm_path_factor_dsdh(
        altitude_nodes,
        pressure_nodes,
        temperature_nodes,
        h2o_nodes,
        observer_altitude_m=float(edges[0]),
        observer_pressure_hpa=float(level_pressure[0]),
        observer_temperature_k=float(level_temperature[0]),
        observer_h2o_mixing_ratio=float(h2o_levels[0]),
        airmass=airmass,
        spherical_slant=spherical_slant,
        refracted_slant=refracted_slant,
        reference_wavenumber_cm=reference_wavenumber_cm,
        earth_radius_m=earth_radius_m,
    )
    weights = (
        0.5
        * vertical[:, None]
        * _LAYER_QUADRATURE_WEIGHTS[None, :]
        * path_factor
    )

    density_integral = np.sum(total_density_nodes * weights, axis=1)
    layer_pressure = np.sum(pressure_nodes * total_density_nodes * weights, axis=1) / density_integral
    layer_temperature = np.sum(pressure_nodes * weights, axis=1) / (
        BOLTZMANN_J_PER_K * density_integral / 100.0
    )
    equivalent_density = layer_pressure * 100.0 / (BOLTZMANN_J_PER_K * layer_temperature)
    effective_paths = density_integral / equivalent_density
    vertical_density_nodes = _exponential_level_interpolation(total_density_levels, fraction)
    vertical_density_integral = np.sum(
        vertical_density_nodes
        * (0.5 * vertical[:, None] * _LAYER_QUADRATURE_WEIGHTS[None, :]),
        axis=1,
    )
    effective_vertical_paths = vertical_density_integral / equivalent_density

    species_names = tuple(sorted(level_ratios))
    h2o_number_density = total_density_levels * h2o_levels / (1.0 + h2o_levels)
    dry_air_density = total_density_levels - h2o_number_density
    layer_ratios: dict[str, np.ndarray] = {}
    for species in species_names:
        ratio_levels = np.asarray(level_ratios[species], dtype=float)
        number_density_levels = (
            h2o_number_density
            if species == "H2O"
            else dry_air_density * ratio_levels
        )
        number_density_nodes = _exponential_level_interpolation(
            np.maximum(number_density_levels, np.finfo(float).tiny),
            fraction,
        )
        species_integral = np.sum(number_density_nodes * weights, axis=1)
        layer_ratios[species] = species_integral / density_integral

    return tuple(
        AtmosphereLayer(
            pressure_atm=float(layer_pressure[index] / HPA_PER_ATM),
            temperature_k=float(layer_temperature[index]),
            # LBLRTM carries molecular amounts independently from PBAR/TBAR.
            # AtmosphereLayer derives amounts from P/T/path, so this effective
            # path preserves the source-integrated total amount exactly.
            path_length_m=float(effective_paths[index]),
            mixing_ratios={
                species: float(layer_ratios[species][index])
                for species in species_names
            },
            vertical_path_length_m=float(effective_vertical_paths[index]),
        )
        for index in range(vertical.size)
    )


def _exponential_level_interpolation(level_values: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    values = np.asarray(level_values, dtype=float)
    log_values = np.log(np.maximum(values, np.finfo(float).tiny))
    return np.exp(log_values[:-1, None] + fraction * np.diff(log_values)[:, None])


def lblrtm_lowtran6_refractivity(
    pressure_hpa: np.ndarray | float,
    temperature_k: np.ndarray | float,
    h2o_mixing_ratio: np.ndarray | float,
    reference_wavenumber_cm: float,
) -> np.ndarray:
    """Return the LOWTRAN6 ``n - 1`` refractivity used by LBLRTM.

    This is the expression in ``lblatm.f90``. The water ratio is a volume
    fraction, so ``pressure * ratio`` is the H2O partial pressure in hPa.
    """

    pressure = np.asarray(pressure_hpa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    h2o = np.asarray(h2o_mixing_ratio, dtype=float)
    wavenumber = float(reference_wavenumber_cm)
    if not np.isfinite(wavenumber) or wavenumber <= 0:
        raise ValueError("reference_wavenumber_cm must be positive and finite")
    if np.any(pressure <= 0) or np.any(temperature <= 0):
        raise ValueError("pressure and temperature must be positive")

    dry_denominator_1 = 1.0 - (wavenumber / 1.14e5) ** 2
    dry_denominator_2 = 1.0 - (wavenumber / 6.24e4) ** 2
    if abs(dry_denominator_1) < 1.0e-12 or abs(dry_denominator_2) < 1.0e-12:
        raise ValueError("reference wavenumber is at a LOWTRAN6 refractivity pole")
    dry_coefficient = (
        83.42
        + 185.08 / dry_denominator_1
        + 4.11 / dry_denominator_2
    )
    water_coefficient = 43.49 - (wavenumber / 1.7e4) ** 2
    h2o_partial_pressure_hpa = pressure * np.clip(h2o, 0.0, None)
    return (
        dry_coefficient * (pressure * 288.15) / (1013.25 * temperature)
        - water_coefficient * h2o_partial_pressure_hpa / 1013.25
    ) * 1.0e-6


def _lblrtm_path_factor_dsdh(
    altitude_m: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    h2o_mixing_ratio: np.ndarray,
    *,
    observer_altitude_m: float,
    observer_pressure_hpa: float,
    observer_temperature_k: float,
    observer_h2o_mixing_ratio: float,
    airmass: float,
    spherical_slant: bool,
    refracted_slant: bool,
    reference_wavenumber_cm: float,
    earth_radius_m: float,
) -> np.ndarray:
    """Return ``ds/dh`` for the LBLRTM radial refracted ray."""

    altitude = np.asarray(altitude_m, dtype=float)
    if airmass <= 0:
        raise ValueError("airmass must be positive")
    if not spherical_slant:
        return np.full(altitude.shape, float(airmass), dtype=float)
    if airmass < 1.0:
        raise ValueError("airmass must be at least 1 for spherical slant geometry")
    if earth_radius_m <= 0:
        raise ValueError("earth_radius_m must be positive")

    cos_zenith = 1.0 / float(airmass)
    sin_zenith = np.sqrt(max(0.0, 1.0 - cos_zenith**2))
    if refracted_slant:
        refractivity = lblrtm_lowtran6_refractivity(
            pressure_hpa,
            temperature_k,
            h2o_mixing_ratio,
            reference_wavenumber_cm,
        )
        observer_refractivity = float(
            lblrtm_lowtran6_refractivity(
                observer_pressure_hpa,
                observer_temperature_k,
                observer_h2o_mixing_ratio,
                reference_wavenumber_cm,
            )
        )
    else:
        refractivity = np.zeros(altitude.shape, dtype=float)
        observer_refractivity = 0.0

    observer_nr = (1.0 + observer_refractivity) * (
        earth_radius_m + float(observer_altitude_m)
    )
    path_invariant = observer_nr * sin_zenith
    local_nr = (1.0 + refractivity) * (earth_radius_m + altitude)
    sine = path_invariant / local_nr
    if np.any(sine >= 1.0):
        raise ValueError("refracted ray does not reach every requested atmospheric layer")
    return 1.0 / np.sqrt(np.maximum(1.0 - sine**2, np.finfo(float).tiny))


def _atmosphere_path_coordinates_m(
    altitude_edges_m: np.ndarray,
    *,
    airmass: float,
    spherical_slant: bool,
    earth_radius_m: float,
) -> np.ndarray:
    edges = np.asarray(altitude_edges_m, dtype=float)
    if not spherical_slant:
        return (edges - edges[0]) * float(airmass)
    cos_z = 1.0 / float(airmass)
    sin_z = np.sqrt(max(0.0, 1.0 - cos_z * cos_z))
    observer_radius = earth_radius_m + float(edges[0])
    impact_parameter = observer_radius * sin_z
    shell_radius = earth_radius_m + edges
    return -observer_radius * cos_z + np.sqrt(np.maximum(shell_radius**2 - impact_parameter**2, 0.0))


def _altitude_on_path_m(
    path_coordinate_m: np.ndarray,
    *,
    observer_altitude_m: float,
    airmass: float,
    spherical_slant: bool,
    earth_radius_m: float,
) -> np.ndarray:
    if not spherical_slant:
        return observer_altitude_m + np.asarray(path_coordinate_m, dtype=float) / float(airmass)
    cos_z = 1.0 / float(airmass)
    sin_z = np.sqrt(max(0.0, 1.0 - cos_z * cos_z))
    observer_radius = earth_radius_m + observer_altitude_m
    impact_parameter = observer_radius * sin_z
    x0 = observer_radius * cos_z
    return np.sqrt(impact_parameter**2 + (x0 + np.asarray(path_coordinate_m, dtype=float)) ** 2) - earth_radius_m


def _mipas_gdas_height_edges_m(
    mipas_height_m: np.ndarray,
    gdas_height_m: np.ndarray,
    *,
    observatory_altitude_m: float,
    top_altitude_m: float,
    include_observatory: bool,
) -> np.ndarray:
    lower = float(observatory_altitude_m)
    heights = np.concatenate(
        [
            np.asarray(mipas_height_m, dtype=float),
            np.asarray(gdas_height_m, dtype=float),
            np.asarray([lower] if include_observatory else [], dtype=float),
            np.asarray([top_altitude_m], dtype=float),
        ]
    )
    heights = heights[np.isfinite(heights)]
    heights = heights[(heights >= lower) & (heights <= top_altitude_m)]
    heights = np.unique(np.round(heights, decimals=6))
    if heights.size < 2:
        raise ValueError("MIPAS/GDAS height grid does not span the requested atmosphere")
    if heights[0] > lower:
        heights = np.insert(heights, 0, lower)
    elif heights[0] < lower:
        heights[0] = lower
    if heights[-1] < top_altitude_m:
        heights = np.append(heights, top_altitude_m)
    if np.any(np.diff(heights) <= 0):
        heights = np.unique(heights)
    if heights.size < 2:
        raise ValueError("MIPAS/GDAS height grid collapsed to fewer than two edges")
    return heights.astype(float)


def _merge_mipas_gdas_at_altitudes(
    altitude_m: np.ndarray,
    *,
    mipas: Mapping[str, object],
    gdas: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    altitude = np.asarray(altitude_m, dtype=float)
    mipas_height = np.asarray(mipas["height_m"], dtype=float)
    gdas_height = np.asarray(gdas["height_m"], dtype=float)

    mipas_pressure = _interp_linear_clipped(altitude, mipas_height, np.asarray(mipas["pressure_hpa"], dtype=float))
    mipas_temperature = _interp_linear_clipped(altitude, mipas_height, np.asarray(mipas["temperature_k"], dtype=float))
    gdas_pressure = _interp_linear_clipped(altitude, gdas_height, np.asarray(gdas["pressure_hpa"], dtype=float))
    gdas_temperature = _interp_linear_clipped(altitude, gdas_height, np.asarray(gdas["temperature_k"], dtype=float))
    gdas_h2o = _interp_linear_clipped(altitude, gdas_height, np.asarray(gdas["h2o_mixing_ratio"], dtype=float))

    pressure_hpa = mipas_pressure.copy()
    temperature_k = mipas_temperature.copy()
    gdas_max = float(np.nanmax(gdas_height))
    merge_height = gdas_max * (1.0 + MIPAS_GDAS_MERGE_FRACTION)
    in_gdas = altitude <= gdas_max
    pressure_hpa[in_gdas] = gdas_pressure[in_gdas]
    temperature_k[in_gdas] = gdas_temperature[in_gdas]

    mixing_ratios: dict[str, np.ndarray] = {
        species: _interp_linear_clipped(altitude, mipas_height, np.asarray(values, dtype=float))
        for species, values in dict(mipas["mixing_ratios"]).items()
    }
    mipas_h2o = mixing_ratios.get("H2O", np.zeros_like(altitude))
    h2o = mipas_h2o.copy()
    h2o[in_gdas] = gdas_h2o[in_gdas]

    in_blend = (altitude > gdas_max) & (altitude < merge_height)
    if np.any(in_blend):
        blend_fraction = (merge_height - altitude[in_blend]) / (merge_height - gdas_max)
        pressure_hpa[in_blend] = mipas_pressure[in_blend] * (
            1.0 + _relative_deviation_at_height(gdas_height, gdas["pressure_hpa"], mipas_height, mipas["pressure_hpa"], gdas_max)
            * blend_fraction
        )
        temperature_k[in_blend] = mipas_temperature[in_blend] * (
            1.0 + _relative_deviation_at_height(gdas_height, gdas["temperature_k"], mipas_height, mipas["temperature_k"], gdas_max)
            * blend_fraction
        )
        h2o[in_blend] = mipas_h2o[in_blend] * (
            1.0
            + _relative_deviation_at_height(gdas_height, gdas["h2o_mixing_ratio"], mipas_height, mipas_h2o, gdas_max)
            * blend_fraction
        )
    mixing_ratios["H2O"] = np.clip(h2o, 0.0, None)
    return np.clip(pressure_hpa, np.finfo(float).tiny, None), np.clip(temperature_k, np.finfo(float).tiny, None), mixing_ratios


def _relative_deviation_at_height(
    reference_height: np.ndarray,
    reference_values: np.ndarray,
    baseline_height: np.ndarray,
    baseline_values: np.ndarray,
    height: float,
) -> float:
    reference = float(_interp_linear_clipped(np.asarray([height]), reference_height, np.asarray(reference_values, dtype=float))[0])
    baseline = float(_interp_linear_clipped(np.asarray([height]), baseline_height, np.asarray(baseline_values, dtype=float))[0])
    if baseline <= 0:
        return 0.0
    return reference / baseline - 1.0


def _adapt_profile_to_local_meteo(
    altitude_m: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    mixing_ratios: dict[str, np.ndarray],
    *,
    observatory_altitude_m: float,
    meteo_mixing_height_m: float,
    pressure_at_observatory_atm: float | None,
    temperature_at_observatory_k: float | None,
    relative_humidity_percent: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if (
        pressure_at_observatory_atm is None
        and temperature_at_observatory_k is None
        and relative_humidity_percent is None
    ):
        return pressure_hpa, temperature_k, mixing_ratios
    if meteo_mixing_height_m <= observatory_altitude_m:
        return pressure_hpa, temperature_k, mixing_ratios

    altitude = np.asarray(altitude_m, dtype=float)
    pressure = np.asarray(pressure_hpa, dtype=float).copy()
    temperature = np.asarray(temperature_k, dtype=float).copy()
    ratios = {species: np.asarray(values, dtype=float).copy() for species, values in mixing_ratios.items()}
    profile_pressure_at_site = _interp_log_clipped_scalar(observatory_altitude_m, altitude, pressure)
    profile_temperature_at_site = _interp_log_clipped_scalar(observatory_altitude_m, altitude, temperature)
    weight = 1.0 - np.clip(
        (altitude - observatory_altitude_m) / (meteo_mixing_height_m - observatory_altitude_m),
        0.0,
        1.0,
    )
    below_site = altitude < observatory_altitude_m

    if pressure_at_observatory_atm is not None and profile_pressure_at_site > 0:
        target_pressure_hpa = float(pressure_at_observatory_atm) * HPA_PER_ATM
        deviation = target_pressure_hpa / profile_pressure_at_site - 1.0
        pressure *= 1.0 + deviation * weight

    if temperature_at_observatory_k is not None and profile_temperature_at_site > 0:
        target_temperature_k = float(temperature_at_observatory_k)
        deviation = target_temperature_k / profile_temperature_at_site - 1.0
        temperature *= 1.0 + deviation * weight
        # Molecfit keeps temperature constant below the observatory because
        # those levels only stabilize the LBLRTM lower boundary.
        temperature[below_site] = target_temperature_k

    if relative_humidity_percent is not None and "H2O" in ratios:
        local_pressure_hpa = (
            float(pressure_at_observatory_atm) * HPA_PER_ATM
            if pressure_at_observatory_atm is not None
            else profile_pressure_at_site
        )
        local_temperature_k = (
            float(temperature_at_observatory_k)
            if temperature_at_observatory_k is not None
            else profile_temperature_at_site
        )
        target_h2o = float(
            _relative_humidity_to_ppmv(
                np.asarray([local_temperature_k]),
                np.asarray([local_pressure_hpa]),
                np.asarray([relative_humidity_percent]),
            )[0]
            * 1.0e-6
        )
        profile_h2o_at_site = _interp_log_clipped_scalar(observatory_altitude_m, altitude, ratios["H2O"])
        if profile_h2o_at_site > 0:
            deviation = target_h2o / profile_h2o_at_site - 1.0
            ratios["H2O"] = np.clip(ratios["H2O"] * (1.0 + deviation * weight), 0.0, None)
            ratios["H2O"][below_site] = target_h2o

    return np.clip(pressure, np.finfo(float).tiny, None), np.clip(temperature, np.finfo(float).tiny, None), ratios


def _relative_humidity_to_ppmv(
    temperature_k: np.ndarray,
    pressure_hpa: np.ndarray,
    relative_humidity_percent: np.ndarray,
) -> np.ndarray:
    """Murphy-Koop saturation vapor pressure conversion used by Molecfit."""

    temperature = np.maximum(np.asarray(temperature_k, dtype=float), np.finfo(float).tiny)
    pressure = np.maximum(np.asarray(pressure_hpa, dtype=float), np.finfo(float).tiny)
    humidity = np.asarray(relative_humidity_percent, dtype=float)
    log_temperature = np.log(temperature)
    log_ew = (
        54.842763
        - 6763.22 / temperature
        - 4.210 * log_temperature
        + 0.000367 * temperature
        + np.tanh(0.0415 * (temperature - 218.8))
        * (53.878 - 1331.22 / temperature - 9.44523 * log_temperature + 0.014025 * temperature)
    )
    log_ei = 9.550426 - 5723.265 / temperature + 3.53068 * log_temperature - 0.00728332 * temperature
    log_e = np.minimum(log_ew, log_ei)
    vapor_saturation_hpa = np.exp(np.minimum(log_e, np.log(np.finfo(float).max))) / 100.0
    water_pressure_hpa = np.minimum(humidity, 100.0) / 100.0 * vapor_saturation_hpa
    return np.maximum(water_pressure_hpa / pressure * 1.0e6, 0.0)


def _interp_linear_clipped(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    return np.interp(np.asarray(x, dtype=float), xp, fp, left=fp[0], right=fp[-1])


def _interp_log_clipped_scalar(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    values = np.asarray(fp, dtype=float)
    values = np.clip(values, np.finfo(float).tiny, None)
    return float(np.exp(_interp_linear_clipped(np.asarray([x], dtype=float), xp, np.log(values))[0]))


def _header_observation_time(header: Mapping[str, object]) -> Time | None:
    for key in ("MJD-OBS", "MJDOBS", "MJD", "JD"):
        value = _header_float(header, (key,), np.nan)
        if np.isfinite(value):
            return Time(value, format="jd" if key == "JD" else "mjd", scale="utc")
    for key in ("DATE_BEG", "DATE-BEG", "DATE-AVG", "DATE-OBS", "UTC-DATE", "DATE"):
        if key not in header:
            continue
        try:
            return Time(str(header[key]), scale="utc")
        except Exception:
            continue
    return None


def _header_gdas_observation_time(header: Mapping[str, object]) -> Time | None:
    """Return the timestamp Molecfit uses for GDAS interpolation.

    ESO Molecfit constructs the calendar date from ``DATE-OBS``/the observing
    date but takes seconds since midnight from the ``UTC`` ambient parameter.
    That can differ from MJD-OBS by a few seconds in reduced archive products.
    """

    utc_seconds = _header_time_of_day_seconds(
        header,
        ("UTC", "ESO TEL UTC", "HIERARCH ESO TEL UTC", "TIME-OBS", "UT", "TIME"),
    )
    if np.isfinite(utc_seconds) and 0.0 <= utc_seconds <= 86_400.0:
        base = None
        for key in ("DATE-OBS", "UTC-DATE", "DATE"):
            if key not in header:
                continue
            try:
                observed = Time(str(header[key]), scale="utc")
                base = Time(observed.utc.datetime.strftime("%Y-%m-%dT00:00:00"), scale="utc")
                break
            except Exception:
                continue
        if base is None:
            observed = _header_observation_time(header)
            if observed is not None:
                base = Time(observed.utc.datetime.strftime("%Y-%m-%dT00:00:00"), scale="utc")
        if base is not None:
            return base + TimeDelta(float(utc_seconds), format="sec")
    return _header_observation_time(header)


def _header_time_of_day_seconds(
    header: Mapping[str, object],
    keys: tuple[str, ...],
) -> float:
    for key in keys:
        if key not in header:
            continue
        try:
            value = header[key]
        except Exception:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            parts = str(value).strip().split(":")
            if len(parts) != 3:
                continue
            try:
                hours, minutes, seconds_part = map(float, parts)
            except ValueError:
                continue
            seconds = hours * 3600.0 + minutes * 60.0 + seconds_part
        if np.isfinite(seconds) and 0.0 <= seconds <= 86_400.0:
            return float(seconds)
    return np.nan


def _normalize_observatory_name(value: object) -> str:
    text = "".join(character if str(character).isalnum() else " " for character in str(value).upper())
    return " ".join(text.split())


def _header_observatory_site(
    header: Mapping[str, object],
) -> tuple[_ObservatorySite | None, str]:
    unidentified_key = None
    for key in ("OBSERVAT", "OBSERVATORY", "SITE", "TELESCOP"):
        if key not in header:
            continue
        normalized = _normalize_observatory_name(header[key])
        if not normalized:
            continue
        if unidentified_key is None:
            unidentified_key = key
        site_key = _OBSERVATORY_ALIASES.get(normalized)
        if site_key is None:
            matches = {
                candidate
                for alias, candidate in _OBSERVATORY_ALIASES.items()
                if normalized.startswith(f"{alias} ") or normalized.endswith(f" {alias}")
            }
            if len(matches) == 1:
                site_key = matches.pop()
        if site_key is not None:
            return _OBSERVATORY_SITES[site_key], f"fits:{key}"
    if unidentified_key is not None:
        return None, f"unrecognized:{unidentified_key}"
    return None, "absent"


def _resolve_header_observatory_geometry(
    header: Mapping[str, object],
    *,
    site: _ObservatorySite | None,
    latitude_deg: float | None,
    longitude_deg: float | None,
    observatory_altitude_m: float | None,
    allow_default_observatory: bool,
) -> tuple[float, float, float, str]:
    latitude, latitude_source = _resolve_coordinate_value(
        explicit=latitude_deg,
        header_value=_header_observatory_latitude_value(header),
        site_value=None if site is None else site.latitude_deg,
        default_value=DEFAULT_OBSERVATORY_LATITUDE_DEG,
        name="latitude_deg",
        allow_default=allow_default_observatory,
    )
    longitude, longitude_source = _resolve_coordinate_value(
        explicit=longitude_deg,
        header_value=_header_observatory_longitude_value(header),
        site_value=None if site is None else site.longitude_deg,
        default_value=DEFAULT_OBSERVATORY_LONGITUDE_DEG,
        name="longitude_deg",
        allow_default=allow_default_observatory,
    )
    altitude, altitude_source = _resolve_coordinate_value(
        explicit=observatory_altitude_m,
        header_value=_header_observatory_altitude_value(header),
        site_value=None if site is None else site.altitude_m,
        default_value=DEFAULT_OBSERVATORY_ALTITUDE_M,
        name="observatory_altitude_m",
        allow_default=allow_default_observatory,
    )
    source = _join_coordinate_sources(latitude_source, longitude_source, altitude_source)
    return latitude, longitude, altitude, source


def _resolve_header_observatory_altitude(
    header: Mapping[str, object],
    *,
    site: _ObservatorySite | None,
    observatory_altitude_m: float | None,
    allow_default_observatory: bool,
) -> tuple[float, str]:
    return _resolve_coordinate_value(
        explicit=observatory_altitude_m,
        header_value=_header_observatory_altitude_value(header),
        site_value=None if site is None else site.altitude_m,
        default_value=DEFAULT_OBSERVATORY_ALTITUDE_M,
        name="observatory_altitude_m",
        allow_default=allow_default_observatory,
    )


def _resolve_coordinate_value(
    *,
    explicit: float | None,
    header_value: float,
    site_value: float | None,
    default_value: float,
    name: str,
    allow_default: bool,
) -> tuple[float, str]:
    if explicit is not None:
        value = float(explicit)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value, "explicit"
    if np.isfinite(header_value):
        return float(header_value), "fits_header"
    if site_value is not None:
        return float(site_value), "observatory_registry"
    if allow_default:
        return float(default_value), "default_paranal"
    raise ValueError(
        f"cannot resolve {name} from the FITS header; provide an explicit value, "
        "include standard observatory coordinates/site metadata, or set "
        "allow_default_observatory=True to explicitly use the Paranal default"
    )


def _join_coordinate_sources(*sources: str) -> str:
    ordered = tuple(dict.fromkeys(sources))
    return ordered[0] if len(ordered) == 1 else "+".join(ordered)


def _header_observatory_latitude_value(header: Mapping[str, object]) -> float:
    return _header_float(header, _LATITUDE_HEADER_KEYS, np.nan)


def _header_observatory_longitude_value(header: Mapping[str, object]) -> float:
    east_positive = _header_float(header, _LONGITUDE_HEADER_KEYS, np.nan)
    if np.isfinite(east_positive):
        return float(east_positive)

    # Legacy Keck headers use LONGITUD as degrees west, unlike the FITS
    # east-positive LONG-OBS/OBSGEO-L convention.
    legacy = _header_float(header, ("LONGITUD",), np.nan)
    telescope = str(header.get("TELESCOP", "")).strip().upper()
    if np.isfinite(legacy):
        return float(-abs(legacy) if telescope.startswith("KECK") else legacy)
    return np.nan


def _header_observatory_altitude_value(header: Mapping[str, object]) -> float:
    return _header_float(header, _ALTITUDE_HEADER_KEYS, np.nan)


def _header_observatory_latitude_deg(
    header: Mapping[str, object],
    *,
    site: _ObservatorySite | None = None,
) -> float:
    latitude = _header_observatory_latitude_value(header)
    if np.isfinite(latitude):
        return float(latitude)
    if site is not None:
        return site.latitude_deg
    return DEFAULT_OBSERVATORY_LATITUDE_DEG


def _header_observatory_coordinate_source(
    header: Mapping[str, object],
    *,
    site: _ObservatorySite | None,
) -> str:
    latitude = _header_observatory_latitude_value(header)
    longitude = _header_observatory_longitude_value(header)
    altitude = _header_observatory_altitude_value(header)
    count = sum(np.isfinite(value) for value in (latitude, longitude, altitude))
    if count == 3:
        return "fits_header"
    if count > 0 and site is not None:
        return "fits_header+observatory_registry"
    if site is not None:
        return "observatory_registry"
    if count > 0:
        return "fits_header+default"
    return "default_paranal"


def _header_observatory_longitude_deg(
    header: Mapping[str, object],
    *,
    site: _ObservatorySite | None = None,
) -> float:
    longitude = _header_observatory_longitude_value(header)
    if np.isfinite(longitude):
        return float(longitude)
    if site is not None:
        return site.longitude_deg
    return DEFAULT_OBSERVATORY_LONGITUDE_DEG


def _header_observatory_altitude_m(
    header: Mapping[str, object],
    *,
    site: _ObservatorySite | None = None,
) -> float:
    altitude = _header_observatory_altitude_value(header)
    if np.isfinite(altitude):
        return float(altitude)
    if site is not None:
        return site.altitude_m
    return DEFAULT_OBSERVATORY_ALTITUDE_M


def _spherical_layer_path_lengths_m(
    altitude_edges_m: np.ndarray,
    *,
    airmass: float,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """Ray path lengths through concentric atmospheric shells.

    The requested airmass is interpreted as the plane-parallel sec(z) value at
    the observatory. Curvature then reduces the effective path length in upper
    shells, which is a better self-contained approximation than applying the
    same secant factor to every layer.
    """

    edges = np.asarray(altitude_edges_m, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("altitude_edges_m must be a one-dimensional edge grid")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("altitude_edges_m must be strictly increasing")
    if airmass <= 0:
        raise ValueError("airmass must be positive")
    if earth_radius_m <= 0:
        raise ValueError("earth_radius_m must be positive")
    if np.isclose(airmass, 1.0):
        return np.diff(edges)

    cos_z = 1.0 / float(airmass)
    if cos_z <= 0 or cos_z > 1:
        raise ValueError("airmass must be at least 1 for spherical slant geometry")
    sin_z = np.sqrt(max(0.0, 1.0 - cos_z**2))
    observer_radius = earth_radius_m + float(edges[0])
    impact_parameter = observer_radius * sin_z
    shell_radius = earth_radius_m + edges
    radicand = shell_radius**2 - impact_parameter**2
    if np.any(radicand < 0):
        raise ValueError("spherical geometry failed for the requested airmass")
    distance_to_shell = -observer_radius * cos_z + np.sqrt(np.maximum(radicand, 0.0))
    return np.diff(distance_to_shell)


def _standard_temperature_at_altitude_k(altitude_m: float) -> float:
    altitude = max(0.0, float(altitude_m))
    if altitude <= 11_000.0:
        return 288.15 - 0.0065 * altitude
    if altitude <= 20_000.0:
        return 216.65
    if altitude <= 32_000.0:
        return 216.65 + 0.0010 * (altitude - 20_000.0)
    if altitude <= 47_000.0:
        return 228.65 + 0.0028 * (altitude - 32_000.0)
    return 270.65


def _standard_pressure_at_altitude_atm(altitude_m: float) -> float:
    altitude = max(0.0, float(altitude_m))
    scale_height_m = 8_400.0
    return float(np.exp(-altitude / scale_height_m))


def _header_float(header: Mapping[str, object], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in header:
            try:
                return float(header[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _header_airmass(header: Mapping[str, object]) -> float:
    start = _header_float(header, ("ESO TEL AIRM START", "HIERARCH ESO TEL AIRM START"), np.nan)
    end = _header_float(header, ("ESO TEL AIRM END", "HIERARCH ESO TEL AIRM END"), np.nan)
    if np.isfinite(start) and np.isfinite(end) and start > 0 and end > 0:
        return 0.5 * (start + end)
    for key in ("AIRMASS", "ESO OBS AIRM", "HIERARCH ESO OBS AIRM"):
        value = _header_float(header, (key,), np.nan)
        if np.isfinite(value) and value > 0:
            return value
    return 1.0


def _resolve_pressure_column(table: Table, requested: str | None) -> str:
    if requested is not None:
        if requested not in table.colnames:
            raise ValueError(f"pressure column {requested!r} not found")
        return requested
    for candidate in ("pressure_atm", "pressure_hpa", "pressure_mbar", "pressure_pa"):
        if candidate in table.colnames:
            return candidate
    raise ValueError("atmosphere table needs pressure_atm, pressure_hpa, pressure_mbar, or pressure_pa")


def _resolve_path_length_column(table: Table, requested: str | None) -> str:
    if requested is not None:
        if requested not in table.colnames:
            raise ValueError(f"path length column {requested!r} not found")
        return requested
    for candidate in ("path_length_m", "thickness_m", "dz_m"):
        if candidate in table.colnames:
            return candidate
    raise ValueError("atmosphere table needs path_length_m, thickness_m, or dz_m")


def _resolve_vertical_path_length_column(table: Table) -> str | None:
    for candidate in ("vertical_path_length_m", "vertical_thickness_m", "vertical_dz_m"):
        if candidate in table.colnames:
            return candidate
    return None


def _pressure_to_atm(table: Table, pressure_col: str) -> np.ndarray:
    pressure = np.asarray(table[pressure_col], dtype=float)
    lower = pressure_col.lower()
    if lower.endswith("_pa"):
        return pressure / PA_PER_ATM
    if lower.endswith("_hpa") or lower.endswith("_mbar"):
        return pressure / 1013.25
    return pressure


def _mixing_columns(table: Table, prefixes: tuple[str, ...]) -> dict[str, str]:
    columns = {}
    for colname in table.colnames:
        for prefix in prefixes:
            if colname.startswith(prefix):
                species = colname[len(prefix) :]
                if species:
                    columns[colname] = species
                break
    return columns
