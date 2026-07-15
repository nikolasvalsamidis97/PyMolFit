from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
from astropy.table import Table

LBLRTM_DENSITY_SHIFT_LINE_SOURCES = frozenset({"aer_lnfl_tape8", "lblrtm_tape3", "lblrtm_tape8"})
LBLRTM_BROADENER_SPECIES = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2")


def _read_line_data_provenance(path: Path, table: Table) -> dict[str, Any]:
    encoded = table.meta.get("data_provenance_json")
    if encoded is None:
        provenance: dict[str, Any] = {}
    else:
        try:
            decoded = json.loads(str(encoded))
        except json.JSONDecodeError as exc:
            raise ValueError("line-list data_provenance_json metadata is invalid") from exc
        if not isinstance(decoded, dict):
            raise ValueError("line-list data provenance must decode to an object")
        provenance = decoded

    request_hash = table.meta.get("line_data_request_sha256")
    source = table.meta.get("line_data_source")
    if request_hash is not None:
        provenance["request_sha256"] = str(request_hash)
    if source is not None:
        provenance["source"] = str(source)

    manifest_path = path.with_suffix(".json")
    if request_hash is None or not manifest_path.is_file():
        return provenance

    from .provenance import file_sha256

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid line-data manifest beside {path}") from exc
    if manifest.get("request_sha256") != str(request_hash):
        raise ValueError("line-data manifest request hash does not match the line table")
    try:
        expected_table_hash = manifest["files"]["table"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("line-data manifest has no line-table hash") from exc
    if expected_table_hash != file_sha256(path):
        raise ValueError("line-data table hash does not match its manifest")
    provenance.update(
        {
            "manifest_name": manifest_path.name,
            "manifest_sha256": file_sha256(manifest_path),
            "request": manifest.get("request", {}),
            "database_edition": manifest.get("database_edition"),
            "created_utc": manifest.get("created_utc"),
            "citations": manifest.get("citations", {}),
        }
    )
    return provenance


@dataclass(frozen=True)
class LineList:
    """Molecular transition data used by the Python telluric model.

    Parameters are stored as arrays so model evaluation can use broadcasting:

    - ``wavelength``: line center in microns
    - ``strength``: relative integrated optical-depth scale
    - ``sigma``: Gaussian Doppler width in microns
    - ``gamma``: Lorentzian pressure width in microns
    - ``species``: molecular species label, for example ``H2O`` or ``CO2``

    HITRAN-style fields are optional. When present, the physical radiative
    transfer model uses them instead of the simplified wavelength-width fields.
    """

    wavelength: np.ndarray
    strength: np.ndarray
    sigma: np.ndarray
    gamma: np.ndarray
    species: np.ndarray
    wavenumber: np.ndarray | None = None
    mol_id: np.ndarray | None = None
    iso_id: np.ndarray | None = None
    air_width: np.ndarray | None = None
    self_width: np.ndarray | None = None
    lower_state_energy: np.ndarray | None = None
    temperature_exponent: np.ndarray | None = None
    pressure_shift: np.ndarray | None = None
    molecular_mass_amu: np.ndarray | None = None
    global_iso_id: np.ndarray | None = None
    natural_abundance: np.ndarray | None = None
    isotopologue_abundance_scale: np.ndarray | None = None
    broadener_flags: np.ndarray | None = None
    broadener_widths: np.ndarray | None = None
    broadener_temperature_exponents: np.ndarray | None = None
    broadener_pressure_shifts: np.ndarray | None = None
    line_flags: np.ndarray | None = None
    line_coupling_a: np.ndarray | None = None
    line_coupling_b: np.ndarray | None = None
    speed_dependence: np.ndarray | None = None
    reference_temperature: float = 296.0
    line_source: str = "generic"
    data_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength, dtype=float)
        strength = np.asarray(self.strength, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        gamma = np.asarray(self.gamma, dtype=float)
        species = np.asarray(self.species, dtype=str)
        optional = {
            "wavenumber": self.wavenumber,
            "mol_id": self.mol_id,
            "iso_id": self.iso_id,
            "air_width": self.air_width,
            "self_width": self.self_width,
            "lower_state_energy": self.lower_state_energy,
            "temperature_exponent": self.temperature_exponent,
            "pressure_shift": self.pressure_shift,
            "molecular_mass_amu": self.molecular_mass_amu,
            "global_iso_id": self.global_iso_id,
            "natural_abundance": self.natural_abundance,
            "isotopologue_abundance_scale": self.isotopologue_abundance_scale,
            "line_flags": self.line_flags,
            "speed_dependence": self.speed_dependence,
        }
        broadener_optional = {
            "broadener_flags": self.broadener_flags,
            "broadener_widths": self.broadener_widths,
            "broadener_temperature_exponents": self.broadener_temperature_exponents,
            "broadener_pressure_shifts": self.broadener_pressure_shifts,
        }
        coupling_optional = {
            "line_coupling_a": self.line_coupling_a,
            "line_coupling_b": self.line_coupling_b,
        }

        shapes = {arr.shape for arr in (wavelength, strength, sigma, gamma, species)}
        if len(shapes) != 1:
            raise ValueError("all line-list arrays must have the same shape")
        if wavelength.ndim != 1:
            raise ValueError("line-list arrays must be one-dimensional")
        if np.any(strength < 0):
            raise ValueError("line strengths must be non-negative")
        if np.any(sigma <= 0) or np.any(gamma < 0):
            raise ValueError("line widths must be positive/non-negative")
        if not self.line_source:
            raise ValueError("line_source must be a non-empty string")
        if self.data_provenance is not None and not isinstance(self.data_provenance, Mapping):
            raise ValueError("data_provenance must be a mapping")

        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "species", species)
        for name, value in optional.items():
            if value is None:
                continue
            dtype = int if name in {"global_iso_id", "line_flags"} else float
            array = np.asarray(value, dtype=dtype)
            if array.shape != wavelength.shape:
                raise ValueError(f"{name} must have the same shape as wavelength")
            object.__setattr__(self, name, array)
        for name, value in broadener_optional.items():
            if value is None:
                continue
            dtype = int if name == "broadener_flags" else float
            array = np.asarray(value, dtype=dtype)
            if array.shape != (wavelength.size, len(LBLRTM_BROADENER_SPECIES)):
                raise ValueError(f"{name} must have shape (n_lines, {len(LBLRTM_BROADENER_SPECIES)})")
            object.__setattr__(self, name, array)
        for name, value in coupling_optional.items():
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != (wavelength.size, 4):
                raise ValueError(f"{name} must have shape (n_lines, 4)")
            object.__setattr__(self, name, array)
        object.__setattr__(self, "line_source", str(self.line_source))
        provenance = {} if self.data_provenance is None else dict(self.data_provenance)
        try:
            provenance = json.loads(json.dumps(provenance, sort_keys=True, ensure_ascii=True))
        except (TypeError, ValueError) as exc:
            raise ValueError("data_provenance must contain JSON-compatible values") from exc
        object.__setattr__(self, "data_provenance", MappingProxyType(provenance))

    @property
    def species_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.species.tolist())))

    @property
    def has_hitran_parameters(self) -> bool:
        return all(
            value is not None
            for value in (
                self.wavenumber,
                self.air_width,
                self.self_width,
                self.lower_state_energy,
                self.temperature_exponent,
                self.pressure_shift,
                self.molecular_mass_amu,
            )
        )

    @property
    def has_isotopologue_metadata(self) -> bool:
        return self.global_iso_id is not None and self.natural_abundance is not None

    @property
    def pressure_shift_convention(self) -> str:
        if self.line_source in LBLRTM_DENSITY_SHIFT_LINE_SOURCES:
            return "lblrtm_density"
        return "hitran"

    @property
    def has_broadener_parameters(self) -> bool:
        return all(
            value is not None
            for value in (
                self.broadener_flags,
                self.broadener_widths,
                self.broadener_temperature_exponents,
                self.broadener_pressure_shifts,
            )
        )

    def select_range(self, wavelength_min: float, wavelength_max: float, *, margin: float = 0.0) -> "LineList":
        keep = (self.wavelength >= wavelength_min - margin) & (self.wavelength <= wavelength_max + margin)
        return self.select(keep)

    def select_species(self, species: Iterable[str]) -> "LineList":
        wanted = set(species)
        keep = np.array([name in wanted for name in self.species], dtype=bool)
        return self.select(keep)

    def select(self, keep: np.ndarray) -> "LineList":
        keep = np.asarray(keep, dtype=bool)
        optional = {
            "wavenumber": self.wavenumber,
            "mol_id": self.mol_id,
            "iso_id": self.iso_id,
            "air_width": self.air_width,
            "self_width": self.self_width,
            "lower_state_energy": self.lower_state_energy,
            "temperature_exponent": self.temperature_exponent,
            "pressure_shift": self.pressure_shift,
            "molecular_mass_amu": self.molecular_mass_amu,
            "global_iso_id": self.global_iso_id,
            "natural_abundance": self.natural_abundance,
            "isotopologue_abundance_scale": self.isotopologue_abundance_scale,
            "broadener_flags": self.broadener_flags,
            "broadener_widths": self.broadener_widths,
            "broadener_temperature_exponents": self.broadener_temperature_exponents,
            "broadener_pressure_shifts": self.broadener_pressure_shifts,
            "line_flags": self.line_flags,
            "line_coupling_a": self.line_coupling_a,
            "line_coupling_b": self.line_coupling_b,
            "speed_dependence": self.speed_dependence,
        }
        return LineList(
            wavelength=self.wavelength[keep],
            strength=self.strength[keep],
            sigma=self.sigma[keep],
            gamma=self.gamma[keep],
            species=self.species[keep],
            reference_temperature=self.reference_temperature,
            line_source=self.line_source,
            data_provenance=self.data_provenance,
            **{name: None if value is None else value[keep] for name, value in optional.items()},
        )

    def with_isotopologue_metadata(
        self,
        metadata,
        *,
        abundance_overrides: Mapping[tuple[int, int] | int, float] | None = None,
    ) -> "LineList":
        """Attach HITRAN isotopologue metadata and optional abundance overrides.

        ``abundance_overrides`` values are absolute isotopologue abundances.
        HITRAN intensities are already weighted by natural terrestrial
        abundance, so the runtime optical depth uses ``override / natural`` as
        a multiplier. With no overrides, all multipliers are one.
        """

        if self.mol_id is None or self.iso_id is None:
            raise ValueError("mol_id and iso_id are required to attach isotopologue metadata")
        mol_id = np.asarray(self.mol_id, dtype=int)
        iso_id = np.asarray(self.iso_id, dtype=int)
        global_ids = metadata.global_ids_for_pair(mol_id, iso_id)
        natural_abundance = metadata.abundance_for_pair(mol_id, iso_id)
        molar_mass = metadata.molar_mass_for_pair(mol_id, iso_id)

        finite_metadata = np.isfinite(global_ids) & np.isfinite(natural_abundance) & np.isfinite(molar_mass)
        global_iso_id = np.full(mol_id.shape, -1, dtype=int)
        global_iso_id[finite_metadata] = global_ids[finite_metadata].astype(int)
        updated_mass = (
            np.asarray(self.molecular_mass_amu, dtype=float).copy()
            if self.molecular_mass_amu is not None
            else np.full(mol_id.shape, np.nan, dtype=float)
        )
        updated_mass[np.isfinite(molar_mass)] = molar_mass[np.isfinite(molar_mass)]

        abundance_scale = np.ones(mol_id.shape, dtype=float)
        if abundance_overrides:
            for key, abundance in abundance_overrides.items():
                abundance = float(abundance)
                if abundance <= 0:
                    raise ValueError("isotopologue abundance overrides must be positive")
                if isinstance(key, tuple):
                    keep = (mol_id == int(key[0])) & (iso_id == int(key[1]))
                else:
                    keep = global_iso_id == int(key)
                if not np.any(keep):
                    continue
                natural = natural_abundance[keep]
                valid = np.isfinite(natural) & (natural > 0)
                abundance_scale[keep] = np.where(valid, abundance / natural, abundance_scale[keep])

        return LineList(
            wavelength=self.wavelength,
            strength=self.strength,
            sigma=self.sigma,
            gamma=self.gamma,
            species=self.species,
            wavenumber=self.wavenumber,
            mol_id=self.mol_id,
            iso_id=self.iso_id,
            air_width=self.air_width,
            self_width=self.self_width,
            lower_state_energy=self.lower_state_energy,
            temperature_exponent=self.temperature_exponent,
            pressure_shift=self.pressure_shift,
            molecular_mass_amu=updated_mass,
            global_iso_id=global_iso_id,
            natural_abundance=natural_abundance,
            isotopologue_abundance_scale=abundance_scale,
            broadener_flags=self.broadener_flags,
            broadener_widths=self.broadener_widths,
            broadener_temperature_exponents=self.broadener_temperature_exponents,
            broadener_pressure_shifts=self.broadener_pressure_shifts,
            line_flags=self.line_flags,
            line_coupling_a=self.line_coupling_a,
            line_coupling_b=self.line_coupling_b,
            speed_dependence=self.speed_dependence,
            reference_temperature=self.reference_temperature,
            line_source=self.line_source,
            data_provenance=self.data_provenance,
        )

    def to_table(self) -> Table:
        table = Table()
        table["wavelength"] = self.wavelength
        table["strength"] = self.strength
        table["sigma"] = self.sigma
        table["gamma"] = self.gamma
        table["species"] = self.species
        optional = {
            "wavenumber": self.wavenumber,
            "mol_id": self.mol_id,
            "iso_id": self.iso_id,
            "air_width": self.air_width,
            "self_width": self.self_width,
            "lower_state_energy": self.lower_state_energy,
            "temperature_exponent": self.temperature_exponent,
            "pressure_shift": self.pressure_shift,
            "molecular_mass_amu": self.molecular_mass_amu,
            "global_iso_id": self.global_iso_id,
            "natural_abundance": self.natural_abundance,
            "isotopologue_abundance_scale": self.isotopologue_abundance_scale,
            "broadener_flags": self.broadener_flags,
            "broadener_widths": self.broadener_widths,
            "broadener_temperature_exponents": self.broadener_temperature_exponents,
            "broadener_pressure_shifts": self.broadener_pressure_shifts,
            "line_flags": self.line_flags,
            "line_coupling_a": self.line_coupling_a,
            "line_coupling_b": self.line_coupling_b,
            "speed_dependence": self.speed_dependence,
        }
        for name, value in optional.items():
            if value is not None:
                table[name] = value
        table.meta["reference_temperature"] = self.reference_temperature
        table.meta["line_source"] = self.line_source
        if self.data_provenance:
            table.meta["data_provenance_json"] = json.dumps(
                dict(self.data_provenance), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        return table

    def write(self, path: str | Path, *, format: str = "ascii.ecsv") -> None:
        self.to_table().write(path, format=format, overwrite=True)

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        wavelength_col: str = "wavelength",
        strength_col: str = "strength",
        sigma_col: str = "sigma",
        gamma_col: str = "gamma",
        species_col: str = "species",
    ) -> "LineList":
        input_path = Path(path)
        table = Table.read(input_path)
        optional = {}
        for name in (
            "wavenumber",
            "mol_id",
            "iso_id",
            "air_width",
            "self_width",
            "lower_state_energy",
            "temperature_exponent",
            "pressure_shift",
            "molecular_mass_amu",
            "global_iso_id",
            "natural_abundance",
            "isotopologue_abundance_scale",
            "broadener_flags",
            "broadener_widths",
            "broadener_temperature_exponents",
            "broadener_pressure_shifts",
            "line_flags",
            "line_coupling_a",
            "line_coupling_b",
            "speed_dependence",
        ):
            if name not in table.colnames:
                optional[name] = None
            else:
                dtype = int if name in ("global_iso_id", "broadener_flags", "line_flags") else float
                optional[name] = np.asarray(table[name], dtype=dtype)
        data_provenance = _read_line_data_provenance(input_path, table)
        return cls(
            wavelength=np.asarray(table[wavelength_col], dtype=float),
            strength=np.asarray(table[strength_col], dtype=float),
            sigma=np.asarray(table[sigma_col], dtype=float),
            gamma=np.asarray(table[gamma_col], dtype=float),
            species=np.asarray(table[species_col], dtype=str),
            reference_temperature=float(table.meta.get("reference_temperature", 296.0)),
            line_source=str(table.meta.get("line_source", "generic")),
            data_provenance=data_provenance,
            **optional,
        )

    @classmethod
    def from_hitran_par(
        cls,
        path: str | Path,
        *,
        wavenumber_min: float | None = None,
        wavenumber_max: float | None = None,
        species: Iterable[str] | None = None,
        min_strength: float | None = None,
        max_lines: int | None = None,
        isotopologue_metadata=None,
        abundance_overrides: Mapping[tuple[int, int] | int, float] | None = None,
    ) -> "LineList":
        from .hitran import read_hitran_par

        return read_hitran_par(
            path,
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
            species=species,
            min_strength=min_strength,
            max_lines=max_lines,
            isotopologue_metadata=isotopologue_metadata,
            abundance_overrides=abundance_overrides,
        )

    @classmethod
    def from_aer_line_file(
        cls,
        path: str | Path,
        *,
        wavenumber_min: float | None = None,
        wavenumber_max: float | None = None,
        wavenumber_ranges: Iterable[tuple[float, float]] | None = None,
        species: Iterable[str] | None = None,
        min_strength: float | None = None,
        max_lines: int | None = None,
        isotopologue_metadata=None,
        abundance_overrides: Mapping[tuple[int, int] | int, float] | None = None,
        extra_broadener_dir: str | Path | None = None,
        assume_sorted: bool = False,
    ) -> "LineList":
        from .hitran import read_aer_line_file

        return read_aer_line_file(
            path,
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
            wavenumber_ranges=wavenumber_ranges,
            species=species,
            min_strength=min_strength,
            max_lines=max_lines,
            isotopologue_metadata=isotopologue_metadata,
            abundance_overrides=abundance_overrides,
            extra_broadener_dir=extra_broadener_dir,
            assume_sorted=assume_sorted,
        )

    @classmethod
    def empty_hitran(cls, *, reference_temperature: float = 296.0) -> "LineList":
        """Return an empty physical line list for continuum-only modelling."""

        empty_float = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return cls(
            wavelength=empty_float,
            strength=empty_float,
            sigma=empty_float,
            gamma=empty_float,
            species=np.array([], dtype=str),
            wavenumber=empty_float,
            mol_id=empty_int,
            iso_id=empty_int,
            air_width=empty_float,
            self_width=empty_float,
            lower_state_energy=empty_float,
            temperature_exponent=empty_float,
            pressure_shift=empty_float,
            molecular_mass_amu=empty_float,
            global_iso_id=empty_int,
            natural_abundance=empty_float,
            isotopologue_abundance_scale=empty_float,
            broadener_flags=np.zeros((0, len(LBLRTM_BROADENER_SPECIES)), dtype=int),
            broadener_widths=np.zeros((0, len(LBLRTM_BROADENER_SPECIES)), dtype=float),
            broadener_temperature_exponents=np.zeros((0, len(LBLRTM_BROADENER_SPECIES)), dtype=float),
            broadener_pressure_shifts=np.zeros((0, len(LBLRTM_BROADENER_SPECIES)), dtype=float),
            line_flags=empty_int,
            line_coupling_a=np.zeros((0, 4), dtype=float),
            line_coupling_b=np.zeros((0, 4), dtype=float),
            speed_dependence=empty_float,
            reference_temperature=reference_temperature,
            line_source="empty_hitran",
        )

    @classmethod
    def demo_near_ir(cls) -> "LineList":
        """Small synthetic line list for development tests.

        These are not real HITRAN values. They only let the package exercise the
        same fitting/correction mechanics before real molecular data ingestion
        is added.
        """

        wavelength = np.array(
            [2.3183, 2.3210, 2.3265, 2.3312, 2.3377, 2.3420, 2.3485, 2.3540],
            dtype=float,
        )
        strength = np.array([0.018, 0.026, 0.014, 0.03, 0.022, 0.012, 0.018, 0.015], dtype=float)
        sigma = np.full(wavelength.shape, 1.8e-4, dtype=float)
        gamma = np.full(wavelength.shape, 9.0e-5, dtype=float)
        species = np.array(["H2O", "H2O", "CO2", "H2O", "CH4", "CO2", "H2O", "CH4"], dtype=str)
        return cls(
            wavelength=wavelength,
            strength=strength,
            sigma=sigma,
            gamma=gamma,
            species=species,
            line_source="demo",
        )
