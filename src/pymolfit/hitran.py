from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import numpy as np

from .linelist import LBLRTM_BROADENER_SPECIES, LineList
from .partition import IsotopologueMetadata
from .physics import wavenumber_cm_to_wavelength_micron

HITRAN_MOLECULES = {
    1: ("H2O", 18.01528),
    2: ("CO2", 44.0095),
    3: ("O3", 47.9982),
    4: ("N2O", 44.0128),
    5: ("CO", 28.0101),
    6: ("CH4", 16.0425),
    7: ("O2", 31.9988),
    8: ("NO", 30.0061),
    9: ("SO2", 64.066),
    10: ("NO2", 46.0055),
    11: ("NH3", 17.0305),
    12: ("HNO3", 63.0128),
    13: ("OH", 17.0073),
    14: ("HF", 20.0063),
    15: ("HCl", 36.4609),
    16: ("HBr", 80.9119),
    17: ("HI", 127.9124),
    18: ("ClO", 51.4529),
    19: ("OCS", 60.0751),
    20: ("H2CO", 30.026),
    21: ("HOCl", 52.4603),
    22: ("N2", 28.0134),
    23: ("HCN", 27.0253),
    24: ("CH3Cl", 50.4875),
    25: ("H2O2", 34.0147),
    26: ("C2H2", 26.0373),
    27: ("C2H6", 30.069),
    28: ("PH3", 33.9976),
    29: ("COF2", 66.0069),
    30: ("SF6", 146.055),
    31: ("H2S", 34.0809),
    32: ("HCOOH", 46.0254),
    33: ("HO2", 33.0067),
    34: ("O", 15.999),
    35: ("ClONO2", 97.4579),
    36: ("NO+", 30.0061),
    37: ("HOBr", 96.9113),
    38: ("C2H4", 28.0532),
    39: ("CH3OH", 32.0419),
    40: ("CH3Br", 94.938),
    41: ("CH3CN", 41.052),
    42: ("CF4", 88.0043),
    43: ("C4H2", 50.0587),
    44: ("HC3N", 51.047),
    45: ("H2", 2.01588),
    46: ("CS", 44.076),
    47: ("SO3", 80.0632),
}

# ``AMWT`` block data from the LBLRTM 12.11 source distributed with
# Molecfit 4.4.4.  Values are indexed by HITRAN molecule and local
# isotopologue identifiers and are used for AER/LNFL records, whose runtime
# Doppler widths otherwise lose isotopologue specificity.
LBLRTM_ISOTOPOLOGUE_MASSES_AMU = {
    1: (18.01, 20.01, 19.01, 19.01, 21.02, 20.02),
    2: (43.99, 44.99, 45.99, 44.99, 47.00, 46.00, 48.00, 47.00, 46.00, 49.00),
    3: (47.98, 49.99, 49.99, 48.99, 48.99, 51.99, 51.99, 50.99, 50.99),
    4: (44.00, 45.00, 45.00, 46.00, 45.00),
    5: (27.99, 28.99, 29.99, 29.00, 31.00, 30.00),
    6: (16.03, 17.03, 17.03, 18.03),
    7: (31.99, 33.99, 32.99),
    8: (30.00, 31.00, 32.00),
    9: (63.96, 65.96),
    10: (45.99,),
    11: (17.03, 18.02),
    12: (62.99, 63.99),
    13: (17.00, 19.01, 18.01),
    14: (20.01, 21.01),
    15: (35.98, 37.97, 36.98, 38.97),
    16: (79.92, 81.92, 80.92, 82.92),
    17: (127.91, 128.91),
    18: (50.96, 52.96),
    19: (59.97, 61.96, 60.97, 60.97, 61.97),
    20: (30.01, 31.01, 32.01),
    21: (51.97, 53.97),
    22: (28.01, 29.01),
    23: (27.01, 28.01, 28.01),
    24: (49.99, 51.99),
    25: (34.01,),
    26: (26.01, 27.02, 27.01),
    27: (30.05, 31.05),
    28: (34.00,),
    29: (65.99, 66.99),
    30: (145.96,),
    31: (33.99, 35.98, 34.99),
    32: (46.01,),
    33: (33.00,),
    34: (15.99,),
    35: (96.96, 98.95),
    36: (30.00,),
    37: (95.92, 97.92),
    38: (28.05, 29.05),
    39: (32.04,),
    40: (93.94, 95.94),
    41: (41.05,),
    42: (88.0043,),
    43: (50.06,),
    44: (51.05,),
    45: (2.016, 3.022),
    46: (44.08, 46.08, 45.08, 45.08),
    47: (80.066,),
}


def lblrtm_isotopologue_mass_amu(mol_id: int, iso_id: int) -> float:
    """Return the LBLRTM source mass, with molecular-average fallback."""

    molecule_id = int(mol_id)
    isotope_id = 10 if int(iso_id) == 0 else int(iso_id)
    masses = LBLRTM_ISOTOPOLOGUE_MASSES_AMU.get(molecule_id, ())
    if 1 <= isotope_id <= len(masses):
        return float(masses[isotope_id - 1])
    return float(HITRAN_MOLECULES.get(molecule_id, ("", np.nan))[1])


def _parse_float(value: str) -> float:
    return float(value.replace("D", "E").strip())


def parse_hitran_local_iso_id(token: str) -> int:
    """Decode HITRAN's one-character local isotopologue identifier."""

    value = str(token).strip().upper()
    if len(value) != 1:
        raise ValueError(f"invalid HITRAN local isotopologue token: {token!r}")
    if value in "123456789":
        return int(value)
    if value == "0":
        return 10
    if "A" <= value <= "Z":
        return 11 + ord(value) - ord("A")
    raise ValueError(f"invalid HITRAN local isotopologue token: {token!r}")


def read_hitran_par(
    path: str | Path,
    *,
    wavenumber_min: float | None = None,
    wavenumber_max: float | None = None,
    species: Iterable[str] | None = None,
    min_strength: float | None = None,
    max_lines: int | None = None,
    isotopologue_metadata: IsotopologueMetadata | None = None,
    abundance_overrides: dict[tuple[int, int] | int, float] | None = None,
) -> LineList:
    """Read the common 160-character HITRAN ``.par`` line format.

    Only the fields needed by PyMolFit's first line-by-line model are parsed:
    molecule, isotopologue, wavenumber, intensity, Einstein A, air/self widths,
    lower-state energy, temperature exponent, and pressure shift.
    """

    wanted_species = None if species is None else set(species)
    rows: list[tuple] = []

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, row in enumerate(handle, start=1):
            if not row.strip() or not row[:2].strip().isdigit():
                continue
            if len(row) < 67:
                raise ValueError(f"HITRAN row {line_number} is too short")

            mol_id = int(row[0:2])
            iso_id = parse_hitran_local_iso_id(row[2:3])
            molecule, mass = HITRAN_MOLECULES.get(mol_id, (f"MOL{mol_id}", np.nan))
            if wanted_species is not None and molecule not in wanted_species:
                continue

            wavenumber = _parse_float(row[3:15])
            if wavenumber_min is not None and wavenumber < wavenumber_min:
                continue
            if wavenumber_max is not None and wavenumber > wavenumber_max:
                continue
            strength = _parse_float(row[15:25])
            if min_strength is not None and strength < min_strength:
                continue

            rows.append(
                (
                    wavenumber,
                    strength,
                    _parse_float(row[35:40]),
                    _parse_float(row[40:45]),
                    _parse_float(row[45:55]),
                    _parse_float(row[55:59]),
                    _parse_float(row[59:67]),
                    mass,
                    mol_id,
                    iso_id,
                    molecule,
                    _parse_float(row[25:35]),
                )
            )

    if not rows:
        raise ValueError("no HITRAN lines matched the requested filters")

    data = np.array(rows, dtype=object)
    if max_lines is not None:
        if max_lines < 1:
            raise ValueError("max_lines must be positive")
        strongest = np.argsort(np.asarray(data[:, 1], dtype=float))[::-1][:max_lines]
        data = data[strongest]
    wavenumber_order = np.argsort(np.asarray(data[:, 0], dtype=float))
    data = data[wavenumber_order]
    wavenumber = np.asarray(data[:, 0], dtype=float)
    strength = np.asarray(data[:, 1], dtype=float)
    wavelength = wavenumber_cm_to_wavelength_micron(wavenumber)
    air_width = np.asarray(data[:, 2], dtype=float)
    self_width = np.asarray(data[:, 3], dtype=float)
    lower_state_energy = np.asarray(data[:, 4], dtype=float)
    temperature_exponent = np.asarray(data[:, 5], dtype=float)
    pressure_shift = np.asarray(data[:, 6], dtype=float)
    molecular_mass_amu = np.asarray(data[:, 7], dtype=float)
    mol_id = np.asarray(data[:, 8], dtype=float)
    iso_id = np.asarray(data[:, 9], dtype=float)
    species_names = np.asarray(data[:, 10], dtype=str)

    wavelength_width = wavelength**2 * 1.0e-4 * np.maximum(air_width, 1.0e-6)
    line_list = LineList(
        wavelength=wavelength,
        strength=strength,
        sigma=np.maximum(wavelength_width * 0.5, 1.0e-8),
        gamma=np.maximum(wavelength_width, 1.0e-8),
        species=species_names,
        wavenumber=wavenumber,
        mol_id=mol_id,
        iso_id=iso_id,
        air_width=air_width,
        self_width=self_width,
        lower_state_energy=lower_state_energy,
        temperature_exponent=temperature_exponent,
        pressure_shift=pressure_shift,
        molecular_mass_amu=molecular_mass_amu,
        line_source="hitran_par",
    )
    if isotopologue_metadata is not None:
        line_list = line_list.with_isotopologue_metadata(
            isotopologue_metadata,
            abundance_overrides=abundance_overrides,
        )
    return line_list


def read_aer_line_file(
    path: str | Path,
    *,
    wavenumber_min: float | None = None,
    wavenumber_max: float | None = None,
    wavenumber_ranges: Iterable[tuple[float, float]] | None = None,
    species: Iterable[str] | None = None,
    min_strength: float | None = None,
    max_lines: int | None = None,
    isotopologue_metadata: IsotopologueMetadata | None = None,
    abundance_overrides: dict[tuple[int, int] | int, float] | None = None,
    extra_broadener_dir: str | Path | None = None,
    assume_sorted: bool = False,
) -> LineList:
    """Read AER/LBLRTM line files used by Molecfit.

    AER line files are close to HITRAN fixed-width records, but the first
    field is a three-character molecule/isotopologue token such as ``" 71"``
    for O2 isotopologue 1 or ``"201"`` for H2CO isotopologue 1. The remaining
    numeric fields used here follow the HITRAN/LBLRTM positions.
    """

    wanted_species = None if species is None else set(species)
    selected_ranges = None
    if wavenumber_ranges is not None:
        selected_ranges = tuple(
            (min(float(lower), float(upper)), max(float(lower), float(upper)))
            for lower, upper in wavenumber_ranges
        )
        if not selected_ranges:
            raise ValueError("wavenumber_ranges must contain at least one interval")
        if not all(np.isfinite(lower) and np.isfinite(upper) and upper > lower for lower, upper in selected_ranges):
            raise ValueError("wavenumber_ranges must contain finite, non-zero intervals")
    rows: list[tuple] = []
    input_path = Path(path)
    if extra_broadener_dir is None:
        extra_broadener_dir = _default_extra_broadener_dir(input_path)
    broadener_tables = _load_extra_broadener_tables(extra_broadener_dir) if extra_broadener_dir is not None else {}

    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        iterator = enumerate(handle, start=1)
        for line_number, row in iterator:
            stripped = row.strip()
            if not stripped or stripped.startswith((">", "%")):
                continue
            if len(row) < 67:
                continue
            token = row[:3].strip()
            if not token.isdigit():
                continue
            mol_iso = int(token)
            mol_id = mol_iso // 10
            iso_id = mol_iso % 10
            if mol_id not in HITRAN_MOLECULES:
                continue

            raw_line_flag = _parse_f100_line_flag(row)
            line_flag = abs(raw_line_flag) if raw_line_flag < 0 else 0
            line_coupling_a = np.zeros(4, dtype=float)
            line_coupling_b = np.zeros(4, dtype=float)
            if raw_line_flag < 0:
                try:
                    _, aux_row = next(iterator)
                except StopIteration:
                    aux_row = ""
                parsed = _parse_line_coupling_auxiliary_row(aux_row)
                if parsed is not None:
                    line_coupling_a, line_coupling_b, _ = parsed

            molecule, _ = HITRAN_MOLECULES[mol_id]
            mass = lblrtm_isotopologue_mass_amu(mol_id, iso_id)
            if wanted_species is not None and molecule not in wanted_species:
                continue

            try:
                wavenumber = _parse_float(row[3:15])
                strength = _parse_float(row[15:25])
                einstein_a = _parse_float(row[25:35])
                air_width = _parse_float(row[35:40])
                self_width = _parse_float(row[40:45])
                lower_state_energy = _parse_float(row[45:55])
                temperature_exponent = _parse_float(row[55:59])
                pressure_shift = _parse_float(row[59:67])
                broadener_flags = _parse_broadener_flags(row)
            except ValueError as exc:
                raise ValueError(f"AER line row {line_number} could not be parsed") from exc

            broadener_widths = np.zeros(len(LBLRTM_BROADENER_SPECIES), dtype=float)
            broadener_temperature_exponents = np.zeros(len(LBLRTM_BROADENER_SPECIES), dtype=float)
            broadener_pressure_shifts = np.zeros(len(LBLRTM_BROADENER_SPECIES), dtype=float)
            if np.any(broadener_flags > 0):
                try:
                    _, next_row = next(iterator)
                except StopIteration:
                    next_row = ""
                next_token = next_row[:2].strip()
                if next_token.isdigit() and int(next_token) == mol_id:
                    broadener_data = _parse_broadener_data(next_row)
                    broadener_widths = broadener_data[:, 0]
                    broadener_temperature_exponents = broadener_data[:, 1]
                    broadener_pressure_shifts = broadener_data[:, 2]

            if assume_sorted and wavenumber_max is not None and wavenumber > wavenumber_max:
                break
            if wavenumber_min is not None and wavenumber < wavenumber_min:
                continue
            if wavenumber_max is not None and wavenumber > wavenumber_max:
                continue
            if selected_ranges is not None and not any(
                lower <= wavenumber <= upper for lower, upper in selected_ranges
            ):
                continue
            if min_strength is not None and strength < min_strength:
                continue

            _apply_extra_broadener_parameters(
                broadener_flags,
                broadener_widths,
                broadener_temperature_exponents,
                broadener_pressure_shifts,
                broadener_tables,
                mol_id=mol_id,
                iso_id=iso_id,
                wavenumber=wavenumber,
            )

            rows.append(
                (
                    wavenumber,
                    strength,
                    air_width,
                    self_width,
                    lower_state_energy,
                    temperature_exponent,
                    pressure_shift,
                    mass,
                    mol_id,
                    iso_id,
                    molecule,
                    einstein_a,
                    broadener_flags,
                    broadener_widths,
                    broadener_temperature_exponents,
                    broadener_pressure_shifts,
                    line_flag,
                    line_coupling_a,
                    line_coupling_b,
                    0.0,
                )
            )

    line_list = _line_list_from_rows(rows, max_lines=max_lines)
    if isotopologue_metadata is not None:
        line_list = line_list.with_isotopologue_metadata(
            isotopologue_metadata,
            abundance_overrides=abundance_overrides,
        )
    return line_list


def _line_list_from_rows(
    rows: list[tuple],
    *,
    max_lines: int | None = None,
) -> LineList:
    if not rows:
        raise ValueError("no HITRAN/AER lines matched the requested filters")

    data = np.array(rows, dtype=object)
    if max_lines is not None:
        if max_lines < 1:
            raise ValueError("max_lines must be positive")
        strongest = np.argsort(np.asarray(data[:, 1], dtype=float))[::-1][:max_lines]
        data = data[strongest]
    wavenumber_order = np.argsort(np.asarray(data[:, 0], dtype=float))
    data = data[wavenumber_order]
    wavenumber = np.asarray(data[:, 0], dtype=float)
    strength = np.asarray(data[:, 1], dtype=float)
    wavelength = wavenumber_cm_to_wavelength_micron(wavenumber)
    air_width = np.asarray(data[:, 2], dtype=float)
    self_width = np.asarray(data[:, 3], dtype=float)
    lower_state_energy = np.asarray(data[:, 4], dtype=float)
    temperature_exponent = np.asarray(data[:, 5], dtype=float)
    pressure_shift = np.asarray(data[:, 6], dtype=float)
    molecular_mass_amu = np.asarray(data[:, 7], dtype=float)
    mol_id = np.asarray(data[:, 8], dtype=float)
    iso_id = np.asarray(data[:, 9], dtype=float)
    species_names = np.asarray(data[:, 10], dtype=str)
    broadener_flags = None
    broadener_widths = None
    broadener_temperature_exponents = None
    broadener_pressure_shifts = None
    line_flags = None
    line_coupling_a = None
    line_coupling_b = None
    speed_dependence = None
    if data.shape[1] >= 16:
        broadener_flags = np.vstack(data[:, 12]).astype(int)
        broadener_widths = np.vstack(data[:, 13]).astype(float)
        broadener_temperature_exponents = np.vstack(data[:, 14]).astype(float)
        broadener_pressure_shifts = np.vstack(data[:, 15]).astype(float)
    if data.shape[1] >= 20:
        line_flags = np.asarray(data[:, 16], dtype=int)
        line_coupling_a = np.vstack(data[:, 17]).astype(float)
        line_coupling_b = np.vstack(data[:, 18]).astype(float)
        speed_dependence = np.asarray(data[:, 19], dtype=float)

    wavelength_width = wavelength**2 * 1.0e-4 * np.maximum(air_width, 1.0e-6)
    return LineList(
        wavelength=wavelength,
        strength=strength,
        sigma=np.maximum(wavelength_width * 0.5, 1.0e-8),
        gamma=np.maximum(wavelength_width, 1.0e-8),
        species=species_names,
        wavenumber=wavenumber,
        mol_id=mol_id,
        iso_id=iso_id,
        air_width=air_width,
        self_width=self_width,
        lower_state_energy=lower_state_energy,
        temperature_exponent=temperature_exponent,
        pressure_shift=pressure_shift,
        molecular_mass_amu=molecular_mass_amu,
        broadener_flags=broadener_flags,
        broadener_widths=broadener_widths,
        broadener_temperature_exponents=broadener_temperature_exponents,
        broadener_pressure_shifts=broadener_pressure_shifts,
        line_flags=line_flags,
        line_coupling_a=line_coupling_a,
        line_coupling_b=line_coupling_b,
        speed_dependence=speed_dependence,
        line_source="aer_lnfl_tape8",
    )


def _parse_broadener_flags(row: str) -> np.ndarray:
    flags = np.zeros(len(LBLRTM_BROADENER_SPECIES), dtype=int)
    if len(row) < 114:
        return flags
    try:
        return np.array([int(row[100 + 2 * i : 102 + 2 * i]) for i in range(7)], dtype=int)
    except ValueError:
        return flags


def _parse_f100_line_flag(row: str) -> int:
    if len(row) < 100:
        return 0
    field = row[98:100].strip()
    if not field:
        return 0
    try:
        return int(field)
    except ValueError:
        return 0


def _parse_line_coupling_auxiliary_row(row: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    numbers = re.findall(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[DEde][+-]?\d+)?", row[2:])
    if len(numbers) < 9:
        return None
    try:
        values = np.asarray([_parse_float(value) for value in numbers[:8]], dtype=float)
        flag = int(float(numbers[8]))
    except ValueError:
        return None
    return values[0::2], values[1::2], flag


def _parse_broadener_data(row: str) -> np.ndarray:
    values: list[float] = []
    for index in range(21):
        start = 2 + 8 * index
        stop = start + 8
        field = row[start:stop]
        if not field.strip():
            values.append(0.0)
            continue
        values.append(_parse_float(field))
    return np.asarray(values, dtype=float).reshape(len(LBLRTM_BROADENER_SPECIES), 3)


def _load_extra_broadener_tables(path: str | Path | None) -> dict[tuple[int, float, int], tuple[float, float, float]]:
    if path is None:
        return {}
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"extra broadener directory does not exist: {directory}")

    tables: dict[tuple[int, float, int], tuple[float, float, float]] = {}
    specs = (
        ("wv_co2_brd_param", 101, "CO2", "wv_co2"),
        ("co2_co2_brd_param", 102, "CO2", "standard"),
        ("co2_h2o_brd_param", 102, "H2O", "standard"),
        ("o2_o2_brd_param", 107, "O2", "standard"),
        ("o2_h2o_brd_param", 107, "H2O", "standard"),
    )
    for filename, mol_id, broadener_name, table_format in specs:
        file_path = directory / filename
        if not file_path.exists():
            continue
        broadener_index = LBLRTM_BROADENER_SPECIES.index(broadener_name)
        for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith((">", "%", "#")):
                continue
            parts = stripped.replace("D", "E").split()
            try:
                if table_format == "wv_co2":
                    if len(parts) < 6:
                        continue
                    wavenumber = float(parts[0])
                    width = float(parts[2])
                    temperature_exponent = float(parts[4])
                    pressure_shift = float(parts[5])
                else:
                    if len(parts) < 5:
                        continue
                    wavenumber = float(parts[1])
                    width = float(parts[2])
                    temperature_exponent = float(parts[3])
                    pressure_shift = float(parts[4])
            except ValueError:
                continue
            tables.setdefault(
                (mol_id, round(wavenumber, 6), broadener_index),
                (width, temperature_exponent, pressure_shift),
            )
    return tables


def _apply_extra_broadener_parameters(
    flags: np.ndarray,
    widths: np.ndarray,
    temperature_exponents: np.ndarray,
    pressure_shifts: np.ndarray,
    tables: dict[tuple[int, float, int], tuple[float, float, float]],
    *,
    mol_id: int,
    iso_id: int,
    wavenumber: float,
) -> None:
    if not tables:
        return
    # LNFL's MOL3 convention is molecule + 100 * isotopologue; the installed
    # extra broadener files used by Molecfit target the main isotopologues
    # (H2O=101, CO2=102, O2=107).
    mol3 = 100 * int(iso_id) + int(mol_id)
    wavenumber_key = round(float(wavenumber), 6)
    for broadener_index in range(len(LBLRTM_BROADENER_SPECIES)):
        parameters = tables.get((mol3, wavenumber_key, broadener_index))
        if parameters is None:
            continue
        width, exponent, shift = parameters
        flags[broadener_index] = 1
        widths[broadener_index] = width
        temperature_exponents[broadener_index] = exponent
        pressure_shifts[broadener_index] = shift


def _default_extra_broadener_dir(line_path: Path) -> Path | None:
    directory = line_path.parent
    known_files = (
        "wv_co2_brd_param",
        "co2_co2_brd_param",
        "co2_h2o_brd_param",
        "o2_o2_brd_param",
        "o2_h2o_brd_param",
    )
    if any((directory / filename).exists() for filename in known_files):
        return directory
    return None
