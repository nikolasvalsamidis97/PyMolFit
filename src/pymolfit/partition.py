from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from html import unescape
from pathlib import Path
from typing import Mapping
import re

import numpy as np
from astropy.table import Table


@dataclass(frozen=True)
class IsotopologueMetadata:
    """HITRAN molecule/isotopologue metadata.

    HITRAN line intensities are normally weighted by the terrestrial natural
    isotopologue abundance. The abundance stored here therefore acts as
    metadata and as the denominator when a caller explicitly asks for a
    non-terrestrial isotopologue mixture.
    """

    global_iso_id: np.ndarray
    mol_id: np.ndarray
    iso_id: np.ndarray
    abundance: np.ndarray
    molar_mass: np.ndarray
    q296: np.ndarray
    q_file: np.ndarray | None = None
    formula: np.ndarray | None = None

    def __post_init__(self) -> None:
        global_iso_id = np.asarray(self.global_iso_id, dtype=int)
        mol_id = np.asarray(self.mol_id, dtype=int)
        iso_id = np.asarray(self.iso_id, dtype=int)
        abundance = np.asarray(self.abundance, dtype=float)
        molar_mass = np.asarray(self.molar_mass, dtype=float)
        q296 = np.asarray(self.q296, dtype=float)
        optional = {
            "q_file": None if self.q_file is None else np.asarray(self.q_file, dtype=str),
            "formula": None if self.formula is None else np.asarray(self.formula, dtype=str),
        }

        shapes = {arr.shape for arr in (global_iso_id, mol_id, iso_id, abundance, molar_mass, q296)}
        for value in optional.values():
            if value is not None:
                shapes.add(value.shape)
        if len(shapes) != 1:
            raise ValueError("isotopologue metadata arrays must have the same shape")
        if global_iso_id.ndim != 1:
            raise ValueError("isotopologue metadata arrays must be one-dimensional")
        if np.any(global_iso_id <= 0) or np.any(mol_id <= 0):
            raise ValueError("global_iso_id and mol_id must be positive")
        if np.any(iso_id < 0):
            raise ValueError("iso_id must be non-negative")
        if np.any(abundance <= 0) or np.any(molar_mass <= 0) or np.any(q296 <= 0):
            raise ValueError("abundance, molar_mass, and q296 must be positive")

        object.__setattr__(self, "global_iso_id", global_iso_id)
        object.__setattr__(self, "mol_id", mol_id)
        object.__setattr__(self, "iso_id", iso_id)
        object.__setattr__(self, "abundance", abundance)
        object.__setattr__(self, "molar_mass", molar_mass)
        object.__setattr__(self, "q296", q296)
        object.__setattr__(self, "q_file", optional["q_file"])
        object.__setattr__(self, "formula", optional["formula"])

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        global_iso_id_col: str = "global_iso_id",
        mol_id_col: str = "mol_id",
        iso_id_col: str = "iso_id",
        abundance_col: str = "abundance",
        molar_mass_col: str = "molar_mass",
        q296_col: str = "q296",
        q_file_col: str = "q_file",
        formula_col: str = "formula",
    ) -> "IsotopologueMetadata":
        table = Table.read(path)
        return cls(
            global_iso_id=np.asarray(table[global_iso_id_col], dtype=int),
            mol_id=np.asarray(table[mol_id_col], dtype=int),
            iso_id=np.asarray(table[iso_id_col], dtype=int),
            abundance=np.asarray(table[abundance_col], dtype=float),
            molar_mass=np.asarray(table[molar_mass_col], dtype=float),
            q296=np.asarray(table[q296_col], dtype=float),
            q_file=np.asarray(table[q_file_col], dtype=str) if q_file_col in table.colnames else None,
            formula=np.asarray(table[formula_col], dtype=str) if formula_col in table.colnames else None,
        )

    @classmethod
    def from_hitran_iso_meta_html(cls, path: str | Path) -> "IsotopologueMetadata":
        """Read HITRAN's isotopologue metadata HTML page.

        This supports a saved copy of ``https://hitran.org/docs/iso-meta/``.
        It extracts the global isotopologue ID, local ID, natural abundance,
        molar mass, Q(296 K), and the linked TIPS q-file name.
        """

        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        global_ids: list[int] = []
        mol_ids: list[int] = []
        iso_ids: list[int] = []
        abundances: list[float] = []
        molar_masses: list[float] = []
        q296_values: list[float] = []
        q_files: list[str] = []
        formulas: list[str] = []

        section_pattern = re.compile(
            r"<h4>\s*(?P<mol>\d+)\s*:.*?</h4>\s*<table[^>]*>.*?<tbody>(?P<body>.*?)</tbody>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        row_pattern = re.compile(r"<tr>(?P<row>.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
        cell_pattern = re.compile(r"<td[^>]*>(?P<cell>.*?)</td>", flags=re.IGNORECASE | re.DOTALL)

        for section in section_pattern.finditer(text):
            mol_id = int(section.group("mol"))
            for row_match in row_pattern.finditer(section.group("body")):
                cells = [match.group("cell") for match in cell_pattern.finditer(row_match.group("row"))]
                if len(cells) < 8:
                    continue
                global_ids.append(int(_html_cell_text(cells[0])))
                mol_ids.append(mol_id)
                iso_ids.append(int(_html_cell_text(cells[1])))
                formulas.append(_html_cell_text(cells[2]))
                abundances.append(_parse_hitran_number(cells[4]))
                molar_masses.append(_parse_hitran_number(cells[5]))
                q296_values.append(_parse_hitran_number(cells[6]))
                q_files.append(_html_cell_text(cells[7]))

        if not global_ids:
            raise ValueError("no isotopologue metadata rows found in HITRAN HTML")
        return cls(
            global_iso_id=np.asarray(global_ids, dtype=int),
            mol_id=np.asarray(mol_ids, dtype=int),
            iso_id=np.asarray(iso_ids, dtype=int),
            abundance=np.asarray(abundances, dtype=float),
            molar_mass=np.asarray(molar_masses, dtype=float),
            q296=np.asarray(q296_values, dtype=float),
            q_file=np.asarray(q_files, dtype=str),
            formula=np.asarray(formulas, dtype=str),
        )

    def to_table(self) -> Table:
        table = Table()
        table["global_iso_id"] = self.global_iso_id
        table["mol_id"] = self.mol_id
        table["iso_id"] = self.iso_id
        table["abundance"] = self.abundance
        table["molar_mass"] = self.molar_mass
        table["q296"] = self.q296
        if self.q_file is not None:
            table["q_file"] = self.q_file
        if self.formula is not None:
            table["formula"] = self.formula
        return table

    def write(self, path: str | Path, *, format: str = "ascii.ecsv") -> None:
        self.to_table().write(path, format=format, overwrite=True)

    def values_for_pair(
        self,
        mol_id: np.ndarray,
        iso_id: np.ndarray,
        values: np.ndarray,
        *,
        missing: float = np.nan,
    ) -> np.ndarray:
        mol_id = np.asarray(mol_id, dtype=int)
        iso_id = np.asarray(iso_id, dtype=int)
        if mol_id.shape != iso_id.shape:
            raise ValueError("mol_id and iso_id must have the same shape")
        values = np.asarray(values)
        if values.shape != self.mol_id.shape:
            raise ValueError("values must come from this metadata table")

        result = np.full(mol_id.shape, missing, dtype=float)
        lookup = {
            (int(mol), int(iso)): value
            for mol, iso, value in zip(self.mol_id, self.iso_id, values, strict=True)
        }
        for pair in set(zip(mol_id.tolist(), iso_id.tolist(), strict=True)):
            if pair in lookup:
                result[(mol_id == pair[0]) & (iso_id == pair[1])] = lookup[pair]
        return result

    def global_ids_for_pair(self, mol_id: np.ndarray, iso_id: np.ndarray) -> np.ndarray:
        return self.values_for_pair(mol_id, iso_id, self.global_iso_id)

    def abundance_for_pair(self, mol_id: np.ndarray, iso_id: np.ndarray) -> np.ndarray:
        return self.values_for_pair(mol_id, iso_id, self.abundance)

    def molar_mass_for_pair(self, mol_id: np.ndarray, iso_id: np.ndarray) -> np.ndarray:
        return self.values_for_pair(mol_id, iso_id, self.molar_mass)

    def q_file_map(self) -> Mapping[int, str]:
        if self.q_file is None:
            return {}
        return {int(global_id): str(path) for global_id, path in zip(self.global_iso_id, self.q_file, strict=True)}


@dataclass(frozen=True)
class PartitionTable:
    """Tabulated total internal partition sums, keyed by molecule/isotopologue."""

    mol_id: np.ndarray
    iso_id: np.ndarray
    temperature_k: np.ndarray
    q: np.ndarray
    interpolation: str = "log_linear"
    _lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _value_cache: dict[tuple[int, int, float], float] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        mol_id = np.asarray(self.mol_id, dtype=int)
        iso_id = np.asarray(self.iso_id, dtype=int)
        temperature_k = np.asarray(self.temperature_k, dtype=float)
        q = np.asarray(self.q, dtype=float)
        shapes = {arr.shape for arr in (mol_id, iso_id, temperature_k, q)}
        if len(shapes) != 1:
            raise ValueError("partition arrays must have the same shape")
        if mol_id.ndim != 1:
            raise ValueError("partition arrays must be one-dimensional")
        if np.any(temperature_k <= 0) or np.any(q <= 0):
            raise ValueError("temperature_k and q must be positive")
        if self.interpolation not in {"log_linear", "lblrtm_lagrange"}:
            raise ValueError("interpolation must be 'log_linear' or 'lblrtm_lagrange'")

        object.__setattr__(self, "mol_id", mol_id)
        object.__setattr__(self, "iso_id", iso_id)
        object.__setattr__(self, "temperature_k", temperature_k)
        object.__setattr__(self, "q", q)
        lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for pair in set(zip(mol_id.tolist(), iso_id.tolist(), strict=True)):
            keep = (mol_id == pair[0]) & (iso_id == pair[1])
            order = np.argsort(temperature_k[keep])
            lookup[(int(pair[0]), int(pair[1]))] = (
                temperature_k[keep][order],
                q[keep][order],
            )
        object.__setattr__(self, "_lookup", lookup)
        object.__setattr__(self, "_value_cache", {})

    @classmethod
    def from_lblrtm_package_data(cls) -> "PartitionTable":
        """Load the TIPS tables embedded in LBLRTM 12.11.

        The packaged data are extracted from LBLRTM's ``TIPS_2011`` source at
        development time. This keeps physical runs self-contained while using
        the same molecule/isotopologue partition sums as Molecfit's LBLRTM.
        """

        resource = resources.files("pymolfit").joinpath("data", "lblrtm_v12_11_tips.npz")
        with resources.as_file(resource) as path, np.load(path) as data:
            return cls(
                mol_id=np.asarray(data["mol_id"], dtype=int),
                iso_id=np.asarray(data["iso_id"], dtype=int),
                temperature_k=np.asarray(data["temperature_k"], dtype=float),
                q=np.asarray(data["q"], dtype=float),
                interpolation="lblrtm_lagrange",
            )

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        mol_id_col: str = "mol_id",
        iso_id_col: str = "iso_id",
        temperature_col: str = "temperature_k",
        q_col: str = "q",
    ) -> "PartitionTable":
        table = Table.read(path)
        return cls(
            mol_id=np.asarray(table[mol_id_col], dtype=int),
            iso_id=np.asarray(table[iso_id_col], dtype=int),
            temperature_k=np.asarray(table[temperature_col], dtype=float),
            q=np.asarray(table[q_col], dtype=float),
        )

    def to_table(self) -> Table:
        table = Table()
        table["mol_id"] = self.mol_id
        table["iso_id"] = self.iso_id
        table["temperature_k"] = self.temperature_k
        table["q"] = self.q
        return table

    def write(self, path: str | Path, *, format: str = "ascii.ecsv") -> None:
        self.to_table().write(path, format=format, overwrite=True)

    @classmethod
    def from_hitran_q_file(
        cls,
        path: str | Path,
        *,
        mol_id: int,
        iso_id: int,
    ) -> "PartitionTable":
        """Read one HITRAN TIPS ``q*.txt`` file.

        HITRAN q files are simple two-column text files: temperature in K and
        total internal partition sum Q(T).
        """

        data = np.loadtxt(path, dtype=float)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 2:
            raise ValueError("HITRAN q file must contain at least two columns")
        temperature = np.asarray(data[:, 0], dtype=float)
        q = np.asarray(data[:, 1], dtype=float)
        return cls(
            mol_id=np.full(temperature.shape, int(mol_id), dtype=int),
            iso_id=np.full(temperature.shape, int(iso_id), dtype=int),
            temperature_k=temperature,
            q=q,
        )

    @classmethod
    def from_hitran_q_directory(
        cls,
        q_dir: str | Path,
        metadata: IsotopologueMetadata,
        *,
        global_iso_ids: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> "PartitionTable":
        """Read HITRAN TIPS q files for the isotopologues in ``metadata``."""

        q_dir = Path(q_dir)
        selected_global_ids = (
            set(int(value) for value in global_iso_ids)
            if global_iso_ids is not None
            else set(int(value) for value in metadata.global_iso_id)
        )
        q_file_map = metadata.q_file_map()
        tables = []
        for index, global_id in enumerate(metadata.global_iso_id):
            global_id = int(global_id)
            if global_id not in selected_global_ids:
                continue
            filename = q_file_map.get(global_id, f"q{global_id}.txt")
            path = q_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"missing HITRAN q file for global isotopologue {global_id}: {path}")
            tables.append(
                cls.from_hitran_q_file(
                    path,
                    mol_id=int(metadata.mol_id[index]),
                    iso_id=int(metadata.iso_id[index]),
                )
            )
        if not tables:
            raise ValueError("no HITRAN q files matched the requested isotopologues")
        return cls.concatenate(tables)

    @classmethod
    def concatenate(cls, tables: list["PartitionTable"] | tuple["PartitionTable", ...]) -> "PartitionTable":
        if not tables:
            raise ValueError("tables must not be empty")
        return cls(
            mol_id=np.concatenate([table.mol_id for table in tables]),
            iso_id=np.concatenate([table.iso_id for table in tables]),
            temperature_k=np.concatenate([table.temperature_k for table in tables]),
            q=np.concatenate([table.q for table in tables]),
        )

    def value(self, mol_id: np.ndarray, iso_id: np.ndarray, temperature_k: float) -> np.ndarray:
        """Interpolate Q(T) for each requested molecule/isotopologue.

        Missing pairs return NaN so callers can fall back to an approximate
        partition model without silently pretending table coverage exists.
        """

        mol_id = np.asarray(mol_id, dtype=int)
        iso_id = np.asarray(iso_id, dtype=int)
        if mol_id.shape != iso_id.shape:
            raise ValueError("mol_id and iso_id must have the same shape")

        values = np.full(mol_id.shape, np.nan, dtype=float)
        pair_codes = (mol_id.astype(np.int64, copy=False) << 32) | iso_id.astype(np.int64, copy=False)
        for code in np.unique(pair_codes):
            mol = int(code >> 32)
            iso = int(code & 0xFFFFFFFF)
            keep_request = pair_codes == code
            pair = (mol, iso)
            if pair not in self._lookup:
                continue
            cache_key = (mol, iso, float(temperature_k))
            cached_value = self._value_cache.get(cache_key)
            if cached_value is None:
                temps, qs = self._lookup[pair]
                if self.interpolation == "lblrtm_lagrange":
                    cached_value = _lblrtm_lagrange_value(temps, qs, float(temperature_k))
                else:
                    cached_value = float(
                        np.exp(
                            np.interp(
                                np.log(float(temperature_k)),
                                np.log(temps),
                                np.log(qs),
                                left=np.log(qs[0]),
                                right=np.log(qs[-1]),
                            )
                        )
                    )
                self._value_cache[cache_key] = cached_value
            values[keep_request] = cached_value
        return values


def _lblrtm_lagrange_value(x: np.ndarray, y: np.ndarray, value: float) -> float:
    """Port LBLRTM 12.11 ``AtoB`` three/four-point interpolation."""

    if value < x[0] or value > x[-1]:
        return np.nan
    upper = int(np.searchsorted(x, value, side="left"))
    upper = max(1, upper)
    if upper < 2 or upper == x.size - 1:
        indices = np.arange(0, 3) if upper < 2 else np.arange(x.size - 3, x.size)
    else:
        indices = np.arange(upper - 2, upper + 2)
    nodes = x[indices]
    values = y[indices]
    delta = value - nodes
    weights = np.ones(nodes.size, dtype=float)
    for index in range(nodes.size):
        others = np.arange(nodes.size) != index
        weights[index] = np.prod(delta[others] / (nodes[index] - nodes[others]))
    return float(np.dot(weights, values))


def _html_cell_text(cell: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", cell))).strip().replace("\xa0", " ")


def _parse_hitran_number(cell: str) -> float:
    text = _html_cell_text(cell)
    text = text.replace("\u2212", "-")
    text = text.replace("×", "x")
    match = re.fullmatch(r"([+-]?\d+(?:\.\d*)?)\s*x\s*10\s*([+-]?\d+)", text)
    if match:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    return float(text)
