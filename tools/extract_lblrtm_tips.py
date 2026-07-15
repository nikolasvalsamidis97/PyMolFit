"""Extract LBLRTM 12.11 TIPS tables into GenMolFit package data.

This is a development-time provenance tool. The generated NPZ is consumed at
runtime, so installing GenMolFit does not require LBLRTM or a Fortran compiler.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


MOLECULE_SUBROUTINES = {
    1: "H2O", 2: "CO2", 3: "O3", 4: "N2O", 5: "CO", 6: "CH4",
    7: "O2", 8: "NO", 9: "SO2", 10: "NO2", 11: "NH3", 12: "HNO3",
    13: "OH", 14: "HF", 15: "HCL", 16: "HBR", 17: "HI", 18: "CLO",
    19: "OCS", 20: "H2CO", 21: "HOCL", 22: "N2", 23: "HCN",
    24: "CH3CL", 25: "H2O2", 26: "C2H2", 27: "C2H6", 28: "PH3",
    29: "COF2", 30: "SF6", 31: "H2S", 32: "HCOOH", 33: "HO2",
    35: "CLONO2", 36: "NOP", 37: "HOBR", 38: "C2H4", 40: "CH3BR",
    41: "CH3CN", 42: "CF4", 43: "C4H2", 44: "HC3N", 45: "H2",
    46: "CS",
}


def _numbers(text: str) -> np.ndarray:
    tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?", text)
    return np.asarray([float(token.replace("D", "E").replace("d", "e")) for token in tokens])


def extract(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="ignore")
    tdat_match = re.search(r"data\s+Tdat\s*/(.*?)/", text, flags=re.IGNORECASE | re.DOTALL)
    if tdat_match is None:
        raise ValueError("could not find LBLRTM Tdat table")
    temperatures = _numbers(tdat_match.group(1))

    mol_rows: list[np.ndarray] = []
    iso_rows: list[np.ndarray] = []
    temperature_rows: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []
    for mol_id, name in MOLECULE_SUBROUTINES.items():
        block_match = re.search(
            rf"Subroutine\s+QT_{re.escape(name)}\b(.*?)end\s+subroutine\s+QT_{re.escape(name)}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block_match is None:
            continue
        block = block_match.group(1)
        for match in re.finditer(
            r"data\s*\(\s*QofT\(\s*(\d+)\s*,\s*J\s*\)\s*,\s*J\s*=\s*1\s*,\s*\d+\s*\)\s*/(.*?)/",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            iso_id = int(match.group(1))
            q_values = _numbers(match.group(2))
            if q_values.size != temperatures.size:
                raise ValueError(
                    f"QT_{name} isotope {iso_id} has {q_values.size} values; "
                    f"expected {temperatures.size}"
                )
            mol_rows.append(np.full(temperatures.size, mol_id, dtype=np.int16))
            iso_rows.append(np.full(temperatures.size, iso_id, dtype=np.int8))
            temperature_rows.append(temperatures.astype(np.float32))
            q_rows.append(q_values.astype(np.float64))

    if not q_rows:
        raise ValueError("no LBLRTM TIPS tables were extracted")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        mol_id=np.concatenate(mol_rows),
        iso_id=np.concatenate(iso_rows),
        temperature_k=np.concatenate(temperature_rows),
        q=np.concatenate(q_rows),
        source=np.asarray("LBLRTM 12.11 oprop_voigt.f90 TIPS_2011 tables"),
    )
    print(f"wrote {output} ({len(q_rows)} isotopologues, {sum(row.size for row in q_rows)} values)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)


if __name__ == "__main__":
    main()
