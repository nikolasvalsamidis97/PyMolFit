"""Extract the LBLRTM 12.11 N2 continuum coefficient tables.

This development-time provenance tool reads the ``bn2f`` and ``bn2f1`` block data from
LBLRTM's ``contnm.f90`` and writes the compact table consumed by GenMolFit.
LBLRTM is not imported or executed at runtime.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _numbers(text: str) -> np.ndarray:
    tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?", text)
    return np.asarray([float(token.replace("D", "E").replace("d", "e")) for token in tokens])


def _data_array(block: str, name: str) -> np.ndarray:
    name_pattern = r"\s*&?\s*,\s*".join(re.escape(part) for part in name.split(","))
    match = re.search(
        rf"DATA\s+{name_pattern}\s*(?:&\s*)*/(.*?)/",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"could not find {name} in LBLRTM bn2f block")
    return _numbers(match.group(1))


def extract(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="ignore")
    block_match = re.search(
        r"BLOCK\s+DATA\s+bn2f\b(.*?)end\s+block\s+data\s+bn2f",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block_match is None:
        raise ValueError("could not find LBLRTM bn2f block")
    block = block_match.group(1)

    overtone_match = re.search(
        r"BLOCK\s+DATA\s+bn2f1\b(.*?)end\s+block\s+data\s+bn2f1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if overtone_match is None:
        raise ValueError("could not find LBLRTM bn2f1 block")
    overtone_block = overtone_match.group(1)

    grid = _data_array(block, "V1n2f,V2n2f,DVn2f,NPTn2f")
    if grid.size != 4:
        raise ValueError(f"expected four bn2f grid values, found {grid.size}")
    v1, v2, dv, n_points_float = grid
    n_points = int(n_points_float)

    n2_272 = _data_array(block, "xn2_272")
    n2_228 = _data_array(block, "xn2_228")
    h2o_efficiency = _data_array(block, "a_h2o")
    if not (n2_272.size == n2_228.size == h2o_efficiency.size == n_points):
        raise ValueError(
            "bn2f table sizes disagree: "
            f"expected {n_points}, got {n2_272.size}, {n2_228.size}, "
            f"and {h2o_efficiency.size}"
        )

    wavenumber = v1 + dv * np.arange(n_points, dtype=float)
    if not np.isclose(wavenumber[-1], v2, rtol=0.0, atol=1.0e-5):
        raise ValueError(f"bn2f grid ends at {wavenumber[-1]}, expected {v2}")

    overtone_grid = _data_array(overtone_block, "V1n2f,V2n2f,DVn2f,NPTn2f")
    if overtone_grid.size != 4:
        raise ValueError(f"expected four bn2f1 grid values, found {overtone_grid.size}")
    overtone_v1, overtone_v2, overtone_dv, overtone_n_float = overtone_grid
    overtone_n = int(overtone_n_float)
    overtone = _data_array(overtone_block, "xn2")
    if overtone.size != overtone_n:
        raise ValueError(f"expected {overtone_n} bn2f1 values, found {overtone.size}")
    overtone_wavenumber = overtone_v1 + overtone_dv * np.arange(overtone_n, dtype=float)
    if not np.isclose(overtone_wavenumber[-1], overtone_v2, rtol=0.0, atol=1.0e-5):
        raise ValueError(
            f"bn2f1 grid ends at {overtone_wavenumber[-1]}, expected {overtone_v2}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        wavenumber_cm=wavenumber,
        n2_n2_272=n2_272,
        n2_n2_228=n2_228,
        h2o_relative_efficiency=h2o_efficiency,
        overtone_wavenumber_cm=overtone_wavenumber,
        overtone_n2_n2=overtone,
        source=np.asarray("LBLRTM 12.11 contnm.f90 bn2f and bn2f1 tables"),
    )
    print(
        f"wrote {output} ({n_points} fundamental and {overtone_n} overtone coefficients)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)


if __name__ == "__main__":
    main()
