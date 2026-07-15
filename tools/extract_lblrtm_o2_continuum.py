"""Extract ground-based O2 continuum tables from LBLRTM 12.11.

This provenance tool is used only while building PyMolFit package data.
PyMolFit does not import or execute LBLRTM at runtime.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _numbers(text: str) -> np.ndarray:
    tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?", text)
    return np.asarray([float(token.replace("D", "E").replace("d", "e")) for token in tokens])


def _block(text: str, name: str) -> str:
    match = re.search(
        rf"BLOCK\s+DATA\s+{re.escape(name)}\b(.*?)end\s+block\s+data\s+{re.escape(name)}",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"could not find LBLRTM block data {name}")
    return match.group(1)


def _data(block: str, name: str) -> np.ndarray:
    match = re.search(
        rf"DATA\s+{re.escape(name)}\s*(?:&\s*)*/(.*?)/",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"could not find DATA {name}")
    return _numbers(match.group(1))


def _series(block: str, prefix: str) -> np.ndarray:
    matches = re.findall(
        rf"DATA\s+({re.escape(prefix)}(\d+))\s*(?:&\s*)*/(.*?)/",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        raise ValueError(f"could not find DATA series {prefix}*")
    ordered = sorted(matches, key=lambda item: int(item[1]))
    return np.concatenate([_numbers(item[2]) for item in ordered])


def _grid(block: str, name: str = "V1,V2,DV,NPT") -> tuple[np.ndarray, int]:
    values = _data(block, name)
    if values.size != 4:
        raise ValueError(f"expected four grid values for {name}, found {values.size}")
    v1, v2, spacing, count_float = values
    count = int(count_float)
    grid = v1 + spacing * np.arange(count, dtype=float)
    if not np.isclose(grid[-1], v2, rtol=0.0, atol=1.0e-5):
        raise ValueError(f"grid ends at {grid[-1]}, expected {v2}")
    return grid, count


def _require_size(name: str, values: np.ndarray, count: int) -> np.ndarray:
    if values.size != count:
        raise ValueError(f"{name}: expected {count} values, found {values.size}")
    return values


def extract(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="ignore")

    fundamental_block = _block(text, "bo2f")
    fundamental_grid, fundamental_count = _grid(
        fundamental_block, "V1S,V2S,DVS,NPTS"
    )
    fundamental = _require_size(
        "O2 fundamental", _series(fundamental_block, "o0"), fundamental_count
    )
    fundamental_t = _require_size(
        "O2 fundamental temperature", _series(fundamental_block, "ot0"), fundamental_count
    )

    inf1_block = _block(text, "bo2inf1")
    inf1_grid, inf1_count = _grid(inf1_block)
    inf1 = _require_size("O2 1.27 micron", _series(inf1_block, "o0"), inf1_count)

    aband_block = _block(text, "bo2inf3")
    aband_grid, aband_count = _grid(aband_block)
    aband = _require_size("O2 A band", _data(aband_block, "x02inf3"), aband_count)

    visible_block = _block(text, "bo2in_vis")
    visible_grid, visible_count = _grid(visible_block)
    visible = _require_size(
        "O2 visible", _series(visible_block, "o2vis"), visible_count
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fundamental_wavenumber_cm=fundamental_grid,
        fundamental_coefficient=fundamental,
        fundamental_temperature_coefficient=fundamental_t,
        inf1_wavenumber_cm=inf1_grid,
        inf1_coefficient=inf1,
        aband_wavenumber_cm=aband_grid,
        aband_coefficient=aband,
        visible_wavenumber_cm=visible_grid,
        visible_coefficient=visible,
        source=np.asarray(
            "LBLRTM 12.11 contnm.f90 bo2f, bo2inf1, bo2inf3, and bo2in_vis tables"
        ),
    )
    print(
        "wrote "
        f"{output} ({fundamental_count}, {inf1_count}, {aband_count}, "
        f"{visible_count} coefficients)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)


if __name__ == "__main__":
    main()
