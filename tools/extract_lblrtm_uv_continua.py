"""Extract ozone and far-UV oxygen tables from LBLRTM 12.11 contnm.f90.

Development-only provenance tool; the installed package needs no Fortran source.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np


def _numbers(value: str) -> np.ndarray:
    values = []
    for token in value.replace("&", " ").replace("\n", " ").split(","):
        token = token.strip()
        if not token:
            continue
        count, number = token.split("*", 1) if "*" in token else ("1", token)
        values.extend([float(number.replace("D", "E").replace("d", "e"))] * int(count))
    return np.asarray(values)


def _block(text: str, name: str) -> str:
    match = re.search(
        rf"BLOCK\s+DATA\s+{name}\b(.*?)end\s+block\s+data\s+{name}\b",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing block {name}")
    return match[1]


def _common_arrays(block: str, common: str) -> list[np.ndarray]:
    declaration = re.search(rf"COMMON\s*/{common}/(.*?)\bDATA\b", block, re.IGNORECASE | re.DOTALL)
    if declaration is None:
        raise ValueError(f"missing COMMON /{common}/")
    arrays = []
    # Preserve COMMON storage order, which is not numerical order for O3HH1/2.
    for name, count in re.findall(r"(\w+)\(\s*(\d+)\s*\)", declaration[1]):
        match = re.search(rf"DATA\s+{name}\s*(?:&\s*)?/(.*?)/", block, re.I | re.S)
        if match is None:
            raise ValueError(f"missing DATA {name}")
        values = _numbers(match[1])
        if values.size != int(count):
            raise ValueError(f"{name}: expected {count} values, found {values.size}")
        arrays.append(values)
    return arrays


def _grid(block: str) -> np.ndarray:
    values = []
    for name in ("V1C", "V2C", "DVC", "NC"):
        match = re.search(rf"\b{name}\s*/([^/]+)/", block, re.I)
        if match is None:
            raise ValueError(f"missing grid field {name}")
        values.append(float(match[1]))
    first, last, step, count = values
    grid = first + step * np.arange(int(count))
    if grid[-1] != last:
        raise ValueError("inconsistent source grid")
    return grid


def extract(source: Path, output_directory: Path) -> None:
    raw = source.read_bytes()
    text = re.sub(r"!.*", "", raw.decode("utf-8"))
    provenance = {
        "source": np.asarray("LBLRTM 12.11 contnm.f90"),
        "source_sha256": np.asarray(hashlib.sha256(raw).hexdigest()),
    }
    chappuis = _block(text, "O3CH")
    chappuis_arrays = []
    for name in ("X", "Y", "Z"):
        values = np.full(3150, np.nan)
        pattern = rf"DATA\s*\({name}\(I\),\s*I\s*=\s*(\d+),\s*(\d+)\)\s*/(.*?)/"
        for start, end, data in re.findall(pattern, chappuis, re.I | re.S):
            values[int(start) - 1 : int(end)] = _numbers(data)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"incomplete Chappuis {name} array")
        chappuis_arrays.append(values)
    ozone = {
        "chappuis_wavenumber_cm": np.arange(8920.0, 24666.0, 5.0),
        "chappuis_coefficient": np.asarray(chappuis_arrays) * 1e-20,
    }
    hh_arrays = []
    for i in range(3):
        block = _block(text, f"BO3HH{i}")
        grid = _grid(block)
        values = np.concatenate(_common_arrays(block, f"O3HH{i}"))
        if values.size != grid.size:
            raise ValueError(f"inconsistent Hartley-Huggins {i} table")
        hh_arrays.append(values)
    ozone["huggins_wavenumber_cm"] = grid
    ozone["huggins_coefficient"] = hh_arrays[0] * 1e-20
    ozone["huggins_temperature_coefficient"] = np.asarray(hh_arrays[1:])
    uv = _block(text, "BO3HUV")
    ozone["hartley_wavenumber_cm"] = _grid(uv)
    ozone["hartley_coefficient"] = np.concatenate(_common_arrays(uv, "O3HUV"))
    fuv = _block(text, "bo2in_fuv")
    oxygen = {
        "wavenumber_cm": np.arange(56740.0, 86961.0, 20.0),
        "cross_section_cm2": np.concatenate(_common_arrays(fuv, "o2_fuv")) * 1e-20,
    }
    if oxygen["wavenumber_cm"].size != oxygen["cross_section_cm2"].size:
        raise ValueError("inconsistent O2 far-UV table")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("o3_continuum", ozone), ("o2_uv_continuum", oxygen)):
        output = output_directory / f"lblrtm_v12_11_{name}.npz"
        np.savez_compressed(output, **data, **provenance)
        print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output_directory)
