from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import astropy
import numpy as np
import scipy

from ._version import __version__


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.dtype.kind.encode("ascii"))
    if array.dtype.kind in "OU":
        for item in array.ravel(order="C"):
            encoded = str(item).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "NaN"
        if np.isposinf(numeric):
            return "Infinity"
        if np.isneginf(numeric):
            return "-Infinity"
        return numeric
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "array_sha256": _array_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)},
        }
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}"}


def scientific_object_sha256(value: Any) -> str:
    """Hash arrays and nested scientific dataclasses in a deterministic form."""

    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _spectrum_summary(spectrum: Any) -> dict[str, Any]:
    payload = {
        "wavelength": np.asarray(spectrum.wavelength),
        "flux": np.asarray(spectrum.flux),
        "uncertainty": None if spectrum.uncertainty is None else np.asarray(spectrum.uncertainty),
        "mask": None if spectrum.mask is None else np.asarray(spectrum.mask),
        "wavelength_unit": spectrum.wavelength_unit,
        "wavelength_medium": spectrum.wavelength_medium,
    }
    return {
        "sha256": scientific_object_sha256(payload),
        "pixels": int(np.asarray(spectrum.wavelength).size),
        "source": str(spectrum.meta.get("source", "")),
        "source_file_sha256": str(spectrum.meta.get("source_file_sha256", "")),
    }


def build_fit_provenance(
    spectra: Any | Sequence[Any],
    *,
    line_list: Any,
    selected_line_list: Any | None,
    config: Any,
    fit_pixel_counts: Sequence[int],
) -> dict[str, Any]:
    """Build machine-readable provenance for a fitted telluric model."""

    if hasattr(spectra, "wavelength"):
        spectrum_items = (spectra,)
    else:
        spectrum_items = tuple(spectra)
    spectrum_summaries = [_spectrum_summary(spectrum) for spectrum in spectrum_items]
    atmosphere = getattr(config, "atmosphere", None)
    components = getattr(config, "components", None)
    partition_table = getattr(config, "partition_table", None)
    config_summary = _jsonable(config)
    selected = line_list if selected_line_list is None else selected_line_list
    atmosphere_metadata = {} if atmosphere is None else _jsonable(atmosphere.metadata)

    return {
        "schema_version": 1,
        "pymolfit_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "astropy_version": astropy.__version__,
        "spectra": spectrum_summaries,
        "fit_pixel_counts": [int(value) for value in fit_pixel_counts],
        "line_source": str(getattr(line_list, "line_source", "unknown")),
        "line_data_provenance": _jsonable(getattr(line_list, "data_provenance", {})),
        "line_count": int(np.asarray(line_list.wavelength).size),
        "selected_line_count": int(np.asarray(selected.wavelength).size),
        "line_species": list(getattr(line_list, "species_names", ())),
        "line_list_sha256": scientific_object_sha256(line_list),
        "selected_line_list_sha256": scientific_object_sha256(selected),
        "atmosphere_profile_sha256": (
            "" if atmosphere is None else scientific_object_sha256(atmosphere.layers)
        ),
        "atmosphere_layer_count": 0 if atmosphere is None else len(atmosphere.layers),
        "atmosphere_metadata": atmosphere_metadata,
        "partition_table_sha256": (
            "" if partition_table is None else scientific_object_sha256(partition_table)
        ),
        "components_sha256": "" if components is None else scientific_object_sha256(components),
        "fit_config_sha256": scientific_object_sha256(config_summary),
        "fit_config": config_summary,
    }


def provenance_json(provenance: Mapping[str, Any]) -> str:
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
