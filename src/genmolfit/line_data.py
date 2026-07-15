from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .hitran import HITRAN_MOLECULES, parse_hitran_local_iso_id
from .linelist import LineList
from .provenance import file_sha256

HITRAN_API_BASE_URL = "https://hitran.org"
HITRAN_API_VERSION = "v2"
HITRAN_API_KEY_ENV = "HITRAN_API_KEY"
HITRAN_CACHE_ENV = "GENMOLFIT_HITRAN_CACHE"
HITRAN_CITATION_URL = "https://hitran.org/citepolicy/"
HITRAN_DOCUMENTATION_URL = "https://hitran.org/docs/"
LINE_DATA_SCHEMA_VERSION = 1

_SPECIES_TO_MOLECULE_ID = {name.upper(): mol_id for mol_id, (name, _) in HITRAN_MOLECULES.items()}
_OpenUrl = Callable[..., Any]


class HitranAcquisitionError(RuntimeError):
    """A HITRAN line-data request failed or returned unusable data."""


@dataclass(frozen=True)
class HitranLineArtifact:
    """Validated local files for one reproducible HITRAN query."""

    par_path: Path
    table_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    line_list: LineList
    cache_hit: bool


def default_hitran_cache_dir() -> Path:
    configured = os.environ.get(HITRAN_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "genmolfit" / "hitran"


def fetch_hitran_lines(
    species: Iterable[str],
    *,
    wavelength_min_micron: float | None = None,
    wavelength_max_micron: float | None = None,
    wavenumber_min_cm: float | None = None,
    wavenumber_max_cm: float | None = None,
    api_key: str | None = None,
    api_key_env: str = HITRAN_API_KEY_ENV,
    cache_dir: str | Path | None = None,
    force: bool = False,
    timeout_s: float = 60.0,
    base_url: str = HITRAN_API_BASE_URL,
    opener: _OpenUrl = urlopen,
) -> HitranLineArtifact:
    """Download and cache a line window through the authenticated HITRAN API.

    The API key is read from ``HITRAN_API_KEY`` by default and is never written
    to the cache or included in an exception. A valid cache hit can be used
    offline without supplying the key again.
    """

    request_spec = _canonical_request(
        species,
        wavelength_min_micron=wavelength_min_micron,
        wavelength_max_micron=wavelength_max_micron,
        wavenumber_min_cm=wavenumber_min_cm,
        wavenumber_max_cm=wavenumber_max_cm,
        source="hitran_api",
    )
    paths = _artifact_paths(_resolve_cache_dir(cache_dir), request_spec)
    if not force:
        cached = _load_cached_artifact(paths, request_spec)
        if cached is not None:
            return cached

    secret = api_key if api_key is not None else os.environ.get(api_key_env)
    if not secret or not secret.strip():
        raise HitranAcquisitionError(
            f"no HITRAN API key supplied; set {api_key_env} or pass api_key to fetch_hitran_lines()"
        )
    secret = secret.strip()
    host = base_url.rstrip("/")
    api_root = f"{host}/api/{HITRAN_API_VERSION}/{secret}"

    try:
        info = _request_json(f"{api_root}/info", opener=opener, timeout_s=timeout_s)
        isotopologues = _request_json(
            f"{api_root}/isotopologues?"
            + urlencode({"molecule_id__in": ",".join(str(value) for value in request_spec["molecule_ids"])}),
            opener=opener,
            timeout_s=timeout_s,
        )
        iso_rows = _response_data(isotopologues, expected="isotopologue metadata")
        if not isinstance(iso_rows, list):
            raise HitranAcquisitionError("HITRAN returned malformed isotopologue metadata")
        global_iso_ids = sorted(
            {
                int(row["id"])
                for row in iso_rows
                if isinstance(row, Mapping) and row.get("id") is not None
            }
        )
        if not global_iso_ids:
            raise HitranAcquisitionError("HITRAN returned no isotopologues for the requested molecules")

        query = urlencode(
            {
                "iso_ids_list": ",".join(str(value) for value in global_iso_ids),
                "numin": _format_query_float(request_spec["wavenumber_min_cm"]),
                "numax": _format_query_float(request_spec["wavenumber_max_cm"]),
                "head": "False",
                "fixwidth": "0",
                "request_params": "par_line,trans_id,global_iso_id",
            }
        )
        transition_header = _request_json(
            f"{api_root}/transitions?{query}", opener=opener, timeout_s=timeout_s
        )
        result_name = _response_data(transition_header, expected="transition result filename")
        if not isinstance(result_name, str) or not result_name.strip():
            raise HitranAcquisitionError("HITRAN returned no transition result filename")
        results_dir = _find_named_value(_response_data(info, expected="server information"), "results_dir")
        if not isinstance(results_dir, str) or not results_dir.strip():
            raise HitranAcquisitionError("HITRAN server information did not include results_dir")
        result_url = f"{host}/{results_dir.strip('/')}/{result_name.strip('/')}"
        transition_text = _request_text(result_url, opener=opener, timeout_s=timeout_s)
    except HitranAcquisitionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive sanitization boundary
        raise HitranAcquisitionError("unexpected failure while acquiring HITRAN line data") from exc

    par_text, row_global_ids = _extract_par_records(transition_text)
    line_list = _validated_line_list(par_text, request_spec)
    line_list = _attach_api_isotopologue_metadata(line_list, row_global_ids, iso_rows)
    server_data = _response_data(info, expected="server information")
    manifest_extra = {
        "api_version": HITRAN_API_VERSION,
        "database_edition": _first_named_value(
            server_data,
            ("database_edition", "edition", "database_version", "version"),
        ),
        "global_isotopologue_ids": global_iso_ids,
        "result_filename": Path(result_name).name,
    }
    return _write_artifact(paths, request_spec, par_text, line_list, manifest_extra=manifest_extra)


def cache_hitran_par(
    path: str | Path,
    *,
    species: Iterable[str],
    wavelength_min_micron: float | None = None,
    wavelength_max_micron: float | None = None,
    wavenumber_min_cm: float | None = None,
    wavenumber_max_cm: float | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> HitranLineArtifact:
    """Validate and cache a user-supplied HITRAN ``.par`` window."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    request_spec = _canonical_request(
        species,
        wavelength_min_micron=wavelength_min_micron,
        wavelength_max_micron=wavelength_max_micron,
        wavenumber_min_cm=wavenumber_min_cm,
        wavenumber_max_cm=wavenumber_max_cm,
        source="user_hitran_par",
    )
    request_spec["input_sha256"] = file_sha256(input_path)
    paths = _artifact_paths(_resolve_cache_dir(cache_dir), request_spec)
    if not force:
        cached = _load_cached_artifact(paths, request_spec)
        if cached is not None:
            return cached

    source_text = input_path.read_text(encoding="utf-8", errors="strict")
    selected_text = _select_par_records(source_text, request_spec)
    line_list = _validated_line_list(selected_text, request_spec)
    return _write_artifact(
        paths,
        request_spec,
        selected_text,
        line_list,
        manifest_extra={"input_path": str(input_path), "input_sha256": request_spec["input_sha256"]},
    )


def _canonical_request(
    species: Iterable[str],
    *,
    wavelength_min_micron: float | None,
    wavelength_max_micron: float | None,
    wavenumber_min_cm: float | None,
    wavenumber_max_cm: float | None,
    source: str,
) -> dict[str, Any]:
    names = tuple(sorted({str(value).strip().upper() for value in species if str(value).strip()}))
    if not names:
        raise ValueError("species must contain at least one HITRAN molecule name")
    unsupported = tuple(name for name in names if name not in _SPECIES_TO_MOLECULE_ID)
    if unsupported:
        supported = ", ".join(sorted(_SPECIES_TO_MOLECULE_ID))
        raise ValueError(f"unsupported HITRAN species {unsupported}; supported names are: {supported}")

    use_wavelength = wavelength_min_micron is not None or wavelength_max_micron is not None
    use_wavenumber = wavenumber_min_cm is not None or wavenumber_max_cm is not None
    if use_wavelength == use_wavenumber:
        raise ValueError("provide exactly one complete wavelength or wavenumber interval")
    if use_wavelength:
        if wavelength_min_micron is None or wavelength_max_micron is None:
            raise ValueError("both wavelength_min_micron and wavelength_max_micron are required")
        lower_wave, upper_wave = _ordered_positive_interval(
            wavelength_min_micron, wavelength_max_micron, "wavelength"
        )
        lower_nu = 1.0e4 / upper_wave
        upper_nu = 1.0e4 / lower_wave
    else:
        if wavenumber_min_cm is None or wavenumber_max_cm is None:
            raise ValueError("both wavenumber_min_cm and wavenumber_max_cm are required")
        lower_nu, upper_nu = _ordered_positive_interval(
            wavenumber_min_cm, wavenumber_max_cm, "wavenumber"
        )
        lower_wave = 1.0e4 / upper_nu
        upper_wave = 1.0e4 / lower_nu

    return {
        "source": source,
        "species": list(names),
        "molecule_ids": [_SPECIES_TO_MOLECULE_ID[name] for name in names],
        "wavelength_min_micron": float(lower_wave),
        "wavelength_max_micron": float(upper_wave),
        "wavenumber_min_cm": float(lower_nu),
        "wavenumber_max_cm": float(upper_nu),
    }


def _ordered_positive_interval(lower: float, upper: float, label: str) -> tuple[float, float]:
    values = np.asarray([lower, upper], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or values[0] == values[1]:
        raise ValueError(f"{label} bounds must be finite, positive, and distinct")
    return float(np.min(values)), float(np.max(values))


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    target = default_hitran_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def _request_json(url: str, *, opener: _OpenUrl, timeout_s: float) -> Mapping[str, Any]:
    text = _request_text(url, opener=opener, timeout_s=timeout_s)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HitranAcquisitionError("HITRAN returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise HitranAcquisitionError("HITRAN returned an invalid response object")
    status = str(payload.get("status", "OK")).upper()
    if status not in {"OK", "SUCCESS", "200"}:
        message = str(payload.get("message", "request rejected"))
        raise HitranAcquisitionError(f"HITRAN request failed: {message}")
    return payload


def _request_text(url: str, *, opener: _OpenUrl, timeout_s: float) -> str:
    request = Request(url, headers={"User-Agent": "GenMolFit line-data client"})
    try:
        response = opener(request, timeout=float(timeout_s))
        with response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code in {401, 403}:
            reason = "authentication was rejected"
        elif exc.code == 429:
            reason = "the HITRAN request limit was reached"
        else:
            reason = f"HTTP {exc.code}"
        raise HitranAcquisitionError(f"HITRAN request failed: {reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HitranAcquisitionError("could not reach the HITRAN service") from exc


def _response_data(payload: Mapping[str, Any], *, expected: str) -> Any:
    content = payload.get("content")
    if not isinstance(content, Mapping) or "data" not in content:
        raise HitranAcquisitionError(f"HITRAN response did not contain {expected}")
    return content["data"]


def _find_named_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
        for item in value.values():
            found = _find_named_value(item, name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_named_value(item, name)
            if found is not None:
                return found
    return None


def _first_named_value(value: Any, names: Iterable[str]) -> Any:
    for name in names:
        found = _find_named_value(value, name)
        if found is not None:
            return found
    return None


def _format_query_float(value: float) -> str:
    return format(float(value), ".12g")


def _extract_par_records(text: str) -> tuple[str, list[int | None]]:
    rows: list[str] = []
    global_ids: list[int | None] = []
    for raw in text.splitlines():
        if len(raw) < 160 or not raw[:2].strip().isdigit():
            continue
        record = raw[:160]
        try:
            parse_hitran_local_iso_id(record[2:3])
            float(record[3:15])
        except ValueError as exc:
            raise HitranAcquisitionError("HITRAN returned a malformed fixed-width transition") from exc
        rows.append(record)
        extras = raw[160:].lstrip(",").split(",") if raw[160:].strip(" ,") else []
        try:
            global_ids.append(int(extras[-1]) if extras else None)
        except ValueError:
            global_ids.append(None)
    if not rows:
        raise HitranAcquisitionError("HITRAN returned no fixed-width transition records")
    return "\n".join(rows) + "\n", global_ids


def _select_par_records(text: str, request_spec: Mapping[str, Any]) -> str:
    wanted_ids = set(int(value) for value in request_spec["molecule_ids"])
    lower = float(request_spec["wavenumber_min_cm"])
    upper = float(request_spec["wavenumber_max_cm"])
    rows = []
    for raw in text.splitlines():
        if len(raw) < 67 or not raw[:2].strip().isdigit():
            continue
        molecule_id = int(raw[:2])
        wavenumber = float(raw[3:15].replace("D", "E"))
        if molecule_id in wanted_ids and lower <= wavenumber <= upper:
            rows.append(raw[:160].ljust(160))
    if not rows:
        raise HitranAcquisitionError("no records in the supplied HITRAN file match the requested window")
    return "\n".join(rows) + "\n"


def _validated_line_list(par_text: str, request_spec: Mapping[str, Any]) -> LineList:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".par", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(par_text)
    try:
        line_list = LineList.from_hitran_par(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    unexpected = set(line_list.species_names) - set(request_spec["species"])
    if unexpected:
        raise HitranAcquisitionError(f"line data contained unexpected species: {sorted(unexpected)}")
    wavenumber = np.asarray(line_list.wavenumber, dtype=float)
    tolerance = 1.0e-7
    if np.any(wavenumber < float(request_spec["wavenumber_min_cm"]) - tolerance) or np.any(
        wavenumber > float(request_spec["wavenumber_max_cm"]) + tolerance
    ):
        raise HitranAcquisitionError("line data extended outside the requested wavenumber interval")
    return line_list


def _attach_api_isotopologue_metadata(
    line_list: LineList,
    row_global_ids: list[int | None],
    iso_rows: list[Any],
) -> LineList:
    if len(row_global_ids) != line_list.wavelength.size:
        raise HitranAcquisitionError("HITRAN isotopologue metadata was not aligned with transition rows")
    metadata = {
        int(row["id"]): row
        for row in iso_rows
        if isinstance(row, Mapping) and row.get("id") is not None
    }
    global_ids = np.asarray([-1 if value is None else value for value in row_global_ids], dtype=int)
    abundance = np.full(global_ids.shape, np.nan, dtype=float)
    mass = np.asarray(line_list.molecular_mass_amu, dtype=float).copy()
    for global_id in np.unique(global_ids[global_ids > 0]):
        row = metadata.get(int(global_id), {})
        keep = global_ids == global_id
        if row.get("abundance") is not None:
            abundance[keep] = float(row["abundance"])
        if row.get("mass") is not None:
            mass[keep] = float(row["mass"])
    return replace(
        line_list,
        global_iso_id=global_ids,
        natural_abundance=abundance,
        isotopologue_abundance_scale=np.ones(global_ids.shape, dtype=float),
        molecular_mass_amu=mass,
        line_source="hitran_api_v2",
    )


def _artifact_paths(cache_dir: Path, request_spec: Mapping[str, Any]) -> dict[str, Path]:
    fingerprint = _request_fingerprint(request_spec)
    species = "-".join(str(value).lower() for value in request_spec["species"])
    stem = f"{species}_{fingerprint[:16]}"
    return {
        "par": cache_dir / f"{stem}.par",
        "table": cache_dir / f"{stem}.ecsv",
        "manifest": cache_dir / f"{stem}.json",
    }


def _request_fingerprint(request_spec: Mapping[str, Any]) -> str:
    encoded = json.dumps(request_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_artifact(
    paths: Mapping[str, Path], request_spec: Mapping[str, Any]
) -> HitranLineArtifact | None:
    if not all(paths[name].is_file() for name in ("par", "table", "manifest")):
        return None
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("request") != request_spec:
            return None
        if manifest.get("request_sha256") != _request_fingerprint(request_spec):
            return None
        files = manifest["files"]
        if files["par"]["sha256"] != file_sha256(paths["par"]):
            return None
        if files["table"]["sha256"] != file_sha256(paths["table"]):
            return None
        line_list = LineList.from_table(paths["table"])
        _validate_manifest_line_list(manifest, line_list)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return HitranLineArtifact(
        par_path=paths["par"],
        table_path=paths["table"],
        manifest_path=paths["manifest"],
        manifest=manifest,
        line_list=line_list,
        cache_hit=True,
    )


def _validate_manifest_line_list(manifest: Mapping[str, Any], line_list: LineList) -> None:
    if int(manifest["line_count"]) != line_list.wavelength.size:
        raise ValueError("cached HITRAN line count does not match its manifest")
    if sorted(manifest["species_with_lines"]) != list(line_list.species_names):
        raise ValueError("cached HITRAN species do not match their manifest")
    wavenumber = np.asarray(line_list.wavenumber, dtype=float)
    actual = manifest["actual_wavenumber_range_cm"]
    if not np.allclose([np.min(wavenumber), np.max(wavenumber)], actual, rtol=0.0, atol=1.0e-8):
        raise ValueError("cached HITRAN line range does not match its manifest")


def _write_artifact(
    paths: Mapping[str, Path],
    request_spec: Mapping[str, Any],
    par_text: str,
    line_list: LineList,
    *,
    manifest_extra: Mapping[str, Any],
) -> HitranLineArtifact:
    request_hash = _request_fingerprint(request_spec)
    table = line_list.to_table()
    table.meta.update(
        {
            "line_data_schema_version": LINE_DATA_SCHEMA_VERSION,
            "line_data_request_sha256": request_hash,
            "line_data_source": request_spec["source"],
            "hitran_citation_url": HITRAN_CITATION_URL,
        }
    )
    _atomic_write_text(paths["par"], par_text)
    _atomic_write_table(paths["table"], table)
    wavenumber = np.asarray(line_list.wavenumber, dtype=float)
    manifest = {
        "schema_version": LINE_DATA_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request": dict(request_spec),
        "request_sha256": request_hash,
        "line_count": int(line_list.wavelength.size),
        "species_with_lines": list(line_list.species_names),
        "actual_wavenumber_range_cm": [float(np.min(wavenumber)), float(np.max(wavenumber))],
        "actual_wavelength_range_micron": [
            float(np.min(line_list.wavelength)),
            float(np.max(line_list.wavelength)),
        ],
        "files": {
            "par": {"name": paths["par"].name, "sha256": file_sha256(paths["par"])},
            "table": {"name": paths["table"].name, "sha256": file_sha256(paths["table"])},
        },
        "citations": {
            "hitran": HITRAN_CITATION_URL,
            "documentation": HITRAN_DOCUMENTATION_URL,
        },
        **dict(manifest_extra),
    }
    _atomic_write_text(paths["manifest"], json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    loaded_line_list = LineList.from_table(paths["table"])
    return HitranLineArtifact(
        par_path=paths["par"],
        table_path=paths["table"],
        manifest_path=paths["manifest"],
        manifest=manifest,
        line_list=loaded_line_list,
        cache_hit=False,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_table(path: Path, table: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".ecsv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        table.write(temporary, format="ascii.ecsv", overwrite=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
