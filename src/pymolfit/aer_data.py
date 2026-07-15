from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .hitran import HITRAN_MOLECULES
from .linelist import LineList
from .provenance import file_sha256


LOGGER = logging.getLogger(__name__)

AER_CATALOG_VERSION = "3.9"
AER_CATALOG_FILENAME = "aer_v_3.9"
AER_ARCHIVE_FILENAME = "aer_v_3.9.tgz"
AER_ARCHIVE_SHA256 = "3a55d5deb9430894ce61e1a23475ad1ed7ef01f0a5c527ee6f08d9e5676a0e4e"
AER_CATALOG_SHA256 = "08c990768fc3b69dc65538325eadd4df633106a0e6456cba3da072cf4357acb8"
AER_ZENODO_RECORD = "18881607"
AER_SOURCE_URL = (
    f"https://zenodo.org/records/{AER_ZENODO_RECORD}/files/"
    f"{AER_ARCHIVE_FILENAME}?download=1"
)
AER_CACHE_ENV = "PYMOLFIT_AER_CACHE"
AER_SOURCE_URL_ENV = "PYMOLFIT_AER_URL"
AER_CATALOG_PATH_ENV = "PYMOLFIT_AER_CATALOG"
AER_MOLECFIT_ROOT_ENV = "PYMOLFIT_MOLECFIT_ROOT"
AER_DATA_SCHEMA_VERSION = 2
AER_WINDOW_SCHEMA_VERSION = 3
AER_LICENSE_URL = (
    f"https://zenodo.org/records/{AER_ZENODO_RECORD}/files/LICENSE?download=1"
)
AER_SOURCE_PAGE = f"https://doi.org/10.5281/zenodo.{AER_ZENODO_RECORD}"
AER_LICENSE_TEXT = """Copyright ©, Atmospheric and Environmental Research, Inc., 2020. All rights reserved.  Atmospheric and Environmental Research,  Inc. (AER) grants USER the right to download, install, use and copy this  database for scientific and research purposes only.  This database may be redistributed as long as this copyright notice is reproduced on any copy made and appropriate acknowledgment is given to AER. This database or any modified version of this database may not be incorporated into any proprietary product or commercial product offered for sale without express written consent of AER. This database is provided as is without any express or implied warranties."""

# These hashes identify the exact payload in the immutable official AER 3.9
# Zenodo record. Files not yet consumed by PyMolFit are retained so the
# managed catalogue remains a complete scientific input.
AER_FILE_SPECS: Mapping[str, tuple[str, str]] = {
    AER_CATALOG_FILENAME: (
        "aer_v_3.9/line_file/aer_v_3.9",
        AER_CATALOG_SHA256,
    ),
    "co2_co2_brd_param": (
        "aer_v_3.9/extra_brd_params/co2_co2_brd_param",
        "049e533568ea4503be2915ee80ee93947d754de997bfb0fe68b2afbe55a98f94",
    ),
    "co2_h2o_brd_param": (
        "aer_v_3.9/extra_brd_params/co2_h2o_brd_param",
        "7de9aab9081fa5de2ba02ea1e902e1e4d4be6af732e4c56234e5658648f3d749",
    ),
    "o2_h2o_brd_param": (
        "aer_v_3.9/extra_brd_params/o2_h2o_brd_param",
        "836382cf27250c980c14f1e48b367b7c5a59e148bd1ec482c2d32a1d2d30ca7a",
    ),
    "o2_o2_brd_param": (
        "aer_v_3.9/extra_brd_params/o2_o2_brd_param",
        "5b48d0b42a19f6831cb4541940adb63600d4c9a0a93875b26f147082a570bb7c",
    ),
    "o2_uv_brd_param": (
        "aer_v_3.9/extra_brd_params/o2_uv_brd_param",
        "4fcc08ab75a7e17b998d76780b97bf3a33186ba75bf5bda8be3c121c1c964032",
    ),
    "spd_dep_param": (
        "aer_v_3.9/extra_brd_params/spd_dep_param",
        "179d513b562130cb75264a2028af2ca13aea4679732998d0020bf24d71574955",
    ),
    "wv_co2_brd_param": (
        "aer_v_3.9/extra_brd_params/wv_co2_brd_param",
        "474a8a65af212124cee9dc5fb60dbfa2fa9b76c933006b3bb3f554a43559d01d",
    ),
    "lncpl_lines": (
        "aer_v_3.9/lncpl_lines",
        "a84cd6ef22e89a3abad4144da752cd3aabc1cccd9825bc8b48cbc9cbaf857a7d",
    ),
}

_OPTIONAL_ARCHIVE_MEMBERS: Mapping[str, str] = {
    "AER_LICENSE.txt": "aer_v_3.9/LICENSE",
    "AER_RELEASE_NOTES.txt": "aer_v_3.9/RELEASE_NOTES_aer_linefile",
}
_DEFAULT_SPECIES = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2")
_OpenUrl = Callable[..., Any]


class AERDataError(RuntimeError):
    """The versioned AER catalogue could not be acquired or verified."""


@dataclass(frozen=True)
class AERCatalogArtifact:
    """A verified AER line catalogue and its associated broadener files."""

    catalog_path: Path
    extra_broadener_dir: Path
    manifest_path: Path | None
    manifest: Mapping[str, Any]
    source: str
    cache_hit: bool
    managed: bool


@dataclass(frozen=True)
class AERLineArtifact:
    """One reproducible, cached line window selected from the AER catalogue."""

    table_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    line_list: LineList
    catalog: AERCatalogArtifact
    cache_hit: bool


def default_aer_cache_dir() -> Path:
    configured = os.environ.get(AER_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "pymolfit" / "aer"


def find_local_aer_catalog(
    *,
    catalog_path: str | Path | None = None,
) -> AERCatalogArtifact | None:
    """Find and verify an exact local AER 3.9 catalogue."""

    for candidate in _local_catalog_candidates(catalog_path):
        artifact = _external_artifact(candidate)
        if artifact is not None:
            return artifact
    return None


def aer_catalog_status(
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    reuse_molecfit: bool = True,
) -> AERCatalogArtifact | None:
    """Return a verified cached/local catalogue without using the network."""

    managed = _load_managed_artifact(_version_dir(cache_dir))
    if managed is not None:
        return managed
    if reuse_molecfit:
        adopted = _load_adopted_artifact(cache_dir)
        if adopted is not None:
            return adopted
        local = find_local_aer_catalog(catalog_path=catalog_path)
        if local is not None:
            _record_adopted_artifact(local, cache_dir)
        return local
    return None


def install_aer_catalog(
    *,
    source: str | Path | None = None,
    source_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    force: bool = False,
    offline: bool = False,
    reuse_molecfit: bool = True,
    timeout_s: float = 120.0,
    opener: _OpenUrl = urlopen,
    progress: Callable[[str], None] | None = None,
) -> AERCatalogArtifact:
    """Resolve the official AER catalogue, installing it on cache miss.

    Resolution does not require a compiler or an authenticated HITRAN request:

    1. use a verified PyMolFit-managed cache;
    2. reuse a verified exact local catalogue without copying it;
    3. download the pinned official AER Zenodo archive.
    """

    target_dir = _version_dir(cache_dir)
    if not force:
        cached = _load_managed_artifact(target_dir)
        if cached is not None:
            return cached
        if reuse_molecfit and source is None and catalog_path is None:
            adopted = _load_adopted_artifact(cache_dir)
            if adopted is not None:
                return adopted
            local = find_local_aer_catalog(catalog_path=catalog_path)
            if local is not None:
                _record_adopted_artifact(local, cache_dir)
                return local

    if catalog_path is not None:
        local = find_local_aer_catalog(catalog_path=catalog_path)
        if local is None:
            raise AERDataError(
                f"catalog_path does not contain the exact AER {AER_CATALOG_VERSION} payload: "
                f"{Path(catalog_path).expanduser()}"
            )
        _record_adopted_artifact(local, cache_dir)
        return local
    if offline:
        raise AERDataError(
            "AER catalogue is not installed and offline=True prevents downloading it; "
            "run `pymolfit install-aer` while online"
        )

    configured_source = source
    if configured_source is None:
        configured_source = os.environ.get(AER_SOURCE_URL_ENV) or AER_SOURCE_URL
    source_text = os.fspath(configured_source)
    expected_source_hash = source_sha256 or _known_source_sha256(source_text)
    cache_root = _cache_root(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    lock_path = cache_root / f".{AER_CATALOG_VERSION}.install.lock"
    with _CacheLock(lock_path, timeout_s=max(float(timeout_s), 30.0)):
        if not force:
            cached = _load_managed_artifact(target_dir)
            if cached is not None:
                return cached

        _report(progress, f"Acquiring AER {AER_CATALOG_VERSION} from {source_text}")
        with tempfile.TemporaryDirectory(prefix="aer-install-", dir=cache_root) as tmp_name:
            temp_root = Path(tmp_name)
            archive_path = temp_root / Path(source_text).name
            if _is_url(source_text):
                if not archive_path.name or archive_path.name == ".":
                    archive_path = temp_root / "aer-source.tar"
                _download_archive(
                    source_text,
                    archive_path,
                    timeout_s=timeout_s,
                    opener=opener,
                    progress=progress,
                )
            else:
                local_source = Path(source_text).expanduser().resolve()
                if not local_source.is_file():
                    raise FileNotFoundError(local_source)
                archive_path = local_source

            archive_sha256 = file_sha256(archive_path)
            if expected_source_hash is not None and archive_sha256 != expected_source_hash:
                raise AERDataError(
                    "AER source archive checksum mismatch: "
                    f"expected {expected_source_hash}, got {archive_sha256}"
                )

            staging = temp_root / "payload"
            staging.mkdir()
            _extract_aer_payload(archive_path, staging, temp_root=temp_root)
            file_records = _verify_payload(staging)
            manifest = _catalog_manifest(
                source=source_text,
                source_archive_sha256=archive_sha256,
                files=file_records,
            )
            manifest_path = staging / "manifest.json"
            _write_json_atomic(manifest_path, manifest)

            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(target_dir)

    installed = _load_managed_artifact(target_dir)
    if installed is None:  # pragma: no cover - defensive postcondition
        raise AERDataError("AER payload failed verification after installation")
    _report(progress, f"Installed AER catalogue at {installed.catalog_path}")
    return replace(installed, cache_hit=False)


def load_aer_line_window(
    *,
    wavelength_min_micron: float | None = None,
    wavelength_max_micron: float | None = None,
    wavenumber_min_cm: float | None = None,
    wavenumber_max_cm: float | None = None,
    species: Iterable[str] | None = None,
    min_strength: float | None = None,
    max_lines: int | None = None,
    catalog: AERCatalogArtifact | str | Path | None = None,
    cache_dir: str | Path | None = None,
    source: str | Path | None = None,
    offline: bool = False,
    reuse_molecfit: bool = True,
    timeout_s: float = 120.0,
    opener: _OpenUrl = urlopen,
) -> AERLineArtifact:
    """Select and cache one wavelength/species window from AER 3.9."""

    request = _canonical_window_request(
        species,
        wavelength_min_micron=wavelength_min_micron,
        wavelength_max_micron=wavelength_max_micron,
        wavenumber_min_cm=wavenumber_min_cm,
        wavenumber_max_cm=wavenumber_max_cm,
        min_strength=min_strength,
        max_lines=max_lines,
    )
    resolved_catalog = _resolve_catalog_argument(
        catalog,
        cache_dir=cache_dir,
        source=source,
        offline=offline,
        reuse_molecfit=reuse_molecfit,
        timeout_s=timeout_s,
        opener=opener,
    )
    paths = _window_paths(_cache_root(cache_dir), request)
    cached = _load_cached_window(paths, request, resolved_catalog)
    if cached is not None:
        return cached

    lock_path = paths[0].with_suffix(".lock")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with _CacheLock(lock_path, timeout_s=max(float(timeout_s), 30.0)):
        cached = _load_cached_window(paths, request, resolved_catalog)
        if cached is not None:
            return cached
        line_list = LineList.from_aer_line_file(
            resolved_catalog.catalog_path,
            wavenumber_min=request["wavenumber_min_cm"],
            wavenumber_max=request["wavenumber_max_cm"],
            species=request["species"],
            min_strength=request["min_strength"],
            max_lines=request["max_lines"],
            extra_broadener_dir=resolved_catalog.extra_broadener_dir,
            # AER 3.9 is grouped by molecule and is not globally monotonic in
            # wavenumber, so an early-stop scan would silently omit species.
            assume_sorted=False,
        )
        catalog_source_page = str(
            resolved_catalog.manifest.get("source_page", resolved_catalog.source)
        )
        source_archive_sha256 = str(
            resolved_catalog.manifest.get("source_archive_sha256", "")
        )
        provenance = {
            "source": "aer_catalog",
            "catalog_version": AER_CATALOG_VERSION,
            "catalog_sha256": AER_CATALOG_SHA256,
            "catalog_source_page": catalog_source_page,
            "source_archive_sha256": source_archive_sha256,
            "request": request,
        }
        line_list = replace(line_list, data_provenance=provenance)
        table_path, manifest_path = paths
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=table_path.parent,
            prefix=table_path.name,
            suffix=".tmp.ecsv",
            delete=False,
        ) as handle:
            temp_table = Path(handle.name)
        try:
            line_list.write(temp_table)
            table_sha256 = file_sha256(temp_table)
            manifest = {
                "schema_version": AER_WINDOW_SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "catalog_version": AER_CATALOG_VERSION,
                "catalog_sha256": AER_CATALOG_SHA256,
                "catalog_source_page": catalog_source_page,
                "source_archive_sha256": source_archive_sha256,
                "request": request,
                "line_count": int(line_list.wavelength.size),
                "species_with_lines": list(line_list.species_names),
                "table_sha256": table_sha256,
            }
            temp_table.replace(table_path)
            _write_json_atomic(manifest_path, manifest)
        finally:
            if temp_table.exists():
                temp_table.unlink()
    return AERLineArtifact(
        table_path=table_path,
        manifest_path=manifest_path,
        manifest=manifest,
        line_list=line_list,
        catalog=resolved_catalog,
        cache_hit=False,
    )


def _cache_root(cache_dir: str | Path | None) -> Path:
    target = default_aer_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    return target.resolve()


def _version_dir(cache_dir: str | Path | None) -> Path:
    return _cache_root(cache_dir) / AER_CATALOG_VERSION


def _local_catalog_candidates(catalog_path: str | Path | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if catalog_path is not None:
        candidates.append(Path(catalog_path).expanduser())
        candidate = candidates[0]
        if candidate.is_dir():
            candidate = candidate / AER_CATALOG_FILENAME
        return (candidate,)
    env_catalog = os.environ.get(AER_CATALOG_PATH_ENV)
    if env_catalog:
        candidates.append(Path(env_catalog).expanduser())
    roots = []
    env_root = os.environ.get(AER_MOLECFIT_ROOT_ENV)
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        (
            Path.home() / ".criresflow" / "molecfit",
            Path.home() / ".local",
            Path("/usr/local"),
            Path("/opt/molecfit"),
        )
    )
    for root in roots:
        candidates.extend(
            (
                root / "share" / "molecfit" / "data" / "hitran" / AER_CATALOG_FILENAME,
                root / "data" / "hitran" / AER_CATALOG_FILENAME,
            )
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / AER_CATALOG_FILENAME
        key = os.fspath(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _external_artifact(catalog_path: Path) -> AERCatalogArtifact | None:
    try:
        catalog_path = catalog_path.resolve()
    except OSError:
        return None
    if not catalog_path.is_file():
        return None
    directory = catalog_path.parent
    for filename, (_, expected_hash) in AER_FILE_SPECS.items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected_hash:
            return None
    manifest = {
        "schema_version": AER_DATA_SCHEMA_VERSION,
        "catalog_version": AER_CATALOG_VERSION,
        "catalog_sha256": AER_CATALOG_SHA256,
        "source": "existing_aer_installation",
        "path": os.fspath(catalog_path),
    }
    return AERCatalogArtifact(
        catalog_path=catalog_path,
        extra_broadener_dir=directory,
        manifest_path=None,
        manifest=manifest,
        source="existing_aer_installation",
        cache_hit=True,
        managed=False,
    )


def _adopted_manifest_path(cache_dir: str | Path | None) -> Path:
    return _cache_root(cache_dir) / f"adopted-{AER_CATALOG_VERSION}.json"


def _load_adopted_artifact(cache_dir: str | Path | None) -> AERCatalogArtifact | None:
    manifest_path = _adopted_manifest_path(cache_dir)
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        catalog_path = Path(manifest["catalog_path"])
        records = manifest["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != AER_DATA_SCHEMA_VERSION
        or manifest.get("catalog_version") != AER_CATALOG_VERSION
        or manifest.get("catalog_sha256") != AER_CATALOG_SHA256
        or not isinstance(records, Mapping)
    ):
        return None
    directory = catalog_path.parent
    for filename, (_, expected_hash) in AER_FILE_SPECS.items():
        path = directory / filename
        record = records.get(filename)
        if not path.is_file() or not isinstance(record, Mapping):
            return None
        stat = path.stat()
        if (
            record.get("sha256") != expected_hash
            or record.get("size_bytes") != stat.st_size
            or record.get("mtime_ns") != stat.st_mtime_ns
        ):
            return None
    return AERCatalogArtifact(
        catalog_path=catalog_path,
        extra_broadener_dir=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        source=str(manifest.get("source", "existing_aer_installation")),
        cache_hit=True,
        managed=False,
    )


def _record_adopted_artifact(
    artifact: AERCatalogArtifact,
    cache_dir: str | Path | None,
) -> None:
    directory = artifact.extra_broadener_dir
    records = {}
    for filename, (_, expected_hash) in AER_FILE_SPECS.items():
        stat = (directory / filename).stat()
        records[filename] = {
            "sha256": expected_hash,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    manifest = {
        "schema_version": AER_DATA_SCHEMA_VERSION,
        "catalog_version": AER_CATALOG_VERSION,
        "catalog_sha256": AER_CATALOG_SHA256,
        "catalog_path": os.fspath(artifact.catalog_path),
        "source": artifact.source,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    try:
        _write_json_atomic(_adopted_manifest_path(cache_dir), manifest)
    except OSError:
        LOGGER.debug("Could not persist adopted AER catalogue manifest", exc_info=True)


def _load_managed_artifact(directory: Path) -> AERCatalogArtifact | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != AER_DATA_SCHEMA_VERSION
        or manifest.get("catalog_version") != AER_CATALOG_VERSION
        or manifest.get("catalog_sha256") != AER_CATALOG_SHA256
    ):
        return None
    expected_records = manifest.get("files")
    if not isinstance(expected_records, Mapping):
        return None
    for filename, (_, expected_hash) in AER_FILE_SPECS.items():
        path = directory / filename
        record = expected_records.get(filename)
        if not path.is_file() or not isinstance(record, Mapping):
            return None
        stat = path.stat()
        unchanged = (
            record.get("size_bytes") == stat.st_size
            and record.get("mtime_ns") == stat.st_mtime_ns
        )
        if record.get("sha256") != expected_hash or (not unchanged and file_sha256(path) != expected_hash):
            return None
    return AERCatalogArtifact(
        catalog_path=directory / AER_CATALOG_FILENAME,
        extra_broadener_dir=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        source=str(manifest.get("source", "managed_cache")),
        cache_hit=True,
        managed=True,
    )


def _known_source_sha256(source: str) -> str | None:
    if source == AER_SOURCE_URL:
        return AER_ARCHIVE_SHA256
    source_name = Path(urlparse(source).path).name
    if source_name == AER_ARCHIVE_FILENAME and source.startswith(("http://", "https://", "ftp://")):
        return AER_ARCHIVE_SHA256
    return None


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "ftp://", "file://"))


def _download_archive(
    url: str,
    destination: Path,
    *,
    timeout_s: float,
    opener: _OpenUrl,
    progress: Callable[[str], None] | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "PyMolFit-AER/0.1"})
    try:
        with opener(request, timeout=timeout_s) as response, destination.open("wb") as handle:
            downloaded = 0
            next_report = 64 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    _report(progress, f"Downloaded {downloaded / (1024**2):.0f} MiB")
                    next_report += 64 * 1024 * 1024
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AERDataError(f"could not download AER data from {url}: {exc}") from exc
    if not destination.is_file() or destination.stat().st_size == 0:
        raise AERDataError(f"AER download from {url} was empty")


def _extract_aer_payload(archive_path: Path, destination: Path, *, temp_root: Path, depth: int = 0) -> None:
    if depth > 3:
        raise AERDataError("AER archive nesting exceeded the supported layout")
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            names = archive.getnames()
            if _find_member(names, AER_FILE_SPECS[AER_CATALOG_FILENAME][0]) is not None:
                _extract_direct_payload(archive, names, destination)
                return
            nested_aer = _find_member(names, AER_ARCHIVE_FILENAME)
            nested_name = nested_aer
            if nested_name is None:
                raise AERDataError(
                    f"archive does not contain the official AER {AER_CATALOG_VERSION} payload"
                )
            suffix = ".tgz" if nested_name.endswith((".tgz", ".tar.gz")) else ".tar"
            nested_path = temp_root / f"nested-{depth}{suffix}"
            _copy_tar_member(archive, nested_name, nested_path)
    except (tarfile.TarError, OSError) as exc:
        raise AERDataError(f"could not read AER source archive {archive_path}") from exc
    _extract_aer_payload(nested_path, destination, temp_root=temp_root, depth=depth + 1)


def _extract_direct_payload(archive: tarfile.TarFile, names: list[str], destination: Path) -> None:
    for filename, (member_suffix, _) in AER_FILE_SPECS.items():
        member_name = _find_member(names, member_suffix)
        if member_name is None:
            raise AERDataError(f"AER archive is missing required member {member_suffix}")
        _copy_tar_member(archive, member_name, destination / filename)
    for filename, member_suffix in _OPTIONAL_ARCHIVE_MEMBERS.items():
        member_name = _find_member(names, member_suffix)
        if member_name is not None:
            _copy_tar_member(archive, member_name, destination / filename)
    license_path = destination / "AER_LICENSE.txt"
    if not license_path.exists():
        license_path.write_text(AER_LICENSE_TEXT + "\n", encoding="utf-8")


def _find_member(names: Iterable[str], suffix: str) -> str | None:
    normalized_suffix = suffix.lstrip("/")
    matches = [name for name in names if name.lstrip("/").endswith(normalized_suffix)]
    if len(matches) > 1:
        raise AERDataError(f"archive contains multiple members matching {suffix}")
    return matches[0] if matches else None


def _copy_tar_member(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    member = archive.getmember(member_name)
    if not member.isfile():
        raise AERDataError(f"archive member is not a regular file: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise AERDataError(f"could not extract archive member: {member_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as handle:
        shutil.copyfileobj(source, handle, length=1024 * 1024)


def _verify_payload(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for filename, (_, expected_hash) in AER_FILE_SPECS.items():
        path = directory / filename
        if not path.is_file():
            raise AERDataError(f"AER payload is missing {filename}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise AERDataError(
                f"AER payload checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        stat = path.stat()
        records[filename] = {
            "sha256": actual_hash,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return records


def _catalog_manifest(
    *,
    source: str,
    source_archive_sha256: str,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": AER_DATA_SCHEMA_VERSION,
        "catalog_version": AER_CATALOG_VERSION,
        "catalog_sha256": AER_CATALOG_SHA256,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_archive_sha256": source_archive_sha256,
        "source_page": AER_SOURCE_PAGE,
        "zenodo_record": AER_ZENODO_RECORD,
        "license_url": AER_LICENSE_URL,
        "license_scope": "scientific and research use; redistribution requires the AER notice",
        "files": dict(files),
    }


def _canonical_window_request(
    species: Iterable[str] | None,
    *,
    wavelength_min_micron: float | None,
    wavelength_max_micron: float | None,
    wavenumber_min_cm: float | None,
    wavenumber_max_cm: float | None,
    min_strength: float | None,
    max_lines: int | None,
) -> dict[str, Any]:
    names = tuple(
        sorted(
            {
                str(value).strip().upper()
                for value in (_DEFAULT_SPECIES if species is None else species)
                if str(value).strip()
            }
        )
    )
    known_species = {name for name, _ in HITRAN_MOLECULES.values()}
    unsupported = tuple(name for name in names if name not in known_species)
    if not names or unsupported:
        raise ValueError(f"unsupported or empty AER species selection: {unsupported or names}")

    use_wavelength = wavelength_min_micron is not None or wavelength_max_micron is not None
    use_wavenumber = wavenumber_min_cm is not None or wavenumber_max_cm is not None
    if use_wavelength == use_wavenumber:
        raise ValueError("provide exactly one complete wavelength or wavenumber interval")
    if use_wavelength:
        lower_wave, upper_wave = _ordered_interval(
            wavelength_min_micron, wavelength_max_micron, "wavelength"
        )
        lower_nu, upper_nu = 1.0e4 / upper_wave, 1.0e4 / lower_wave
    else:
        lower_nu, upper_nu = _ordered_interval(
            wavenumber_min_cm, wavenumber_max_cm, "wavenumber"
        )
        lower_wave, upper_wave = 1.0e4 / upper_nu, 1.0e4 / lower_nu
    if min_strength is not None and (not _finite_positive(min_strength)):
        raise ValueError("min_strength must be finite and positive")
    if max_lines is not None and int(max_lines) < 1:
        raise ValueError("max_lines must be positive")
    return {
        "species": list(names),
        "wavelength_min_micron": float(lower_wave),
        "wavelength_max_micron": float(upper_wave),
        "wavenumber_min_cm": float(lower_nu),
        "wavenumber_max_cm": float(upper_nu),
        "min_strength": None if min_strength is None else float(min_strength),
        "max_lines": None if max_lines is None else int(max_lines),
    }


def _ordered_interval(lower: float | None, upper: float | None, label: str) -> tuple[float, float]:
    if lower is None or upper is None:
        raise ValueError(f"both {label} bounds are required")
    values = (float(lower), float(upper))
    if not all(_finite_positive(value) for value in values) or values[0] == values[1]:
        raise ValueError(f"{label} bounds must be finite, positive, and distinct")
    return min(values), max(values)


def _finite_positive(value: float) -> bool:
    return float("-inf") < float(value) < float("inf") and float(value) > 0


def _resolve_catalog_argument(
    catalog: AERCatalogArtifact | str | Path | None,
    *,
    cache_dir: str | Path | None,
    source: str | Path | None,
    offline: bool,
    reuse_molecfit: bool,
    timeout_s: float,
    opener: _OpenUrl,
) -> AERCatalogArtifact:
    if isinstance(catalog, AERCatalogArtifact):
        return catalog
    explicit_path = None if catalog is None or os.fspath(catalog) == "auto" else catalog
    return install_aer_catalog(
        source=source,
        cache_dir=cache_dir,
        catalog_path=explicit_path,
        offline=offline,
        reuse_molecfit=reuse_molecfit,
        timeout_s=timeout_s,
        opener=opener,
    )


def _window_paths(cache_root: Path, request: Mapping[str, Any]) -> tuple[Path, Path]:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    directory = cache_root / AER_CATALOG_VERSION / "windows"
    stem = directory / f"window-{digest}"
    return stem.with_suffix(".ecsv"), stem.with_suffix(".manifest.json")


def _load_cached_window(
    paths: tuple[Path, Path],
    request: Mapping[str, Any],
    catalog: AERCatalogArtifact,
) -> AERLineArtifact | None:
    table_path, manifest_path = paths
    if not table_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != AER_WINDOW_SCHEMA_VERSION
        or manifest.get("catalog_version") != AER_CATALOG_VERSION
        or manifest.get("catalog_sha256") != AER_CATALOG_SHA256
        or manifest.get("request") != dict(request)
        or manifest.get("table_sha256") != file_sha256(table_path)
    ):
        return None
    try:
        line_list = LineList.from_table(table_path)
    except Exception:
        return None
    return AERLineArtifact(
        table_path=table_path,
        manifest_path=manifest_path,
        manifest=manifest,
        line_list=line_list,
        catalog=catalog,
        cache_hit=True,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    LOGGER.info(message)
    if progress is not None:
        progress(message)


class _CacheLock:
    def __init__(self, path: Path, *, timeout_s: float, stale_s: float = 6 * 3600.0):
        self.path = path
        self.timeout_s = timeout_s
        self.stale_s = stale_s
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                try:
                    stale = time.time() - self.path.stat().st_mtime > self.stale_s
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise AERDataError(f"timed out waiting for AER cache lock {self.path}")
                time.sleep(0.1)
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(f"pid={os.getpid()}\n")
            self.acquired = True
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return False
