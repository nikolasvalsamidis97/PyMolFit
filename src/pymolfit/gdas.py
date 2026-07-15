from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from astropy.table import Table
from astropy.time import Time

GDAS_BASE_URLS = (
    "https://ftp.eso.org/pub/dfs/pipelines/skytools/molecfit/gdas",
    "ftp://ftp.eso.org/pub/dfs/pipelines/skytools/molecfit/gdas",
)
GDAS_INTERVAL_HOURS = 3
GDAS_SEARCH_HOURS = 6


class GDASProfileUnavailable(RuntimeError):
    """Raised when a strict GDAS mode cannot find a time-local profile."""


@dataclass(frozen=True)
class GDASProfileResolution:
    path: Path
    source: str
    before_member: str
    after_member: str


def resolve_time_local_gdas_profile(
    *,
    observation_time: Time | datetime | str | float | None,
    latitude_deg: float | None,
    longitude_deg: float | None,
    mode: str = "auto",
    cache_dir: str | Path | None = None,
    timeout_s: float = 15.0,
    base_urls: Iterable[str] = GDAS_BASE_URLS,
) -> GDASProfileResolution | None:
    """Return a cached FITS GDAS profile interpolated to the observation time.

    ``auto`` downloads/cache-resolves exact GDAS and returns ``None`` on failure
    so callers can use a monthly average fallback. ``online`` and ``cache`` are
    strict modes: they raise if a time-local exact profile cannot be resolved.
    """

    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "average":
        return None

    time = _coerce_time(observation_time)
    if time is None or latitude_deg is None or longitude_deg is None:
        if normalized_mode in {"online", "cache"}:
            raise GDASProfileUnavailable("exact GDAS requires observation time, latitude, and longitude")
        return None

    cache_root = _cache_root(cache_dir)
    site_id = _site_id(longitude_deg, latitude_deg)
    profile_path = _interpolated_profile_path(cache_root, site_id, time)
    if profile_path.exists():
        try:
            cached = Table.read(profile_path, hdu=1)
            return GDASProfileResolution(
                path=profile_path,
                source="cache",
                before_member=str(cached.meta.get("GDASBFR", "")),
                after_member=str(cached.meta.get("GDASAFT", "")),
            )
        except Exception:
            # A partial/interrupted cache file is reproducible from the source
            # archive and must not silently become an average-atmosphere run.
            profile_path.unlink()

    tarball_path = _tarball_path(cache_root, site_id)
    downloaded = False
    if not tarball_path.exists():
        if normalized_mode == "cache":
            raise GDASProfileUnavailable(f"GDAS tarball not found in cache: {tarball_path}")
        try:
            _download_tarball(tarball_path, base_urls=base_urls, timeout_s=timeout_s)
            downloaded = True
        except Exception as exc:
            if normalized_mode == "online":
                raise GDASProfileUnavailable(f"could not download GDAS tarball {tarball_path.name}") from exc
            return None

    try:
        before_member, after_member, before_time, after_time = _find_bracketing_members(
            tarball_path,
            site_id=site_id,
            observation_time=time,
        )
        table_before = _read_gdas_member(tarball_path, before_member)
        table_after = _read_gdas_member(tarball_path, after_member)
        interpolated = _interpolate_gdas_tables(
            table_before,
            table_after,
            observation_time=time,
            before_time=before_time,
            after_time=after_time,
        )
        interpolated.meta["GDASSRC"] = "ESO_TIME_LOCAL"
        interpolated.meta["GDASBFR"] = before_member
        interpolated.meta["GDASAFT"] = after_member
        interpolated.meta["GDASSITE"] = site_id
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        interpolated.write(profile_path, overwrite=True)
        return GDASProfileResolution(
            path=profile_path,
            source="download" if downloaded else "cache",
            before_member=before_member,
            after_member=after_member,
        )
    except Exception as exc:
        if normalized_mode in {"online", "cache"}:
            raise GDASProfileUnavailable("could not resolve exact GDAS profiles from cached tarball") from exc
        return None


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "online": "online",
        "download": "online",
        "cache": "cache",
        "cached": "cache",
        "average": "average",
        "monthly": "average",
        "offline": "average",
        "none": "average",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("gdas_mode must be 'auto', 'online', 'cache', or 'average'") from exc


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env_cache = os.environ.get("PYMOLFIT_GDAS_CACHE")
    if env_cache:
        return Path(env_cache).expanduser()
    return Path.home() / ".cache" / "pymolfit" / "gdas"


def _site_id(longitude_deg: float, latitude_deg: float) -> str:
    return f"C{float(longitude_deg):+.1f}{float(latitude_deg):+.1f}"


def _tarball_path(cache_root: Path, site_id: str) -> Path:
    return cache_root / "tarballs" / f"gdas_profiles_{site_id}.tar.gz"


def _interpolated_profile_path(cache_root: Path, site_id: str, observation_time: Time) -> Path:
    dt = observation_time.utc.datetime
    stamp = dt.strftime("%Y-%m-%dT%H%M%S")
    return cache_root / "profiles" / site_id / f"{site_id}D{stamp}.fits"


def _download_tarball(
    path: Path,
    *,
    base_urls: Iterable[str],
    timeout_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[Exception] = []
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        for base_url in base_urls:
            url = f"{str(base_url).rstrip('/')}/{path.name}"
            try:
                with urlopen(url, timeout=timeout_s) as response, tmp_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                if tmp_path.stat().st_size <= 0:
                    raise URLError("empty GDAS tarball download")
                tmp_path.replace(path)
                return
            except Exception as exc:
                errors.append(exc)
        raise GDASProfileUnavailable(f"all GDAS download URLs failed for {path.name}: {errors[-1]}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _find_bracketing_members(
    tarball_path: Path,
    *,
    site_id: str,
    observation_time: Time,
) -> tuple[str, str, Time, Time]:
    dt = observation_time.utc.datetime
    floor_hour = (dt.hour // GDAS_INTERVAL_HOURS) * GDAS_INTERVAL_HOURS
    before_guess = dt.replace(hour=floor_hour, minute=0, second=0, microsecond=0)
    after_guess = before_guess + timedelta(hours=GDAS_INTERVAL_HOURS)
    with tarfile.open(tarball_path, "r:gz") as tar:
        names = set(tar.getnames())
    before_time = _find_existing_time(site_id, names, before_guess, direction=-1)
    after_time = _find_existing_time(site_id, names, after_guess, direction=1)
    if before_time is None or after_time is None:
        raise GDASProfileUnavailable(f"no GDAS profiles bracketing {observation_time.isot}")
    before_member = _member_name(site_id, before_time)
    after_member = _member_name(site_id, after_time)
    return before_member, after_member, Time(before_time, scale="utc"), Time(after_time, scale="utc")


def _find_existing_time(
    site_id: str,
    members: set[str],
    start: datetime,
    *,
    direction: int,
) -> datetime | None:
    for offset in range(GDAS_SEARCH_HOURS + 1):
        candidate = start + timedelta(hours=direction * offset)
        if _member_name(site_id, candidate) in members:
            return candidate
    return None


def _member_name(site_id: str, when: datetime) -> str:
    return f"{site_id}D{when:%Y-%m-%d}T{when.hour:02d}.gdas"


def _read_gdas_member(tarball_path: Path, member_name: str) -> Table:
    with tarfile.open(tarball_path, "r:gz") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise GDASProfileUnavailable(f"GDAS member {member_name} not found")
        data = np.loadtxt(extracted, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] < 4:
        raise GDASProfileUnavailable(f"GDAS member {member_name} has invalid columns")
    table = Table()
    table["press"] = np.asarray(data[:, 0], dtype=float)
    table["height"] = np.asarray(data[:, 1], dtype=float) / 1000.0
    table["temp"] = np.asarray(data[:, 2], dtype=float)
    table["relhum"] = np.asarray(data[:, 3], dtype=float)
    return table


def _interpolate_gdas_tables(
    before: Table,
    after: Table,
    *,
    observation_time: Time,
    before_time: Time,
    after_time: Time,
) -> Table:
    if len(before) != len(after):
        before_delta = abs((observation_time - before_time).to_value("hour"))
        after_delta = abs((after_time - observation_time).to_value("hour"))
        return before.copy() if before_delta <= after_delta else after.copy()
    interval_h = (after_time - before_time).to_value("hour")
    weight = 0.0 if interval_h == 0 else (observation_time - before_time).to_value("hour") / interval_h
    weight = float(np.clip(weight, 0.0, 1.0))
    table = Table()
    for colname in before.colnames:
        before_values = np.asarray(before[colname], dtype=float)
        after_values = np.asarray(after[colname], dtype=float)
        table[colname] = before_values + weight * (after_values - before_values)
    return table


def _coerce_time(value: Time | datetime | str | float | None) -> Time | None:
    if value is None:
        return None
    if isinstance(value, Time):
        return value
    if isinstance(value, datetime):
        return Time(value, scale="utc")
    if isinstance(value, str):
        try:
            return Time(value, scale="utc")
        except Exception:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 10_000.0:
        return Time(numeric, format="mjd", scale="utc")
    return None
