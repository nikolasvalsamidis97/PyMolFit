import io
import tarfile

import numpy as np
import pytest
from astropy.table import Table

from genmolfit.gdas import GDASProfileUnavailable, resolve_time_local_gdas_profile


def _add_gdas_member(tar, name, *, temp, relhum):
    text = "\n".join(
        [
            "# P[hPa] HGT[m] T[K] RELHUM[%]",
            f"1000 100.0 {temp:.1f} {relhum:.1f}",
            f"900 1000.0 {temp - 10:.1f} {relhum + 5:.1f}",
        ]
    )
    payload = text.encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def test_resolve_time_local_gdas_profile_from_cached_tarball(tmp_path):
    site = "C-70.4-24.6"
    tarball = tmp_path / "tarballs" / f"gdas_profiles_{site}.tar.gz"
    tarball.parent.mkdir(parents=True)
    with tarfile.open(tarball, "w:gz") as tar:
        _add_gdas_member(tar, f"{site}D2022-01-02T00.gdas", temp=280.0, relhum=10.0)
        _add_gdas_member(tar, f"{site}D2022-01-02T03.gdas", temp=290.0, relhum=20.0)

    resolution = resolve_time_local_gdas_profile(
        observation_time="2022-01-02T01:30:00",
        longitude_deg=-70.4,
        latitude_deg=-24.6,
        mode="cache",
        cache_dir=tmp_path,
    )

    assert resolution is not None
    assert resolution.before_member.endswith("T00.gdas")
    assert resolution.after_member.endswith("T03.gdas")
    table = Table.read(resolution.path, hdu=1)
    np.testing.assert_allclose(table["temp"], [285.0, 275.0])
    np.testing.assert_allclose(table["relhum"], [15.0, 20.0])
    np.testing.assert_allclose(table["height"], [0.1, 1.0])


def test_strict_cache_mode_raises_when_exact_gdas_is_missing(tmp_path):
    with pytest.raises(GDASProfileUnavailable):
        resolve_time_local_gdas_profile(
            observation_time="2022-01-02T01:30:00",
            longitude_deg=-70.4,
            latitude_deg=-24.6,
            mode="cache",
            cache_dir=tmp_path,
        )


def test_auto_mode_returns_none_when_download_fails(tmp_path):
    resolution = resolve_time_local_gdas_profile(
        observation_time="2022-01-02T01:30:00",
        longitude_deg=-70.4,
        latitude_deg=-24.6,
        mode="auto",
        cache_dir=tmp_path,
        base_urls=("file:///definitely/not/a/real/gdas/archive",),
    )

    assert resolution is None
