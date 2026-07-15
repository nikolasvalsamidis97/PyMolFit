from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from genmolfit import Spectrum
from genmolfit import aer_data
from genmolfit.aer_data import AERDataError, install_aer_catalog, load_aer_line_window
from genmolfit.workflow import _resolve_line_list


def _fixed_decimal(value, width, decimals):
    text = f"{value:.{decimals}f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return f"{text:>{width}}"[-width:]


def _aer_row(*, mol_id=1, iso_id=1, wavenumber=4320.0, strength=1.0e-24):
    row = (
        f"{10 * mol_id + iso_id:3d}"
        f"{wavenumber:12.6f}"
        f"{strength:10.3E}"
        f"{1.0:10.3E}"
        f"{_fixed_decimal(0.07, 5, 4)}"
        f"{_fixed_decimal(0.30, 5, 4)}"
        f"{100.0:10.4f}"
        f"{0.75:4.2f}"
        f"{_fixed_decimal(-0.001, 8, 6)}"
    )
    return row + " " * (160 - len(row))


def _payload_text():
    return "\n".join(
        (
            "> tiny AER test catalogue",
            _aer_row(wavenumber=4319.5),
            _aer_row(wavenumber=4320.5),
            _aer_row(mol_id=7, wavenumber=4320.0),
            _aer_row(wavenumber=4400.0),
            "",
        )
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_tar(path: Path, members: dict[str, bytes], *, gzip: bool = True):
    mode = "w:gz" if gzip else "w"
    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


@pytest.fixture
def tiny_aer(monkeypatch, tmp_path):
    line_payload = _payload_text().encode("ascii")
    extra_payload = b"# tiny extra broadener table\n"
    monkeypatch.setattr(aer_data, "AER_CATALOG_SHA256", _sha256_bytes(line_payload))
    monkeypatch.setattr(
        aer_data,
        "AER_FILE_SPECS",
        {
            aer_data.AER_CATALOG_FILENAME: (
                f"aer_v_{aer_data.AER_CATALOG_VERSION}/line_file/{aer_data.AER_CATALOG_FILENAME}",
                _sha256_bytes(line_payload),
            ),
            "o2_h2o_brd_param": (
                f"aer_v_{aer_data.AER_CATALOG_VERSION}/extra_brd_params/o2_h2o_brd_param",
                _sha256_bytes(extra_payload),
            ),
        },
    )
    archive_path = tmp_path / "tiny-aer.tar.gz"
    _write_tar(
        archive_path,
        {
            f"aer_v_{aer_data.AER_CATALOG_VERSION}/line_file/{aer_data.AER_CATALOG_FILENAME}": line_payload,
            f"aer_v_{aer_data.AER_CATALOG_VERSION}/extra_brd_params/o2_h2o_brd_param": extra_payload,
            f"aer_v_{aer_data.AER_CATALOG_VERSION}/LICENSE": b"test AER license\n",
            f"aer_v_{aer_data.AER_CATALOG_VERSION}/RELEASE_NOTES_aer_linefile": b"test release\n",
        },
    )
    return archive_path, line_payload, extra_payload


def test_install_direct_aer_archive_is_verified_and_reused(tiny_aer, tmp_path):
    archive_path, line_payload, _ = tiny_aer
    cache = tmp_path / "cache"

    first = install_aer_catalog(
        source=archive_path,
        cache_dir=cache,
        reuse_molecfit=False,
    )
    second = install_aer_catalog(
        cache_dir=cache,
        offline=True,
        reuse_molecfit=False,
    )

    assert first.managed and not first.cache_hit
    assert second.managed and second.cache_hit
    assert first.catalog_path.read_bytes() == line_payload
    assert (first.extra_broadener_dir / "o2_h2o_brd_param").is_file()
    assert (first.extra_broadener_dir / "AER_LICENSE.txt").is_file()
    assert first.manifest["catalog_version"] == aer_data.AER_CATALOG_VERSION
    assert first.manifest["source_archive_sha256"]


def test_install_supports_nested_aer_archive_layout(tiny_aer, tmp_path):
    direct_archive, _, _ = tiny_aer
    direct_bytes = direct_archive.read_bytes()
    middle = tmp_path / "third-party.tar"
    _write_tar(
        middle,
        {f"data/{aer_data.AER_ARCHIVE_FILENAME}": direct_bytes},
        gzip=False,
    )
    outer = tmp_path / "aer-wrapper.tar.gz"
    _write_tar(
        outer,
        {aer_data.AER_ARCHIVE_FILENAME: direct_bytes},
    )

    artifact = install_aer_catalog(
        source=outer,
        cache_dir=tmp_path / "cache",
        reuse_molecfit=False,
    )

    assert artifact.catalog_path.is_file()
    assert artifact.manifest["source"] == str(outer)


def test_default_source_is_pinned_official_zenodo_release():
    assert aer_data.AER_CATALOG_VERSION == "3.9"
    assert aer_data.AER_ZENODO_RECORD in aer_data.AER_SOURCE_URL
    assert aer_data._known_source_sha256(aer_data.AER_SOURCE_URL) == aer_data.AER_ARCHIVE_SHA256


def test_downloaded_archive_is_atomic_and_works_offline(tiny_aer, tmp_path):
    archive_path, _, _ = tiny_aer
    calls = []

    class Response:
        def __init__(self, payload):
            self.stream = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return self.stream.read(size)

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(archive_path.read_bytes())

    cache = tmp_path / "cache"
    artifact = install_aer_catalog(
        source="https://example.test/catalogue.bundle",
        cache_dir=cache,
        reuse_molecfit=False,
        opener=opener,
    )
    cached = install_aer_catalog(
        cache_dir=cache,
        offline=True,
        reuse_molecfit=False,
        opener=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    assert len(calls) == 1
    assert artifact.catalog_path == cached.catalog_path
    assert not list(cache.glob("*.lock"))


def test_payload_checksum_mismatch_is_rejected(tiny_aer, tmp_path, monkeypatch):
    archive_path, _, _ = tiny_aer
    specs = dict(aer_data.AER_FILE_SPECS)
    suffix, _ = specs[aer_data.AER_CATALOG_FILENAME]
    specs[aer_data.AER_CATALOG_FILENAME] = (suffix, "0" * 64)
    monkeypatch.setattr(aer_data, "AER_FILE_SPECS", specs)

    with pytest.raises(AERDataError, match="checksum mismatch"):
        install_aer_catalog(
            source=archive_path,
            cache_dir=tmp_path / "cache",
            reuse_molecfit=False,
        )

    assert not (tmp_path / "cache" / aer_data.AER_CATALOG_VERSION / "manifest.json").exists()


def test_aer_line_windows_are_filtered_provenanced_and_cached(tiny_aer, tmp_path):
    archive_path, _, _ = tiny_aer
    cache = tmp_path / "cache"
    catalog = install_aer_catalog(
        source=archive_path,
        cache_dir=cache,
        reuse_molecfit=False,
    )

    first = load_aer_line_window(
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        species=("H2O",),
        catalog=catalog,
        cache_dir=cache,
    )
    second = load_aer_line_window(
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        species=("H2O",),
        catalog=catalog,
        cache_dir=cache,
    )

    np.testing.assert_allclose(first.line_list.wavenumber, [4319.5, 4320.5])
    assert first.line_list.species_names == ("H2O",)
    assert first.line_list.data_provenance["catalog_version"] == aer_data.AER_CATALOG_VERSION
    assert first.line_list.data_provenance["catalog_source_page"] == aer_data.AER_SOURCE_PAGE
    assert first.line_list.data_provenance["source_archive_sha256"] == _sha256_bytes(
        archive_path.read_bytes()
    )
    assert not first.cache_hit
    assert second.cache_hit
    assert first.table_path == second.table_path
    assert json.loads(first.manifest_path.read_text())["line_count"] == 2


def test_workflow_resolves_automatic_aer_window(monkeypatch):
    expected = aer_data.LineList.demo_near_ir()
    captured = {}

    class Artifact:
        line_list = expected

    def fake_load(**kwargs):
        captured.update(kwargs)
        return Artifact()

    monkeypatch.setattr("genmolfit.workflow.load_aer_line_window", fake_load)
    wavenumber = np.linspace(4319.0, 4321.0, 20)
    spectrum = Spectrum(wavelength=1.0e4 / wavenumber, flux=np.ones(wavenumber.size))

    resolved = _resolve_line_list(
        spectrum,
        line_list=None,
        line_list_path=None,
        hitran_par=None,
        hitran_species=("H2O",),
        hitran_min_strength=None,
        hitran_max_lines=None,
        line_cutoff_cm=None,
        line_wing_mode="full",
        lblrtm_sample=4.0,
        lblrtm_alfal0=0.04,
        lblrtm_hwf3=64.0,
        allow_empty_hitran=True,
    )

    assert resolved is expected
    assert captured["species"] == ("H2O",)
    assert captured["wavenumber_min_cm"] <= 4319.0 - 25.0
    assert captured["wavenumber_max_cm"] >= 4321.0 + 25.0
