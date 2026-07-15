from __future__ import annotations

import json
import numpy as np
import pytest

from genmolfit import cache_hitran_par, fetch_hitran_lines
from genmolfit.hitran import parse_hitran_local_iso_id
from genmolfit.line_data import HitranAcquisitionError


def _fixed_decimal(value, width, decimals):
    text = f"{value:.{decimals}f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return f"{text:>{width}}"[-width:]


def _hitran_row(*, mol_id=1, iso_token="1", wavenumber=4320.0, strength=1.0e-24):
    row = (
        f"{mol_id:2d}"
        f"{iso_token:1s}"
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


class _Response:
    def __init__(self, text):
        self._payload = text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._payload


def test_parse_hitran_local_isotopologue_ids():
    assert parse_hitran_local_iso_id("1") == 1
    assert parse_hitran_local_iso_id("0") == 10
    assert parse_hitran_local_iso_id("A") == 11
    assert parse_hitran_local_iso_id("Z") == 36
    with pytest.raises(ValueError):
        parse_hitran_local_iso_id("?")


def test_cache_local_hitran_file_is_deterministic_and_validated(tmp_path):
    source = tmp_path / "source.par"
    source.write_text(
        _hitran_row(wavenumber=4319.5)
        + "\n"
        + _hitran_row(wavenumber=4320.5)
        + "\n"
        + _hitran_row(mol_id=7, wavenumber=4320.0)
        + "\n"
    )
    cache = tmp_path / "cache"
    first = cache_hitran_par(
        source,
        species=("H2O",),
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        cache_dir=cache,
    )
    second = cache_hitran_par(
        source,
        species=("H2O",),
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        cache_dir=cache,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert first.table_path == second.table_path
    assert first.line_list.wavelength.size == 2
    assert first.manifest["species_with_lines"] == ["H2O"]
    assert "input_sha256" in first.manifest
    assert first.line_list.data_provenance["manifest_sha256"]
    assert first.line_list.data_provenance["request"] == first.manifest["request"]
    assert set(cache.iterdir()) == {first.par_path, first.table_path, first.manifest_path}


def test_fetch_hitran_uses_api_v2_and_preserves_isotopologue_metadata(tmp_path):
    calls = []
    transition_data = (
        _hitran_row(wavenumber=4319.5) + ",101,1\n" + _hitran_row(wavenumber=4320.5) + ",102,1\n"
    )

    def opener(request, timeout):
        url = request.full_url
        calls.append((url, timeout))
        if url.endswith("/info"):
            return _Response(json.dumps({"status": "OK", "content": {"data": {"results_dir": "results", "edition": "test-edition"}}}))
        if "/isotopologues?" in url:
            return _Response(json.dumps({"status": "OK", "content": {"data": [{"id": 1, "isoid": 1, "abundance": 0.997317, "mass": 18.010565}]}}))
        if "/transitions?" in url:
            return _Response(json.dumps({"status": "OK", "content": {"data": "window.data"}}))
        if url.endswith("/results/window.data"):
            return _Response(transition_data)
        raise AssertionError(url)

    artifact = fetch_hitran_lines(
        ("H2O",),
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        api_key="top-secret",
        cache_dir=tmp_path,
        opener=opener,
    )

    assert artifact.line_list.line_source == "hitran_api_v2"
    np.testing.assert_array_equal(artifact.line_list.global_iso_id, [1, 1])
    np.testing.assert_allclose(artifact.line_list.natural_abundance, 0.997317)
    np.testing.assert_allclose(artifact.line_list.molecular_mass_amu, 18.010565)
    assert artifact.manifest["database_edition"] == "test-edition"
    assert artifact.line_list.data_provenance["database_edition"] == "test-edition"
    assert "top-secret" not in artifact.manifest_path.read_text()
    assert any("/api/v2/top-secret/transitions?" in url for url, _ in calls)

    cached = fetch_hitran_lines(
        ("H2O",),
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        cache_dir=tmp_path,
        opener=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert cached.cache_hit


def test_fetch_hitran_requires_key_only_on_cache_miss(tmp_path, monkeypatch):
    monkeypatch.delenv("HITRAN_API_KEY", raising=False)
    with pytest.raises(HitranAcquisitionError, match="API key"):
        fetch_hitran_lines(
            ("O2",),
            wavelength_min_micron=0.75,
            wavelength_max_micron=0.78,
            cache_dir=tmp_path,
        )


def test_tampered_cache_is_not_accepted_offline(tmp_path):
    source = tmp_path / "source.par"
    source.write_text(_hitran_row() + "\n")
    artifact = cache_hitran_par(
        source,
        species=("H2O",),
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        cache_dir=tmp_path / "cache",
    )
    artifact.table_path.write_text(artifact.table_path.read_text() + "# tampered\n")
    refreshed = cache_hitran_par(
        source,
        species=("H2O",),
        wavenumber_min_cm=4319.0,
        wavenumber_max_cm=4321.0,
        cache_dir=tmp_path / "cache",
    )
    assert not refreshed.cache_hit
    assert "tampered" not in refreshed.table_path.read_text()
