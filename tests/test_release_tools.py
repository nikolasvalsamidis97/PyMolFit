from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "local_tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_blind_review_packet = _load_tool("make_blind_review_packet")
validate_authenticated_hitran = _load_tool("validate_authenticated_hitran")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reviewer_archive_is_deterministic_and_excludes_sibling_answer_key(
    tmp_path, monkeypatch
):
    packet = tmp_path / "results" / "independent_review"
    packet.mkdir(parents=True)
    for name, content in {
        "README.md": "instructions\n",
        "review.csv": "case,decision\n",
        "held_out_review.csv": "reviewer,decision\n",
        "case_01.ecsv": "# table\n",
        "case_01.png": "not-a-real-png",
    }.items():
        (packet / name).write_text(content, encoding="utf-8")
    answer_key = packet.parent / "independent_review_answer_key.csv"
    answer_key.write_text("private answer\n", encoding="utf-8")
    archive = packet.parent / "independent_review_packet.zip"
    monkeypatch.setattr(make_blind_review_packet, "PACKET", packet)

    first = make_blind_review_packet._write_reviewer_archive()
    second = make_blind_review_packet._write_reviewer_archive()

    assert first == second == _sha256(archive)
    with zipfile.ZipFile(archive) as handle:
        members = handle.namelist()
        assert all("answer_key" not in name for name in members)
        assert answer_key.read_bytes() not in {handle.read(name) for name in members}
        manifest = json.loads(handle.read("independent_review/packet_manifest.json"))
    assert {entry["path"] for entry in manifest["files"]} == {
        "README.md",
        "review.csv",
        "held_out_review.csv",
        "case_01.ecsv",
        "case_01.png",
    }


def test_reviewer_archive_rejects_unexpected_files(tmp_path, monkeypatch):
    packet = tmp_path / "independent_review"
    packet.mkdir()
    for name in ("README.md", "review.csv", "held_out_review.csv"):
        (packet / name).write_text("safe\n", encoding="utf-8")
    (packet / "private.csv").write_text("must not ship\n", encoding="utf-8")
    monkeypatch.setattr(make_blind_review_packet, "PACKET", packet)

    with pytest.raises(RuntimeError, match="unexpected file"):
        make_blind_review_packet._write_reviewer_archive()


def test_authenticated_hitran_receipt_is_redacted_and_source_bound(tmp_path, monkeypatch):
    secret = "test-hitran-secret-that-must-not-be-written"
    monkeypatch.setenv("HITRAN_API_KEY", secret)

    def fetcher(species, **kwargs):
        assert species == ("O2",)
        assert kwargs["force"] is True
        assert kwargs["wavenumber_min_cm"] == 13160.0
        assert kwargs["wavenumber_max_cm"] == 13161.0
        directory = Path(kwargs["cache_dir"])
        par_path = directory / "o2.par"
        table_path = directory / "o2.ecsv"
        manifest_path = directory / "o2.json"
        par_path.write_text("fixed-width line\n", encoding="utf-8")
        table_path.write_text("# ECSV\n", encoding="utf-8")
        request = {
            "source": "hitran_api",
            "species": ["O2"],
            "molecule_ids": [7],
            "wavelength_min_micron": 1.0e4 / 13161.0,
            "wavelength_max_micron": 1.0e4 / 13160.0,
            "wavenumber_min_cm": 13160.0,
            "wavenumber_max_cm": 13161.0,
        }
        request_hash = validate_authenticated_hitran._request_sha256(request)
        manifest = {
            "request": request,
            "request_sha256": request_hash,
            "api_version": "v2",
            "database_edition": "mock-edition",
            "line_count": 2,
            "species_with_lines": ["O2"],
            "actual_wavenumber_range_cm": [13160.2, 13160.8],
            "result_filename": "mock.data",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(
            cache_hit=False,
            manifest=manifest,
            par_path=par_path,
            table_path=table_path,
            manifest_path=manifest_path,
        )

    output = tmp_path / "receipt.json"
    validate_authenticated_hitran.run(output, fetcher=fetcher)
    text = output.read_text(encoding="utf-8")
    receipt = json.loads(text)

    assert secret not in text
    assert receipt["line_count"] == 2
    assert receipt["cache_hit"] is False
    assert receipt["credential_persisted"] is False
    assert receipt["client_source_sha256"] == _sha256(
        Path(validate_authenticated_hitran.line_data_impl.__file__)
    )
    assert receipt["validator_source_sha256"] == _sha256(
        Path(validate_authenticated_hitran.__file__)
    )
