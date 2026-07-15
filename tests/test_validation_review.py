from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from local_tests import make_blind_review_packet as packet
from local_tests import run_science_readiness_validation as campaign


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_independent_review_gate_pass_fail_and_manual(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "OUTPUT_DIR", tmp_path)
    review_dir = tmp_path / "independent_review"
    case_hash = "a" * 64
    answer_fields = ("case", "case_sha256", "source", "candidate_A", "candidate_B")
    review_fields = (
        "case",
        "case_sha256",
        "preferred_candidate",
        "candidate_A_material_artifact",
        "candidate_B_material_artifact",
        "candidate_A_intrinsic_lines_preserved",
        "candidate_B_intrinsic_lines_preserved",
        "notes",
    )
    held_fields = (
        "reviewer",
        "date",
        "dataset",
        "instrument",
        "molecfit_version",
        "genmolfit_version",
        "decision",
        "intrinsic_lines_preserved",
        "masks_usable",
        "uncertainties_usable",
        "settings_and_notes",
    )
    _write_csv(
        tmp_path / "independent_review_answer_key.csv",
        answer_fields,
        [{
            "case": "case_01",
            "case_sha256": case_hash,
            "source": "test",
            "candidate_A": "GenMolFit",
            "candidate_B": "Molecfit",
        }],
    )
    review = {
        "case": "case_01",
        "case_sha256": case_hash,
        "preferred_candidate": "EQUIVALENT",
        "candidate_A_material_artifact": "NO",
        "candidate_B_material_artifact": "NO",
        "candidate_A_intrinsic_lines_preserved": "YES",
        "candidate_B_intrinsic_lines_preserved": "YES",
        "notes": "blind assessment",
    }
    _write_csv(review_dir / "review.csv", review_fields, [review])
    _write_csv(
        review_dir / "held_out_review.csv",
        held_fields,
        [{
            "reviewer": "Reviewer",
            "date": "2026-07-13",
            "dataset": "held-out",
            "instrument": "other",
            "molecfit_version": "4",
            "genmolfit_version": "0.1.0",
            "decision": "PASS",
            "intrinsic_lines_preserved": "YES",
            "masks_usable": "YES",
            "uncertainties_usable": "YES",
            "settings_and_notes": "same fit windows",
        }],
    )
    assert campaign._independent_review_check().status == "PASS"

    review["candidate_A_material_artifact"] = "YES"
    _write_csv(review_dir / "review.csv", review_fields, [review])
    assert campaign._independent_review_check().status == "FAIL"

    review["preferred_candidate"] = ""
    _write_csv(review_dir / "review.csv", review_fields, [review])
    assert campaign._independent_review_check().status == "MANUAL"


def test_blind_packet_preserves_only_matching_hashed_responses(tmp_path, monkeypatch):
    packet_dir = tmp_path / "independent_review"
    monkeypatch.setattr(packet, "PACKET", packet_dir)
    monkeypatch.setattr(packet, "ANSWER_KEY", tmp_path / "answer_key.csv")
    wavelength = np.linspace(0.75, 0.76, 8)
    raw = np.ones(8)
    gen = np.linspace(0.98, 1.01, 8)
    mol = np.linspace(0.97, 1.00, 8)
    cases = [("case_01", wavelength, raw, gen, mol, "source")]
    monkeypatch.setattr(packet, "_xshooter_cases", lambda: cases)
    monkeypatch.setattr(packet, "_crires_cases", lambda start: [])

    packet.build_packet()
    review_path = packet_dir / "review.csv"
    fields, rows = campaign._read_csv_rows(review_path)
    rows[0]["preferred_candidate"] = "A"
    _write_csv(review_path, fields, rows)
    packet.build_packet()
    _, retained = campaign._read_csv_rows(review_path)
    assert retained[0]["preferred_candidate"] == "A"

    changed_cases = [("case_01", wavelength, raw, gen + 0.01, mol, "source")]
    monkeypatch.setattr(packet, "_xshooter_cases", lambda: changed_cases)
    packet.build_packet()
    _, reset = campaign._read_csv_rows(review_path)
    assert reset[0]["preferred_candidate"] == ""
    assert reset[0]["case_sha256"] != retained[0]["case_sha256"]


def test_reviewer_archive_campaign_gate(tmp_path, monkeypatch):
    packet_dir = tmp_path / "independent_review"
    packet_dir.mkdir()
    for name in ("README.md", "review.csv", "held_out_review.csv"):
        (packet_dir / name).write_text("review material\n", encoding="utf-8")
    for case in range(1, 10):
        (packet_dir / f"case_{case:02d}.ecsv").write_text("# table\n", encoding="utf-8")
        (packet_dir / f"case_{case:02d}.png").write_bytes(b"PNG")
    monkeypatch.setattr(packet, "PACKET", packet_dir)
    packet._write_reviewer_archive()
    monkeypatch.setattr(campaign, "REVIEW_PACKET", packet_dir)
    monkeypatch.setattr(
        campaign, "REVIEW_ARCHIVE", tmp_path / "independent_review_packet.zip"
    )

    assert campaign._review_packet_archive_check().status == "PASS"
    (packet_dir / "case_01.ecsv").write_text("changed\n", encoding="utf-8")
    assert campaign._review_packet_archive_check().status == "FAIL"


def test_authenticated_hitran_campaign_gate_is_source_bound(tmp_path, monkeypatch):
    receipt_path = tmp_path / "authenticated_hitran_receipt.json"
    monkeypatch.setattr(campaign, "HITRAN_RECEIPT", receipt_path)
    assert campaign._authenticated_hitran_check().status == "MANUAL"

    request = {
        "source": "hitran_api",
        "species": ["O2"],
        "molecule_ids": [7],
        "wavelength_min_micron": 1.0e4 / 13161.0,
        "wavelength_max_micron": 1.0e4 / 13160.0,
        "wavenumber_min_cm": 13160.0,
        "wavenumber_max_cm": 13161.0,
    }
    request_sha256 = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    source_sha256 = hashlib.sha256(
        (campaign.ROOT / "src" / "genmolfit" / "line_data.py").read_bytes()
    ).hexdigest()
    validator_sha256 = hashlib.sha256(
        (campaign.ROOT / "local_tests" / "validate_authenticated_hitran.py").read_bytes()
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "online_request_completed": True,
        "cache_hit": False,
        "credential_environment_variable": "HITRAN_API_KEY",
        "credential_persisted": False,
        "api_base_url": "https://hitran.org",
        "api_version": "v2",
        "database_edition": "test",
        "genmolfit_version": campaign.__version__,
        "client_source_sha256": source_sha256,
        "validator_source_sha256": validator_sha256,
        "request": request,
        "request_sha256": request_sha256,
        "line_count": 12,
        "species_with_lines": ["O2"],
        "artifact_sha256": {"par": "a" * 64, "table": "b" * 64, "manifest": "c" * 64},
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert campaign._authenticated_hitran_check().status == "PASS"

    receipt["client_source_sha256"] = "d" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert campaign._authenticated_hitran_check().status == "FAIL"
