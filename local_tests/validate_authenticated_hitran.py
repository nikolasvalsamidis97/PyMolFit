from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pymolfit.line_data as line_data_impl
from pymolfit import __version__, fetch_hitran_lines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    ROOT
    / "local_tests"
    / "science_readiness"
    / "results"
    / "authenticated_hitran_receipt.json"
)
API_KEY_ENV = "HITRAN_API_KEY"
SPECIES = ("O2",)
WAVENUMBER_MIN_CM = 13160.0
WAVENUMBER_MAX_CM = 13161.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_sha256(request: object) -> str:
    encoded = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def run(
    output: Path = DEFAULT_RECEIPT,
    *,
    timeout_s: float = 60.0,
    fetcher=fetch_hitran_lines,
) -> Path:
    secret = os.environ.get(API_KEY_ENV, "").strip()
    if not secret:
        raise RuntimeError(
            f"set {API_KEY_ENV} to a real HITRAN API v2 key before running this check"
        )

    client_source = Path(line_data_impl.__file__).resolve()
    validator_source = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="pymolfit_hitran_live_") as directory:
        artifact = fetcher(
            SPECIES,
            wavenumber_min_cm=WAVENUMBER_MIN_CM,
            wavenumber_max_cm=WAVENUMBER_MAX_CM,
            cache_dir=directory,
            force=True,
            timeout_s=timeout_s,
        )
        manifest = dict(artifact.manifest)
        request = manifest.get("request")
        if artifact.cache_hit:
            raise RuntimeError("live HITRAN check unexpectedly returned a cache hit")
        if not isinstance(request, dict) or request.get("source") != "hitran_api":
            raise RuntimeError("HITRAN response did not produce an API-backed manifest")
        if int(manifest.get("line_count", 0)) < 1:
            raise RuntimeError("HITRAN returned no O2 lines in the fixed validation interval")
        for path in (artifact.par_path, artifact.table_path, artifact.manifest_path):
            if not path.is_file():
                raise RuntimeError(f"HITRAN acquisition did not produce {path.name}")

        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "online_request_completed": True,
            "cache_hit": False,
            "credential_environment_variable": API_KEY_ENV,
            "credential_persisted": False,
            "api_base_url": line_data_impl.HITRAN_API_BASE_URL,
            "api_version": manifest.get("api_version"),
            "database_edition": manifest.get("database_edition"),
            "pymolfit_version": __version__,
            "client_source": "src/pymolfit/line_data.py",
            "client_source_sha256": _sha256(client_source),
            "validator_source": "local_tests/validate_authenticated_hitran.py",
            "validator_source_sha256": _sha256(validator_source),
            "request": request,
            "request_sha256": manifest.get("request_sha256"),
            "line_count": int(manifest["line_count"]),
            "species_with_lines": list(manifest.get("species_with_lines", ())),
            "actual_wavenumber_range_cm": list(manifest["actual_wavenumber_range_cm"]),
            "artifact_sha256": {
                "par": _sha256(artifact.par_path),
                "table": _sha256(artifact.table_path),
                "manifest": _sha256(artifact.manifest_path),
            },
            "service_result_filename": manifest.get("result_filename"),
        }
        if receipt["request_sha256"] != _request_sha256(request):
            raise RuntimeError("HITRAN manifest request fingerprint is inconsistent")
        serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if secret in serialized or secret in artifact.manifest_path.read_text(encoding="utf-8"):
            raise RuntimeError("refusing to persist a receipt containing the HITRAN credential")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise PyMolFit's authenticated HITRAN client and write a redacted receipt."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    try:
        receipt = run(args.output, timeout_s=args.timeout)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Authenticated HITRAN acquisition passed: {receipt}")


if __name__ == "__main__":
    main()
