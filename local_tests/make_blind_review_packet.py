from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "local_tests" / "science_readiness" / "results"
PACKET = RESULTS / "independent_review"
ANSWER_KEY = RESULTS / "independent_review_answer_key.csv"


def _packet_manifest_path() -> Path:
    return PACKET / "packet_manifest.json"


def _review_archive_path() -> Path:
    return PACKET.parent / "independent_review_packet.zip"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reviewer_files() -> list[Path]:
    manifest_path = _packet_manifest_path()
    files = []
    for path in PACKET.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"refusing to package symlink: {path}")
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(PACKET)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe reviewer-packet path: {relative}")
        allowed = path.name in {"README.md", "review.csv", "held_out_review.csv"} or (
            len(relative.parts) == 1
            and path.name.startswith("case_")
            and path.suffix in {".ecsv", ".png"}
            and path.stem[5:].isdigit()
        )
        if not allowed:
            raise RuntimeError(f"unexpected file in reviewer packet: {relative}")
        files.append(path)
    required = {"README.md", "review.csv", "held_out_review.csv"}
    present = {path.name for path in files}
    if not required.issubset(present):
        raise RuntimeError(f"reviewer packet is missing files: {sorted(required - present)}")
    return sorted(files, key=lambda path: path.relative_to(PACKET).as_posix())


def _write_packet_manifest() -> None:
    entries = [
        {
            "path": path.relative_to(PACKET).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _reviewer_files()
    ]
    manifest = {
        "schema_version": 1,
        "purpose": "PyMolFit independent blind telluric-correction review",
        "files": entries,
    }
    _packet_manifest_path().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_reviewer_archive() -> str:
    _write_packet_manifest()
    manifest_path = _packet_manifest_path()
    review_archive = _review_archive_path()
    files = _reviewer_files() + [manifest_path]
    temporary = review_archive.with_suffix(review_archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(files, key=lambda item: item.relative_to(PACKET).as_posix()):
                relative = path.relative_to(PACKET).as_posix()
                member = f"independent_review/{relative}"
                if "answer_key" in member.lower():
                    raise RuntimeError("private answer key cannot be included in reviewer archive")
                info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        temporary.replace(review_archive)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(review_archive)


def _finite_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    scale = float(np.nanmedian(values[finite])) if np.any(finite) else 1.0
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return values / scale


def _assignment(case_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(case_id.encode("ascii")).digest()
    return ("PyMolFit", "Molecfit") if digest[0] % 2 == 0 else ("Molecfit", "PyMolFit")


def _write_case(
    case_id: str,
    wavelength: np.ndarray,
    raw: np.ndarray,
    pymolfit: np.ndarray,
    molecfit: np.ndarray,
    *,
    source: str,
) -> dict[str, str]:
    candidate_a, candidate_b = _assignment(case_id)
    candidates = {"PyMolFit": pymolfit, "Molecfit": molecfit}
    candidate_a_flux = np.asarray(candidates[candidate_a], dtype=float)
    candidate_b_flux = np.asarray(candidates[candidate_b], dtype=float)

    table = Table()
    table["wavelength_micron"] = wavelength
    table["raw_relative_flux"] = raw
    table["candidate_A_relative_flux"] = candidate_a_flux
    table["candidate_B_relative_flux"] = candidate_b_flux
    table_path = PACKET / f"{case_id}.ecsv"
    table.write(table_path, format="ascii.ecsv", overwrite=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    panels = (
        (raw, "Observed spectrum", "0.2"),
        (candidate_a_flux, "Candidate A", "#2b6cb0"),
        (candidate_b_flux, "Candidate B", "#c05621"),
    )
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values, _, _ in panels if np.any(np.isfinite(values))]
    )
    lower, upper = np.nanpercentile(finite_values, [0.5, 99.5])
    padding = 0.08 * max(upper - lower, 1.0e-3)
    for axis, (values, title, color) in zip(axes, panels, strict=True):
        axis.plot(wavelength, values, color=color, lw=0.75)
        axis.set_ylabel("Relative flux")
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylim(lower - padding, upper + padding)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Wavelength [micron]")
    fig.suptitle(f"Blind telluric-correction review: {case_id}")
    fig.savefig(PACKET / f"{case_id}.png", dpi=180)
    plt.close(fig)
    return {
        "case": case_id,
        "case_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "source": source,
        "candidate_A": candidate_a,
        "candidate_B": candidate_b,
    }


def _xshooter_cases() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    result = []
    names = (
        "xshooter_o2_a",
        "xshooter_h2o_j",
        "xshooter_h2o_h",
        "xshooter_h2o_co2_k",
    )
    for index, name in enumerate(names, start=1):
        path = RESULTS / "xshooter" / name / "comparison.ecsv"
        if not path.exists():
            continue
        table = Table.read(path, format="ascii.ecsv")
        result.append(
            (
                f"case_{index:02d}",
                np.asarray(table["wavelength_air_micron"], dtype=float),
                _finite_normalize(table["raw_flux"]),
                np.asarray(table["pymolfit_corrected_relative"], dtype=float),
                np.asarray(table["molecfit_corrected_relative"], dtype=float),
                str(path.relative_to(ROOT)),
            )
        )
    return result


def _crires_cases(start_index: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    result = []
    base = ROOT / "local_tests" / "rho01_molecfit_vs_pymolfit_lband"
    for offset, chip in enumerate((2, 6, 10, 14, 18)):
        path = base / f"chip_{chip:02d}_comparison.ecsv"
        if not path.exists():
            continue
        table = Table.read(path, format="ascii.ecsv")
        wavelength = np.asarray(table["wavelength"], dtype=float)
        flux = np.asarray(table["flux"], dtype=float)
        gen_continuum = np.asarray(table["continuum"], dtype=float)
        molecfit_continuum = np.asarray(table["molecfit_continuum"], dtype=float)
        gen_transmission = np.asarray(table["transmission"], dtype=float)
        molecfit_transmission = np.asarray(table["molecfit_transmission"], dtype=float)
        raw = flux / np.where(np.abs(gen_continuum) > 0, gen_continuum, np.nan)
        gen_corrected = flux / np.where(
            (gen_transmission > 0.03) & (np.abs(gen_continuum) > 0),
            gen_transmission * gen_continuum,
            np.nan,
        )
        molecfit_corrected = flux / np.where(
            (molecfit_transmission > 0.03) & (np.abs(molecfit_continuum) > 0),
            molecfit_transmission * molecfit_continuum,
            np.nan,
        )
        result.append(
            (
                f"case_{start_index + offset:02d}",
                wavelength,
                raw,
                gen_corrected,
                molecfit_corrected,
                str(path.relative_to(ROOT)),
            )
        )
    return result


def _existing_review_rows() -> dict[tuple[str, str], dict[str, str]]:
    path = PACKET / "review.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    if "case_sha256" not in fieldnames:
        response_fields = fieldnames - {"case"}
        if any(any(str(row.get(key, "")).strip() for key in response_fields) for row in rows):
            raise RuntimeError(
                "Refusing to overwrite completed legacy review.csv; archive it before rebuilding the packet"
            )
        return {}
    return {
        (str(row.get("case", "")), str(row.get("case_sha256", ""))): row
        for row in rows
    }


def _prepare_held_out_form() -> None:
    path = PACKET / "held_out_review.csv"
    fieldnames = (
        "reviewer",
        "date",
        "dataset",
        "instrument",
        "molecfit_version",
        "pymolfit_version",
        "decision",
        "intrinsic_lines_preserved",
        "masks_usable",
        "uncertainties_usable",
        "settings_and_notes",
    )
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fields = tuple(reader.fieldnames or ())
        if existing_rows:
            if existing_fields != fieldnames:
                raise RuntimeError(
                    "Refusing to overwrite completed held_out_review.csv with an older schema"
                )
            return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def build_packet() -> None:
    PACKET.mkdir(parents=True, exist_ok=True)
    existing_reviews = _existing_review_rows()
    for pattern in ("case_*.png", "case_*.ecsv"):
        for path in PACKET.glob(pattern):
            path.unlink()

    cases = _xshooter_cases()
    cases.extend(_crires_cases(len(cases) + 1))
    answer_rows = [
        _write_case(case_id, wavelength, raw, pymolfit, molecfit, source=source)
        for case_id, wavelength, raw, pymolfit, molecfit, source in cases
    ]

    with ANSWER_KEY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "case_sha256", "source", "candidate_A", "candidate_B"),
        )
        writer.writeheader()
        writer.writerows(answer_rows)

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
    with (PACKET / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for answer in answer_rows:
            key = (answer["case"], answer["case_sha256"])
            retained = existing_reviews.get(key, {})
            writer.writerow(
                {
                    "case": answer["case"],
                    "case_sha256": answer["case_sha256"],
                    **{field: retained.get(field, "") for field in review_fields[2:]},
                }
            )
    (PACKET / "README.md").write_text(
        """# Independent blind-review packet

Give this packet to an experienced telluric-correction user. The private answer
key is retained separately by the developer. Candidate identity is
independently permuted in each case.

For every case, inspect the PNG and ECSV for residual telluric structure,
over-correction, discontinuities, and preservation of plausible intrinsic
features. Record the assessment in `review.csv`. Use `A`, `B`, or `EQUIVALENT`
for `preferred_candidate`; use `YES` or `NO` for each material-artifact and
intrinsic-line field. A preference is not required when the candidates are
scientifically equivalent. Do not alter `case_sha256`.

The reviewer must also process at least one familiar held-out spectrum that
was not selected by the PyMolFit developer, using documented settings for
both programs. Record that dataset, versions, settings, residual diagnostics,
mask and uncertainty usability, and the final signed `PASS` or `FAIL` decision
in `held_out_review.csv`. Use `YES` or `NO` in its three assessment fields.

Acceptance requires no scientifically material regression relative to
Molecfit, no unexplained modification of intrinsic spectral lines, usable
masks/uncertainties, and a passing held-out assessment.
""",
        encoding="utf-8",
    )
    _prepare_held_out_form()
    archive_sha256 = _write_reviewer_archive()
    print(f"Wrote {len(answer_rows)} blind cases to {PACKET}")
    print(f"Reviewer archive: {_review_archive_path()}")
    print(f"Reviewer archive SHA-256: {archive_sha256}")
    print(f"Keep the answer key private: {ANSWER_KEY}")


if __name__ == "__main__":
    build_packet()
