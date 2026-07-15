from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "local_tests" / "compare_rho01_genmolfit_molecfit_lband.py"
OUTPUT = PROJECT / "local_tests" / "rho01_convention_calibration"


VARIANTS = {
    "baseline_integrate": {},
    "center_rebin": {
        "GENMOLFIT_HIGH_RESOLUTION_REBIN_MODE": "center",
    },
    "molecfit_voigt_lsf": {
        "GENMOLFIT_MOLECFIT_VOIGT_LSF": "1",
    },
    "center_rebin_molecfit_voigt_lsf": {
        "GENMOLFIT_HIGH_RESOLUTION_REBIN_MODE": "center",
        "GENMOLFIT_MOLECFIT_VOIGT_LSF": "1",
    },
    "extra_cia_rayleigh_n2": {
        "GENMOLFIT_EXTRA_CIA": "1",
        "GENMOLFIT_RAYLEIGH": "1",
        "GENMOLFIT_N2_CONTINUUM": "1",
    },
    "self_contained_header_atmosphere": {
        "GENMOLFIT_ATMOSPHERE_SOURCE": "header_slant",
    },
}


def selected_variants() -> dict[str, dict[str, str]]:
    requested = os.environ.get("GENMOLFIT_VARIANTS")
    if requested is None:
        return VARIANTS
    names = [name.strip() for name in requested.split(",") if name.strip()]
    missing = [name for name in names if name not in VARIANTS]
    if missing:
        raise ValueError(f"unknown variants: {', '.join(missing)}")
    return {name: VARIANTS[name] for name in names}


def run_variant(name: str, settings: dict[str, str]) -> dict[str, object]:
    output = OUTPUT / name
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(settings)
    env["GENMOLFIT_WRITE_PLOTS"] = "0"
    env["GENMOLFIT_COMPARISON_OUTPUT"] = str(output)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT / "src") if not existing_pythonpath else f"{PROJECT / 'src'}:{existing_pythonpath}"
    )
    t0 = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=PROJECT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    seconds = time.perf_counter() - t0
    if completed.returncode != 0:
        (output / "stdout.txt").write_text(completed.stdout)
        (output / "stderr.txt").write_text(completed.stderr)
        raise RuntimeError(f"{name} failed; see {output}")

    summary = Table.read(output / "summary.ecsv", format="ascii.ecsv")
    rms = np.asarray(summary["transmission_rms_difference"], dtype=float)
    median_abs = np.asarray(summary["transmission_median_abs_difference"], dtype=float)
    row = {
        "variant": name,
        "seconds": seconds,
        "median_rms": float(np.nanmedian(rms)),
        "mean_rms": float(np.nanmean(rms)),
        "max_rms": float(np.nanmax(rms)),
        "median_abs": float(np.nanmedian(median_abs)),
        "mean_median_abs": float(np.nanmean(median_abs)),
        "high_resolution_rebin_mode": str(summary["high_resolution_rebin_mode"][0]),
        "lsf_molecfit_voigt": bool(summary["lsf_molecfit_voigt"][0]),
        "atmosphere_source": str(summary["atmosphere_source"][0]),
        "extra_cia": bool(summary["extra_cia"][0]),
        "rayleigh": bool(summary["rayleigh"][0]),
        "n2_continuum": bool(summary["n2_continuum"][0]),
        "success": bool(np.all(np.asarray(summary["success"], dtype=bool))),
        "output": str(output),
    }
    (output / "stdout.txt").write_text(completed.stdout)
    (output / "stderr.txt").write_text(completed.stderr)
    return row


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, settings in selected_variants().items():
        print(f"Running {name}...")
        row = run_variant(name, settings)
        rows.append(row)
        print(
            f"  median RMS={row['median_rms']:.6g}, "
            f"mean RMS={row['mean_rms']:.6g}, seconds={row['seconds']:.1f}"
        )
    table = Table(rows=rows)
    table.sort("median_rms")
    table.write(OUTPUT / "summary.ecsv", format="ascii.ecsv", overwrite=True)
    table.write(OUTPUT / "summary.csv", format="ascii.csv", overwrite=True)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
