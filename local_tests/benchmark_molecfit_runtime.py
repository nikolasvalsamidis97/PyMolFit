from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MOLECFIT_ROOT = Path.home() / ".criresflow" / "molecfit"
ESOREX = MOLECFIT_ROOT / "bin" / "esorex"
INPUT = Path(
    os.environ.get(
        "PYMOLFIT_RHO01_MOLECFIT_INPUT",
        PROJECT / "local_tests" / "data" / "rho01" / "molecfit_input",
    )
)
OUTPUT = Path("/tmp/pymolfit_molecfit_speed_benchmark")
FTOL = os.environ.get("MOLECFIT_BENCHMARK_FTOL", "1e-10")
XTOL = os.environ.get("MOLECFIT_BENCHMARK_XTOL", "1e-10")


def main() -> None:
    if not ESOREX.exists():
        raise FileNotFoundError(ESOREX)
    science = INPUT / "SCIENCE_A.fits"
    wave_include = INPUT / "WAVE_INCLUDE_A.fits"
    if not science.exists() or not wave_include.exists():
        raise FileNotFoundError("rho01 Molecfit inputs are unavailable")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pymolfit_molecfit_runtime_") as tmp:
        work = Path(tmp)
        products = work / "products"
        products.mkdir()
        staged_science = work / "SCIENCE_A.fits"
        staged_wave_include = work / "WAVE_INCLUDE_A.fits"
        shutil.copy2(science, staged_science)
        shutil.copy2(wave_include, staged_wave_include)
        sof = work / "model.sof"
        sof.write_text(
            f"{staged_science} SCIENCE\n{staged_wave_include} WAVE_INCLUDE\n",
            encoding="utf-8",
        )
        command = [
            str(ESOREX),
            f"--output-dir={products}",
            "molecfit_model",
            "--LIST_MOLEC=H2O,CO2,CO,CH4,O2",
            "--FIT_MOLEC=1,1,1,1,1",
            "--REL_COL=1.0,1.0,1.0,1.0,1.0",
            "--COLUMN_LAMBDA=WAVE",
            "--COLUMN_FLUX=SPEC",
            "--COLUMN_DFLUX=ERR",
            "--WLG_TO_MICRON=1.0",
            "--WAVELENGTH_FRAME=VAC",
            "--FIT_RES_BOX=TRUE",
            "--RES_BOX=1.0",
            "--FIT_RES_GAUSS=TRUE",
            "--RES_GAUSS=1.0",
            "--FIT_RES_LORENTZ=TRUE",
            "--RES_LORENTZ=1.0",
            "--KERNMODE=FALSE",
            "--KERNFAC=3.0",
            "--VARKERN=TRUE",
            f"--FTOL={FTOL}",
            f"--XTOL={XTOL}",
            "--FIT_TELESCOPE_BACKGROUND=TRUE",
            "--REFERENCE_ATMOSPHERIC=equ.fits",
            "--GDAS_PROFILE=auto",
            "--LBLRTM_ALFAL0=0.0",
            str(sof),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=work, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - start
        (OUTPUT / "molecfit_model.log").write_text(
            completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Molecfit failed; see {OUTPUT / 'molecfit_model.log'}")

    summary = OUTPUT / "molecfit_speed_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "seconds"))
        writer.writeheader()
        writer.writerow({"stage": "molecfit_model", "seconds": elapsed})
    print(f"Molecfit model: {elapsed:.3f} s")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
