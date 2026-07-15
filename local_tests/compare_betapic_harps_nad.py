from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from pymolfit import LineList, correct_file


PROJECT = Path(__file__).resolve().parents[1]
SPECTRA_DIR = Path(
    os.environ.get(
        "PYMOLFIT_BETAPIC_SPECTRA_DIR",
        PROJECT / "local_tests" / "data" / "betapic",
    )
)
OUTPUT_DIR = Path(__file__).resolve().parent / "betapic_harps_nad_comparison"
MOLECFIT_ROOT = Path.home() / ".criresflow" / "molecfit"
MOLECFIT_ESOREX = MOLECFIT_ROOT / "bin" / "esorex"
MOLECFIT_AER_LINE_DB = MOLECFIT_ROOT / "share" / "molecfit" / "data" / "hitran" / "aer_v_3.8.1.2"

NAD_CROP = (0.5882, 0.5907)
NAD_FIT = (0.58825, 0.59065)
NAD_EXCLUDE = (
    (0.58888, 0.58912),
    (0.58948, 0.58978),
)
SPEED_OF_LIGHT_KM_S = 299_792.458


def _read_harps_adp(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        wave_angstrom = np.asarray(hdul[1].data["WAVE"][0], dtype=float)
        flux = np.asarray(hdul[1].data["FLUX"][0], dtype=float)
    return wave_angstrom * 1e-4, flux, header


def _clean_flux(wavelength: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0)
    wavelength = wavelength[finite]
    flux = flux[finite]
    order = np.argsort(wavelength)
    return wavelength[order], flux[order]


def _write_crop_fits(source_path: Path, crop_path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    wavelength, flux, header = _read_harps_adp(source_path)
    keep = (wavelength >= NAD_CROP[0]) & (wavelength <= NAD_CROP[1])
    wavelength, flux = _clean_flux(wavelength[keep], flux[keep])

    primary = fits.PrimaryHDU(header=header)
    primary.header["ESO INS SLIT1 WID"] = (0.4, "Default HARPS fibre/slit width for Molecfit test")
    primary.header.setdefault("ESO TEL TH M1 TEMP", primary.header.get("ESO TEL AMBI TEMP", 15.0))
    primary.header["HIERARCH PYMOLFIT SOURCE"] = source_path.name
    primary.header["HIERARCH PYMOLFIT WAVE"] = "air micron"

    cols = [
        fits.Column(name="lambda", array=wavelength, format="D", unit="um"),
        fits.Column(name="flux", array=flux, format="D", unit="adu"),
    ]
    table = fits.BinTableHDU.from_columns(cols, name="SCIENCE")
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, table]).writeto(crop_path, overwrite=True)
    return wavelength, flux, primary.header


def _load_h2o_lines() -> tuple[LineList, str]:
    if not MOLECFIT_AER_LINE_DB.exists():
        raise FileNotFoundError(f"Missing Molecfit AER line database: {MOLECFIT_AER_LINE_DB}")
    wavenumber_min = 1.0e4 / (NAD_CROP[1] + 0.001)
    wavenumber_max = 1.0e4 / (NAD_CROP[0] - 0.001)
    return (
        LineList.from_aer_line_file(
            MOLECFIT_AER_LINE_DB,
            species=("H2O",),
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
        ),
        str(MOLECFIT_AER_LINE_DB),
    )


def _run_pymolfit(
    crop_path: Path,
    out_dir: Path,
    airmass: float,
    line_list: LineList,
    *,
    line_wing_mode: str = "lblrtm_dynamic",
    output_suffix: str = "",
) -> object:
    return correct_file(
        crop_path,
        out_dir / f"pymolfit_corrected{output_suffix}.txt",
        input_format="fits",
        wavelength_col="lambda",
        flux_col="flux",
        wavelength_unit="micron",
        wavelength_medium="air",
        line_list=line_list,
        physical=True,
        atmosphere_mode="mipas_gdas",
        mipas_profile="equ",
        gdas_mode="auto",
        gdas_download_timeout_s=30.0,
        airmass=airmass,
        continuum_order=2,
        rayleigh=False,
        lsf_box_width_pixels=1.0,
        lsf_sigma_pixels=1.0,
        lsf_lorentz_fwhm_pixels=2.0,
        high_resolution_grid=True,
        high_resolution_oversampling=5.0,
        high_resolution_rebin_mode="molecfit_overlap",
        line_wing_mode=line_wing_mode,
        fit_wavelength_shift=True,
        wavelength_shift_bounds=(-2.0e-4, 2.0e-4),
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.0, 5.0),
        fit_lsf_lorentz_fwhm=True,
        lsf_lorentz_fwhm_bounds=(0.0, 10.0),
        lsf_kernel_width_fwhm=3.0,
        lsf_molecfit_voigt=False,
        fit_ranges=(NAD_FIT,),
        exclude_ranges=NAD_EXCLUDE,
        product_path=out_dir / f"pymolfit_product{output_suffix}.ecsv",
        plot_path=out_dir / f"pymolfit_diagnostic{output_suffix}.png",
    )


def _run_molecfit(crop_path: Path, out_dir: Path) -> Path | None:
    if not MOLECFIT_ESOREX.exists():
        return None

    products_dir = out_dir / "molecfit_products"
    products_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "molecfit_model.log"
    with fits.open(crop_path) as hdul:
        header = hdul[0].header
        wavelength_frame, rv_args = _molecfit_wavelength_frame_args(header)

    with tempfile.TemporaryDirectory(prefix="pymolfit_harps_nad_") as tmp:
        stage = Path(tmp)
        staged_input = stage / "input.fits"
        staged_sof = stage / "model.sof"
        staged_out = stage / "out"
        staged_tmp = stage / "tmp"
        staged_out.mkdir()
        staged_tmp.mkdir()
        shutil.copy2(crop_path, staged_input)
        staged_sof.write_text(f"{staged_input} SCIENCE\n", encoding="utf-8")

        wave_exclude = ",".join(str(value) for interval in NAD_EXCLUDE for value in interval)
        cmd = [
            str(MOLECFIT_ESOREX),
            f"--output-dir={staged_out}",
            "molecfit_model",
            "--LIST_MOLEC=H2O",
            "--FIT_MOLEC=1",
            "--REL_COL=1.0",
            f"--WAVE_INCLUDE={NAD_FIT[0]},{NAD_FIT[1]}",
            f"--WAVE_EXCLUDE={wave_exclude}",
            f"--WAVELENGTH_FRAME={wavelength_frame}",
            *rv_args,
            "--COLUMN_LAMBDA=lambda",
            "--COLUMN_FLUX=flux",
            "--WLG_TO_MICRON=1.0",
            "--FIT_WLC=FALSE",
            "--WLC_CONST=0.0",
            "--FIT_RES_BOX=FALSE",
            "--FIT_RES_GAUSS=TRUE",
            "--FIT_RES_LORENTZ=TRUE",
            "--CONTINUUM_N=2",
            "--GDAS_PROFILE=auto",
            "--UTC_KEYWORD=UTC",
            "--MIRROR_TEMPERATURE_KEYWORD=ESO TEL TH M1 TEMP",
            "--SLIT_WIDTH_KEYWORD=ESO INS SLIT1 WID",
            f"--TMP_PATH={staged_tmp}",
            str(staged_sof),
        ]
        completed = subprocess.run(cmd, cwd=stage, text=True, capture_output=True, check=False)
        log_path.write_text(completed.stdout + "\n\nSTDERR:\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Molecfit failed for {crop_path.name}; see {log_path}")

        for product in staged_out.iterdir():
            if product.is_file():
                shutil.copy2(product, products_dir / product.name)
    return products_dir / "BEST_FIT_MODEL.fits"


def _molecfit_wavelength_frame_args(header: fits.Header) -> tuple[str, list[str]]:
    specs = str(header.get("SPECSYS", "")).strip().upper()
    berv = _header_float(header, "ESO DRS BERV")
    if specs == "BARYCENT" and np.isfinite(berv):
        return "AIR_RV", ["--OBS_ERF_RV_KEY=NONE", f"--OBS_ERF_RV_VALUE={berv}"]
    return "AIR", []


def _header_float(header: fits.Header, key: str) -> float:
    try:
        return float(header[key])
    except Exception:
        return np.nan


def _airmass_from_header(header: fits.Header) -> float:
    values = [
        header.get("AIRMASS"),
        header.get("ESO TEL AIRM START"),
        header.get("ESO TEL AIRM END"),
    ]
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 1.0


def _molecfit_corrected_on_raw_grid(
    best_fit_model: Path,
    raw_wavelength_air: np.ndarray,
    raw_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(best_fit_model) as hdul:
        data = hdul[1].data
        flux = np.asarray(data["flux"], dtype=float)
        transmission = np.asarray(data["mtrans"], dtype=float)
    # With WAVELENGTH_FRAME=AIR_RV, Molecfit reports a converted wavelength
    # grid, but the rows still correspond to the original input pixels.
    # Apply the correction row-by-row, then plot it on the original raw grid.
    corrected = np.full(raw_wavelength_air.shape, np.nan, dtype=float)
    transmission_on_raw = np.full(raw_wavelength_air.shape, np.nan, dtype=float)
    n_pixels = min(raw_wavelength_air.size, flux.size, transmission.size, raw_flux.size)
    transmission_on_raw[:n_pixels] = transmission[:n_pixels]
    corrected[:n_pixels] = flux[:n_pixels] / np.where(transmission[:n_pixels] > 0.2, transmission[:n_pixels], np.nan)
    return corrected, transmission_on_raw


def _normalise(flux: np.ndarray) -> np.ndarray:
    finite = np.isfinite(flux)
    if not np.any(finite):
        return flux
    scale = np.nanmedian(flux[finite])
    if not np.isfinite(scale) or scale == 0:
        return flux
    return flux / scale


def _plot_nad(
    out_path: Path,
    wavelength_air: np.ndarray,
    raw_flux: np.ndarray,
    gen_result: object,
    molecfit_model: Path | None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(wavelength_air * 1.0e4, _normalise(raw_flux), color="black", lw=0.9)
    axes[0].set_ylabel("Raw norm. flux")
    axes[0].set_title("Beta Pic HARPS Na D region")

    if molecfit_model is not None:
        mol_corrected, mol_trans = _molecfit_corrected_on_raw_grid(molecfit_model, wavelength_air, raw_flux)
        reliable = mol_trans > 0.8
        axes[1].plot(wavelength_air[reliable] * 1.0e4, _normalise(mol_corrected)[reliable], color="C0", lw=0.9)
    axes[1].set_ylabel("Molecfit corrected")

    # Corrections remain row-aligned with the input spectrum even though the
    # physical fit is performed on observatory-frame vacuum wavelengths.
    gen_wave_air = wavelength_air * 1.0e4
    gen_corrected = np.asarray(gen_result.corrected.flux, dtype=float)
    gen_reliable = np.asarray(gen_result.transmission, dtype=float) > 0.8
    axes[2].plot(gen_wave_air[gen_reliable], _normalise(gen_corrected)[gen_reliable], color="C1", lw=0.9)
    axes[2].set_ylabel("PyMolFit corrected")
    axes[2].set_xlabel("Wavelength air [Å]")

    for ax in axes:
        for lower, upper in NAD_EXCLUDE:
            ax.axvspan(lower * 1.0e4, upper * 1.0e4, color="0.85", alpha=0.5, lw=0)
        ax.grid(alpha=0.25)
        ax.set_xlim(NAD_CROP[0] * 1.0e4, NAD_CROP[1] * 1.0e4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_all(spectra_dir: Path, output_dir: Path, *, skip_molecfit: bool, max_files: int | None) -> None:
    line_list, line_source = _load_h2o_lines()
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra = sorted(spectra_dir.glob("ADP*.fits"))
    if max_files is not None:
        spectra = spectra[:max_files]
    if not spectra:
        raise FileNotFoundError(f"No ADP*.fits files found in {spectra_dir}")

    rows: list[dict[str, str | float | bool | int]] = []
    for source in spectra:
        tag = source.stem.replace(".", "_").replace(":", "_")
        case_dir = output_dir / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        crop_path = case_dir / "harps_nad_crop_air.fits"
        wavelength, flux, header = _write_crop_fits(source, crop_path)
        airmass = _airmass_from_header(header)

        gen_result = _run_pymolfit(crop_path, case_dir, airmass, line_list)
        molecfit_model = None if skip_molecfit else _run_molecfit(crop_path, case_dir)
        _plot_nad(case_dir / "nad_raw_molecfit_pymolfit.png", wavelength, flux, gen_result, molecfit_model)

        rows.append(
            {
                "source": str(source),
                "case": tag,
                "date_obs": str(header.get("DATE-OBS", "")),
                "mjd_obs": float(header.get("MJD-OBS", np.nan)),
                "airmass": airmass,
                "speccys": str(header.get("SPECSYS", "")),
                "berv_km_s": _header_float(header, "ESO DRS BERV"),
                "line_source": line_source,
                "n_pixels": int(wavelength.size),
                "pymolfit_h2o_scale": float(gen_result.species_scales.get("H2O", np.nan)),
                "pymolfit_wavelength_shift_micron": float(gen_result.wavelength_shift),
                "pymolfit_lsf_sigma_pixels": float(gen_result.lsf_sigma_pixels),
                "pymolfit_lsf_lorentz_fwhm_pixels": float(gen_result.lsf_lorentz_fwhm_pixels),
                "pymolfit_min_transmission": float(np.nanmin(gen_result.transmission)),
                "pymolfit_median_transmission": float(np.nanmedian(gen_result.transmission)),
                "molecfit_ran": bool(molecfit_model),
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Table(rows=rows).write(output_dir / "summary.ecsv", overwrite=True)
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyMolFit and Molecfit around beta Pic HARPS Na D.")
    parser.add_argument("--spectra-dir", type=Path, default=SPECTRA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-molecfit", action="store_true")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    run_all(args.spectra_dir, args.output_dir, skip_molecfit=args.skip_molecfit, max_files=args.max_files)


if __name__ == "__main__":
    main()
