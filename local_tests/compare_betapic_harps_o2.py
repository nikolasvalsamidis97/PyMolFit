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
OUTPUT_DIR = Path(__file__).resolve().parent / "betapic_harps_o2_comparison"
HITRAN_O2_PAR = Path(
    os.environ.get(
        "PYMOLFIT_HITRAN_O2_PAR",
        PROJECT / "local_tests" / "data" / "hitran" / "o2.par",
    )
)
MOLECFIT_ROOT = Path.home() / ".criresflow" / "molecfit"
MOLECFIT_ESOREX = MOLECFIT_ROOT / "bin" / "esorex"
MOLECFIT_AER_LINE_DB = MOLECFIT_ROOT / "share" / "molecfit" / "data" / "hitran" / "aer_v_3.8.1.2"

O2_BBAND_CROP = (0.6867, 0.6909)
O2_BBAND_FIT = (0.6869, 0.6906)
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
    keep = (wavelength >= O2_BBAND_CROP[0]) & (wavelength <= O2_BBAND_CROP[1])
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


def _load_pymolfit_o2_lines() -> tuple[LineList | None, str]:
    wavenumber_min = 1.0e4 / 0.6915
    wavenumber_max = 1.0e4 / 0.6860
    if MOLECFIT_AER_LINE_DB.exists():
        return (
            LineList.from_aer_line_file(
                MOLECFIT_AER_LINE_DB,
                species=("O2",),
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
            ),
            str(MOLECFIT_AER_LINE_DB),
        )
    return None, str(HITRAN_O2_PAR)


def _run_pymolfit(
    crop_path: Path,
    out_dir: Path,
    airmass: float,
    line_list: LineList | None,
    initial_wavelength_shift: float | None = None,
    *,
    line_wing_mode: str = "lblrtm_dynamic",
    high_resolution_oversampling: float = 5.0,
    output_suffix: str = "",
) -> object:
    corrected_path = out_dir / f"pymolfit_corrected{output_suffix}.txt"
    product_path = out_dir / f"pymolfit_product{output_suffix}.ecsv"
    plot_path = out_dir / f"pymolfit_diagnostic{output_suffix}.png"
    line_kwargs = (
        {"line_list": line_list}
        if line_list is not None
        else {"hitran_par": HITRAN_O2_PAR, "hitran_species": ("O2",), "hitran_min_strength": 1e-32}
    )
    return correct_file(
        crop_path,
        corrected_path,
        input_format="fits",
        wavelength_col="lambda",
        flux_col="flux",
        wavelength_unit="micron",
        wavelength_medium="air",
        **line_kwargs,
        physical=True,
        atmosphere_mode="mipas_gdas",
        mipas_profile="equ",
        gdas_mode="auto",
        gdas_download_timeout_s=30.0,
        airmass=airmass,
        continuum_order=1,
        rayleigh=False,
        lsf_box_width_pixels=1.0,
        lsf_sigma_pixels=1.0,
        lsf_lorentz_fwhm_pixels=3.0,
        high_resolution_grid=True,
        high_resolution_oversampling=high_resolution_oversampling,
        high_resolution_rebin_mode="molecfit_overlap",
        line_wing_mode=line_wing_mode,
        fit_wavelength_shift=True,
        initial_wavelength_shift=initial_wavelength_shift,
        wavelength_shift_bounds=(-2.0e-4, 2.0e-4),
        lsf_kernel_width_fwhm=3.0,
        lsf_molecfit_voigt=False,
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.0, 10.0),
        fit_lsf_lorentz_fwhm=True,
        lsf_lorentz_fwhm_bounds=(0.0, 10.0),
        fit_ranges=(O2_BBAND_FIT,),
        product_path=product_path,
        plot_path=plot_path,
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

    with tempfile.TemporaryDirectory(prefix="pymolfit_harps_o2_") as tmp:
        stage = Path(tmp)
        staged_input = stage / "input.fits"
        staged_sof = stage / "model.sof"
        staged_out = stage / "out"
        staged_tmp = stage / "tmp"
        staged_out.mkdir()
        staged_tmp.mkdir()
        shutil.copy2(crop_path, staged_input)
        staged_sof.write_text(f"{staged_input} SCIENCE\n", encoding="utf-8")

        cmd = [
            str(MOLECFIT_ESOREX),
            f"--output-dir={staged_out}",
            "molecfit_model",
            "--LIST_MOLEC=O2",
            "--FIT_MOLEC=1",
            "--REL_COL=1.0",
            f"--WAVE_INCLUDE={O2_BBAND_FIT[0]},{O2_BBAND_FIT[1]}",
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
            "--CONTINUUM_N=1",
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


def _molecfit_arrays(best_fit_model: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(best_fit_model) as hdul:
        data = hdul[1].data
        data_wavelength = np.asarray(data["lambda"], dtype=float)
        model_wavelength = np.asarray(data["mlambda"], dtype=float)
        model_flux = np.asarray(data["mflux"], dtype=float)
        transmission = np.asarray(data["mtrans"], dtype=float)
        continuum = np.asarray(data["mscal"], dtype=float)
    return data_wavelength, model_wavelength, model_flux, transmission, continuum


def _normalise(flux: np.ndarray) -> np.ndarray:
    finite = np.isfinite(flux)
    if not np.any(finite):
        return flux
    scale = np.nanmedian(flux[finite])
    if not np.isfinite(scale) or scale == 0:
        return flux
    return flux / scale


def _plot_comparison(
    out_path: Path,
    wavelength: np.ndarray,
    raw_flux: np.ndarray,
    gen_result: object,
    molecfit_model: Path | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    gen_data_wave_vacuum = gen_result.spectrum.wavelength
    gen_model_wave_vacuum = gen_result.spectrum.wavelength - gen_result.wavelength_shift
    fig, axes = plt.subplots(4 if molecfit_model else 3, 1, figsize=(12, 10), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    axes[0].plot(gen_data_wave_vacuum, _normalise(gen_result.model_flux), color="C1", lw=1.0, label="PyMolFit model")
    axes[0].set_ylabel("Norm. flux")

    gen_corrected_flux = _corrected_flux_from_result(gen_result, raw_flux)
    gen_corrected_plot = _continuum_normalised_corrected_flux(gen_result, gen_corrected_flux)
    reliable_transmission = 0.8
    gen_corrected_plot = np.where(gen_result.transmission > reliable_transmission, gen_corrected_plot, np.nan)
    axes[1].plot(
        gen_data_wave_vacuum,
        gen_corrected_plot,
        color="black",
        lw=1.0,
        label=f"Telluric-corrected spectrum / continuum (T > {reliable_transmission:g})",
    )
    axes[1].set_ylabel("Corrected / continuum")
    axes[1].legend(loc="best")

    axes[2].plot(gen_model_wave_vacuum, gen_result.transmission, color="C1", lw=1.0, label="PyMolFit transmission")
    axes[2].set_ylabel("Transmission")
    axes[2].legend(loc="best")

    if molecfit_model is not None:
        mol_data_wave, mol_model_wave, mol_model_flux, mol_trans, _ = _molecfit_arrays(molecfit_model)
        use_mol = (
            np.isfinite(mol_model_wave)
            & np.isfinite(mol_trans)
            & np.isfinite(mol_model_flux)
            & (mol_trans > 0)
            & (mol_model_flux > 0)
        )
        axes[0].plot(mol_model_wave[use_mol], _normalise(mol_model_flux[use_mol]), color="C0", lw=1.0, alpha=0.8, label="Molecfit model")
        axes[2].plot(mol_model_wave[use_mol], mol_trans[use_mol], color="C0", lw=1.0, alpha=0.8, label="Molecfit transmission")
        axes[2].legend(loc="best")

        gen_trans_on_mol = np.interp(mol_model_wave[use_mol], gen_model_wave_vacuum, gen_result.transmission)
        diff = gen_trans_on_mol - mol_trans[use_mol]
        axes[3].plot(mol_model_wave[use_mol], diff, color="0.15", lw=0.8)
        axes[3].axhline(0, color="0.5", lw=0.8)
        axes[3].set_ylabel("Gen - Molecfit")
        metrics["transmission_rms"] = float(np.sqrt(np.nanmean(diff**2)))
        metrics["transmission_max_abs"] = float(np.nanmax(np.abs(diff)))
        metrics["molecfit_model_minus_data_wavelength_micron"] = float(
            np.nanmedian(mol_model_wave[use_mol] - mol_data_wave[use_mol])
        )
        metrics["pymolfit_model_minus_data_wavelength_micron"] = float(-gen_result.wavelength_shift)

    axes[0].plot(
        gen_data_wave_vacuum,
        _normalise(raw_flux),
        color="black",
        lw=1.15,
        alpha=0.9,
        label="Raw crop",
        zorder=10,
    )
    axes[0].legend(loc="best")

    axes[1].set_ylim(*_robust_ylim([gen_corrected_plot]))
    axes[-1].set_xlabel("Wavelength [vacuum micron]")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return metrics


def _corrected_flux_from_result(gen_result: object, raw_flux: np.ndarray) -> np.ndarray:
    if hasattr(gen_result, "corrected_flux"):
        return np.asarray(gen_result.corrected_flux, dtype=float)
    corrected = getattr(gen_result, "corrected", None)
    if corrected is not None and hasattr(corrected, "flux"):
        return np.asarray(corrected.flux, dtype=float)
    transmission = np.asarray(gen_result.transmission, dtype=float)
    return raw_flux / np.where(transmission > 0.03, transmission, np.nan)


def _continuum_normalised_corrected_flux(gen_result: object, corrected_flux: np.ndarray) -> np.ndarray:
    if hasattr(gen_result, "continuum"):
        continuum = np.asarray(gen_result.continuum, dtype=float)
    else:
        model_flux = np.asarray(gen_result.model_flux, dtype=float)
        transmission = np.asarray(gen_result.transmission, dtype=float)
        continuum = model_flux / np.where(transmission > 0, transmission, np.nan)
    return np.asarray(corrected_flux, dtype=float) / np.where(continuum > 0, continuum, np.nan)


def _robust_ylim(fluxes: list[np.ndarray]) -> tuple[float, float]:
    normalised = [_normalise(np.asarray(flux, dtype=float)) for flux in fluxes]
    values = np.concatenate([flux[np.isfinite(flux)] for flux in normalised if np.any(np.isfinite(flux))])
    if values.size == 0:
        return (0.0, 1.5)
    low, high = np.nanpercentile(values, [1, 99])
    padding = 0.1 * max(high - low, 1.0e-6)
    return float(low - padding), float(high + padding)


def _airmass_from_header(header: fits.Header) -> float:
    values = [
        header.get("AIRMASS"),
        header.get("ESO TEL AIRM START"),
        header.get("ESO TEL AIRM END"),
    ]
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 1.0


def run_all(spectra_dir: Path, output_dir: Path, *, skip_molecfit: bool, max_files: int | None) -> None:
    if not HITRAN_O2_PAR.exists():
        raise FileNotFoundError(f"Missing HITRAN O2 line list: {HITRAN_O2_PAR}")
    line_list, line_source = _load_pymolfit_o2_lines()
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra = sorted(spectra_dir.glob("ADP*.fits"))
    if max_files is not None:
        spectra = spectra[:max_files]
    if not spectra:
        raise FileNotFoundError(f"No ADP*.fits files found in {spectra_dir}")

    rows: list[dict[str, str | float]] = []
    for source in spectra:
        tag = source.stem.replace(".", "_").replace(":", "_")
        case_dir = output_dir / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        crop_path = case_dir / "harps_o2_bband_crop_air.fits"
        wavelength, flux, header = _write_crop_fits(source, crop_path)
        airmass = _airmass_from_header(header)
        berv = _header_float(header, "ESO DRS BERV")
        expected_berv_shift = float(np.nanmedian(wavelength) * berv / SPEED_OF_LIGHT_KM_S) if np.isfinite(berv) else np.nan
        wavelength_frame, _ = _molecfit_wavelength_frame_args(header)

        gen_result = _run_pymolfit(crop_path, case_dir, airmass, line_list)
        molecfit_model = None if skip_molecfit else _run_molecfit(crop_path, case_dir)
        metrics = _plot_comparison(case_dir / "pymolfit_vs_molecfit_o2_bband.png", wavelength, flux, gen_result, molecfit_model)

        rows.append(
            {
                "source": str(source),
                "case": tag,
                "date_obs": str(header.get("DATE-OBS", "")),
                "mjd_obs": float(header.get("MJD-OBS", np.nan)),
                "airmass": airmass,
                "speccys": str(header.get("SPECSYS", "")),
                "berv_km_s": berv,
                "expected_berv_shift_micron": expected_berv_shift,
                "molecfit_wavelength_frame": wavelength_frame,
                "pymolfit_line_source": line_source,
                "n_pixels": int(wavelength.size),
                "pymolfit_min_transmission": float(np.nanmin(gen_result.transmission)),
                "pymolfit_median_transmission": float(np.nanmedian(gen_result.transmission)),
                "pymolfit_o2_scale": float(gen_result.species_scales.get("O2", np.nan)),
                "pymolfit_rayleigh_scale": float(gen_result.species_scales.get("Rayleigh", np.nan)),
                "pymolfit_wavelength_shift_micron": float(gen_result.wavelength_shift),
                "pymolfit_lsf_sigma_pixels": float(gen_result.lsf_sigma_pixels),
                "pymolfit_lsf_lorentz_fwhm_pixels": float(gen_result.lsf_lorentz_fwhm_pixels),
                "molecfit_ran": bool(molecfit_model),
                **metrics,
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
    parser = argparse.ArgumentParser(description="Compare PyMolFit and Molecfit on beta Pic HARPS O2 B-band spectra.")
    parser.add_argument("--spectra-dir", type=Path, default=SPECTRA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-molecfit", action="store_true")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    run_all(args.spectra_dir, args.output_dir, skip_molecfit=args.skip_molecfit, max_files=args.max_files)


if __name__ == "__main__":
    main()
