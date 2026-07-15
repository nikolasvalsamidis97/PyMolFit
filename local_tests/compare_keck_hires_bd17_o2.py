from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from genmolfit import (
    AtmosphereProfile,
    FitConfig,
    LineList,
    PartitionTable,
    Spectrum,
    cross_validate_telluric_segments,
    fit_telluric_segments,
)
from genmolfit.aer_data import (
    AER_CATALOG_FILENAME,
    AER_CATALOG_VERSION,
    load_aer_line_window,
)
from genmolfit.io import load_spectrum
from genmolfit.workflow import _spectrum_to_observatory_vacuum

try:
    from local_tests.molecfit_reference_data import stage_aer_molecfit_data
except ModuleNotFoundError:  # Direct script execution.
    from molecfit_reference_data import stage_aer_molecfit_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "local_tests" / "science_readiness" / "data" / "keck_hires_bd17"
OUTPUT = ROOT / "local_tests" / "keck_hires_bd17_o2_comparison"
LINE_CACHE = DATA_DIR / "hires_o2_lines.fits"
MOLECFIT_ROOT = Path.home() / ".criresflow" / "molecfit"
MOLECFIT_ESOREX = MOLECFIT_ROOT / "bin" / "esorex"
MOLECFIT_DATA_ROOT = MOLECFIT_ROOT / "share" / "molecfit" / "data"

ORDER_PATHS = (
    DATA_DIR / "binaryfits" / "ccd3" / "flux" / "HI.20040824.18925_3_04_flux.fits.gz",
    DATA_DIR / "binaryfits" / "ccd3" / "flux" / "HI.20040824.18925_3_09_flux.fits.gz",
)
# Established O2 B- and A-band intervals, chosen before examining residuals.
FIT_RANGES_VACUUM_MICRON = (
    (0.6868, 0.6905),
    (0.7592, 0.76515),
)
SPECIES = ("O2",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_float(header: fits.Header, *keys: str, default: float = np.nan) -> float:
    for key in keys:
        try:
            value = float(header[key])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _load_segments() -> tuple[list[Spectrum], fits.Header]:
    segments: list[Spectrum] = []
    header: fits.Header | None = None
    for path, (lower, upper) in zip(ORDER_PATHS, FIT_RANGES_VACUUM_MICRON, strict=True):
        with fits.open(path) as hdul:
            current_header = hdul[0].header.copy()
        if header is None:
            header = current_header
        loaded = load_spectrum(
            path,
            wavelength_col="wave",
            flux_col="Flux",
            uncertainty_col="Error",
            wavelength_unit="angstrom",
            wavelength_medium="vacuum",
        )
        observatory = _spectrum_to_observatory_vacuum(loaded, current_header).to_unit("micron")
        selected = (
            observatory.valid
            & np.isfinite(observatory.uncertainty)
            & (observatory.uncertainty > 0)
            & (observatory.wavelength >= lower)
            & (observatory.wavelength <= upper)
        )
        if np.count_nonzero(selected) < 500:
            raise RuntimeError(f"Too few valid pixels in {path.name}")
        segments.append(
            Spectrum(
                wavelength=observatory.wavelength[selected],
                flux=observatory.flux[selected],
                uncertainty=observatory.uncertainty[selected],
                wavelength_unit="micron",
                wavelength_medium="vacuum",
                meta=observatory.meta,
            )
        )
    if header is None:
        raise RuntimeError("No HIRES order was loaded")
    return segments, header


def _load_lines() -> LineList:
    lower = min(interval[0] for interval in FIT_RANGES_VACUUM_MICRON) - 0.003
    upper = max(interval[1] for interval in FIT_RANGES_VACUUM_MICRON) + 0.003
    artifact = load_aer_line_window(
        wavelength_min_micron=lower,
        wavelength_max_micron=upper,
        species=SPECIES,
    )
    lines = artifact.line_list
    lines.write(LINE_CACHE, format="fits")
    return lines


def _build_atmosphere(header: fits.Header, segments: list[Spectrum]) -> AtmosphereProfile:
    reference_wavenumber = float(
        np.nanmedian(np.concatenate([1.0e4 / segment.wavelength for segment in segments]))
    )
    return AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        gdas_mode="auto",
        gdas_download_timeout_s=30.0,
        reference_wavenumber_cm=reference_wavenumber,
    )


def _run_genmolfit(
    segments: list[Spectrum],
    header: fits.Header,
    line_list: LineList,
):
    atmosphere = _build_atmosphere(header, segments)
    config = _genmolfit_config(atmosphere)
    started = time.perf_counter()
    result = fit_telluric_segments(segments, line_list=line_list, config=config)
    elapsed = time.perf_counter() - started
    return result, atmosphere, config, elapsed


def _genmolfit_config(atmosphere: AtmosphereProfile) -> FitConfig:
    return FitConfig(
        species=SPECIES,
        atmosphere=atmosphere,
        partition_table=PartitionTable.from_lblrtm_package_data(),
        continuum_order=3,
        solve_continuum_linear=True,
        fit_wavelength_polynomial=True,
        wavelength_polynomial_order=1,
        wavelength_shift_bounds=(-5.0e-5, 5.0e-5),
        lsf_sigma_pixels=0.9,
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.3, 2.0),
        high_resolution_grid=True,
        high_resolution_oversampling=5.0,
        high_resolution_rebin_mode="molecfit_overlap",
        line_wing_mode="lblrtm_panel",
        lsf_kernel_width_fwhm=4.0,
        loss="linear",
        scale_bounds=(0.2, 5.0),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        estimate_uncertainties=True,
    )


def _run_genmolfit_cross_validation(
    segments: list[Spectrum],
    line_list: LineList,
    config: FitConfig,
):
    started = time.perf_counter()
    result = cross_validate_telluric_segments(
        segments,
        line_list=line_list,
        config=replace(config, estimate_uncertainties=False),
        block_size=64,
        n_folds=2,
    )
    elapsed = time.perf_counter() - started
    result.write(OUTPUT / "cross_validation", prefix="hires_segment")
    return result, elapsed


def _sanitized_molecfit_header(source: fits.Header) -> fits.Header:
    utc = str(source.get("UTC", "05:15:25.65"))
    hours, minutes, seconds = (float(value) for value in utc.split(":"))
    utc_seconds = hours * 3600.0 + minutes * 60.0 + seconds
    header = fits.Header()
    header["OBJECT"] = str(source.get("OBJECT", "BD+17 3248"))
    header["INSTRUME"] = "HIRES"
    header["TELESCOP"] = "Keck I"
    header["DATE-OBS"] = f"{source.get('DATE-OBS', '2004-08-24')}T{utc}"
    header["MJD-OBS"] = _header_float(source, "MJD", default=53241.219065)
    header["UTC"] = utc_seconds
    header["AIRMASS"] = _header_float(source, "AIRMASS", default=1.01)
    header["HIERARCH ESO TEL AIRM START"] = header["AIRMASS"]
    header["HIERARCH ESO TEL AIRM END"] = header["AIRMASS"]
    header["HIERARCH ESO TEL ALT"] = float(
        np.rad2deg(np.arcsin(np.clip(1.0 / header["AIRMASS"], 0.0, 1.0)))
    )
    header["HIERARCH ESO TEL GEOLAT"] = _header_float(source, "LATITUDE", default=19.82658656)
    header["HIERARCH ESO TEL GEOLON"] = -abs(
        _header_float(source, "LONGITUD", default=155.4722)
    )
    header["HIERARCH ESO TEL GEOELEV"] = 4145.0
    header["HIERARCH ESO TEL AMBI PRES START"] = _header_float(
        source,
        "WXPRESS",
        default=623.3,
    )
    header["HIERARCH ESO TEL AMBI TEMP"] = _header_float(
        source,
        "WXOUTTMP",
        "WXDOMTMP",
        default=5.23,
    )
    header["HIERARCH ESO TEL AMBI RHUM"] = _header_float(
        source,
        "RELHUM",
        "WXOUTHUM",
        default=11.16,
    )
    header["HIERARCH ESO INS SLIT1 WID"] = _header_float(source, "SLITWIDT", default=0.4)
    header["HIERARCH ESO TEL TH M1 TEMP"] = header["HIERARCH ESO TEL AMBI TEMP"]
    header["SPECSYS"] = "TOPOCENT"
    return header


def _write_molecfit_input(segments: list[Spectrum], header: fits.Header) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wavelength = np.concatenate([segment.wavelength for segment in segments])
    flux = np.concatenate([segment.flux for segment in segments])
    uncertainty = np.concatenate([segment.uncertainty for segment in segments])
    columns = [
        fits.Column(name="lambda", array=wavelength, format="D", unit="um"),
        fits.Column(name="flux", array=flux, format="D", unit="count"),
        fits.Column(name="dflux", array=uncertainty, format="D", unit="count"),
    ]
    path = OUTPUT / "hires_bd17_o2_topocentric_vacuum.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=_sanitized_molecfit_header(header)),
            fits.BinTableHDU.from_columns(columns, name="SCIENCE"),
        ]
    ).writeto(path, overwrite=True)
    return path


def _run_molecfit(input_path: Path, *, force: bool) -> tuple[Path, float]:
    model_path = OUTPUT / "BEST_FIT_MODEL.fits"
    if model_path.exists() and not force:
        previous = OUTPUT / "summary.csv"
        if previous.exists():
            with previous.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle), {})
            if row.get("molecfit_line_catalog_version") == AER_CATALOG_VERSION:
                try:
                    return model_path, float(row["molecfit_seconds"])
                except (KeyError, TypeError, ValueError):
                    return model_path, np.nan
    if not MOLECFIT_ESOREX.exists():
        raise FileNotFoundError(f"Molecfit executable not found: {MOLECFIT_ESOREX}")

    with tempfile.TemporaryDirectory(prefix="genmolfit_hires_bd17_") as temporary:
        stage = Path(temporary)
        staged_input = stage / "input.fits"
        staged_output = stage / "out"
        staged_tmp = stage / "tmp"
        staged_output.mkdir()
        staged_tmp.mkdir()
        molecfit_data, _ = stage_aer_molecfit_data(
            MOLECFIT_DATA_ROOT,
            stage / "molecfit-data",
        )
        shutil.copy2(input_path, staged_input)
        sof = stage / "model.sof"
        sof.write_text(f"{staged_input} SCIENCE\n", encoding="utf-8")
        ranges = ",".join(
            str(value) for interval in FIT_RANGES_VACUUM_MICRON for value in interval
        )
        command = [
            str(MOLECFIT_ESOREX),
            f"--output-dir={staged_output}",
            "molecfit_model",
            f"--TELLURICCORR_DATA_PATH={molecfit_data}",
            f"--LNFL_LINE_DB={AER_CATALOG_FILENAME}",
            "--LIST_MOLEC=O2",
            "--FIT_MOLEC=1",
            "--REL_COL=1.0",
            f"--WAVE_INCLUDE={ranges}",
            "--MAP_REGIONS_TO_CHIP=1,2",
            "--WAVELENGTH_FRAME=VAC",
            "--COLUMN_LAMBDA=lambda",
            "--COLUMN_FLUX=flux",
            "--COLUMN_DFLUX=dflux",
            "--WLG_TO_MICRON=1.0",
            "--FIT_CONTINUUM=1,1",
            "--CONTINUUM_N=3,3",
            "--FIT_WLC=1,1",
            "--WLC_N=1",
            "--WLC_CONST=0.0",
            "--FIT_RES_BOX=FALSE",
            "--RES_BOX=0.0",
            "--FIT_RES_GAUSS=TRUE",
            "--RES_GAUSS=2.1",
            "--FIT_RES_LORENTZ=FALSE",
            "--RES_LORENTZ=0.0",
            "--KERNMODE=FALSE",
            "--KERNFAC=4.0",
            "--GDAS_PROFILE=auto",
            "--UTC_KEYWORD=UTC",
            "--SLIT_WIDTH_KEYWORD=ESO INS SLIT1 WID",
            "--MIRROR_TEMPERATURE_KEYWORD=ESO TEL TH M1 TEMP",
            "--FTOL=1e-10",
            "--XTOL=1e-10",
            f"--TMP_PATH={staged_tmp}",
            str(sof),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=stage,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        (OUTPUT / "molecfit.log").write_text(
            completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Molecfit HIRES fit failed; see {OUTPUT / 'molecfit.log'}")
        for product in staged_output.iterdir():
            if product.is_file():
                shutil.copy2(product, OUTPUT / product.name)
    return model_path, elapsed


def _load_molecfit_model(path: Path) -> dict[str, np.ndarray]:
    with fits.open(path) as hdul:
        data = hdul[1].data
        return {
            name: np.asarray(data[name], dtype=float)
            for name in ("lambda", "flux", "mflux", "mtrans")
        }


def _weighted_objective(flux: np.ndarray, model: np.ndarray, error: np.ndarray) -> float:
    valid = np.isfinite(flux) & np.isfinite(model) & np.isfinite(error) & (error > 0)
    return float(np.sum(np.square((flux[valid] - model[valid]) / error[valid])))


def _write_outputs(
    result,
    cross_validation,
    atmosphere: AtmosphereProfile,
    genmolfit_seconds: float,
    cross_validation_seconds: float,
    molecfit: dict[str, np.ndarray],
    molecfit_seconds: float,
    line_list: LineList,
) -> dict[str, float]:
    for index, segment in enumerate(result.segment_results, 1):
        segment.write(OUTPUT / f"genmolfit_segment_{index}.ecsv")

    wavelength = np.concatenate([segment.spectrum.wavelength for segment in result.segment_results])
    flux = np.concatenate([segment.spectrum.flux for segment in result.segment_results])
    error = np.concatenate([segment.spectrum.uncertainty for segment in result.segment_results])
    gen_model = np.concatenate([segment.model_flux for segment in result.segment_results])
    gen_continuum = np.concatenate([segment.continuum for segment in result.segment_results])
    gen_transmission = np.concatenate([segment.transmission for segment in result.segment_results])
    n_pixels = min(wavelength.size, *(values.size for values in molecfit.values()))
    wavelength = wavelength[:n_pixels]
    flux = flux[:n_pixels]
    error = error[:n_pixels]
    gen_model = gen_model[:n_pixels]
    gen_continuum = gen_continuum[:n_pixels]
    gen_transmission = gen_transmission[:n_pixels]
    molecule = {name: values[:n_pixels] for name, values in molecfit.items()}

    mol_continuum = molecule["mflux"] / np.where(
        molecule["mtrans"] > 0,
        molecule["mtrans"],
        np.nan,
    )
    raw_gen_normalized = flux / gen_continuum
    raw_mol_normalized = flux / mol_continuum
    gen_corrected = raw_gen_normalized / np.where(gen_transmission > 0.2, gen_transmission, np.nan)
    mol_corrected = raw_mol_normalized / np.where(molecule["mtrans"] > 0.2, molecule["mtrans"], np.nan)
    reliable = (
        np.isfinite(gen_transmission)
        & np.isfinite(molecule["mtrans"])
        & (gen_transmission > 0.2)
        & (molecule["mtrans"] > 0.2)
    )
    telluric = reliable & (
        (gen_transmission < 0.995) | (molecule["mtrans"] < 0.995)
    )
    quiet = reliable & (gen_transmission > 0.999) & (molecule["mtrans"] > 0.999)
    delta = gen_transmission - molecule["mtrans"]

    table = Table()
    table["wavelength_vacuum_micron"] = wavelength
    table["flux"] = flux
    table["uncertainty"] = error
    table["genmolfit_model_flux"] = gen_model
    table["molecfit_model_flux"] = molecule["mflux"]
    table["genmolfit_continuum"] = gen_continuum
    table["molecfit_continuum"] = mol_continuum
    table["genmolfit_transmission"] = gen_transmission
    table["molecfit_transmission"] = molecule["mtrans"]
    table["genmolfit_corrected_normalized"] = gen_corrected
    table["molecfit_corrected_normalized"] = mol_corrected
    table["reliable"] = reliable
    table["telluric"] = telluric
    table.meta.update(
        {
            "source_orders": json.dumps([path.name for path in ORDER_PATHS]),
            "source_sha256": json.dumps([_sha256(path) for path in ORDER_PATHS]),
            "line_cache_sha256": _sha256(LINE_CACHE),
            "line_catalog_version": str(
                line_list.data_provenance.get("catalog_version", "")
            ),
            "line_catalog_sha256": str(
                line_list.data_provenance.get("catalog_sha256", "")
            ),
            "fit_ranges_vacuum_micron": repr(FIT_RANGES_VACUUM_MICRON),
            "gdas_source": str(atmosphere.metadata.get("gdas_source")),
            "observation_time_utc": str(atmosphere.metadata.get("observation_time_utc")),
            "latitude_deg": atmosphere.metadata.get("latitude_deg"),
            "longitude_deg": atmosphere.metadata.get("longitude_deg"),
            "observatory_altitude_m": atmosphere.metadata.get("observatory_altitude_m"),
        }
    )
    table.write(OUTPUT / "comparison.ecsv", overwrite=True)

    transmission_rms = float(np.sqrt(np.mean(np.square(delta[reliable]))))
    telluric_transmission_rms = float(np.sqrt(np.mean(np.square(delta[telluric]))))
    metrics = {
        "n_pixels": float(n_pixels),
        "n_lines": float(line_list.wavelength.size),
        "line_catalog_version": str(
            line_list.data_provenance.get("catalog_version", "")
        ),
        "line_catalog_sha256": str(
            line_list.data_provenance.get("catalog_sha256", "")
        ),
        "molecfit_line_catalog_version": AER_CATALOG_VERSION,
        "n_telluric_pixels": float(np.count_nonzero(telluric)),
        "transmission_rms": transmission_rms,
        "telluric_transmission_rms": telluric_transmission_rms,
        "transmission_max_abs": float(np.max(np.abs(delta[reliable]))),
        "genmolfit_weighted_objective": _weighted_objective(flux, gen_model, error),
        "molecfit_weighted_objective": _weighted_objective(flux, molecule["mflux"], error),
        "weighted_objective_ratio": (
            _weighted_objective(flux, gen_model, error)
            / _weighted_objective(flux, molecule["mflux"], error)
        ),
        "raw_telluric_rms_from_unity": float(
            np.sqrt(np.nanmean(np.square(raw_gen_normalized[telluric] - 1.0)))
        ),
        "genmolfit_telluric_rms_from_unity": float(
            np.sqrt(np.nanmean(np.square(gen_corrected[telluric] - 1.0)))
        ),
        "molecfit_telluric_rms_from_unity": float(
            np.sqrt(np.nanmean(np.square(mol_corrected[telluric] - 1.0)))
        ),
        "genmolfit_quiet_rms_from_unity": float(
            np.sqrt(np.nanmean(np.square(gen_corrected[quiet] - 1.0)))
        ),
        "molecfit_quiet_rms_from_unity": float(
            np.sqrt(np.nanmean(np.square(mol_corrected[quiet] - 1.0)))
        ),
        "genmolfit_seconds": genmolfit_seconds,
        "genmolfit_cross_validation_seconds": cross_validation_seconds,
        "molecfit_seconds": molecfit_seconds,
        "genmolfit_nfev": float(result.nfev),
        "genmolfit_covariance_rank": float(result.covariance_rank),
        "genmolfit_parameter_count": float(len(result.parameter_names)),
        "genmolfit_o2_scale": float(result.species_scales["O2"]),
        "genmolfit_lsf_sigma_pixels": float(result.lsf_sigma_pixels),
        "cross_validation_block_size": float(cross_validation.block_size),
        "cross_validation_n_folds": float(cross_validation.n_folds),
        "cross_validation_prediction_coverage": float(
            cross_validation.metrics["prediction_coverage"]
        ),
        "cross_validation_reliable_correction_coverage": float(
            cross_validation.metrics["reliable_correction_coverage"]
        ),
        "cross_validation_telluric_pixels": float(
            cross_validation.metrics["n_telluric_pixels"]
        ),
        "cross_validation_relative_rms_improvement": float(
            cross_validation.metrics["telluric_relative_rms_improvement"]
        ),
        "cross_validation_weighted_rms_improvement": float(
            cross_validation.metrics["telluric_weighted_rms_improvement"]
        ),
        "cross_validation_all_folds_successful": float(
            cross_validation.metrics["all_folds_successful"]
        ),
    }
    metrics["genmolfit_telluric_rms_improvement"] = (
        metrics["raw_telluric_rms_from_unity"]
        / metrics["genmolfit_telluric_rms_from_unity"]
    )
    metrics["molecfit_telluric_rms_improvement"] = (
        metrics["raw_telluric_rms_from_unity"]
        / metrics["molecfit_telluric_rms_from_unity"]
    )
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex="col")
    gen_relative = np.where(gen_transmission > 0.2, flux / gen_model - 1.0, np.nan)
    mol_relative = np.where(
        molecule["mtrans"] > 0.2,
        flux / molecule["mflux"] - 1.0,
        np.nan,
    )
    start = 0
    for column, (lower, upper) in enumerate(FIT_RANGES_VACUUM_MICRON):
        count = result.segment_results[column].spectrum.wavelength.size
        region = slice(start, start + count)
        x = wavelength[region]
        axes[0, column].plot(x, raw_gen_normalized[region], color="0.35", lw=0.7, label="Input")
        axes[0, column].plot(x, gen_transmission[region], color="C1", lw=0.9, label="GenMolFit T")
        axes[0, column].plot(x, molecule["mtrans"][region], color="C0", lw=0.8, label="Molecfit T")
        axes[1, column].plot(x, gen_corrected[region], color="C1", lw=0.7, label="GenMolFit")
        axes[1, column].plot(x, mol_corrected[region], color="C0", lw=0.7, alpha=0.8, label="Molecfit")
        axes[2, column].plot(x, gen_relative[region], color="C1", lw=0.65)
        axes[2, column].plot(x, mol_relative[region], color="C0", lw=0.65, alpha=0.8)
        axes[3, column].plot(x, delta[region], color="C3", lw=0.7)
        axes[0, column].set_title("O2 B band" if column == 0 else "O2 A band")
        axes[3, column].set_xlim(lower, upper)
        start += count

    axes[0, 0].set_ylabel("Normalized flux / T")
    axes[1, 0].set_ylabel("Corrected / continuum")
    axes[2, 0].set_ylabel("Relative residual")
    axes[3, 0].set_ylabel("Gen T - Molecfit T")
    for column in range(2):
        axes[1, column].axhline(1.0, color="black", lw=0.6)
        axes[2, column].axhline(0.0, color="black", lw=0.6)
        axes[3, column].axhline(0.0, color="black", lw=0.6)
        axes[3, column].set_xlabel("Topocentric vacuum wavelength [micron]")
        for row in range(4):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=3)
    axes[1, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Keck/HIRES BD+17 3248: non-ESO O2 validation")
    fig.tight_layout()
    fig.savefig(OUTPUT / "keck_hires_bd17_o2_comparison.png", dpi=170)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-molecfit", action="store_true")
    args = parser.parse_args()
    missing = [path for path in ORDER_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing KOA HIRES validation orders: {missing}")

    segments, header = _load_segments()
    line_list = _load_lines()
    result, atmosphere, config, genmolfit_seconds = _run_genmolfit(
        segments,
        header,
        line_list,
    )
    cross_validation, cross_validation_seconds = _run_genmolfit_cross_validation(
        segments,
        line_list,
        config,
    )
    input_path = _write_molecfit_input(segments, header)
    molecfit_path, molecfit_seconds = _run_molecfit(
        input_path,
        force=args.force_molecfit,
    )
    metrics = _write_outputs(
        result,
        cross_validation,
        atmosphere,
        genmolfit_seconds,
        cross_validation_seconds,
        _load_molecfit_model(molecfit_path),
        molecfit_seconds,
        line_list,
    )
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
