from __future__ import annotations

import argparse
import csv
import hashlib
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
    fit_telluric_segments,
)
from genmolfit.aer_data import (
    AER_CATALOG_FILENAME,
    AER_CATALOG_VERSION,
    load_aer_line_window,
)

try:
    from local_tests.molecfit_reference_data import stage_aer_molecfit_data
except ModuleNotFoundError:  # Direct script execution.
    from molecfit_reference_data import stage_aer_molecfit_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "local_tests" / "science_readiness" / "data" / "uves_demo"
SOURCE = DATA_DIR / "ADP.2020-06-08T15_07_14.471.fits"
LINE_CACHE = DATA_DIR / "uves_h2o_o2_lines.fits"
OUTPUT = ROOT / "local_tests" / "uves_official_demo_comparison"
MOLECFIT_ROOT = Path.home() / ".criresflow" / "molecfit"
MOLECFIT_ESOREX = MOLECFIT_ROOT / "bin" / "esorex"
MOLECFIT_DATA_ROOT = MOLECFIT_ROOT / "share" / "molecfit" / "data"

# These are the optimized ranges published for the UVES demo spectrum in the
# ESO Molecfit Reflex tutorial 4.4.2, section 8.3.2. They were not selected
# from GenMolFit residuals.
FIT_RANGES = (
    (0.586, 0.600),
    (0.625, 0.640),
    (0.645, 0.653),
)
SPECIES = ("H2O", "O2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_float(header: fits.Header, *keys: str, default: float) -> float:
    for key in keys:
        value = header.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return default


def _representative_airmass(header: fits.Header) -> float:
    start = _header_float(header, "ESO TEL AIRM START", default=np.nan)
    end = _header_float(header, "ESO TEL AIRM END", default=start)
    if np.isfinite(start) and np.isfinite(end):
        return 0.5 * (start + end)
    altitude = _header_float(header, "ESO TEL ALT", default=90.0)
    return float(1.0 / np.sin(np.deg2rad(altitude)))


def _load_source() -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    with fits.open(SOURCE, memmap=True) as hdul:
        header = hdul[0].header.copy()
        data = hdul[1].data
        wavelength = np.asarray(data["WAVE"][0], dtype=float) * 1.0e-4
        flux = np.asarray(data["FLUX_REDUCED"][0], dtype=float)
        uncertainty = np.asarray(data["ERR_REDUCED"][0], dtype=float)
    order = np.argsort(wavelength)
    return wavelength[order], flux[order], uncertainty[order], header


def _range_masks(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
) -> list[np.ndarray]:
    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )
    return [valid & (wavelength >= lower) & (wavelength <= upper) for lower, upper in FIT_RANGES]


def _write_input(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    header: fits.Header,
    masks: list[np.ndarray],
) -> Path:
    selected = np.logical_or.reduce(masks)
    columns = [
        fits.Column(name="lambda", array=wavelength[selected], format="D", unit="um"),
        fits.Column(name="flux", array=flux[selected], format="D", unit="adu"),
        fits.Column(name="dflux", array=uncertainty[selected], format="D", unit="adu"),
    ]
    path = OUTPUT / "uves_demo_fit_regions_air.fits"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.BinTableHDU.from_columns(columns, name="SCIENCE"),
        ]
    ).writeto(path, overwrite=True)
    return path


def _load_lines() -> LineList:
    lower = min(value[0] for value in FIT_RANGES) - 0.003
    upper = max(value[1] for value in FIT_RANGES) + 0.003
    artifact = load_aer_line_window(
        wavelength_min_micron=lower,
        wavelength_max_micron=upper,
        species=SPECIES,
    )
    lines = artifact.line_list
    lines.write(LINE_CACHE, format="fits")
    return lines


def _build_atmosphere(
    header: fits.Header,
    spectra_vacuum: list[Spectrum],
    *,
    gdas_profile: str | Path | None = None,
) -> AtmosphereProfile:
    reference_wavenumber = float(
        np.nanmedian(
            np.concatenate(
                [1.0e4 / spectrum.wavelength for spectrum in spectra_vacuum]
            )
        )
    )
    return AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        airmass=_representative_airmass(header),
        observatory_altitude_m=_header_float(
            header,
            "ESO TEL GEOELEV",
            default=2648.0,
        ),
        gdas_profile=gdas_profile,
        gdas_mode="auto",
        reference_wavenumber_cm=reference_wavenumber,
    )


def _run_genmolfit(
    wavelength_air: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    header: fits.Header,
    masks: list[np.ndarray],
    line_list: LineList,
    *,
    gdas_profile: str | Path | None = None,
    atmosphere: AtmosphereProfile | None = None,
):
    spectra_air = [
        Spectrum(
            wavelength=wavelength_air[mask],
            flux=flux[mask],
            uncertainty=uncertainty[mask],
            wavelength_unit="micron",
            wavelength_medium="air",
        )
        for mask in masks
    ]
    spectra_vacuum = [spectrum.to_vacuum() for spectrum in spectra_air]
    if atmosphere is None:
        atmosphere = _build_atmosphere(
            header,
            spectra_vacuum,
            gdas_profile=gdas_profile,
        )
    started = time.perf_counter()
    result = fit_telluric_segments(
        spectra_vacuum,
        line_list=line_list,
        config=FitConfig(
            species=SPECIES,
            atmosphere=atmosphere,
            partition_table=PartitionTable.from_lblrtm_package_data(),
            continuum_order=2,
            solve_continuum_linear=True,
            fit_wavelength_polynomial=True,
            wavelength_polynomial_order=1,
            initial_wavelength_shift=0.0,
            wavelength_shift_bounds=(-5.0e-5, 5.0e-5),
            lsf_box_width_pixels=0.0,
            fit_lsf_box_width=False,
            lsf_sigma_pixels=1.0 / 2.354820045,
            fit_lsf_sigma=True,
            lsf_sigma_bounds=(0.0, 4.0),
            lsf_lorentz_fwhm_pixels=0.0,
            fit_lsf_lorentz_fwhm=False,
            high_resolution_grid=True,
            high_resolution_oversampling=5.0,
            high_resolution_rebin_mode="molecfit_overlap",
            line_wing_mode="lblrtm_panel",
            lsf_kernel_width_fwhm=3.0,
            loss="linear",
            scale_bounds=(1.0e-3, 1.0e3),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
            estimate_uncertainties=True,
        ),
    )
    return result, atmosphere, time.perf_counter() - started


def _run_molecfit(
    input_path: Path,
    *,
    force: bool,
    gdas_profile: str | Path | None = None,
) -> tuple[Path, float]:
    model_path = OUTPUT / "BEST_FIT_MODEL.fits"
    parameters_path = OUTPUT / "BEST_FIT_PARAMETERS.fits"
    if model_path.exists() and parameters_path.exists() and not force:
        summary_path = OUTPUT / "summary.csv"
        cache_current = False
        if summary_path.exists():
            with summary_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle), {})
            try:
                previous_seconds = float(row["molecfit_seconds"])
            except (KeyError, TypeError, ValueError):
                previous_seconds = np.nan
            cache_current = row.get("molecfit_line_catalog_version") == AER_CATALOG_VERSION
        else:
            previous_seconds = np.nan
        if cache_current:
            return model_path, previous_seconds
    if not MOLECFIT_ESOREX.exists():
        raise FileNotFoundError(f"Molecfit executable not found: {MOLECFIT_ESOREX}")

    with tempfile.TemporaryDirectory(prefix="genmolfit_uves_demo_") as temporary:
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
        ranges = ",".join(str(value) for interval in FIT_RANGES for value in interval)
        command = [
            str(MOLECFIT_ESOREX),
            f"--output-dir={staged_output}",
            "molecfit_model",
            f"--TELLURICCORR_DATA_PATH={molecfit_data}",
            f"--LNFL_LINE_DB={AER_CATALOG_FILENAME}",
            f"--LIST_MOLEC={','.join(SPECIES)}",
            "--FIT_MOLEC=1,1",
            "--REL_COL=1.0,1.0",
            f"--WAVE_INCLUDE={ranges}",
            "--MAP_REGIONS_TO_CHIP=1,1,1",
            "--WAVELENGTH_FRAME=AIR",
            "--COLUMN_LAMBDA=lambda",
            "--COLUMN_FLUX=flux",
            "--COLUMN_DFLUX=dflux",
            "--WLG_TO_MICRON=1.0",
            "--FIT_CONTINUUM=1,1,1",
            "--CONTINUUM_N=2,2,2",
            "--FIT_WLC=1,1,1",
            "--WLC_N=1",
            "--WLC_CONST=0.0",
            "--FIT_RES_BOX=FALSE",
            "--RES_BOX=0.0",
            "--FIT_RES_GAUSS=TRUE",
            "--RES_GAUSS=1.0",
            "--FIT_RES_LORENTZ=FALSE",
            "--RES_LORENTZ=0.0",
            "--KERNMODE=FALSE",
            "--KERNFAC=3.0",
            f"--GDAS_PROFILE={'auto' if gdas_profile is None else Path(gdas_profile).resolve()}",
            "--UTC_KEYWORD=UTC",
            "--SLIT_WIDTH_KEYWORD=ESO INS SLIT3 WID",
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
            raise RuntimeError(f"Molecfit UVES demo failed; see {OUTPUT / 'molecfit.log'}")
        for product in staged_output.iterdir():
            if product.is_file():
                shutil.copy2(product, OUTPUT / product.name)
    return model_path, elapsed


def _load_molecfit_model(path: Path) -> dict[str, np.ndarray]:
    with fits.open(path) as hdul:
        data = hdul[1].data
        return {
            name: np.asarray(data[name], dtype=float)
            for name in ("lambda", "flux", "dflux", "mflux", "mtrans")
            if name in data.names
        }


def _concatenate_genmolfit(result) -> dict[str, np.ndarray]:
    return {
        "wavelength_vacuum": np.concatenate(
            [segment.spectrum.wavelength for segment in result.segment_results]
        ),
        "flux": np.concatenate([segment.spectrum.flux for segment in result.segment_results]),
        "uncertainty": np.concatenate(
            [segment.spectrum.uncertainty for segment in result.segment_results]
        ),
        "model_flux": np.concatenate(
            [segment.model_flux for segment in result.segment_results]
        ),
        "continuum": np.concatenate(
            [segment.continuum for segment in result.segment_results]
        ),
        "transmission": np.concatenate(
            [segment.transmission for segment in result.segment_results]
        ),
        "corrected": np.concatenate(
            [segment.corrected.flux for segment in result.segment_results]
        ),
    }


def _weighted_objective(flux: np.ndarray, model: np.ndarray, uncertainty: np.ndarray) -> float:
    valid = np.isfinite(flux) & np.isfinite(model) & np.isfinite(uncertainty) & (uncertainty > 0)
    return float(np.sum(np.square((flux[valid] - model[valid]) / uncertainty[valid])))


def _write_outputs(
    wavelength_air: np.ndarray,
    masks: list[np.ndarray],
    gen_result,
    atmosphere: AtmosphereProfile,
    gen_seconds: float,
    molecfit_model: dict[str, np.ndarray],
    molecfit_seconds: float,
    line_list: LineList,
) -> dict[str, float]:
    wavelength_fit_air = np.concatenate([wavelength_air[mask] for mask in masks])
    gen = _concatenate_genmolfit(gen_result)
    n_pixels = min(
        wavelength_fit_air.size,
        *(values.size for values in gen.values()),
        *(values.size for values in molecfit_model.values()),
    )
    wavelength_fit_air = wavelength_fit_air[:n_pixels]
    gen = {name: values[:n_pixels] for name, values in gen.items()}
    mol = {name: values[:n_pixels] for name, values in molecfit_model.items()}

    reliable = (
        np.isfinite(gen["transmission"])
        & np.isfinite(mol["mtrans"])
        & (gen["transmission"] > 0.2)
        & (mol["mtrans"] > 0.2)
    )
    telluric = reliable & (
        (gen["transmission"] < 0.995) | (mol["mtrans"] < 0.995)
    )
    delta = gen["transmission"] - mol["mtrans"]
    direct_rms = float(np.sqrt(np.mean(np.square(delta[reliable]))))
    telluric_rms = float(np.sqrt(np.mean(np.square(delta[telluric]))))
    direct_max = float(np.max(np.abs(delta[reliable])))
    gen_objective = _weighted_objective(gen["flux"], gen["model_flux"], gen["uncertainty"])
    mol_objective = _weighted_objective(gen["flux"], mol["mflux"], gen["uncertainty"])

    gen_relative = gen["flux"] / gen["model_flux"] - 1.0
    mol_relative = gen["flux"] / mol["mflux"] - 1.0
    gen_scatter = float(np.nanstd(gen_relative[reliable]))
    mol_scatter = float(np.nanstd(mol_relative[reliable]))

    mol_continuum = mol["mflux"] / np.where(mol["mtrans"] > 0, mol["mtrans"], np.nan)
    mol_corrected = gen["flux"] / np.where(mol["mtrans"] > 0.03, mol["mtrans"], np.nan)
    table = Table()
    table["wavelength_air_micron"] = wavelength_fit_air
    table["wavelength_vacuum_micron"] = gen["wavelength_vacuum"]
    table["flux"] = gen["flux"]
    table["uncertainty"] = gen["uncertainty"]
    table["genmolfit_model_flux"] = gen["model_flux"]
    table["molecfit_model_flux"] = mol["mflux"]
    table["genmolfit_continuum"] = gen["continuum"]
    table["molecfit_continuum"] = mol_continuum
    table["genmolfit_transmission"] = gen["transmission"]
    table["molecfit_transmission"] = mol["mtrans"]
    table["genmolfit_corrected"] = gen["corrected"]
    table["molecfit_corrected"] = mol_corrected
    table["reliable"] = reliable
    table["telluric"] = telluric
    table.meta.update(
        {
            "source": SOURCE.name,
            "source_sha256": _sha256(SOURCE),
            "line_cache_sha256": _sha256(LINE_CACHE),
            "line_catalog_version": str(
                line_list.data_provenance.get("catalog_version", "")
            ),
            "line_catalog_sha256": str(
                line_list.data_provenance.get("catalog_sha256", "")
            ),
            "fit_ranges_micron": repr(FIT_RANGES),
            "species": repr(SPECIES),
            "genmolfit_species_scales": repr(gen_result.species_scales),
            "genmolfit_lsf_sigma_pixels": float(gen_result.lsf_sigma_pixels),
            "genmolfit_lsf_box_width_pixels": float(gen_result.lsf_box_width_pixels),
            "genmolfit_lsf_lorentz_fwhm_pixels": float(gen_result.lsf_lorentz_fwhm_pixels),
            "gdas_source": str(atmosphere.metadata.get("gdas_source", "")),
            "gdas_profile": str(atmosphere.metadata.get("gdas_profile", "")),
        }
    )
    table.write(OUTPUT / "comparison.ecsv", overwrite=True)

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
        "transmission_rms": direct_rms,
        "telluric_transmission_rms": telluric_rms,
        "transmission_max_abs": direct_max,
        "genmolfit_weighted_objective": gen_objective,
        "molecfit_weighted_objective": mol_objective,
        "weighted_objective_ratio": gen_objective / mol_objective,
        "genmolfit_relative_scatter": gen_scatter,
        "molecfit_relative_scatter": mol_scatter,
        "relative_scatter_ratio": gen_scatter / mol_scatter,
        "genmolfit_seconds": gen_seconds,
        "molecfit_seconds": molecfit_seconds,
        "genmolfit_nfev": float(gen_result.nfev),
        "genmolfit_covariance_rank": float(gen_result.covariance_rank),
        "genmolfit_parameter_count": float(len(gen_result.parameter_names)),
    }
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for range_index, (lower, upper) in enumerate(FIT_RANGES):
        region = (wavelength_fit_air >= lower) & (wavelength_fit_air <= upper)
        label_suffix = range_index == 0
        axes[0].plot(
            wavelength_fit_air[region],
            (gen["flux"] / gen["continuum"])[region],
            color="0.35",
            lw=0.65,
            label="Input / Gen continuum" if label_suffix else None,
        )
        axes[0].plot(
            wavelength_fit_air[region],
            gen["transmission"][region],
            color="C1",
            lw=0.9,
            label="GenMolFit" if label_suffix else None,
        )
        axes[0].plot(
            wavelength_fit_air[region],
            mol["mtrans"][region],
            color="C0",
            lw=0.8,
            alpha=0.85,
            label="Molecfit" if label_suffix else None,
        )
        axes[1].plot(
            wavelength_fit_air[region],
            gen_relative[region],
            color="C1",
            lw=0.65,
            label="GenMolFit" if label_suffix else None,
        )
        axes[1].plot(
            wavelength_fit_air[region],
            mol_relative[region],
            color="C0",
            lw=0.65,
            alpha=0.8,
            label="Molecfit" if label_suffix else None,
        )
        axes[2].plot(wavelength_fit_air[region], delta[region], color="C3", lw=0.7)

    axes[0].set_ylabel("Normalized flux / T")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].axhline(0.0, color="black", lw=0.6)
    axes[1].set_ylabel("(data-model)/model")
    axes[1].legend(fontsize=8)

    axes[2].axhline(0.0, color="black", lw=0.6)
    axes[2].set_ylabel("Gen T - Molecfit T")
    axes[2].set_xlabel("Air wavelength [micron]")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Official ESO UVES Molecfit demo: independently published fit regions")
    fig.tight_layout()
    fig.savefig(OUTPUT / "uves_demo_comparison.png", dpi=170)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-molecfit", action="store_true")
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(f"Download the official UVES demo spectrum to {SOURCE}")

    wavelength, flux, uncertainty, header = _load_source()
    masks = _range_masks(wavelength, flux, uncertainty)
    if any(np.count_nonzero(mask) < 100 for mask in masks):
        raise RuntimeError("One or more published UVES fit regions contain too few valid pixels")
    input_path = _write_input(wavelength, flux, uncertainty, header, masks)
    line_list = _load_lines()
    gen_result, atmosphere, gen_seconds = _run_genmolfit(
        wavelength,
        flux,
        uncertainty,
        header,
        masks,
        line_list,
    )
    molecfit_path, molecfit_seconds = _run_molecfit(
        input_path,
        force=args.force_molecfit,
    )
    metrics = _write_outputs(
        wavelength,
        masks,
        gen_result,
        atmosphere,
        gen_seconds,
        _load_molecfit_model(molecfit_path),
        molecfit_seconds,
        line_list,
    )
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
