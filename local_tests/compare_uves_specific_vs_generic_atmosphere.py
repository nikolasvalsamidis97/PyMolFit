from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from pymolfit import AtmosphereProfile, LineList, Spectrum

try:
    from local_tests import compare_uves_official_demo as uves
except ModuleNotFoundError:  # Direct execution places local_tests on sys.path.
    import compare_uves_official_demo as uves


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "local_tests" / "uves_specific_vs_generic_atmosphere"
SHARED_GDAS = ROOT / "local_tests" / "uves_official_demo_comparison" / "GDAS.fits"
SHARED_LINES = (
    ROOT
    / "local_tests"
    / "uves_broadspan_timed_comparison"
    / "selected_aer_lines.fits"
)
TRANSMISSION_FLOOR = 0.2
GENERIC_ALTITUDE_M = 2635.0
GENERIC_PRESSURE_ATM = 0.75
GENERIC_TEMPERATURE_K = 280.0
GENERIC_LAYER_COUNT = 36


def _stripped_input(source: Path, destination: Path) -> tuple[str, ...]:
    explicit = {
        "DATE",
        "DATE-OBS",
        "MJD-OBS",
        "MJD-END",
        "UTC",
        "LST",
        "TELESCOP",
        "OBSERVAT",
        "SITENAME",
    }
    atmospheric_prefixes = (
        "ESO TEL GEO",
        "ESO TEL AIRM",
        "ESO TEL AMBI",
        "ESO OBS",
    )
    with fits.open(source) as hdul:
        copied = fits.HDUList([hdu.copy() for hdu in hdul])
    removed: list[str] = []
    header = copied[0].header
    for key in list(header):
        normalized = str(key).upper()
        if normalized in explicit or normalized.startswith(atmospheric_prefixes):
            del header[key]
            removed.append(str(key))
    header["HIERARCH PYMOLFIT ATM MODE"] = "generic standard_midlatitude"
    header["HIERARCH PYMOLFIT META STRIPPED"] = True
    copied.writeto(destination, overwrite=True)
    copied.close()
    return tuple(sorted(removed))


def _spectra_vacuum(
    wavelength_air: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    masks: list[np.ndarray],
) -> list[Spectrum]:
    return [
        Spectrum(
            wavelength=wavelength_air[mask],
            flux=flux[mask],
            uncertainty=uncertainty[mask],
            wavelength_unit="micron",
            wavelength_medium="air",
        ).to_vacuum()
        for mask in masks
    ]


def _load_shared_lines() -> LineList:
    if SHARED_LINES.is_file():
        return LineList.from_table(SHARED_LINES)
    return uves._load_lines()


def _generic_atmosphere(airmass: float) -> AtmosphereProfile:
    profile = AtmosphereProfile.standard_midlatitude(
        airmass=airmass,
        observatory_altitude_m=GENERIC_ALTITUDE_M,
        pressure_at_observatory_atm=GENERIC_PRESSURE_ATM,
        temperature_at_observatory_k=GENERIC_TEMPERATURE_K,
        n_layers=GENERIC_LAYER_COUNT,
    )
    return AtmosphereProfile(
        profile.layers,
        metadata={
            "atmosphere_mode": "standard_midlatitude",
            "metadata_free": True,
            "airmass_source": "retained spectrum metadata for controlled comparison",
            "airmass": float(airmass),
            "observatory_altitude_m": GENERIC_ALTITUDE_M,
            "pressure_at_observatory_atm": GENERIC_PRESSURE_ATM,
            "temperature_at_observatory_k": GENERIC_TEMPERATURE_K,
            "n_layers": GENERIC_LAYER_COUNT,
            "observation_time_utc": None,
            "latitude_deg": None,
            "longitude_deg": None,
            "gdas_source": None,
        },
    )


def _run_fit(
    label: str,
    atmosphere: AtmosphereProfile,
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    header: fits.Header,
    masks: list[np.ndarray],
    lines: LineList,
    output: Path,
) -> tuple[object, float]:
    result, _, fit_seconds = uves._run_pymolfit(
        wavelength,
        flux,
        uncertainty,
        header,
        masks,
        lines,
        atmosphere=atmosphere,
    )
    if not result.success:
        raise RuntimeError(f"{label} atmosphere fit failed: {result.message}")
    for index, segment in enumerate(result.segment_results, start=1):
        segment.write(output / f"{label}_segment_{index}.ecsv")
    return result, fit_seconds


def _concatenate(result: object) -> dict[str, np.ndarray]:
    return uves._concatenate_pymolfit(result)


def _fit_wavelength(wavelength: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([wavelength[mask] for mask in masks])


def _relative_scatter(data: dict[str, np.ndarray], mask: np.ndarray) -> float:
    relative = data["flux"] / data["model_flux"] - 1.0
    return float(np.nanstd(relative[mask]))


def _column_metrics(
    prefix: str,
    atmosphere: AtmosphereProfile,
    result: object,
) -> dict[str, float]:
    h2o_scale = float(result.species_scales["H2O"])
    o2_scale = float(result.species_scales["O2"])
    return {
        f"{prefix}_h2o_scale": h2o_scale,
        f"{prefix}_o2_scale": o2_scale,
        f"{prefix}_base_vertical_h2o_column_cm2": float(
            atmosphere.total_vertical_column_cm2("H2O")
        ),
        f"{prefix}_fitted_vertical_h2o_column_cm2": float(
            atmosphere.total_vertical_column_cm2("H2O") * h2o_scale
        ),
        f"{prefix}_base_vertical_o2_column_cm2": float(
            atmosphere.total_vertical_column_cm2("O2")
        ),
        f"{prefix}_fitted_vertical_o2_column_cm2": float(
            atmosphere.total_vertical_column_cm2("O2") * o2_scale
        ),
    }


def _metrics(
    wavelength: np.ndarray,
    specific: dict[str, np.ndarray],
    generic: dict[str, np.ndarray],
    specific_result: object,
    generic_result: object,
    specific_atmosphere: AtmosphereProfile,
    generic_atmosphere: AtmosphereProfile,
    specific_build_seconds: float,
    generic_build_seconds: float,
    specific_fit_seconds: float,
    generic_fit_seconds: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    reliable = (
        np.isfinite(specific["transmission"])
        & np.isfinite(generic["transmission"])
        & (specific["transmission"] > TRANSMISSION_FLOOR)
        & (generic["transmission"] > TRANSMISSION_FLOOR)
    )
    telluric = reliable & (
        (specific["transmission"] < 0.995) | (generic["transmission"] < 0.995)
    )
    relative_percent = np.full(wavelength.shape, np.nan)
    relative_percent[reliable] = (
        100.0
        * (generic["transmission"][reliable] - specific["transmission"][reliable])
        / specific["transmission"][reliable]
    )
    absolute = generic["transmission"][reliable] - specific["transmission"][reliable]
    telluric_absolute = (
        generic["transmission"][telluric] - specific["transmission"][telluric]
    )
    specific_objective = uves._weighted_objective(
        specific["flux"],
        specific["model_flux"],
        specific["uncertainty"],
    )
    generic_objective = uves._weighted_objective(
        generic["flux"],
        generic["model_flux"],
        generic["uncertainty"],
    )
    values = relative_percent[reliable]
    metrics: dict[str, object] = {
        "source": str(uves.SOURCE),
        "instrument": "UVES",
        "species": ",".join(uves.SPECIES),
        "fit_ranges_micron": repr(uves.FIT_RANGES),
        "pixel_count": int(wavelength.size),
        "comparison_pixel_count": int(np.count_nonzero(reliable)),
        "telluric_pixel_count": int(np.count_nonzero(telluric)),
        "wavelength_min_angstrom": float(wavelength.min() * 1.0e4),
        "wavelength_max_angstrom": float(wavelength.max() * 1.0e4),
        "specific_atmosphere_layers": len(specific_atmosphere.layers),
        "generic_atmosphere_layers": len(generic_atmosphere.layers),
        "specific_atmosphere_build_seconds": float(specific_build_seconds),
        "generic_atmosphere_build_seconds": float(generic_build_seconds),
        "specific_fit_seconds": float(specific_fit_seconds),
        "generic_fit_seconds": float(generic_fit_seconds),
        "specific_total_seconds": float(specific_build_seconds + specific_fit_seconds),
        "generic_total_seconds": float(generic_build_seconds + generic_fit_seconds),
        "specific_weighted_objective": float(specific_objective),
        "generic_weighted_objective": float(generic_objective),
        "generic_over_specific_objective": float(generic_objective / specific_objective),
        "specific_relative_residual_scatter": _relative_scatter(specific, reliable),
        "generic_relative_residual_scatter": _relative_scatter(generic, reliable),
        "transmission_rms_absolute": float(np.sqrt(np.mean(absolute**2))),
        "telluric_transmission_rms_absolute": float(
            np.sqrt(np.mean(telluric_absolute**2))
        ),
        "transmission_max_abs": float(np.max(np.abs(absolute))),
        "relative_transmission_median_percent": float(np.median(values)),
        "relative_transmission_rms_percent": float(np.sqrt(np.mean(values**2))),
        "relative_transmission_p95_abs_percent": float(
            np.percentile(np.abs(values), 95.0)
        ),
        "relative_transmission_max_abs_percent": float(np.max(np.abs(values))),
        "specific_wavelength_shift_micron": float(specific_result.wavelength_shift),
        "generic_wavelength_shift_micron": float(generic_result.wavelength_shift),
        "specific_lsf_sigma_pixels": float(specific_result.lsf_sigma_pixels),
        "generic_lsf_sigma_pixels": float(generic_result.lsf_sigma_pixels),
    }
    metrics.update(_column_metrics("specific", specific_atmosphere, specific_result))
    metrics.update(_column_metrics("generic", generic_atmosphere, generic_result))
    return metrics, reliable, relative_percent


def _normalise(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    median = float(np.nanmedian(values[finite])) if np.any(finite) else np.nan
    return values / median if np.isfinite(median) and median != 0 else values


def _plot_spectrum(
    output: Path,
    wavelength: np.ndarray,
    specific: dict[str, np.ndarray],
    generic: dict[str, np.ndarray],
    reliable: np.ndarray,
    relative_percent: np.ndarray,
    metrics: dict[str, object],
) -> None:
    x = wavelength * 1.0e4
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8.7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.07},
    )
    for index, (lower, upper) in enumerate(uves.FIT_RANGES):
        region = (wavelength >= lower) & (wavelength <= upper)
        first = index == 0
        axes[0].plot(
            x[region],
            _normalise(specific["flux"][region]),
            color="0.15",
            lw=0.72,
            label="Observed spectrum" if first else None,
            zorder=3,
        )
        axes[0].plot(
            x[region],
            _normalise(specific["model_flux"][region]),
            color="#1f77b4",
            lw=0.82,
            label="Time-local MIPAS + GDAS" if first else None,
        )
        axes[0].plot(
            x[region],
            _normalise(generic["model_flux"][region]),
            color="#d95f02",
            lw=0.78,
            alpha=0.9,
            label="Metadata-free generic atmosphere" if first else None,
        )
        plotted = region & reliable
        axes[1].plot(
            x[plotted],
            relative_percent[plotted],
            color="#7a3e9d",
            lw=0.68,
        )
    axes[0].set_ylabel("Region-normalized flux")
    axes[0].set_title(
        "PyMolFit atmosphere sensitivity: observation-specific versus metadata-free generic"
    )
    axes[0].legend(loc="lower left", ncol=3, frameon=False)
    axes[0].text(
        0.985,
        0.96,
        f"Specific fit: {float(metrics['specific_fit_seconds']):.2f} s\n"
        f"Generic fit: {float(metrics['generic_fit_seconds']):.2f} s\n"
        f"Objective ratio: {float(metrics['generic_over_specific_objective']):.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75"},
    )
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    limit = max(0.5, 1.12 * float(np.percentile(np.abs(relative_percent[reliable]), 99.5)))
    axes[1].set_ylim(-limit, limit)
    axes[1].set_ylabel("Relative transmission\ndifference (%)")
    axes[1].set_xlabel(r"Observed air wavelength ($\AA$)")
    axes[1].text(
        0.012,
        0.08,
        "100 × (generic − time-local) / time-local\n"
        f"RMS = {float(metrics['relative_transmission_rms_percent']):.3f}%   "
        f"95th |difference| = {float(metrics['relative_transmission_p95_abs_percent']):.3f}%\n"
        f"Only T > {TRANSMISSION_FLOOR:.1f} in both models",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.8"},
    )
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.margins(x=0.01)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _profile_coordinates(
    atmosphere: AtmosphereProfile,
    base_altitude_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertical = np.asarray(
        [
            layer.path_length_m
            if layer.vertical_path_length_m is None
            else layer.vertical_path_length_m
            for layer in atmosphere.layers
        ],
        dtype=float,
    )
    altitude_m = base_altitude_m + np.cumsum(vertical) - 0.5 * vertical
    pressure = np.asarray([layer.pressure_atm for layer in atmosphere.layers], dtype=float)
    temperature = np.asarray([layer.temperature_k for layer in atmosphere.layers], dtype=float)
    h2o_ppmv = (
        np.asarray([layer.mixing_ratios.get("H2O", 0.0) for layer in atmosphere.layers])
        * 1.0e6
    )
    return altitude_m / 1000.0, pressure, temperature, h2o_ppmv


def _plot_atmospheres(
    output: Path,
    specific: AtmosphereProfile,
    generic: AtmosphereProfile,
) -> None:
    specific_altitude = float(specific.metadata.get("observatory_altitude_m", 2648.0))
    exact = _profile_coordinates(specific, specific_altitude)
    fallback = _profile_coordinates(generic, GENERIC_ALTITUDE_M)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 7.2), sharey=True)
    labels = ("Pressure (atm)", "Temperature (K)", "H$_2$O mixing ratio (ppmv)")
    for axis, exact_values, generic_values, label in zip(
        axes,
        exact[1:],
        fallback[1:],
        labels,
        strict=True,
    ):
        axis.plot(exact_values, exact[0], color="#1f77b4", lw=1.8, label="Time-local MIPAS + GDAS")
        axis.plot(
            generic_values,
            fallback[0],
            color="#d95f02",
            lw=1.6,
            label="Generic standard-midlatitude",
        )
        axis.set_xlabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_xscale("log")
    axes[2].set_xscale("log")
    axes[0].set_ylabel("Approximate altitude (km)")
    axes[0].set_ylim(2.0, 50.0)
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Atmospheric profiles supplied to the two PyMolFit runs")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_comparison(
    output: Path,
    wavelength: np.ndarray,
    specific: dict[str, np.ndarray],
    generic: dict[str, np.ndarray],
    reliable: np.ndarray,
    relative_percent: np.ndarray,
) -> None:
    table = Table()
    table["wavelength_air_angstrom"] = wavelength * 1.0e4
    table["observed_flux"] = specific["flux"]
    table["uncertainty"] = specific["uncertainty"]
    table["specific_model_flux"] = specific["model_flux"]
    table["generic_model_flux"] = generic["model_flux"]
    table["specific_transmission"] = specific["transmission"]
    table["generic_transmission"] = generic["transmission"]
    table["relative_transmission_difference_percent"] = relative_percent
    table["comparison_mask"] = reliable
    table.meta["relative_difference_definition"] = "100 * (generic - specific) / specific"
    table.write(output, format="ascii.ecsv", overwrite=True)


def run(output: Path) -> dict[str, object]:
    if not uves.SOURCE.is_file():
        raise FileNotFoundError(uves.SOURCE)
    if not SHARED_GDAS.is_file():
        raise FileNotFoundError(SHARED_GDAS)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    uves.OUTPUT = output

    wavelength, flux, uncertainty, header = uves._load_source()
    masks = uves._range_masks(wavelength, flux, uncertainty)
    input_path = uves._write_input(wavelength, flux, uncertainty, header, masks)
    stripped_path = output / "uves_input_without_time_place_weather.fits"
    removed_keywords = _stripped_input(input_path, stripped_path)
    shared_gdas = output / "specific_time_local_GDAS.fits"
    shutil.copy2(SHARED_GDAS, shared_gdas)
    lines = _load_shared_lines()
    spectra_vacuum = _spectra_vacuum(wavelength, flux, uncertainty, masks)
    airmass = uves._representative_airmass(header)

    started = time.perf_counter()
    specific_atmosphere = uves._build_atmosphere(
        header,
        spectra_vacuum,
        gdas_profile=shared_gdas,
    )
    specific_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    generic_atmosphere = _generic_atmosphere(airmass)
    generic_build_seconds = time.perf_counter() - started
    specific_atmosphere.write(output / "specific_atmosphere.ecsv")
    generic_atmosphere.write(output / "generic_atmosphere.ecsv")

    specific_result, specific_fit_seconds = _run_fit(
        "specific",
        specific_atmosphere,
        wavelength,
        flux,
        uncertainty,
        header,
        masks,
        lines,
        output,
    )
    generic_result, generic_fit_seconds = _run_fit(
        "generic",
        generic_atmosphere,
        wavelength,
        flux,
        uncertainty,
        header,
        masks,
        lines,
        output,
    )
    fit_wavelength = _fit_wavelength(wavelength, masks)
    specific = _concatenate(specific_result)
    generic = _concatenate(generic_result)
    metrics, reliable, relative_percent = _metrics(
        fit_wavelength,
        specific,
        generic,
        specific_result,
        generic_result,
        specific_atmosphere,
        generic_atmosphere,
        specific_build_seconds,
        generic_build_seconds,
        specific_fit_seconds,
        generic_fit_seconds,
    )
    metrics["retained_airmass"] = float(airmass)
    metrics["stripped_keyword_count"] = len(removed_keywords)
    _write_comparison(
        output / "comparison.ecsv",
        fit_wavelength,
        specific,
        generic,
        reliable,
        relative_percent,
    )
    _plot_spectrum(
        output / "specific_vs_generic_spectral_fit.png",
        fit_wavelength,
        specific,
        generic,
        reliable,
        relative_percent,
        metrics,
    )
    _plot_atmospheres(
        output / "specific_vs_generic_atmospheric_profiles.png",
        specific_atmosphere,
        generic_atmosphere,
    )
    Table(rows=[metrics]).write(output / "summary.ecsv", format="ascii.ecsv", overwrite=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    manifest = {
        "test": "PyMolFit observation-specific versus metadata-free atmosphere",
        "source": str(uves.SOURCE),
        "shared_line_list": str(SHARED_LINES if SHARED_LINES.is_file() else uves.LINE_CACHE),
        "shared_gdas": str(shared_gdas),
        "fit_ranges_micron": [list(interval) for interval in uves.FIT_RANGES],
        "retained_control_variable": {
            "airmass": airmass,
            "reason": "retained in both runs to isolate missing date/site/weather metadata",
        },
        "specific_atmosphere": dict(specific_atmosphere.metadata),
        "generic_assumptions": dict(generic_atmosphere.metadata),
        "stripped_fits": str(stripped_path),
        "stripped_keywords": list(removed_keywords),
        "shared_fit_configuration": {
            "species": list(uves.SPECIES),
            "continuum_order": 2,
            "wavelength_polynomial_order": 1,
            "initial_gaussian_fwhm_pixels": 1.0,
            "ftol": 1.0e-10,
            "xtol": 1.0e-10,
            "gtol": 1.0e-10,
        },
        "metrics": metrics,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare time-local and metadata-free PyMolFit atmospheres on UVES data."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = run(args.output_dir.resolve())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(args.output_dir.resolve() / "specific_vs_generic_spectral_fit.png")


if __name__ == "__main__":
    main()
