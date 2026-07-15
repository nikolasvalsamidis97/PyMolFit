from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

import genmolfit
from genmolfit import LineList

try:
    from local_tests import compare_uves_official_demo as uves
except ModuleNotFoundError:  # Direct execution places local_tests on sys.path.
    import compare_uves_official_demo as uves


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "local_tests" / "uves_broadspan_timed_comparison"
REFERENCE_GDAS = ROOT / "local_tests" / "uves_official_demo_comparison" / "GDAS.fits"
TRANSMISSION_FLOOR = 0.2
COMPARISON_AER_VERSION = "3.8.1.2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fresh_shared_lines(output: Path) -> LineList:
    lower = min(interval[0] for interval in uves.FIT_RANGES) - 0.003
    upper = max(interval[1] for interval in uves.FIT_RANGES) + 0.003
    line_list = LineList.from_aer_line_file(
        uves.AER_LINE_DB,
        species=uves.SPECIES,
        wavenumber_min=1.0e4 / upper,
        wavenumber_max=1.0e4 / lower,
        extra_broadener_dir=uves.AER_LINE_DB.parent,
        assume_sorted=False,
    )
    line_list.write(output / "selected_aer_lines.fits", format="fits")
    return line_list


def _concatenated_air_wavelength(
    wavelength: np.ndarray,
    masks: list[np.ndarray],
) -> np.ndarray:
    return np.concatenate([wavelength[mask] for mask in masks])


def _comparison_arrays(
    wavelength_air: np.ndarray,
    masks: list[np.ndarray],
    result: object,
    molecfit_model: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    wavelength_fit = _concatenated_air_wavelength(wavelength_air, masks)
    gen = uves._concatenate_genmolfit(result)
    n_pixels = min(
        wavelength_fit.size,
        *(values.size for values in gen.values()),
        *(values.size for values in molecfit_model.values()),
    )
    return (
        wavelength_fit[:n_pixels],
        {name: values[:n_pixels] for name, values in gen.items()},
        {name: values[:n_pixels] for name, values in molecfit_model.items()},
    )


def _profile_max_difference(shared: Path, molecfit_output: Path) -> float:
    shared_table = Table.read(shared, hdu=1)
    molecfit_table = Table.read(molecfit_output, hdu=1)
    if len(shared_table) != len(molecfit_table):
        return np.inf
    maxima = []
    for column in ("press", "height", "temp", "relhum"):
        if column not in shared_table.colnames or column not in molecfit_table.colnames:
            return np.inf
        maxima.append(
            np.nanmax(
                np.abs(
                    np.asarray(shared_table[column], dtype=float)
                    - np.asarray(molecfit_table[column], dtype=float)
                )
            )
        )
    return float(max(maxima))


def _calculate_metrics(
    wavelength: np.ndarray,
    gen: dict[str, np.ndarray],
    mol: dict[str, np.ndarray],
    gen_result: object,
    gen_seconds: float,
    molecfit_seconds: float,
    line_count: int,
    gdas_max_difference: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    reliable = (
        np.isfinite(gen["transmission"])
        & np.isfinite(mol["mtrans"])
        & (gen["transmission"] > TRANSMISSION_FLOOR)
        & (mol["mtrans"] > TRANSMISSION_FLOOR)
    )
    telluric = reliable & (
        (gen["transmission"] < 0.995) | (mol["mtrans"] < 0.995)
    )
    relative = np.full(wavelength.shape, np.nan)
    relative[reliable] = (
        100.0
        * (gen["transmission"][reliable] - mol["mtrans"][reliable])
        / mol["mtrans"][reliable]
    )
    direct = gen["transmission"][reliable] - mol["mtrans"][reliable]
    direct_telluric = gen["transmission"][telluric] - mol["mtrans"][telluric]
    gen_objective = uves._weighted_objective(
        gen["flux"],
        gen["model_flux"],
        gen["uncertainty"],
    )
    molecfit_objective = uves._weighted_objective(
        gen["flux"],
        mol["mflux"],
        gen["uncertainty"],
    )
    relative_values = relative[reliable]
    metrics: dict[str, object] = {
        "source": str(uves.SOURCE),
        "object": "UVES official Molecfit demo target",
        "instrument": "UVES",
        "species": ",".join(uves.SPECIES),
        "aer_version": COMPARISON_AER_VERSION,
        "fit_ranges_micron": repr(uves.FIT_RANGES),
        "wavelength_min_angstrom": float(wavelength.min() * 1.0e4),
        "wavelength_max_angstrom": float(wavelength.max() * 1.0e4),
        "wavelength_span_angstrom": float(np.ptp(wavelength) * 1.0e4),
        "fitted_wavelength_coverage_angstrom": float(
            sum(upper - lower for lower, upper in uves.FIT_RANGES) * 1.0e4
        ),
        "pixel_count": int(wavelength.size),
        "comparison_pixel_count": int(np.count_nonzero(reliable)),
        "telluric_pixel_count": int(np.count_nonzero(telluric)),
        "line_count": int(line_count),
        "genmolfit_seconds": float(gen_seconds),
        "molecfit_seconds": float(molecfit_seconds),
        "genmolfit_over_molecfit_time": float(gen_seconds / molecfit_seconds),
        "transmission_rms_absolute": float(np.sqrt(np.mean(direct**2))),
        "telluric_transmission_rms_absolute": float(
            np.sqrt(np.mean(direct_telluric**2))
        ),
        "transmission_max_abs": float(np.max(np.abs(direct))),
        "relative_transmission_median_percent": float(np.median(relative_values)),
        "relative_transmission_rms_percent": float(
            np.sqrt(np.mean(relative_values**2))
        ),
        "relative_transmission_p95_abs_percent": float(
            np.percentile(np.abs(relative_values), 95.0)
        ),
        "relative_transmission_max_abs_percent": float(
            np.max(np.abs(relative_values))
        ),
        "genmolfit_weighted_objective": float(gen_objective),
        "molecfit_weighted_objective": float(molecfit_objective),
        "weighted_objective_ratio": float(gen_objective / molecfit_objective),
        "genmolfit_nfev": int(gen_result.nfev),
        "genmolfit_covariance_rank": int(gen_result.covariance_rank),
        "genmolfit_parameter_count": int(len(gen_result.parameter_names)),
        "shared_gdas_max_abs_difference": float(gdas_max_difference),
    }
    return metrics, reliable, relative


def _write_table(
    output: Path,
    wavelength: np.ndarray,
    gen: dict[str, np.ndarray],
    mol: dict[str, np.ndarray],
    reliable: np.ndarray,
    relative: np.ndarray,
) -> None:
    table = Table()
    table["wavelength_air_angstrom"] = wavelength * 1.0e4
    table["observed_flux"] = gen["flux"]
    table["uncertainty"] = gen["uncertainty"]
    table["genmolfit_model_flux"] = gen["model_flux"]
    table["molecfit_model_flux"] = mol["mflux"]
    table["genmolfit_transmission"] = gen["transmission"]
    table["molecfit_transmission"] = mol["mtrans"]
    table["relative_transmission_difference_percent"] = relative
    table["comparison_mask"] = reliable
    table.meta["relative_difference_definition"] = "100 * (GenMolFit - Molecfit) / Molecfit"
    table.meta["transmission_floor"] = TRANSMISSION_FLOOR
    table.write(output, format="ascii.ecsv", overwrite=True)


def _normalise_region(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    median = float(np.nanmedian(values[finite])) if np.any(finite) else np.nan
    return values / median if np.isfinite(median) and median != 0 else values


def _plot(
    output: Path,
    wavelength: np.ndarray,
    gen: dict[str, np.ndarray],
    mol: dict[str, np.ndarray],
    reliable: np.ndarray,
    relative: np.ndarray,
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
            _normalise_region(gen["flux"][region]),
            color="0.15",
            lw=0.75,
            label="Observed spectrum" if first else None,
            zorder=3,
        )
        axes[0].plot(
            x[region],
            _normalise_region(gen["model_flux"][region]),
            color="#d95f02",
            lw=0.85,
            label="GenMolFit model" if first else None,
        )
        axes[0].plot(
            x[region],
            _normalise_region(mol["mflux"][region]),
            color="#1f77b4",
            lw=0.8,
            alpha=0.9,
            label="Molecfit model" if first else None,
        )
        plotted = region & reliable
        axes[1].plot(x[plotted], relative[plotted], color="#5b2a86", lw=0.7)

    axes[0].set_ylabel("Region-normalized flux")
    axes[0].set_title(
        "Broad-span UVES comparison using ESO-published fit windows "
        f"(shared GDAS and AER {COMPARISON_AER_VERSION})"
    )
    axes[0].legend(loc="lower left", ncol=3, frameon=False)
    timing = (
        f"GenMolFit: {float(metrics['genmolfit_seconds']):.2f} s\n"
        f"Molecfit: {float(metrics['molecfit_seconds']):.2f} s\n"
        f"GenMolFit / Molecfit: {float(metrics['genmolfit_over_molecfit_time']):.2f}x"
    )
    axes[0].text(
        0.985,
        0.96,
        timing,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75"},
    )

    axes[1].axhline(0.0, color="0.4", lw=0.8)
    robust_limit = float(np.percentile(np.abs(relative[reliable]), 99.5))
    axes[1].set_ylim(-max(0.5, 1.12 * robust_limit), max(0.5, 1.12 * robust_limit))
    axes[1].set_ylabel("Relative transmission\ndifference (%)")
    axes[1].set_xlabel(r"Observed air wavelength ($\AA$)")
    axes[1].text(
        0.012,
        0.08,
        "100 × (GenMolFit − Molecfit) / Molecfit\n"
        f"RMS = {float(metrics['relative_transmission_rms_percent']):.3f}%   "
        f"95th |difference| = {float(metrics['relative_transmission_p95_abs_percent']):.3f}%\n"
        f"Only T > {TRANSMISSION_FLOOR:.1f} in both models; display limits use the 99.5th percentile",
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


def run(output: Path) -> dict[str, object]:
    if not uves.SOURCE.is_file():
        raise FileNotFoundError(uves.SOURCE)
    if not uves.AER_LINE_DB.is_file():
        raise FileNotFoundError(uves.AER_LINE_DB)
    if not uves.MOLECFIT_ESOREX.is_file():
        raise FileNotFoundError(uves.MOLECFIT_ESOREX)
    if not REFERENCE_GDAS.is_file():
        raise FileNotFoundError(REFERENCE_GDAS)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shared_gdas = output / "shared_GDAS.fits"
    shutil.copy2(REFERENCE_GDAS, shared_gdas)

    # Reuse the established ingestion and fitting configuration while keeping
    # this comparison's products isolated from the science-readiness baseline.
    uves.OUTPUT = output
    wavelength, flux, uncertainty, header = uves._load_source()
    masks = uves._range_masks(wavelength, flux, uncertainty)
    if any(np.count_nonzero(mask) < 100 for mask in masks):
        raise RuntimeError("one or more published UVES fit regions contain too few valid pixels")
    input_path = uves._write_input(wavelength, flux, uncertainty, header, masks)

    gen_started = time.perf_counter()
    line_list = _fresh_shared_lines(output)
    gen_result, atmosphere, _ = uves._run_genmolfit(
        wavelength,
        flux,
        uncertainty,
        header,
        masks,
        line_list,
        gdas_profile=shared_gdas,
    )
    for index, segment in enumerate(gen_result.segment_results, start=1):
        segment.write(output / f"genmolfit_segment_{index}.ecsv")
    gen_seconds = time.perf_counter() - gen_started
    if not gen_result.success:
        raise RuntimeError(f"GenMolFit failed: {gen_result.message}")

    molecfit_path, molecfit_seconds = uves._run_molecfit(
        input_path,
        force=True,
        gdas_profile=shared_gdas,
    )
    wavelength_fit, gen, mol = _comparison_arrays(
        wavelength,
        masks,
        gen_result,
        uves._load_molecfit_model(molecfit_path),
    )
    gdas_difference = _profile_max_difference(shared_gdas, output / "GDAS.fits")
    metrics, reliable, relative = _calculate_metrics(
        wavelength_fit,
        gen,
        mol,
        gen_result,
        gen_seconds,
        molecfit_seconds,
        line_list.wavelength.size,
        gdas_difference,
    )
    _write_table(output / "comparison.ecsv", wavelength_fit, gen, mol, reliable, relative)
    plot_path = output / "uves_broadspan_model_comparison.png"
    _plot(plot_path, wavelength_fit, gen, mol, reliable, relative, metrics)

    Table(rows=[metrics]).write(output / "summary.ecsv", format="ascii.ecsv", overwrite=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    manifest = {
        "test": "broad-span UVES GenMolFit versus Molecfit comparison",
        "source": str(uves.SOURCE),
        "source_sha256": _sha256(uves.SOURCE),
        "input_sha256": _sha256(input_path),
        "shared_gdas_sha256": _sha256(shared_gdas),
        "aer_catalog": str(uves.AER_LINE_DB),
        "aer_catalog_version": COMPARISON_AER_VERSION,
        "aer_catalog_sha256": _sha256(uves.AER_LINE_DB),
        "selected_lines_sha256": _sha256(output / "selected_aer_lines.fits"),
        "fit_ranges_micron": [list(interval) for interval in uves.FIT_RANGES],
        "species": list(uves.SPECIES),
        "genmolfit_version": genmolfit.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timing_definition": (
            "Spectrum download/cropping, installed-catalog acquisition, and GDAS acquisition are "
            "excluded for both. GenMolFit includes fresh AER line selection, atmosphere assembly, "
            "fit, and native segment products. Molecfit includes fresh LNFL/TAPE3 construction, "
            "atmosphere assembly, fit, and native recipe products. Plotting is excluded."
        ),
        "fairness": {
            "same_input_pixels_and_uncertainties": True,
            "same_aer_catalog": True,
            "same_gdas_profile": gdas_difference <= 1.0e-12,
            "same_species": True,
            "same_published_fit_windows": True,
            "same_continuum_order": 2,
            "same_wavelength_polynomial_order": 1,
            "same_gaussian_lsf_initial_fwhm_pixels": 1.0,
            "same_optimizer_tolerances": "FTOL=XTOL=1e-10; GenMolFit also GTOL=1e-10",
            "known_unmatched_component": "SciPy least_squares versus Molecfit MPFIT",
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
        description="Run a cold, matched, broad-span UVES GenMolFit/Molecfit comparison."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = run(args.output_dir.resolve())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(args.output_dir.resolve() / "uves_broadspan_model_comparison.png")


if __name__ == "__main__":
    main()
