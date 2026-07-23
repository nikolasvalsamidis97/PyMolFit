from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymolfit import correct_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "local_tests" / "betapic_harps_nad_comparison"
OUTPUT_DIR = PROJECT_ROOT / "local_tests" / "nad_exclude_range_comparison"

FIT_RANGE = ((0.58825, 0.59065),)
OLD_EXCLUDE_RANGES = (
    (0.58888, 0.58912),
    (0.58948, 0.58978),
)
NEW_EXCLUDE_RANGES = (
    (0.58888, 0.58934),
    (0.58936, 0.58993),
)


def run_fit(input_path: Path, exclude_ranges: tuple[tuple[float, float], ...]):
    started = time.perf_counter()
    result = correct_file(
        input_path=input_path,
        wavelength_medium="air",
        aer_catalog="auto",
        atmosphere_mode="mipas_gdas",
        gdas_mode="average",
        continuum_order=2,
        solve_continuum_linear=True,
        lsf_box_width_pixels=1.0,
        lsf_sigma_pixels=2.17,
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.5, 4.0),
        lsf_lorentz_fwhm_pixels=0.5,
        fit_lsf_lorentz_fwhm=True,
        lsf_lorentz_fwhm_bounds=(0.0, 6.0),
        fit_wavelength_shift=True,
        wavelength_shift_bounds=(-2e-5, 2e-5),
        fit_ranges=FIT_RANGE,
        exclude_ranges=exclude_ranges,
    )
    return result, time.perf_counter() - started


def outside_ranges(
    wavelength_micron: np.ndarray,
    ranges: tuple[tuple[float, float], ...],
) -> np.ndarray:
    selected = np.ones(wavelength_micron.shape, dtype=bool)
    for lower, upper in ranges:
        selected &= ~((wavelength_micron >= lower) & (wavelength_micron <= upper))
    return selected


def normalized_model_residual(result) -> np.ndarray:
    continuum = np.asarray(result.continuum, dtype=float)
    residual = np.asarray(result.spectrum.flux - result.model_flux, dtype=float)
    return np.divide(
        residual,
        continuum,
        out=np.full(residual.shape, np.nan),
        where=np.isfinite(continuum) & (continuum != 0),
    )


def rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan


def metrics(old_result, new_result) -> dict[str, float | int]:
    wavelength = old_result.spectrum.to_air().to_unit("micron").wavelength
    common = (
        (wavelength >= FIT_RANGE[0][0])
        & (wavelength <= FIT_RANGE[0][1])
        & outside_ranges(wavelength, NEW_EXCLUDE_RANGES)
    )
    old_residual = normalized_model_residual(old_result)
    new_residual = normalized_model_residual(new_result)
    common &= np.isfinite(old_residual) & np.isfinite(new_residual)

    old_transmission = np.asarray(old_result.transmission, dtype=float)
    new_transmission = np.asarray(new_result.transmission, dtype=float)
    telluric = common & (
        (old_transmission < 0.995)
        | (new_transmission < 0.995)
    )

    old_abs = np.abs(old_residual[common])
    new_abs = np.abs(new_residual[common])
    old_telluric_abs = np.abs(old_residual[telluric])
    new_telluric_abs = np.abs(new_residual[telluric])

    return {
        "shared_pixels": int(np.count_nonzero(common)),
        "shared_telluric_pixels": int(np.count_nonzero(telluric)),
        "old_shared_rms": rms(old_residual[common]),
        "new_shared_rms": rms(new_residual[common]),
        "old_shared_median_abs": float(np.nanmedian(old_abs)),
        "new_shared_median_abs": float(np.nanmedian(new_abs)),
        "old_telluric_rms": rms(old_residual[telluric]),
        "new_telluric_rms": rms(new_residual[telluric]),
        "old_telluric_p95_abs": float(np.nanpercentile(old_telluric_abs, 95)),
        "new_telluric_p95_abs": float(np.nanpercentile(new_telluric_abs, 95)),
    }


def plot_comparison(path: Path, old_result, new_result) -> None:
    observed = old_result.spectrum.to_air().to_unit("angstrom")
    wavelength = observed.wavelength
    visible = (wavelength >= 5882.5) & (wavelength <= 5906.5)
    scale = np.nanmedian(observed.flux[visible])

    old_corrected = np.asarray(old_result.corrected.flux, dtype=float) / scale
    new_corrected = np.asarray(new_result.corrected.flux, dtype=float) / scale
    old_residual = normalized_model_residual(old_result)
    new_residual = normalized_model_residual(new_result)

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0, 1.15, 1.0)},
    )

    axes[0].plot(wavelength, observed.flux / scale, color="black", linewidth=0.8)
    axes[0].set_ylabel("Observed / median")
    axes[0].set_title("Beta Pic HARPS Na D: old versus proposed exclusion ranges")

    axes[1].plot(
        wavelength,
        old_result.transmission,
        color="tab:blue",
        linewidth=0.9,
        label="Old mask fit",
    )
    axes[1].plot(
        wavelength,
        new_result.transmission,
        color="tab:orange",
        linewidth=0.9,
        label="Proposed mask fit",
    )
    axes[1].set_ylabel("Transmission")
    axes[1].legend()

    axes[2].plot(
        wavelength,
        old_corrected,
        color="tab:blue",
        linewidth=0.8,
        label="Corrected with old mask",
    )
    axes[2].plot(
        wavelength,
        new_corrected,
        color="tab:orange",
        linewidth=0.8,
        label="Corrected with proposed mask",
    )
    axes[2].set_ylabel("Corrected / median")
    axes[2].legend()

    axes[3].plot(
        wavelength,
        old_residual,
        color="tab:blue",
        linewidth=0.8,
        label="Old normalized residual",
    )
    axes[3].plot(
        wavelength,
        new_residual,
        color="tab:orange",
        linewidth=0.8,
        label="Proposed normalized residual",
    )
    axes[3].axhline(0, color="black", linewidth=0.7)
    axes[3].set_ylabel("(data-model) / continuum")
    axes[3].set_xlabel("Air wavelength [Angstrom]")
    axes[3].legend()

    for axis in axes:
        for lower, upper in NEW_EXCLUDE_RANGES:
            axis.axvspan(
                lower * 1e4,
                upper * 1e4,
                color="tab:orange",
                alpha=0.08,
            )
        for lower, upper in OLD_EXCLUDE_RANGES:
            axis.axvline(lower * 1e4, color="tab:blue", linestyle=":", linewidth=0.8)
            axis.axvline(upper * 1e4, color="tab:blue", linestyle=":", linewidth=0.8)
        axis.set_xlim(5882.5, 5906.5)
        axis.grid(alpha=0.15)

    figure.text(
        0.99,
        0.01,
        "Orange shading: proposed exclusions; blue dotted boundaries: old exclusions",
        ha="right",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    inputs = sorted(SOURCE_ROOT.glob("ADP_*/harps_nad_crop_air.fits"))
    if not inputs:
        inputs = [PROJECT_ROOT / "tutorials" / "data" / "harps_nad_crop_air.fits"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for input_path in inputs:
        case = input_path.parent.name
        print(f"Fitting {case} with old exclusions...")
        old_result, old_elapsed = run_fit(input_path, OLD_EXCLUDE_RANGES)
        print(f"Fitting {case} with proposed exclusions...")
        new_result, new_elapsed = run_fit(input_path, NEW_EXCLUDE_RANGES)
        comparison = metrics(old_result, new_result)

        row: dict[str, object] = {
            "case": case,
            "input": str(input_path),
            "old_elapsed_s": old_elapsed,
            "new_elapsed_s": new_elapsed,
            "old_fit_pixels": int(np.count_nonzero(old_result.fit_mask)),
            "new_fit_pixels": int(np.count_nonzero(new_result.fit_mask)),
            "old_cost_per_fit_pixel": float(
                np.sqrt(2.0 * old_result.cost / np.count_nonzero(old_result.fit_mask))
            ),
            "new_cost_per_fit_pixel": float(
                np.sqrt(2.0 * new_result.cost / np.count_nonzero(new_result.fit_mask))
            ),
            "old_h2o_scale": float(old_result.species_scales.get("H2O", np.nan)),
            "new_h2o_scale": float(new_result.species_scales.get("H2O", np.nan)),
            "old_shift_angstrom": float(old_result.wavelength_shift * 1e4),
            "new_shift_angstrom": float(new_result.wavelength_shift * 1e4),
            "old_lsf_sigma_pixels": float(old_result.lsf_sigma_pixels),
            "new_lsf_sigma_pixels": float(new_result.lsf_sigma_pixels),
            "old_lsf_lorentz_fwhm_pixels": float(old_result.lsf_lorentz_fwhm_pixels),
            "new_lsf_lorentz_fwhm_pixels": float(new_result.lsf_lorentz_fwhm_pixels),
            **comparison,
        }
        rows.append(row)

        if "2017-04-07" in case or len(inputs) == 1:
            plot_comparison(
                OUTPUT_DIR / "nad_exclude_ranges_2017_comparison.png",
                old_result,
                new_result,
            )

    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    old_shared = np.asarray([row["old_shared_rms"] for row in rows], dtype=float)
    new_shared = np.asarray([row["new_shared_rms"] for row in rows], dtype=float)
    old_telluric = np.asarray([row["old_telluric_rms"] for row in rows], dtype=float)
    new_telluric = np.asarray([row["new_telluric_rms"] for row in rows], dtype=float)

    print(f"Wrote {summary_path}")
    print(
        "Shared-region RMS median, old -> proposed:",
        f"{np.nanmedian(old_shared):.6f} -> {np.nanmedian(new_shared):.6f}",
    )
    print(
        "Telluric-pixel RMS median, old -> proposed:",
        f"{np.nanmedian(old_telluric):.6f} -> {np.nanmedian(new_telluric):.6f}",
    )
    print(
        "Proposed-mask wins:",
        f"{np.count_nonzero(new_shared < old_shared)}/{len(rows)} shared-region cases,",
        f"{np.count_nonzero(new_telluric < old_telluric)}/{len(rows)} telluric-pixel cases",
    )


if __name__ == "__main__":
    main()
