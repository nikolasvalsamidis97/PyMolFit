from __future__ import annotations

import numpy as np

from .fit import TelluricFitResult


def correction_summary(result: TelluricFitResult) -> dict[str, float]:
    """Return scalar diagnostics for a telluric correction result."""

    finite = (
        np.isfinite(result.spectrum.flux)
        & np.isfinite(result.corrected.flux)
        & np.isfinite(result.transmission)
        & np.isfinite(result.continuum)
        & (result.continuum != 0)
    )
    if not np.any(finite):
        return {
            "n_pixels": 0,
            "median_transmission": np.nan,
            "deep_absorption_fraction": np.nan,
            "raw_scatter": np.nan,
            "corrected_scatter": np.nan,
        }

    raw_norm = result.spectrum.flux[finite] / result.continuum[finite]
    corrected_norm = result.corrected.flux[finite] / result.continuum[finite]
    transmission = result.transmission[finite]
    return {
        "n_pixels": int(np.count_nonzero(finite)),
        "median_transmission": float(np.nanmedian(transmission)),
        "deep_absorption_fraction": float(np.nanmean(transmission < 0.5)),
        "raw_scatter": float(np.nanstd(raw_norm - np.nanmedian(raw_norm))),
        "corrected_scatter": float(np.nanstd(corrected_norm - np.nanmedian(corrected_norm))),
    }


def residual_by_window(
    result: TelluricFitResult,
    windows: tuple[tuple[float, float], ...],
) -> list[dict[str, float]]:
    """Summarize model residuals in wavelength windows."""

    rows = []
    residual = result.spectrum.flux - result.model_flux
    for start, stop in windows:
        lo, hi = sorted((start, stop))
        keep = (result.spectrum.wavelength >= lo) & (result.spectrum.wavelength <= hi)
        keep &= np.isfinite(residual)
        if np.any(keep):
            rms = float(np.sqrt(np.nanmean(residual[keep] ** 2)))
            median_abs = float(np.nanmedian(np.abs(residual[keep])))
        else:
            rms = np.nan
            median_abs = np.nan
        rows.append(
            {
                "start": float(lo),
                "stop": float(hi),
                "n_pixels": int(np.count_nonzero(keep)),
                "rms_residual": rms,
                "median_abs_residual": median_abs,
            }
        )
    return rows
