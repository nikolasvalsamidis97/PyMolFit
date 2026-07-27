from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import TextIO

import numpy as np

from .fit import TelluricFitResult


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _format_scalar(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _format_sequence(value: object) -> str:
    if value is None:
        return "none"
    if not isinstance(value, (tuple, list, np.ndarray)):
        return _format_scalar(value)
    return "[" + ", ".join(_format_scalar(item) for item in value) + "]"


def _format_ranges(value: object) -> str:
    if value is None:
        return "all valid pixels"
    if not isinstance(value, (tuple, list)):
        return _format_scalar(value)
    if not value:
        return "none"
    ranges = []
    for bounds in value:
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            ranges.append(
                f"({_format_scalar(bounds[0])}, {_format_scalar(bounds[1])})"
            )
        else:
            ranges.append(_format_scalar(bounds))
    return ", ".join(ranges)


def _class_label(value: object) -> str:
    details = _mapping(value)
    qualified_name = details.get("class")
    if qualified_name is None:
        return "disabled" if value is None else "configured"
    return str(qualified_name).rsplit(".", 1)[-1]


def format_fit_summary(
    result: TelluricFitResult,
    *,
    input_path: str | Path | None = None,
) -> str:
    """Return a readable report of the effective parameters used by a fit.

    Automatic choices are reported after resolution, so this summary describes
    the actual continuum solver, atmosphere, line data, instrumental profile,
    wavelength model, segmentation, and optimizer settings used for the final
    correction rather than only the values originally requested by the caller.
    """

    provenance = _mapping(result.provenance)
    fit_config = _mapping(provenance.get("fit_config"))
    config = _mapping(fit_config.get("fields"))
    atmosphere = _mapping(provenance.get("atmosphere_metadata"))
    continuum = _mapping(provenance.get("continuum_solver"))
    lsf_sigma = _mapping(provenance.get("lsf_sigma"))
    lsf_lorentz = _mapping(provenance.get("lsf_lorentz"))
    lsf_variable = _mapping(provenance.get("lsf_variable_width"))
    wavelength_alignment = _mapping(provenance.get("wavelength_alignment"))
    segmentation = _mapping(provenance.get("segmentation"))

    wavelength = result.spectrum.to_unit("micron").wavelength
    finite_wavelength = wavelength[np.isfinite(wavelength)]
    if finite_wavelength.size:
        wavelength_range = (
            f"{np.nanmin(finite_wavelength):.8g} to "
            f"{np.nanmax(finite_wavelength):.8g} micron"
        )
    else:
        wavelength_range = "no finite wavelengths"
    fit_pixels = (
        0
        if result.fit_mask is None
        else int(np.count_nonzero(np.asarray(result.fit_mask, dtype=bool)))
    )
    source = input_path
    if source is None:
        source = result.spectrum.meta.get("source")

    lines = ["PyMolFit effective fit configuration", "Data"]
    if source:
        lines.append(f"  input: {Path(str(source)).expanduser()}")
    lines.extend(
        (
            f"  wavelength range: {wavelength_range}",
            "  wavelength frame: observatory-frame "
            f"{result.spectrum.wavelength_medium}",
            f"  pixels: {result.spectrum.wavelength.size} total, {fit_pixels} fitted",
            f"  uncertainty weighting: {'yes' if result.spectrum.uncertainty is not None else 'no'}",
            "Line data",
            f"  source: {_format_scalar(provenance.get('line_source'))}",
            "  lines: "
            f"{_format_scalar(provenance.get('selected_line_count'))} selected from "
            f"{_format_scalar(provenance.get('line_count'))}",
            "  species: "
            + (
                ", ".join(str(item) for item in provenance.get("line_species", ()))
                or "none"
            ),
            "Atmosphere",
            f"  physical layers: {_format_scalar(provenance.get('atmosphere_layer_count'))}",
            f"  MIPAS profile: {_format_scalar(atmosphere.get('mipas_profile'))}",
            f"  GDAS source: {_format_scalar(atmosphere.get('gdas_source'))}",
            f"  observation time UTC: {_format_scalar(atmosphere.get('observation_time_utc'))}",
            "  observatory: "
            f"{_format_scalar(atmosphere.get('observatory_site'))}; "
            f"lat={_format_scalar(atmosphere.get('latitude_deg'))} deg, "
            f"lon={_format_scalar(atmosphere.get('longitude_deg'))} deg, "
            f"alt={_format_scalar(atmosphere.get('observatory_altitude_m'))} m",
            f"  airmass used by fit: {_format_scalar(config.get('airmass'))}",
            "Fit masks and segmentation",
            "  fit ranges (observatory vacuum micron): "
            f"{_format_ranges(config.get('fit_ranges'))}",
            "  excluded ranges (observatory vacuum micron): "
            f"{_format_ranges(config.get('exclude_ranges')) if config.get('exclude_ranges') is not None else 'none'}",
            "  segmentation: "
            + (
                f"{_format_scalar(segmentation.get('segment_count'))} automatic segments; "
                f"maximum width={_format_scalar(segmentation.get('segment_size_micron'))} micron"
                if segmentation
                else "single fit"
            ),
            "Continuum and optimizer",
            f"  continuum order: {_format_scalar(config.get('continuum_order'))}",
            "  continuum solver: "
            f"{_format_scalar(continuum.get('selected', config.get('solve_continuum_linear')))} "
            f"(requested={_format_scalar(continuum.get('requested'))}, "
            f"fallback={_format_scalar(continuum.get('fallback_used'))})",
            f"  loss: {_format_scalar(config.get('loss'))}; "
            f"f_scale={_format_scalar(config.get('f_scale'))}",
            "  tolerances: "
            f"ftol={_format_scalar(config.get('ftol'))}, "
            f"xtol={_format_scalar(config.get('xtol'))}, "
            f"gtol={_format_scalar(config.get('gtol'))}",
            f"  estimate uncertainties: {_format_scalar(config.get('estimate_uncertainties'))}",
            "Instrument model",
            "  Gaussian LSF sigma: "
            f"{result.lsf_sigma_pixels:.8g} pixels "
            f"(source={_format_scalar(lsf_sigma.get('source'))}, "
            f"fitted={_format_scalar(lsf_sigma.get('fit_enabled'))}, "
            f"bounds={_format_sequence(lsf_sigma.get('bounds_pixels'))})",
            f"  boxcar LSF width: {result.lsf_box_width_pixels:.8g} pixels "
            f"(fitted={_format_scalar(config.get('fit_lsf_box_width'))})",
            "  Lorentzian LSF FWHM: "
            f"{result.lsf_lorentz_fwhm_pixels:.8g} pixels "
            f"(model={_format_scalar(lsf_lorentz.get('selected_model'))}, "
            f"fitted={_format_scalar(lsf_lorentz.get('fit_enabled_in_full_fit'))}, "
            f"bounds={_format_sequence(lsf_lorentz.get('bounds_pixels'))})",
            "  wavelength-dependent LSF: "
            f"{_format_scalar(lsf_variable.get('selected_model'))}; "
            f"exponent={result.lsf_wavelength_exponent:.8g}, "
            "reference="
            f"{_format_scalar(lsf_variable.get('reference_wavelength_micron'))} micron",
            "  wavelength alignment: "
            f"{_format_scalar(wavelength_alignment.get('selected_model'))}; "
            f"coefficients={_format_sequence(wavelength_alignment.get('final_coefficients'))} "
            f"{_format_scalar(wavelength_alignment.get('coefficient_unit'))}; "
            f"bounds={_format_sequence(wavelength_alignment.get('bounds'))}",
            "Radiative transfer",
            "  internal grid: "
            f"high_resolution={_format_scalar(config.get('high_resolution_grid'))}, "
            f"oversampling={_format_scalar(config.get('high_resolution_oversampling'))}, "
            f"margin={_format_scalar(config.get('high_resolution_margin_pixels'))} pixels",
            "  transfer grid: "
            f"{_format_scalar(config.get('radiative_transfer_grid'))}; "
            f"step={_format_scalar(config.get('radiative_transfer_step_cm'))} cm^-1; "
            f"maximum points={_format_scalar(config.get('radiative_transfer_max_points'))}",
            f"  rebin mode: {_format_scalar(config.get('high_resolution_rebin_mode'))}",
            "  line wings: "
            f"{_format_scalar(config.get('line_wing_mode'))}; "
            f"cutoff={_format_scalar(config.get('line_cutoff_cm'))} cm^-1; "
            f"taper={_format_scalar(config.get('line_taper_cm'))} cm^-1",
            "  continua/components: "
            f"H2O={_class_label(config.get('h2o_continuum'))}, "
            f"components={_class_label(config.get('components'))}, "
            f"Rayleigh={_format_scalar(config.get('rayleigh'))}, "
            f"N2={_format_scalar(config.get('n2_continuum'))}, "
            f"O2={_format_scalar(config.get('o2_continuum'))}",
            f"  minimum corrected transmission: {_format_scalar(config.get('min_transmission'))}",
            "Fitted parameters",
        )
    )
    if result.species_scales:
        for species, scale in sorted(result.species_scales.items()):
            uncertainty = result.species_scale_uncertainties.get(species)
            suffix = "" if uncertainty is None else f" +/- {uncertainty:.4g}"
            lines.append(f"  {species} scale: {scale:.8g}{suffix}")
    else:
        lines.append("  species scales: none")
    lines.extend(
        (
            f"  continuum coefficients: {_format_sequence(result.continuum_coefficients)}",
            f"  parameter bounds reached: "
            f"{_format_scalar(result.parameter_bound_status or 'none')}",
            "Result",
            f"  success: {_format_scalar(result.success)}",
            f"  cost: {result.cost:.8g}",
            f"  function evaluations: {result.nfev}",
            f"  reduced chi-square: {_format_scalar(result.reduced_chi_square)}",
            f"  median transmission: {np.nanmedian(result.transmission):.8g}",
            f"  termination: {result.message}",
        )
    )
    return "\n".join(lines)


def print_fit_summary(
    result: TelluricFitResult,
    *,
    input_path: str | Path | None = None,
    file: TextIO | None = None,
) -> None:
    """Print the effective parameters and final values used by a fit."""

    destination = sys.stdout if file is None else file
    print(format_fit_summary(result, input_path=input_path), file=destination)


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
