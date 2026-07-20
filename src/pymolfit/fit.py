from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from astropy.table import Table
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .atmosphere import AtmosphereProfile
from .components import AbsorptionComponent, HitranLineAbsorption, line_wing_effective_cutoff_cm
from .continuum import MTCKDH2OContinuum
from .linelist import LineList
from .model import (
    PiecewiseConstantRebinPlan,
    SampleAverageRebinPlan,
    high_resolution_wavelength_grid,
    lsf_kernel_half_width_pixels,
    optical_depth_basis,
    prepare_piecewise_constant_rebin,
    prepare_sample_average_rebin,
    radiative_transfer_grid_point_count,
    radiative_transfer_wavelength_grid,
    transmission_from_basis,
    transmission_from_high_resolution_basis,
)
from .partition import PartitionTable
from .physics import (
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_AVMASS_AMU,
    LBLRTM_DEFAULT_SAMPLE,
    LBLRTM_VOIGT_DOMAIN_HWF3,
    lblrtm_dynamic_max_line_cutoff_cm,
    wavelength_micron_to_wavenumber_cm,
)
from .provenance import build_fit_provenance, provenance_json
from .radiative_transfer import PhysicalModelConfig, physical_optical_depth_basis
from .spectrum import Spectrum, correct_spectrum


@dataclass(frozen=True)
class FitConfig:
    airmass: float = 1.0
    continuum_order: int = 1
    species: tuple[str, ...] | None = None
    initial_species_scales: Mapping[str, float] | None = None
    fixed_species_scales: Mapping[str, float] | None = None
    continuum_prior_weight: float = 0.0
    continuum_prior_fractional_sigma: float = 0.05
    solve_continuum_linear: bool = False
    lsf_sigma_pixels: float = 0.0
    lsf_box_width_pixels: float = 0.0
    lsf_lorentz_fwhm_pixels: float = 0.0
    lsf_variable_width: bool = False
    lsf_reference_wavelength_micron: float | None = None
    lsf_kernel_width_fwhm: float = 3.0
    lsf_molecfit_voigt: bool = False
    high_resolution_grid: bool = False
    high_resolution_oversampling: float = 5.0
    high_resolution_margin_pixels: float = 2.0
    high_resolution_rebin_mode: str = "integrate"
    radiative_transfer_grid: str = "auto"
    radiative_transfer_step_cm: float | None = None
    radiative_transfer_max_points: int = 2_000_000
    line_cutoff_cm: float | None = None
    subtract_cutoff_profile: bool = False
    line_taper_cm: float = 0.0
    line_wing_mode: str = "full"
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3
    line_margin_micron: float = 0.01
    chunk_size: int = 0
    min_transmission: float = 0.03
    scale_bounds: tuple[float, float] = (1.0e-3, 1.0e3)
    atmosphere: AtmosphereProfile | None = None
    partition_exponent: float = 1.5
    partition_table: PartitionTable | None = None
    h2o_continuum: MTCKDH2OContinuum | None = None
    h2o_continuum_foreign_closure: bool = False
    rayleigh: bool = False
    rayleigh_xrayl: float = 1.0
    n2_continuum: bool = False
    n2_continuum_xn2cn: float = 1.0
    o2_continuum: bool = False
    o2_continuum_xo2cn: float = 1.0
    components: tuple[AbsorptionComponent, ...] | None = None
    fit_wavelength_shift: bool = False
    fit_wavelength_polynomial: bool = False
    wavelength_polynomial_order: int = 1
    fit_segment_wavelength_shifts: bool = False
    fit_segment_wavelength_polynomial: bool = False
    segment_wavelength_polynomial_order: int = 1
    initial_wavelength_shift: float = 0.0
    wavelength_shift_bounds: tuple[float, float] = (-5.0e-4, 5.0e-4)
    fit_lsf_sigma: bool = False
    lsf_sigma_bounds: tuple[float, float] = (0.0, 5.0)
    fit_lsf_box_width: bool = False
    lsf_box_width_bounds: tuple[float, float] = (0.0, 10.0)
    fit_lsf_lorentz_fwhm: bool = False
    lsf_lorentz_fwhm_bounds: tuple[float, float] = (0.0, 10.0)
    fit_ranges: tuple[tuple[float, float], ...] | None = None
    exclude_ranges: tuple[tuple[float, float], ...] | None = None
    loss: str = "linear"
    f_scale: float = 1.0
    ftol: float = 1.0e-10
    xtol: float = 1.0e-10
    gtol: float = 1.0e-10
    max_nfev: int | None = None
    use_jacobian_sparsity: bool = True
    basis_workers: int = 0
    estimate_uncertainties: bool = False


@dataclass(frozen=True)
class TelluricFitResult:
    spectrum: Spectrum
    corrected: Spectrum
    transmission: np.ndarray
    continuum: np.ndarray
    model_flux: np.ndarray
    species_scales: dict[str, float]
    wavelength_shift: float
    wavelength_coefficients: np.ndarray
    lsf_sigma_pixels: float
    lsf_box_width_pixels: float
    lsf_lorentz_fwhm_pixels: float
    continuum_coefficients: np.ndarray
    metrics: dict[str, float]
    success: bool
    message: str
    cost: float
    nfev: int
    parameter_names: tuple[str, ...] = ()
    parameter_covariance: np.ndarray | None = None
    parameter_standard_errors: dict[str, float] = field(default_factory=dict)
    species_scale_uncertainties: dict[str, float] = field(default_factory=dict)
    transmission_uncertainty: np.ndarray | None = None
    reduced_chi_square: float = np.nan
    covariance_rank: int = 0
    fit_mask: np.ndarray | None = None
    parameter_bound_status: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    def to_table(self) -> Table:
        table = Table()
        table["wavelength"] = self.spectrum.wavelength
        table["wavelength"].unit = self.spectrum.wavelength_unit
        table["flux"] = self.spectrum.flux
        table["model_flux"] = self.model_flux
        table["continuum"] = self.continuum
        table["transmission"] = self.transmission
        if self.transmission_uncertainty is not None:
            table["transmission_uncertainty"] = self.transmission_uncertainty
        table["corrected_flux"] = self.corrected.flux
        if self.spectrum.uncertainty is not None:
            table["uncertainty"] = self.spectrum.uncertainty
        if self.corrected.uncertainty is not None:
            table["corrected_uncertainty"] = self.corrected.uncertainty
        table["input_mask"] = self.spectrum.valid
        if self.fit_mask is not None:
            table["fit_mask"] = np.asarray(self.fit_mask, dtype=bool)
        table["corrected_mask"] = self.corrected.valid
        table.meta.update(
            {
                "wavelength_medium": self.spectrum.wavelength_medium,
                "fit_success": bool(self.success),
                "fit_message": self.message,
                "fit_cost": float(self.cost),
                "fit_nfev": int(self.nfev),
                "species_scales": json.dumps(self.species_scales, sort_keys=True),
                "wavelength_shift_micron": float(self.wavelength_shift),
                "wavelength_coefficients": json.dumps(self.wavelength_coefficients.tolist()),
                "lsf_sigma_pixels": float(self.lsf_sigma_pixels),
                "lsf_box_width_pixels": float(self.lsf_box_width_pixels),
                "lsf_lorentz_fwhm_pixels": float(self.lsf_lorentz_fwhm_pixels),
                "continuum_coefficients": json.dumps(self.continuum_coefficients.tolist()),
                "parameter_names": json.dumps(self.parameter_names),
                "parameter_standard_errors": json.dumps(
                    self.parameter_standard_errors,
                    sort_keys=True,
                ),
                "reduced_chi_square": float(self.reduced_chi_square),
                "covariance_rank": int(self.covariance_rank),
                "covariance_parameter_count": len(self.parameter_names),
                "covariance_full_rank": bool(
                    self.parameter_covariance is not None
                    and self.covariance_rank == len(self.parameter_names)
                ),
                "parameter_bound_status": json.dumps(
                    self.parameter_bound_status,
                    sort_keys=True,
                ),
                "provenance_json": provenance_json(self.provenance),
            }
        )
        return table

    def write(self, path: str | Path, *, format: str = "ascii.ecsv") -> None:
        self.to_table().write(path, format=format, overwrite=True)


@dataclass(frozen=True)
class MultiTelluricFitResult:
    segment_results: tuple[TelluricFitResult, ...]
    species_scales: dict[str, float]
    wavelength_shift: float
    lsf_sigma_pixels: float
    lsf_box_width_pixels: float
    lsf_lorentz_fwhm_pixels: float
    success: bool
    message: str
    cost: float
    nfev: int
    parameter_names: tuple[str, ...] = ()
    parameter_covariance: np.ndarray | None = None
    parameter_standard_errors: dict[str, float] = field(default_factory=dict)
    species_scale_uncertainties: dict[str, float] = field(default_factory=dict)
    reduced_chi_square: float = np.nan
    covariance_rank: int = 0
    parameter_bound_status: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, float]:
        residuals = [
            result.metrics.get("corrected_scatter", np.nan)
            for result in self.segment_results
        ]
        return {
            "median_segment_corrected_scatter": float(np.nanmedian(residuals)),
            "mean_segment_corrected_scatter": float(np.nanmean(residuals)),
        }


def _poly_design(wavelength: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 0:
        raise ValueError("continuum_order must be non-negative")
    center = 0.5 * (np.nanmin(wavelength) + np.nanmax(wavelength))
    span = np.nanmax(wavelength) - np.nanmin(wavelength)
    if span <= 0:
        raise ValueError("wavelength must span a non-zero range")
    x = 2.0 * (wavelength - center) / span
    design = np.vstack([x**degree for degree in range(order + 1)]).T
    return x, design


def _poly_design_over_bounds(
    wavelength: np.ndarray,
    order: int,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a polynomial design on a coordinate shared by several segments."""

    if order < 0:
        raise ValueError("wavelength polynomial order must be non-negative")
    lower, upper = map(float, bounds)
    span = upper - lower
    if not np.isfinite(lower) or not np.isfinite(upper) or span <= 0:
        raise ValueError("global wavelength bounds must be finite and increasing")
    x = 2.0 * (np.asarray(wavelength, dtype=float) - 0.5 * (lower + upper)) / span
    return x, np.vstack([x**degree for degree in range(order + 1)]).T


def _windows_mask(
    wavelength: np.ndarray,
    include: tuple[tuple[float, float], ...] | None,
    exclude: tuple[tuple[float, float], ...] | None,
) -> np.ndarray:
    mask = np.ones(wavelength.shape, dtype=bool)
    if include is not None:
        mask[:] = False
        for start, stop in include:
            lo, hi = sorted((start, stop))
            mask |= (wavelength >= lo) & (wavelength <= hi)
    if exclude is not None:
        for start, stop in exclude:
            lo, hi = sorted((start, stop))
            mask &= ~((wavelength >= lo) & (wavelength <= hi))
    return mask


def _shift_basis(wavelength: np.ndarray, basis: np.ndarray, shift_micron: float | np.ndarray) -> np.ndarray:
    shift = np.asarray(shift_micron, dtype=float)
    if shift.ndim == 0 and float(shift) == 0:
        return basis
    if shift.ndim > 0 and shift.shape != wavelength.shape:
        raise ValueError("wavelength-dependent shift must match wavelength shape")
    sample_at = wavelength - shift
    shifted = np.empty_like(basis)
    for index in range(basis.shape[0]):
        shifted[index] = np.interp(sample_at, wavelength, basis[index], left=0.0, right=0.0)
    return shifted


def _segment_wavelength_polynomial_order(config: FitConfig) -> int | None:
    if config.fit_segment_wavelength_polynomial:
        if config.segment_wavelength_polynomial_order < 0:
            raise ValueError("segment_wavelength_polynomial_order must be non-negative")
        return int(config.segment_wavelength_polynomial_order)
    if config.fit_segment_wavelength_shifts:
        return 0
    return None


def _global_wavelength_polynomial_order(config: FitConfig) -> int | None:
    if config.fit_wavelength_polynomial:
        if config.fit_wavelength_shift:
            raise ValueError(
                "fit_wavelength_polynomial cannot be combined with fit_wavelength_shift"
            )
        if config.wavelength_polynomial_order < 0:
            raise ValueError("wavelength_polynomial_order must be non-negative")
        return int(config.wavelength_polynomial_order)
    if config.fit_wavelength_shift:
        return 0
    return None


def _wavelength_coefficients_initial(order: int, initial_shift: float) -> np.ndarray:
    coefficients = np.zeros(order + 1, dtype=float)
    coefficients[0] = float(initial_shift)
    return coefficients


def _wavelength_shift_from_coefficients(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(design, dtype=float) @ np.asarray(coefficients, dtype=float)


def _validate_lsf_fit_config(config: FitConfig) -> None:
    if config.fit_lsf_sigma and config.lsf_sigma_bounds[1] <= config.lsf_sigma_bounds[0]:
        raise ValueError("lsf_sigma_bounds must be increasing")
    if config.fit_lsf_box_width and config.lsf_box_width_bounds[1] <= config.lsf_box_width_bounds[0]:
        raise ValueError("lsf_box_width_bounds must be increasing")
    if config.fit_lsf_lorentz_fwhm and config.lsf_lorentz_fwhm_bounds[1] <= config.lsf_lorentz_fwhm_bounds[0]:
        raise ValueError("lsf_lorentz_fwhm_bounds must be increasing")
    for name, value in (
        ("lsf_sigma_pixels", config.lsf_sigma_pixels),
        ("lsf_box_width_pixels", config.lsf_box_width_pixels),
        ("lsf_lorentz_fwhm_pixels", config.lsf_lorentz_fwhm_pixels),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _validate_optimizer_tolerances(config: FitConfig) -> None:
    for name, value in (("ftol", config.ftol), ("xtol", config.xtol), ("gtol", config.gtol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")


def _initial_lsf_width(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if np.isfinite(value) and lower < value < upper:
        return float(value)
    midpoint = 0.5 * (lower + upper)
    return float(np.clip(midpoint, lower + np.finfo(float).eps, upper - np.finfo(float).eps))


def _components_for_selected_lines(
    components: tuple[AbsorptionComponent, ...] | None,
    selected_lines: LineList,
) -> tuple[AbsorptionComponent, ...] | None:
    if components is None:
        return None
    selected = []
    for component in components:
        if isinstance(component, HitranLineAbsorption):
            selected.append(replace(component, line_list=selected_lines))
        else:
            selected.append(component)
    return tuple(selected)


def _line_selection_margin(wavelength_fit: np.ndarray, config: FitConfig) -> float:
    if config.line_margin_micron < 0:
        raise ValueError("line_margin_micron must be non-negative")
    if config.line_taper_cm < 0:
        raise ValueError("line_taper_cm must be non-negative")

    cutoff_values = []
    config_cutoff = _line_wing_selection_cutoff_cm(
        wavelength_fit,
        line_wing_mode=config.line_wing_mode,
        line_cutoff_cm=config.line_cutoff_cm,
        lblrtm_sample=config.lblrtm_sample,
        lblrtm_alfal0=config.lblrtm_alfal0,
        lblrtm_hwf3=config.lblrtm_hwf3,
    )
    if config_cutoff is not None:
        cutoff_values.append(config_cutoff)
    if config.components is not None:
        for component in config.components:
            if isinstance(component, HitranLineAbsorption):
                component_cutoff = _line_wing_selection_cutoff_cm(
                    wavelength_fit,
                    line_wing_mode=component.line_wing_mode,
                    line_cutoff_cm=component.line_cutoff_cm,
                    lblrtm_sample=component.lblrtm_sample,
                    lblrtm_alfal0=component.lblrtm_alfal0,
                    lblrtm_hwf3=component.lblrtm_hwf3,
                )
                if component_cutoff is not None:
                    cutoff_values.append(component_cutoff)
    if not cutoff_values:
        return config.line_margin_micron
    if not np.all(np.isfinite(cutoff_values)):
        return config.line_margin_micron

    max_wavelength = float(np.nanmax(np.abs(wavelength_fit)))
    cutoff_margin = 1.1 * max_wavelength**2 * max(cutoff_values) / 1.0e4
    return max(config.line_margin_micron, cutoff_margin)


def _line_wing_selection_cutoff_cm(
    wavelength_micron: np.ndarray,
    *,
    line_wing_mode: str,
    line_cutoff_cm: float | None,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_hwf3: float,
) -> float | None:
    cutoff = line_wing_effective_cutoff_cm(line_wing_mode, line_cutoff_cm)
    if str(line_wing_mode).strip().lower() not in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}:
        return cutoff

    dynamic_cutoff = lblrtm_dynamic_max_line_cutoff_cm(
        _wavenumber_grid_spacing_cm(wavelength_micron),
        sample=lblrtm_sample,
        alfal0=lblrtm_alfal0,
        hwf3=lblrtm_hwf3,
    )
    if line_cutoff_cm is not None:
        return min(dynamic_cutoff, float(line_cutoff_cm))
    if cutoff is None:
        return dynamic_cutoff
    return max(float(cutoff), dynamic_cutoff)


def _wavenumber_grid_spacing_cm(wavelength_micron: np.ndarray) -> float:
    wavenumber = wavelength_micron_to_wavenumber_cm(np.asarray(wavelength_micron, dtype=float))
    finite = np.sort(wavenumber[np.isfinite(wavenumber)])
    if finite.size < 2:
        raise ValueError("line selection for LBLRTM line-wing modes requires at least two wavelength pixels")
    spacing = np.diff(finite)
    spacing = spacing[spacing > 0]
    if spacing.size == 0:
        raise ValueError("wavelength grid must span a non-zero range")
    return float(np.nanmedian(spacing))


def _species_scale_setup(
    species_names: tuple[str, ...],
    config: FitConfig,
) -> tuple[tuple[str, ...], dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    fixed_scales = dict(config.fixed_species_scales or {})
    for name, value in fixed_scales.items():
        if value <= 0:
            raise ValueError(f"fixed scale for {name} must be positive")
    fitted_names = tuple(name for name in species_names if name not in fixed_scales)
    initial_scales = config.initial_species_scales or {}
    p0 = np.log(np.array([initial_scales.get(name, 1.0) for name in fitted_names], dtype=float))
    lower = np.full(len(fitted_names), np.log(config.scale_bounds[0]))
    upper = np.full(len(fitted_names), np.log(config.scale_bounds[1]))
    return fitted_names, fixed_scales, p0, lower, upper


def _fix_zero_basis_species(
    config: FitConfig,
    species_names: tuple[str, ...],
    basis_entries: Sequence[tuple[tuple[str, ...], np.ndarray]],
) -> FitConfig:
    """Remove exactly unconstrained scale directions from the optimizer."""

    fixed = dict(config.fixed_species_scales or {})
    initial = config.initial_species_scales or {}
    for name in species_names:
        if name in fixed:
            continue
        active = False
        for local_names, basis in basis_entries:
            if name not in local_names:
                continue
            row = basis[local_names.index(name)]
            if np.any(np.isfinite(row) & (row != 0.0)):
                active = True
                break
        if not active:
            fixed[name] = float(initial.get(name, 1.0))
    return replace(config, fixed_species_scales=fixed)


def _scales_from_params(
    fitted_species_names: tuple[str, ...],
    fixed_species_scales: Mapping[str, float],
    log_scales: np.ndarray,
) -> dict[str, float]:
    scales = dict(fixed_species_scales)
    scales.update(
        {name: float(np.exp(value)) for name, value in zip(fitted_species_names, log_scales, strict=True)}
    )
    return scales


def _fit_metrics(flux: np.ndarray, model_flux: np.ndarray, continuum: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(flux) & np.isfinite(model_flux) & np.isfinite(continuum) & (continuum != 0)
    if not np.any(finite):
        return {"rms_residual": np.nan, "median_abs_residual": np.nan, "corrected_scatter": np.nan}
    residual = flux[finite] - model_flux[finite]
    normalized = residual / continuum[finite]
    return {
        "rms_residual": float(np.sqrt(np.nanmean(residual**2))),
        "median_abs_residual": float(np.nanmedian(np.abs(residual))),
        "corrected_scatter": float(np.nanstd(normalized)),
    }


def _parameter_bound_status(
    names: Sequence[str],
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, str]:
    """Report fitted parameters that terminate numerically on a bound."""

    statuses: dict[str, str] = {}
    for name, value, low, high in zip(names, values, lower, upper, strict=True):
        scale = max(1.0, abs(float(value)))
        tolerance = 1.0e-8 * scale
        if np.isfinite(low) and float(value) <= float(low) + tolerance:
            statuses[str(name)] = "lower"
        elif np.isfinite(high) and float(value) >= float(high) - tolerance:
            statuses[str(name)] = "upper"
    return statuses


def _solve_linear_continuum(
    design: np.ndarray,
    transmission: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    *,
    continuum_prior: np.ndarray | None = None,
    continuum_prior_weight: float = 0.0,
    continuum_prior_fractional_sigma: float = 0.05,
) -> np.ndarray:
    """Solve the multiplicative continuum coefficients for fixed tellurics."""

    design = np.asarray(design, dtype=float)
    transmission = np.asarray(transmission, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    rows = (
        np.all(np.isfinite(design), axis=1)
        & np.isfinite(transmission)
        & np.isfinite(flux)
        & np.isfinite(sigma)
        & (sigma > 0)
    )
    if not np.any(rows):
        raise ValueError("no finite pixels available for continuum solve")

    matrix = design[rows] * transmission[rows, None]
    target = flux[rows]
    matrix = matrix / sigma[rows, None]
    target = target / sigma[rows]

    if continuum_prior_weight > 0 and continuum_prior is not None:
        continuum_prior = np.asarray(continuum_prior, dtype=float)
        prior_sigma = continuum_prior_fractional_sigma * np.maximum(
            np.abs(continuum_prior),
            np.nanmedian(np.abs(continuum_prior)) * 1.0e-6,
        )
        prior_rows = (
            np.all(np.isfinite(design), axis=1)
            & np.isfinite(continuum_prior)
            & np.isfinite(prior_sigma)
            & (prior_sigma > 0)
        )
        if np.any(prior_rows):
            prior_weight = np.sqrt(continuum_prior_weight)
            matrix = np.vstack(
                [
                    matrix,
                    prior_weight * design[prior_rows] / prior_sigma[prior_rows, None],
                ]
            )
            target = np.concatenate(
                [
                    target,
                    prior_weight * continuum_prior[prior_rows] / prior_sigma[prior_rows],
                ]
            )

    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return coefficients


def _continuum_prior_residual(
    continuum: np.ndarray,
    continuum_prior: np.ndarray,
    *,
    weight: float,
    fractional_sigma: float,
) -> np.ndarray:
    prior_sigma = fractional_sigma * np.maximum(
        np.abs(continuum_prior),
        np.nanmedian(np.abs(continuum_prior)) * 1.0e-6,
    )
    return np.sqrt(weight) * (continuum - continuum_prior) / prior_sigma


def _validate_high_resolution_config(config: FitConfig) -> None:
    if config.high_resolution_oversampling <= 0:
        raise ValueError("high_resolution_oversampling must be positive")
    if config.high_resolution_margin_pixels < 0:
        raise ValueError("high_resolution_margin_pixels must be non-negative")
    mode = str(config.high_resolution_rebin_mode).strip().lower().replace("-", "_")
    allowed = {
        "integrate",
        "integral",
        "average",
        "bin_average",
        "pixel_average",
        "center",
        "centre",
        "sample",
        "pixel_center",
        "centre_sample",
        "center_sample",
        "sample_average",
        "sample_mean",
        "average_samples",
        "molecfit",
        "molecfit_overlap",
        "molecfit_rebin",
        "molecfit_average",
    }
    if mode not in allowed:
        raise ValueError(
            "high_resolution_rebin_mode must be 'integrate', 'center', "
            "'sample_average', or 'molecfit_overlap'"
        )
    rt_grid = str(config.radiative_transfer_grid).strip().lower().replace("-", "_")
    if rt_grid not in {"auto", "model", "disabled", "off"}:
        raise ValueError("radiative_transfer_grid must be 'auto' or 'model'")
    if config.radiative_transfer_step_cm is not None:
        step = float(config.radiative_transfer_step_cm)
        if step <= 0 or not np.isfinite(step):
            raise ValueError("radiative_transfer_step_cm must be positive and finite")
    if config.radiative_transfer_max_points < 2:
        raise ValueError("radiative_transfer_max_points must be at least two")
    if config.lblrtm_avmass_amu <= 0 or not np.isfinite(config.lblrtm_avmass_amu):
        raise ValueError("lblrtm_avmass_amu must be positive and finite")


def _uses_native_radiative_transfer_grid(config: FitConfig) -> bool:
    mode = str(config.radiative_transfer_grid).strip().lower().replace("-", "_")
    return config.atmosphere is not None and mode == "auto"


def _prepare_high_resolution_grids(
    observed_wavelength: np.ndarray,
    config: FitConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    SampleAverageRebinPlan | None,
    PiecewiseConstantRebinPlan,
]:
    """Prepare the native RT, internal model, and detector rebin grids."""

    model_wavelength, pixels_per_observed = _high_resolution_model_grid(
        observed_wavelength,
        config,
    )
    basis_wavelength = model_wavelength
    native_to_model_plan = None
    if _uses_native_radiative_transfer_grid(config):
        basis_wavelength, _ = radiative_transfer_wavelength_grid(
            model_wavelength,
            config.atmosphere,
            sample=config.lblrtm_sample,
            alfal0=config.lblrtm_alfal0,
            avmass_amu=config.lblrtm_avmass_amu,
            step_cm=config.radiative_transfer_step_cm,
            max_points=config.radiative_transfer_max_points,
        )
        native_to_model_plan = prepare_sample_average_rebin(
            model_wavelength,
            basis_wavelength,
        )
    overlap_aliases = {
        "molecfit",
        "molecfit_overlap",
        "molecfit_rebin",
    }
    detector_input_wavelength = (
        basis_wavelength
        if str(config.high_resolution_rebin_mode).strip().lower().replace("-", "_")
        in overlap_aliases
        else model_wavelength
    )
    detector_rebin_plan = prepare_piecewise_constant_rebin(
        observed_wavelength,
        detector_input_wavelength,
    )
    return (
        basis_wavelength,
        model_wavelength,
        pixels_per_observed,
        native_to_model_plan,
        detector_rebin_plan,
    )


def _high_resolution_model_grid(
    observed_wavelength: np.ndarray,
    config: FitConfig,
) -> tuple[np.ndarray, float]:
    gaussian_sigma = (
        config.lsf_sigma_bounds[1] if config.fit_lsf_sigma else config.lsf_sigma_pixels
    )
    box_width = (
        config.lsf_box_width_bounds[1]
        if config.fit_lsf_box_width
        else config.lsf_box_width_pixels
    )
    lorentz_fwhm = (
        config.lsf_lorentz_fwhm_bounds[1]
        if config.fit_lsf_lorentz_fwhm
        else config.lsf_lorentz_fwhm_pixels
    )
    if config.lsf_variable_width:
        finite_wavelength = np.asarray(observed_wavelength, dtype=float)
        finite_wavelength = finite_wavelength[np.isfinite(finite_wavelength)]
        reference = config.lsf_reference_wavelength_micron
        if reference is None:
            reference = float(np.nanmedian(finite_wavelength))
        width_scale = float(np.nanmax(finite_wavelength) / reference)
        gaussian_sigma *= width_scale
        box_width *= width_scale
        lorentz_fwhm *= width_scale
    kernel_margin = lsf_kernel_half_width_pixels(
        gaussian_sigma_pixels=gaussian_sigma,
        box_width_pixels=box_width,
        lorentz_fwhm_pixels=lorentz_fwhm,
        kernel_width_fwhm=config.lsf_kernel_width_fwhm,
        molecfit_voigt=config.lsf_molecfit_voigt,
    )
    observed = np.sort(np.asarray(observed_wavelength, dtype=float))
    observed = observed[np.isfinite(observed)]
    wavelength_step = float(np.nanmedian(np.diff(observed)))
    shift_margin = 0.0
    if config.fit_wavelength_shift or config.fit_wavelength_polynomial:
        shift_margin = max(abs(value) for value in config.wavelength_shift_bounds) / wavelength_step
    effective_margin = max(
        float(config.high_resolution_margin_pixels),
        kernel_margin + shift_margin + 1.0,
    )
    return high_resolution_wavelength_grid(
        observed_wavelength,
        oversampling=config.high_resolution_oversampling,
        margin_pixels=effective_margin,
    )


def _radiative_transfer_point_count(
    observed_wavelength: np.ndarray,
    config: FitConfig,
) -> int:
    if not _uses_native_radiative_transfer_grid(config):
        return 0
    model_wavelength, _ = _high_resolution_model_grid(observed_wavelength, config)
    return radiative_transfer_grid_point_count(
        model_wavelength,
        config.atmosphere,
        sample=config.lblrtm_sample,
        alfal0=config.lblrtm_alfal0,
        avmass_amu=config.lblrtm_avmass_amu,
        step_cm=config.radiative_transfer_step_cm,
    )


def _transmission_from_prepared_basis(
    observed_wavelength: np.ndarray,
    species_names: tuple[str, ...],
    basis: np.ndarray,
    *,
    config: FitConfig,
    species_scales: Mapping[str, float],
    airmass: float,
    wavelength_shift: float,
    lsf_sigma_pixels: float,
    lsf_box_width_pixels: float,
    lsf_lorentz_fwhm_pixels: float,
    basis_wavelength: np.ndarray,
    model_wavelength: np.ndarray,
    highres_pixels_per_observed_pixel: float,
    rebin_plan: PiecewiseConstantRebinPlan | None = None,
    native_to_model_plan: SampleAverageRebinPlan | None = None,
) -> np.ndarray:
    if config.high_resolution_grid:
        return transmission_from_high_resolution_basis(
            observed_wavelength,
            basis_wavelength,
            species_names,
            _shift_basis(basis_wavelength, basis, wavelength_shift),
            species_scales=species_scales,
            airmass=airmass,
            lsf_sigma_pixels=lsf_sigma_pixels,
            lsf_box_width_pixels=lsf_box_width_pixels,
            lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
            highres_pixels_per_observed_pixel=highres_pixels_per_observed_pixel,
            lsf_variable_width=config.lsf_variable_width,
            lsf_reference_wavelength_micron=config.lsf_reference_wavelength_micron,
            lsf_kernel_width_fwhm=config.lsf_kernel_width_fwhm,
            lsf_molecfit_voigt=config.lsf_molecfit_voigt,
            rebin_mode=config.high_resolution_rebin_mode,
            rebin_plan=rebin_plan,
            model_wavelength_micron=model_wavelength,
            native_to_model_plan=native_to_model_plan,
        )
    return transmission_from_basis(
        species_names,
        _shift_basis(observed_wavelength, basis, wavelength_shift),
        species_scales=species_scales,
        airmass=airmass,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        wavelength_micron=observed_wavelength,
        lsf_variable_width=config.lsf_variable_width,
        lsf_reference_wavelength_micron=config.lsf_reference_wavelength_micron,
        lsf_kernel_width_fwhm=config.lsf_kernel_width_fwhm,
        lsf_molecfit_voigt=config.lsf_molecfit_voigt,
    )


def fit_tellurics(
    spectrum: Spectrum,
    *,
    line_list: LineList,
    config: FitConfig | None = None,
    fit_mask: np.ndarray | None = None,
) -> TelluricFitResult:
    """Fit telluric transmission and return a corrected spectrum.

    The current version fits species scale factors and a multiplicative
    polynomial continuum. Future versions should add wavelength-shift and LSF
    fitting once real HITRAN/profile ingestion is in place.
    """

    config = FitConfig() if config is None else config
    if config.scale_bounds[0] <= 0 or config.scale_bounds[1] <= config.scale_bounds[0]:
        raise ValueError("scale_bounds must be positive and increasing")
    _validate_lsf_fit_config(config)
    global_wavelength_order = _global_wavelength_polynomial_order(config)
    if global_wavelength_order is not None and config.wavelength_shift_bounds[1] <= config.wavelength_shift_bounds[0]:
        raise ValueError("wavelength_shift_bounds must be increasing")
    if config.fit_segment_wavelength_shifts:
        raise ValueError("fit_segment_wavelength_shifts is only supported by fit_telluric_segments")
    if config.fit_segment_wavelength_polynomial:
        raise ValueError("fit_segment_wavelength_polynomial is only supported by fit_telluric_segments")
    if config.solve_continuum_linear and config.loss != "linear":
        raise ValueError("solve_continuum_linear currently requires loss='linear'")
    if config.max_nfev is not None and config.max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    _validate_optimizer_tolerances(config)
    _validate_high_resolution_config(config)

    spectrum = spectrum.to_unit("micron")
    sort_order = np.argsort(spectrum.wavelength)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != spectrum.wavelength.shape:
            raise ValueError("fit_mask must have the same shape as the spectrum")
        fit_mask = fit_mask[sort_order]
    spectrum = spectrum.sorted()
    valid = spectrum.valid.copy()
    valid &= _windows_mask(spectrum.wavelength, config.fit_ranges, config.exclude_ranges)
    if fit_mask is not None:
        valid &= fit_mask
    if np.count_nonzero(valid) < config.continuum_order + 2:
        raise ValueError("not enough valid pixels to fit")

    wavelength_fit = spectrum.wavelength[valid]
    flux_fit = spectrum.flux[valid]
    _, design_fit = _poly_design(wavelength_fit, config.continuum_order)
    _, design_all = _poly_design(spectrum.wavelength, config.continuum_order)

    selected_lines = line_list.select_range(
        np.nanmin(wavelength_fit),
        np.nanmax(wavelength_fit),
        margin=_line_selection_margin(wavelength_fit, config),
    )
    basis_wavelength = spectrum.wavelength
    model_wavelength = spectrum.wavelength
    highres_pixels_per_observed_pixel = 1.0
    native_to_model_plan = None
    rebin_plan = None
    if config.high_resolution_grid:
        (
            basis_wavelength,
            model_wavelength,
            highres_pixels_per_observed_pixel,
            native_to_model_plan,
            rebin_plan,
        ) = _prepare_high_resolution_grids(
            spectrum.wavelength,
            config,
        )
    wavelength_bounds = (
        float(np.nanmin(spectrum.wavelength)),
        float(np.nanmax(spectrum.wavelength)),
    )
    wavelength_design_all = wavelength_design_fit = wavelength_design_basis = None
    if global_wavelength_order is not None:
        _, wavelength_design_all = _poly_design_over_bounds(
            spectrum.wavelength,
            global_wavelength_order,
            wavelength_bounds,
        )
        wavelength_design_fit = wavelength_design_all[valid]
        _, wavelength_design_basis = _poly_design_over_bounds(
            basis_wavelength,
            global_wavelength_order,
            wavelength_bounds,
        )

    if config.atmosphere is not None:
        selected_components = _components_for_selected_lines(config.components, selected_lines)
        physical_config = PhysicalModelConfig(
            chunk_size=config.chunk_size,
            partition_exponent=config.partition_exponent,
            partition_table=config.partition_table,
            h2o_continuum=config.h2o_continuum,
            h2o_continuum_foreign_closure=config.h2o_continuum_foreign_closure,
            line_cutoff_cm=config.line_cutoff_cm,
            subtract_cutoff_profile=config.subtract_cutoff_profile,
            line_taper_cm=config.line_taper_cm,
            line_wing_mode=config.line_wing_mode,
            lblrtm_sample=config.lblrtm_sample,
            lblrtm_alfal0=config.lblrtm_alfal0,
            lblrtm_avmass_amu=config.lblrtm_avmass_amu,
            lblrtm_hwf3=config.lblrtm_hwf3,
            rayleigh=config.rayleigh,
            rayleigh_xrayl=config.rayleigh_xrayl,
            n2_continuum=config.n2_continuum,
            n2_continuum_xn2cn=config.n2_continuum_xn2cn,
            o2_continuum=config.o2_continuum,
            o2_continuum_xo2cn=config.o2_continuum_xo2cn,
            components=selected_components,
        )
        species_names, basis_all = physical_optical_depth_basis(
            basis_wavelength,
            selected_lines,
            config.atmosphere,
            physical_config,
            species=config.species,
        )
        basis_airmass = config.airmass
    else:
        species_names, basis_all = optical_depth_basis(
            basis_wavelength,
            selected_lines,
            species=config.species,
            chunk_size=config.chunk_size or 512,
        )
        basis_airmass = config.airmass
    basis_fit = basis_all[:, valid] if not config.high_resolution_grid else np.zeros((basis_all.shape[0], 0))

    scale_config = _fix_zero_basis_species(config, species_names, ((species_names, basis_all),))
    fitted_species_names, fixed_species_scales, p0_scales, lower_scales, upper_scales = _species_scale_setup(
        species_names,
        scale_config,
    )
    continuum0 = np.zeros(config.continuum_order + 1, dtype=float)
    continuum0[0] = np.nanmedian(flux_fit)
    if not np.isfinite(continuum0[0]) or continuum0[0] == 0:
        continuum0[0] = 1.0
    p0_parts = [p0_scales]
    lower_parts = [
        lower_scales,
    ]
    upper_parts = [
        upper_scales,
    ]
    if not config.solve_continuum_linear:
        p0_parts.append(continuum0)
        lower_parts.append(np.full(config.continuum_order + 1, -np.inf))
        upper_parts.append(np.full(config.continuum_order + 1, np.inf))
    if global_wavelength_order is not None:
        p0_parts.append(
            _wavelength_coefficients_initial(
                global_wavelength_order,
                config.initial_wavelength_shift,
            )
        )
        lower_parts.append(
            np.full(global_wavelength_order + 1, config.wavelength_shift_bounds[0])
        )
        upper_parts.append(
            np.full(global_wavelength_order + 1, config.wavelength_shift_bounds[1])
        )
    if config.fit_lsf_sigma:
        p0_parts.append(np.array([_initial_lsf_width(config.lsf_sigma_pixels, config.lsf_sigma_bounds)], dtype=float))
        lower_parts.append(np.array([config.lsf_sigma_bounds[0]], dtype=float))
        upper_parts.append(np.array([config.lsf_sigma_bounds[1]], dtype=float))
    if config.fit_lsf_box_width:
        p0_parts.append(
            np.array([_initial_lsf_width(config.lsf_box_width_pixels, config.lsf_box_width_bounds)], dtype=float)
        )
        lower_parts.append(np.array([config.lsf_box_width_bounds[0]], dtype=float))
        upper_parts.append(np.array([config.lsf_box_width_bounds[1]], dtype=float))
    if config.fit_lsf_lorentz_fwhm:
        p0_parts.append(
            np.array(
                [
                    _initial_lsf_width(
                        config.lsf_lorentz_fwhm_pixels,
                        config.lsf_lorentz_fwhm_bounds,
                    )
                ],
                dtype=float,
            )
        )
        lower_parts.append(np.array([config.lsf_lorentz_fwhm_bounds[0]], dtype=float))
        upper_parts.append(np.array([config.lsf_lorentz_fwhm_bounds[1]], dtype=float))

    p0 = np.concatenate(p0_parts)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    if spectrum.uncertainty is None:
        # Molecfit uses DEFAULT_ERROR=0.01 times the mean fitted flux when the
        # input has no uncertainty column.
        mean_flux = float(np.nanmean(flux_fit))
        sigma = np.full_like(flux_fit, 0.01 * mean_flux if mean_flux > 0 else 1.0)
    else:
        sigma = spectrum.uncertainty[valid]

    def residual(params: np.ndarray) -> np.ndarray:
        log_scales = params[: len(fitted_species_names)]
        continuum_start = len(fitted_species_names)
        continuum_stop = continuum_start
        continuum_coeff = None
        if not config.solve_continuum_linear:
            continuum_stop = continuum_start + config.continuum_order + 1
            continuum_coeff = params[continuum_start:continuum_stop]
        cursor = continuum_stop
        if global_wavelength_order is not None:
            stop = cursor + global_wavelength_order + 1
            wavelength_coefficients = np.asarray(params[cursor:stop], dtype=float)
            cursor = stop
            wavelength_shift_fit: float | np.ndarray = (
                wavelength_design_fit @ wavelength_coefficients
            )
            wavelength_shift_basis: float | np.ndarray = (
                wavelength_design_basis @ wavelength_coefficients
            )
        else:
            wavelength_shift_fit = 0.0
            wavelength_shift_basis = 0.0
        lsf_sigma_pixels = float(params[cursor]) if config.fit_lsf_sigma else config.lsf_sigma_pixels
        cursor += int(config.fit_lsf_sigma)
        lsf_box_width_pixels = (
            float(params[cursor]) if config.fit_lsf_box_width else config.lsf_box_width_pixels
        )
        cursor += int(config.fit_lsf_box_width)
        lsf_lorentz_fwhm_pixels = (
            float(params[cursor]) if config.fit_lsf_lorentz_fwhm else config.lsf_lorentz_fwhm_pixels
        )
        scales = _scales_from_params(fitted_species_names, fixed_species_scales, log_scales)
        if config.high_resolution_grid:
            transmission = _transmission_from_prepared_basis(
                spectrum.wavelength,
                species_names,
                basis_all,
                config=config,
                species_scales=scales,
                airmass=basis_airmass,
                wavelength_shift=wavelength_shift_basis,
                lsf_sigma_pixels=lsf_sigma_pixels,
                lsf_box_width_pixels=lsf_box_width_pixels,
                lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
                basis_wavelength=basis_wavelength,
                model_wavelength=model_wavelength,
                highres_pixels_per_observed_pixel=highres_pixels_per_observed_pixel,
                rebin_plan=rebin_plan,
                native_to_model_plan=native_to_model_plan,
            )[valid]
        else:
            transmission = transmission_from_basis(
                species_names,
                _shift_basis(wavelength_fit, basis_fit, wavelength_shift_fit),
                species_scales=scales,
                airmass=basis_airmass,
                lsf_sigma_pixels=lsf_sigma_pixels,
                lsf_box_width_pixels=lsf_box_width_pixels,
                lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
                wavelength_micron=wavelength_fit,
                lsf_variable_width=config.lsf_variable_width,
                lsf_reference_wavelength_micron=config.lsf_reference_wavelength_micron,
                lsf_kernel_width_fwhm=config.lsf_kernel_width_fwhm,
                lsf_molecfit_voigt=config.lsf_molecfit_voigt,
            )
        if config.solve_continuum_linear:
            continuum_coeff = _solve_linear_continuum(
                design_fit,
                transmission,
                flux_fit,
                sigma,
            )
        continuum = design_fit @ continuum_coeff
        return (flux_fit - continuum * transmission) / sigma

    fit = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        method="trf",
        loss=config.loss,
        f_scale=config.f_scale,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=config.max_nfev,
    )

    log_scales = fit.x[: len(fitted_species_names)]
    continuum_start = len(fitted_species_names)
    continuum_stop = continuum_start
    continuum_coeff = None
    if not config.solve_continuum_linear:
        continuum_stop = continuum_start + config.continuum_order + 1
        continuum_coeff = fit.x[continuum_start:continuum_stop]
    cursor = continuum_stop
    if global_wavelength_order is not None:
        stop = cursor + global_wavelength_order + 1
        wavelength_coefficients = np.asarray(fit.x[cursor:stop], dtype=float)
        cursor = stop
        wavelength_shift_all: float | np.ndarray = wavelength_design_all @ wavelength_coefficients
        wavelength_shift_basis: float | np.ndarray = wavelength_design_basis @ wavelength_coefficients
    else:
        wavelength_coefficients = np.array([0.0], dtype=float)
        wavelength_shift_all = 0.0
        wavelength_shift_basis = 0.0
    wavelength_shift = float(np.nanmedian(wavelength_shift_all))
    lsf_sigma_pixels = float(fit.x[cursor]) if config.fit_lsf_sigma else config.lsf_sigma_pixels
    cursor += int(config.fit_lsf_sigma)
    lsf_box_width_pixels = float(fit.x[cursor]) if config.fit_lsf_box_width else config.lsf_box_width_pixels
    cursor += int(config.fit_lsf_box_width)
    lsf_lorentz_fwhm_pixels = (
        float(fit.x[cursor]) if config.fit_lsf_lorentz_fwhm else config.lsf_lorentz_fwhm_pixels
    )
    species_scales = _scales_from_params(fitted_species_names, fixed_species_scales, log_scales)
    transmission = _transmission_from_prepared_basis(
        spectrum.wavelength,
        species_names,
        basis_all,
        config=config,
        species_scales=species_scales,
        airmass=basis_airmass,
        wavelength_shift=wavelength_shift_basis,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        basis_wavelength=basis_wavelength,
        model_wavelength=model_wavelength,
        highres_pixels_per_observed_pixel=highres_pixels_per_observed_pixel,
        rebin_plan=rebin_plan,
        native_to_model_plan=native_to_model_plan,
    )
    if config.solve_continuum_linear:
        continuum_coeff = _solve_linear_continuum(
            design_fit,
            transmission[valid],
            flux_fit,
            sigma,
        )
    continuum = design_all @ continuum_coeff
    model_flux = continuum * transmission
    parameter_names = [f"log_scale:{name}" for name in fitted_species_names]
    active_transmission_indices = list(range(len(fitted_species_names)))
    parameter_cursor = len(fitted_species_names)
    if not config.solve_continuum_linear:
        parameter_names.extend(
            f"continuum:{degree}" for degree in range(config.continuum_order + 1)
        )
        parameter_cursor += config.continuum_order + 1
    if global_wavelength_order is not None:
        if config.fit_wavelength_shift:
            parameter_names.append("wavelength_shift_micron")
        else:
            parameter_names.extend(
                f"wavelength_coefficient:{degree}"
                for degree in range(global_wavelength_order + 1)
            )
        active_transmission_indices.extend(
            range(parameter_cursor, parameter_cursor + global_wavelength_order + 1)
        )
        parameter_cursor += global_wavelength_order + 1
    if config.fit_lsf_sigma:
        parameter_names.append("lsf_sigma_pixels")
        active_transmission_indices.append(parameter_cursor)
        parameter_cursor += 1
    if config.fit_lsf_box_width:
        parameter_names.append("lsf_box_width_pixels")
        active_transmission_indices.append(parameter_cursor)
        parameter_cursor += 1
    if config.fit_lsf_lorentz_fwhm:
        parameter_names.append("lsf_lorentz_fwhm_pixels")
        active_transmission_indices.append(parameter_cursor)

    bound_status = _parameter_bound_status(parameter_names, fit.x, lower, upper)

    parameter_covariance = None
    parameter_standard_errors: dict[str, float] = {}
    species_scale_uncertainties: dict[str, float] = {}
    transmission_uncertainty = None
    reduced_chi_square = np.nan
    covariance_rank = 0
    if config.estimate_uncertainties:
        parameter_covariance, reduced_chi_square, covariance_rank = _linearized_parameter_covariance(
            fit.jac,
            fit.cost,
            fit.fun.size,
            fit.x.size,
        )
        standard_errors = np.sqrt(np.maximum(0.0, np.diag(parameter_covariance)))
        parameter_standard_errors = {
            name: float(error)
            for name, error in zip(parameter_names, standard_errors, strict=True)
        }
        species_scale_uncertainties = {
            name: float(species_scales[name] * standard_errors[index])
            for index, name in enumerate(fitted_species_names)
        }

        def transmission_for_parameters(parameters: np.ndarray) -> np.ndarray:
            local_log_scales = parameters[: len(fitted_species_names)]
            local_cursor = len(fitted_species_names)
            if not config.solve_continuum_linear:
                local_cursor += config.continuum_order + 1
            if global_wavelength_order is not None:
                local_stop = local_cursor + global_wavelength_order + 1
                local_wavelength_coefficients = np.asarray(
                    parameters[local_cursor:local_stop],
                    dtype=float,
                )
                local_cursor = local_stop
                local_shift: float | np.ndarray = (
                    wavelength_design_basis @ local_wavelength_coefficients
                )
            else:
                local_shift = 0.0
            local_sigma = float(parameters[local_cursor]) if config.fit_lsf_sigma else config.lsf_sigma_pixels
            local_cursor += int(config.fit_lsf_sigma)
            local_box = (
                float(parameters[local_cursor])
                if config.fit_lsf_box_width
                else config.lsf_box_width_pixels
            )
            local_cursor += int(config.fit_lsf_box_width)
            local_lorentz = (
                float(parameters[local_cursor])
                if config.fit_lsf_lorentz_fwhm
                else config.lsf_lorentz_fwhm_pixels
            )
            local_scales = _scales_from_params(
                fitted_species_names,
                fixed_species_scales,
                local_log_scales,
            )
            return _transmission_from_prepared_basis(
                spectrum.wavelength,
                species_names,
                basis_all,
                config=config,
                species_scales=local_scales,
                airmass=basis_airmass,
                wavelength_shift=local_shift,
                lsf_sigma_pixels=local_sigma,
                lsf_box_width_pixels=local_box,
                lsf_lorentz_fwhm_pixels=local_lorentz,
                basis_wavelength=basis_wavelength,
                model_wavelength=model_wavelength,
                highres_pixels_per_observed_pixel=highres_pixels_per_observed_pixel,
                rebin_plan=rebin_plan,
                native_to_model_plan=native_to_model_plan,
            )

        transmission_uncertainty = _finite_difference_output_uncertainty(
            transmission_for_parameters,
            fit.x,
            parameter_covariance,
            lower,
            upper,
            np.asarray(active_transmission_indices, dtype=int),
        )

    corrected = correct_spectrum(
        spectrum,
        transmission,
        transmission_uncertainty=transmission_uncertainty,
        min_transmission=config.min_transmission,
    )
    metrics = _fit_metrics(spectrum.flux, model_flux, continuum)
    provenance = build_fit_provenance(
        spectrum,
        line_list=line_list,
        selected_line_list=selected_lines,
        config=config,
        fit_pixel_counts=(int(np.count_nonzero(valid)),),
    )

    return TelluricFitResult(
        spectrum=spectrum,
        corrected=corrected,
        transmission=transmission,
        continuum=continuum,
        model_flux=model_flux,
        species_scales=species_scales,
        wavelength_shift=wavelength_shift,
        wavelength_coefficients=wavelength_coefficients,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        continuum_coefficients=continuum_coeff,
        metrics=metrics,
        success=bool(fit.success),
        message=str(fit.message),
        cost=float(fit.cost),
        nfev=int(fit.nfev),
        parameter_names=tuple(parameter_names),
        parameter_covariance=parameter_covariance,
        parameter_standard_errors=parameter_standard_errors,
        species_scale_uncertainties=species_scale_uncertainties,
        transmission_uncertainty=transmission_uncertainty,
        reduced_chi_square=reduced_chi_square,
        covariance_rank=covariance_rank,
        fit_mask=valid.copy(),
        parameter_bound_status=bound_status,
        provenance=provenance,
    )


@dataclass(frozen=True)
class _PreparedSegment:
    spectrum: Spectrum
    valid: np.ndarray
    design_all: np.ndarray
    design_fit: np.ndarray
    flux_fit: np.ndarray
    sigma: np.ndarray
    species_names: tuple[str, ...]
    basis_all: np.ndarray
    basis_fit: np.ndarray
    basis_wavelength: np.ndarray
    model_wavelength: np.ndarray
    wavelength_shift_design_all: np.ndarray
    wavelength_shift_design_fit: np.ndarray
    wavelength_shift_design_basis: np.ndarray
    highres_pixels_per_observed_pixel: float = 1.0
    continuum_prior_fit: np.ndarray | None = None
    rebin_plan: PiecewiseConstantRebinPlan | None = None
    native_to_model_plan: SampleAverageRebinPlan | None = None


def _multi_fit_jacobian_sparsity(
    prepared: Sequence[_PreparedSegment],
    *,
    n_parameters: int,
    n_fitted_species: int,
    continuum_block: int,
    solve_continuum_linear: bool,
    global_wavelength_order: int | None,
    segment_wavelength_order: int | None,
    fit_lsf_sigma: bool,
    fit_lsf_box_width: bool,
    fit_lsf_lorentz_fwhm: bool,
    continuum_prior_weight: float,
) -> object | None:
    """Return the exact residual/parameter dependency pattern for segments."""

    if len(prepared) < 2:
        return None
    row_ranges = []
    row_cursor = 0
    for segment in prepared:
        rows = segment.flux_fit.size
        if continuum_prior_weight > 0 and segment.continuum_prior_fit is not None:
            rows *= 2
        row_ranges.append(slice(row_cursor, row_cursor + rows))
        row_cursor += rows

    pattern = lil_matrix((row_cursor, n_parameters), dtype=bool)
    column_cursor = n_fitted_species
    local_continuum_columns = []
    if not solve_continuum_linear:
        for _ in prepared:
            local_continuum_columns.append(slice(column_cursor, column_cursor + continuum_block))
            column_cursor += continuum_block
    else:
        local_continuum_columns = [None] * len(prepared)

    global_columns = list(range(n_fitted_species))
    if global_wavelength_order is not None:
        global_columns.extend(
            range(column_cursor, column_cursor + global_wavelength_order + 1)
        )
        column_cursor += global_wavelength_order + 1

    local_wavelength_columns = [None] * len(prepared)
    if segment_wavelength_order is not None:
        block = segment_wavelength_order + 1
        local_wavelength_columns = []
        for _ in prepared:
            local_wavelength_columns.append(slice(column_cursor, column_cursor + block))
            column_cursor += block

    for enabled in (fit_lsf_sigma, fit_lsf_box_width, fit_lsf_lorentz_fwhm):
        if enabled:
            global_columns.append(column_cursor)
            column_cursor += 1
    if column_cursor != n_parameters:
        raise RuntimeError("Jacobian sparsity parameter layout does not match the fit")

    if global_columns:
        pattern[:, global_columns] = True
    has_local_columns = False
    for rows, continuum_columns, wavelength_columns in zip(
        row_ranges,
        local_continuum_columns,
        local_wavelength_columns,
        strict=True,
    ):
        if continuum_columns is not None:
            pattern[rows, continuum_columns] = True
            has_local_columns = True
        if wavelength_columns is not None:
            pattern[rows, wavelength_columns] = True
            has_local_columns = True
    return pattern.tocsr() if has_local_columns else None


def _jacobian_column_groups(pattern) -> tuple[np.ndarray, ...]:
    """Greedily group columns whose nonzero residual rows do not overlap."""

    columns = pattern.tocsc()
    groups: list[list[int]] = []
    occupied_rows: list[set[int]] = []
    for column in range(columns.shape[1]):
        rows = set(columns.indices[columns.indptr[column]:columns.indptr[column + 1]].tolist())
        for group, occupied in zip(groups, occupied_rows, strict=True):
            if rows.isdisjoint(occupied):
                group.append(column)
                occupied.update(rows)
                break
        else:
            groups.append([column])
            occupied_rows.append(set(rows))
    return tuple(np.asarray(group, dtype=int) for group in groups)


def _bounded_forward_steps(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return stable two-point finite-difference steps inside parameter bounds."""

    x = np.asarray(x, dtype=float)
    relative_step = np.sqrt(np.finfo(float).eps)
    direction = np.where(x >= 0, 1.0, -1.0)
    step = relative_step * direction * np.maximum(1.0, np.abs(x))
    for index in range(x.size):
        forward = upper[index] - x[index]
        backward = x[index] - lower[index]
        requested = abs(step[index])
        if step[index] > 0 and forward >= requested:
            continue
        if step[index] < 0 and backward >= requested:
            continue
        if backward >= requested:
            step[index] = -requested
        elif forward >= requested:
            step[index] = requested
        elif forward >= backward and forward > 0:
            step[index] = forward
        elif backward > 0:
            step[index] = -backward
        else:
            raise ValueError("cannot finite-difference a parameter fixed at both bounds")
    return (x + step) - x


def _linearized_parameter_covariance(
    jacobian: np.ndarray,
    cost: float,
    n_residuals: int,
    n_parameters: int,
) -> tuple[np.ndarray, float, int]:
    """Estimate a scaled local covariance from a weighted least-squares fit.

    The covariance is conditional on the supplied atmosphere, line database,
    fit mask, and LSF model. It quantifies local statistical uncertainty, not
    systematic uncertainty in those inputs.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (n_residuals, n_parameters):
        raise ValueError("jacobian shape does not match residual and parameter counts")
    degrees_of_freedom = max(0, int(n_residuals) - int(n_parameters))
    reduced_chi_square = (
        float(2.0 * cost / degrees_of_freedom)
        if degrees_of_freedom > 0
        else np.nan
    )
    if n_parameters == 0:
        return np.zeros((0, 0), dtype=float), reduced_chi_square, 0

    # Rank and conditioning must not depend on whether a parameter is measured
    # in microns, pixels, or logarithmic column scale. Normalize every
    # Jacobian column before the SVD, then transform the covariance back to the
    # original parameter units. Without this step, wavelength derivatives can
    # dominate molecular-scale columns by many orders of magnitude and create
    # false rank-deficiency reports.
    column_scales = np.linalg.norm(jacobian, axis=0)
    active_columns = np.isfinite(column_scales) & (column_scales > 0)
    safe_scales = np.where(active_columns, column_scales, 1.0)
    scaled_jacobian = jacobian / safe_scales[None, :]
    _, singular_values, vt = np.linalg.svd(scaled_jacobian, full_matrices=False)
    if singular_values.size == 0:
        return np.full((n_parameters, n_parameters), np.nan), reduced_chi_square, 0
    # The optimizer Jacobian is obtained with finite differences whose useful
    # relative precision is O(sqrt(machine epsilon)), not machine epsilon.
    # Using the latter mistakes differencing noise for an identifiable
    # parameter direction in nearly degenerate molecular/continuum fits.
    threshold = np.sqrt(np.finfo(float).eps) * max(jacobian.shape) * singular_values[0]
    keep = singular_values > threshold
    rank = int(np.count_nonzero(keep))
    inverse_squares = np.zeros_like(singular_values)
    inverse_squares[keep] = 1.0 / singular_values[keep] ** 2
    scaled_covariance = (vt.T * inverse_squares) @ vt
    covariance = scaled_covariance / np.outer(safe_scales, safe_scales)
    if np.isfinite(reduced_chi_square):
        covariance *= reduced_chi_square
    rank = min(rank, int(np.count_nonzero(active_columns)))
    if rank < n_parameters:
        # A pseudoinverse can assign a deceptively small variance to a null
        # direction. A rank-deficient fit has no identifiable full covariance,
        # so expose that limitation instead of reporting false precision.
        covariance[:] = np.nan
    return covariance, reduced_chi_square, rank


def _finite_difference_output_uncertainty(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    covariance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    """Propagate a parameter covariance through an array-valued function."""

    parameters = np.asarray(parameters, dtype=float)
    baseline = np.asarray(function(parameters), dtype=float)
    if active_indices.size == 0:
        return np.zeros_like(baseline)
    if not np.all(np.isfinite(covariance)):
        return np.full_like(baseline, np.nan, dtype=float)
    steps = _bounded_forward_steps(parameters, lower, upper)
    output_jacobian = np.zeros((baseline.size, parameters.size), dtype=float)
    for index in np.asarray(active_indices, dtype=int):
        shifted = parameters.copy()
        shifted[index] += steps[index]
        output_jacobian[:, index] = (
            np.asarray(function(shifted), dtype=float) - baseline
        ) / steps[index]
    variance = np.einsum(
        "ij,jk,ik->i",
        output_jacobian,
        covariance,
        output_jacobian,
        optimize=True,
    )
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.nanmax(np.abs(variance)))
    variance = np.where((variance < 0) & (variance > -tolerance), 0.0, variance)
    return np.sqrt(np.where(variance >= 0, variance, np.nan))


def _grouped_dense_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    pattern,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a dense finite-difference Jacobian using independent column groups."""

    columns = pattern.tocsc()
    groups = _jacobian_column_groups(pattern)

    def jacobian(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        baseline = function(x)
        steps = _bounded_forward_steps(x, lower, upper)
        result = np.zeros((baseline.size, x.size), dtype=float)
        for group in groups:
            shifted = x.copy()
            shifted[group] += steps[group]
            difference = function(shifted) - baseline
            for column in group:
                rows = columns.indices[columns.indptr[column]:columns.indptr[column + 1]]
                result[rows, column] = difference[rows] / steps[column]
        return result

    return jacobian


def _prepare_multi_fit_segment(
    spectrum: Spectrum,
    fit_mask: np.ndarray | None,
    continuum_prior: np.ndarray | None,
    *,
    line_list: LineList,
    config: FitConfig,
    global_wavelength_order: int | None,
    global_wavelength_bounds: tuple[float, float] | None,
    segment_wavelength_order: int | None,
) -> _PreparedSegment:
    unit_spectrum = spectrum.to_unit("micron")
    order = np.argsort(unit_spectrum.wavelength)
    segment = unit_spectrum.sorted()
    valid = segment.valid.copy()
    valid &= _windows_mask(segment.wavelength, config.fit_ranges, config.exclude_ranges)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != valid.shape:
            raise ValueError("each fit mask must match its spectrum")
        valid &= fit_mask[order]
    continuum_prior_fit = None
    if continuum_prior is not None:
        continuum_prior = np.asarray(continuum_prior, dtype=float)
        if continuum_prior.shape != valid.shape:
            raise ValueError("each continuum prior must match its spectrum")
        continuum_prior_fit = continuum_prior[order][valid]
    if np.count_nonzero(valid) < config.continuum_order + 2:
        raise ValueError("not enough valid pixels to fit one of the segments")

    wavelength_fit = segment.wavelength[valid]
    flux_fit = segment.flux[valid]
    _, design_fit = _poly_design(wavelength_fit, config.continuum_order)
    _, design_all = _poly_design(segment.wavelength, config.continuum_order)
    selected_lines = line_list.select_range(
        np.nanmin(wavelength_fit),
        np.nanmax(wavelength_fit),
        margin=_line_selection_margin(wavelength_fit, config),
    )

    basis_wavelength = segment.wavelength
    model_wavelength = segment.wavelength
    highres_pixels_per_observed_pixel = 1.0
    native_to_model_plan = None
    rebin_plan = None
    if config.high_resolution_grid:
        (
            basis_wavelength,
            model_wavelength,
            highres_pixels_per_observed_pixel,
            native_to_model_plan,
            rebin_plan,
        ) = _prepare_high_resolution_grids(
            segment.wavelength,
            config,
        )
    wavelength_shift_order = (
        segment_wavelength_order
        if segment_wavelength_order is not None
        else (global_wavelength_order if global_wavelength_order is not None else 0)
    )
    if global_wavelength_order is not None:
        if global_wavelength_bounds is None:
            raise RuntimeError("global wavelength bounds were not prepared")
        _, wavelength_shift_design_all = _poly_design_over_bounds(
            segment.wavelength,
            wavelength_shift_order,
            global_wavelength_bounds,
        )
        _, wavelength_shift_design_basis = _poly_design_over_bounds(
            basis_wavelength,
            wavelength_shift_order,
            global_wavelength_bounds,
        )
    else:
        _, wavelength_shift_design_all = _poly_design(
            segment.wavelength,
            wavelength_shift_order,
        )
        _, wavelength_shift_design_basis = _poly_design(
            basis_wavelength,
            wavelength_shift_order,
        )
    wavelength_shift_design_fit = wavelength_shift_design_all[valid]

    if config.atmosphere is not None:
        selected_components = _components_for_selected_lines(config.components, selected_lines)
        physical_config = PhysicalModelConfig(
            chunk_size=config.chunk_size,
            partition_exponent=config.partition_exponent,
            partition_table=config.partition_table,
            h2o_continuum=config.h2o_continuum,
            h2o_continuum_foreign_closure=config.h2o_continuum_foreign_closure,
            line_cutoff_cm=config.line_cutoff_cm,
            subtract_cutoff_profile=config.subtract_cutoff_profile,
            line_taper_cm=config.line_taper_cm,
            line_wing_mode=config.line_wing_mode,
            lblrtm_sample=config.lblrtm_sample,
            lblrtm_alfal0=config.lblrtm_alfal0,
            lblrtm_avmass_amu=config.lblrtm_avmass_amu,
            lblrtm_hwf3=config.lblrtm_hwf3,
            rayleigh=config.rayleigh,
            rayleigh_xrayl=config.rayleigh_xrayl,
            n2_continuum=config.n2_continuum,
            n2_continuum_xn2cn=config.n2_continuum_xn2cn,
            o2_continuum=config.o2_continuum,
            o2_continuum_xo2cn=config.o2_continuum_xo2cn,
            components=selected_components,
        )
        species_names, basis_all = physical_optical_depth_basis(
            basis_wavelength,
            selected_lines,
            config.atmosphere,
            physical_config,
            species=config.species,
        )
    else:
        species_names, basis_all = optical_depth_basis(
            basis_wavelength,
            selected_lines,
            species=config.species,
            chunk_size=config.chunk_size or 512,
        )
    basis_fit = basis_all[:, valid] if not config.high_resolution_grid else np.zeros((basis_all.shape[0], 0))
    if segment.uncertainty is None:
        mean_flux = float(np.nanmean(flux_fit))
        sigma = np.full_like(flux_fit, 0.01 * mean_flux if mean_flux > 0 else 1.0)
    else:
        sigma = segment.uncertainty[valid]

    return _PreparedSegment(
        spectrum=segment,
        valid=valid,
        design_all=design_all,
        design_fit=design_fit,
        flux_fit=flux_fit,
        sigma=sigma,
        species_names=species_names,
        basis_all=basis_all,
        basis_fit=basis_fit,
        basis_wavelength=basis_wavelength,
        model_wavelength=model_wavelength,
        wavelength_shift_design_all=wavelength_shift_design_all,
        wavelength_shift_design_fit=wavelength_shift_design_fit,
        wavelength_shift_design_basis=wavelength_shift_design_basis,
        highres_pixels_per_observed_pixel=highres_pixels_per_observed_pixel,
        continuum_prior_fit=continuum_prior_fit,
        rebin_plan=rebin_plan,
        native_to_model_plan=native_to_model_plan,
    )


def fit_telluric_segments(
    spectra: Sequence[Spectrum],
    *,
    line_list: LineList,
    config: FitConfig | None = None,
    fit_masks: Sequence[np.ndarray | None] | None = None,
    continuum_priors: Sequence[np.ndarray | None] | None = None,
    global_wavelength_bounds: tuple[float, float] | None = None,
) -> MultiTelluricFitResult:
    """Fit several spectral segments with shared telluric scales.

    This mirrors the most important Molecfit fitting pattern: molecular column
    scale factors are global, while each order/chip gets its own continuum
    polynomial. It is especially important for heavily absorbed segments where
    a single segment cannot distinguish a low continuum from deep telluric
    absorption.
    """

    if not spectra:
        raise ValueError("spectra must contain at least one segment")
    config = FitConfig() if config is None else config
    if config.scale_bounds[0] <= 0 or config.scale_bounds[1] <= config.scale_bounds[0]:
        raise ValueError("scale_bounds must be positive and increasing")
    _validate_lsf_fit_config(config)
    global_wavelength_order = _global_wavelength_polynomial_order(config)
    if global_wavelength_order is not None and config.wavelength_shift_bounds[1] <= config.wavelength_shift_bounds[0]:
        raise ValueError("wavelength_shift_bounds must be increasing")
    segment_wavelength_order = _segment_wavelength_polynomial_order(config)
    if global_wavelength_order is not None and segment_wavelength_order is not None:
        raise ValueError(
            "global wavelength fitting cannot be combined with per-segment wavelength fitting"
        )
    if config.wavelength_shift_bounds[1] <= config.wavelength_shift_bounds[0]:
        raise ValueError("wavelength_shift_bounds must be increasing")
    if config.continuum_prior_weight < 0:
        raise ValueError("continuum_prior_weight must be non-negative")
    if config.continuum_prior_fractional_sigma <= 0:
        raise ValueError("continuum_prior_fractional_sigma must be positive")
    if config.solve_continuum_linear and config.loss != "linear":
        raise ValueError("solve_continuum_linear currently requires loss='linear'")
    if config.max_nfev is not None and config.max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if config.basis_workers < 0:
        raise ValueError("basis_workers must be non-negative")
    _validate_optimizer_tolerances(config)
    _validate_high_resolution_config(config)
    if fit_masks is None:
        fit_masks = [None] * len(spectra)
    if len(fit_masks) != len(spectra):
        raise ValueError("fit_masks must have the same length as spectra")
    if continuum_priors is None:
        continuum_priors = [None] * len(spectra)
    if len(continuum_priors) != len(spectra):
        raise ValueError("continuum_priors must have the same length as spectra")

    if global_wavelength_order is not None:
        if global_wavelength_bounds is None:
            wavelength_arrays = [
                spectrum.to_unit("micron").wavelength
                for spectrum in spectra
            ]
            global_wavelength_bounds = (
                float(min(np.nanmin(values) for values in wavelength_arrays)),
                float(max(np.nanmax(values) for values in wavelength_arrays)),
            )
        else:
            global_wavelength_bounds = tuple(
                float(value) for value in global_wavelength_bounds
            )
            if (
                len(global_wavelength_bounds) != 2
                or not np.all(np.isfinite(global_wavelength_bounds))
                or global_wavelength_bounds[1] <= global_wavelength_bounds[0]
            ):
                raise ValueError("global_wavelength_bounds must be finite and increasing")

    jobs = list(zip(spectra, fit_masks, continuum_priors, strict=True))
    worker_count = min(4, len(jobs)) if config.basis_workers == 0 else min(config.basis_workers, len(jobs))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            prepared = list(
                executor.map(
                    lambda job: _prepare_multi_fit_segment(
                        job[0],
                        job[1],
                        job[2],
                        line_list=line_list,
                        config=config,
                        global_wavelength_order=global_wavelength_order,
                        global_wavelength_bounds=global_wavelength_bounds,
                        segment_wavelength_order=segment_wavelength_order,
                    ),
                    jobs,
                )
            )
    else:
        prepared = [
            _prepare_multi_fit_segment(
                spectrum,
                fit_mask,
                continuum_prior,
                line_list=line_list,
                config=config,
                global_wavelength_order=global_wavelength_order,
                global_wavelength_bounds=global_wavelength_bounds,
                segment_wavelength_order=segment_wavelength_order,
            )
            for spectrum, fit_mask, continuum_prior in jobs
        ]

    global_species: list[str] = []
    for segment in prepared:
        for name in segment.species_names:
            if name not in global_species:
                global_species.append(name)

    species_names = tuple(global_species)
    scale_config = _fix_zero_basis_species(
        config,
        species_names,
        tuple((segment.species_names, segment.basis_all) for segment in prepared),
    )
    fitted_species_names, fixed_species_scales, p0_scales, lower_scales, upper_scales = _species_scale_setup(
        species_names,
        scale_config,
    )
    p0_parts = [p0_scales]
    lower_parts = [lower_scales]
    upper_parts = [upper_scales]
    for segment in prepared:
        continuum0 = np.zeros(config.continuum_order + 1, dtype=float)
        continuum0[0] = np.nanmedian(segment.flux_fit)
        if not np.isfinite(continuum0[0]) or continuum0[0] == 0:
            continuum0[0] = 1.0
        if not config.solve_continuum_linear:
            p0_parts.append(continuum0)
            lower_parts.append(np.full(config.continuum_order + 1, -np.inf))
            upper_parts.append(np.full(config.continuum_order + 1, np.inf))
    if global_wavelength_order is not None:
        p0_parts.append(
            _wavelength_coefficients_initial(
                global_wavelength_order,
                config.initial_wavelength_shift,
            )
        )
        lower_parts.append(
            np.full(global_wavelength_order + 1, config.wavelength_shift_bounds[0])
        )
        upper_parts.append(
            np.full(global_wavelength_order + 1, config.wavelength_shift_bounds[1])
        )
    if segment_wavelength_order is not None:
        initial_wavelength_coefficients = _wavelength_coefficients_initial(
            segment_wavelength_order,
            config.initial_wavelength_shift,
        )
        p0_parts.append(np.tile(initial_wavelength_coefficients, len(prepared)))
        lower_parts.append(
            np.full(len(prepared) * (segment_wavelength_order + 1), config.wavelength_shift_bounds[0])
        )
        upper_parts.append(
            np.full(len(prepared) * (segment_wavelength_order + 1), config.wavelength_shift_bounds[1])
        )
    if config.fit_lsf_sigma:
        p0_parts.append(np.array([_initial_lsf_width(config.lsf_sigma_pixels, config.lsf_sigma_bounds)], dtype=float))
        lower_parts.append(np.array([config.lsf_sigma_bounds[0]], dtype=float))
        upper_parts.append(np.array([config.lsf_sigma_bounds[1]], dtype=float))
    if config.fit_lsf_box_width:
        p0_parts.append(
            np.array([_initial_lsf_width(config.lsf_box_width_pixels, config.lsf_box_width_bounds)], dtype=float)
        )
        lower_parts.append(np.array([config.lsf_box_width_bounds[0]], dtype=float))
        upper_parts.append(np.array([config.lsf_box_width_bounds[1]], dtype=float))
    if config.fit_lsf_lorentz_fwhm:
        p0_parts.append(
            np.array(
                [
                    _initial_lsf_width(
                        config.lsf_lorentz_fwhm_pixels,
                        config.lsf_lorentz_fwhm_bounds,
                    )
                ],
                dtype=float,
            )
        )
        lower_parts.append(np.array([config.lsf_lorentz_fwhm_bounds[0]], dtype=float))
        upper_parts.append(np.array([config.lsf_lorentz_fwhm_bounds[1]], dtype=float))

    p0 = np.concatenate(p0_parts)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    continuum_block = config.continuum_order + 1
    jacobian_sparsity = None
    if config.use_jacobian_sparsity:
        jacobian_sparsity = _multi_fit_jacobian_sparsity(
            prepared,
            n_parameters=p0.size,
            n_fitted_species=len(fitted_species_names),
            continuum_block=continuum_block,
            solve_continuum_linear=config.solve_continuum_linear,
            global_wavelength_order=global_wavelength_order,
            segment_wavelength_order=segment_wavelength_order,
            fit_lsf_sigma=config.fit_lsf_sigma,
            fit_lsf_box_width=config.fit_lsf_box_width,
            fit_lsf_lorentz_fwhm=config.fit_lsf_lorentz_fwhm,
            continuum_prior_weight=config.continuum_prior_weight,
        )

    def unpack(params: np.ndarray):
        cursor = len(fitted_species_names)
        continuum_coefficients = []
        if not config.solve_continuum_linear:
            for _ in prepared:
                continuum_coefficients.append(params[cursor:cursor + continuum_block])
                cursor += continuum_block
        if global_wavelength_order is not None:
            stop = cursor + global_wavelength_order + 1
            global_wavelength_coefficients = np.asarray(params[cursor:stop], dtype=float)
            cursor = stop
        else:
            global_wavelength_coefficients = None
        if segment_wavelength_order is not None:
            n_wavelength_coefficients = segment_wavelength_order + 1
            stop = cursor + len(prepared) * n_wavelength_coefficients
            segment_wavelength_coefficients = np.asarray(
                params[cursor:stop],
                dtype=float,
            ).reshape(len(prepared), n_wavelength_coefficients)
            cursor = stop
        else:
            segment_wavelength_coefficients = None
        lsf_sigma_pixels = float(params[cursor]) if config.fit_lsf_sigma else config.lsf_sigma_pixels
        cursor += int(config.fit_lsf_sigma)
        lsf_box_width_pixels = (
            float(params[cursor]) if config.fit_lsf_box_width else config.lsf_box_width_pixels
        )
        cursor += int(config.fit_lsf_box_width)
        lsf_lorentz_fwhm_pixels = (
            float(params[cursor]) if config.fit_lsf_lorentz_fwhm else config.lsf_lorentz_fwhm_pixels
        )
        species_scales = _scales_from_params(
            fitted_species_names,
            fixed_species_scales,
            params[: len(fitted_species_names)],
        )
        return (
            species_scales,
            continuum_coefficients,
            global_wavelength_coefficients,
            segment_wavelength_coefficients,
            lsf_sigma_pixels,
            lsf_box_width_pixels,
            lsf_lorentz_fwhm_pixels,
        )

    def residual(params: np.ndarray) -> np.ndarray:
        (
            species_scales,
            continuum_coefficients,
            global_wavelength_coefficients,
            segment_wavelength_coefficients,
            lsf_sigma_pixels,
            lsf_box_width_pixels,
            lsf_lorentz_fwhm_pixels,
        ) = unpack(params)
        residuals = []
        for index, segment in enumerate(prepared):
            if segment_wavelength_coefficients is not None:
                coefficients = segment_wavelength_coefficients[index]
                segment_shift_fit = _wavelength_shift_from_coefficients(
                    segment.wavelength_shift_design_fit,
                    coefficients,
                )
                segment_shift_basis = _wavelength_shift_from_coefficients(
                    segment.wavelength_shift_design_basis,
                    coefficients,
                )
            elif global_wavelength_coefficients is not None:
                segment_shift_fit = _wavelength_shift_from_coefficients(
                    segment.wavelength_shift_design_fit,
                    global_wavelength_coefficients,
                )
                segment_shift_basis = _wavelength_shift_from_coefficients(
                    segment.wavelength_shift_design_basis,
                    global_wavelength_coefficients,
                )
            else:
                segment_shift_fit = 0.0
                segment_shift_basis = 0.0
            if config.high_resolution_grid:
                transmission = _transmission_from_prepared_basis(
                    segment.spectrum.wavelength,
                    segment.species_names,
                    segment.basis_all,
                    config=config,
                    species_scales=species_scales,
                    airmass=config.airmass,
                    wavelength_shift=segment_shift_basis,
                    lsf_sigma_pixels=lsf_sigma_pixels,
                    lsf_box_width_pixels=lsf_box_width_pixels,
                    lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
                    basis_wavelength=segment.basis_wavelength,
                    model_wavelength=segment.model_wavelength,
                    highres_pixels_per_observed_pixel=segment.highres_pixels_per_observed_pixel,
                    rebin_plan=segment.rebin_plan,
                    native_to_model_plan=segment.native_to_model_plan,
                )
            else:
                transmission = _transmission_from_prepared_basis(
                    segment.spectrum.wavelength[segment.valid],
                    segment.species_names,
                    segment.basis_fit,
                    config=config,
                    species_scales=species_scales,
                    airmass=config.airmass,
                    wavelength_shift=segment_shift_fit,
                    lsf_sigma_pixels=lsf_sigma_pixels,
                    lsf_box_width_pixels=lsf_box_width_pixels,
                    lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
                    basis_wavelength=segment.spectrum.wavelength[segment.valid],
                    model_wavelength=segment.spectrum.wavelength[segment.valid],
                    highres_pixels_per_observed_pixel=1.0,
                    rebin_plan=None,
                    native_to_model_plan=None,
                )
            if config.high_resolution_grid:
                transmission = transmission[segment.valid]
            if config.solve_continuum_linear:
                continuum_coeff = _solve_linear_continuum(
                    segment.design_fit,
                    transmission,
                    segment.flux_fit,
                    segment.sigma,
                    continuum_prior=segment.continuum_prior_fit,
                    continuum_prior_weight=config.continuum_prior_weight,
                    continuum_prior_fractional_sigma=config.continuum_prior_fractional_sigma,
                )
            else:
                continuum_coeff = continuum_coefficients[index]
            continuum = segment.design_fit @ continuum_coeff
            residuals.append((segment.flux_fit - continuum * transmission) / segment.sigma)
            if config.continuum_prior_weight > 0 and segment.continuum_prior_fit is not None:
                residuals.append(
                    _continuum_prior_residual(
                        continuum,
                        segment.continuum_prior_fit,
                        weight=config.continuum_prior_weight,
                        fractional_sigma=config.continuum_prior_fractional_sigma,
                    )
                )
        return np.concatenate(residuals)

    jacobian = (
        _grouped_dense_jacobian(residual, jacobian_sparsity, lower, upper)
        if jacobian_sparsity is not None
        else "2-point"
    )

    fit = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        method="trf",
        loss=config.loss,
        f_scale=config.f_scale,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=config.max_nfev,
        jac=jacobian,
    )

    (
        species_scales,
        continuum_coefficients,
        global_wavelength_coefficients,
        segment_wavelength_coefficients,
        lsf_sigma_pixels,
        lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels,
    ) = unpack(fit.x)

    parameter_names = [f"log_scale:{name}" for name in fitted_species_names]
    common_transmission_indices = list(range(len(fitted_species_names)))
    segment_transmission_indices = [list(common_transmission_indices) for _ in prepared]
    parameter_cursor = len(fitted_species_names)
    if not config.solve_continuum_linear:
        for segment_index in range(len(prepared)):
            parameter_names.extend(
                f"segment:{segment_index}:continuum:{degree}"
                for degree in range(config.continuum_order + 1)
            )
            parameter_cursor += continuum_block
    if global_wavelength_order is not None:
        if config.fit_wavelength_shift:
            parameter_names.append("wavelength_shift_micron")
        else:
            parameter_names.extend(
                f"wavelength_coefficient:{degree}"
                for degree in range(global_wavelength_order + 1)
            )
        global_wavelength_indices = list(
            range(parameter_cursor, parameter_cursor + global_wavelength_order + 1)
        )
        for indices in segment_transmission_indices:
            indices.extend(global_wavelength_indices)
        parameter_cursor += global_wavelength_order + 1
    if segment_wavelength_order is not None:
        for segment_index in range(len(prepared)):
            local_indices = list(
                range(
                    parameter_cursor,
                    parameter_cursor + segment_wavelength_order + 1,
                )
            )
            parameter_names.extend(
                f"segment:{segment_index}:wavelength_coefficient:{degree}"
                for degree in range(segment_wavelength_order + 1)
            )
            segment_transmission_indices[segment_index].extend(local_indices)
            parameter_cursor += segment_wavelength_order + 1
    for enabled, name in (
        (config.fit_lsf_sigma, "lsf_sigma_pixels"),
        (config.fit_lsf_box_width, "lsf_box_width_pixels"),
        (config.fit_lsf_lorentz_fwhm, "lsf_lorentz_fwhm_pixels"),
    ):
        if enabled:
            parameter_names.append(name)
            for indices in segment_transmission_indices:
                indices.append(parameter_cursor)
            parameter_cursor += 1
    if parameter_cursor != fit.x.size:
        raise RuntimeError("multi-segment fit parameter bookkeeping is inconsistent")
    bound_status = _parameter_bound_status(parameter_names, fit.x, lower, upper)

    selected_keep = np.zeros(line_list.wavelength.shape, dtype=bool)
    for segment in prepared:
        fitted_wavelength = segment.spectrum.wavelength[segment.valid]
        margin = _line_selection_margin(fitted_wavelength, config)
        selected_keep |= (
            (line_list.wavelength >= np.nanmin(fitted_wavelength) - margin)
            & (line_list.wavelength <= np.nanmax(fitted_wavelength) + margin)
        )
    selected_lines = line_list.select(selected_keep)
    provenance = build_fit_provenance(
        tuple(segment.spectrum for segment in prepared),
        line_list=line_list,
        selected_line_list=selected_lines,
        config=config,
        fit_pixel_counts=tuple(int(np.count_nonzero(segment.valid)) for segment in prepared),
    )

    parameter_covariance = None
    parameter_standard_errors: dict[str, float] = {}
    species_scale_uncertainties: dict[str, float] = {}
    reduced_chi_square = np.nan
    covariance_rank = 0
    if config.estimate_uncertainties:
        parameter_covariance, reduced_chi_square, covariance_rank = _linearized_parameter_covariance(
            fit.jac,
            fit.cost,
            fit.fun.size,
            fit.x.size,
        )
        standard_errors = np.sqrt(np.maximum(0.0, np.diag(parameter_covariance)))
        parameter_standard_errors = {
            name: float(error)
            for name, error in zip(parameter_names, standard_errors, strict=True)
        }
        species_scale_uncertainties = {
            name: float(species_scales[name] * standard_errors[index])
            for index, name in enumerate(fitted_species_names)
        }

    segment_results = []
    reported_segment_shifts = []
    for index, segment in enumerate(prepared):
        if segment_wavelength_coefficients is not None:
            coefficients = segment_wavelength_coefficients[index]
            segment_shift_all = _wavelength_shift_from_coefficients(
                segment.wavelength_shift_design_all,
                coefficients,
            )
            segment_shift_basis = _wavelength_shift_from_coefficients(
                segment.wavelength_shift_design_basis,
                coefficients,
            )
        elif global_wavelength_coefficients is not None:
            coefficients = global_wavelength_coefficients
            segment_shift_all = _wavelength_shift_from_coefficients(
                segment.wavelength_shift_design_all,
                coefficients,
            )
            segment_shift_basis = _wavelength_shift_from_coefficients(
                segment.wavelength_shift_design_basis,
                coefficients,
            )
        else:
            coefficients = np.array([0.0], dtype=float)
            segment_shift_all = 0.0
            segment_shift_basis = 0.0
        reported_shift = float(np.nanmedian(segment_shift_all))
        reported_segment_shifts.append(reported_shift)
        transmission = _transmission_from_prepared_basis(
            segment.spectrum.wavelength,
            segment.species_names,
            segment.basis_all,
            config=config,
            species_scales=species_scales,
            airmass=config.airmass,
            wavelength_shift=segment_shift_basis if config.high_resolution_grid else segment_shift_all,
            lsf_sigma_pixels=lsf_sigma_pixels,
            lsf_box_width_pixels=lsf_box_width_pixels,
            lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
            basis_wavelength=segment.basis_wavelength,
            model_wavelength=segment.model_wavelength,
            highres_pixels_per_observed_pixel=segment.highres_pixels_per_observed_pixel,
            rebin_plan=segment.rebin_plan,
            native_to_model_plan=segment.native_to_model_plan,
        )
        if config.solve_continuum_linear:
            continuum_coeff = _solve_linear_continuum(
                segment.design_fit,
                transmission[segment.valid],
                segment.flux_fit,
                segment.sigma,
                continuum_prior=segment.continuum_prior_fit,
                continuum_prior_weight=config.continuum_prior_weight,
                continuum_prior_fractional_sigma=config.continuum_prior_fractional_sigma,
            )
        else:
            continuum_coeff = continuum_coefficients[index]
        continuum = segment.design_all @ continuum_coeff
        model_flux = continuum * transmission
        transmission_uncertainty = None
        if parameter_covariance is not None:

            def transmission_for_parameters(parameters: np.ndarray) -> np.ndarray:
                (
                    local_scales,
                    _,
                    local_global_coefficients,
                    local_segment_coefficients,
                    local_lsf_sigma,
                    local_lsf_box,
                    local_lsf_lorentz,
                ) = unpack(parameters)
                if local_segment_coefficients is not None:
                    local_coefficients = local_segment_coefficients[index]
                    local_shift_all = _wavelength_shift_from_coefficients(
                        segment.wavelength_shift_design_all,
                        local_coefficients,
                    )
                    local_shift_basis = _wavelength_shift_from_coefficients(
                        segment.wavelength_shift_design_basis,
                        local_coefficients,
                    )
                elif local_global_coefficients is not None:
                    local_shift_all = _wavelength_shift_from_coefficients(
                        segment.wavelength_shift_design_all,
                        local_global_coefficients,
                    )
                    local_shift_basis = _wavelength_shift_from_coefficients(
                        segment.wavelength_shift_design_basis,
                        local_global_coefficients,
                    )
                else:
                    local_shift_all = 0.0
                    local_shift_basis = 0.0
                return _transmission_from_prepared_basis(
                    segment.spectrum.wavelength,
                    segment.species_names,
                    segment.basis_all,
                    config=config,
                    species_scales=local_scales,
                    airmass=config.airmass,
                    wavelength_shift=(
                        local_shift_basis if config.high_resolution_grid else local_shift_all
                    ),
                    lsf_sigma_pixels=local_lsf_sigma,
                    lsf_box_width_pixels=local_lsf_box,
                    lsf_lorentz_fwhm_pixels=local_lsf_lorentz,
                    basis_wavelength=segment.basis_wavelength,
                    model_wavelength=segment.model_wavelength,
                    highres_pixels_per_observed_pixel=segment.highres_pixels_per_observed_pixel,
                    rebin_plan=segment.rebin_plan,
                    native_to_model_plan=segment.native_to_model_plan,
                )

            transmission_uncertainty = _finite_difference_output_uncertainty(
                transmission_for_parameters,
                fit.x,
                parameter_covariance,
                lower,
                upper,
                np.asarray(segment_transmission_indices[index], dtype=int),
            )
        corrected = correct_spectrum(
            segment.spectrum,
            transmission,
            transmission_uncertainty=transmission_uncertainty,
            min_transmission=config.min_transmission,
        )
        metrics = _fit_metrics(segment.spectrum.flux, model_flux, continuum)
        segment_results.append(
            TelluricFitResult(
                spectrum=segment.spectrum,
                corrected=corrected,
                transmission=transmission,
                continuum=continuum,
                model_flux=model_flux,
                species_scales=dict(species_scales),
                wavelength_shift=reported_shift,
                wavelength_coefficients=np.asarray(coefficients, dtype=float),
                lsf_sigma_pixels=lsf_sigma_pixels,
                lsf_box_width_pixels=lsf_box_width_pixels,
                lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
                continuum_coefficients=continuum_coeff,
                metrics=metrics,
                success=bool(fit.success),
                message=str(fit.message),
                cost=float(fit.cost),
                nfev=int(fit.nfev),
                parameter_names=tuple(parameter_names),
                parameter_covariance=parameter_covariance,
                parameter_standard_errors=dict(parameter_standard_errors),
                species_scale_uncertainties=dict(species_scale_uncertainties),
                transmission_uncertainty=transmission_uncertainty,
                reduced_chi_square=reduced_chi_square,
                covariance_rank=covariance_rank,
                fit_mask=segment.valid.copy(),
                parameter_bound_status=dict(bound_status),
                provenance={**provenance, "segment_index": index},
            )
        )

    return MultiTelluricFitResult(
        segment_results=tuple(segment_results),
        species_scales=dict(species_scales),
        wavelength_shift=float(np.nanmedian(reported_segment_shifts)),
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        success=bool(fit.success),
        message=str(fit.message),
        cost=float(fit.cost),
        nfev=int(fit.nfev),
        parameter_names=tuple(parameter_names),
        parameter_covariance=parameter_covariance,
        parameter_standard_errors=parameter_standard_errors,
        species_scale_uncertainties=species_scale_uncertainties,
        reduced_chi_square=reduced_chi_square,
        covariance_rank=covariance_rank,
        parameter_bound_status=bound_status,
        provenance=provenance,
    )


def _apply_multi_fit_to_segment(
    spectrum: Spectrum,
    *,
    line_list: LineList,
    config: FitConfig,
    fit_result: MultiTelluricFitResult,
    global_wavelength_bounds: tuple[float, float] | None = None,
) -> TelluricFitResult:
    """Apply shared fitted parameters to a segment that did not constrain the fit."""

    evaluation_config = replace(config, fit_ranges=None, exclude_ranges=None)
    global_wavelength_order = _global_wavelength_polynomial_order(config)
    prepared = _prepare_multi_fit_segment(
        spectrum,
        None,
        None,
        line_list=line_list,
        config=evaluation_config,
        global_wavelength_order=global_wavelength_order,
        global_wavelength_bounds=global_wavelength_bounds,
        segment_wavelength_order=None,
    )
    if global_wavelength_order is None:
        wavelength_coefficients = np.array([0.0], dtype=float)
        wavelength_shift_all: float | np.ndarray = 0.0
        wavelength_shift_basis: float | np.ndarray = 0.0
    else:
        wavelength_coefficients = np.asarray(
            fit_result.segment_results[0].wavelength_coefficients,
            dtype=float,
        )
        wavelength_shift_all = _wavelength_shift_from_coefficients(
            prepared.wavelength_shift_design_all,
            wavelength_coefficients,
        )
        wavelength_shift_basis = _wavelength_shift_from_coefficients(
            prepared.wavelength_shift_design_basis,
            wavelength_coefficients,
        )

    transmission = _transmission_from_prepared_basis(
        prepared.spectrum.wavelength,
        prepared.species_names,
        prepared.basis_all,
        config=evaluation_config,
        species_scales=fit_result.species_scales,
        airmass=evaluation_config.airmass,
        wavelength_shift=(
            wavelength_shift_basis
            if evaluation_config.high_resolution_grid
            else wavelength_shift_all
        ),
        lsf_sigma_pixels=fit_result.lsf_sigma_pixels,
        lsf_box_width_pixels=fit_result.lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=fit_result.lsf_lorentz_fwhm_pixels,
        basis_wavelength=prepared.basis_wavelength,
        model_wavelength=prepared.model_wavelength,
        highres_pixels_per_observed_pixel=prepared.highres_pixels_per_observed_pixel,
        rebin_plan=prepared.rebin_plan,
        native_to_model_plan=prepared.native_to_model_plan,
    )
    continuum_coefficients = _solve_linear_continuum(
        prepared.design_fit,
        transmission[prepared.valid],
        prepared.flux_fit,
        prepared.sigma,
    )
    continuum = prepared.design_all @ continuum_coefficients
    model_flux = continuum * transmission
    corrected = correct_spectrum(
        prepared.spectrum,
        transmission,
        min_transmission=config.min_transmission,
    )
    provenance = {
        **dict(fit_result.provenance),
        "application_only_segment": True,
    }
    return TelluricFitResult(
        spectrum=prepared.spectrum,
        corrected=corrected,
        transmission=transmission,
        continuum=continuum,
        model_flux=model_flux,
        species_scales=dict(fit_result.species_scales),
        wavelength_shift=float(np.nanmedian(wavelength_shift_all)),
        wavelength_coefficients=wavelength_coefficients,
        lsf_sigma_pixels=float(fit_result.lsf_sigma_pixels),
        lsf_box_width_pixels=float(fit_result.lsf_box_width_pixels),
        lsf_lorentz_fwhm_pixels=float(fit_result.lsf_lorentz_fwhm_pixels),
        continuum_coefficients=continuum_coefficients,
        metrics=_fit_metrics(prepared.spectrum.flux, model_flux, continuum),
        success=bool(fit_result.success),
        message=f"{fit_result.message} (shared model applied outside fit ranges)",
        cost=float(fit_result.cost),
        nfev=int(fit_result.nfev),
        parameter_names=tuple(fit_result.parameter_names),
        parameter_covariance=fit_result.parameter_covariance,
        parameter_standard_errors=dict(fit_result.parameter_standard_errors),
        species_scale_uncertainties=dict(fit_result.species_scale_uncertainties),
        transmission_uncertainty=None,
        reduced_chi_square=float(fit_result.reduced_chi_square),
        covariance_rank=int(fit_result.covariance_rank),
        fit_mask=np.zeros(prepared.spectrum.wavelength.size, dtype=bool),
        parameter_bound_status=dict(fit_result.parameter_bound_status),
        provenance=provenance,
    )
