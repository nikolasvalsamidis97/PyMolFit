from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping

import numpy as np
from scipy.special import wofz

from .atmosphere import AtmosphereProfile
from .linelist import LineList
from .physics import (
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_AVMASS_AMU,
    LBLRTM_DEFAULT_SAMPLE,
    lblrtm_layer_wavenumber_spacings_cm,
)

_LSF_RELATIVE_WIDTH_STEP = 0.01
_MOLECFIT_KERNEL_BINS_PER_FWHM = 200.0


@dataclass(frozen=True)
class ModelConfig:
    airmass: float = 1.0
    species_scales: Mapping[str, float] = field(default_factory=dict)
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
    chunk_size: int = 512


def _voigt_profile_matrix(
    wavelength: np.ndarray,
    centers: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    x = wavelength[None, :] - centers[:, None]
    z = (x + 1j * gamma[:, None]) / (sigma[:, None] * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma[:, None] * np.sqrt(2.0 * np.pi))


def optical_depth_basis(
    wavelength: np.ndarray,
    line_list: LineList,
    *,
    species: tuple[str, ...] | None = None,
    chunk_size: int = 512,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return one optical-depth basis vector per molecular species.

    The expensive part is vectorized over line chunks and pixels. Chunking keeps
    memory bounded for large line lists while avoiding per-pixel Python loops.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim != 1:
        raise ValueError("wavelength must be one-dimensional")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    species_names = line_list.species_names if species is None else tuple(species)
    basis = np.zeros((len(species_names), wavelength.size), dtype=float)
    species_index = {name: idx for idx, name in enumerate(species_names)}

    for start in range(0, line_list.wavelength.size, chunk_size):
        stop = min(start + chunk_size, line_list.wavelength.size)
        chunk_species = line_list.species[start:stop]
        active = np.array([name in species_index for name in chunk_species], dtype=bool)
        if not np.any(active):
            continue

        centers = line_list.wavelength[start:stop][active]
        strengths = line_list.strength[start:stop][active]
        sigma = line_list.sigma[start:stop][active]
        gamma = line_list.gamma[start:stop][active]
        profile = _voigt_profile_matrix(wavelength, centers, sigma, gamma) * strengths[:, None]

        active_species = chunk_species[active]
        for name in set(active_species.tolist()):
            line_keep = active_species == name
            basis[species_index[name]] += np.sum(profile[line_keep], axis=0)

    return species_names, basis


def transmission_from_basis(
    species_names: tuple[str, ...],
    basis: np.ndarray,
    *,
    wavelength_micron: np.ndarray | None = None,
    species_scales: Mapping[str, float] | None = None,
    airmass: float = 1.0,
    lsf_sigma_pixels: float = 0.0,
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: float = 0.0,
    lsf_variable_width: bool = False,
    lsf_reference_wavelength_micron: float | None = None,
    lsf_kernel_width_fwhm: float = 3.0,
    lsf_molecfit_voigt: bool = False,
) -> np.ndarray:
    if airmass <= 0:
        raise ValueError("airmass must be positive")

    scales = np.array(
        [1.0 if species_scales is None else species_scales.get(name, 1.0) for name in species_names],
        dtype=float,
    )
    if np.any(scales < 0):
        raise ValueError("species scales must be non-negative")

    tau = airmass * np.sum(scales[:, None] * basis, axis=0)
    transmission = np.exp(-tau)
    transmission = convolve_lsf(
        transmission,
        gaussian_sigma_pixels=lsf_sigma_pixels,
        box_width_pixels=lsf_box_width_pixels,
        lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        wavelength_micron=wavelength_micron,
        variable_width=lsf_variable_width,
        reference_wavelength_micron=lsf_reference_wavelength_micron,
        kernel_width_fwhm=lsf_kernel_width_fwhm,
        molecfit_voigt=lsf_molecfit_voigt,
    )
    return np.clip(transmission, 0.0, 1.0)


def high_resolution_wavelength_grid(
    observed_wavelength_micron: np.ndarray,
    *,
    oversampling: float = 5.0,
    margin_pixels: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Return a uniform-wavenumber grid covering observed wavelength pixels.

    LBLRTM computes transmission on an internal wavenumber grid and only later
    convolves/rebins to observed pixels. This helper creates the self-contained
    analogue used by PyMolFit. The returned wavelength grid is monotonically
    increasing, while its reciprocal wavenumber grid is uniformly spaced.

    The second return value is the median number of high-resolution samples per
    observed pixel in wavenumber space. Instrumental widths expressed in
    observed-pixel units can be multiplied by this factor before convolution on
    the high-resolution grid.
    """

    wavelength = np.asarray(observed_wavelength_micron, dtype=float)
    finite = np.sort(wavelength[np.isfinite(wavelength)])
    if finite.ndim != 1 or finite.size < 2:
        raise ValueError("at least two finite wavelength points are required")
    if oversampling <= 0:
        raise ValueError("oversampling must be positive")
    if margin_pixels < 0:
        raise ValueError("margin_pixels must be non-negative")

    observed_wavenumber = 1.0e4 / finite
    sorted_wavenumber = np.sort(observed_wavenumber)
    spacing = np.diff(sorted_wavenumber)
    spacing = spacing[spacing > 0]
    if spacing.size == 0:
        raise ValueError("observed wavelength grid must span a non-zero range")
    observed_step = float(np.nanmedian(spacing))
    highres_step = observed_step / float(oversampling)
    margin = float(margin_pixels) * observed_step
    start = float(sorted_wavenumber[0] - margin)
    stop = float(sorted_wavenumber[-1] + margin)
    n_points = max(2, int(np.ceil((stop - start) / highres_step)) + 1)
    wavenumber_ascending = start + highres_step * np.arange(n_points, dtype=float)
    # Reverse so wavelength is ascending. The wavenumber spacing is still
    # uniform in absolute value.
    wavelength_ascending = 1.0e4 / wavenumber_ascending[::-1]
    return wavelength_ascending, observed_step / highres_step


def radiative_transfer_wavelength_grid(
    model_wavelength_micron: np.ndarray,
    atmosphere: AtmosphereProfile,
    *,
    sample: float = LBLRTM_DEFAULT_SAMPLE,
    alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
    step_cm: float | None = None,
    max_points: int = 2_000_000,
) -> tuple[np.ndarray, float]:
    """Build the native uniform-wavenumber radiative-transfer grid.

    LBLRTM does not evaluate opacity directly on Molecfit's five-times
    oversampled model grid. Its ``MANE`` path chooses a layer-dependent
    wavenumber spacing from the representative Voigt half-width and ``SAMPLE``;
    the finest layer grid is retained for the combined transmission. This
    function implements that source equation and covers the complete model-bin
    extent so native samples can subsequently be averaged into model pixels.

    ``step_cm`` is an explicit reproducibility override. The automatic path is
    entirely determined by the wavelength interval and atmospheric profile.
    """

    model_wavelength = np.asarray(model_wavelength_micron, dtype=float)
    finite = np.sort(model_wavelength[np.isfinite(model_wavelength)])
    if finite.ndim != 1 or finite.size < 2:
        raise ValueError("at least two finite model wavelengths are required")
    if np.any(finite <= 0):
        raise ValueError("model wavelengths must be positive")
    if sample <= 0 or not np.isfinite(sample):
        raise ValueError("sample must be positive and finite")
    if alfal0 < 0 or not np.isfinite(alfal0):
        raise ValueError("alfal0 must be non-negative and finite")
    if avmass_amu <= 0 or not np.isfinite(avmass_amu):
        raise ValueError("avmass_amu must be positive and finite")
    if max_points < 2:
        raise ValueError("max_points must be at least two")

    wavelength_edges = _pixel_edges_from_centers(finite)
    if np.any(wavelength_edges <= 0):
        raise ValueError("model wavelength-bin edges must be positive")
    edge_wavenumber = 1.0e4 / wavelength_edges
    start = float(np.nanmin(edge_wavenumber))
    stop = float(np.nanmax(edge_wavenumber))
    model_wavenumber = np.sort(1.0e4 / finite)
    model_spacing = np.diff(model_wavenumber)
    model_spacing = model_spacing[model_spacing > 0]
    if model_spacing.size == 0:
        raise ValueError("model wavelength grid must span a non-zero range")
    model_step = float(np.nanmin(model_spacing))

    if step_cm is None:
        pressure_ratio = np.asarray(
            [layer.pressure_atm for layer in atmosphere.layers], dtype=float
        )
        temperature = np.asarray(
            [layer.temperature_k for layer in atmosphere.layers], dtype=float
        )
        h2o_fraction = np.asarray(
            [layer.mixing_ratios.get("H2O", 0.0) for layer in atmosphere.layers],
            dtype=float,
        )
        if not (
            np.all(np.isfinite(pressure_ratio))
            and np.all(np.isfinite(temperature))
            and np.all(temperature > 0)
        ):
            raise ValueError("atmospheric pressure and temperature must be finite and positive")

        representative_wavenumber = 0.5 * (start + stop)
        layer_steps = lblrtm_layer_wavenumber_spacings_cm(
            representative_wavenumber,
            pressure_ratio,
            temperature,
            h2o_fraction=np.clip(h2o_fraction, 0.0, 1.0),
            sample=sample,
            alfal0=alfal0,
            avmass_amu=avmass_amu,
        )
        desired_step = float(np.nanmin(layer_steps))
    else:
        desired_step = float(step_cm)
        if desired_step <= 0 or not np.isfinite(desired_step):
            raise ValueError("step_cm must be positive and finite")

    desired_step = min(desired_step, model_step)
    if desired_step <= 0 or not np.isfinite(desired_step):
        raise ValueError("the radiative-transfer grid spacing is not positive and finite")
    n_points = max(2, int(np.ceil((stop - start) / desired_step)) + 1)
    if n_points > max_points:
        raise ValueError(
            "automatic radiative-transfer grid requires "
            f"{n_points:,} points, exceeding max_points={max_points:,}; "
            "split the spectrum into narrower segments or raise the explicit limit"
        )
    actual_step = (stop - start) / float(n_points - 1)
    wavenumber = start + actual_step * np.arange(n_points, dtype=float)
    return 1.0e4 / wavenumber[::-1], actual_step


def transmission_from_high_resolution_basis(
    observed_wavelength_micron: np.ndarray,
    highres_wavelength_micron: np.ndarray,
    species_names: tuple[str, ...],
    highres_basis: np.ndarray,
    *,
    species_scales: Mapping[str, float] | None = None,
    airmass: float = 1.0,
    lsf_sigma_pixels: float = 0.0,
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: float = 0.0,
    highres_pixels_per_observed_pixel: float,
    lsf_variable_width: bool = False,
    lsf_reference_wavelength_micron: float | None = None,
    lsf_kernel_width_fwhm: float = 3.0,
    lsf_molecfit_voigt: bool = False,
    rebin_mode: str = "integrate",
    rebin_plan: PiecewiseConstantRebinPlan | None = None,
    model_wavelength_micron: np.ndarray | None = None,
    native_to_model_plan: SampleAverageRebinPlan | None = None,
) -> np.ndarray:
    """Evaluate, convolve, and rebin a high-resolution transmission model.

    When ``model_wavelength_micron`` is supplied, ``highres_basis`` lives on a
    finer native radiative-transfer grid. Transmission is exponentiated there
    before discrete native samples are averaged into model pixels, matching the
    ordering in Molecfit's LBLRTM reader. The model is then rebinned/convolved
    with the selected detector-grid convention.
    """

    if highres_pixels_per_observed_pixel <= 0:
        raise ValueError("highres_pixels_per_observed_pixel must be positive")
    observed = np.asarray(observed_wavelength_micron, dtype=float)
    highres = np.asarray(highres_wavelength_micron, dtype=float)
    mode = _normalized_high_resolution_rebin_mode(rebin_mode)
    raw_highres_transmission = transmission_from_basis(
        species_names,
        highres_basis,
        wavelength_micron=highres,
        species_scales=species_scales,
        airmass=airmass,
    )
    if mode == "molecfit_overlap":
        # mf_convolution first overlap-rebins the LBLRTM model directly onto
        # detector wavelengths and only then applies the synthetic kernel.
        plan_matches_native_grid = (
            rebin_plan is not None
            and rebin_plan.input_indices.size == highres.size
        )
        rebinned = (
            rebin_plan.apply(raw_highres_transmission)
            if plan_matches_native_grid
            else rebin_piecewise_constant_values(observed, highres, raw_highres_transmission)
        )
        return np.clip(
            convolve_lsf(
                rebinned,
                gaussian_sigma_pixels=lsf_sigma_pixels,
                box_width_pixels=lsf_box_width_pixels,
                lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
                wavelength_micron=observed,
                variable_width=lsf_variable_width,
                reference_wavelength_micron=lsf_reference_wavelength_micron,
                kernel_width_fwhm=lsf_kernel_width_fwhm,
                molecfit_voigt=lsf_molecfit_voigt,
            ),
            0.0,
            1.0,
        )

    if model_wavelength_micron is None:
        model_wavelength = highres
        model_transmission = raw_highres_transmission
    else:
        model_wavelength = np.asarray(model_wavelength_micron, dtype=float)
        if native_to_model_plan is None:
            model_transmission = average_high_resolution_values(
                model_wavelength,
                highres,
                raw_highres_transmission,
            )
        else:
            model_transmission = native_to_model_plan.apply(raw_highres_transmission)

    model_transmission = convolve_lsf(
        model_transmission,
        gaussian_sigma_pixels=lsf_sigma_pixels * highres_pixels_per_observed_pixel,
        box_width_pixels=lsf_box_width_pixels * highres_pixels_per_observed_pixel,
        lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels * highres_pixels_per_observed_pixel,
        wavelength_micron=model_wavelength,
        variable_width=lsf_variable_width,
        reference_wavelength_micron=lsf_reference_wavelength_micron,
        kernel_width_fwhm=lsf_kernel_width_fwhm,
        molecfit_voigt=lsf_molecfit_voigt,
    )
    if mode == "integrate":
        return rebin_high_resolution_values(observed, model_wavelength, model_transmission)
    if mode == "center":
        return sample_high_resolution_values(observed, model_wavelength, model_transmission)
    if mode == "sample_average":
        return average_high_resolution_values(observed, model_wavelength, model_transmission)
    raise AssertionError(f"unhandled high-resolution rebin mode {mode!r}")


def _normalized_high_resolution_rebin_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "integral": "integrate",
        "average": "integrate",
        "bin_average": "integrate",
        "pixel_average": "integrate",
        "sample": "center",
        "centre": "center",
        "pixel_center": "center",
        "centre_sample": "center",
        "center_sample": "center",
        "mean": "sample_average",
        "average_samples": "sample_average",
        "sample_mean": "sample_average",
        "sample_average": "sample_average",
        "molecfit": "molecfit_overlap",
        "molecfit_overlap": "molecfit_overlap",
        "molecfit_rebin": "molecfit_overlap",
        "molecfit_average": "sample_average",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"integrate", "center", "sample_average", "molecfit_overlap"}:
        raise ValueError(
            "high_resolution_rebin_mode must be 'integrate', 'center', "
            "'sample_average', or 'molecfit_overlap'"
        )
    return normalized


def sample_high_resolution_values(
    observed_wavelength_micron: np.ndarray,
    highres_wavelength_micron: np.ndarray,
    highres_values: np.ndarray,
) -> np.ndarray:
    """Sample high-resolution values at observed pixel centers."""

    observed = np.asarray(observed_wavelength_micron, dtype=float)
    highres_wavelength = np.asarray(highres_wavelength_micron, dtype=float)
    values = np.asarray(highres_values, dtype=float)
    if observed.ndim != 1 or highres_wavelength.ndim != 1 or values.ndim != 1:
        raise ValueError("observed wavelengths, high-resolution wavelengths, and values must be one-dimensional")
    if highres_wavelength.shape != values.shape:
        raise ValueError("high-resolution wavelength and value arrays must have the same shape")
    if observed.size == 0:
        return np.array([], dtype=float)
    if highres_wavelength.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)

    order = np.argsort(highres_wavelength)
    x = highres_wavelength[order]
    y = values[order]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)
    unique = np.r_[True, np.diff(x) > 0]
    x = x[unique]
    y = y[unique]
    if x.size == 1:
        return np.full(observed.shape, y[0], dtype=float)
    return np.interp(observed, x, y, left=y[0], right=y[-1])


def rebin_high_resolution_values(
    observed_wavelength_micron: np.ndarray,
    highres_wavelength_micron: np.ndarray,
    highres_values: np.ndarray,
) -> np.ndarray:
    """Integrate high-resolution values over observed pixel bins.

    This is more faithful than sampling the high-resolution transmission at the
    pixel centers, especially for narrow saturated telluric lines. The
    implementation uses a cumulative trapezoid integral and is vectorized over
    all output pixels.
    """

    observed = np.asarray(observed_wavelength_micron, dtype=float)
    highres_wavelength = np.asarray(highres_wavelength_micron, dtype=float)
    values = np.asarray(highres_values, dtype=float)
    if observed.ndim != 1 or highres_wavelength.ndim != 1 or values.ndim != 1:
        raise ValueError("observed wavelengths, high-resolution wavelengths, and values must be one-dimensional")
    if highres_wavelength.shape != values.shape:
        raise ValueError("high-resolution wavelength and value arrays must have the same shape")
    if observed.size == 0:
        return np.array([], dtype=float)
    if highres_wavelength.size < 2:
        return np.full(observed.shape, values[0] if values.size else np.nan, dtype=float)

    order = np.argsort(highres_wavelength)
    x = highres_wavelength[order]
    y = values[order]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return np.full(observed.shape, np.nan, dtype=float)
    unique = np.r_[True, np.diff(x) > 0]
    x = x[unique]
    y = y[unique]
    if x.size < 2:
        return np.full(observed.shape, np.nan, dtype=float)

    edges = _pixel_edges_from_centers(observed)
    cumulative = np.empty(x.size, dtype=float)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    integral_edges = np.interp(edges, x, cumulative, left=cumulative[0], right=cumulative[-1])
    widths = np.diff(edges)
    averaged = np.empty(observed.shape, dtype=float)
    valid_width = np.isfinite(widths) & (widths != 0)
    averaged[valid_width] = np.diff(integral_edges)[valid_width] / widths[valid_width]
    if not np.all(valid_width):
        averaged[~valid_width] = np.interp(observed[~valid_width], x, y)
    return np.clip(averaged, np.nanmin(y), np.nanmax(y))


@dataclass(frozen=True)
class SampleAverageRebinPlan:
    """Precomputed geometry for Molecfit's native-sample averaging step."""

    input_indices: np.ndarray
    start_indices: np.ndarray
    stop_indices: np.ndarray
    counts: np.ndarray
    fallback_left_indices: np.ndarray
    fallback_right_indices: np.ndarray
    fallback_right_weights: np.ndarray
    output_order: np.ndarray

    def apply(self, input_values: np.ndarray) -> np.ndarray:
        values = np.asarray(input_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("input_values must be one-dimensional")
        if values.size <= int(np.max(self.input_indices, initial=-1)):
            raise ValueError("input_values do not match the sample-average plan")

        y = values[self.input_indices]
        cumulative = np.r_[0.0, np.cumsum(y)]
        averaged_sorted = np.empty(self.counts.shape, dtype=float)
        has_samples = self.counts > 0
        averaged_sorted[has_samples] = (
            cumulative[self.stop_indices[has_samples]]
            - cumulative[self.start_indices[has_samples]]
        ) / self.counts[has_samples]
        if not np.all(has_samples):
            left = y[self.fallback_left_indices[~has_samples]]
            right = y[self.fallback_right_indices[~has_samples]]
            weight = self.fallback_right_weights[~has_samples]
            averaged_sorted[~has_samples] = left + weight * (right - left)

        averaged = np.empty_like(averaged_sorted)
        averaged[self.output_order] = averaged_sorted
        return averaged


def prepare_sample_average_rebin(
    model_wavelength_micron: np.ndarray,
    native_wavelength_micron: np.ndarray,
) -> SampleAverageRebinPlan:
    """Precompute native-sample membership for every internal model bin."""

    model = np.asarray(model_wavelength_micron, dtype=float)
    native = np.asarray(native_wavelength_micron, dtype=float)
    if model.ndim != 1 or native.ndim != 1:
        raise ValueError("wavelength arrays must be one-dimensional")
    if model.size == 0 or native.size < 2:
        raise ValueError("a sample-average plan requires model pixels and two native samples")
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(native)):
        raise ValueError("sample-average wavelengths must be finite")

    input_order = np.argsort(native)
    native_sorted = native[input_order]
    unique = np.r_[True, np.diff(native_sorted) > 0]
    input_indices = input_order[unique]
    x = native_sorted[unique]
    if x.size < 2:
        raise ValueError("native wavelength grid must contain at least two unique samples")

    output_order = np.argsort(model)
    model_sorted = model[output_order]
    edges = _pixel_edges_from_centers(model_sorted)
    left_edges = np.minimum(edges[:-1], edges[1:])
    right_edges = np.maximum(edges[:-1], edges[1:])
    # This reproduces the source inequalities lmin < lambda <= lmax.
    start_indices = np.searchsorted(x, left_edges, side="right")
    stop_indices = np.searchsorted(x, right_edges, side="right")
    counts = stop_indices - start_indices

    insertion = np.searchsorted(x, model_sorted, side="left")
    fallback_right = np.clip(insertion, 1, x.size - 1)
    fallback_left = fallback_right - 1
    denominator = x[fallback_right] - x[fallback_left]
    fallback_weight = np.divide(
        model_sorted - x[fallback_left],
        denominator,
        out=np.zeros(model_sorted.shape, dtype=float),
        where=denominator > 0,
    )
    fallback_weight = np.clip(fallback_weight, 0.0, 1.0)

    return SampleAverageRebinPlan(
        input_indices=input_indices,
        start_indices=start_indices,
        stop_indices=stop_indices,
        counts=counts,
        fallback_left_indices=fallback_left,
        fallback_right_indices=fallback_right,
        fallback_right_weights=fallback_weight,
        output_order=output_order,
    )


@dataclass(frozen=True)
class PiecewiseConstantRebinPlan:
    """Precomputed geometry for repeated Molecfit-style overlap rebinning."""

    input_indices: np.ndarray
    input_widths: np.ndarray
    cumulative_indices: np.ndarray
    data_indices: np.ndarray
    edge_offsets: np.ndarray
    output_widths: np.ndarray
    output_order: np.ndarray

    def apply(self, input_values: np.ndarray) -> np.ndarray:
        values = np.asarray(input_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("input_values must be one-dimensional")
        if values.size <= int(np.max(self.input_indices, initial=-1)):
            raise ValueError("input_values do not match the rebin plan")
        y = values[self.input_indices]
        cumulative = np.empty(y.size + 1, dtype=float)
        cumulative[0] = 0.0
        cumulative[1:] = np.cumsum(y * self.input_widths)
        primitive = cumulative[self.cumulative_indices] + y[self.data_indices] * self.edge_offsets
        rebinned_sorted = np.divide(
            np.diff(primitive),
            self.output_widths,
            out=np.zeros(self.output_widths.shape, dtype=float),
            where=self.output_widths > 0,
        )
        rebinned = np.empty_like(rebinned_sorted)
        rebinned[self.output_order] = rebinned_sorted
        return rebinned


def prepare_piecewise_constant_rebin(
    observed_wavelength_micron: np.ndarray,
    input_wavelength_micron: np.ndarray,
) -> PiecewiseConstantRebinPlan:
    """Precompute the wavelength-only part of overlap-weighted rebinning."""

    observed = np.asarray(observed_wavelength_micron, dtype=float)
    input_wavelength = np.asarray(input_wavelength_micron, dtype=float)
    if observed.ndim != 1 or input_wavelength.ndim != 1:
        raise ValueError("wavelength arrays must be one-dimensional")
    if observed.size == 0 or input_wavelength.size < 2:
        raise ValueError("a rebin plan requires output pixels and at least two input pixels")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(input_wavelength)):
        raise ValueError("rebin-plan wavelengths must be finite")

    input_order = np.argsort(input_wavelength)
    x_sorted = input_wavelength[input_order]
    unique = np.r_[True, np.diff(x_sorted) > 0]
    input_indices = input_order[unique]
    x = x_sorted[unique]
    if x.size < 2:
        raise ValueError("input wavelength grid must contain at least two unique pixels")

    output_order = np.argsort(observed)
    output_centers = observed[output_order]
    input_edges = _pixel_edges_from_centers(x)
    output_edges = _pixel_edges_from_centers(output_centers)
    output_widths = np.diff(output_edges)

    below = output_edges <= input_edges[0]
    above = output_edges >= input_edges[-1]
    middle = ~(below | above)
    cumulative_indices = np.zeros(output_edges.shape, dtype=int)
    data_indices = np.zeros(output_edges.shape, dtype=int)
    edge_offsets = np.zeros(output_edges.shape, dtype=float)
    cumulative_indices[above] = x.size
    data_indices[above] = x.size - 1
    if np.any(middle):
        indices = np.searchsorted(input_edges, output_edges[middle], side="right") - 1
        indices = np.clip(indices, 0, x.size - 1)
        cumulative_indices[middle] = indices
        data_indices[middle] = indices
        edge_offsets[middle] = output_edges[middle] - input_edges[indices]

    return PiecewiseConstantRebinPlan(
        input_indices=input_indices,
        input_widths=np.diff(input_edges),
        cumulative_indices=cumulative_indices,
        data_indices=data_indices,
        edge_offsets=edge_offsets,
        output_widths=output_widths,
        output_order=output_order,
    )


def rebin_piecewise_constant_values(
    observed_wavelength_micron: np.ndarray,
    input_wavelength_micron: np.ndarray,
    input_values: np.ndarray,
) -> np.ndarray:
    """Port Molecfit's overlap-weighted ``mf_convolution_rebin`` routine.

    Every input value represents a piecewise-constant wavelength bin bounded
    by neighboring pixel midpoints. Output values are overlap-weighted bin
    averages. This differs from :func:`rebin_high_resolution_values`, which
    integrates the linearly interpolated samples with a trapezoid rule.
    """

    observed = np.asarray(observed_wavelength_micron, dtype=float)
    input_wavelength = np.asarray(input_wavelength_micron, dtype=float)
    values = np.asarray(input_values, dtype=float)
    if observed.ndim != 1 or input_wavelength.ndim != 1 or values.ndim != 1:
        raise ValueError("wavelength and value arrays must be one-dimensional")
    if input_wavelength.shape != values.shape:
        raise ValueError("input wavelength and value arrays must have the same shape")
    if observed.size == 0:
        return np.array([], dtype=float)
    if input_wavelength.size == 0:
        return np.zeros(observed.shape, dtype=float)
    if input_wavelength.size == 1:
        return np.full(observed.shape, values[0], dtype=float)

    input_order = np.argsort(input_wavelength)
    x = input_wavelength[input_order]
    y = values[input_order]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    unique = np.r_[True, np.diff(x) > 0]
    x = x[unique]
    y = y[unique]
    if x.size < 2:
        return np.full(observed.shape, y[0] if y.size else 0.0, dtype=float)

    output_order = np.argsort(observed)
    output_centers = observed[output_order]
    input_edges = _pixel_edges_from_centers(x)
    output_edges = _pixel_edges_from_centers(output_centers)
    input_widths = np.diff(input_edges)
    cumulative = np.r_[0.0, np.cumsum(y * input_widths)]

    def primitive(points: np.ndarray) -> np.ndarray:
        result = np.empty(points.shape, dtype=float)
        below = points <= input_edges[0]
        above = points >= input_edges[-1]
        middle = ~(below | above)
        result[below] = 0.0
        result[above] = cumulative[-1]
        if np.any(middle):
            indices = np.searchsorted(input_edges, points[middle], side="right") - 1
            indices = np.clip(indices, 0, y.size - 1)
            result[middle] = cumulative[indices] + y[indices] * (
                points[middle] - input_edges[indices]
            )
        return result

    widths = np.diff(output_edges)
    rebinned_sorted = np.divide(
        np.diff(primitive(output_edges)),
        widths,
        out=np.zeros(output_centers.shape, dtype=float),
        where=widths > 0,
    )
    rebinned = np.empty_like(rebinned_sorted)
    rebinned[output_order] = rebinned_sorted
    return rebinned


def average_high_resolution_values(
    observed_wavelength_micron: np.ndarray,
    highres_wavelength_micron: np.ndarray,
    highres_values: np.ndarray,
) -> np.ndarray:
    """Average high-resolution sample values inside observed pixel bins.

    Molecfit's LBLRTM-reader path averages the discrete LBLRTM samples falling
    in each observed wavelength bin. Empty bins are sampled at the pixel center
    here so the vectorized model remains finite for sparse internal grids.
    """

    observed = np.asarray(observed_wavelength_micron, dtype=float)
    highres_wavelength = np.asarray(highres_wavelength_micron, dtype=float)
    values = np.asarray(highres_values, dtype=float)
    if observed.ndim != 1 or highres_wavelength.ndim != 1 or values.ndim != 1:
        raise ValueError("observed wavelengths, high-resolution wavelengths, and values must be one-dimensional")
    if highres_wavelength.shape != values.shape:
        raise ValueError("high-resolution wavelength and value arrays must have the same shape")
    if observed.size == 0:
        return np.array([], dtype=float)
    if highres_wavelength.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)

    order = np.argsort(highres_wavelength)
    x = highres_wavelength[order]
    y = values[order]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)
    unique = np.r_[True, np.diff(x) > 0]
    x = x[unique]
    y = y[unique]
    if x.size == 1:
        return np.full(observed.shape, y[0], dtype=float)

    edges = _pixel_edges_from_centers(observed)
    left = np.minimum(edges[:-1], edges[1:])
    right = np.maximum(edges[:-1], edges[1:])
    start = np.searchsorted(x, left, side="right")
    stop = np.searchsorted(x, right, side="right")
    cumsum = np.r_[0.0, np.cumsum(y)]
    count = stop - start
    averaged = np.empty(observed.shape, dtype=float)
    has_samples = count > 0
    averaged[has_samples] = (cumsum[stop[has_samples]] - cumsum[start[has_samples]]) / count[has_samples]
    if not np.all(has_samples):
        averaged[~has_samples] = np.interp(observed[~has_samples], x, y, left=y[0], right=y[-1])
    return averaged


def _pixel_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1:
        raise ValueError("pixel centers must be one-dimensional")
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def convolve_lsf(
    values: np.ndarray,
    *,
    gaussian_sigma_pixels: float = 0.0,
    box_width_pixels: float = 0.0,
    lorentz_fwhm_pixels: float = 0.0,
    wavelength_micron: np.ndarray | None = None,
    variable_width: bool = False,
    reference_wavelength_micron: float | None = None,
    kernel_width_fwhm: float = 3.0,
    molecfit_voigt: bool = False,
) -> np.ndarray:
    """Convolve a 1D model with simple instrumental-profile components."""

    result = np.asarray(values, dtype=float)
    if gaussian_sigma_pixels < 0 or box_width_pixels < 0 or lorentz_fwhm_pixels < 0:
        raise ValueError("LSF widths must be non-negative")
    if kernel_width_fwhm < 0:
        raise ValueError("kernel_width_fwhm must be non-negative")
    if not (gaussian_sigma_pixels > 0 or box_width_pixels > 0 or lorentz_fwhm_pixels > 0):
        return result.copy()
    if variable_width:
        return _convolve_variable_lsf(
            result,
            gaussian_sigma_pixels=gaussian_sigma_pixels,
            box_width_pixels=box_width_pixels,
            lorentz_fwhm_pixels=lorentz_fwhm_pixels,
            wavelength_micron=wavelength_micron,
            reference_wavelength_micron=reference_wavelength_micron,
            kernel_width_fwhm=kernel_width_fwhm,
            molecfit_voigt=molecfit_voigt,
        )

    kernel = _composite_lsf_kernel(
        gaussian_sigma_pixels=gaussian_sigma_pixels,
        box_width_pixels=box_width_pixels,
        lorentz_fwhm_pixels=lorentz_fwhm_pixels,
        kernel_width_fwhm=kernel_width_fwhm,
        molecfit_voigt=molecfit_voigt,
    )
    return _convolve_with_kernel(result, kernel)


def lsf_kernel_half_width_pixels(
    *,
    gaussian_sigma_pixels: float = 0.0,
    box_width_pixels: float = 0.0,
    lorentz_fwhm_pixels: float = 0.0,
    kernel_width_fwhm: float = 3.0,
    molecfit_voigt: bool = False,
) -> float:
    """Return the discrete half-support of the composite LSF kernel."""

    if gaussian_sigma_pixels < 0 or box_width_pixels < 0 or lorentz_fwhm_pixels < 0:
        raise ValueError("LSF widths must be non-negative")
    if kernel_width_fwhm < 0:
        raise ValueError("kernel_width_fwhm must be non-negative")
    kernel = _composite_lsf_kernel(
        gaussian_sigma_pixels=float(gaussian_sigma_pixels),
        box_width_pixels=float(box_width_pixels),
        lorentz_fwhm_pixels=float(lorentz_fwhm_pixels),
        kernel_width_fwhm=float(kernel_width_fwhm),
        molecfit_voigt=bool(molecfit_voigt),
    )
    return 0.5 * float(kernel.size - 1)


def _convolve_with_kernel(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.size // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _convolve_variable_lsf(
    values: np.ndarray,
    *,
    gaussian_sigma_pixels: float,
    box_width_pixels: float,
    lorentz_fwhm_pixels: float,
    wavelength_micron: np.ndarray | None,
    reference_wavelength_micron: float | None,
    kernel_width_fwhm: float,
    molecfit_voigt: bool,
) -> np.ndarray:
    if wavelength_micron is None:
        raise ValueError("wavelength_micron is required when variable_width=True")
    wavelength = np.asarray(wavelength_micron, dtype=float)
    if wavelength.shape != values.shape:
        raise ValueError("wavelength_micron must have the same shape as values")
    if reference_wavelength_micron is None:
        reference_wavelength_micron = float(np.nanmedian(wavelength))
    if not np.isfinite(reference_wavelength_micron) or reference_wavelength_micron <= 0:
        raise ValueError("reference_wavelength_micron must be positive")

    output = values.copy()
    finite = np.isfinite(wavelength)
    if not np.any(finite):
        return output
    scale_values = wavelength[finite] / reference_wavelength_micron
    log_step = np.log1p(_LSF_RELATIVE_WIDTH_STEP)
    scale_keys = np.rint(np.log(scale_values) / log_step).astype(int)
    finite_pixels = np.nonzero(finite)[0]
    for key in np.unique(scale_keys):
        group = finite_pixels[scale_keys == key]
        scale = float(np.nanmedian(wavelength[group]) / reference_wavelength_micron)
        kernel = _composite_lsf_kernel(
            gaussian_sigma_pixels=gaussian_sigma_pixels * scale,
            box_width_pixels=box_width_pixels * scale,
            lorentz_fwhm_pixels=lorentz_fwhm_pixels * scale,
            kernel_width_fwhm=kernel_width_fwhm,
            molecfit_voigt=molecfit_voigt,
        )
        convolved = _convolve_with_kernel(values, kernel)
        output[group] = convolved[group]
    return output


@lru_cache(maxsize=256)
def _composite_lsf_kernel(
    *,
    gaussian_sigma_pixels: float,
    box_width_pixels: float,
    lorentz_fwhm_pixels: float,
    kernel_width_fwhm: float,
    molecfit_voigt: bool = False,
) -> np.ndarray:
    kernel = np.array([1.0], dtype=float)
    if box_width_pixels > 0:
        kernel = np.convolve(kernel, _fractional_box_kernel(box_width_pixels))
    if molecfit_voigt and (gaussian_sigma_pixels > 0 or lorentz_fwhm_pixels > 0):
        gaussian_fwhm_pixels = gaussian_sigma_pixels * (2.0 * np.sqrt(2.0 * np.log(2.0)))
        kernel = np.convolve(
            kernel,
            _molecfit_voigt_approx_kernel(
                gaussian_fwhm_pixels,
                lorentz_fwhm_pixels,
                kernel_width_fwhm=kernel_width_fwhm,
            ),
        )
    else:
        if gaussian_sigma_pixels > 0:
            kernel = np.convolve(
                kernel,
                _integrated_gaussian_kernel(
                    gaussian_sigma_pixels,
                    kernel_width_fwhm=kernel_width_fwhm,
                ),
            )
        if lorentz_fwhm_pixels > 0:
            kernel = np.convolve(
                kernel,
                _integrated_lorentz_kernel(
                    lorentz_fwhm_pixels,
                    kernel_width_fwhm=kernel_width_fwhm,
                ),
            )
    return kernel / np.sum(kernel)


def _fractional_box_kernel(width_pixels: float) -> np.ndarray:
    term = 0.5 * (width_pixels - 1.0)
    n_pixels = int(2 * np.ceil(term) + 1)
    if n_pixels <= 1:
        return np.array([1.0])
    edge_value = np.mod(term, 1.0) / width_pixels
    center_value = 1.0 / width_pixels
    weights = np.full(n_pixels, center_value, dtype=float)
    if edge_value > 0:
        weights[0] = edge_value
        weights[-1] = edge_value
    return weights / np.sum(weights)


def _integrated_gaussian_kernel(sigma_pixels: float, *, kernel_width_fwhm: float) -> np.ndarray:
    fwhm_pixels = sigma_pixels * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    n_pixels = int(2 * np.ceil(fwhm_pixels * kernel_width_fwhm / 2.0 - 0.5) + 1)
    if n_pixels <= 1:
        return np.array([1.0])
    offsets = np.arange(n_pixels, dtype=float) - (n_pixels // 2)
    n_sub = max(1, int(np.ceil(_MOLECFIT_KERNEL_BINS_PER_FWHM / fwhm_pixels)))
    subpixel = (np.arange(n_sub, dtype=float) + 0.5) / n_sub - 0.5
    x = offsets[:, None] + subpixel[None, :]
    weights = np.mean(np.exp(-0.5 * (x / sigma_pixels) ** 2), axis=1)
    weights = np.maximum(weights, 0.0)
    return weights / np.sum(weights)


def _integrated_lorentz_kernel(fwhm_pixels: float, *, kernel_width_fwhm: float) -> np.ndarray:
    gamma = 0.5 * fwhm_pixels
    n_pixels = int(2 * np.ceil(fwhm_pixels * kernel_width_fwhm / 2.0 - 0.5) + 1)
    if n_pixels <= 1:
        return np.array([1.0])
    offsets = np.arange(n_pixels, dtype=float) - (n_pixels // 2)
    n_sub = max(1, int(np.ceil(_MOLECFIT_KERNEL_BINS_PER_FWHM / fwhm_pixels)))
    subpixel = (np.arange(n_sub, dtype=float) + 0.5) / n_sub - 0.5
    x = offsets[:, None] + subpixel[None, :]
    weights = np.mean(gamma / (x * x + gamma * gamma), axis=1)
    weights = np.maximum(weights, 0.0)
    return weights / np.sum(weights)


def _molecfit_voigt_approx_kernel(
    gaussian_fwhm_pixels: float,
    lorentz_fwhm_pixels: float,
    *,
    kernel_width_fwhm: float,
) -> np.ndarray:
    """Molecfit synthetic-kernel Voigt approximation.

    This ports the `mf_kernel_synthetic_voigt` formula used when Molecfit's
    `kern_mode` option is enabled. It is not the HITRAN/LBLRTM absorption
    Voigt profile; it is an instrumental kernel approximation in pixel units.
    """

    if gaussian_fwhm_pixels < 0 or lorentz_fwhm_pixels < 0:
        raise ValueError("LSF FWHM values must be non-negative")
    if kernel_width_fwhm < 0:
        raise ValueError("kernel_width_fwhm must be non-negative")

    gamma = 0.5 * lorentz_fwhm_pixels
    voigt_fwhm = gamma + np.sqrt(gamma * gamma + gaussian_fwhm_pixels * gaussian_fwhm_pixels)
    n_pixels = int(2 * np.ceil(voigt_fwhm * kernel_width_fwhm / 2.0 - 0.5) + 1)
    if n_pixels <= 1 or voigt_fwhm <= 0:
        return np.array([1.0])

    lorentz_ratio = lorentz_fwhm_pixels / voigt_fwhm
    n_sub = max(1, int(np.ceil(_MOLECFIT_KERNEL_BINS_PER_FWHM / voigt_fwhm)))
    weights = np.zeros(n_pixels, dtype=float)
    refpix = int(np.floor(n_pixels / 2.0))
    right_indices = np.arange(n_pixels - 1, refpix - 1, -1)
    right_offsets = np.arange(right_indices.size, dtype=float)
    subpixel_offsets = (np.arange(n_sub, dtype=float) + 0.5) / n_sub
    x = (0.5 * n_pixels - right_offsets[:, None] - 1.0) + subpixel_offsets[None, :]
    xv = x / voigt_fwhm
    xv2 = xv * xv
    xv225 = np.abs(xv) ** 2.25
    values = (
        (1.0 - lorentz_ratio) * np.exp(-2.772 * xv2)
        + lorentz_ratio / (1.0 + 4.0 * xv2)
        + 0.016
        * (1.0 - lorentz_ratio)
        * lorentz_ratio
        * (np.exp(-0.4 * xv225) - 10.0 / (10.0 + xv225))
    )
    weights[right_indices] = np.mean(values, axis=1)

    for k in range(refpix - 1, -1, -1):
        weights[k] = weights[n_pixels - k - 1]
    weights = np.maximum(weights, 0.0)
    total = np.sum(weights)
    if total <= 0:
        return np.array([1.0])
    return weights / total


def transmission_model(
    wavelength: np.ndarray,
    line_list: LineList,
    config: ModelConfig | None = None,
) -> np.ndarray:
    config = ModelConfig() if config is None else config
    if config.high_resolution_oversampling <= 0:
        raise ValueError("high_resolution_oversampling must be positive")
    if config.high_resolution_margin_pixels < 0:
        raise ValueError("high_resolution_margin_pixels must be non-negative")
    if config.high_resolution_grid:
        highres_wavelength, pixels_per_observed = high_resolution_wavelength_grid(
            wavelength,
            oversampling=config.high_resolution_oversampling,
            margin_pixels=config.high_resolution_margin_pixels,
        )
        species_names, basis = optical_depth_basis(
            highres_wavelength,
            line_list,
            chunk_size=config.chunk_size,
        )
        return transmission_from_high_resolution_basis(
            wavelength,
            highres_wavelength,
            species_names,
            basis,
            species_scales=config.species_scales,
            airmass=config.airmass,
            lsf_sigma_pixels=config.lsf_sigma_pixels,
            lsf_box_width_pixels=config.lsf_box_width_pixels,
            lsf_lorentz_fwhm_pixels=config.lsf_lorentz_fwhm_pixels,
            highres_pixels_per_observed_pixel=pixels_per_observed,
            lsf_variable_width=config.lsf_variable_width,
            lsf_reference_wavelength_micron=config.lsf_reference_wavelength_micron,
            lsf_kernel_width_fwhm=config.lsf_kernel_width_fwhm,
            lsf_molecfit_voigt=config.lsf_molecfit_voigt,
            rebin_mode=config.high_resolution_rebin_mode,
        )
    species_names, basis = optical_depth_basis(
        wavelength,
        line_list,
        chunk_size=config.chunk_size,
    )
    return transmission_from_basis(
        species_names,
        basis,
        species_scales=config.species_scales,
        airmass=config.airmass,
        lsf_sigma_pixels=config.lsf_sigma_pixels,
        lsf_box_width_pixels=config.lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=config.lsf_lorentz_fwhm_pixels,
        wavelength_micron=wavelength,
        lsf_variable_width=config.lsf_variable_width,
        lsf_reference_wavelength_micron=config.lsf_reference_wavelength_micron,
        lsf_kernel_width_fwhm=config.lsf_kernel_width_fwhm,
        lsf_molecfit_voigt=config.lsf_molecfit_voigt,
    )
