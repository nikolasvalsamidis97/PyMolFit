from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

from .physics import SPEED_OF_LIGHT_M_PER_S
from .regions import RegionSelection
from .spectrum import Spectrum, wavelength_scale_to_micron

SPEED_OF_LIGHT_KM_S = SPEED_OF_LIGHT_M_PER_S / 1000.0
MaskDepth = float | Literal["auto"]
MaskPadding = float | Literal["auto"]


@dataclass(frozen=True)
class StellarMaskResult:
    """A stellar-template model sampled on an observed spectrum.

    ``mask`` is ``True`` where the broadened theoretical spectrum predicts a
    stellar feature. Those pixels are excluded only from telluric parameter
    estimation. The fitted atmospheric transmission is still evaluated there.
    """

    normalized_flux: np.ndarray
    intrinsic_wavelength_micron: np.ndarray
    intrinsic_normalized_flux: np.ndarray
    mask: np.ndarray
    selection: RegionSelection
    fit_weights: np.ndarray | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def selection_for_spectrum(self, spectrum: Spectrum) -> RegionSelection:
        """Return the mask as regions in ``spectrum``'s native coordinates."""

        if self.mask.shape != spectrum.wavelength.shape:
            raise ValueError("spectrum shape differs from the stellar-mask target")
        padding_kms = float(self.diagnostics.get("mask_padding_kms", 0.0))
        return RegionSelection(
            exclude_ranges=_mask_to_ranges(
                spectrum.wavelength,
                self.mask,
                padding_kms=padding_kms,
            ),
            wavelength_unit=spectrum.wavelength_unit,
            wavelength_medium=spectrum.wavelength_medium,
        )


@dataclass(frozen=True)
class TheoreticalSpectrum:
    """A rest-frame stellar spectrum used to mask astrophysical features.

    The input is a two-column ASCII spectrum. SVO theoretical spectra are
    supported directly, including their commented wavelength-unit metadata and
    physical (non-normalized) flux. PyMolFit estimates a pseudo-continuum,
    shifts the template by ``radial_velocity_kms``, broadens it for stellar
    rotation and instrumental resolution, and optionally refines a small
    residual velocity offset by cross-correlation with the observation.

    The template is used only to decide which observed pixels must not
    constrain the telluric fit. Its flux never replaces or modifies the science
    spectrum.

    :param path: Two-column ASCII theoretical spectrum, including files
        downloaded from the SVO Theoretical Spectra Web Server.
    :param radial_velocity_kms: Stellar barycentric radial velocity in km/s;
        positive values redshift the rest-frame template.
    :param vsini_kms: Projected stellar rotation speed in km/s. ``0`` disables
        rotational broadening.
    :param wavelength_unit: Template wavelength unit. ``"auto"`` reads SVO
        comments and otherwise requires an explicit unit.
    :param wavelength_medium: ``"air"``, ``"vacuum"``, or ``"auto"``. Auto
        recognizes the SVO ASCII signature; SVO BT-family spectra are supplied
        in air wavelengths.
    :param resolving_power: Optional instrumental resolving power. ``None``
        uses observation/FITS metadata when available.
    :param limb_darkening: Linear limb-darkening coefficient in the standard
        Gray rotational kernel; it must lie between 0 and 1.
    :param mask_depth: Minimum normalized stellar-line depth. ``"auto"`` uses
        at least 5 percent and raises that threshold for noisier observations,
        avoiding masks driven by small template/continuum mismatches.
    :param continuum_window_kms: Velocity width used to estimate the template
        pseudo-continuum. It should be wider than the stellar lines.
    :param mask_padding_kms: Extra velocity padding around detected stellar
        features so their wings are also excluded from fitting. ``"auto"``
        uses half the quadrature sum of ``v sin(i)`` and instrumental FWHM,
        with a 10 km/s floor. Supply a non-negative number for explicit
        padding.
    :param fit_velocity_offset: Refine a residual template/observation velocity
        offset after applying the supplied radial velocity.
    :param velocity_search_kms: Symmetric residual-alignment search range.
    :param confidence_weighted_masking: Replace binary template exclusions in
        :func:`correct` with continuous residual weights. Pixels dominated by
        predicted stellar structure constrain the atmosphere less strongly,
        while continuum pixels retain full weight. This experimental mode is
        disabled by default; set it to ``True`` to enable it.
    :param confidence_weight_floor: Minimum residual weight assigned to the
        deepest predicted stellar pixels.
    :param macroturbulence_kms: Optional Gaussian stellar macroturbulent FWHM
        in km/s, applied in addition to rotational and instrumental broadening.
    :param wavelength_col: Zero-based wavelength column in the ASCII file.
    :param flux_col: Zero-based flux column in the ASCII file.
    """

    path: str | Path
    radial_velocity_kms: float
    vsini_kms: float
    wavelength_unit: str = "auto"
    wavelength_medium: str = "auto"
    resolving_power: float | None = None
    limb_darkening: float = 0.6
    mask_depth: MaskDepth = "auto"
    continuum_window_kms: float = 3_000.0
    mask_padding_kms: MaskPadding = "auto"
    fit_velocity_offset: bool = True
    velocity_search_kms: float = 20.0
    confidence_weighted_masking: bool = False
    confidence_weight_floor: float = 0.05
    macroturbulence_kms: float = 0.0
    wavelength_col: int = 0
    flux_col: int = 1
    wavelength: np.ndarray = field(init=False, repr=False, compare=False)
    flux: np.ndarray = field(init=False, repr=False, compare=False)
    metadata: dict[str, object] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        source = Path(self.path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"theoretical spectrum does not exist: {source}")
        if not np.isfinite(self.radial_velocity_kms):
            raise ValueError("radial_velocity_kms must be finite")
        if not np.isfinite(self.vsini_kms) or self.vsini_kms < 0:
            raise ValueError("vsini_kms must be finite and non-negative")
        if not 0.0 <= self.limb_darkening <= 1.0:
            raise ValueError("limb_darkening must lie between 0 and 1")
        if self.resolving_power is not None and (
            not np.isfinite(self.resolving_power) or self.resolving_power <= 0
        ):
            raise ValueError("resolving_power must be positive when supplied")
        if self.mask_depth != "auto" and (
            not np.isfinite(float(self.mask_depth)) or not 0.0 < float(self.mask_depth) < 1.0
        ):
            raise ValueError("mask_depth must be 'auto' or lie between 0 and 1")
        if not np.isfinite(self.continuum_window_kms) or self.continuum_window_kms <= 0:
            raise ValueError("continuum_window_kms must be positive")
        if self.mask_padding_kms != "auto" and (
            not np.isfinite(float(self.mask_padding_kms)) or float(self.mask_padding_kms) < 0
        ):
            raise ValueError("mask_padding_kms must be 'auto' or non-negative")
        if not np.isfinite(self.velocity_search_kms) or self.velocity_search_kms < 0:
            raise ValueError("velocity_search_kms must be non-negative")
        if (
            not np.isfinite(self.confidence_weight_floor)
            or not 0.0 < self.confidence_weight_floor <= 1.0
        ):
            raise ValueError("confidence_weight_floor must lie in (0, 1]")
        if not np.isfinite(self.macroturbulence_kms) or self.macroturbulence_kms < 0:
            raise ValueError("macroturbulence_kms must be finite and non-negative")

        comments = _read_ascii_comments(source)
        unit = self.wavelength_unit
        if unit.strip().lower() == "auto":
            unit = _infer_ascii_wavelength_unit(comments)
            if unit is None:
                raise ValueError(
                    "the theoretical spectrum does not declare a wavelength unit; "
                    "set wavelength_unit explicitly"
                )
        wavelength_scale_to_micron(unit)
        medium = self.wavelength_medium
        if medium.strip().lower() == "auto":
            medium = _infer_ascii_wavelength_medium(comments)
            if medium is None:
                raise ValueError(
                    "the theoretical spectrum does not declare an air/vacuum "
                    "wavelength medium; set wavelength_medium explicitly"
                )
        if medium.strip().lower() not in {"air", "vacuum", "vac"}:
            raise ValueError("wavelength_medium must be 'auto', 'air', or 'vacuum'")
        if medium.strip().lower() == "vac":
            medium = "vacuum"

        columns = np.loadtxt(
            source,
            comments="#",
            usecols=(int(self.wavelength_col), int(self.flux_col)),
            ndmin=2,
        )
        wavelength = np.asarray(columns[:, 0], dtype=float)
        flux = np.asarray(columns[:, 1], dtype=float)
        valid = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
        wavelength = wavelength[valid]
        flux = flux[valid]
        if wavelength.size < 3:
            raise ValueError("theoretical spectrum must contain at least three valid rows")

        order = np.argsort(wavelength, kind="stable")
        wavelength = wavelength[order]
        flux = flux[order]
        unique = np.concatenate(([True], np.diff(wavelength) > 0))
        wavelength = wavelength[unique]
        flux = flux[unique]
        if wavelength.size < 3:
            raise ValueError("theoretical spectrum needs at least three unique wavelengths")

        object.__setattr__(self, "path", source.resolve())
        object.__setattr__(self, "wavelength_unit", unit)
        object.__setattr__(self, "wavelength_medium", medium)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "metadata", _parse_ascii_metadata(comments))

    def build_mask(
        self,
        spectrum: Spectrum,
        *,
        frame_correction_factor: float = 1.0,
        resolving_power: float | None = None,
    ) -> StellarMaskResult:
        """Build stellar exclusion regions on an observatory-frame spectrum.

        ``spectrum`` must already use observatory-frame vacuum wavelengths.
        ``frame_correction_factor`` maps a barycentric wavelength product back
        to that frame and is normally supplied internally by :func:`correct`.
        """

        target = spectrum.to_unit("micron").to_vacuum()
        if not np.isfinite(frame_correction_factor) or frame_correction_factor <= 0:
            raise ValueError("frame_correction_factor must be positive")

        template = (
            Spectrum(
                wavelength=self.wavelength,
                flux=self.flux,
                wavelength_unit=self.wavelength_unit,
                wavelength_medium=self.wavelength_medium,
                meta={"source": str(self.path)},
            )
            .to_unit("micron")
            .to_vacuum()
        )
        radial_factor = _relativistic_doppler_factor(self.radial_velocity_kms)
        shifted_wavelength = template.wavelength * radial_factor / frame_correction_factor

        target_valid = target.valid & (target.wavelength > 0)
        if np.count_nonzero(target_valid) < 3:
            raise ValueError("observed spectrum has fewer than three usable wavelengths")
        target_min = float(np.nanmin(target.wavelength[target_valid]))
        target_max = float(np.nanmax(target.wavelength[target_valid]))
        margin_kms = max(
            self.continuum_window_kms,
            2.0 * self.vsini_kms,
            2.0 * self.velocity_search_kms,
            100.0,
        )
        margin_factor = _relativistic_doppler_factor(margin_kms)
        in_window = (shifted_wavelength >= target_min / margin_factor) & (
            shifted_wavelength <= target_max * margin_factor
        )
        if np.count_nonzero(in_window) < 10:
            raise ValueError(
                "the theoretical spectrum does not cover the observed wavelength range"
            )
        shifted_wavelength = shifted_wavelength[in_window]
        template_flux = template.flux[in_window]

        normalized_template, _ = _normalize_pseudo_continuum(
            shifted_wavelength,
            template_flux,
            window_kms=self.continuum_window_kms,
        )
        active_resolving_power = (
            self.resolving_power if self.resolving_power is not None else resolving_power
        )
        log_grid, intrinsic_profile, velocity_step = _broaden_on_velocity_grid(
            shifted_wavelength,
            normalized_template,
            target.wavelength[target_valid],
            vsini_kms=self.vsini_kms,
            limb_darkening=self.limb_darkening,
            macroturbulence_kms=self.macroturbulence_kms,
            resolving_power=None,
        )
        alignment_profile = _apply_instrumental_broadening(
            intrinsic_profile,
            velocity_step_kms=velocity_step,
            resolving_power=active_resolving_power,
        )

        threshold = _resolve_mask_depth(self.mask_depth, target)
        residual_velocity = 0.0
        alignment_score = np.nan
        if self.fit_velocity_offset and self.velocity_search_kms > 0:
            residual_velocity, alignment_score = _fit_residual_velocity(
                target,
                log_grid,
                alignment_profile,
                threshold=threshold,
                search_kms=self.velocity_search_kms,
                velocity_step_kms=velocity_step,
                continuum_window_kms=self.continuum_window_kms,
            )

        velocity_offsets = np.full(
            target.wavelength.shape,
            residual_velocity,
            dtype=float,
        )
        sampled = _sample_velocity_shifted_profile(
            target.wavelength,
            velocity_offsets,
            log_grid,
            alignment_profile,
        )
        intrinsic_wavelength = np.exp(log_grid) * _relativistic_doppler_factor(residual_velocity)
        stellar_mask = target_valid & np.isfinite(sampled) & (np.abs(sampled - 1.0) >= threshold)
        resolved_mask_padding_kms = _resolve_mask_padding_kms(
            self.mask_padding_kms,
            vsini_kms=self.vsini_kms,
            resolving_power=active_resolving_power,
        )
        ranges = _mask_to_ranges(
            target.wavelength,
            stellar_mask,
            padding_kms=resolved_mask_padding_kms,
        )
        selection = RegionSelection(
            exclude_ranges=ranges,
            wavelength_unit="micron",
            wavelength_medium="vacuum",
        )
        expanded_mask = np.zeros(stellar_mask.shape, dtype=bool)
        for lower, upper in selection.exclude_ranges:
            expanded_mask |= (
                target_valid & (target.wavelength >= lower) & (target.wavelength <= upper)
            )
        covered = target_valid & np.isfinite(sampled)
        fit_weights = None
        if self.confidence_weighted_masking:
            significance = np.zeros(sampled.shape, dtype=float)
            significance[covered] = np.abs(sampled[covered] - 1.0) / threshold
            fit_weights = np.ones(sampled.shape, dtype=float)
            fit_weights[covered] = np.clip(
                1.0 / (1.0 + np.square(significance[covered])),
                self.confidence_weight_floor,
                1.0,
            )
            padded_only = expanded_mask & (significance < 1.0)
            fit_weights[padded_only] = np.minimum(
                fit_weights[padded_only],
                0.5,
            )
            fit_weights[~target_valid] = 0.0
        diagnostics: dict[str, object] = {
            "source": str(self.path),
            "source_metadata": dict(self.metadata),
            "radial_velocity_kms": float(self.radial_velocity_kms),
            "frame_correction_factor": float(frame_correction_factor),
            "residual_velocity_kms": float(residual_velocity),
            "alignment_score": float(alignment_score),
            "vsini_kms": float(self.vsini_kms),
            "macroturbulence_kms": float(self.macroturbulence_kms),
            "limb_darkening": float(self.limb_darkening),
            "resolving_power": (
                None
                if (self.resolving_power is None and resolving_power is None)
                else float(
                    self.resolving_power if self.resolving_power is not None else resolving_power
                )
            ),
            "mask_depth": float(threshold),
            "mask_padding_kms": float(resolved_mask_padding_kms),
            "mask_padding_mode": ("auto" if self.mask_padding_kms == "auto" else "explicit"),
            "core_pixel_count": int(np.count_nonzero(stellar_mask)),
            "masked_pixel_count": int(np.count_nonzero(expanded_mask)),
            "covered_pixel_count": int(np.count_nonzero(covered)),
            "masked_fraction_of_covered": (
                float(np.count_nonzero(expanded_mask) / np.count_nonzero(covered))
                if np.count_nonzero(covered)
                else 0.0
            ),
            "exclude_region_count": len(selection.exclude_ranges),
            "confidence_weighted_masking": bool(self.confidence_weighted_masking),
            "confidence_weight_floor": float(self.confidence_weight_floor),
            "fit_weight_median": (
                None if fit_weights is None else float(np.nanmedian(fit_weights[covered]))
            ),
        }
        return StellarMaskResult(
            normalized_flux=sampled,
            intrinsic_wavelength_micron=intrinsic_wavelength,
            intrinsic_normalized_flux=np.asarray(intrinsic_profile, dtype=float),
            mask=stellar_mask,
            selection=selection,
            fit_weights=fit_weights,
            diagnostics=diagnostics,
        )


def _read_ascii_comments(path: Path) -> tuple[str, ...]:
    comments: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#"):
                comments.append(stripped.lstrip("#").strip())
            elif stripped:
                break
    return tuple(comments)


def _infer_ascii_wavelength_unit(comments: tuple[str, ...]) -> str | None:
    text = " ".join(comments).lower()
    if "angstrom" in text or "ångström" in text:
        return "angstrom"
    if re.search(r"\bmicron(?:s)?\b|\bmicromet(?:er|re)s?\b|\bum\b", text):
        return "micron"
    if re.search(r"\bnanomet(?:er|re)s?\b|\bnm\b", text):
        return "nm"
    return None


def _infer_ascii_wavelength_medium(comments: tuple[str, ...]) -> str | None:
    text = " ".join(comments).lower()
    if "vacuum wavelength" in text or "wavelength (vacuum" in text:
        return "vacuum"
    if "air wavelength" in text or "wavelength (air" in text:
        return "air"
    svo_ascii = (
        "column 1: wavelength" in text and "column 2: flux" in text and "erg/cm2/s/a" in text
    )
    svo_air_collections = (
        "bt-cond",
        "bt-dusty",
        "bt-nextgen",
        "bt-settl",
    )
    if svo_ascii and any(name in text for name in svo_air_collections):
        return "air"
    return None


def _parse_ascii_metadata(comments: tuple[str, ...]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if comments:
        metadata["description"] = comments[0]
    for line in comments:
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        normalized_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if not normalized_key:
            continue
        number = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
        metadata[normalized_key] = float(number.group(0)) if number else value
    return metadata


def _relativistic_doppler_factor(velocity_kms: float) -> float:
    beta = float(velocity_kms) / SPEED_OF_LIGHT_KM_S
    if abs(beta) >= 1.0:
        raise ValueError("velocity magnitude must be below the speed of light")
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def _normalize_pseudo_continuum(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    *,
    window_kms: float,
) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength_micron, dtype=float)
    values = np.asarray(flux, dtype=float)
    valid = np.isfinite(wavelength) & np.isfinite(values) & (wavelength > 0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("not enough finite template samples to estimate a continuum")
    valid_indices = np.flatnonzero(valid)
    order = np.argsort(wavelength[valid], kind="stable")
    sorted_indices = valid_indices[order]
    sorted_wavelength = wavelength[sorted_indices]
    sorted_flux = values[sorted_indices]
    coordinate = SPEED_OF_LIGHT_KM_S * np.log(sorted_wavelength)
    span = float(np.ptp(coordinate))
    width = min(float(window_kms), max(span / 3.0, 50.0))
    step = max(width / 3.0, 10.0)
    centers = np.arange(coordinate[0], coordinate[-1] + step, step)
    envelope = np.empty(centers.size, dtype=float)
    half = 0.5 * width
    for index, center in enumerate(centers):
        lower = np.searchsorted(coordinate, center - half, side="left")
        upper = np.searchsorted(coordinate, center + half, side="right")
        local = sorted_flux[lower:upper]
        local = local[np.isfinite(local)]
        envelope[index] = np.nanpercentile(local, 95.0) if local.size else np.nan
    finite_envelope = np.isfinite(envelope) & (envelope > 0)
    if np.count_nonzero(finite_envelope) < 2:
        scale = float(np.nanmedian(sorted_flux))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("theoretical flux has no positive continuum")
        continuum_valid = np.full(coordinate.shape, scale)
    else:
        smoothed = gaussian_filter1d(
            envelope[finite_envelope],
            sigma=1.0,
            mode="nearest",
        )
        continuum_valid = np.interp(
            coordinate,
            centers[finite_envelope],
            smoothed,
        )
    positive = continuum_valid[np.isfinite(continuum_valid) & (continuum_valid > 0)]
    floor = max(float(np.nanmedian(positive)) * 1.0e-12, np.finfo(float).tiny)
    continuum_valid = np.maximum(continuum_valid, floor)
    normalized = np.full(values.shape, np.nan)
    continuum = np.full(values.shape, np.nan)
    normalized[sorted_indices] = np.clip(sorted_flux / continuum_valid, 0.0, 2.0)
    continuum[sorted_indices] = continuum_valid
    return normalized, continuum


def _broaden_on_velocity_grid(
    wavelength_micron: np.ndarray,
    normalized_flux: np.ndarray,
    target_wavelength_micron: np.ndarray,
    *,
    vsini_kms: float,
    limb_darkening: float,
    macroturbulence_kms: float = 0.0,
    resolving_power: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    valid = np.isfinite(wavelength_micron) & np.isfinite(normalized_flux) & (wavelength_micron > 0)
    wavelength = np.asarray(wavelength_micron, dtype=float)[valid]
    profile = np.asarray(normalized_flux, dtype=float)[valid]
    template_dv = _median_velocity_spacing(wavelength)
    target_dv = _median_velocity_spacing(target_wavelength_micron)
    finite_steps = [value for value in (template_dv, target_dv) if np.isfinite(value)]
    velocity_step = float(np.clip(min(finite_steps or [1.0]), 0.25, 5.0))
    log_step = velocity_step / SPEED_OF_LIGHT_KM_S
    log_grid = np.arange(np.log(wavelength[0]), np.log(wavelength[-1]) + log_step, log_step)
    broadened = np.interp(log_grid, np.log(wavelength), profile)

    if vsini_kms > 0:
        half_pixels = max(1, int(np.ceil(vsini_kms / velocity_step)))
        velocity = np.arange(-half_pixels, half_pixels + 1) * velocity_step
        x = velocity / vsini_kms
        inside = np.abs(x) < 1.0
        kernel = np.zeros_like(x)
        kernel[inside] = 2.0 * (1.0 - limb_darkening) * np.sqrt(
            1.0 - x[inside] ** 2
        ) + 0.5 * np.pi * limb_darkening * (1.0 - x[inside] ** 2)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0:
            broadened = fftconvolve(broadened, kernel / kernel_sum, mode="same")

    if resolving_power is not None and resolving_power > 0:
        broadened = _apply_instrumental_broadening(
            broadened,
            velocity_step_kms=velocity_step,
            resolving_power=resolving_power,
        )
    if macroturbulence_kms > 0:
        macro_sigma_pixels = macroturbulence_kms / (2.354820045 * velocity_step)
        if macro_sigma_pixels >= 0.15:
            broadened = gaussian_filter1d(
                broadened,
                sigma=macro_sigma_pixels,
                mode="nearest",
            )
    return log_grid, np.asarray(broadened, dtype=float), velocity_step


def _apply_instrumental_broadening(
    profile: np.ndarray,
    *,
    velocity_step_kms: float,
    resolving_power: float | None,
) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    if resolving_power is None or resolving_power <= 0:
        return values.copy()
    sigma_velocity = SPEED_OF_LIGHT_KM_S / (2.354820045 * resolving_power)
    sigma_pixels = sigma_velocity / velocity_step_kms
    if sigma_pixels < 0.15:
        return values.copy()
    return gaussian_filter1d(values, sigma=sigma_pixels, mode="nearest")


def _sample_velocity_shifted_profile(
    wavelength_micron: np.ndarray,
    velocity_offsets_kms: np.ndarray,
    log_grid: np.ndarray,
    profile: np.ndarray,
) -> np.ndarray:
    wavelength = np.asarray(wavelength_micron, dtype=float)
    offsets = np.asarray(velocity_offsets_kms, dtype=float)
    if offsets.shape != wavelength.shape:
        raise ValueError("velocity offsets must match the target wavelength shape")
    coordinate = np.full(wavelength.shape, np.nan)
    valid = np.isfinite(wavelength) & (wavelength > 0) & np.isfinite(offsets)
    beta = offsets[valid] / SPEED_OF_LIGHT_KM_S
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError("velocity magnitude must be below the speed of light")
    factors = np.sqrt((1.0 + beta) / (1.0 - beta))
    coordinate[valid] = np.log(wavelength[valid]) - np.log(factors)
    return np.interp(coordinate, log_grid, profile, left=np.nan, right=np.nan)


def _median_velocity_spacing(wavelength: np.ndarray) -> float:
    values = np.asarray(wavelength, dtype=float)
    values = np.unique(values[np.isfinite(values) & (values > 0)])
    if values.size < 2:
        return np.nan
    differences = np.diff(np.log(values)) * SPEED_OF_LIGHT_KM_S
    positive = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.nanmedian(positive)) if positive.size else np.nan


def _resolve_mask_depth(requested: MaskDepth, spectrum: Spectrum) -> float:
    if requested != "auto":
        return float(requested)
    if spectrum.uncertainty is None:
        return 0.05
    valid = spectrum.valid & (np.abs(spectrum.flux) > 0)
    noise = np.abs(spectrum.uncertainty[valid] / spectrum.flux[valid])
    noise = noise[np.isfinite(noise) & (noise > 0)]
    if noise.size == 0:
        return 0.05
    return float(np.clip(5.0 * np.nanmedian(noise), 0.05, 0.12))


def _resolve_mask_padding_kms(
    requested: MaskPadding,
    *,
    vsini_kms: float,
    resolving_power: float | None,
) -> float:
    if requested != "auto":
        return float(requested)
    instrumental_fwhm_kms = (
        0.0
        if resolving_power is None or resolving_power <= 0
        else SPEED_OF_LIGHT_KM_S / float(resolving_power)
    )
    broadening_fwhm_kms = float(np.hypot(vsini_kms, instrumental_fwhm_kms))
    return max(10.0, 0.5 * broadening_fwhm_kms)


def _fit_residual_velocity(
    spectrum: Spectrum,
    log_grid: np.ndarray,
    template_flux: np.ndarray,
    *,
    threshold: float,
    search_kms: float,
    velocity_step_kms: float,
    continuum_window_kms: float,
) -> tuple[float, float]:
    normalized_observed, _ = _normalize_pseudo_continuum(
        spectrum.wavelength,
        spectrum.flux,
        window_kms=continuum_window_kms,
    )
    usable = spectrum.valid & np.isfinite(normalized_observed) & (spectrum.wavelength > 0)
    if np.count_nonzero(usable) < 20:
        return 0.0, np.nan
    step = float(np.clip(velocity_step_kms, 0.25, 2.0))
    offsets = np.arange(-search_kms, search_kms + 0.5 * step, step)
    scores = np.full(offsets.shape, -np.inf)
    observed_depth = np.clip(1.0 - normalized_observed, -0.5, 1.5)
    log_wavelength = np.full(spectrum.wavelength.shape, np.nan)
    positive_wavelength = np.isfinite(spectrum.wavelength) & (spectrum.wavelength > 0)
    log_wavelength[positive_wavelength] = np.log(spectrum.wavelength[positive_wavelength])
    for index, offset in enumerate(offsets):
        shifted = np.interp(
            log_wavelength - np.log(_relativistic_doppler_factor(offset)),
            log_grid,
            template_flux,
            left=np.nan,
            right=np.nan,
        )
        template_depth = 1.0 - shifted
        active = usable & np.isfinite(template_depth) & (np.abs(template_depth) >= 0.5 * threshold)
        if np.count_nonzero(active) < 20:
            continue
        x = template_depth[active]
        y = observed_depth[active]
        x = x - np.nanmedian(x)
        y = y - np.nanmedian(y)
        denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
        if denominator > 0:
            scores[index] = float(np.dot(x, y) / denominator)
    if not np.any(np.isfinite(scores)):
        return 0.0, np.nan
    best = int(np.nanargmax(scores))
    zero = int(np.argmin(np.abs(offsets)))
    best_score = float(scores[best])
    zero_score = float(scores[zero])
    if best in {0, offsets.size - 1}:
        return 0.0, zero_score
    if best_score < 0.1 or best_score - zero_score < 0.01:
        return 0.0, zero_score
    return float(offsets[best]), best_score


def _mask_to_ranges(
    wavelength_micron: np.ndarray,
    mask: np.ndarray,
    *,
    padding_kms: float,
) -> tuple[tuple[float, float], ...]:
    wavelength = np.asarray(wavelength_micron, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    order = np.argsort(wavelength, kind="stable")
    wavelength = wavelength[order]
    selected = selected[order]
    finite = np.isfinite(wavelength) & (wavelength > 0)
    wavelength = wavelength[finite]
    selected = selected[finite]
    if wavelength.size == 0 or not np.any(selected):
        return ()

    spacing = np.diff(wavelength)
    positive = spacing[spacing > 0]
    typical_spacing = float(np.nanmedian(positive)) if positive.size else 0.0
    discontinuity = np.concatenate(([True], spacing > max(10.0 * typical_spacing, 0.0)))
    starts = selected & np.concatenate(([True], ~selected[:-1] | discontinuity[1:]))
    stops = selected & np.concatenate((~selected[1:] | discontinuity[1:], [True]))
    start_values = wavelength[starts]
    stop_values = wavelength[stops]
    factor = _relativistic_doppler_factor(padding_kms)
    ranges = tuple(
        (float(lower / factor), float(upper * factor))
        for lower, upper in zip(start_values, stop_values, strict=True)
    )
    return RegionSelection(
        exclude_ranges=ranges,
        wavelength_unit="micron",
        wavelength_medium="vacuum",
    ).exclude_ranges
