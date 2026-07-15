from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy.table import Table

from .fit import FitConfig, MultiTelluricFitResult, fit_telluric_segments
from .linelist import LineList
from .spectrum import Spectrum


@dataclass(frozen=True)
class SpectrumComparison:
    n_pixels: int
    overlap_min: float
    overlap_max: float
    bias: float
    rms: float
    median_abs: float
    max_abs: float
    relative_rms: float
    correlation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n_pixels": self.n_pixels,
            "overlap_min": self.overlap_min,
            "overlap_max": self.overlap_max,
            "bias": self.bias,
            "rms": self.rms,
            "median_abs": self.median_abs,
            "max_abs": self.max_abs,
            "relative_rms": self.relative_rms,
            "correlation": self.correlation,
        }


@dataclass(frozen=True)
class TelluricCrossValidationResult:
    """Out-of-fold predictions from blocked multi-segment telluric fits."""

    spectra: tuple[Spectrum, ...]
    fold_results: tuple[MultiTelluricFitResult, ...]
    fold_assignment: tuple[np.ndarray, ...]
    transmission: tuple[np.ndarray, ...]
    continuum: tuple[np.ndarray, ...]
    model_flux: tuple[np.ndarray, ...]
    corrected_relative_flux: tuple[np.ndarray, ...]
    telluric_mask: tuple[np.ndarray, ...]
    segment_metrics: tuple[Mapping[str, float], ...]
    metrics: Mapping[str, float]
    block_size: int
    n_folds: int

    def to_table(self, segment: int) -> Table:
        if segment < 0 or segment >= len(self.spectra):
            raise IndexError("cross-validation segment index is out of range")
        spectrum = self.spectra[segment]
        table = Table()
        table["wavelength"] = spectrum.wavelength
        table["wavelength"].unit = spectrum.wavelength_unit
        table["flux"] = spectrum.flux
        if spectrum.uncertainty is not None:
            table["uncertainty"] = spectrum.uncertainty
        table["input_valid"] = spectrum.valid
        table["holdout_fold"] = self.fold_assignment[segment]
        table["oof_continuum"] = self.continuum[segment]
        table["oof_transmission"] = self.transmission[segment]
        table["oof_model_flux"] = self.model_flux[segment]
        table["oof_corrected_relative_flux"] = self.corrected_relative_flux[segment]
        table["oof_telluric_mask"] = self.telluric_mask[segment]
        table.meta.update(
            {
                "cross_validation_block_size": self.block_size,
                "cross_validation_n_folds": self.n_folds,
                "cross_validation_split": "contiguous_pixel_blocks_modulo_fold",
                "cross_validation_metrics": json.dumps(dict(self.metrics), sort_keys=True),
                "cross_validation_segment_metrics": json.dumps(
                    dict(self.segment_metrics[segment]),
                    sort_keys=True,
                ),
            }
        )
        return table

    def write(
        self,
        directory: str | Path,
        *,
        format: str = "ascii.ecsv",
        prefix: str = "segment",
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "ecsv" if format == "ascii.ecsv" else format.rsplit(".", 1)[-1]
        for index in range(len(self.spectra)):
            self.to_table(index).write(
                directory / f"{prefix}_{index + 1:02d}.{suffix}",
                format=format,
                overwrite=True,
            )
        (directory / f"{prefix}_cross_validation_summary.json").write_text(
            json.dumps(dict(self.metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def cross_validate_telluric_segments(
    spectra: Sequence[Spectrum],
    *,
    line_list: LineList,
    config: FitConfig,
    block_size: int = 64,
    n_folds: int = 2,
    telluric_threshold: float = 0.995,
    min_transmission: float | None = None,
    require_success: bool = True,
) -> TelluricCrossValidationResult:
    """Measure prediction quality on wavelength blocks excluded from fitting.

    Pixels are assigned to contiguous blocks using only their array position;
    block numbers are then distributed modulo ``n_folds``. Every valid pixel
    is predicted exactly once by a fit that did not consume that pixel. The
    split therefore cannot adapt to observed residuals or a reference model.
    """

    if block_size < 1:
        raise ValueError("block_size must be positive")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not 0.0 < telluric_threshold < 1.0:
        raise ValueError("telluric_threshold must be between 0 and 1")
    threshold = config.min_transmission if min_transmission is None else float(min_transmission)
    if not 0.0 < threshold < telluric_threshold:
        raise ValueError("min_transmission must be positive and below telluric_threshold")

    ordered = tuple(spectrum.to_unit("micron").sorted() for spectrum in spectra)
    if not ordered:
        raise ValueError("spectra must contain at least one segment")

    assignments: list[np.ndarray] = []
    for index, spectrum in enumerate(ordered):
        assignment = (np.arange(spectrum.wavelength.size) // block_size) % n_folds
        assignment = assignment.astype(int)
        assignment[~spectrum.valid] = -1
        active = np.unique(assignment[assignment >= 0])
        if active.size != n_folds:
            raise ValueError(
                f"segment {index} does not contain at least one valid block for every fold"
            )
        assignments.append(assignment)

    oof_transmission = [np.full(spectrum.flux.shape, np.nan) for spectrum in ordered]
    oof_continuum = [np.full(spectrum.flux.shape, np.nan) for spectrum in ordered]
    oof_model = [np.full(spectrum.flux.shape, np.nan) for spectrum in ordered]
    fold_results: list[MultiTelluricFitResult] = []
    for fold in range(n_folds):
        fit_masks = []
        for index, (spectrum, assignment) in enumerate(zip(ordered, assignments, strict=True)):
            training = spectrum.valid & (assignment != fold)
            holdout = spectrum.valid & (assignment == fold)
            minimum_pixels = config.continuum_order + 2
            if np.count_nonzero(training) < minimum_pixels or not np.any(holdout):
                raise ValueError(f"segment {index} has insufficient training or holdout pixels")
            fit_masks.append(training)
        result = fit_telluric_segments(
            ordered,
            line_list=line_list,
            config=config,
            fit_masks=fit_masks,
        )
        if require_success and not result.success:
            raise RuntimeError(f"cross-validation fold {fold} failed: {result.message}")
        fold_results.append(result)
        for index, segment in enumerate(result.segment_results):
            holdout = assignments[index] == fold
            oof_transmission[index][holdout] = segment.transmission[holdout]
            oof_continuum[index][holdout] = segment.continuum[holdout]
            oof_model[index][holdout] = segment.model_flux[holdout]

    corrected_relative: list[np.ndarray] = []
    telluric_masks: list[np.ndarray] = []
    segment_metrics: list[dict[str, float]] = []
    aggregate: dict[str, list[np.ndarray]] = {
        "raw_relative": [],
        "corrected_relative": [],
        "continuum_z": [],
        "model_z": [],
    }
    total_valid = 0
    total_predicted = 0
    total_reliable = 0
    for spectrum, transmission, continuum, model in zip(
        ordered,
        oof_transmission,
        oof_continuum,
        oof_model,
        strict=True,
    ):
        predicted = (
            spectrum.valid
            & np.isfinite(transmission)
            & np.isfinite(continuum)
            & np.isfinite(model)
            & (continuum != 0.0)
        )
        valid = predicted & (transmission >= threshold)
        telluric = valid & (transmission < telluric_threshold)
        corrected = np.full(spectrum.flux.shape, np.nan)
        corrected[valid] = spectrum.flux[valid] / (continuum[valid] * transmission[valid])
        raw_relative = np.full(spectrum.flux.shape, np.nan)
        raw_relative[valid] = spectrum.flux[valid] / continuum[valid] - 1.0
        corrected_residual = corrected - 1.0
        if spectrum.uncertainty is None:
            continuum_z = np.full(spectrum.flux.shape, np.nan)
            model_z = np.full(spectrum.flux.shape, np.nan)
        else:
            continuum_z = np.full(spectrum.flux.shape, np.nan)
            model_z = np.full(spectrum.flux.shape, np.nan)
            weighted = valid & np.isfinite(spectrum.uncertainty) & (spectrum.uncertainty > 0)
            continuum_z[weighted] = (
                spectrum.flux[weighted] - continuum[weighted]
            ) / spectrum.uncertainty[weighted]
            model_z[weighted] = (
                spectrum.flux[weighted] - model[weighted]
            ) / spectrum.uncertainty[weighted]

        raw_rms = _masked_rms(raw_relative, telluric)
        corrected_rms = _masked_rms(corrected_residual, telluric)
        continuum_weighted_rms = _masked_rms(continuum_z, telluric)
        model_weighted_rms = _masked_rms(model_z, telluric)
        segment_metrics.append(
            {
                "n_valid_pixels": float(np.count_nonzero(spectrum.valid)),
                "n_predicted_pixels": float(np.count_nonzero(predicted)),
                "n_reliable_correction_pixels": float(np.count_nonzero(valid)),
                "n_telluric_pixels": float(np.count_nonzero(telluric)),
                "telluric_relative_rms_raw": raw_rms,
                "telluric_relative_rms_corrected": corrected_rms,
                "telluric_relative_rms_improvement": _safe_ratio(raw_rms, corrected_rms),
                "telluric_weighted_rms_continuum_only": continuum_weighted_rms,
                "telluric_weighted_rms_model": model_weighted_rms,
                "telluric_weighted_rms_improvement": _safe_ratio(
                    continuum_weighted_rms,
                    model_weighted_rms,
                ),
            }
        )
        corrected_relative.append(corrected)
        telluric_masks.append(telluric)
        total_valid += int(np.count_nonzero(spectrum.valid))
        total_predicted += int(np.count_nonzero(predicted))
        total_reliable += int(np.count_nonzero(valid))
        aggregate["raw_relative"].append(raw_relative[telluric])
        aggregate["corrected_relative"].append(corrected_residual[telluric])
        aggregate["continuum_z"].append(continuum_z[telluric])
        aggregate["model_z"].append(model_z[telluric])

    raw = _concatenate_finite(aggregate["raw_relative"])
    corrected = _concatenate_finite(aggregate["corrected_relative"])
    continuum_z = _concatenate_finite(aggregate["continuum_z"])
    model_z = _concatenate_finite(aggregate["model_z"])
    raw_rms = _array_rms(raw)
    corrected_rms = _array_rms(corrected)
    continuum_weighted_rms = _array_rms(continuum_z)
    model_weighted_rms = _array_rms(model_z)
    metrics = {
        "block_size": float(block_size),
        "n_folds": float(n_folds),
        "n_segments": float(len(ordered)),
        "n_valid_pixels": float(total_valid),
        "n_predicted_pixels": float(total_predicted),
        "prediction_coverage": float(total_predicted / total_valid),
        "n_reliable_correction_pixels": float(total_reliable),
        "reliable_correction_coverage": float(total_reliable / total_valid),
        "n_telluric_pixels": float(raw.size),
        "telluric_relative_rms_raw": raw_rms,
        "telluric_relative_rms_corrected": corrected_rms,
        "telluric_relative_rms_improvement": _safe_ratio(raw_rms, corrected_rms),
        "telluric_weighted_rms_continuum_only": continuum_weighted_rms,
        "telluric_weighted_rms_model": model_weighted_rms,
        "telluric_weighted_rms_improvement": _safe_ratio(
            continuum_weighted_rms,
            model_weighted_rms,
        ),
        "all_folds_successful": float(all(result.success for result in fold_results)),
    }
    return TelluricCrossValidationResult(
        spectra=ordered,
        fold_results=tuple(fold_results),
        fold_assignment=tuple(assignments),
        transmission=tuple(oof_transmission),
        continuum=tuple(oof_continuum),
        model_flux=tuple(oof_model),
        corrected_relative_flux=tuple(corrected_relative),
        telluric_mask=tuple(telluric_masks),
        segment_metrics=tuple(segment_metrics),
        metrics=metrics,
        block_size=block_size,
        n_folds=n_folds,
    )


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    return _array_rms(selected[np.isfinite(selected)])


def _array_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan


def _concatenate_finite(parts: Sequence[np.ndarray]) -> np.ndarray:
    finite = [np.asarray(part, dtype=float)[np.isfinite(part)] for part in parts]
    finite = [part for part in finite if part.size]
    return np.concatenate(finite) if finite else np.asarray([], dtype=float)


VALIDATION_STATUSES = frozenset({"PASS", "FAIL", "WARN", "MANUAL", "SKIP"})


@dataclass(frozen=True)
class ValidationCheck:
    category: str
    name: str
    status: str
    value: float | None = None
    threshold: str = ""
    units: str = ""
    details: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        normalized = str(self.status).strip().upper()
        if normalized not in VALIDATION_STATUSES:
            raise ValueError(f"unsupported validation status: {self.status!r}")
        object.__setattr__(self, "status", normalized)
        if not self.category.strip() or not self.name.strip():
            raise ValueError("validation category and name must be non-empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "units": self.units,
            "details": self.details,
            "required": self.required,
        }


@dataclass(frozen=True)
class ScienceReadinessReport:
    checks: tuple[ValidationCheck, ...]
    metadata: Mapping[str, object]
    generated_at_utc: str

    @classmethod
    def create(
        cls,
        checks: Sequence[ValidationCheck],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> "ScienceReadinessReport":
        return cls(
            checks=tuple(checks),
            metadata={} if metadata is None else dict(metadata),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def verdict(self) -> str:
        required = tuple(check for check in self.checks if check.required)
        if any(check.status == "FAIL" for check in required):
            return "NOT_SCIENCE_READY"
        if any(check.status in {"MANUAL", "SKIP"} for check in required):
            return "VALIDATION_INCOMPLETE"
        if any(check.status == "WARN" for check in required):
            return "CONDITIONALLY_READY"
        return "SCIENCE_READY"

    def status_counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in sorted(VALIDATION_STATUSES)
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "generated_at_utc": self.generated_at_utc,
            "metadata": dict(self.metadata),
            "status_counts": self.status_counts(),
            "checks": [check.as_dict() for check in self.checks],
        }

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "science_readiness_report.json").write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (directory / "science_readiness_checks.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            fieldnames = list(ValidationCheck("x", "x", "PASS").as_dict())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(check.as_dict() for check in self.checks)


def compare_spectra(
    candidate: Spectrum,
    reference: Spectrum,
    *,
    normalize: bool = False,
) -> SpectrumComparison:
    """Compare two spectra on their common wavelength grid.

    The reference spectrum is linearly interpolated onto the candidate grid.
    This is deliberately generic so it can compare PyMolFit output to a
    Molecfit product, a simulation, or another reduction package.
    """

    candidate = candidate.to_unit("micron").sorted()
    reference = reference.to_unit("micron").sorted()
    overlap_min = max(float(np.nanmin(candidate.wavelength)), float(np.nanmin(reference.wavelength)))
    overlap_max = min(float(np.nanmax(candidate.wavelength)), float(np.nanmax(reference.wavelength)))
    if overlap_max <= overlap_min:
        raise ValueError("spectra do not overlap in wavelength")

    keep = (candidate.wavelength >= overlap_min) & (candidate.wavelength <= overlap_max)
    keep &= candidate.valid
    reference_valid = reference.valid
    if np.count_nonzero(keep) < 2 or np.count_nonzero(reference_valid) < 2:
        raise ValueError("not enough valid pixels to compare")

    candidate_flux = candidate.flux[keep]
    reference_flux = np.interp(
        candidate.wavelength[keep],
        reference.wavelength[reference_valid],
        reference.flux[reference_valid],
    )
    finite = np.isfinite(candidate_flux) & np.isfinite(reference_flux)
    candidate_flux = candidate_flux[finite]
    reference_flux = reference_flux[finite]
    if candidate_flux.size < 2:
        raise ValueError("not enough finite overlap pixels to compare")

    if normalize:
        candidate_scale = np.nanmedian(candidate_flux)
        reference_scale = np.nanmedian(reference_flux)
        if candidate_scale != 0:
            candidate_flux = candidate_flux / candidate_scale
        if reference_scale != 0:
            reference_flux = reference_flux / reference_scale

    delta = candidate_flux - reference_flux
    reference_scale = np.nanmedian(np.abs(reference_flux))
    if np.nanstd(candidate_flux) == 0 or np.nanstd(reference_flux) == 0:
        correlation = np.nan
    else:
        correlation = np.corrcoef(candidate_flux, reference_flux)[0, 1]
    return SpectrumComparison(
        n_pixels=int(candidate_flux.size),
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        bias=float(np.nanmean(delta)),
        rms=float(np.sqrt(np.nanmean(delta**2))),
        median_abs=float(np.nanmedian(np.abs(delta))),
        max_abs=float(np.nanmax(np.abs(delta))),
        relative_rms=float(np.sqrt(np.nanmean(delta**2)) / reference_scale) if reference_scale != 0 else np.nan,
        correlation=float(correlation),
    )
