from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from astropy.table import Table

from .fit import (
    FitConfig,
    MultiTelluricFitResult,
    TelluricFitResult,
    fit_telluric_segments,
    fit_tellurics,
)
from .linelist import LineList
from .spectrum import Spectrum, correct_spectrum


@dataclass(frozen=True)
class ModelSystematicsResult:
    """A baseline fit plus refitted model variants and their correction spread."""

    baseline: TelluricFitResult
    variants: Mapping[str, TelluricFitResult]
    transmission_systematic_uncertainty: np.ndarray
    transmission_systematic_envelope: np.ndarray
    combined_transmission_uncertainty: np.ndarray
    corrected: Spectrum
    metrics: Mapping[str, float]

    def to_table(self) -> Table:
        table = self.baseline.to_table()
        table["transmission_systematic_uncertainty"] = self.transmission_systematic_uncertainty
        table["transmission_systematic_envelope"] = self.transmission_systematic_envelope
        table["combined_transmission_uncertainty"] = self.combined_transmission_uncertainty
        table["corrected_flux_with_systematics"] = self.corrected.flux
        if self.corrected.uncertainty is not None:
            table["corrected_uncertainty_with_systematics"] = self.corrected.uncertainty
        table["corrected_mask_with_systematics"] = self.corrected.valid

        used_columns: set[str] = set(table.colnames)
        variant_columns: dict[str, str] = {}
        for label, result in self.variants.items():
            stem = _column_label(label)
            column = f"transmission_variant_{stem}"
            suffix = 2
            while column in used_columns:
                column = f"transmission_variant_{stem}_{suffix}"
                suffix += 1
            used_columns.add(column)
            table[column] = result.transmission
            variant_columns[str(label)] = column

        table.meta.update(
            {
                "systematic_variant_labels": json.dumps(list(self.variants), ensure_ascii=True),
                "systematic_variant_columns": json.dumps(variant_columns, sort_keys=True),
                "systematic_metrics": json.dumps(dict(self.metrics), sort_keys=True),
                "systematic_combination": "root_mean_square_about_baseline",
            }
        )
        return table

    def write(self, path: str | Path, *, format: str = "ascii.ecsv") -> None:
        self.to_table().write(path, format=format, overwrite=True)


@dataclass(frozen=True)
class MultiModelSystematicsResult:
    """Joint multi-segment fits and per-segment model-systematic products."""

    baseline: MultiTelluricFitResult
    variants: Mapping[str, MultiTelluricFitResult]
    segment_results: tuple[ModelSystematicsResult, ...]
    metrics: Mapping[str, float]

    def write(
        self,
        directory: str | Path,
        *,
        format: str = "ascii.ecsv",
        prefix: str = "segment",
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = _format_suffix(format)
        for index, result in enumerate(self.segment_results, 1):
            result.write(directory / f"{prefix}_{index:02d}.{suffix}", format=format)
        (directory / f"{prefix}_systematics_summary.json").write_text(
            json.dumps(dict(self.metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def fit_tellurics_with_systematics(
    spectrum: Spectrum,
    line_list: LineList,
    baseline_config: FitConfig,
    variant_configs: Mapping[str, FitConfig],
    *,
    baseline_result: TelluricFitResult | None = None,
    max_workers: int = 1,
    require_success: bool = True,
) -> ModelSystematicsResult:
    """Refit a spectrum under alternative physical/model assumptions.

    Each variant is a complete :class:`FitConfig`, so the ensemble can test
    atmosphere profiles, continuum choices, line-wing settings, or LSF
    families without embedding observation-specific calibration factors.
    The per-pixel RMS about the baseline is treated as model-systematic
    transmission uncertainty and is combined in quadrature with the baseline
    fit's local statistical transmission uncertainty.
    """

    configs = {str(label): config for label, config in variant_configs.items()}
    if not configs:
        raise ValueError("variant_configs must contain at least one named FitConfig")
    if any(not label.strip() for label in configs):
        raise ValueError("systematic variant labels must be non-empty")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if any(not isinstance(config, FitConfig) for config in configs.values()):
        raise TypeError("every systematic variant must be a FitConfig")

    baseline = (
        fit_tellurics(spectrum, line_list=line_list, config=baseline_config)
        if baseline_result is None
        else baseline_result
    )
    _validate_result_grid(baseline, spectrum, label="baseline")
    if require_success and not baseline.success:
        raise RuntimeError(f"baseline telluric fit failed: {baseline.message}")

    def run(item: tuple[str, FitConfig]) -> tuple[str, TelluricFitResult]:
        label, config = item
        return label, fit_tellurics(spectrum, line_list=line_list, config=config)

    items = tuple(configs.items())
    if max_workers == 1:
        fitted = tuple(run(item) for item in items)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fitted = tuple(executor.map(run, items))
    variants = dict(fitted)

    for label, result in variants.items():
        _validate_result_grid(result, spectrum, label=label)
        if require_success and not result.success:
            raise RuntimeError(f"systematic variant {label!r} failed: {result.message}")

    return _combine_single_systematics(
        baseline,
        variants,
        min_transmission=baseline_config.min_transmission,
    )


def fit_telluric_segments_with_systematics(
    spectra: Sequence[Spectrum],
    line_list: LineList,
    baseline_config: FitConfig,
    variant_configs: Mapping[str, FitConfig],
    *,
    baseline_result: MultiTelluricFitResult | None = None,
    fit_masks: Sequence[np.ndarray | None] | None = None,
    continuum_priors: Sequence[np.ndarray | None] | None = None,
    max_workers: int = 1,
    require_success: bool = True,
) -> MultiModelSystematicsResult:
    """Refit shared-parameter spectral segments under named model variants."""

    spectra = tuple(spectra)
    if not spectra:
        raise ValueError("spectra must contain at least one segment")
    configs = _validate_variant_configs(variant_configs, max_workers=max_workers)
    baseline = (
        fit_telluric_segments(
            spectra,
            line_list=line_list,
            config=baseline_config,
            fit_masks=fit_masks,
            continuum_priors=continuum_priors,
        )
        if baseline_result is None
        else baseline_result
    )
    _validate_multi_result(baseline, spectra, label="baseline")
    if require_success and not baseline.success:
        raise RuntimeError(f"baseline telluric fit failed: {baseline.message}")

    def run(item: tuple[str, FitConfig]) -> tuple[str, MultiTelluricFitResult]:
        label, config = item
        return label, fit_telluric_segments(
            spectra,
            line_list=line_list,
            config=config,
            fit_masks=fit_masks,
            continuum_priors=continuum_priors,
        )

    items = tuple(configs.items())
    if max_workers == 1:
        fitted = tuple(run(item) for item in items)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fitted = tuple(executor.map(run, items))
    variants = dict(fitted)
    for label, result in variants.items():
        _validate_multi_result(result, spectra, label=label)
        if require_success and not result.success:
            raise RuntimeError(f"systematic variant {label!r} failed: {result.message}")

    segment_results = tuple(
        _combine_single_systematics(
            baseline.segment_results[index],
            {
                label: result.segment_results[index]
                for label, result in variants.items()
            },
            min_transmission=baseline_config.min_transmission,
        )
        for index in range(len(spectra))
    )
    rms = np.concatenate(
        [result.transmission_systematic_uncertainty for result in segment_results]
    )
    envelope = np.concatenate(
        [result.transmission_systematic_envelope for result in segment_results]
    )
    finite = np.isfinite(rms)
    corrected_squares = []
    for result in segment_results:
        value = result.metrics["corrected_flux_variant_rms"]
        if np.isfinite(value):
            corrected_squares.append(float(value) ** 2)
    metrics = {
        "variant_count": float(len(variants)),
        "segment_count": float(len(segment_results)),
        "transmission_systematic_rms_median": _finite_percentile(rms[finite], 50.0),
        "transmission_systematic_rms_p95": _finite_percentile(rms[finite], 95.0),
        "transmission_systematic_envelope_max": _finite_max(envelope[finite]),
        "corrected_flux_variant_rms": (
            float(np.sqrt(np.mean(corrected_squares))) if corrected_squares else np.nan
        ),
        "finite_systematic_fraction": float(np.mean(finite)),
    }
    return MultiModelSystematicsResult(
        baseline=baseline,
        variants=variants,
        segment_results=segment_results,
        metrics=metrics,
    )


def _combine_single_systematics(
    baseline: TelluricFitResult,
    variants: Mapping[str, TelluricFitResult],
    *,
    min_transmission: float,
) -> ModelSystematicsResult:
    variant_transmissions = np.stack(
        [np.asarray(result.transmission, dtype=float) for result in variants.values()],
        axis=0,
    )
    baseline_transmission = np.asarray(baseline.transmission, dtype=float)
    deltas = variant_transmissions - baseline_transmission[None, :]
    finite = np.isfinite(deltas)
    count = np.count_nonzero(finite, axis=0)
    sum_squares = np.sum(np.where(finite, deltas**2, 0.0), axis=0)
    rms = np.full(baseline_transmission.shape, np.nan, dtype=float)
    np.sqrt(sum_squares, out=rms, where=count > 0)
    rms[count > 0] /= np.sqrt(count[count > 0])
    envelope = np.full(baseline_transmission.shape, np.nan, dtype=float)
    envelope[count > 0] = np.max(
        np.where(finite[:, count > 0], np.abs(deltas[:, count > 0]), -np.inf),
        axis=0,
    )

    statistical = baseline.transmission_uncertainty
    if statistical is None:
        combined = rms.copy()
    else:
        statistical = np.asarray(statistical, dtype=float)
        if statistical.shape != rms.shape:
            raise ValueError("baseline transmission uncertainty has the wrong shape")
        combined = np.sqrt(statistical**2 + rms**2)

    corrected = correct_spectrum(
        baseline.spectrum,
        baseline_transmission,
        transmission_uncertainty=combined,
        min_transmission=min_transmission,
    )
    corrected = Spectrum(
        wavelength=corrected.wavelength,
        flux=corrected.flux,
        uncertainty=corrected.uncertainty,
        mask=corrected.mask,
        wavelength_unit=corrected.wavelength_unit,
        wavelength_medium=corrected.wavelength_medium,
        meta={
            **dict(corrected.meta),
            "model_systematic_uncertainty_propagated": True,
            "systematic_variant_labels": tuple(variants),
        },
    )

    valid = np.isfinite(baseline_transmission) & np.isfinite(rms)
    corrected_deltas = []
    for result in variants.values():
        keep = baseline.corrected.valid & result.corrected.valid
        if np.any(keep):
            corrected_deltas.append(result.corrected.flux[keep] - baseline.corrected.flux[keep])
    concatenated_corrected = (
        np.concatenate(corrected_deltas) if corrected_deltas else np.asarray([], dtype=float)
    )
    metrics = {
        "variant_count": float(len(variants)),
        "transmission_systematic_rms_median": _finite_percentile(rms[valid], 50.0),
        "transmission_systematic_rms_p95": _finite_percentile(rms[valid], 95.0),
        "transmission_systematic_envelope_max": _finite_max(envelope[valid]),
        "corrected_flux_variant_rms": (
            float(np.sqrt(np.mean(concatenated_corrected**2)))
            if concatenated_corrected.size
            else np.nan
        ),
        "finite_systematic_fraction": float(np.mean(valid)),
    }
    return ModelSystematicsResult(
        baseline=baseline,
        variants=dict(variants),
        transmission_systematic_uncertainty=rms,
        transmission_systematic_envelope=envelope,
        combined_transmission_uncertainty=combined,
        corrected=corrected,
        metrics=metrics,
    )


def _validate_variant_configs(
    variant_configs: Mapping[str, FitConfig],
    *,
    max_workers: int,
) -> dict[str, FitConfig]:
    configs = {str(label): config for label, config in variant_configs.items()}
    if not configs:
        raise ValueError("variant_configs must contain at least one named FitConfig")
    if any(not label.strip() for label in configs):
        raise ValueError("systematic variant labels must be non-empty")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if any(not isinstance(config, FitConfig) for config in configs.values()):
        raise TypeError("every systematic variant must be a FitConfig")
    return configs


def _validate_multi_result(
    result: MultiTelluricFitResult,
    spectra: Sequence[Spectrum],
    *,
    label: str,
) -> None:
    if len(result.segment_results) != len(spectra):
        raise ValueError(f"{label} result segment count does not match the spectra")
    for index, (segment, spectrum) in enumerate(
        zip(result.segment_results, spectra, strict=True)
    ):
        _validate_result_grid(segment, spectrum.to_unit("micron").sorted(), label=f"{label}[{index}]")


def _format_suffix(format: str) -> str:
    normalized = str(format).strip().lower()
    return "ecsv" if normalized == "ascii.ecsv" else normalized.rsplit(".", 1)[-1]


def _validate_result_grid(
    result: TelluricFitResult,
    spectrum: Spectrum,
    *,
    label: str,
) -> None:
    if result.transmission.shape != spectrum.flux.shape:
        raise ValueError(f"{label} result transmission shape does not match the spectrum")
    if result.spectrum.wavelength.shape != spectrum.wavelength.shape or not np.allclose(
        result.spectrum.wavelength,
        spectrum.wavelength,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError(f"{label} result wavelength grid does not match the spectrum")


def _column_label(label: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", label.strip()).strip("_").lower()
    return value or "variant"


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else np.nan


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else np.nan
