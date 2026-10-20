"""Guard a reduced automatic-support fit against the existing broad-support fit."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from ._automatic import (
    _available_pixel_ranges,
    _continuum_prediction_score,
    continuum_guarded_plan,
    sampling_indices,
    validation_pixels,
)
from ._joint_selection import cv_comparison, domain_masks
from .fit import _windows_mask
from .regions import _normalize_ranges


def local_support_ranges(segments, catalogue_ranges, exclude_ranges, *, continuum_order):
    """Add only the nearest eligible pixels where a catalogue window exists."""
    ranges = list(catalogue_ranges)
    for spectrum in segments:
        wave = spectrum.to_unit("micron").wavelength
        available = spectrum.valid & _windows_mask(wave, None, exclude_ranges)
        selected = available & _windows_mask(wave, catalogue_ranges, None)
        required = min(
            int(available.sum()),
            max(32, 8 * (continuum_order + 1), int(np.ceil(0.25 * available.sum()))),
        )
        if not selected.any():
            # With no telluric seed there is no supported local alternative.
            ranges.extend(_available_pixel_ranges(spectrum, available))
            continue
        if selected.sum() >= required:
            continue
        distance = np.full(wave.size, np.inf)
        for indices in sampling_indices(spectrum):
            seeds = np.flatnonzero(selected[indices])
            if not seeds.size:
                continue
            pixels = np.arange(indices.size)
            insertion = np.searchsorted(seeds, pixels)
            left = seeds[np.maximum(insertion - 1, 0)]
            right = seeds[np.minimum(insertion, seeds.size - 1)]
            distance[indices] = np.minimum(np.abs(pixels - left), np.abs(pixels - right))
        candidates = np.flatnonzero(available & ~selected & np.isfinite(distance))
        nearest = candidates[np.argsort(distance[candidates], kind="stable")]
        selected[nearest[: required - int(selected.sum())]] = True
        ranges.extend(_available_pixel_ranges(spectrum, selected))
    return _normalize_ranges(ranges)


def _reference_mask(reference, mask):
    return SimpleNamespace(**{**vars(reference), "fit_mask": mask})


def refine_region_support(
    reference, config, segments, catalogue_ranges, *, run_fit, starting_config
):
    """Accept reduced support only after paired, guarded telluric prediction tests."""
    details = {
        "method": "guarded_local_automatic_support",
        "accepted": False,
        "conditional_on_full_data_structure": True,
        "score_scope": "original_valid_nonexcluded_catalogue_windows",
        "baseline_fit_pixels": int(reference.fit_mask.sum()),
        "full_data_fitted_parameters_reused": False,
    }
    if not reference.success:
        return reference, config, {**details, "reason": "unsuccessful_reference"}
    ranges = local_support_ranges(
        segments, catalogue_ranges, config.exclude_ranges, continuum_order=config.continuum_order
    )
    wave = reference.spectrum.to_unit("micron").wavelength
    valid = validation_pixels(reference, config.exclude_ranges)
    local_mask = valid & _windows_mask(wave, ranges, None)
    catalogue_mask = valid & _windows_mask(wave, catalogue_ranges, None)
    details["candidate_fit_pixels"] = int(local_mask.sum())
    details["candidate_fit_ranges_micron"] = ranges
    if np.array_equal(local_mask, reference.fit_mask):
        return reference, config, {**details, "reason": "support_unchanged"}
    if np.any(local_mask & ~reference.fit_mask) or not catalogue_mask.any():
        return reference, config, {**details, "reason": "unsupported_support_geometry"}
    local_reference = _reference_mask(reference, local_mask)
    local_config = replace(config, fit_ranges=ranges)
    folds, plan = continuum_guarded_plan(
        local_reference, local_config, config.continuum_order, score_pixels=catalogue_mask
    )
    details["plan"] = plan
    if not folds:
        return reference, config, {**details, "reason": "insufficient_guarded_support"}

    try:
        baseline_scores, candidate_scores = [], []
        for fold in folds:
            exclusions = tuple(config.exclude_ranges or ()) + tuple(fold["exclude_ranges_micron"])
            for mask_reference, fit_ranges, scores in (
                (reference, config.fit_ranges, baseline_scores),
                (local_reference, ranges, candidate_scores),
            ):
                trial = run_fit(
                    replace(
                        starting_config,
                        fit_ranges=fit_ranges,
                        exclude_ranges=exclusions,
                        estimate_uncertainties=False,
                    )
                )
                score, residuals = _continuum_prediction_score(
                    trial, mask_reference, fold["score_mask"], fold["guard_mask"]
                )
                scores.append({**score, "residuals": residuals})
        comparison = cv_comparison(
            baseline_scores, candidate_scores, domain_masks(reference, config.exclude_ranges), folds
        )
        details["validation"] = comparison
        if not comparison["accepted"]:
            return reference, config, {**details, "reason": "no_predictive_support"}
        candidate_config = replace(starting_config, fit_ranges=ranges)
        candidate = run_fit(candidate_config)
        before, _ = _continuum_prediction_score(reference, reference, catalogue_mask)
        after, _ = _continuum_prediction_score(candidate, local_reference, catalogue_mask)
        details["catalogue_scores"] = {"baseline": before, "candidate": after}
        old_bounds = getattr(reference, "parameter_bound_status", {})
        new_bounds = {
            name: state
            for name, state in getattr(candidate, "parameter_bound_status", {}).items()
            if old_bounds.get(name) != state
        }
        details["new_parameter_bound_hits"] = new_bounds
        if (
            after["rms"] > before["rms"]
            or after["median_absolute"] > before["median_absolute"]
            or np.any(candidate.transmission[valid] < 0)
            or not np.any(candidate.transmission[valid] >= config.min_transmission)
            or new_bounds
        ):
            return reference, config, {**details, "reason": "catalogue_or_physical_regression"}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return reference, config, {**details, "reason": "candidate_invalid", "error": str(exc)}
    return (
        candidate,
        candidate_config,
        {**details, "accepted": True, "reason": "guarded_support_improved"},
    )
