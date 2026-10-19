"""Guarded prediction checks for automatic wavelength-model selection.

The reference transmission locates features; it is not independent ground truth.
Callers must refit every candidate using only ``train_masks`` and must preserve
the input grids, errors, physical groups, and continuum-segment boundaries.
This module builds masks and scores predictions; it never runs a fit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks, peak_widths


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2], strict=True)


def _sampling_runs(mask: np.ndarray, wavelength: np.ndarray):
    spacing = np.abs(np.diff(wavelength))
    finite_spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    typical = float(np.median(finite_spacing)) if finite_spacing.size else np.inf
    for start, stop in _runs(mask):
        breaks = np.flatnonzero(spacing[start : stop - 1] > 5 * typical) + start + 1
        edges = np.r_[start, breaks, stop]
        yield from pairwise(edges)


def _signature(wavelengths, masks) -> str:
    digest = hashlib.sha256()
    for wave, mask in zip(wavelengths, masks, strict=True):
        digest.update(np.asarray(wave, dtype="<f8").tobytes())
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def build_wavelength_validation_plan(
    segment_results: Sequence,
    *,
    continuum_order: int,
    lsf_sigma_pixels: float,
    wavelength_guard_pixels: float = 0.0,
) -> tuple[list[dict], dict]:
    """Make three deterministic folds containing complete absorption features.

    Feature cores extend to 95% of their prominence width and at least one
    instrumental FWHM. A further FWHM on each side is withheld but not scored;
    ``wavelength_guard_pixels`` enlarges this to cover allowed line displacement.
    Peaks are located in continuous valid stretches, never across detector gaps.
    Score pixels also respect the reference fit mask and exclude T < 0.1.
    Features with deep centres require at least four eligible moderate pixels
    in each wing. Entire feature cores and guards must fit within a continuous
    valid stretch; a processing boundary or masked gap never truncates a guard.
    At most twelve nonoverlapping guarded features are distributed in wavelength.
    Every affected segment retains enough training pixels and support on both
    sides. Insufficient support returns no folds with an explanatory diagnostic.

    Fold dictionaries contain Boolean train/score masks on the original grids,
    JSON-compatible block records, and private copies of the original wavelength,
    flux and uncertainty arrays for checking correspondence when scoring.
    """
    if continuum_order < 0:
        raise ValueError("continuum_order must be non-negative")
    if not np.isfinite(lsf_sigma_pixels) or lsf_sigma_pixels < 0:
        raise ValueError("lsf_sigma_pixels must be finite and non-negative")
    if not np.isfinite(wavelength_guard_pixels) or wavelength_guard_pixels < 0:
        raise ValueError("wavelength_guard_pixels must be finite and non-negative")
    parts = tuple(segment_results)
    minimum_training = max(32, 8 * (int(continuum_order) + 1))
    side_support = max(4, int(continuum_order) + 2)
    fwhm = max(2.0, 2.354820045 * float(lsf_sigma_pixels))
    guard = int(np.ceil(max(fwhm, float(wavelength_guard_pixels))))
    diagnostics = {
        "method": "three_fold_guarded_reference_absorption_features",
        "reference_dependent_screening": True,
        "minimum_training_pixels": minimum_training,
        "minimum_side_training_pixels": side_support,
        "feature_prominence_fraction": 0.95,
        "minimum_scored_transmission": 0.1,
        "deep_feature_minimum_moderate_wing_pixels": 4,
        "guard_pixels": guard,
        "instrumental_fwhm_pixels": fwhm,
        "wavelength_guard_pixels": float(wavelength_guard_pixels),
        "fold_count": 0,
        "block_count": 0,
        "reason": "insufficient_usable_absorption_features",
    }
    if not parts:
        diagnostics["reason"] = "no_segments"
        return [], diagnostics

    eligible, valid_runs, waves, fluxes, errors, transmissions = [], [], [], [], [], []
    for part in parts:
        spectrum = part.spectrum
        wave = np.asarray(spectrum.wavelength, dtype=float)
        flux = np.asarray(spectrum.flux, dtype=float)
        transmission = np.asarray(part.transmission, dtype=float)
        valid = np.asarray(spectrum.valid, dtype=bool)
        if spectrum.uncertainty is None:
            diagnostics["reason"] = "missing_original_uncertainty"
            return [], diagnostics
        sigma = np.asarray(spectrum.uncertainty, dtype=float)
        if any(values.shape != wave.shape for values in (flux, transmission, valid, sigma)):
            raise ValueError("segment arrays must have identical shapes")
        finite = valid & np.isfinite(wave) & np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
        fit_mask = finite if part.fit_mask is None else np.asarray(part.fit_mask, dtype=bool)
        if fit_mask.shape != wave.shape:
            raise ValueError("fit mask must match its original segment grid")
        eligible.append(finite & fit_mask)
        valid_runs.append(finite & np.isfinite(transmission))
        waves.append(wave.copy())
        fluxes.append(flux.copy())
        errors.append(sigma.copy())
        transmissions.append(transmission)

    candidates = []
    for index, (wave, transmission, usable, valid) in enumerate(
        zip(waves, transmissions, eligible, valid_runs, strict=True)
    ):
        for start, stop in _sampling_runs(valid, wave):
            if stop - start < 7:
                continue
            absorption = 1.0 - transmission[start:stop]
            peaks, properties = find_peaks(absorption, prominence=0.03)
            if not peaks.size:
                continue
            _, _, lefts, rights = peak_widths(
                absorption,
                peaks,
                rel_height=0.95,
                prominence_data=(
                    properties["prominences"],
                    properties["left_bases"],
                    properties["right_bases"],
                ),
            )
            for local, peak in enumerate(peaks):
                center = int(start + peak)
                if not usable[center] or not 0.0 <= transmission[center] <= 0.9:
                    continue
                left = max(int(start), int(np.floor(start + min(lefts[local], peak - fwhm / 2))))
                right = min(
                    int(stop), int(np.ceil(start + max(rights[local], peak + fwhm / 2))) + 1
                )
                lower, upper = left - guard, right + guard
                if lower < start or upper > stop:
                    continue
                if transmission[center] < 0.1:
                    moderate = usable & (transmission >= 0.1) & (transmission <= 0.95)
                    if (
                        np.count_nonzero(moderate[left:center]) < 4
                        or np.count_nonzero(moderate[center + 1 : right]) < 4
                    ):
                        continue
                scoring = usable[left:right] & (transmission[left:right] >= 0.1)
                if np.count_nonzero(scoring) < 3:
                    continue
                remaining = usable.copy()
                remaining[lower:upper] = False
                if (
                    np.count_nonzero(remaining) < minimum_training
                    or np.count_nonzero(remaining[:lower]) < side_support
                    or np.count_nonzero(remaining[upper:]) < side_support
                ):
                    continue
                candidates.append(
                    {
                        "segment_index": index,
                        "center_index": center,
                        "center_wavelength": float(wave[center]),
                        "core_start": left,
                        "core_stop": right,
                        "withhold_start": lower,
                        "withhold_stop": upper,
                        "prominence": float(properties["prominences"][local]),
                    }
                )

    # Keep the more prominent member of overlapping blends, deterministically.
    chosen = []
    for block in sorted(
        candidates, key=lambda b: (-b["prominence"], b["segment_index"], b["center_index"])
    ):
        if any(
            block["segment_index"] == other["segment_index"]
            and block["withhold_start"] < other["withhold_stop"]
            and other["withhold_start"] < block["withhold_stop"]
            for other in chosen
        ):
            continue
        chosen.append(block)
    chosen.sort(key=lambda b: (b["center_wavelength"], b["segment_index"], b["center_index"]))
    if len(chosen) > 12:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, 12, dtype=int)]
    diagnostics["candidate_block_count"] = len(candidates)

    # Removing an unsupported block and rebuilding all folds prevents combined
    # holdouts from starving a chunk even when each individual block was safe.
    while len(chosen) >= 3:
        folds, unsupported = [], []
        for fold_index in range(3):
            blocks = chosen[fold_index::3]
            train = [mask.copy() for mask in eligible]
            score = [np.zeros_like(mask) for mask in eligible]
            for block in blocks:
                index = block["segment_index"]
                train[index][block["withhold_start"] : block["withhold_stop"]] = False
                core = slice(block["core_start"], block["core_stop"])
                score[index][core] = eligible[index][core] & (transmissions[index][core] >= 0.1)
            for index in {block["segment_index"] for block in blocks}:
                local = [block for block in blocks if block["segment_index"] == index]
                lower = min(block["withhold_start"] for block in local)
                upper = max(block["withhold_stop"] for block in local)
                if (
                    np.count_nonzero(train[index]) < minimum_training
                    or np.count_nonzero(train[index][:lower]) < side_support
                    or np.count_nonzero(train[index][upper:]) < side_support
                ):
                    unsupported.extend(local)
            folds.append(
                {
                    "fold_index": fold_index,
                    "train_masks": train,
                    "score_masks": score,
                    "eligible_masks": [mask.copy() for mask in eligible],
                    "blocks": [dict(block) for block in blocks],
                    "score_signature": _signature(waves, score),
                    "_wavelengths": waves,
                    "_fluxes": fluxes,
                    "_uncertainties": errors,
                }
            )
        if not unsupported:
            diagnostics.update(
                {
                    "reason": "ready",
                    "fold_count": 3,
                    "block_count": len(chosen),
                    "blocks": [dict(block, fold_index=i % 3) for i, block in enumerate(chosen)],
                    "fold_score_pixel_counts": [
                        int(sum(np.count_nonzero(mask) for mask in fold["score_masks"]))
                        for fold in folds
                    ],
                }
            )
            return folds, diagnostics
        remove = min(
            unsupported, key=lambda b: (b["prominence"], b["segment_index"], b["center_index"])
        )
        chosen.remove(remove)
    return [], diagnostics


def score_wavelength_validation_fold(result, fold: dict) -> dict:
    """Score untouched observations using only this fold's withheld pixels.

    The original errors are used without rescaling. Failed fits, changed grids,
    changed input observations/errors, invalid predictions, and nonpositive
    continua in originally eligible pixels invalidate the entire fold.
    """
    summary = {
        "valid": False,
        "reason": "invalid_fit",
        "weighted_rss": None,
        "point_count": 0,
        "rms": None,
        "median_absolute_residual": None,
        "score_signature": fold.get("score_signature"),
    }
    if hasattr(result, "segment_results"):
        if not result.success:
            summary["reason"] = "fit_did_not_converge"
            return summary
        parts = tuple(result.segment_results)
    else:
        parts = tuple(result)
    if len(parts) != len(fold["score_masks"]):
        summary["reason"] = "segment_count_changed"
        return summary
    residuals = []
    for index, (part, mask) in enumerate(zip(parts, fold["score_masks"], strict=True)):
        if not part.success:
            summary["reason"] = "fit_did_not_converge"
            return summary
        if part.fit_mask is None or not np.array_equal(
            np.asarray(part.fit_mask, dtype=bool), fold["train_masks"][index]
        ):
            summary["reason"] = "training_mask_mismatch"
            return summary
        spectrum = part.spectrum
        originals = (
            fold["_wavelengths"][index],
            fold["_fluxes"][index],
            fold["_uncertainties"][index],
        )
        actuals = (spectrum.wavelength, spectrum.flux, spectrum.uncertainty)
        if any(
            actual is None or not np.array_equal(actual, original, equal_nan=True)
            for actual, original in zip(actuals, originals, strict=True)
        ):
            summary["reason"] = "input_grid_flux_or_uncertainty_changed"
            return summary
        eligible = fold["eligible_masks"][index]
        model = np.asarray(part.model_flux, dtype=float)
        continuum = np.asarray(part.continuum, dtype=float)
        if model.shape != mask.shape or continuum.shape != mask.shape:
            summary["reason"] = "prediction_grid_changed"
            return summary
        if np.any(~np.isfinite(model[eligible])) or np.any(~np.isfinite(continuum[eligible])):
            summary["reason"] = "nonfinite_prediction"
            return summary
        if np.any(continuum[eligible] <= 0):
            summary["reason"] = "nonpositive_continuum"
            return summary
        residuals.append((originals[1][mask] - model[mask]) / originals[2][mask])
    values = np.concatenate(residuals) if residuals else np.array([])
    if not values.size or not np.all(np.isfinite(values)):
        summary["reason"] = "empty_or_nonfinite_residuals"
        return summary
    rss = float(np.sum(np.square(values)))
    summary.update(
        {
            "valid": True,
            "reason": "scored",
            "weighted_rss": rss,
            "point_count": int(values.size),
            "rms": float(np.sqrt(rss / values.size)),
            "median_absolute_residual": float(np.median(np.abs(values))),
        }
    )
    return summary


def evaluate_wavelength_validation(reference_fold_metrics, candidate_fold_metrics) -> dict:
    """Require a repeatable >=5% prediction improvement across three folds.

    At least two folds must improve RMS. The mean fold median absolute residual
    must not worsen, with at least two individual median scores nonworse. These
    are conservative prediction checks, not independent-pixel significance tests.
    """
    answer = {
        "accepted": False,
        "relative_rms_improvement": None,
        "fold_improvements": [],
        "reason": "three_valid_paired_folds_required",
    }
    reference, candidate = list(reference_fold_metrics), list(candidate_fold_metrics)
    if len(reference) != 3 or len(candidate) != 3:
        return answer
    for before, after in zip(reference, candidate, strict=True):
        if not before.get("valid") or not after.get("valid"):
            return answer
        if before.get("point_count", 0) <= 0 or before["point_count"] != after.get("point_count"):
            answer["reason"] = "score_pixels_differ"
            return answer
        if before.get("score_signature") != after.get("score_signature"):
            answer["reason"] = "score_pixels_differ"
            return answer
        for metric in ("weighted_rss", "rms", "median_absolute_residual"):
            if any(
                value.get(metric) is None or not np.isfinite(value[metric]) or value[metric] < 0
                for value in (before, after)
            ):
                return answer
    before_rss = sum(item["weighted_rss"] for item in reference)
    after_rss = sum(item["weighted_rss"] for item in candidate)
    if before_rss <= np.finfo(float).tiny:
        answer["reason"] = "reference_residual_already_zero"
        return answer
    improvement = 1.0 - np.sqrt(after_rss / before_rss)
    fold_improvements = [
        float(1.0 - after["rms"] / before["rms"])
        if before["rms"] > 0
        else (0.0 if after["rms"] == 0 else None)
        for before, after in zip(reference, candidate, strict=True)
    ]
    before_medians = np.asarray([item["median_absolute_residual"] for item in reference])
    after_medians = np.asarray([item["median_absolute_residual"] for item in candidate])
    answer.update(
        {
            "relative_rms_improvement": float(improvement),
            "fold_improvements": fold_improvements,
            "reference_mean_fold_median_absolute_residual": float(np.mean(before_medians)),
            "candidate_mean_fold_median_absolute_residual": float(np.mean(after_medians)),
        }
    )
    if improvement < 0.05 - 1e-12:
        answer["reason"] = "aggregate_rms_improvement_below_five_percent"
    elif sum(value is not None and value > 0 for value in fold_improvements) < 2:
        answer["reason"] = "rms_improvement_not_repeated_across_folds"
    elif (
        np.mean(after_medians) > np.mean(before_medians) + 1e-12
        or np.count_nonzero(after_medians <= before_medians + 1e-12) < 2
    ):
        answer["reason"] = "median_absolute_residual_tradeoff"
    else:
        answer.update({"accepted": True, "reason": "supported_by_guarded_prediction"})
    return answer
