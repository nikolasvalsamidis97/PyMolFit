"""Paired continuum rescue with an unchanged baseline fallback."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._automatic import _continuum_prediction_score, validation_pixels
from ._wavelength_validation import evaluate_wavelength_validation

COMPLEXITY = {
    "none": 0,
    "constant_pixel_shift": 1,
    "linear_pixel_trend": 2,
    "quadratic_pixel_trend": 3,
}
MINIMUM_TRIM_GAIN = 0.02
MAXIMUM_FULL_REGRESSION = 0.01
MAXIMUM_DOMAIN_REGRESSION = 0.02
MINIMUM_DOMAIN_PIXELS = 32


def defer_rescued_pair(original, candidates, original_matrix, rescued, details):
    """Restore the original decision and retain a supported pair for a late refit.

    The original-matrix replay performs no additional fitting and uses exactly
    the baseline selection policy.  The full native workflow must finish before
    the rescued pair can replace it, so physical rejection has a real fallback.
    """
    rescue = details.get("paired_continuum_rescue", {})
    if "selected_continuum_order" not in rescue:
        return rescued, details
    rescue["rescued_comparisons"] = details.get("comparisons", [])
    rescue["rescued_reference_fold_scores"] = details.get("reference_fold_scores")
    rescue["rescued_selected_model"] = rescued[0]
    baseline = original
    comparisons = []
    if isinstance(original_matrix[0], dict):
        details.update(status="validation_error", error=original_matrix[0]["error"])
        details.pop("reference_fold_scores", None)
        details.pop("reference_model", None)
    else:
        current = original_matrix[0]
        details["reference_model"] = original[0]
        details["reference_fold_scores"] = current
        for candidate, scores in zip(candidates, original_matrix[1:], strict=True):
            if isinstance(scores, dict):
                comparison = {
                    "from_model": baseline[0],
                    "to_model": candidate[0],
                    "accepted": False,
                    "reason": "candidate_validation_error",
                    "error": scores["error"],
                }
            else:
                comparison = evaluate_wavelength_validation(current, scores)
                comparison.update(
                    from_model=baseline[0],
                    to_model=candidate[0],
                    bounds_pixels=list(candidate[1].wavelength_shift_bounds),
                    fold_scores=scores,
                )
                if comparison["accepted"]:
                    baseline, current = candidate, scores
            comparisons.append(comparison)
    details["comparisons"] = comparisons
    details["selected_model"] = baseline[0]
    # Rescue must have accepted a wavelength upgrade and must not replace a
    # stronger original-matrix choice with a less complex family.
    upgraded = (
        rescued is not original
        and rescued is not baseline
        and COMPLEXITY[rescued[0]] >= COMPLEXITY[baseline[0]]
    )
    if upgraded:
        details["deferred_continuum_wavelength_pair"] = {
            "continuum_order": int(rescue["selected_continuum_order"]),
            "wavelength_model": rescued[0],
            "wavelength_shift_bounds": list(rescued[1].wavelength_shift_bounds),
            "original_selected_model": baseline[0],
            "validation": "common_fixed_degree_guarded_absorption_prediction",
        }
    return baseline, details


def _domains(reference, exclude_ranges):
    valid = validation_pixels(reference, exclude_ranges)
    valid &= np.isfinite(reference.spectrum.uncertainty) & (reference.spectrum.uncertainty > 0)
    trimmed = valid.copy()
    trimmed[:40] = False
    trimmed[-40:] = False
    result = {"full": valid, "trim40": trimmed}
    for name, lo, hi in (
        ("opaque", 0.0, 0.1),
        ("deep", 0.1, 0.5),
        ("moderate", 0.5, 0.9),
        ("shallow", 0.9, 0.98),
        ("transparent", 0.98, np.inf),
        ("telluric", 0.1, 0.98),
    ):
        result[name] = trimmed & (reference.transmission >= lo) & (reference.transmission < hi)
    return result


def _fixed_full_comparison(reference, candidate, config):
    valid = reference.spectrum.valid
    if np.any(candidate.transmission[valid] < 0) or np.any(candidate.model_flux[valid] < 0):
        raise ValueError("negative physical model or transmission")
    scores, reasons = {}, []
    for name, mask in _domains(reference, config.exclude_ranges).items():
        count = int(mask.sum())
        if count < MINIMUM_DOMAIN_PIXELS and name not in {"full", "trim40"}:
            scores[name] = {"pixels": count, "evaluated": False}
            continue
        before, _ = _continuum_prediction_score(reference, reference, mask)
        after, _ = _continuum_prediction_score(candidate, reference, mask)
        if before["rms"] <= np.finfo(float).tiny and after["rms"] > np.finfo(float).tiny:
            raise ValueError("zero reference error cannot be improved by a nonzero residual")
        improvement = (
            1.0 - after["rms"] / before["rms"] if before["rms"] > np.finfo(float).tiny else 0.0
        )
        scores[name] = {
            "pixels": count,
            "evaluated": True,
            "before_rms": before["rms"],
            "after_rms": after["rms"],
            "rms_improvement_fraction": float(improvement),
        }
        threshold = MAXIMUM_FULL_REGRESSION if name == "full" else MAXIMUM_DOMAIN_REGRESSION
        if improvement < -threshold:
            reasons.append(name + "_regression")
    if scores["trim40"]["rms_improvement_fraction"] < MINIMUM_TRIM_GAIN:
        reasons.append("trim40_improvement_below_two_percent")
    hits = {k: v for k, v in candidate.parameter_bound_status.items() if "wavelength_" in k}
    if config.fit_wavelength_shift or config.fit_wavelength_polynomial:
        lo, hi = config.wavelength_shift_bounds
        margin = 0.01 * (hi - lo)
        for index, value in enumerate(candidate.wavelength_coefficients):
            if value <= lo + margin or value >= hi - margin:
                hits[f"coefficient_{index}"] = "near_bound"
    if hits:
        reasons.append("wavelength_coefficient_at_bound")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "scores": scores,
        "wavelength_bound_status": hits,
    }


def apply_rescued_pair(reference, config, deferred, *, run_fit, starting_config):
    """Try one already-validated pair after the complete native fitting pipeline."""
    details = {
        "method": "deferred_paired_continuum_wavelength_rescue",
        "accepted": False,
        "pair": dict(deferred),
        "baseline_continuum_order": int(config.continuum_order),
        "conditional_on_full_data_structure": True,
        "score_scope": "original_valid_nonexcluded_pixels",
        "thresholds": {
            "minimum_trim_gain": MINIMUM_TRIM_GAIN,
            "maximum_full_regression": MAXIMUM_FULL_REGRESSION,
            "maximum_domain_regression": MAXIMUM_DOMAIN_REGRESSION,
            "minimum_domain_pixels": MINIMUM_DOMAIN_PIXELS,
        },
    }
    if (
        config.wavelength_shift_unit != "pixel"
        or config.fit_segment_wavelength_shifts
        or config.fit_segment_wavelength_polynomial
        or config.wavelength_group_polynomial_orders
    ):
        return reference, config, {**details, "reason": "unsupported_wavelength_geometry"}
    if tuple(deferred["wavelength_shift_bounds"]) != tuple(config.wavelength_shift_bounds):
        return reference, config, {**details, "reason": "deferred_bounds_differ_from_baseline"}
    name = deferred["wavelength_model"]
    candidate_config = replace(
        starting_config,
        continuum_order=int(deferred["continuum_order"]),
        fit_wavelength_shift=name == "constant_pixel_shift",
        fit_wavelength_polynomial=name in {"linear_pixel_trend", "quadratic_pixel_trend"},
        wavelength_polynomial_order=2 if name == "quadratic_pixel_trend" else 1,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=config.wavelength_shift_bounds,
        fit_ranges=config.fit_ranges,
        exclude_ranges=config.exclude_ranges,
        initial_wavelength_shift=0.0,
        initialize_wavelength_shift_grid=True,
        initialize_lsf_sigma_grid=starting_config.fit_lsf_sigma,
    )
    details["starting_policy"] = {
        "full_data_fitted_parameters_reused": False,
        "initial_species_scales": candidate_config.initial_species_scales,
        "lsf_sigma_pixels": candidate_config.lsf_sigma_pixels,
        "initial_wavelength_shift": candidate_config.initial_wavelength_shift,
    }
    try:
        candidate = run_fit(candidate_config)
        comparison = _fixed_full_comparison(reference, candidate, candidate_config)
        details["full_comparison"] = comparison
        if not comparison["passed"]:
            return reference, config, {**details, "reason": "full_fixed_domain_screen_rejected"}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return reference, config, {**details, "reason": "candidate_invalid", "error": str(exc)}
    return (
        candidate,
        candidate_config,
        {
            **details,
            "accepted": True,
            "reason": "rescued_pair_passed_guarded_prediction_and_full_fixed_domain_screen",
        },
    )
