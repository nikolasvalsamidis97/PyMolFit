"""Bounded joint continuum/wavelength model reconsideration.

Full-data screening and reference-defined masks make the inner validation
conditional; external measurements are still needed to assess generalization.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._automatic import _continuum_prediction_score, continuum_guarded_plan, validation_pixels

RECONSIDER_CONTINUUM = True
FAMILIES = ("none", "constant", "linear", "quadratic")
MINIMUM_FULL_GAIN = 0.02
MINIMUM_CV_GAIN = 0.05
MAXIMUM_FULL_REGRESSION = 0.01
MAXIMUM_DOMAIN_REGRESSION = 0.02
MAXIMUM_FOLD_REGRESSION = 0.05
MAXIMUM_SHORTLIST = 2
MINIMUM_DOMAIN_PIXELS = 32


def family(config):
    if config.fit_wavelength_polynomial:
        return "quadratic" if config.wavelength_polynomial_order == 2 else "linear"
    return "constant" if config.fit_wavelength_shift else "none"


def model_config(start, degree, wavelength_family):
    """Reset every fitted start; only caller/header starts may enter ``start``."""
    return replace(
        start,
        continuum_order=int(degree),
        fit_wavelength_shift=wavelength_family == "constant",
        fit_wavelength_polynomial=wavelength_family in {"linear", "quadratic"},
        wavelength_polynomial_order=2 if wavelength_family == "quadratic" else 1,
        wavelength_shift_unit="pixel",
        initial_wavelength_shift=0.0,
        fit_segment_wavelength_shifts=False,
        fit_segment_wavelength_polynomial=False,
        wavelength_group_polynomial_orders=None,
        wavelength_group_coefficient_bounds=None,
        initialize_lsf_sigma_grid=start.fit_lsf_sigma,
        initialize_wavelength_shift_grid=True,
    )


def _bound_hits(result, config):
    hits = {k: v for k, v in result.parameter_bound_status.items() if "wavelength_" in k}
    if config.fit_wavelength_shift or config.fit_wavelength_polynomial:
        lower, upper = config.wavelength_shift_bounds
        margin = 0.01 * (upper - lower)
        for index, value in enumerate(result.wavelength_coefficients):
            if value <= lower + margin or value >= upper - margin:
                hits[f"coefficient_{index}"] = "near_bound"
    return hits


def domain_masks(reference, exclude_ranges=()):
    valid = validation_pixels(reference, exclude_ranges)
    valid &= np.isfinite(reference.spectrum.uncertainty) & (reference.spectrum.uncertainty > 0)
    trim = valid.copy()
    trim[:40] = False
    trim[-40:] = False
    transmission = reference.transmission
    domains = {"full": valid, "trim40": trim}
    for name, lo, hi in (
        ("opaque", 0.0, 0.1),
        ("deep", 0.1, 0.5),
        ("moderate", 0.5, 0.9),
        ("shallow", 0.9, 0.98),
        ("transparent", 0.98, np.inf),
        ("telluric", 0.1, 0.98),
    ):
        domains[name] = trim & (transmission >= lo) & (transmission < hi)
    return domains


def _gain(before, after):
    if before <= np.finfo(float).tiny:
        if after <= np.finfo(float).tiny:
            return 0.0
        raise ValueError("zero reference residual cannot support a relative-improvement comparison")
    return float(1 - after / before)


def full_comparison(reference, result, config, domains):
    """Reject fit failures, changed inputs/masks and regression in supported domains."""
    records = {}
    reasons = []
    for label, mask in domains.items():
        count = int(mask.sum())
        if count < MINIMUM_DOMAIN_PIXELS and label not in {"full", "trim40"}:
            records[label] = {"pixels": count, "evaluated": False}
            continue
        before, _ = _continuum_prediction_score(reference, reference, mask)
        after, _ = _continuum_prediction_score(result, reference, mask)
        gain = _gain(before["rms"], after["rms"])
        records[label] = {
            "pixels": count,
            "evaluated": True,
            "before_rms": before["rms"],
            "after_rms": after["rms"],
            "rms_improvement_fraction": gain,
        }
        permitted = MAXIMUM_FULL_REGRESSION if label == "full" else MAXIMUM_DOMAIN_REGRESSION
        if gain < -permitted:
            reasons.append(label + "_regression")
    if records["trim40"]["rms_improvement_fraction"] < MINIMUM_FULL_GAIN:
        reasons.append("trim40_improvement_below_two_percent")
    hits = _bound_hits(result, config)
    if hits:
        reasons.append("wavelength_coefficient_at_bound")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "scores": records,
        "wavelength_bound_status": hits,
    }


def cv_comparison(before, after, domains, folds):
    """Compare three guarded refits on identical original-error score vectors."""
    a = np.concatenate([item["residuals"] for item in before])
    b = np.concatenate([item["residuals"] for item in after])

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values))))

    gain = _gain(rms(a), rms(b))
    gains = [
        _gain(rms(x["residuals"]), rms(y["residuals"])) for x, y in zip(before, after, strict=True)
    ]
    reasons = []
    if gain < MINIMUM_CV_GAIN:
        reasons.append("pooled_prediction_improvement_below_five_percent")
    if sum(value > 1e-6 for value in gains) < 2:
        reasons.append("prediction_improvement_not_repeated")
    if min(gains) < -MAXIMUM_FOLD_REGRESSION:
        reasons.append("fold_regression_above_five_percent")
    medians_a = [float(np.median(np.abs(x["residuals"]))) for x in before]
    medians_b = [float(np.median(np.abs(x["residuals"]))) for x in after]
    if (
        np.mean(medians_b) > np.mean(medians_a) + 1e-12
        or sum(y <= x + 1e-12 for x, y in zip(medians_a, medians_b, strict=True)) < 2
    ):
        reasons.append("median_absolute_residual_tradeoff")
    scores = {}
    for name, mask in domains.items():
        if name in {"full", "trim40"}:
            continue
        chosen = np.concatenate([mask[fold["score_mask"]] for fold in folds])
        count = int(chosen.sum())
        if count < MINIMUM_DOMAIN_PIXELS:
            scores[name] = {"pixels": count, "evaluated": False}
            continue
        local_gain = _gain(rms(a[chosen]), rms(b[chosen]))
        scores[name] = {"pixels": count, "evaluated": True, "rms_improvement_fraction": local_gain}
        if local_gain < -MAXIMUM_DOMAIN_REGRESSION:
            reasons.append(name + "_prediction_regression")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "pooled_rms_improvement_fraction": gain,
        "fold_improvement_fractions": gains,
        "baseline_pooled_rms": rms(a),
        "candidate_pooled_rms": rms(b),
        "domain_scores": scores,
    }


def joint_model_reconsideration(reference, config, *, run_fit, starting_config):
    """Screen a fixed small grid, validate at most two and otherwise retain baseline."""
    baseline_family = family(config)
    degrees = (
        sorted({int(config.continuum_order), 1, 2})
        if RECONSIDER_CONTINUUM
        else [int(config.continuum_order)]
    )
    details = {
        "method": "bounded_joint_continuum_wavelength_reconsideration",
        "reconsider_continuum": RECONSIDER_CONTINUUM,
        "conditional_on_full_data_structure": True,
        "score_scope": "original_valid_nonexcluded_pixels",
        "accepted": False,
        "baseline_degree": config.continuum_order,
        "baseline_family": baseline_family,
        "degrees": degrees,
        "families": list(FAMILIES),
        "shortlist_limit": MAXIMUM_SHORTLIST,
        "screen": [],
        "comparisons": [],
        "starting_policy": {
            "full_data_fitted_parameters_reused": False,
            "initial_species_scales": starting_config.initial_species_scales,
            "lsf_sigma_pixels": starting_config.lsf_sigma_pixels,
            "initial_wavelength_shift": 0.0,
        },
        "thresholds": {
            "full_trim_gain": MINIMUM_FULL_GAIN,
            "cv_gain": MINIMUM_CV_GAIN,
            "maximum_full_regression": MAXIMUM_FULL_REGRESSION,
            "maximum_domain_regression": MAXIMUM_DOMAIN_REGRESSION,
            "maximum_fold_regression": MAXIMUM_FOLD_REGRESSION,
            "minimum_domain_pixels": MINIMUM_DOMAIN_PIXELS,
        },
    }
    if (
        config.wavelength_shift_unit != "pixel"
        or config.fit_segment_wavelength_shifts
        or config.fit_segment_wavelength_polynomial
        or config.wavelength_group_polynomial_orders
        or (config.fit_wavelength_polynomial and config.wavelength_polynomial_order > 2)
    ):
        return reference, config, {**details, "reason": "unsupported_wavelength_geometry"}
    # Plan against the largest possible displacement and polynomial degrees,
    # so every compared model uses the same conservative guards and score pixels.
    plan_config = model_config(config, max(degrees), "quadratic")
    folds, plan = continuum_guarded_plan(reference, plan_config, max(degrees))
    details["plan"] = plan
    if not folds:
        return reference, config, {**details, "reason": "insufficient_validation_support"}
    domains = domain_masks(reference, config.exclude_ranges)
    shortlist = []
    for degree in degrees:
        for name in FAMILIES:
            if degree == config.continuum_order and name == baseline_family:
                continue
            candidate_config = model_config(starting_config, degree, name)
            item = {"degree": degree, "family": name}
            try:
                result = run_fit(candidate_config)
                screen = full_comparison(reference, result, candidate_config, domains)
                item.update(screen)
                if screen["passed"]:
                    shortlist.append(
                        (
                            screen["scores"]["trim40"]["after_rms"],
                            degree,
                            FAMILIES.index(name),
                            result,
                            candidate_config,
                            item,
                        )
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                item.update(passed=False, reasons=["full_candidate_invalid"], error=str(exc))
            details["screen"].append(item)
    shortlist.sort(key=lambda item: item[:3])
    shortlist = shortlist[:MAXIMUM_SHORTLIST]
    if not shortlist:
        return reference, config, {**details, "reason": "no_full_candidate_passed"}

    def evaluate(candidate_config):
        output = []
        for fold in folds:
            exclusions = tuple(config.exclude_ranges or ()) + tuple(fold["exclude_ranges_micron"])
            fold_config = replace(
                candidate_config, exclude_ranges=exclusions, estimate_uncertainties=False
            )
            result = run_fit(fold_config)
            score, residuals = _continuum_prediction_score(
                result, reference, fold["score_mask"], fold["guard_mask"]
            )
            hits = _bound_hits(result, fold_config)
            if hits:
                raise ValueError("guarded wavelength coefficient at bound")
            output.append({"score": score, "residuals": residuals})
        return output

    try:
        baseline_start = model_config(starting_config, config.continuum_order, baseline_family)
        baseline_scores = evaluate(baseline_start)
        details["baseline_fold_scores"] = [item["score"] for item in baseline_scores]
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return (
            reference,
            config,
            {**details, "reason": "baseline_validation_invalid", "error": str(exc)},
        )
    accepted = []
    for _, degree, family_index, result, candidate_config, _screen_record in shortlist:
        item = {"degree": degree, "family": FAMILIES[family_index]}
        try:
            scores = evaluate(candidate_config)
            comparison = cv_comparison(baseline_scores, scores, domains, folds)
            item.update(comparison, fold_scores=[entry["score"] for entry in scores])
            if comparison["accepted"]:
                accepted.append(
                    (
                        comparison["candidate_pooled_rms"],
                        degree,
                        family_index,
                        result,
                        candidate_config,
                        item,
                    )
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            item.update(accepted=False, reasons=["candidate_validation_invalid"], error=str(exc))
        details["comparisons"].append(item)
    if not accepted:
        return reference, config, {**details, "reason": "no_predictively_supported_candidate"}
    accepted.sort(key=lambda item: item[:3])
    # Prefer the simpler model if validation RMS differs by <=1% from best.
    equivalent = [item for item in accepted if item[0] <= accepted[0][0] * 1.01]
    winner = min(equivalent, key=lambda item: (item[1] + item[2], item[1], item[2], item[0]))
    _, degree, family_index, result, candidate_config, comparison = winner
    details.update(
        accepted=True,
        reason="joint_guarded_prediction_supported",
        selected_degree=degree,
        selected_family=FAMILIES[family_index],
        selected_comparison=comparison,
    )
    return result, candidate_config, details
