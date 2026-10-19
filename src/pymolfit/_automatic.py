"""Conservative data-dependent policies for the high-level correction API."""

import hashlib
from dataclasses import replace

import numpy as np

from .fit import _poly_design, _robust_cost, _solve_profiled_continuum
from .model import lsf_kernel_half_width_pixels
from .regions import _expected_aer_absorption_score, _normalize_ranges


def sampling_indices(spectrum):
    """Keep index mappings, physical groups, and wavelength gaps intact."""
    wave = spectrum.to_unit("micron").wavelength
    labels = spectrum.group_id
    labels = np.zeros(wave.size, dtype=int) if labels is None else labels
    sections = []
    for label in np.unique(labels):
        indices = np.flatnonzero((labels == label) & np.isfinite(wave))
        indices = indices[np.argsort(wave[indices], kind="stable")]
        steps = np.diff(wave[indices])
        positive = steps[steps > 0]
        if not positive.size:
            continue
        cuts = np.flatnonzero(steps > 10 * np.median(positive)) + 1
        sections.extend(part for part in np.split(indices, cuts) if part.size >= 3)
    return sections


def automatic_regions(spectrum, line_list, sigma_pixels, exclude_ranges):
    """Select distributed catalogue windows without imposing a global line count."""
    wave = spectrum.to_unit("micron").wavelength
    centers = line_list.wavelength
    scores = (
        _expected_aer_absorption_score(line_list.strength, line_list.species)
        if line_list.has_hitran_parameters
        else np.asarray(line_list.strength, dtype=float)
    )
    eligible = np.isfinite(scores) & (scores > 0) & np.isfinite(centers)
    for lower, upper in exclude_ranges or ():
        eligible &= ~((centers >= lower) & (centers <= upper))
    half_width = int(np.clip(np.ceil(4 * sigma_pixels + 12), 24, 64))
    ranges, records = [], []
    for indices in sampling_indices(spectrum):
        x = wave[indices]
        covered = eligible & (centers >= x[0]) & (centers <= x[-1])
        candidates = np.flatnonzero(covered)
        if not candidates.size:
            continue
        candidates = candidates[scores[candidates] >= 1.0e-3 * np.max(scores[candidates])]
        bin_count = max(1, int(np.ceil(indices.size / (8 * half_width))))
        edges = np.linspace(x[0], x[-1], bin_count + 1)
        selected = []
        for bin_index in range(bin_count):
            local = candidates[
                (centers[candidates] >= edges[bin_index])
                & (centers[candidates] <= edges[bin_index + 1])
            ]
            # Retain different molecules, not just the most abundant species.
            for species in np.unique(line_list.species[local]):
                members = local[line_list.species[local] == species]
                selected.append(int(members[np.argmax(scores[members])]))
        for index in sorted(set(selected)):
            pixel = int(np.clip(np.searchsorted(x, centers[index]), 0, x.size - 1))
            start, stop = max(0, pixel - half_width), min(x.size - 1, pixel + half_width)
            if np.count_nonzero(spectrum.valid[indices[start : stop + 1]]) >= 8:
                ranges.append((float(x[start]), float(x[stop])))
        records.append(
            {
                "lower_micron": float(x[0]),
                "upper_micron": float(x[-1]),
                "selected_lines": len(set(selected)),
                "spatial_bins": bin_count,
            }
        )
    return _normalize_ranges(tuple(ranges)), {
        "source": "distributed_catalogue_windows",
        "half_width_pixels": half_width,
        "sections": records,
        "selected_line_count": sum(r["selected_lines"] for r in records),
    }


def ensure_automatic_region_coverage(segments, fit_ranges, exclude_ranges, *, continuum_order):
    """Broaden sparse automatic windows within the affected processing chunks.

    This checks pixel coverage, not transmission or continuum identifiability.
    Ranges have the same global wavelength scope as the public fitting ranges;
    overlapping physical groups may therefore gain coverage together.
    """
    minimum_fraction = 0.25
    minimum_pixels = max(32, 8 * (continuum_order + 1))
    ranges = list(fit_ranges)
    records, masks = [], []
    for segment in segments:
        wave = segment.to_unit("micron").wavelength
        available = segment.valid.copy()
        for lower, upper in exclude_ranges or ():
            available &= ~((wave >= lower) & (wave <= upper))
        selected = np.zeros(wave.size, dtype=bool)
        for lower, upper in fit_ranges:
            selected |= (wave >= lower) & (wave <= upper)
        count = int(np.count_nonzero(available))
        before = int(np.count_nonzero(available & selected))
        required = min(count, max(minimum_pixels, int(np.ceil(minimum_fraction * count))))
        expand = before < required
        if expand:
            ranges.extend(_available_pixel_ranges(segment, available))
        finite = wave[np.isfinite(wave)]
        records.append(
            {
                "lower_micron": float(np.min(finite)) if finite.size else None,
                "upper_micron": float(np.max(finite)) if finite.size else None,
                "available_pixels": count,
                "selected_before": before,
                "required_pixels": required,
                "expanded": expand,
                "available_below_minimum": count < minimum_pixels,
            }
        )
        masks.append((wave, available))
    expanded_ranges = _normalize_ranges(tuple(ranges))
    # A wavelength interval can also cover samples in an overlapping group.
    # Report the final union rather than only each chunk's requested additions.
    for record, (wave, available) in zip(records, masks, strict=True):
        selected = np.zeros(wave.size, dtype=bool)
        for lower, upper in expanded_ranges:
            selected |= (wave >= lower) & (wave <= upper)
        record["selected_after"] = int(np.count_nonzero(available & selected))
    return expanded_ranges, {
        "method": "expand_undercovered_processing_segments",
        "minimum_fraction": minimum_fraction,
        "minimum_pixels": minimum_pixels,
        "range_scope": "global_wavelength_union",
        "expanded_segment_count": sum(record["expanded"] for record in records),
        "segments": records,
    }


def _available_pixel_ranges(spectrum, available):
    """Include eligible samples without bridging exclusions or sampling gaps."""
    wave = spectrum.to_unit("micron").wavelength
    labels = spectrum.group_id
    labels = np.zeros(wave.size, dtype=int) if labels is None else labels
    ranges = []
    for label in np.unique(labels):
        indices = np.flatnonzero((labels == label) & np.isfinite(wave))
        indices = indices[np.argsort(wave[indices], kind="stable")]
        if not indices.size:
            continue
        steps = np.diff(wave[indices])
        positive = steps[steps > 0]
        gap = 10 * np.median(positive) if positive.size else np.inf
        chosen = np.flatnonzero(available[indices])
        cuts = np.flatnonzero((np.diff(chosen) > 1) | (np.diff(wave[indices[chosen]]) > gap)) + 1
        for run in np.split(chosen, cuts):
            if not run.size:
                continue
            lower, upper = float(wave[indices[run[0]]]), float(wave[indices[run[-1]]])
            if lower == upper:
                lower, upper = (
                    float(np.nextafter(lower, -np.inf)),
                    float(np.nextafter(upper, np.inf)),
                )
            ranges.append((lower, upper))
    return ranges


def estimate_missing_uncertainty(spectrum, line_ranges=()):
    if spectrum.uncertainty is not None:
        return spectrum, {"source": "supplied", "rescaled": False}
    wave = spectrum.to_unit("micron").wavelength
    clean = spectrum.valid.copy()
    for lower, upper in line_ranges or ():
        clean &= ~((wave >= lower) & (wave <= upper))
    sigma = np.full(wave.size, np.nan)
    records = []
    for section in sampling_indices(spectrum):
        for indices in np.array_split(section, max(1, int(np.ceil(section.size / 512)))):
            flux = spectrum.flux[indices]
            valid = spectrum.valid[indices]
            triples = valid[:-2] & valid[1:-1] & valid[2:]
            clean_triples = clean[indices[:-2]] & clean[indices[1:-1]] & clean[indices[2:]]
            source = "line_free_second_differences"
            if np.count_nonzero(triples & clean_triples) >= 24:
                triples &= clean_triples
            else:
                source = "robust_second_differences"
            x = wave[indices]
            span = x[2:] - x[:-2]
            fraction = np.divide(
                x[1:-1] - x[:-2], span, out=np.full(span.size, 0.5), where=span > 0
            )
            triples &= (span > 0) & (fraction > 0.1) & (fraction < 0.9)
            differences = (
                (flux[1:-1] - (1 - fraction) * flux[:-2] - fraction * flux[2:])
                / np.sqrt(1 + (1 - fraction) ** 2 + fraction**2)
            )[triples]
            scatter = (
                (1.4826 * np.median(np.abs(differences - np.median(differences))))
                if differences.size >= 12
                else np.nan
            )
            first = np.diff(flux)[1:][triples]
            first_scatter = (
                (1.4826 * np.median(np.abs(first - np.median(first))) / np.sqrt(2))
                if first.size >= 12
                else np.nan
            )
            level = float(np.nanmedian(np.abs(flux[valid]))) if np.any(valid) else 1.0
            if (
                not np.isfinite(scatter)
                or scatter <= max(level, 1.0e-30) * 1.0e-10
                or scatter < 0.1 * first_scatter
            ):
                scatter, source = 0.01 * max(level, 1.0e-30), "fractional_fallback"
            # Smooth, noiseless inputs must not acquire effectively infinite weight.
            floor = 0.0001 * max(level, 1.0e-30)
            floor_applied = bool(scatter < floor)
            scatter = max(scatter, floor)
            sigma[indices] = scatter
            records.append(
                {
                    "source": source,
                    "sigma": float(scatter),
                    "sample_count": int(differences.size),
                    "floor_applied": floor_applied,
                }
            )
    missing = ~np.isfinite(sigma)
    level = np.nanmedian(np.abs(spectrum.flux[spectrum.valid])) if np.any(spectrum.valid) else 1.0
    sigma[missing] = 0.01 * max(float(level), 1.0e-30)
    return replace(spectrum, uncertainty=sigma, mask=spectrum.valid), {
        "source": "estimated",
        "blocks": records,
        "fractional_noise_floor": 0.0001,
        "warning": "Estimated errors are approximate and exclude model systematics.",
    }


def choose_continuum_order(samples, *, loss, f_scale, initial_order=2):
    """Compare low-order continua on held-out blocks at a fixed transmission."""
    totals = {
        degree: {"validation_cost": 0.0, "score": 0.0, "segments": 0} for degree in (1, 2, 3, 4)
    }
    for wave, flux, sigma, transmission, mask in samples:
        usable = mask & (transmission > 0.1) & np.isfinite(sigma) & (sigma > 0)
        if np.count_nonzero(usable) < 60:
            continue
        # Adjacent pixels are correlated; hold out blocks rather than alternate pixels.
        validation = usable & ((np.arange(wave.size) // max(4, wave.size // 25)) % 5 == 2)
        training = usable & ~validation
        if np.count_nonzero(validation) < 10 or np.count_nonzero(training) < 30:
            continue
        for degree in totals:
            _, design = _poly_design(wave, degree)
            coefficients = _solve_profiled_continuum(
                design[training],
                transmission[training],
                flux[training],
                sigma[training],
                loss=loss,
                f_scale=f_scale,
            )
            continuum = design @ coefficients
            if not np.all(np.isfinite(continuum)) or np.any(continuum <= 0):
                totals[degree]["score"] = np.inf
                totals[degree]["validation_cost"] = np.inf
                continue
            residual = (
                flux[validation] - continuum[validation] * transmission[validation]
            ) / sigma[validation]
            cost = 2 * _robust_cost(residual, loss=loss, f_scale=f_scale)
            totals[degree]["validation_cost"] += cost
            totals[degree]["score"] += cost + (degree + 1) * np.log(np.count_nonzero(validation))
            totals[degree]["segments"] += 1
    selected = initial_order
    base = totals[initial_order]
    if base["segments"]:
        for degree in (1, 3, 4):
            candidate, current = totals[degree], totals[selected]
            if candidate["segments"] != base["segments"] or not np.isfinite(candidate["score"]):
                continue
            supported = candidate["validation_cost"] <= current["validation_cost"] * (
                1.01 if degree < selected else 0.98
            )
            if supported and current["score"] - candidate["score"] >= (
                2 if degree < selected else 6
            ):
                selected = degree
    for record in totals.values():
        for key in ("score", "validation_cost"):
            record[key] = float(record[key]) if np.isfinite(record[key]) else None
    return selected, {
        "requested": "auto",
        "initial_order": initial_order,
        "selected_order": selected,
        "method": "blocked_validation_fixed_transmission",
        "candidates": totals,
    }


def automatic_robust_scale(result):
    mask = result.spectrum.valid & (result.transmission > 0.1)
    if result.fit_mask is not None:
        mask &= result.fit_mask
    sigma = result.spectrum.uncertainty
    if sigma is None or np.count_nonzero(mask) < 30:
        return 1.0
    residual = (result.spectrum.flux[mask] - result.model_flux[mask]) / sigma[mask]
    if getattr(result, "fit_weights", None) is not None:
        residual *= np.sqrt(result.fit_weights[mask])
    scatter = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    return float(np.clip(scatter, 1.0, 2.0)) if np.isfinite(scatter) else 1.0


def promising_continuum_degree(resolution, selected_order):
    """Screen one rejected higher degree; acceptance requires new guarded fits."""
    candidates = resolution.get("candidates", {})
    baseline = candidates.get(selected_order, candidates.get(str(selected_order), {}))
    cost = baseline.get("validation_cost")
    if cost is None or not np.isfinite(cost) or cost <= 0:
        return None
    eligible = []
    for degree, entry in candidates.items():
        value = entry.get("validation_cost")
        if (
            int(degree) > selected_order
            and entry.get("segments") == baseline.get("segments")
            and entry.get("score") is not None
            and value is not None
            and np.isfinite(value)
            and value <= 0.95 * cost
        ):
            eligible.append((int(degree), float(value)))
    if not eligible:
        return None
    best = min(value for _, value in eligible)
    return min(degree for degree, value in eligible if value <= 1.01 * best)


def _separated_continuum_centers(feasible, size):
    """Choose feasible centers nearest fixed targets, never using residuals."""
    centers = np.asarray(feasible, dtype=int)
    targets = np.rint(np.asarray((0.2, 0.5, 0.8)) * (size - 1)).astype(int)
    separation = max(16, int(np.ceil(0.15 * size)))
    best = None
    for middle in centers:
        left = centers[centers <= middle - separation]
        right = centers[centers >= middle + separation]
        if not left.size or not right.size:
            continue
        trial = (
            int(left[np.argmin(np.abs(left - targets[0]))]),
            int(middle),
            int(right[np.argmin(np.abs(right - targets[2]))]),
        )
        key = (int(np.sum((np.asarray(trial) - targets) ** 2)), *trial)
        if best is None or key < best:
            best = key
    return None if best is None else best[1:]


def continuum_guarded_plan(reference, config, degree):
    """Build three common geometry-only folds; fail closed on weak support."""
    spectrum = reference.spectrum.to_unit("micron")
    wave = spectrum.wavelength
    if spectrum.uncertainty is None or reference.fit_mask is None:
        return [], {"reason": "missing_errors_or_fit_mask"}
    if (
        config.lsf_variable_width
        or config.fit_lsf_wavelength_exponent
        or config.lsf_box_width_pixels > 0
        or config.fit_lsf_box_width
        or config.lsf_lorentz_fwhm_pixels > 0
        or config.fit_lsf_lorentz_fwhm
        or config.fit_segment_wavelength_shifts
        or config.fit_segment_wavelength_polynomial
        or config.wavelength_group_polynomial_orders
    ):
        return [], {"reason": "unsupported_kernel_or_wavelength_geometry"}
    if spectrum.group_id is not None and len(np.unique(spectrum.group_id)) > 1:
        return [], {"reason": "multiple_physical_groups"}
    moves = config.fit_wavelength_shift or config.fit_wavelength_polynomial
    if moves and config.wavelength_shift_unit != "pixel":
        return [], {"reason": "nonpixel_wavelength_bounds"}
    displacement = (
        max(abs(value) for value in config.wavelength_shift_bounds)
        * (config.wavelength_polynomial_order + 1 if config.fit_wavelength_polynomial else 1)
        if moves
        else 0.0
    )
    maximum_sigma = config.lsf_sigma_bounds[1] if config.fit_lsf_sigma else config.lsf_sigma_pixels
    kernel_support = lsf_kernel_half_width_pixels(
        gaussian_sigma_pixels=maximum_sigma,
        kernel_width_fwhm=config.lsf_kernel_width_fwhm,
    )
    guard = max(16, int(np.ceil(kernel_support + 2 * displacement)))
    boundaries = reference.provenance.get("segmentation", {}).get(
        "boundaries_micron", [(float(wave.min()), float(wave.max()))]
    )
    valid = spectrum.valid & np.isfinite(spectrum.uncertainty) & (spectrum.uncertainty > 0)
    eligible = valid & np.asarray(reference.fit_mask, bool)
    moderate = eligible & (reference.transmission > 0.1)
    minimum_training, side = max(48, 12 * (degree + 1)), max(8, 2 * (degree + 1))
    details = {
        "reason": "ready",
        "conditional_on_full_data_structure": True,
        "core_pixels": 16,
        "guard_pixels_per_side": guard,
        "maximum_kernel_half_support_pixels": kernel_support,
        "maximum_wavelength_displacement_pixels": displacement,
        "target_center_fractions": [0.2, 0.5, 0.8],
        "center_policy": "nearest_feasible_separated_centers_without_residuals",
        "minimum_center_separation_segment_fraction": 0.15,
        "minimum_original_valid_core_pixels": 8,
        "minimum_training_pixels_per_segment": minimum_training,
        "minimum_side_moderate_pixels": side,
        "folds": [],
    }
    chunks = []
    for lo, hi in boundaries:
        indices = np.flatnonzero((wave >= lo) & (wave <= hi))
        if indices.size < 16 + 2 * guard + minimum_training:
            return [], {**details, "reason": "segment_too_short_for_guards_and_training"}
        spacing = np.diff(wave[indices])
        if (
            np.any(np.diff(indices) != 1)
            or np.any(spacing <= 0)
            or np.any(spacing > 5 * np.median(spacing))
        ):
            return [], {**details, "reason": "noncontiguous_processing_segment"}
        feasible = {}
        for center in range(8 + guard, indices.size - 8 - guard + 1):
            left, right = center - 8, center + 8
            lower, upper = left - guard, right + guard
            if np.count_nonzero(valid[indices[left:right]]) < 8:
                continue
            training = eligible[indices].copy()
            training[lower:upper] = False
            moderate_training = moderate[indices] & training
            if (
                np.count_nonzero(training) >= minimum_training
                and np.count_nonzero(moderate_training) >= max(60, minimum_training)
                and np.count_nonzero(moderate_training[:lower]) >= side
                and np.count_nonzero(moderate_training[upper:]) >= side
            ):
                feasible[center] = int(training.sum())
        centers = _separated_continuum_centers(list(feasible), indices.size)
        if centers is None:
            return [], {
                **details,
                "reason": "insufficient_separated_guarded_training_support",
                "unsupported_segment_bounds_micron": [lo, hi],
                "feasible_center_count": len(feasible),
            }
        chunks.append((indices, centers, feasible))
    folds = []
    for fold_index in range(3):
        score = np.zeros(wave.size, bool)
        withheld = np.zeros(wave.size, bool)
        blocks, ranges = [], []
        for indices, centers, feasible in chunks:
            center = centers[fold_index]
            left, right = center - 8, center + 8
            lower, upper = left - guard, right + guard
            guard_indices, core_indices = indices[lower:upper], indices[left:right]
            score[core_indices] = valid[core_indices]
            withheld[guard_indices] = True
            ranges.append((float(wave[guard_indices[0]]), float(wave[guard_indices[-1]])))
            blocks.append(
                {
                    "core_indices": [int(core_indices[0]), int(core_indices[-1])],
                    "guard_indices": [int(guard_indices[0]), int(guard_indices[-1])],
                    "training_pixels": feasible[center],
                    "original_valid_core_pixels": int(valid[core_indices].sum()),
                    "already_invalid_guard_pixels": int((~valid[guard_indices]).sum()),
                    "center_fraction": float(center / (indices.size - 1)),
                }
            )
        record = {
            "fold_index": fold_index,
            "blocks": blocks,
            "exclude_ranges_micron": ranges,
            "score_pixels": int(score.sum()),
            "score_mask_sha256": hashlib.sha256(score.astype(np.uint8).tobytes()).hexdigest(),
            "guard_mask_sha256": hashlib.sha256(withheld.astype(np.uint8).tobytes()).hexdigest(),
        }
        details["folds"].append(record)
        folds.append({**record, "score_mask": score, "guard_mask": withheld})
    return folds, details


def _continuum_prediction_score(result, reference, mask, guard=None):
    original, prediction = reference.spectrum, result.spectrum
    if not result.success or any(
        not np.array_equal(getattr(original, key), getattr(prediction, key), equal_nan=True)
        for key in ("wavelength", "flux", "uncertainty", "valid")
    ):
        raise ValueError("fold did not converge or original arrays changed")
    valid = original.valid
    if (
        not np.all(np.isfinite(result.model_flux[valid]))
        or not np.all(np.isfinite(result.continuum[valid]) & (result.continuum[valid] > 0))
        or not np.all(np.isfinite(result.transmission[valid]))
    ):
        raise ValueError("invalid fold prediction on original-valid samples")
    expected_mask = reference.fit_mask if guard is None else reference.fit_mask & ~guard
    if result.fit_mask is None or not np.array_equal(result.fit_mask, expected_mask):
        raise ValueError("training fit mask differs from the exact guarded reference mask")
    values = (original.flux[mask] - result.model_flux[mask]) / original.uncertainty[mask]
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("invalid or empty fixed score domain")
    return {
        "pixels": int(values.size),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median_absolute": float(np.median(np.abs(values))),
        "nfev": int(result.nfev),
        "guarded_fit_pixels": 0
        if guard is None
        else int(np.count_nonzero(result.fit_mask & guard)),
    }, values


def guarded_continuum_refinement(reference, config, resolution, *, run_fit, starting_config):
    """Conditionally test one degree using independent, fully refitted folds."""
    degree = promising_continuum_degree(resolution, config.continuum_order)
    details = {
        "method": "three_guarded_refitted_continuum_folds",
        "conditional_on_full_data_structure": True,
        "accepted": False,
        "baseline_degree": config.continuum_order,
        "candidate_degree": degree,
    }
    if degree is None:
        return reference, config, {**details, "reason": "no_promising_rejected_degree"}
    folds, plan = continuum_guarded_plan(reference, config, degree)
    details["plan"] = plan
    if not folds:
        return reference, config, {**details, "reason": "insufficient_validation_support"}
    details["starting_policy"] = {
        "initial_species_scales": starting_config.initial_species_scales,
        "lsf_sigma_pixels": starting_config.lsf_sigma_pixels,
        "initial_wavelength_shift": starting_config.initial_wavelength_shift,
        "conditional_fixed_robust_scale": starting_config.f_scale,
        "full_data_fitted_parameters_reused": False,
    }
    comparisons, original_values, candidate_values = [], [], []
    try:
        for fold in folds:
            exclusions = tuple(config.exclude_ranges or ()) + tuple(fold["exclude_ranges_micron"])
            values = []
            record = {
                "fold_index": fold["fold_index"],
                "score_mask_sha256": fold["score_mask_sha256"],
            }
            for label, order in (("baseline", config.continuum_order), ("candidate", degree)):
                trial_config = replace(
                    starting_config,
                    continuum_order=order,
                    exclude_ranges=exclusions,
                    estimate_uncertainties=False,
                )
                result = run_fit(trial_config)
                score, residual = _continuum_prediction_score(
                    result, reference, fold["score_mask"], fold["guard_mask"]
                )
                record[label] = score
                values.append(residual)
            original_values.append(values[0])
            candidate_values.append(values[1])
            comparisons.append(record)
        details["folds"] = comparisons
        a, b = np.concatenate(original_values), np.concatenate(candidate_values)
        rms_a, rms_b = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
        if rms_a <= 0:
            return reference, config, {**details, "reason": "zero_reference_prediction_error"}
        gain = 1 - rms_b / rms_a
        details.update(
            pooled_rms_before=rms_a,
            pooled_rms_after=rms_b,
            pooled_rms_improvement_percent=100 * gain,
            pooled_median_absolute_before=float(np.median(np.abs(a))),
            pooled_median_absolute_after=float(np.median(np.abs(b))),
        )
        if not (
            gain >= 0.05
            and all(r["candidate"]["rms"] < r["baseline"]["rms"] for r in comparisons)
            and np.median(np.abs(b)) <= np.median(np.abs(a))
        ):
            return reference, config, {**details, "reason": "guarded_prediction_threshold_not_met"}
        trial_config = replace(config, continuum_order=degree)
        trial = run_fit(trial_config)
        valid = reference.spectrum.valid.copy()
        full_scores = {}
        for label, mask in (("all_original_valid", valid), ("trim40", valid.copy())):
            if label == "trim40":
                mask[:40] = False
                mask[-40:] = False
            before, _ = _continuum_prediction_score(reference, reference, mask)
            after, _ = _continuum_prediction_score(trial, reference, mask)
            full_scores[label] = {"baseline": before, "candidate": after}
            if after["rms"] > before["rms"] * (1 + 1e-10) or after["median_absolute"] > before[
                "median_absolute"
            ] * (1 + 1e-10):
                return (
                    reference,
                    config,
                    {**details, "full_scores": full_scores, "reason": "full_data_score_regression"},
                )
        return (
            trial,
            trial_config,
            {
                **details,
                "full_scores": full_scores,
                "accepted": True,
                "reason": "guarded_prediction_and_full_scores_supported",
            },
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return (
            reference,
            config,
            {**details, "folds": comparisons, "reason": "validation_error", "error": str(exc)},
        )


def mask_uncertain_correction(result, *, relative_limit):
    """Mask unreliable divisions without changing the observation or transmission."""
    sigma = result.corrected.uncertainty
    if sigma is None:
        return result, {"source": "unavailable_uncertainty", "additional_masked_pixels": 0}
    input_sigma = result.spectrum.uncertainty
    limit = relative_limit * np.abs(result.continuum)
    # Do not reject noisy input solely for being noisy before any correction.
    if input_sigma is not None:
        limit = np.maximum(limit, 2 * input_sigma)
    unreliable = (result.transmission < 0.98) & (sigma > limit)
    additional = unreliable & result.corrected.valid
    keep = result.corrected.valid & ~unreliable
    corrected = replace(
        result.corrected,
        mask=keep,
        flux=np.where(keep, result.corrected.flux, np.nan),
        uncertainty=np.where(keep, sigma, np.nan),
    )
    return replace(result, corrected=corrected), {
        "source": "propagated_uncertainty",
        "minimum_transmission": 0.01,
        "relative_continuum_uncertainty_limit": relative_limit,
        "minimum_allowed_noise_amplification": 2.0,
        "additional_masked_pixels": int(np.count_nonzero(additional)),
    }
