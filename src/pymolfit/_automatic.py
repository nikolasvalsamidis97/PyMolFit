"""Conservative data-dependent policies for the high-level correction API."""

from dataclasses import replace

import numpy as np

from .fit import _poly_design, _robust_cost, _solve_profiled_continuum
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
