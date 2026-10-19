"""Conditional numerical recovery for an unresponsive fitted Gaussian width."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult


def retry_unresponsive_lsf(
    fit: OptimizeResult,
    *,
    initial: np.ndarray,
    parameter_index: int | None,
    lower: np.ndarray,
    upper: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    optimize: Callable[[np.ndarray], OptimizeResult],
    valid_candidate: Callable[[np.ndarray], bool],
) -> tuple[OptimizeResult, dict[str, object] | None]:
    """Retain the original fit unless a valid retry has a lower same-data loss.

    The small-width screen is only an inexpensive diagnostic gate, never a
    constraint on the answer. An actual residual perturbation must also show a
    weak local derivative and a responsive wider allowed width before retrying.
    """
    if parameter_index is None:
        return fit, None
    index = parameter_index
    width = float(fit.x[index])
    metadata: dict[str, object] = {
        "triggered": False,
        "initial_width_pixels": float(initial[index]),
        "original_fitted_width_pixels": width,
        "diagnostic_width_screen_pixels": 0.25,
        "local_to_wide_sensitivity_threshold": 1.0e-3,
        "attempts": [],
        "selected_attempt": 0,
        "total_optimizer_nfev": int(fit.nfev),
        "optimizer_nfev_complete": True,
    }
    if not np.isfinite(width) or width > 0.25:
        metadata["reason"] = "fitted_width_outside_narrow_diagnostic_screen"
        return fit, metadata
    lo = float(np.nextafter(lower[index], upper[index]))
    hi = float(np.nextafter(upper[index], lower[index]))
    starts = np.unique(np.clip(np.array([0.5, 1.0, 2.0]), lo, hi))
    starts = starts[starts > width + 1.0e-6]
    if not starts.size:
        metadata["reason"] = "no_wider_allowed_start"
        return fit, metadata
    local_width = min(width + 1.0e-4, hi)
    if local_width <= width:
        metadata["reason"] = "no_forward_width_perturbation_within_bounds"
        return fit, metadata
    try:
        base = np.asarray(residual(fit.x), dtype=float)
        local_parameters = np.asarray(fit.x, dtype=float).copy()
        local_parameters[index] = local_width
        local = np.asarray(residual(local_parameters), dtype=float)
        probe_parameters = np.asarray(fit.x, dtype=float).copy()
        probe_parameters[index] = float(starts[0])
        probe = np.asarray(residual(probe_parameters), dtype=float)
        if not all(np.all(np.isfinite(values)) for values in (base, local, probe)):
            raise ValueError("nonfinite diagnostic residual")
        local_sensitivity = float(np.sqrt(np.mean((local - base) ** 2))) / (local_width - width)
        wide_sensitivity = float(np.sqrt(np.mean((probe - base) ** 2))) / (float(starts[0]) - width)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        metadata["reason"] = "sensitivity_diagnostic_failed"
        metadata["diagnostic_error"] = str(exc)
        return fit, metadata
    metadata.update(
        local_probe_width_pixels=local_width,
        wide_probe_width_pixels=float(starts[0]),
        local_residual_sensitivity=local_sensitivity,
        wide_residual_sensitivity=wide_sensitivity,
    )
    if wide_sensitivity <= 1.0e-6 or local_sensitivity >= 1.0e-3 * wide_sensitivity:
        metadata["reason"] = "no_weak_local_width_sensitivity"
        return fit, metadata

    metadata["triggered"] = True
    metadata["reason"] = "weak_local_width_sensitivity_with_responsive_wider_probe"
    original_cost = float(fit.cost)
    best_cost = original_cost if np.isfinite(original_cost) else np.inf
    best = fit
    total_nfev = int(fit.nfev)
    attempts: list[dict[str, object]] = [
        {
            "start_width_pixels": float(initial[index]),
            "final_width_pixels": width,
            "cost": original_cost,
            "success": bool(fit.success),
            "valid": bool(valid_candidate(fit.x)),
            "nfev": int(fit.nfev),
            "original": True,
        }
    ]
    for start in starts:
        parameters = np.asarray(initial, dtype=float).copy()
        parameters[index] = float(start)
        attempt: dict[str, object] = {"start_width_pixels": float(start), "original": False}
        try:
            candidate = optimize(parameters)
            nfev = int(candidate.nfev)
            total_nfev += nfev
            valid = bool(
                candidate.success
                and np.isfinite(candidate.cost)
                and np.all(np.isfinite(candidate.x))
                and np.all(np.isfinite(candidate.fun))
                and valid_candidate(candidate.x)
            )
            attempt.update(
                final_width_pixels=float(candidate.x[index]),
                cost=float(candidate.cost),
                success=bool(candidate.success),
                valid=valid,
                nfev=nfev,
            )
            # Require more than roundoff improvement, preserving the original
            # candidate on ties and preventing numerical noise from replacing it.
            improvement_floor = (
                1.0e-10 * max(1.0, abs(best_cost)) if np.isfinite(best_cost) else 0.0
            )
            if valid and float(candidate.cost) < best_cost - improvement_floor:
                best = candidate
                best_cost = float(candidate.cost)
                metadata["selected_attempt"] = len(attempts)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            # SciPy does not expose work done before an exception. Do not
            # silently label that unreported work as zero evaluations.
            attempt.update(success=False, valid=False, error=str(exc), nfev=None)
            metadata["optimizer_nfev_complete"] = False
        attempts.append(attempt)
    metadata["attempts"] = attempts
    metadata["total_optimizer_nfev"] = total_nfev
    metadata["selected_cost"] = float(best.cost)
    # Count all optimizer work, retaining the chosen solution's Jacobian/status.
    best.nfev = total_nfev
    return best, metadata
