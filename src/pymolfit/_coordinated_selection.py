"""Staged model selection with guards against the untouched baseline.

The fitting-window safeguard limits regressions on the original fitting pixels
when coordinating continuum and wavelength choices.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._automatic import _continuum_prediction_score
from ._coordinated_rescue import _fixed_full_comparison, apply_rescued_pair
from ._joint_selection import joint_model_reconsideration

MAXIMUM_NATIVE_FIT_WINDOW_REGRESSION = 0.02


def native_fit_window_guard(native, candidate):
    """Use all original-native fitting pixels and original errors, without trimming."""
    details = {
        "passed": False,
        "maximum_rms_regression_fraction": MAXIMUM_NATIVE_FIT_WINDOW_REGRESSION,
        "mask_policy": "original_native_valid_and_fit_mask_no_trimming",
    }
    try:
        if native.fit_mask is None:
            raise ValueError("original native fit mask is unavailable")
        mask = np.asarray(native.spectrum.valid, bool) & np.asarray(native.fit_mask, bool)
        before, _ = _continuum_prediction_score(native, native, mask)
        after, _ = _continuum_prediction_score(candidate, native, mask)
        if before["rms"] <= np.finfo(float).tiny:
            gain = 0.0 if after["rms"] <= np.finfo(float).tiny else None
        else:
            gain = float(1 - after["rms"] / before["rms"])
        passed = gain is not None and gain >= -MAXIMUM_NATIVE_FIT_WINDOW_REGRESSION - 1e-12
        details.update(
            passed=passed,
            baseline=before,
            candidate=after,
            rms_improvement_fraction=gain,
            reason="native_fit_window_guard_passed"
            if passed
            else "native_fit_window_rms_regression",
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        details.update(reason="native_fit_window_guard_invalid", error=str(exc))
    return details


def configured_selection_start(
    config,
    *,
    initial_species_scales,
    lsf_sigma_pixels,
    resolved_lorentz_fwhm_pixels,
    automatic_lorentz,
):
    """Reset all fitted starts from caller/header defaults, retaining structure.

    An automatically selected Lorentzian uses the same neutral half-Gaussian-
    FWHM rule as its initial pilot, evaluated from the fresh Gaussian start.
    Neither its fitted pilot width nor the fitted power-law exponent is reused.
    """
    lorentz = float(resolved_lorentz_fwhm_pixels)
    if automatic_lorentz:
        if config.fit_lsf_lorentz_fwhm:
            lower, upper = config.lsf_lorentz_fwhm_bounds
            margin = 0.01 * (upper - lower)
            lorentz = float(
                np.clip(
                    0.5 * 2.354820045 * max(float(lsf_sigma_pixels), 0.2),
                    lower + margin,
                    upper - margin,
                )
            )
        else:
            lorentz = 0.0
    exponent = 1.0 if config.lsf_variable_width else 0.0
    if config.fit_lsf_wavelength_exponent:
        exponent = float(np.clip(1.0, *config.lsf_wavelength_exponent_bounds))
    return replace(
        config,
        initial_species_scales=initial_species_scales,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_lorentz_fwhm_pixels=lorentz,
        lsf_wavelength_exponent=exponent,
        initial_wavelength_shift=0.0,
        initialize_lsf_sigma_grid=config.fit_lsf_sigma,
        initialize_wavelength_shift_grid=True,
    )


def _fresh_start(current_config, starting_config):
    """Keep current structural choices while resetting fitted-start quantities."""
    return replace(
        current_config,
        initial_species_scales=starting_config.initial_species_scales,
        lsf_sigma_pixels=starting_config.lsf_sigma_pixels,
        lsf_lorentz_fwhm_pixels=starting_config.lsf_lorentz_fwhm_pixels,
        lsf_wavelength_exponent=starting_config.lsf_wavelength_exponent,
        initial_wavelength_shift=0.0,
        initialize_lsf_sigma_grid=current_config.fit_lsf_sigma,
        initialize_wavelength_shift_grid=True,
    )


def coordinated_model_selection(
    native, native_config, deferred_pair, *, run_fit, starting_config, enable_joint=True
):
    """Native -> guarded rescued pair -> guarded joint winner -> unchanged masking.

    Joint failure cannot undo the rescued pair.  A second-stage winner is checked
    against the original native domains as well as its own reference, so allowed
    per-stage regressions cannot accumulate or change their scoring bands.
    """
    current, current_config = native, native_config
    details = {
        "method": "coordinated_rescue_then_joint_with_original_native_guards",
        "accepted": False,
        "selected_stage": "native",
        "conditional_on_full_data_structure": True,
        "native_continuum_order": int(native_config.continuum_order),
        "new_prespecified_safeguard": "whole_original_native_fit_window_rms_regression_at_most_two_percent",
        "sequence": ["native", "coordinated_rescue", "joint", "quality_mask_after_return"],
    }
    if deferred_pair is not None:
        rescued, rescued_config, stage = apply_rescued_pair(
            native,
            native_config,
            deferred_pair,
            run_fit=run_fit,
            starting_config=_fresh_start(native_config, starting_config),
        )
        stage = dict(stage)
        stage["accepted_before_native_fit_window_guard"] = bool(stage["accepted"])
        if stage["accepted"]:
            guard = native_fit_window_guard(native, rescued)
            stage["original_native_fit_window_guard"] = guard
            if guard["passed"]:
                current, current_config = rescued, rescued_config
                details.update(accepted=True, selected_stage="coordinated_rescue")
            else:
                stage.update(accepted=False, reason="native_fit_window_guard_rejected_rescue")
        details["rescue_stage"] = stage

    if enable_joint:
        candidate, candidate_config, stage = joint_model_reconsideration(
            current,
            current_config,
            run_fit=run_fit,
            starting_config=_fresh_start(current_config, starting_config),
        )
        stage = dict(stage)
        stage["accepted_before_original_native_guards"] = bool(stage["accepted"])
        stage["reference_stage"] = details["selected_stage"]
        if stage["accepted"]:
            try:
                full_guard = _fixed_full_comparison(native, candidate, candidate_config)
                stage["original_native_fixed_domain_guard"] = full_guard
                fit_guard = native_fit_window_guard(native, candidate)
                stage["original_native_fit_window_guard"] = fit_guard
                if full_guard["passed"] and fit_guard["passed"]:
                    current, current_config = candidate, candidate_config
                    details.update(accepted=True, selected_stage="joint")
                else:
                    stage.update(
                        accepted=False, reason="original_native_guards_rejected_joint_winner"
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                stage.update(accepted=False, reason="original_native_guard_invalid", error=str(exc))
        details["joint_stage"] = stage
    return current, current_config, details
