"""Recovery checks using actual synthetic spectra and controlled failure cases."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, least_squares

from pymolfit import (
    FitConfig,
    LineList,
    ModelConfig,
    Spectrum,
    fit_telluric_segments,
    fit_tellurics,
    transmission_model,
)
from pymolfit._lsf_retry import retry_unresponsive_lsf


def synthetic_fit(segmented, profile=True, initial=0.088, fit_width=True, loss="linear"):
    wavelength = np.linspace(1.0, 1.003, 601)
    lines = LineList(
        wavelength=np.array([1.0007, 1.0014, 1.0023]),
        strength=np.array([1.0e-5, 2.0e-5, 1.6e-5]),
        sigma=np.full(3, 2.0e-6),
        gamma=np.full(3, 5.0e-7),
        species=np.full(3, "H2O"),
    )
    transmission = transmission_model(
        wavelength, lines, ModelConfig(lsf_sigma_pixels=1.0, species_scales={"H2O": 1.3})
    )
    coordinate = np.linspace(-1.0, 1.0, wavelength.size)
    flux = (1.2 + 0.05 * coordinate) * transmission
    spectrum = Spectrum(wavelength=wavelength, flux=flux, uncertainty=np.full(601, 0.002))
    config = FitConfig(
        continuum_order=1,
        solve_continuum_linear=profile,
        lsf_sigma_pixels=initial,
        fit_lsf_sigma=fit_width,
        lsf_sigma_bounds=(0.0, 3.0),
        staged_optimization=False,
        loss=loss,
        max_nfev=100,
    )
    if segmented:
        result = fit_telluric_segments([spectrum], line_list=lines, config=config)
        return result, result.segment_results[0], spectrum
    result = fit_tellurics(spectrum, line_list=lines, config=config)
    return result, result, spectrum


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_real_fitter_recovers_broadening_and_free_species(segmented, profile):
    result, segment, spectrum = synthetic_fit(segmented, profile)
    assert result.success
    assert result.lsf_sigma_pixels == pytest.approx(1.0, abs=2e-3)
    assert result.species_scales["H2O"] == pytest.approx(1.3, abs=2e-3)
    assert np.sqrt(np.mean((segment.model_flux - spectrum.flux) ** 2)) < 1e-6
    recovery = result.provenance["lsf_sigma_recovery"]
    assert recovery["triggered"]
    assert recovery["selected_attempt"] > 0
    assert recovery["attempts"][0]["cost"] > 10 * result.cost
    assert result.nfev == sum(a["nfev"] for a in recovery["attempts"])


@pytest.mark.parametrize("segmented", [False, True])
def test_healthy_initial_width_avoids_retries(segmented):
    result, _, _ = synthetic_fit(segmented, initial=0.7)
    assert result.lsf_sigma_pixels == pytest.approx(1.0, abs=2e-3)
    assert not result.provenance["lsf_sigma_recovery"]["triggered"]


@pytest.mark.parametrize("segmented", [False, True])
def test_fixed_width_is_untouched(segmented):
    result, _, _ = synthetic_fit(segmented, fit_width=False)
    assert result.lsf_sigma_pixels == 0.088
    assert "lsf_sigma_recovery" not in result.provenance


def test_robust_loss_uses_its_own_cost_for_retry_comparison():
    result, _, _ = synthetic_fit(False, loss="soft_l1")
    recovery = result.provenance["lsf_sigma_recovery"]
    assert recovery["triggered"]
    assert result.cost == min(a["cost"] for a in recovery["attempts"] if a["valid"])


def helper_case(*, upper=3.0, optimize=None, valid=lambda _: True, responsive=True):
    # Deliberately flat numerical width response below .14, coupled to a
    # freely refitted second parameter; reproduces the failure mechanism.
    def residual(x):
        effective_width = x[0] if x[0] > 0.14 and responsive else 0.0
        return np.array([effective_width + x[1] - 1.3, x[1] - 0.3])

    lower = np.array([0.0, -2.0])
    bounds = np.array([upper, 2.0])
    initial = np.array([0.088, 0.0])
    fit = least_squares(residual, initial, bounds=(lower, bounds), gtol=1e-12)
    if optimize is None:

        def optimize(x):
            return least_squares(residual, x, bounds=(lower, bounds), gtol=1e-12)

    return retry_unresponsive_lsf(
        fit,
        initial=initial,
        parameter_index=0,
        lower=lower,
        upper=bounds,
        residual=residual,
        optimize=optimize,
        valid_candidate=valid,
    )


def test_bounds_are_preserved_even_when_upper_below_recommended_starts():
    result, recovery = helper_case(upper=0.2)
    assert recovery["triggered"]
    assert len(recovery["attempts"]) == 2
    assert all(a["start_width_pixels"] < 0.2 for a in recovery["attempts"])
    assert result.x[0] <= 0.2


def test_no_sensitive_allowed_width_does_not_retry():
    _, recovery = helper_case(upper=0.1)
    assert not recovery["triggered"]
    _, recovery = helper_case(responsive=False)
    assert not recovery["triggered"]


def test_failed_retry_does_not_discard_original_fit():
    def optimize(_):
        raise ValueError("synthetic optimization failure")

    result, recovery = helper_case(optimize=optimize)
    assert result.success
    assert recovery["selected_attempt"] == 0
    assert all(not a["valid"] for a in recovery["attempts"][1:])


@pytest.mark.parametrize("failure", ["nonconverged", "nonfinite", "invalid_continuum"])
def test_invalid_candidates_cannot_replace_original(failure):
    def optimize(x):
        return OptimizeResult(
            x=x,
            fun=np.array([0.0, 0.0]),
            cost=np.nan if failure == "nonfinite" else 0.0,
            nfev=2,
            success=failure != "nonconverged",
        )

    result, recovery = helper_case(
        optimize=optimize, valid=lambda _: failure != "invalid_continuum"
    )
    assert recovery["selected_attempt"] == 0
    assert result.cost > 0


def test_smooth_but_weak_width_derivative_still_triggers():
    from scipy.special import erf

    def residual(x):
        return np.array([1.0 - erf(0.5 / (np.sqrt(2.0) * x[0])) - 0.3])

    fit = OptimizeResult(x=np.array([0.088]), cost=0.045, nfev=1, success=True)
    result, recovery = retry_unresponsive_lsf(
        fit,
        initial=fit.x,
        parameter_index=0,
        lower=np.array([0.001]),
        upper=np.array([3.0]),
        residual=residual,
        optimize=lambda x: least_squares(residual, x, bounds=([0.001], [3.0])),
        valid_candidate=lambda _: True,
    )
    assert recovery["triggered"]
    assert result.cost < 1e-10
