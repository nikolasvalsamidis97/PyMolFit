import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymolfit import _coordinated_selection as combined
from pymolfit import workflow
from pymolfit.spectrum import Spectrum
from tests.test_wavelength_selection import _correct, _spectrum


@pytest.fixture
def case():
    n = 400
    spectrum = Spectrum(
        np.linspace(5.14, 5.15, n),
        np.full(n, 10.0),
        uncertainty=np.full(n, 2.0),
        wavelength_unit="micron",
    )
    config = workflow.FitConfig(
        continuum_order=2,
        fit_wavelength_shift=True,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=(-12.0, 12.0),
        lsf_sigma_pixels=1.2,
        fit_lsf_sigma=True,
        initial_species_scales={"H2O": 5.0},
    )
    start = replace(config, initial_species_scales=None, lsf_sigma_pixels=1.0)

    def make(residual=1.0):
        transmission = np.full(n, 0.7)
        transmission[::3] = 0.99
        return SimpleNamespace(
            spectrum=spectrum,
            fit_mask=spectrum.valid.copy(),
            continuum=np.full(n, 12.0),
            model_flux=10.0 - 2.0 * np.broadcast_to(residual, (n,)).copy(),
            transmission=transmission,
            wavelength_coefficients=np.array([-0.2, 0.01, 0.1]),
            success=True,
            parameter_bound_status={},
            nfev=8,
        )

    native = make()
    deferred = {
        "continuum_order": 1,
        "wavelength_model": "quadratic_pixel_trend",
        "wavelength_shift_bounds": [-12.0, 12.0],
    }
    return native, config, start, make, deferred


def run(case, *, deferred=True, enable_joint=True, fit=None):
    native, config, start, _, pair = case
    return combined.coordinated_model_selection(
        native,
        config,
        pair if deferred else None,
        run_fit=fit or (lambda c: pytest.fail("unexpected fit")),
        starting_config=start,
        enable_joint=enable_joint,
    )


def test_no_stages_keep_exact_native_objects(case, monkeypatch):
    monkeypatch.setattr(
        combined, "apply_rescued_pair", lambda *a, **k: pytest.fail("unexpected rescue")
    )
    monkeypatch.setattr(
        combined, "joint_model_reconsideration", lambda *a, **k: pytest.fail("unexpected joint")
    )
    result, config, details = run(case, deferred=False, enable_joint=False)
    assert result is case[0] and config is case[1]
    assert not details["accepted"] and details["selected_stage"] == "native"


@pytest.mark.parametrize("joint_failure", ["no_candidates", "invalid_fold", "unsupported_geometry"])
def test_joint_rejection_keeps_exact_rescued_result_and_config(case, monkeypatch, joint_failure):
    _native, _config, _start, make, _pair = case
    rescued = make(0.8)
    seen = []

    def joint(reference, conf, **kwargs):
        seen.append((reference, conf, kwargs["starting_config"]))
        return reference, conf, {"accepted": False, "reason": joint_failure}

    monkeypatch.setattr(combined, "joint_model_reconsideration", joint)
    fitted = []

    def fit(conf):
        fitted.append(conf)
        return rescued

    result, selected, details = run(case, fit=fit)
    assert result is rescued and selected is fitted[0]
    assert seen[0][0] is rescued and seen[0][1] is selected
    assert details["selected_stage"] == "coordinated_rescue"
    assert details["rescue_stage"]["accepted"]
    assert not details["joint_stage"]["accepted"]


def test_both_accept_joint_uses_rescued_structure_and_fresh_starts(case, monkeypatch):
    _native, config, _start, make, _pair = case
    rescued, candidate = make(0.8), make(0.6)
    sequence = []
    winner_config = replace(
        config,
        continuum_order=1,
        fit_wavelength_shift=True,
        fit_wavelength_polynomial=False,
        lsf_sigma_pixels=1.0,
    )

    def fit(conf):
        sequence.append("rescue")
        return rescued

    def joint(reference, conf, **kwargs):
        sequence.append("joint")
        assert reference is rescued
        assert conf.continuum_order == 1 and conf.wavelength_polynomial_order == 2
        fresh = kwargs["starting_config"]
        assert fresh.continuum_order == conf.continuum_order
        assert fresh.fit_wavelength_polynomial == conf.fit_wavelength_polynomial
        assert fresh.initial_species_scales is None
        assert fresh.initial_wavelength_shift == 0.0 and fresh.lsf_sigma_pixels == 1.0
        assert fresh.fit_ranges == conf.fit_ranges and fresh.exclude_ranges == conf.exclude_ranges
        assert fresh.wavelength_shift_bounds == config.wavelength_shift_bounds
        return candidate, winner_config, {"accepted": True}

    monkeypatch.setattr(combined, "joint_model_reconsideration", joint)
    result, selected, details = run(case, fit=fit)
    assert sequence == ["rescue", "joint"]
    assert result is candidate and selected is winner_config
    assert details["selected_stage"] == "joint"
    assert details["joint_stage"]["reference_stage"] == "coordinated_rescue"
    json.dumps(details, allow_nan=False)


def test_rescue_fit_window_regression_rejected_before_joint(case, monkeypatch):
    native, config, _, make, _ = case
    native.fit_mask[80:] = False
    candidate = make(0.7)
    candidate.fit_mask = native.fit_mask.copy()
    candidate.model_flux[:80] = 10 - 2 * 1.03
    assert combined._fixed_full_comparison(native, candidate, config)["passed"]

    def joint(reference, conf, **kwargs):
        assert reference is native and conf is config
        return reference, conf, {"accepted": False}

    monkeypatch.setattr(combined, "joint_model_reconsideration", joint)
    result, selected, details = run(case, fit=lambda c: candidate)
    assert result is native and selected is config
    assert details["rescue_stage"]["accepted_before_native_fit_window_guard"]
    assert not details["rescue_stage"]["accepted"]
    assert details["rescue_stage"]["reason"] == "native_fit_window_guard_rejected_rescue"


@pytest.mark.parametrize("excluded_flux", [0.0, 10.0, -1e6])
def test_rescue_scores_ignore_original_exclusions(case, excluded_flux):
    native, config, _, make, _ = case
    wave = native.spectrum.wavelength
    config = replace(config, exclude_ranges=((wave[150], wave[249]),))
    native.fit_mask[150:250] = False
    candidate = make(0.8)
    candidate.fit_mask = native.fit_mask.copy()
    flux = native.spectrum.flux.copy()
    flux[150:250] = excluded_flux
    native.spectrum = replace(native.spectrum, flux=flux)
    candidate.spectrum = native.spectrum
    candidate.model_flux[150:250] = 1e4
    comparison = combined._fixed_full_comparison(native, candidate, config)
    assert comparison["passed"]
    assert comparison["scores"]["full"]["pixels"] == 300
    assert comparison["scores"]["full"]["rms_improvement_fraction"] == pytest.approx(0.2)


def test_joint_only_can_accept_with_exact_winning_config(case, monkeypatch):
    _native, config, _, make, _ = case
    candidate = make(0.8)
    chosen = replace(config, continuum_order=1)
    monkeypatch.setattr(
        combined,
        "joint_model_reconsideration",
        lambda *a, **k: (candidate, chosen, {"accepted": True}),
    )
    result, selected, details = run(case, deferred=False)
    assert result is candidate and selected is chosen
    assert details["accepted"] and details["selected_stage"] == "joint"
    assert "rescue_stage" not in details


def test_cumulative_band_regression_is_checked_on_original_native_domains(case, monkeypatch):
    native, config, _, make, _ = case
    rescued = make(0.8)
    candidate = make(0.6)
    rescued.model_flux[::3] = 10 - 2 * 1.015
    candidate.model_flux[::3] = 10 - 2 * 1.029
    # Moving transmission bins would hide the original transparent-band loss.
    rescued.transmission[:] = 0.7
    candidate.transmission[:] = 0.7
    assert combined._fixed_full_comparison(native, rescued, config)["passed"]
    assert combined._fixed_full_comparison(rescued, candidate, config)["passed"]
    original = combined._fixed_full_comparison(native, candidate, config)
    assert not original["passed"] and "transparent_regression" in original["reasons"]
    fitted = []

    def fit(conf):
        fitted.append(conf)
        return rescued

    monkeypatch.setattr(
        combined,
        "joint_model_reconsideration",
        lambda *a, **k: (candidate, config, {"accepted": True}),
    )
    result, selected, details = run(case, fit=fit)
    assert result is rescued and selected is fitted[0]
    scored = details["joint_stage"]["original_native_fixed_domain_guard"]
    assert scored["scores"]["transparent"]["pixels"] == 106
    assert not details["joint_stage"]["accepted"]
    assert details["selected_stage"] == "coordinated_rescue"


def test_joint_native_fitwindow_regression_retains_rescue(case, monkeypatch):
    native, config, _, make, _ = case
    native.fit_mask[80:] = False
    rescued = make(0.9)
    candidate = make(0.6)
    rescued.fit_mask = native.fit_mask.copy()
    candidate.fit_mask = native.fit_mask.copy()
    candidate.model_flux[:80] = 10 - 2 * 1.03
    assert combined._fixed_full_comparison(native, candidate, config)["passed"]
    monkeypatch.setattr(
        combined,
        "joint_model_reconsideration",
        lambda *a, **k: (candidate, config, {"accepted": True}),
    )
    result, _selected, details = run(case, fit=lambda c: rescued)
    assert result is rescued
    assert not details["joint_stage"]["original_native_fit_window_guard"]["passed"]
    assert details["selected_stage"] == "coordinated_rescue"


@pytest.mark.parametrize("change", ["errors", "fit_mask", "input_valid", "flux", "wavelength"])
def test_changed_input_or_fitmask_cannot_replace_current(case, monkeypatch, change):
    native, config, _, make, _ = case
    rescued = make(0.8)
    candidate = make(0.6)
    snapshots = {
        name: getattr(native, name).copy()
        for name in ("model_flux", "continuum", "transmission", "fit_mask")
    }
    if change == "errors":
        candidate.spectrum = replace(native.spectrum, uncertainty=np.ones(400))
    elif change == "fit_mask":
        candidate.fit_mask[20] = False
    elif change == "input_valid":
        valid = native.spectrum.valid.copy()
        valid[20] = False
        candidate.spectrum = replace(native.spectrum, mask=~valid)
    elif change == "flux":
        candidate.spectrum = replace(native.spectrum, flux=np.full(400, 11.0))
    elif change == "wavelength":
        candidate.spectrum = replace(native.spectrum, wavelength=native.spectrum.wavelength + 0.001)
    monkeypatch.setattr(
        combined,
        "joint_model_reconsideration",
        lambda *a, **k: (candidate, config, {"accepted": True}),
    )
    result, _, details = run(case, fit=lambda c: rescued)
    assert result is rescued and not details["joint_stage"]["accepted"]
    for name, value in snapshots.items():
        np.testing.assert_array_equal(getattr(native, name), value)


@pytest.mark.parametrize("residual,expected", [(1.02, True), (1.02001, False), (0.8, True)])
def test_native_fitwindow_exact_two_percent_boundary(case, residual, expected):
    native, _, _, make, _ = case
    candidate = make(residual)
    details = combined.native_fit_window_guard(native, candidate)
    assert details["passed"] == expected
    assert details["baseline"]["pixels"] == 400


def test_fitwindow_guard_includes_order_edges_and_uses_original_errors(case):
    native, _, _, make, _ = case
    native.fit_mask[:] = False
    native.fit_mask[:20] = True
    native.fit_mask[-20:] = True
    candidate = make(0.1)
    candidate.fit_mask = native.fit_mask.copy()
    candidate.model_flux[native.fit_mask] = 10 - 2 * 1.03
    assert (
        combined.native_fit_window_guard(native, candidate)["reason"]
        == "native_fit_window_rms_regression"
    )
    changed = make(0.1)
    changed.fit_mask = native.fit_mask.copy()
    changed.spectrum = replace(native.spectrum, uncertainty=np.full(400, 200.0))
    assert (
        combined.native_fit_window_guard(native, changed)["reason"]
        == "native_fit_window_guard_invalid"
    )


@pytest.mark.parametrize("residual,passed", [(0.0, True), (0.1, False)])
def test_zero_native_residual_diagnostics_are_json_serializable(case, residual, passed):
    native, _, _, make, _ = case
    native.model_flux[:] = 10.0
    details = combined.native_fit_window_guard(native, make(residual))
    assert details["passed"] == passed
    json.dumps(details, allow_nan=False)


def test_no_post_selection_continuum_or_wavelength_refits_and_one_mask():
    source = Path(workflow.__file__).read_text()
    late = source.index(
        "selected_result, fit_config, coordinated_details = coordinated_model_selection("
    )
    finalmask = source.index(
        "selected_result, quality_mask_resolution = mask_uncertain_correction("
    )
    refinement = source.index(
        "selected_result, fit_config, validation = guarded_continuum_refinement("
    )
    assert refinement < late < finalmask
    line = source[:late].count("\n") + 1
    owner = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.lineno <= line <= n.end_lineno
    )
    end = sum(len(s) for s in source.splitlines(keepends=True)[: owner.end_lineno])
    tail = source[late:end]
    for function in (
        "choose_continuum_order(",
        "guarded_continuum_refinement(",
        "run_fit(",
        "_validate_wavelength_candidates(",
    ):
        assert function not in tail
    assert tail.count("mask_uncertain_correction(") == 1
    eligibility = source[source.index("    joint_eligible = bool(") : late]
    assert (
        "automatic_continuum_order" in eligibility and "auto_select_wavelength_model" in eligibility
    )
    assert "stellar_fit_weights is None" in eligibility


@pytest.mark.parametrize(
    "auto_lorentz,fit_lorentz,variable,fit_exponent,expected_lorentz,expected_exponent",
    [
        (True, True, True, True, 1.1774100225, 1.0),
        (True, False, False, False, 0.0, 0.0),
        (False, False, True, False, 0.4, 1.0),
        (False, True, True, True, 0.4, 1.0),
    ],
)
def test_neutral_non_gaussian_lsf_starts_replace_pilot_values(
    case, auto_lorentz, fit_lorentz, variable, fit_exponent, expected_lorentz, expected_exponent
):
    config = replace(
        case[1],
        lsf_lorentz_fwhm_pixels=4.9,
        fit_lsf_lorentz_fwhm=fit_lorentz,
        lsf_lorentz_fwhm_bounds=(0.0, 6.0),
        lsf_variable_width=variable,
        lsf_wavelength_exponent=2.8,
        fit_lsf_wavelength_exponent=fit_exponent,
    )
    start = combined.configured_selection_start(
        config,
        initial_species_scales=None,
        lsf_sigma_pixels=1.0,
        resolved_lorentz_fwhm_pixels=0.4,
        automatic_lorentz=auto_lorentz,
    )
    assert start.lsf_lorentz_fwhm_pixels == pytest.approx(expected_lorentz)
    assert start.lsf_wavelength_exponent == expected_exponent
    assert start.initial_species_scales is None
    assert start.lsf_sigma_pixels == 1.0 and start.initial_wavelength_shift == 0.0
    assert start.fit_lsf_lorentz_fwhm == fit_lorentz
    assert start.fit_lsf_wavelength_exponent == fit_exponent
    assert start.lsf_variable_width == variable
    assert start.lsf_lorentz_fwhm_bounds == config.lsf_lorentz_fwhm_bounds
    assert start.continuum_order == config.continuum_order
    # Both later stage starts retain the neutral values, even if the accepted
    # reference config happened to contain pilot-derived fitted values.
    fresh = combined._fresh_start(config, start)
    assert fresh.lsf_lorentz_fwhm_pixels == start.lsf_lorentz_fwhm_pixels
    assert fresh.lsf_wavelength_exponent == start.lsf_wavelength_exponent


def test_neutral_lsf_starts_respect_original_bounds_without_using_fitted_values(case):
    config = replace(
        case[1],
        lsf_lorentz_fwhm_pixels=3.5,
        fit_lsf_lorentz_fwhm=True,
        lsf_lorentz_fwhm_bounds=(2.0, 4.0),
        lsf_variable_width=True,
        lsf_wavelength_exponent=2.9,
        fit_lsf_wavelength_exponent=True,
        lsf_wavelength_exponent_bounds=(2.0, 3.0),
    )
    fresh = combined.configured_selection_start(
        config,
        initial_species_scales={"H2O": 1.0},
        lsf_sigma_pixels=1.0,
        resolved_lorentz_fwhm_pixels=0.0,
        automatic_lorentz=True,
    )
    assert fresh.lsf_lorentz_fwhm_pixels == pytest.approx(2.02)
    assert fresh.lsf_wavelength_exponent == 2.0
    assert fresh.lsf_lorentz_fwhm_bounds == (2.0, 4.0)
    assert fresh.lsf_wavelength_exponent_bounds == (2.0, 3.0)


def test_workflow_passes_pre_pilot_resolved_starts_to_final_stages():
    source = Path(workflow.__file__).read_text()
    late = source.index("coordinated_start = configured_selection_start(")
    call = source[
        late : source.index("        selected_result, fit_config, coordinated_details", late)
    ]
    assert "resolved_lorentz_fwhm_pixels=resolved_lsf_lorentz_fwhm_pixels" in call
    assert 'automatic_lorentz=lsf_lorentz_fwhm_pixels == "auto"' in call
    assert "lsf_sigma_pixels=starting_sigma" in call
    assert "selected_result." not in call


@pytest.mark.parametrize(
    "options",
    [
        {"continuum_order": 2},
        {"fit_wavelength_shift": False},
        {"fit_wavelength_shift": True, "wavelength_shift_bounds": (-0.0001, 0.0001)},
        {
            "fit_wavelength_polynomial": True,
            "wavelength_polynomial_order": 1,
            "wavelength_shift_bounds": (-0.0001, 0.0001),
        },
    ],
)
def test_explicit_model_controls_bypass_coordinated_selection(monkeypatch, options):
    def unexpected_selection(*args, **kwargs):
        pytest.fail("An explicit model control entered coordinated automatic selection")

    monkeypatch.setattr(workflow, "coordinated_model_selection", unexpected_selection)
    data = _spectrum([0.8], uncertainty=0.002)
    result = _correct(data, **{"continuum_order": "auto", **options})
    assert result.success
    assert "coordinated_joint_selection" not in result.provenance["wavelength_alignment"]
    np.testing.assert_array_equal(result.spectrum.wavelength, data[0])
    np.testing.assert_array_equal(result.spectrum.flux, data[1])
    np.testing.assert_array_equal(result.spectrum.uncertainty, data[2])


def test_automatic_single_order_runs_coordinated_selection_before_quality_mask(monkeypatch):
    calls = []

    def retain_native(native, config, deferred_pair, **kwargs):
        assert kwargs["enable_joint"]
        assert native.success
        calls.append(native)
        return native, config, {"accepted": False, "selected_stage": "native"}

    monkeypatch.setattr(workflow, "coordinated_model_selection", retain_native)
    result = _correct(_spectrum([0.8], uncertainty=0.002), continuum_order="auto")
    assert result.success
    assert len(calls) == 1
    assert result.provenance["wavelength_alignment"]["coordinated_joint_selection"] == {
        "accepted": False,
        "selected_stage": "native",
    }
