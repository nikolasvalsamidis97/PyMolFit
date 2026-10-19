from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import pymolfit.workflow as workflow


def exercise(
    monkeypatch,
    *,
    automatic=True,
    invalid_degrees=(2,),
    bad_bounds=False,
    changed_input=False,
    quadratic_gain=True,
    bad_degree1_residual=False,
):
    size = 120
    spectrum = SimpleNamespace(
        wavelength=np.linspace(4.9, 4.91, size),
        flux=np.full(size, 10.0),
        uncertainty=np.full(size, 2.0),
        valid=np.ones(size, dtype=bool),
    )
    folds = []
    for fold_index in range(3):
        score = np.zeros(size, dtype=bool)
        score[20 + fold_index * 30 : 25 + fold_index * 30] = True
        train = np.ones(size, dtype=bool)
        train[15 + fold_index * 30 : 30 + fold_index * 30] = False
        folds.append(
            {
                "train_masks": (train,),
                "score_masks": (score,),
                "eligible_masks": (spectrum.valid.copy(),),
                "_wavelengths": (spectrum.wavelength.copy(),),
                "_fluxes": (spectrum.flux.copy(),),
                "_uncertainties": (spectrum.uncertainty.copy(),),
                "score_signature": f"frozen_fold_{fold_index}",
            }
        )
    calls = []

    def fit(parts, *, config, fit_masks=None, **kwargs):
        is_wave = config.fit_wavelength_shift or config.fit_wavelength_polynomial
        index = next(
            (
                i
                for i, f in enumerate(folds)
                if fit_masks is not None and np.array_equal(fit_masks[0], f["train_masks"][0])
            ),
            None,
        )
        mask = np.ones(size, dtype=bool) if fit_masks is None else fit_masks[0].copy()
        calls.append(
            {
                "config": config,
                "fold": index,
                "mask": mask,
                "flux": parts[0].flux.copy(),
                "error": parts[0].uncertainty.copy(),
            }
        )
        continuum = np.full(size, 12.0)
        if index in (1, 2) and is_wave and config.continuum_order in invalid_degrees:
            continuum[100] = -1.0
        residual = 1.0
        if config.fit_wavelength_polynomial and config.wavelength_polynomial_order == 2:
            residual = 0.8 if quadratic_gain else 1.1
        if bad_degree1_residual and config.continuum_order == 1:
            residual *= 2.0
        coefficients = np.zeros(
            config.wavelength_polynomial_order + 1 if config.fit_wavelength_polynomial else 1
        )
        if bad_bounds and config.continuum_order < 2 and is_wave:
            coefficients[0] = config.wavelength_shift_bounds[1]
        part_spectrum = parts[0]
        if changed_input and is_wave and config.continuum_order < 2:
            part_spectrum = SimpleNamespace(**vars(parts[0]))
            part_spectrum.uncertainty = parts[0].uncertainty * 2.0
        part = SimpleNamespace(
            spectrum=part_spectrum,
            continuum=continuum,
            model_flux=np.full(size, 10.0 - 2.0 * residual),
            transmission=np.full(size, 0.8),
            fit_mask=mask,
            success=True,
            wavelength_coefficients=coefficients,
        )
        return SimpleNamespace(
            segment_results=(part,),
            success=True,
            species_scales={"H2O": 99.0 if index is None else 1.1 + index},
            lsf_sigma_pixels=1.0,
            lsf_lorentz_fwhm_pixels=0.0,
            lsf_wavelength_exponent=0.0,
            parameter_bound_status={},
            cost=residual**2,
            nfev=10,
        )

    monkeypatch.setattr(workflow, "fit_telluric_segments", fit)
    monkeypatch.setattr(workflow, "_segment_has_fit_pixels", lambda *a: True)
    monkeypatch.setattr(
        workflow, "build_wavelength_validation_plan", lambda *a, **k: (folds, {"reason": "ready"})
    )
    monkeypatch.setattr(
        workflow,
        "choose_continuum_order",
        lambda samples, **kwargs: (
            kwargs["initial_order"],
            {"selected_order": kwargs["initial_order"]},
        ),
    )
    config = workflow.FitConfig(
        continuum_order=2, wavelength_shift_unit="pixel", wavelength_shift_bounds=(-12.0, 12.0)
    )
    constant = replace(config, fit_wavelength_shift=True)
    linear = replace(config, fit_wavelength_polynomial=True, wavelength_polynomial_order=1)
    quadratic = replace(config, fit_wavelength_polynomial=True, wavelength_polynomial_order=2)
    records = [
        ("constant_pixel_shift", constant, None, {}),
        ("linear_pixel_trend", linear, None, {}),
        ("quadratic_pixel_trend", quadratic, None, {}),
    ]
    selected, details = workflow._validate_wavelength_candidates(
        spectrum,
        line_list=None,
        config=config,
        segment_size=0.01,
        global_bounds=(4.9, 4.91),
        selected=records[0],
        candidates=records[1:],
        use_processing_segments=False,
        automatic_continuum_order=automatic,
    )
    return selected, details, calls, folds


def test_invalid_continua_retried_as_common_matrix_without_forcing_final_degree(monkeypatch):
    selected, details, _calls, _folds = exercise(monkeypatch)
    rescue = details["paired_continuum_rescue"]
    assert rescue["triggered"] and rescue["selected_continuum_order"] == 1
    assert [a["continuum_order"] for a in rescue["attempts"]] == [1]
    assert selected[0] == "constant_pixel_shift"
    assert (
        details["deferred_continuum_wavelength_pair"]["wavelength_model"] == "quadratic_pixel_trend"
    )
    assert selected[1].continuum_order == 2  # final refinement remains automatic
    scores = rescue["attempts"][0]["fold_scores"]
    assert len(scores) == 3
    assert all(len(row["scores"]) == 3 for row in scores)
    assert all(score["selected_continuum_order"] == 1 for row in scores for score in row["scores"])
    assert rescue["rescued_comparisons"][-1]["accepted"]
    assert rescue["rescued_comparisons"][-1]["relative_rms_improvement"] == pytest.approx(0.2)


def test_retry_preserves_every_training_scoring_pixel_and_original_error(monkeypatch):
    _, details, calls, folds = exercise(monkeypatch)
    for call in calls:
        if call["fold"] is not None:
            fold = folds[call["fold"]]
            np.testing.assert_array_equal(call["mask"], fold["train_masks"][0])
            assert not np.any(call["mask"] & fold["score_masks"][0])
        np.testing.assert_array_equal(call["flux"], np.full(120, 10.0))
        np.testing.assert_array_equal(call["error"], np.full(120, 2.0))
    for row in details["paired_continuum_rescue"]["attempts"][0]["fold_scores"]:
        for index, score in enumerate(row["scores"]):
            assert score["point_count"] == 5
            assert score["score_signature"] == f"frozen_fold_{index}"


def test_retry_starts_are_training_only_and_shared_across_wavelength_models(monkeypatch):
    _, _, calls, _ = exercise(monkeypatch)
    retry = [c for c in calls if c["config"].continuum_order == 1]
    assert len(retry) == 12  # three neutral starts plus three models by three folds
    for call in retry:
        config = call["config"]
        if config.fit_wavelength_shift or config.fit_wavelength_polynomial:
            assert config.initial_species_scales == {"H2O": 1.1 + call["fold"]}
            assert config.initial_wavelength_shift == pytest.approx(0.48)
        else:
            assert config.initial_species_scales is None
            assert config.initial_wavelength_shift == 0.0


def test_valid_original_matrix_is_unchanged_and_never_retried(monkeypatch):
    selected, details, calls, _ = exercise(monkeypatch, invalid_degrees=())
    assert not details["paired_continuum_rescue"]["triggered"]
    assert all(c["config"].continuum_order == 2 for c in calls)
    assert len(calls) == 13
    assert selected[0] == "quadratic_pixel_trend"


def test_user_fixed_continuum_never_overridden(monkeypatch):
    selected, details, calls, _ = exercise(monkeypatch, automatic=False)
    assert not details["paired_continuum_rescue"]["triggered"]
    assert selected[0] == "constant_pixel_shift"
    assert all(c["config"].continuum_order == 2 for c in calls)


def test_constant_degree_only_used_if_linear_physical_retry_failed(monkeypatch):
    selected, details, _, _ = exercise(monkeypatch, invalid_degrees=(2, 1))
    rescue = details["paired_continuum_rescue"]
    assert [a["continuum_order"] for a in rescue["attempts"]] == [1, 0]
    assert rescue["selected_continuum_order"] == 0
    assert selected[0] == "constant_pixel_shift"
    assert details["deferred_continuum_wavelength_pair"]["continuum_order"] == 0


def test_unrepairable_continua_keep_original_selection_and_scores(monkeypatch):
    selected, details, _, _ = exercise(monkeypatch, invalid_degrees=(2, 1, 0))
    rescue = details["paired_continuum_rescue"]
    assert "selected_continuum_order" not in rescue
    assert not any(a["accepted"] for a in rescue["attempts"])
    assert selected[0] == "constant_pixel_shift"
    assert details["reference_fold_scores"][1]["reason"] == "nonpositive_continuum"


@pytest.mark.parametrize("option", ["bad_bounds", "changed_input"])
def test_retry_cannot_bypass_bounds_or_original_input_checks(monkeypatch, option):
    selected, details, _, _ = exercise(monkeypatch, **{option: True})
    assert "selected_continuum_order" not in details["paired_continuum_rescue"]
    assert selected[0] == "constant_pixel_shift"


def test_physical_retry_is_not_chosen_by_best_heldout_residual(monkeypatch):
    _, details, calls, _ = exercise(monkeypatch, bad_degree1_residual=True)
    rescue = details["paired_continuum_rescue"]
    assert rescue["selected_continuum_order"] == 1
    # Degree zero would score better in this fixture; it is never evaluated.
    assert all(call["config"].continuum_order != 0 for call in calls)


def test_rescue_does_not_force_a_wavelength_upgrade(monkeypatch):
    selected, details, _, _ = exercise(monkeypatch, quadratic_gain=False)
    assert details["paired_continuum_rescue"]["selected_continuum_order"] == 1
    assert selected[0] == "constant_pixel_shift"
    assert not any(c["accepted"] for c in details["comparisons"])
