"""Contract tests for isolated automatic model-reconsideration experiments."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from pymolfit import _joint_selection as joint
from pymolfit.fit import FitConfig
from pymolfit.spectrum import Spectrum


class JointSelectionTests(TestCase):
    def setUp(self):
        self.wave = np.linspace(4.9, 4.94, 600)
        self.spectrum = Spectrum(
            self.wave, np.ones(600) * 10, uncertainty=np.ones(600), wavelength_unit="micron"
        )
        self.fit_mask = self.spectrum.valid.copy()
        self.transmission = np.ones(600) * 0.7
        self.transmission[::3] = 0.99
        self.config = FitConfig(
            continuum_order=2,
            wavelength_shift_unit="pixel",
            wavelength_shift_bounds=(-12, 12),
            initial_species_scales={"H2O": 1.0},
            lsf_sigma_pixels=1.0,
            fit_lsf_sigma=True,
            lsf_sigma_bounds=(0.1, 2.0),
        )
        self.reference = self.result(1.0, self.fit_mask)
        self.calls = []
        self.folds = []
        for i in range(3):
            score = np.zeros(600, bool)
            guard = np.zeros(600, bool)
            lo = 100 + 150 * i
            score[lo : lo + 40] = True
            guard[lo - 10 : lo + 50] = True
            self.folds.append(
                {
                    "fold_index": i,
                    "score_mask": score,
                    "guard_mask": guard,
                    "exclude_ranges_micron": [(self.wave[lo - 10], self.wave[lo + 49])],
                }
            )

    def result(self, residual, mask, *, continuum=10.0):
        return SimpleNamespace(
            spectrum=self.spectrum,
            success=True,
            model_flux=np.ones(600) * (10 - residual),
            continuum=np.ones(600) * continuum,
            transmission=self.transmission.copy(),
            fit_mask=mask.copy(),
            nfev=4,
            parameter_bound_status={},
            wavelength_coefficients=np.array([0.1]),
            species_scales={"H2O": 9.0},
            provenance={},
        )

    def fit(self, config):
        self.calls.append(config)
        mask = self.fit_mask.copy()
        for lo, hi in config.exclude_ranges or ():
            mask &= ~((self.wave >= lo) & (self.wave <= hi))
        residual = 0.8 if config.continuum_order == 1 and joint.family(config) == "linear" else 1.0
        return self.result(residual, mask)

    def reconsider(self, **kwargs):
        with patch.object(
            joint, "continuum_guarded_plan", return_value=(self.folds, {"ready": True})
        ):
            return joint.joint_model_reconsideration(
                self.reference,
                self.config,
                run_fit=kwargs.get("run_fit", self.fit),
                starting_config=self.config,
            )

    def test_joint_accepts_validated_pair_and_keeps_matching_config(self):
        result, config, details = self.reconsider()
        self.assertTrue(details["accepted"])
        self.assertEqual(config.continuum_order, 1)
        self.assertEqual(joint.family(config), "linear")
        self.assertAlmostEqual(
            details["selected_comparison"]["pooled_rms_improvement_fraction"], 0.2
        )
        self.assertTrue(np.allclose(result.model_flux, 9.2))

    def test_folds_exclude_every_scored_and_guard_pixel(self):
        self.reconsider()
        guarded = [c for c in self.calls if c.exclude_ranges]
        self.assertEqual(len(guarded), 6)
        for config in guarded:
            lo, hi = config.exclude_ranges[0]
            mask = (self.wave >= lo) & (self.wave <= hi)
            self.assertTrue(any(np.array_equal(mask, fold["guard_mask"]) for fold in self.folds))

    def test_full_data_fitted_starts_are_never_reused(self):
        self.reconsider()
        self.assertTrue(all(c.initial_species_scales == {"H2O": 1.0} for c in self.calls))
        self.assertTrue(all(c.initial_wavelength_shift == 0.0 for c in self.calls))
        self.assertTrue(all(c.lsf_sigma_pixels == 1.0 for c in self.calls))
        self.assertTrue(all(c.wavelength_shift_bounds == (-12, 12) for c in self.calls))

    def test_wave_only_never_changes_continuum_degree(self):
        with patch.object(joint, "RECONSIDER_CONTINUUM", False):
            result, config, details = self.reconsider()
        self.assertIs(result, self.reference)
        self.assertIs(config, self.config)
        self.assertFalse(details["accepted"])
        self.assertEqual({c.continuum_order for c in self.calls}, {2})

    def test_invalid_continuum_does_not_win_low_residual_screen(self):
        def invalid(config):
            result = self.fit(config)
            if joint.family(config) == "linear" and config.continuum_order == 1:
                result.continuum[50] = -1.0
            return result

        result, _config, details = self.reconsider(run_fit=invalid)
        self.assertIs(result, self.reference)
        self.assertFalse(details["accepted"])
        self.assertTrue(
            any("invalid fold prediction" in x.get("error", "") for x in details["screen"])
        )

    def test_changed_training_mask_fails_closed(self):
        def leakage(config):
            result = self.fit(config)
            if config.exclude_ranges:
                result.fit_mask[:] = True
            return result

        result, _config, details = self.reconsider(run_fit=leakage)
        self.assertIs(result, self.reference)
        self.assertEqual(details["reason"], "baseline_validation_invalid")

    def test_changed_original_errors_are_rejected(self):
        def changed(config):
            result = self.fit(config)
            if config.exclude_ranges:
                result.spectrum = replace(self.spectrum, uncertainty=np.ones(600) * 2)
            return result

        result, _, details = self.reconsider(run_fit=changed)
        self.assertIs(result, self.reference)
        self.assertEqual(details["reason"], "baseline_validation_invalid")

    def test_missing_transparent_domain_is_explicitly_inconclusive(self):
        self.reference.transmission[:] = 0.7
        _result, _config, details = self.reconsider()
        self.assertTrue(details["accepted"])
        self.assertEqual(
            details["selected_comparison"]["domain_scores"]["transparent"],
            {"pixels": 0, "evaluated": False},
        )

    def test_full_domain_regression_rejects_good_global_candidate(self):
        domains = joint.domain_masks(self.reference)
        result = self.result(0.5, self.fit_mask)
        result.model_flux[self.transmission >= 0.98] = 8.9
        score = joint.full_comparison(self.reference, result, self.config, domains)
        self.assertFalse(score["passed"])
        self.assertIn("transparent_regression", score["reasons"])

    def test_joint_choice_does_not_depend_on_flux_in_original_exclusions(self):
        self.config = replace(self.config, exclude_ranges=((self.wave[40], self.wave[79]),))
        self.fit_mask[40:80] = False
        for level in (0.0, 10.0, -1e6):
            with self.subTest(excluded_flux=level):
                flux = np.full(600, 10.0)
                flux[40:80] = level
                self.spectrum = replace(self.spectrum, flux=flux)
                self.reference = self.result(1.0, self.fit_mask)
                result, config, details = self.reconsider()
                self.assertTrue(details["accepted"])
                self.assertEqual(config.continuum_order, 1)
                self.assertEqual(joint.family(config), "linear")
                self.assertTrue(np.allclose(result.model_flux, 9.2))
                domains = joint.domain_masks(self.reference, self.config.exclude_ranges)
                self.assertFalse(np.any(domains["full"][40:80]))
                self.assertEqual(int(domains["full"].sum()), 560)

    def test_bad_prediction_cannot_be_selected_despite_good_training_fit(self):
        def overfit(config):
            result = self.fit(config)
            if (
                config.exclude_ranges
                and config.continuum_order == 1
                and joint.family(config) == "linear"
            ):
                result.model_flux[:] = 8.0
            return result

        result, _config, details = self.reconsider(run_fit=overfit)
        self.assertIs(result, self.reference)
        self.assertFalse(details["accepted"])
        self.assertEqual(details["reason"], "no_predictively_supported_candidate")

    def test_absent_validation_support_returns_exact_baseline_without_refits(self):
        with patch.object(joint, "continuum_guarded_plan", return_value=([], {"reason": "weak"})):
            result, config, _details = joint.joint_model_reconsideration(
                self.reference, self.config, run_fit=self.fit, starting_config=self.config
            )
        self.assertIs(result, self.reference)
        self.assertIs(config, self.config)
        self.assertEqual(self.calls, [])

    def test_explicit_nonpixel_geometry_is_not_overwritten(self):
        config = replace(self.config, wavelength_shift_unit="micron")
        _result, returned, details = joint.joint_model_reconsideration(
            self.reference, config, run_fit=self.fit, starting_config=config
        )
        self.assertIs(returned, config)
        self.assertEqual(details["reason"], "unsupported_wavelength_geometry")
        self.assertEqual(self.calls, [])

    def test_bound_hit_is_not_promoted(self):
        result = self.result(0.5, self.fit_mask)
        result.wavelength_coefficients[:] = 11.999
        config = replace(self.config, fit_wavelength_shift=True)
        score = joint.full_comparison(
            self.reference, result, config, joint.domain_masks(self.reference)
        )
        self.assertIn("wavelength_coefficient_at_bound", score["reasons"])

    def test_actual_guard_plan_uses_largest_wavelength_family(self):
        config = joint.model_config(self.config, 2, "quadratic")
        folds, plan = joint.continuum_guarded_plan(self.reference, config, 2)
        self.assertEqual(len(folds), 3)
        self.assertEqual(plan["maximum_wavelength_displacement_pixels"], 36.0)
        self.assertGreaterEqual(plan["guard_pixels_per_side"], 72)
        for fold in folds:
            self.assertFalse(np.any(fold["score_mask"] & ~fold["guard_mask"]))


if __name__ == "__main__":
    main()
