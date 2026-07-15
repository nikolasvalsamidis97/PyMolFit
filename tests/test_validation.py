import json

import numpy as np

from pymolfit import (
    FitConfig,
    LineList,
    ModelConfig,
    ScienceReadinessReport,
    Spectrum,
    ValidationCheck,
    compare_spectra,
    cross_validate_telluric_segments,
    transmission_model,
)


def test_compare_spectra_interpolates_overlap():
    candidate = Spectrum(
        wavelength=np.array([2.31, 2.32, 2.33, 2.34]),
        flux=np.array([1.0, 0.9, 1.1, 1.0]),
    )
    reference = Spectrum(
        wavelength=np.array([2.315, 2.325, 2.335]),
        flux=np.array([0.95, 1.0, 1.05]),
    )

    comparison = compare_spectra(candidate, reference)

    assert comparison.n_pixels == 2
    assert comparison.overlap_min == 2.315
    assert comparison.overlap_max == 2.335
    assert comparison.rms >= 0


def test_compare_spectra_normalizes_flux_scales():
    wavelength = np.linspace(2.31, 2.36, 20)
    candidate = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength) * 2.0)
    reference = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength) * 4.0)

    comparison = compare_spectra(candidate, reference, normalize=True)

    assert comparison.rms == 0.0


def test_science_readiness_report_required_fail_blocks_readiness(tmp_path):
    report = ScienceReadinessReport.create(
        [
            ValidationCheck("synthetic", "H2O", "PASS", value=0.01, threshold="<= 0.05"),
            ValidationCheck("external", "CRIRES", "FAIL", value=0.02, threshold="<= 0.01"),
            ValidationCheck("review", "independent", "MANUAL"),
        ],
        metadata={"campaign": "test"},
    )

    report.write(tmp_path)

    assert report.verdict == "NOT_SCIENCE_READY"
    payload = json.loads((tmp_path / "science_readiness_report.json").read_text())
    assert payload["status_counts"]["FAIL"] == 1
    assert (tmp_path / "science_readiness_checks.csv").exists()


def test_science_readiness_report_manual_check_is_incomplete():
    report = ScienceReadinessReport.create(
        [
            ValidationCheck("synthetic", "H2O", "PASS"),
            ValidationCheck("review", "independent", "MANUAL"),
        ]
    )

    assert report.verdict == "VALIDATION_INCOMPLETE"


def test_blocked_cross_validation_predicts_pixels_excluded_from_each_fold(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 420)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.35}, lsf_sigma_pixels=0.8),
    )
    coordinate = np.linspace(-1.0, 1.0, wavelength.size)
    uncertainty = np.full(wavelength.shape, 0.002)
    spectra = (
        Spectrum(
            wavelength=wavelength,
            flux=(1.0 + 0.08 * coordinate) * transmission,
            uncertainty=uncertainty,
        ),
        Spectrum(
            wavelength=wavelength,
            flux=(0.85 - 0.04 * coordinate) * transmission,
            uncertainty=uncertainty,
        ),
    )
    config = FitConfig(
        species=("H2O",),
        continuum_order=1,
        solve_continuum_linear=True,
        lsf_sigma_pixels=0.8,
    )

    result = cross_validate_telluric_segments(
        spectra,
        line_list=line_list,
        config=config,
        block_size=35,
        n_folds=2,
    )

    assert len(result.fold_results) == 2
    assert result.metrics["prediction_coverage"] == 1.0
    assert result.metrics["n_telluric_pixels"] > 0
    assert result.metrics["telluric_relative_rms_improvement"] > 2.0
    assert result.metrics["telluric_weighted_rms_improvement"] > 2.0
    assert all(np.all(assignment >= 0) for assignment in result.fold_assignment)
    assert all(np.all(np.isfinite(values)) for values in result.transmission)
    result.write(tmp_path)
    assert (tmp_path / "segment_01.ecsv").exists()
    assert (tmp_path / "segment_02.ecsv").exists()
    assert (tmp_path / "segment_cross_validation_summary.json").exists()


def test_blocked_cross_validation_rejects_invalid_split():
    wavelength = np.linspace(2.31, 2.36, 20)
    spectrum = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))
    line_list = LineList.demo_near_ir().select_species(("H2O",))

    with np.testing.assert_raises_regex(ValueError, "every fold"):
        cross_validate_telluric_segments(
            (spectrum,),
            line_list=line_list,
            config=FitConfig(species=("H2O",), continuum_order=0),
            block_size=20,
            n_folds=2,
        )
