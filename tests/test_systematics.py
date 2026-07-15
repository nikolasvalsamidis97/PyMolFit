from dataclasses import replace

import numpy as np

from pymolfit import (
    FitConfig,
    LineList,
    ModelConfig,
    Spectrum,
    fit_telluric_segments_with_systematics,
    fit_tellurics_with_systematics,
    transmission_model,
)


def test_systematic_refits_produce_and_propagate_transmission_envelope(tmp_path):
    rng = np.random.default_rng(841)
    wavelength = np.linspace(2.31, 2.36, 420)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    uncertainty = np.full(wavelength.shape, 0.002)
    flux = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.35}, lsf_sigma_pixels=1.0),
    ) + rng.normal(0.0, uncertainty)
    spectrum = Spectrum(wavelength=wavelength, flux=flux, uncertainty=uncertainty)
    baseline_config = FitConfig(
        continuum_order=0,
        species=("H2O",),
        lsf_sigma_pixels=0.6,
        estimate_uncertainties=True,
    )

    result = fit_tellurics_with_systematics(
        spectrum,
        line_list,
        baseline_config,
        {
            "broader Gaussian LSF": replace(
                baseline_config,
                lsf_sigma_pixels=1.4,
            )
        },
    )

    assert result.baseline.success
    assert result.variants["broader Gaussian LSF"].success
    expected = np.abs(
        result.variants["broader Gaussian LSF"].transmission
        - result.baseline.transmission
    )
    np.testing.assert_allclose(result.transmission_systematic_uncertainty, expected)
    np.testing.assert_allclose(result.transmission_systematic_envelope, expected)
    assert np.nanmax(result.transmission_systematic_uncertainty) > 0
    assert result.corrected.meta["model_systematic_uncertainty_propagated"] is True
    assert result.corrected.uncertainty is not None
    assert result.metrics["variant_count"] == 1

    table = result.to_table()
    assert "transmission_systematic_uncertainty" in table.colnames
    assert "corrected_uncertainty_with_systematics" in table.colnames
    assert "transmission_variant_broader_gaussian_lsf" in table.colnames
    output = tmp_path / "systematics.ecsv"
    result.write(output)
    assert output.exists()


def test_systematic_refits_require_named_variants():
    wavelength = np.linspace(2.31, 2.36, 80)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    spectrum = Spectrum(wavelength=wavelength, flux=np.ones_like(wavelength))

    with np.testing.assert_raises_regex(ValueError, "at least one"):
        fit_tellurics_with_systematics(
            spectrum,
            line_list,
            FitConfig(continuum_order=0, species=("H2O",)),
            {},
        )


def test_multi_segment_systematics_preserve_shared_refit_and_write_products(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 260)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.25}, lsf_sigma_pixels=0.9),
    )
    spectra = (
        Spectrum(wavelength=wavelength, flux=transmission, uncertainty=np.full(260, 0.003)),
        Spectrum(
            wavelength=wavelength,
            flux=1.1 * transmission,
            uncertainty=np.full(260, 0.003),
        ),
    )
    baseline = FitConfig(
        species=("H2O",),
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels=0.7,
        estimate_uncertainties=True,
    )

    result = fit_telluric_segments_with_systematics(
        spectra,
        line_list,
        baseline,
        {"broader_lsf": replace(baseline, lsf_sigma_pixels=1.1)},
    )

    assert result.baseline.success
    assert result.variants["broader_lsf"].success
    assert len(result.segment_results) == 2
    assert result.metrics["segment_count"] == 2
    assert result.metrics["transmission_systematic_rms_p95"] > 0
    assert all(segment.corrected.uncertainty is not None for segment in result.segment_results)
    result.write(tmp_path)
    assert (tmp_path / "segment_01.ecsv").exists()
    assert (tmp_path / "segment_02.ecsv").exists()
    assert (tmp_path / "segment_systematics_summary.json").exists()
