import json

import numpy as np
from astropy.table import Table

from pymolfit import (
    FitConfig,
    LineList,
    ModelConfig,
    Spectrum,
    fit_telluric_segments,
    fit_tellurics,
    transmission_model,
)
from pymolfit.fit import _linearized_parameter_covariance


def test_fit_tellurics_improves_synthetic_spectrum():
    rng = np.random.default_rng(4)
    wavelength = np.linspace(2.31, 2.36, 900)
    line_list = LineList.demo_near_ir()
    true_transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(airmass=1.1, species_scales={"H2O": 2.0, "CO2": 0.6, "CH4": 1.3}),
    )
    continuum = 1.2 + 0.08 * (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    flux = continuum * true_transmission + rng.normal(0.0, 0.003, wavelength.size)
    spectrum = Spectrum(wavelength=wavelength, flux=flux, uncertainty=np.full_like(flux, 0.003))

    result = fit_tellurics(
        spectrum,
        line_list=line_list,
        config=FitConfig(airmass=1.1, continuum_order=1),
    )

    raw_scatter = np.nanstd(flux / continuum - 1.0)
    corrected_scatter = np.nanstd(result.corrected.flux / result.continuum - 1.0)
    assert result.success
    assert corrected_scatter < raw_scatter
    assert all(scale > 0 for scale in result.species_scales.values())


def test_fit_tellurics_accepts_fit_mask():
    wavelength = np.linspace(2.31, 2.36, 300)
    line_list = LineList.demo_near_ir()
    flux = transmission_model(wavelength, line_list, ModelConfig(species_scales={"H2O": 1.2}))
    spectrum = Spectrum(wavelength=wavelength, flux=flux)
    fit_mask = wavelength < 2.345

    result = fit_tellurics(spectrum, line_list=line_list, fit_mask=fit_mask)

    assert result.transmission.shape == wavelength.shape
    assert result.corrected.flux.shape == wavelength.shape
    np.testing.assert_array_equal(result.fit_mask, fit_mask)
    product = result.to_table()
    np.testing.assert_array_equal(product["fit_mask"], fit_mask)
    provenance = json.loads(product.meta["provenance_json"])
    assert provenance["line_source"] == "demo"
    assert provenance["line_count"] == line_list.wavelength.size
    assert provenance["selected_line_count"] <= provenance["line_count"]
    assert len(provenance["line_list_sha256"]) == 64
    assert len(provenance["fit_config_sha256"]) == 64


def test_fit_tellurics_reorders_fit_mask_with_unsorted_wavelength():
    wavelength = np.linspace(2.31, 2.36, 300)[::-1]
    line_list = LineList.demo_near_ir()
    flux = transmission_model(wavelength, line_list, ModelConfig())
    fit_mask = wavelength < 2.345

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        fit_mask=fit_mask,
    )

    assert np.all(np.diff(result.spectrum.wavelength) > 0)
    np.testing.assert_array_equal(
        result.fit_mask,
        result.spectrum.wavelength < 2.345,
    )


def test_fit_tellurics_estimates_and_propagates_local_uncertainties():
    rng = np.random.default_rng(19)
    wavelength = np.linspace(2.31, 2.36, 500)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.4}),
    )
    uncertainty = np.full(wavelength.shape, 0.003)
    flux = transmission + rng.normal(0.0, uncertainty)

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux, uncertainty=uncertainty),
        line_list=line_list,
        config=FitConfig(
            continuum_order=0,
            species=("H2O",),
            estimate_uncertainties=True,
        ),
    )

    assert result.parameter_covariance is not None
    assert result.parameter_covariance.shape == (2, 2)
    assert result.covariance_rank == 2
    assert result.reduced_chi_square > 0
    assert result.species_scale_uncertainties["H2O"] > 0
    assert result.transmission_uncertainty is not None
    assert np.all(np.isfinite(result.transmission_uncertainty))
    assert result.corrected.uncertainty is not None
    valid = np.isfinite(result.corrected.uncertainty)
    assert np.all(
        result.corrected.uncertainty[valid]
        >= uncertainty[valid] / result.transmission[valid]
    )
    assert "transmission_uncertainty" in result.to_table().colnames
    assert "corrected_uncertainty" in result.to_table().colnames


def test_multi_segment_fit_estimates_and_propagates_shared_uncertainties(tmp_path):
    rng = np.random.default_rng(29)
    wavelength = np.linspace(2.31, 2.36, 350)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.35}),
    )
    uncertainty = np.full(wavelength.shape, 0.004)
    spectra = tuple(
        Spectrum(
            wavelength=wavelength,
            flux=continuum * transmission + rng.normal(0.0, uncertainty),
            uncertainty=uncertainty,
        )
        for continuum in (0.9, 1.1)
    )

    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            species=("H2O",),
            continuum_order=0,
            estimate_uncertainties=True,
        ),
    )

    assert result.success
    assert result.parameter_covariance is not None
    assert result.parameter_covariance.shape == (3, 3)
    assert result.covariance_rank == 3
    assert result.species_scale_uncertainties["H2O"] > 0
    for segment in result.segment_results:
        assert segment.transmission_uncertainty is not None
        assert np.all(np.isfinite(segment.transmission_uncertainty))
        assert segment.corrected.uncertainty is not None
        product = tmp_path / f"segment_{len(list(tmp_path.iterdir()))}.ecsv"
        segment.write(product)
        table = Table.read(product)
        assert "input_mask" in table.colnames
        assert "fit_mask" in table.colnames
        assert "corrected_mask" in table.colnames
        assert json.loads(table.meta["provenance_json"])["schema_version"] == 1
        assert table.meta["covariance_full_rank"]
        assert table.meta["wavelength_medium"] == "vacuum"


def test_rank_deficient_covariance_is_not_reported_as_false_precision():
    jacobian = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    covariance, reduced_chi_square, rank = _linearized_parameter_covariance(
        jacobian,
        cost=1.0,
        n_residuals=3,
        n_parameters=2,
    )

    assert rank == 1
    assert reduced_chi_square == 2.0
    assert np.all(np.isnan(covariance))


def test_covariance_rank_is_invariant_to_parameter_units():
    jacobian = np.array(
        [
            [1.0e9, 1.0],
            [2.0e9, -1.0],
            [3.0e9, 0.5],
            [4.0e9, -0.5],
        ]
    )

    covariance, reduced_chi_square, rank = _linearized_parameter_covariance(
        jacobian,
        cost=1.0,
        n_residuals=4,
        n_parameters=2,
    )

    assert rank == 2
    assert reduced_chi_square == 1.0
    assert np.all(np.isfinite(covariance))
    assert covariance[0, 0] < covariance[1, 1]
