from dataclasses import replace

import numpy as np
import pytest

from pymolfit import LineList, Observation, Spectrum, correct_arrays
from pymolfit.fit import FitConfig, fit_telluric_segments
from pymolfit.workflow import _estimate_lsf_sigma_from_resolving_power as from_header
from pymolfit.workflow import _estimate_lsf_sigma_from_spectral_features as from_features


def _make_spectrum(offset=0.0, step=1e-5, size=500, label=0):
    pixel = np.arange(size, dtype=float)
    flux = np.ones(size)
    for center in (70, 170, 270, 370):
        flux -= 0.3 * np.exp(-0.5 * ((pixel - center) / 2.0) ** 2)
    return Spectrum(
        wavelength=1.5 + offset + pixel * step,
        flux=flux,
        group_id=np.full(size, label),
        uncertainty=np.ones(size) * 0.002,
    )


def _combine(*spectra):
    return Spectrum(
        wavelength=np.concatenate([s.wavelength for s in spectra]),
        flux=np.concatenate([s.flux for s in spectra]),
        uncertainty=np.concatenate([s.uncertainty for s in spectra]),
        group_id=np.concatenate([s.group_id for s in spectra]),
    )


def _features(s):
    return from_features(s, fit_ranges=None, exclude_ranges=None)


def test_overlap_uses_true_detector_spacing_not_interleaving():
    first, second = _make_spectrum(label=7), _make_spectrum(offset=5e-6, label=9)
    joint = from_header(_combine(first, second), {"RPOWER": 100000})
    individual = [from_header(s, {"RPOWER": 100000}) for s in (first, second)]
    assert joint["sampling_count"] == 998
    assert joint["initial_sigma_pixels"] == pytest.approx(
        np.mean([r["initial_sigma_pixels"] for r in individual]), rel=2e-10
    )
    assert joint["initial_sigma_pixels"] < 0.65


def test_feature_width_does_not_double_for_offset_overlap():
    first, second = _make_spectrum(label=7), _make_spectrum(offset=5e-6, label=9)
    joint = _features(_combine(first, second))
    reference = _features(first)
    assert joint["feature_count"] == 2 * reference["feature_count"]
    assert joint["feature_fwhm_pixels"] == pytest.approx(
        reference["feature_fwhm_pixels"], rel=1e-13
    )
    assert joint["initial_sigma_pixels"] == pytest.approx(
        reference["initial_sigma_pixels"], rel=1e-13
    )


@pytest.mark.parametrize("method", [_features, lambda s: from_header(s, {"RPOWER": 100000})])
def test_group_label_and_sample_permutation_invariance(method):
    original = _combine(_make_spectrum(label=7), _make_spectrum(offset=2.5e-6, label=9))
    permutation = np.random.default_rng(42).permutation(len(original.flux))
    shuffled = replace(
        original,
        wavelength=original.wavelength[permutation],
        flux=original.flux[permutation],
        uncertainty=original.uncertainty[permutation],
        group_id=original.group_id[permutation],
    )
    relabeled = replace(original, group_id=np.where(original.group_id == 7, 99, -2))
    assert method(original) == method(shuffled) == method(relabeled)


@pytest.mark.parametrize("method", [_features, lambda s: from_header(s, {"RPOWER": 100000})])
def test_single_group_matches_unlabeled_and_units(method):
    original = _make_spectrum()
    assert method(original) == method(replace(original, group_id=None))
    converted = replace(original, wavelength=original.wavelength * 1000, wavelength_unit="nm")
    assert method(original)["initial_sigma_pixels"] == pytest.approx(
        method(converted)["initial_sigma_pixels"], rel=1e-10
    )


def test_different_dispersion_scalar_is_pooled_pixel_median():
    # Majority of pixels are from the lower-resolution sampling; no claim a
    # single width can represent both groups exactly.
    dense, sparse = _make_spectrum(size=500, label=1), _make_spectrum(step=2e-5, size=900, label=2)
    joint = from_header(_combine(dense, sparse), {"RPOWER": 100000})
    a, b = [from_header(s, {"RPOWER": 100000})["initial_sigma_pixels"] for s in (dense, sparse)]
    assert b < joint["initial_sigma_pixels"] < a
    assert abs(joint["initial_sigma_pixels"] - b) < 0.01


@pytest.mark.parametrize("method", [_features, lambda s: from_header(s, {"RPOWER": 100000})])
def test_degenerate_group_does_not_invent_sampling(method):
    good = _make_spectrum(label=1)
    bad = Spectrum(
        wavelength=np.array([1.7]), flux=np.ones(1), uncertainty=np.ones(1), group_id=np.array([2])
    )
    assert method(_combine(good, bad)) == method(good)


def test_feature_exclusion_remains_applied_per_group():
    first, second = _make_spectrum(label=1), _make_spectrum(offset=0.02, label=2)
    combined = _combine(first, second)
    actual = from_features(combined, fit_ranges=((1.52, 1.525),), exclude_ranges=None)
    assert actual == _features(second)

    excluded = from_features(combined, fit_ranges=None, exclude_ranges=((1.5, 1.505),))
    assert excluded == _features(second)


def test_overlapping_orders_recover_injected_line_width_and_molecular_scale():
    # Offsetting three otherwise identical detector grids by fractions of a pixel
    # must not turn their combined sampling into three times the true resolution.
    # The old estimate put the injected sigma below the automatic fitting bounds.
    sigma_pixels = 1.5
    step = 1e-5
    grids = [1.5 + (np.arange(600) + offset) * step for offset in (0, 1 / 3, 2 / 3)]
    lines = LineList(
        wavelength=np.array([1.5008, 1.502, 1.5032, 1.5044, 1.5053]),
        strength=np.array([0.00002, 0.00001, 0.00003, 0.000015, 0.000025]),
        sigma=np.full(5, 1.5e-6),
        gamma=np.full(5, 1e-6),
        species=np.full(5, "H2O"),
    )
    spectra = tuple(
        Spectrum(wavelength=wave, flux=np.ones(wave.size), uncertainty=np.full(wave.size, 0.001))
        for wave in grids
    )
    # Generate noiseless data on separate physical grids, with a known continuum,
    # molecular scale, and instrumental width. This is a geometry regression,
    # not a test of the accuracy of the physical line model.
    truth = fit_telluric_segments(
        spectra,
        line_list=lines,
        wavelength_group_ids=(0, 1, 2),
        config=FitConfig(
            continuum_order=0,
            solve_continuum_linear=True,
            fixed_species_scales={"H2O": 1.0},
            lsf_sigma_pixels=sigma_pixels,
            fit_lsf_sigma=False,
            high_resolution_grid=False,
            high_resolution_oversampling=5.0,
            high_resolution_rebin_mode="molecfit_overlap",
            loss="linear",
        ),
    )
    wavelength = np.concatenate(grids)
    flux = np.concatenate([result.transmission for result in truth.segment_results])
    result = correct_arrays(
        wavelength,
        flux,
        uncertainty=np.full(wavelength.size, 0.001),
        group_id=np.repeat(np.arange(3), 600),
        observation=Observation(
            wavelength_frame="observatory",
            metadata={"SPEC_RES": 1.503 / (step * 2.354820045 * sigma_pixels)},
        ),
        line_list=lines,
        physical=False,
        continuum_order=0,
        solve_continuum_linear=True,
        lsf_sigma_pixels="auto",
        fit_lsf_sigma="auto",
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_variable_width=False,
        fit_wavelength_shift=False,
        fit_wavelength_polynomial=False,
        # Negligible bounded group shifts retain separate detector model grids.
        fit_segment_wavelength_shifts=True,
        wavelength_shift_bounds=(-1e-12, 1e-12),
        automatic_fit_regions=False,
        loss="linear",
        auto_segment=True,
        segment_size=0.01,
        high_resolution_grid=False,
        high_resolution_oversampling=5.0,
        max_nfev=200,
    )

    order = np.lexsort((result.spectrum.wavelength, result.spectrum.group_id))
    np.testing.assert_array_equal(result.spectrum.wavelength[order], wavelength)
    np.testing.assert_array_equal(result.spectrum.flux[order], flux)
    assert result.success
    assert result.lsf_sigma_pixels == pytest.approx(sigma_pixels, abs=1e-4)
    assert result.species_scales["H2O"] == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(result.model_flux[order], flux, rtol=0, atol=1e-7)
