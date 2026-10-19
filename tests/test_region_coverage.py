"""Regression coverage for automatic windows starving local continua."""

import json

import numpy as np
import pytest

from pymolfit import LineList, Spectrum, correct_arrays
from pymolfit._automatic import ensure_automatic_region_coverage
from pymolfit.model import optical_depth_basis, transmission_from_basis


def selected_pixels(spectrum, ranges, exclusions=()):
    wave = spectrum.to_unit("micron").wavelength
    selected = np.zeros(wave.size, dtype=bool)
    for lower, upper in ranges:
        selected |= (wave >= lower) & (wave <= upper)
    selected &= spectrum.valid
    for lower, upper in exclusions:
        selected &= ~((wave >= lower) & (wave <= upper))
    return selected


def make_segment(start, count=500, **kwargs):
    return Spectrum(np.linspace(start, start + 0.009, count), np.ones(count), **kwargs)


def test_sparse_and_empty_chunks_expand_without_changing_supported_chunk():
    segments = tuple(make_segment(1.0 + 0.02 * index) for index in range(3))
    ranges = (
        (segments[0].wavelength[200], segments[0].wavelength[220]),
        (segments[2].wavelength[100], segments[2].wavelength[300]),
    )
    expanded, details = ensure_automatic_region_coverage(segments, ranges, (), continuum_order=2)
    assert details["expanded_segment_count"] == 2
    assert details["minimum_fraction"] == 0.25
    for segment in segments[:2]:
        np.testing.assert_array_equal(selected_pixels(segment, expanded), segment.valid)
    np.testing.assert_array_equal(
        selected_pixels(segments[2], expanded), selected_pixels(segments[2], ranges)
    )
    assert [record["selected_before"] for record in details["segments"]] == [21, 0, 201]
    assert [record["selected_after"] for record in details["segments"]] == [500, 500, 201]
    assert [record["required_pixels"] for record in details["segments"]] == [125] * 3
    json.dumps(details, allow_nan=False)


@pytest.mark.parametrize("count", [124, 125])
def test_coverage_floor_is_inclusive(count):
    segment = make_segment(1.0)
    ranges = ((segment.wavelength[100], segment.wavelength[100 + count - 1]),)
    expanded, details = ensure_automatic_region_coverage((segment,), ranges, (), continuum_order=2)
    assert details["expanded_segment_count"] == int(count < 125)
    assert selected_pixels(segment, expanded).sum() == (500 if count < 125 else count)


def test_degree_floor_and_insufficient_available_samples_are_reported():
    segment = make_segment(1.0, count=35)
    expanded, details = ensure_automatic_region_coverage((segment,), (), (), continuum_order=4)
    assert details["minimum_pixels"] == 40
    record = details["segments"][0]
    assert record["available_below_minimum"]
    assert record["required_pixels"] == 35
    np.testing.assert_array_equal(selected_pixels(segment, expanded), segment.valid)


def test_expansion_preserves_masks_and_exclusions_including_fully_excluded_chunk():
    mask = np.ones(500, dtype=bool)
    mask[::7] = False
    first = make_segment(1.0, mask=mask)
    second = make_segment(1.02)
    exclusions = ((1.003, 1.006), (1.02, 1.029))
    expanded, details = ensure_automatic_region_coverage(
        (first, second), (), exclusions, continuum_order=2
    )
    expected = first.valid & ((first.wavelength < 1.003) | (first.wavelength > 1.006))
    np.testing.assert_array_equal(selected_pixels(first, expanded, exclusions), expected)
    assert not selected_pixels(second, expanded, exclusions).any()
    assert details["segments"][0]["available_pixels"] == int(expected.sum())
    assert details["segments"][1]["available_pixels"] == 0
    assert not details["segments"][1]["expanded"]


def test_added_windows_do_not_bridge_large_gaps_or_physical_groups():
    wave = np.concatenate((np.linspace(1.0, 1.004, 100), np.linspace(1.1, 1.104, 100)))
    with_gap = Spectrum(wave, np.ones(200))
    grouped = Spectrum(wave, np.ones(200), group_id=np.repeat([7, 9], 100))
    for segment in (with_gap, grouped):
        expanded, _ = ensure_automatic_region_coverage((segment,), (), (), continuum_order=2)
        assert all(upper < 1.01 or lower > 1.09 for lower, upper in expanded)
        np.testing.assert_array_equal(selected_pixels(segment, expanded), segment.valid)


def test_overlapping_groups_report_final_global_union_counts():
    # Ranges have wavelength scope, so an expanded group's range also applies
    # to another group at the same wavelengths. Report the resulting truth.
    wave = np.linspace(1.0, 1.009, 500)
    sparse = Spectrum(wave, np.ones(500), group_id=np.full(500, 3))
    mask = np.zeros(500, dtype=bool)
    mask[100:300] = True
    supported = Spectrum(wave, np.ones(500), mask=mask, group_id=np.full(500, 8))
    original = ((wave[100], wave[199]),)
    expanded, details = ensure_automatic_region_coverage(
        (supported, sparse), original, (), continuum_order=2
    )
    assert details["segments"][0]["selected_before"] == 100
    assert details["segments"][0]["selected_after"] == 200
    for segment, record in zip((supported, sparse), details["segments"], strict=True):
        assert record["selected_after"] == selected_pixels(segment, expanded).sum()


def test_nan_flux_and_invalid_errors_never_become_fit_pixels():
    wave = np.linspace(1.0, 1.009, 200)
    flux = np.ones(200)
    errors = np.full(200, 0.01)
    flux[20] = np.nan
    errors[50] = 0.0
    spectrum = Spectrum(wave, flux, uncertainty=errors)
    expanded, details = ensure_automatic_region_coverage((spectrum,), (), (), continuum_order=2)
    np.testing.assert_array_equal(selected_pixels(spectrum, expanded), spectrum.valid)
    assert details["segments"][0]["available_pixels"] == 198


def test_isolated_samples_and_nonmicron_input_preserve_exact_eligible_pixels():
    wave = np.linspace(1000.0, 1009.0, 20)
    mask = np.zeros(20, dtype=bool)
    mask[[0, 7, 19]] = True
    spectrum = Spectrum(wave, np.ones(20), mask=mask, wavelength_unit="nm")
    expanded, details = ensure_automatic_region_coverage((spectrum,), (), (), continuum_order=2)
    assert len(expanded) == 3
    assert all(np.isfinite(lower) and lower < upper for lower, upper in expanded)
    np.testing.assert_array_equal(selected_pixels(spectrum, expanded), mask)
    assert details["segments"][0]["selected_after"] == 3
    assert details["segments"][0]["available_below_minimum"]


def test_sparse_automatic_windows_expand_in_real_correction_and_keep_input_errors():
    wave = np.linspace(1.0, 1.009, 500)
    lines = LineList(
        wavelength=np.array([1.0045]),
        strength=np.array([1.0e-5]),
        sigma=np.array([1.2e-5]),
        gamma=np.array([3.0e-6]),
        species=np.array(["H2O"]),
    )
    species, basis = optical_depth_basis(wave, lines)
    flux = (1.0 + 0.1 * np.linspace(-1.0, 1.0, wave.size)) * transmission_from_basis(
        species, basis, species_scales={"H2O": 1.2}
    )
    errors = np.full(wave.size, 0.01)
    excluded = (1.001, 1.0015)
    result = correct_arrays(
        wave,
        flux,
        uncertainty=errors,
        line_list=lines,
        physical=False,
        continuum_order=1,
        lsf_sigma_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_variable_width=False,
        fit_wavelength_shift=False,
        high_resolution_grid=False,
        auto_segment=False,
        exclude_ranges=(excluded,),
        estimate_uncertainties=False,
    )
    assert result.success
    expected = (wave < excluded[0]) | (wave > excluded[1])
    np.testing.assert_array_equal(result.fit_mask, expected)
    np.testing.assert_array_equal(result.spectrum.uncertainty, errors)
    np.testing.assert_allclose(result.model_flux, flux, atol=1.0e-5)
