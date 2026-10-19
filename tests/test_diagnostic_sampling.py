"""Residual alignment is measured in native pixels, including excluded pixels."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymolfit import Spectrum
from pymolfit.diagnostics import fit_quality_diagnostics
from pymolfit.spectrum import correct_spectrum


def _shifted_result(shift, selected, groups=False):
    pixel = np.arange(selected.size, dtype=float)
    optical_depth = np.zeros(selected.size)
    for index, center in enumerate(range(20, selected.size - 20, 27)):
        optical_depth += (0.2 + 0.03 * (index % 5)) * np.exp(-0.5 * ((pixel - center) / 1.7) ** 2)
    transmission = np.exp(-optical_depth)
    flux = np.interp(pixel, pixel + shift, transmission, left=1, right=1)
    wave = 1 + pixel * 1e-5
    labels = None
    if groups:
        # Two overlapping detector orders interleave in wavelength, but each
        # has its own native pixel coordinate and independent alignment.
        wave = np.column_stack([wave, wave + 0.3e-5]).ravel()
        flux = np.repeat(flux, 2)
        transmission = np.repeat(transmission, 2)
        selected = np.repeat(selected, 2)
        labels = np.tile([7, 12], pixel.size)
    spectrum = Spectrum(
        wave,
        flux,
        uncertainty=np.full(wave.size, 0.01),
        group_id=labels,
        wavelength_unit="micron",
        wavelength_medium="vacuum",
    )
    return SimpleNamespace(
        spectrum=spectrum,
        corrected=correct_spectrum(spectrum, transmission),
        continuum=np.ones(wave.size),
        transmission=transmission,
        model_flux=transmission.copy(),
        fit_mask=selected,
        parameter_bound_status={},
        provenance={},
        success=True,
    )


@pytest.mark.parametrize("mask_kind", ["dense", "stride", "blocked", "irregular"])
@pytest.mark.parametrize("shift", [-0.6, -0.4, 0.0, 0.4, 0.6])
@pytest.mark.parametrize("groups", [False, True])
def test_preserves_native_pixels_across_fit_exclusions(mask_kind, shift, groups):
    pixels = np.arange(320)
    if mask_kind == "dense":
        selected = np.ones(pixels.size, dtype=bool)
    elif mask_kind == "stride":
        selected = pixels % 3 == 0
    elif mask_kind == "blocked":
        selected = ~(((pixels > 70) & (pixels < 110)) | ((pixels > 170) & (pixels < 230)))
    else:
        selected = np.random.default_rng(41).random(pixels.size) > 0.45
    result = _shifted_result(shift, selected, groups)
    before = {
        name: getattr(result, name).copy()
        for name in ("continuum", "transmission", "fit_mask", "model_flux")
    }
    measured = fit_quality_diagnostics(result)
    assert measured["measured_group_count"] == (2 if groups else 1)
    for row in measured["groups"]:
        assert row["status"] == "measured"
        assert row["residual_shift_pixels"] == pytest.approx(shift)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(result, name), value)


def test_nonfinite_model_outside_fit_mask_does_not_compress_grid():
    selected = np.arange(320) % 3 == 0
    result = _shifted_result(-0.4, selected)
    result.transmission[40:50] = np.nan
    rows = fit_quality_diagnostics(result)["groups"]
    assert rows[0]["status"] == "measured"
    assert rows[0]["residual_shift_pixels"] == pytest.approx(-0.4)


@pytest.mark.parametrize("ordering", ["reverse", "permuted"])
def test_group_ordering_does_not_change_native_shift(ordering):
    result = _shifted_result(0.6, np.arange(320) % 3 == 0, groups=True)
    size = result.spectrum.wavelength.size
    indices = (
        np.arange(size)[::-1]
        if ordering == "reverse"
        else np.random.default_rng(7).permutation(size)
    )
    # An unsorted result-like object exercises diagnostic sorting directly.
    for name in ("spectrum", "corrected"):
        spectrum = getattr(result, name)
        setattr(
            result,
            name,
            SimpleNamespace(
                **{
                    key: getattr(spectrum, key)[indices]
                    for key in ("wavelength", "flux", "valid", "group_id")
                }
            ),
        )
    for name in ("continuum", "transmission", "fit_mask", "model_flux"):
        setattr(result, name, getattr(result, name)[indices])
    rows = fit_quality_diagnostics(result)["groups"]
    assert len(rows) == 2
    assert all(row["residual_shift_pixels"] == pytest.approx(0.6) for row in rows)


def test_too_few_telluric_pixels_remains_unmeasured():
    result = _shifted_result(0.4, np.arange(320) < 10)
    row = fit_quality_diagnostics(result)["groups"][0]
    assert row["status"] == "insufficient_telluric_pixels"
