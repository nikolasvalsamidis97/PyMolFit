from dataclasses import replace

import numpy as np
import pytest

from pymolfit import FitConfig, LineList, Spectrum, fit_telluric_segments, fit_tellurics
from pymolfit.fit import _shift_basis
from pymolfit.model import optical_depth_basis, transmission_from_basis


def _group_spectra(coefficients=((5.0,), (7.0,))):
    spectra, catalogs = [], []
    for group, values in enumerate(coefficients):
        wavelength = np.linspace(1.0 + 0.02 * group, 1.01 + 0.02 * group, 500)
        lines = LineList(
            wavelength=np.linspace(wavelength[70], wavelength[-70], 5),
            strength=np.full(5, 3.0e-5),
            sigma=np.full(5, 3.0e-5),
            gamma=np.full(5, 5.0e-6),
            species=np.full(5, "H2O"),
        )
        species, basis = optical_depth_basis(wavelength, lines)
        x = np.linspace(-1.0, 1.0, wavelength.size)
        shift = np.polynomial.polynomial.polyval(x, values)
        flux = transmission_from_basis(
            species,
            _shift_basis(wavelength, basis, shift * np.gradient(wavelength)),
            species_scales={"H2O": 1.2},
        )
        spectra.append(Spectrum(wavelength, flux, uncertainty=np.full(wavelength.size, 0.002)))
        catalogs.append(lines)
    combined = LineList(
        **{
            field: np.concatenate([getattr(lines, field) for lines in catalogs])
            for field in ("wavelength", "strength", "sigma", "gamma", "species")
        }
    )
    return spectra, combined


def _config(**changes):
    return replace(
        FitConfig(
            continuum_order=0,
            solve_continuum_linear=True,
            fixed_species_scales={"H2O": 1.2},
            fit_segment_wavelength_shifts=True,
            wavelength_shift_unit="pixel",
            wavelength_shift_bounds=(-12.0, 12.0),
            use_jacobian_sparsity=False,
        ),
        **changes,
    )


@pytest.mark.parametrize("initialize_grid", [True, False])
def test_one_groups_expanded_bounds_do_not_loosen_another_group(initialize_grid):
    spectra, lines = _group_spectra()
    result = fit_telluric_segments(
        spectra,
        line_list=lines,
        wavelength_group_ids=(10, 20),
        config=_config(
            initial_wavelength_shift=9.0,
            initialize_wavelength_shift_grid=initialize_grid,
            wavelength_group_coefficient_bounds={10: (-3.0, 3.0), 20: (-12.0, 12.0)},
        ),
    )

    assert result.success
    np.testing.assert_allclose(result.wavelength_group_coefficients[10], [3.0], atol=0.01)
    np.testing.assert_allclose(result.wavelength_group_coefficients[20], [7.0], atol=0.01)
    if initialize_grid:
        search = result.provenance["segment_wavelength_coarse_search"]
        assert search[0]["grid"][0] == -3.0
        assert search[0]["grid"][-1] == 3.0
        assert search[1]["grid"][0] == -12.0
        assert search[1]["grid"][-1] == 12.0


def test_polynomial_groups_keep_their_own_bounds_for_all_coefficients():
    spectra, lines = _group_spectra(((4.0, 0.7, 0.5), (7.0, 0.8, 0.6)))
    result = fit_telluric_segments(
        spectra,
        line_list=lines,
        wavelength_group_ids=(10, 20),
        config=_config(
            fit_segment_wavelength_shifts=False,
            fit_segment_wavelength_polynomial=True,
            segment_wavelength_polynomial_order=2,
            wavelength_group_coefficient_bounds={10: (-3.0, 3.0)},
        ),
    )

    assert result.success
    assert np.all(np.abs(result.wavelength_group_coefficients[10]) <= 3.0)
    assert result.wavelength_group_coefficients[10][0] == pytest.approx(3.0, abs=0.01)
    # An unspecified group inherits the conservative global bounds.
    np.testing.assert_allclose(result.wavelength_group_coefficients[20], [7.0, 0.8, 0.6], atol=0.01)


def test_omitted_and_uniform_group_bounds_produce_identical_fits():
    spectra, lines = _group_spectra()
    options = {"spectra": spectra, "line_list": lines, "wavelength_group_ids": (10, 20)}
    original = fit_telluric_segments(**options, config=_config())
    explicit = fit_telluric_segments(
        **options,
        config=_config(wavelength_group_coefficient_bounds={10: (-12.0, 12.0), 20: (-12.0, 12.0)}),
    )
    assert original.success and explicit.success
    for group in (10, 20):
        np.testing.assert_array_equal(
            original.wavelength_group_coefficients[group],
            explicit.wavelength_group_coefficients[group],
        )
    for first, second in zip(original.segment_results, explicit.segment_results, strict=True):
        np.testing.assert_array_equal(first.model_flux, second.model_flux)


@pytest.mark.parametrize(
    "bounds",
    [(-np.inf, 3.0), (-3.0, np.nan), (3.0, 3.0), (3.0, -3.0), (1.0,), None],
)
def test_invalid_group_coefficient_bounds_are_rejected(bounds):
    spectra, lines = _group_spectra()
    with pytest.raises(ValueError, match="finite and increasing pairs"):
        fit_telluric_segments(
            spectra,
            line_list=lines,
            wavelength_group_ids=(10, 20),
            config=_config(wavelength_group_coefficient_bounds={10: bounds}),
        )


@pytest.mark.parametrize("group", [30, "10", 10.0, True])
def test_group_coefficient_bounds_require_existing_integer_group_ids(group):
    spectra, lines = _group_spectra()
    with pytest.raises(ValueError, match="existing integer group IDs"):
        fit_telluric_segments(
            spectra,
            line_list=lines,
            wavelength_group_ids=(10, 20),
            config=_config(wavelength_group_coefficient_bounds={group: (-3.0, 3.0)}),
        )


def test_nonfitting_group_can_have_bounds_without_creating_parameters():
    spectra, lines = _group_spectra(((0.0,), (7.0,)))
    result = fit_telluric_segments(
        spectra,
        line_list=lines,
        wavelength_group_ids=(10, 20),
        config=_config(
            wavelength_group_polynomial_orders={10: -1},
            wavelength_group_coefficient_bounds={10: (1.0, 3.0), 20: (-12.0, 12.0)},
        ),
    )
    assert result.success
    assert not any("group:10:" in name for name in result.parameter_names)
    np.testing.assert_array_equal(result.wavelength_group_coefficients[10], [0.0])
    np.testing.assert_allclose(result.wavelength_group_coefficients[20], [7.0], atol=0.01)


def test_group_bounds_require_group_fitting():
    spectra, lines = _group_spectra()
    with pytest.raises(ValueError, match="require per-group wavelength fitting"):
        fit_telluric_segments(
            spectra,
            line_list=lines,
            wavelength_group_ids=(10, 20),
            config=_config(
                fit_segment_wavelength_shifts=False,
                wavelength_group_coefficient_bounds={10: (-3.0, 3.0)},
            ),
        )


def test_single_spectrum_api_rejects_group_bounds():
    spectra, lines = _group_spectra()
    with pytest.raises(ValueError, match="only supported by fit_telluric_segments"):
        fit_tellurics(
            spectra[0],
            line_list=lines,
            config=_config(
                fit_segment_wavelength_shifts=False,
                wavelength_group_coefficient_bounds={10: (-3.0, 3.0)},
            ),
        )
