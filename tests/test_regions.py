from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymolfit import (
    RegionSelection,
    Spectrum,
    correct_arrays,
    load_region_file,
    save_region_file,
    select_telluric_regions,
)
from pymolfit.spectrum import air_to_vacuum_wavelength


def test_region_selection_normalizes_and_merges_ranges() -> None:
    selection = RegionSelection(
        fit_ranges=((5910.0, 5900.0), (5905.0, 5920.0)),
        exclude_ranges=((5908.0, 5909.0),),
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )

    assert selection.fit_ranges == ((5900.0, 5920.0),)
    assert selection.exclude_ranges == ((5908.0, 5909.0),)


def test_region_file_round_trip_preserves_coordinates(tmp_path) -> None:
    selection = RegionSelection(
        fit_ranges=((5900.0, 5920.0),),
        exclude_ranges=((5908.0, 5909.0),),
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )
    path = save_region_file(selection, tmp_path / "regions.ecsv")

    loaded = load_region_file(path)

    assert loaded == selection


def test_selector_reuses_existing_output_file_without_opening_window(
    tmp_path,
) -> None:
    existing = RegionSelection(
        fit_ranges=((5005.0, 5010.0),),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )
    save_region_file(existing, tmp_path / "regions.ecsv")

    reused = select_telluric_regions(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        output_path=tmp_path / "regions",
        show=False,
    )

    assert isinstance(reused, RegionSelection)
    assert reused == existing


def test_selector_can_open_existing_output_file_for_editing(tmp_path) -> None:
    existing = RegionSelection(
        fit_ranges=((5005.0, 5010.0),),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )
    path = save_region_file(existing, tmp_path / "regions.ecsv")

    selector = select_telluric_regions(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        output_path=path,
        reuse_existing=False,
        show_telluric_lines=False,
        show=False,
    )

    assert not isinstance(selector, RegionSelection)
    assert selector.selection == existing
    selector.close()


def test_region_selection_converts_unit_and_medium() -> None:
    selection = RegionSelection(
        fit_ranges=((5900.0, 5920.0),),
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )

    converted = selection.converted(
        wavelength_unit="micron",
        wavelength_medium="vacuum",
    )
    expected = air_to_vacuum_wavelength(
        np.array([0.5900, 0.5920]),
        unit="micron",
    )

    np.testing.assert_allclose(converted.fit_ranges[0], expected)
    assert converted.wavelength_unit == "micron"
    assert converted.wavelength_medium == "vacuum"


def test_interactive_selector_add_delete_undo_and_save(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    spectrum = Spectrum(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )
    path = tmp_path / "selected.ecsv"
    selector = select_telluric_regions(
        spectrum,
        output_path=path,
        show_telluric_lines=False,
        show=False,
    )
    selector.add_region(5010.0, 5020.0, kind="fit")
    selector.add_region(5030.0, 5040.0, kind="exclude")
    selector.delete_regions(5035.0, 5036.0)
    assert selector.selection.exclude_ranges == ()

    selector.undo()
    assert selector.selection.exclude_ranges == ((5030.0, 5040.0),)
    assert selector.save() == path
    assert load_region_file(path) == selector.selection
    selector.close()


def test_interactive_selector_preserves_spectrum_wavelength_limits() -> None:
    wavelength = np.linspace(15_984.0, 16_198.0, 100)
    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show_telluric_lines=False,
        show=False,
    )

    lower, upper = selector.axis.get_xlim()

    assert lower > 15_900.0
    assert upper < 16_300.0
    assert lower < wavelength.min()
    assert upper > wavelength.max()
    selector.close()


def test_interactive_selector_marks_and_numbers_visible_regions() -> None:
    wavelength = np.linspace(15_984.0, 16_198.0, 100)
    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show_telluric_lines=False,
        show=False,
    )
    selector.axis.set_xlim(16_000.0, 16_010.0)
    selector.mark_visible_region(kind="fit")
    selector.axis.set_xlim(16_020.0, 16_025.0)
    selector.mark_visible_region(kind="exclude")

    assert selector.selection.fit_ranges == ((16_000.0, 16_010.0),)
    assert selector.selection.exclude_ranges == ((16_020.0, 16_025.0),)
    assert [label.get_text() for label in selector._region_labels] == ["R1", "R2"]
    assert "Regions in memory: 2" in selector.status_text.get_text()
    assert "R1 fit:" in selector.status_text.get_text()
    assert "R2 exclude:" in selector.status_text.get_text()
    selector.close()


def test_interactive_selector_adds_rectangle_wavelength_interval() -> None:
    wavelength = np.linspace(5000.0, 5100.0, 101)
    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show_telluric_lines=False,
        show=False,
    )

    assert not selector.rectangle_selector.active
    selector.draw_checkbox.set_active(0)
    assert selector.rectangle_selector.active
    selector._on_rectangle(
        SimpleNamespace(xdata=5012.0, ydata=0.95),
        SimpleNamespace(xdata=5024.0, ydata=1.05),
    )
    selector._on_rectangle(
        SimpleNamespace(xdata=5030.0, ydata=0.95),
        SimpleNamespace(xdata=5036.0, ydata=1.05),
    )

    assert selector.selection.fit_ranges == (
        (5012.0, 5024.0),
        (5030.0, 5036.0),
    )
    assert selector.rectangle_selector.active
    selector.draw_checkbox.set_active(0)
    assert not selector.rectangle_selector.active
    selector.close()


def test_interactive_selector_edits_output_filename(tmp_path) -> None:
    wavelength = np.linspace(5000.0, 5100.0, 101)
    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        output_path=tmp_path / "initial.ecsv",
        show_telluric_lines=False,
        show=False,
    )
    selector.add_region(5010.0, 5020.0)
    selector.save_name_box.set_val("renamed_regions")

    written = selector.save()

    assert written == tmp_path / "renamed_regions.ecsv"
    assert load_region_file(written) == selector.selection
    selector.close()


def test_interactive_selector_shows_aer_markers_by_default(monkeypatch) -> None:
    wavelength = np.linspace(5000.0, 5100.0, 101)
    marker_wavelength = np.array([5010.0, 5020.0, 5080.0])
    marker_species = np.array(["H2O", "O2", "H2O"])
    monkeypatch.setattr(
        "pymolfit.regions._aer_markers_for_spectrum",
        lambda _spectrum, *, max_lines: (
            marker_wavelength,
            marker_species,
        ),
    )

    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show=False,
    )

    labels = [collection.get_label() for collection in selector.axis.collections]
    assert selector._telluric_marker_count == 3
    assert labels == ["AER H2O", "AER O2"]
    selector.close()


def test_interactive_selector_requests_expanded_default_marker_limit(
    monkeypatch,
) -> None:
    requested: list[int] = []

    def markers(_spectrum, *, max_lines):
        requested.append(max_lines)
        return np.array([5050.0]), np.array(["H2O"])

    monkeypatch.setattr(
        "pymolfit.regions._aer_markers_for_spectrum",
        markers,
    )
    selector = select_telluric_regions(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show=False,
    )

    assert requested == [10_000]
    selector.close()


def test_automatic_fit_regions_selects_strongest_covered_lines(
    monkeypatch,
) -> None:
    from pymolfit.regions import _automatic_fit_regions

    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        lambda _spectrum, *, max_lines: (
            np.array([5010.0, 5030.0, 5080.0]),
            np.array(["H2O", "O2", "H2O"]),
            np.array([1.0, 10.0, 5.0]),
        ),
    )
    spectrum = Spectrum(
        wavelength=np.arange(5000.0, 5101.0),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )

    regions = _automatic_fit_regions(
        spectrum,
        count=2,
        max_lines=100,
        half_width_pixels=2.0,
    )

    assert regions == ((5028.0, 5032.0), (5078.0, 5082.0))


def test_automatic_fit_regions_skips_catalogue_lines_in_spectral_gaps(
    monkeypatch,
) -> None:
    from pymolfit.regions import _automatic_fit_regions

    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        lambda _spectrum, *, max_lines: (
            np.array([5005.0, 5050.0]),
            np.array(["H2O", "O2"]),
            np.array([10.0, 100.0]),
        ),
    )
    wavelength = np.concatenate(
        (np.arange(5000.0, 5011.0), np.arange(5100.0, 5111.0))
    )
    spectrum = Spectrum(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )

    regions = _automatic_fit_regions(
        spectrum,
        count=1,
        max_lines=100,
        half_width_pixels=2.0,
    )

    assert regions == ((5003.0, 5007.0),)


def test_interactive_selector_adds_automatic_regions_as_one_undoable_edit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pymolfit.regions._automatic_fit_regions",
        lambda _spectrum, *, count, max_lines, half_width_pixels: (
            (5010.0, 5020.0),
            (5030.0, 5040.0),
        ),
    )
    selector = select_telluric_regions(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show_telluric_lines=False,
        show=False,
    )
    selector.auto_count_box.set_val("2")

    added = selector.add_automatic_fit_regions()

    assert added == ((5010.0, 5020.0), (5030.0, 5040.0))
    assert selector.selection.fit_ranges == added
    assert "Automatic: 2 windows / 2 lines" in selector.status_text.get_text()
    selector.undo()
    assert selector.selection.is_empty
    selector.close()


def test_interactive_selector_markers_preserve_small_flux_limits(
    monkeypatch,
) -> None:
    wavelength = np.linspace(5000.0, 5100.0, 101)
    flux = 2.0e-12 + 0.4e-12 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, wavelength.size)
    )
    monkeypatch.setattr(
        "pymolfit.regions._aer_markers_for_spectrum",
        lambda _spectrum, *, max_lines: (
            np.array([5010.0, 5050.0]),
            np.array(["H2O", "O2"]),
        ),
    )

    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=flux,
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show=False,
    )

    lower, upper = selector.axis.get_ylim()
    assert 0.0 < lower < float(np.nanmin(flux))
    assert float(np.nanmax(flux)) < upper < 1.0e-11
    selector.close()


def test_correct_arrays_region_file_matches_explicit_ranges(tmp_path) -> None:
    wavelength = np.linspace(2.31, 2.36, 300)
    flux = np.ones_like(wavelength)
    explicit_fit = ((2.315, 2.355),)
    explicit_exclude = ((2.330, 2.332),)
    region_file = save_region_file(
        RegionSelection(
            fit_ranges=((23_150.0, 23_550.0),),
            exclude_ranges=((23_300.0, 23_320.0),),
            wavelength_unit="angstrom",
            wavelength_medium="vacuum",
        ),
        tmp_path / "regions.ecsv",
    )
    options = {
        "demo_line_list": True,
        "physical": False,
        "continuum_order": 0,
        "solve_continuum_linear": True,
        "lsf_sigma_pixels": 0.0,
        "fit_lsf_sigma": False,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "fit_lsf_lorentz_fwhm": False,
        "lsf_variable_width": False,
        "high_resolution_grid": False,
        "fit_wavelength_shift": False,
        "auto_segment": False,
    }

    from_file = correct_arrays(
        wavelength,
        flux,
        wavelength_medium="vacuum",
        region_file=region_file,
        **options,
    )
    explicit = correct_arrays(
        wavelength,
        flux,
        wavelength_medium="vacuum",
        fit_ranges=explicit_fit,
        exclude_ranges=explicit_exclude,
        **options,
    )

    np.testing.assert_allclose(from_file.model_flux, explicit.model_flux)
    np.testing.assert_allclose(from_file.transmission, explicit.transmission)


def test_region_file_rejects_explicit_ranges(tmp_path) -> None:
    region_file = save_region_file(
        RegionSelection(
            fit_ranges=((2.31, 2.36),),
        ),
        tmp_path / "regions.ecsv",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        correct_arrays(
            np.linspace(2.31, 2.36, 20),
            np.ones(20),
            region_file=region_file,
            fit_ranges=((2.31, 2.36),),
        )
