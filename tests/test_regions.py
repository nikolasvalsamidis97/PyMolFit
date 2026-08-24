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


def test_automatic_region_default_includes_line_wings_and_continuum() -> None:
    from pymolfit.regions import DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS

    assert DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS == 12.0


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
    assert loaded.output_path == path


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
    assert reused.output_path == tmp_path / "regions.ecsv"


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
    assert selector.output_path == path
    assert selector.selection.output_path == path
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
    selector.axis.set_xlim(15_999.0, 16_026.0)

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


def test_extrema_preserving_display_keeps_narrow_absorption_and_emission() -> None:
    from pymolfit.regions import _extrema_preserving_indices

    wavelength = np.linspace(5000.0, 5100.0, 20_000)
    flux = np.ones_like(wavelength)
    flux[1234] = -4.0
    flux[15_678] = 7.0

    selected = _extrema_preserving_indices(
        wavelength,
        flux,
        max_points=400,
    )

    assert selected.size <= 400
    assert 1234 in selected
    assert 15_678 in selected


def test_interactive_selector_adapts_spectrum_to_viewport() -> None:
    wavelength = np.linspace(5000.0, 5100.0, 100_001)
    flux = 1.0 + 0.01 * np.sin(np.linspace(0.0, 30.0 * np.pi, wavelength.size))
    flux[50_000] = 0.2
    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=flux,
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show_telluric_lines=False,
        show=False,
    )

    overview_wavelength = np.asarray(selector._spectrum_artist.get_xdata())
    overview_flux = np.asarray(selector._spectrum_artist.get_ydata())
    np.testing.assert_array_equal(selector.spectrum.wavelength, wavelength)
    np.testing.assert_array_equal(selector.spectrum.flux, flux)
    assert overview_wavelength.size < wavelength.size
    assert 0.2 in overview_flux

    selector.axis.set_xlim(5049.95, 5050.05)
    detailed_wavelength = np.asarray(selector._spectrum_artist.get_xdata())
    expected = wavelength[(wavelength >= 5049.95) & (wavelength <= 5050.05)]
    assert np.all(np.isin(expected, detailed_wavelength))

    selector.axis.set_xlim(5070.0, 5070.1)
    panned_wavelength = np.asarray(selector._spectrum_artist.get_xdata())
    assert not np.array_equal(detailed_wavelength, panned_wavelength)
    assert np.nanmin(panned_wavelength) < 5070.0
    assert np.nanmax(panned_wavelength) > 5070.1
    selector.close()


def test_interactive_selector_batches_and_filters_region_artists(tmp_path) -> None:
    wavelength = np.linspace(0.0, 100.0, 10_001)
    fit_ranges = tuple((float(index), float(index) + 0.1) for index in range(100))
    exclude_ranges = tuple((float(index) + 0.2, float(index) + 0.3) for index in range(100))
    initial = RegionSelection(
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )
    destination = tmp_path / "all_regions.ecsv"
    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        initial_regions=initial,
        output_path=destination,
        show_telluric_lines=False,
        show=False,
    )

    assert len(selector.selection.fit_ranges) == 100
    assert len(selector.selection.exclude_ranges) == 100
    assert selector._visible_region_count == 200
    assert len(selector._patches) == 2
    assert selector._region_labels == []

    cached_selection = selector.selection
    selector.axis.set_xlim(40.0, 42.0)
    visible_count = sum(
        lower <= 42.0 and upper >= 40.0
        for lower, upper in selector.selection.fit_ranges + selector.selection.exclude_ranges
    )
    rendered_count = sum(len(collection.get_paths()) for collection in selector._patches)
    assert selector.selection is cached_selection
    assert selector._visible_region_count == visible_count
    assert rendered_count == visible_count
    assert len(selector._region_labels) == visible_count

    assert selector.save() == destination
    saved = load_region_file(destination)
    assert len(saved.fit_ranges) == 100
    assert len(saved.exclude_ranges) == 100
    selector.close()


def test_interactive_selector_hides_then_progressively_reveals_aer_markers(
    monkeypatch,
) -> None:
    wavelength = np.linspace(5000.0, 5100.0, 101)
    marker_wavelength = np.linspace(5000.0, 5100.0, 1001)
    marker_species = np.where(np.arange(marker_wavelength.size) % 2, "H2O", "O2")
    marker_strength = np.linspace(1.0, 2.0, marker_wavelength.size)

    def markers(_spectrum, *, max_lines):
        assert max_lines is None
        return marker_wavelength, marker_species, marker_strength

    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        markers,
    )

    selector = select_telluric_regions(
        wavelength=wavelength,
        flux=np.ones_like(wavelength),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show=False,
    )

    assert selector._telluric_marker_count == 1001
    assert selector._visible_telluric_marker_count == 0
    assert selector._telluric_marker_artists == []

    selector.axis.set_xlim(5000.0, 5060.0)
    medium_count = selector._visible_telluric_marker_count
    assert 0 < medium_count < 601

    selector.axis.set_xlim(5030.0, 5050.0)
    narrow_count = selector._visible_telluric_marker_count
    assert narrow_count > medium_count

    selector.axis.set_xlim(5049.0, 5050.0)
    expected_count = np.count_nonzero((marker_wavelength >= 5049.0) & (marker_wavelength <= 5050.0))
    assert selector._visible_telluric_marker_count == expected_count
    rendered_count = sum(len(artist.get_segments()) for artist in selector._telluric_marker_artists)
    assert rendered_count == expected_count
    selector.close()


def test_interactive_selector_loads_all_markers_for_adaptive_display(
    monkeypatch,
) -> None:
    requested: list[int | None] = []

    def markers(_spectrum, *, max_lines):
        requested.append(max_lines)
        return np.array([5050.0]), np.array(["H2O"]), np.array([1.0])

    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        markers,
    )
    selector = select_telluric_regions(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        show=False,
    )

    assert requested == [None]
    assert selector._telluric_overview_count == 1
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


def test_automatic_fit_regions_compares_species_by_expected_absorption(
    monkeypatch,
) -> None:
    from pymolfit.regions import _automatic_fit_regions

    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        lambda _spectrum, *, max_lines: (
            np.array([5010.0, 5030.0]),
            np.array(["CO2", "CH4"]),
            # CH4 is stronger in the catalogue, but its atmospheric column is
            # much smaller than CO2's, so the CO2 line has greater expected
            # integrated telluric absorption.
            np.array([1.0, 100.0]),
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
        count=1,
        max_lines=100,
        half_width_pixels=2.0,
    )

    assert regions == ((5008.0, 5012.0),)


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
    wavelength = np.concatenate((np.arange(5000.0, 5011.0), np.arange(5100.0, 5111.0)))
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
    flux = 2.0e-12 + 0.4e-12 * np.sin(np.linspace(0.0, 2.0 * np.pi, wavelength.size))
    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        lambda _spectrum, *, max_lines: (
            np.array([5010.0, 5050.0]),
            np.array(["H2O", "O2"]),
            np.array([1.0, 2.0]),
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
