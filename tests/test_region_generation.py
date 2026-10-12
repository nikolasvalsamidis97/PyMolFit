from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from pymolfit import (
    Observation,
    RegionSelection,
    Spectrum,
    load_region_file,
    select_telluric_regions,
)
from pymolfit.errors import WavelengthMetadataError


@pytest.fixture
def spectrum():
    return Spectrum(
        wavelength=np.linspace(5000.0, 5100.0, 101),
        flux=np.ones(101),
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
    )


@pytest.fixture
def catalog(monkeypatch):
    calls = []

    def load(spectrum, *, max_lines):
        calls.append((spectrum, max_lines))
        return (
            np.array([5010.0, 5030.0, 5080.0]),
            np.array(["H2O", "H2O", "H2O"]),
            np.array([1.0, 3.0, 2.0]),
        )

    monkeypatch.setattr("pymolfit.regions._aer_catalog_for_spectrum", load)
    return calls


@pytest.mark.parametrize("options", [{}, {"show_interactive_window": False}])
def test_generation_is_headless_without_matplotlib(spectrum, catalog, monkeypatch, options):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise AssertionError("headless region generation must not import Matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    selector = select_telluric_regions(spectrum, **options)

    assert selector.figure is None
    assert selector.axis is None
    assert selector.selection.fit_ranges == ((5000.0, 5042.0), (5068.0, 5092.0))
    assert selector.selection.exclude_ranges == ()
    assert len(catalog) == 1
    np.testing.assert_array_equal(selector.spectrum.flux, spectrum.flux)
    selector.close()


def test_requested_count_and_width_generate_and_save_current_regions(spectrum, catalog, tmp_path):
    selector = select_telluric_regions(
        spectrum,
        automatic_region_count=2,
        automatic_region_half_width_pixels=2,
        output_path=tmp_path / "regions",
    )
    assert selector.selection.fit_ranges == ((5028.0, 5032.0), (5078.0, 5082.0))
    assert not (tmp_path / "regions.ecsv").exists()
    selector.add_region(5029.0, 5030.0, kind="exclude")
    written = selector.save()
    assert written == tmp_path / "regions.ecsv"
    assert load_region_file(written) == selector.selection
    selector.clear()
    assert selector.selection.is_empty
    selector.undo()
    assert selector.selection.exclude_ranges == ((5029.0, 5030.0),)
    assert selector.save(tmp_path / "copy.ecsv") == tmp_path / "copy.ecsv"
    selector.close()


def test_headless_add_automatic_uses_requested_count_without_widgets(spectrum, catalog):
    selector = select_telluric_regions(
        spectrum, automatic_region_count=1, automatic_region_half_width_pixels=2
    )
    selector.clear()
    assert selector.add_automatic_fit_regions() == ((5028.0, 5032.0),)
    assert selector.selection.fit_ranges == ((5028.0, 5032.0),)


@pytest.mark.parametrize("options", [{"show_interactive_window": True}])
def test_window_opens_with_automatic_regions_already_drawn(
    spectrum, catalog, monkeypatch, tmp_path, options
):
    import matplotlib.pyplot as plt

    displayed_regions = []

    def show():
        axis = plt.gcf().axes[0]
        displayed_regions.append(sum(len(artist.get_paths()) for artist in axis.collections))

    monkeypatch.setattr(plt, "show", show)
    selector = select_telluric_regions(
        spectrum,
        automatic_region_count=2,
        automatic_region_half_width_pixels=2,
        show_telluric_lines=False,
        **options,
    )
    try:
        assert displayed_regions == [2]
        assert selector.auto_count_box.text == "2"
        selector.add_region(5029.0, 5030.0, kind="exclude")
        saved = load_region_file(selector.save(tmp_path / "edited.ecsv"))
        assert saved == selector.selection
        assert saved.exclude_ranges == ((5029.0, 5030.0),)
    finally:
        selector.close()


def test_reusing_saved_regions_does_not_generate_new_ones(spectrum, catalog, tmp_path):
    path = tmp_path / "saved.ecsv"
    saved = RegionSelection(
        fit_ranges=((5011.0, 5012.0),),
        exclude_ranges=((5011.5, 5011.8),),
        wavelength_unit="angstrom",
    )
    saved.save(path)
    reused = select_telluric_regions(spectrum, output_path=path)
    assert reused == saved
    assert catalog == []
    assert reused.save() == path
    copied = reused.save(tmp_path / "copy.ecsv")
    assert load_region_file(copied) == saved


def test_window_can_edit_existing_file_without_regenerating(
    spectrum, catalog, monkeypatch, tmp_path
):
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    path = tmp_path / "saved.ecsv"
    saved = RegionSelection(fit_ranges=((5011.0, 5012.0),), wavelength_unit="angstrom")
    saved.save(path)
    selector = select_telluric_regions(
        spectrum, output_path=path, show_interactive_window=True, show_telluric_lines=False
    )
    try:
        assert selector.selection == saved
        assert shown == [True]
        assert catalog == []
        selector.add_region(5040.0, 5041.0)
        assert load_region_file(path) == saved
        selector.save()
        assert load_region_file(path) == selector.selection
    finally:
        selector.close()


def test_explicit_initial_regions_are_preserved(spectrum, catalog):
    initial = RegionSelection(fit_ranges=((5011.0, 5012.0),), wavelength_unit="angstrom")
    selector = select_telluric_regions(spectrum, initial_regions=initial)
    assert selector.selection == initial
    assert catalog == []


def test_automatic_regions_use_requested_observatory_frame(spectrum, catalog, tmp_path):
    velocity = 25.0
    spectrum.meta["observation"] = Observation(
        wavelength_frame="barycentric", frame_velocity_km_s=velocity
    ).to_header()
    selector = select_telluric_regions(spectrum, wavelength_frame="observatory")
    factor = (1.0 + 1.55e-8) * (1.0 + velocity / 299_792.458)
    np.testing.assert_allclose(catalog[0][0].wavelength, spectrum.wavelength / factor)
    np.testing.assert_array_equal(selector.spectrum.wavelength, catalog[0][0].wavelength)
    saved = load_region_file(selector.save(tmp_path / "observer.ecsv"))
    assert saved.wavelength_frame == "observatory"


def test_headless_generation_retains_optional_stellar_exclusions(spectrum, catalog, monkeypatch):
    template = object()
    exclusions = RegionSelection(exclude_ranges=((5029.0, 5030.0),), wavelength_unit="angstrom")
    mask = SimpleNamespace(selection_for_spectrum=lambda _spectrum: exclusions)
    monkeypatch.setattr("pymolfit.regions._prepare_stellar_mask_for_selector", lambda *_: mask)
    selector = select_telluric_regions(spectrum, theoretical_spectrum=template)
    assert selector.selection.fit_ranges
    assert selector.selection.exclude_ranges == exclusions.exclude_ranges
    assert selector.theoretical_spectrum is template
    assert selector.stellar_mask_result is mask


@pytest.mark.parametrize(
    "options",
    [
        {"automatic_region_count": 0},
        {"automatic_region_count": -1},
        {"automatic_region_count": 2.5},
        {"automatic_region_half_width_pixels": 0},
        {"automatic_region_half_width_pixels": np.nan},
    ],
)
def test_invalid_options_fail_before_catalog_or_window_creation(spectrum, catalog, options):
    with pytest.raises(ValueError):
        select_telluric_regions(spectrum, **options)
    assert catalog == []


def test_no_usable_lines_raise_instead_of_silently_saving_empty_regions(spectrum, monkeypatch):
    monkeypatch.setattr("pymolfit.regions._automatic_fit_regions", lambda *_, **__: ())
    with pytest.raises(ValueError, match="no covered positive-strength AER transitions"):
        select_telluric_regions(spectrum)


def test_save_requires_destination(spectrum, catalog):
    selector = select_telluric_regions(spectrum)
    with pytest.raises(ValueError, match="call save"):
        selector.save()
    with pytest.raises(ValueError, match="call save"):
        RegionSelection().save()


@pytest.mark.parametrize("path_type", [str, Path])
@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_selector_loads_fits_path_and_saves_fit_and_exclusion_regions(
    spectrum, catalog, tmp_path, monkeypatch, path_type, medium
):
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVE", format="D", unit="Angstrom", array=spectrum.wavelength),
            fits.Column(name="FLUX", format="D", array=spectrum.flux),
            fits.Column(name="ERR", format="D", array=np.full(101, 0.02)),
        ]
    )
    table.header["TUCD1"] = "em.wl;obs.atmos" if medium == "air" else "em.wl"
    path = tmp_path / "spectrum.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    template = object()
    exclusions = RegionSelection(
        exclude_ranges=((5029.0, 5030.0),),
        wavelength_unit="angstrom",
        wavelength_medium=medium,
    )
    mask = SimpleNamespace(selection_for_spectrum=lambda _spectrum: exclusions)
    monkeypatch.setattr("pymolfit.regions._prepare_stellar_mask_for_selector", lambda *_: mask)

    selector = select_telluric_regions(
        path_type(path),
        theoretical_spectrum=template,
        automatic_region_count=2,
        automatic_region_half_width_pixels=2,
    )

    np.testing.assert_array_equal(selector.spectrum.wavelength, spectrum.wavelength)
    np.testing.assert_array_equal(selector.spectrum.flux, spectrum.flux)
    np.testing.assert_array_equal(selector.spectrum.uncertainty, np.full(101, 0.02))
    assert selector.spectrum.wavelength_unit.lower() == "angstrom"
    assert selector.spectrum.wavelength_medium == medium
    assert selector.spectrum.meta["source"] == str(path.resolve())
    assert selector.figure is None
    assert selector.selection.fit_ranges == ((5028.0, 5032.0), (5078.0, 5082.0))
    assert selector.selection.exclude_ranges == exclusions.exclude_ranges
    assert set(tmp_path.iterdir()) == {path}
    saved = load_region_file(selector.save(tmp_path / "regions.ecsv"))
    assert saved == selector.selection


def test_selector_forwards_file_read_options(spectrum, catalog, monkeypatch):
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return spectrum

    monkeypatch.setattr("pymolfit.regions.load_spectrum", load)
    options = {
        "wavelength_col": "LAMBDA",
        "flux_col": "SCI",
        "uncertainty_col": "ERR",
        "hdu": 2,
        "image_index": 3,
        "wavelength_unit": "angstrom",
        "wavelength_medium": "air",
    }
    select_telluric_regions("spectrum.data", input_format="fits", **options)
    assert calls == [("spectrum.data", {"format": "fits", "save_header": False, **options})]


def test_selector_requires_explicit_medium_for_ambiguous_fits(spectrum, catalog, tmp_path):
    path = tmp_path / "ambiguous.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVE", format="D", unit="Angstrom", array=spectrum.wavelength),
            fits.Column(name="FLUX", format="D", array=spectrum.flux),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    with pytest.raises(WavelengthMetadataError):
        select_telluric_regions(path)
    assert catalog == []
    selector = select_telluric_regions(path, wavelength_medium="air")
    assert selector.spectrum.wavelength_medium == "air"


def test_selector_rejects_missing_file_and_mixed_inputs(tmp_path, catalog):
    missing = tmp_path / "missing.fits"
    with pytest.raises(FileNotFoundError, match="spectrum file does not exist"):
        select_telluric_regions(missing)
    with pytest.raises(ValueError, match="not both"):
        select_telluric_regions(missing, wavelength=np.arange(10), flux=np.ones(10))
    with pytest.raises(TypeError, match="file path or Spectrum"):
        select_telluric_regions(123)
    assert catalog == []


@pytest.mark.parametrize("options, medium", [({}, "vacuum"), ({"wavelength_medium": "air"}, "air")])
def test_selector_keeps_array_wavelength_conventions(spectrum, catalog, options, medium):
    selector = select_telluric_regions(
        wavelength=spectrum.wavelength,
        flux=spectrum.flux,
        wavelength_unit="angstrom",
        **options,
    )
    assert selector.spectrum.wavelength_medium == medium
    assert selector.selection.fit_ranges


@pytest.mark.parametrize("show", [True, False])
def test_removed_show_alias_is_rejected(spectrum, show):
    with pytest.raises(TypeError, match="unexpected keyword argument 'show'"):
        select_telluric_regions(spectrum, show=show)
