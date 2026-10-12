import inspect

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from pymolfit import (
    AdvancedOptions,
    correct,
    correct_file,
    load_spectrum,
    select_telluric_regions,
    workflow,
)
from pymolfit.errors import SpectrumFormatError, WavelengthMetadataError
from pymolfit.io import infer_fits_wavelength_medium


def spectrum_hdu(name="SCI", wave_name="WAVE", flux_name="FLUX", medium="vacuum"):
    table = Table({wave_name: np.linspace(589.0, 590.0, 20), flux_name: np.ones(20)})
    table[wave_name].unit = "nm"
    extension = fits.BinTableHDU(table, name=name)
    extension.header["TUCD1"] = "em.wl;obs.atmos" if medium == "air" else "em.wl"
    extension.header["SPECSYS"] = "TOPOCENT"
    extension.header["AIRMASS"] = 1.6
    return extension


def image_hdu(name="SCI", primary=False, rows=None):
    data = np.ones(20) if rows is None else np.ones((rows, 20))
    extension = fits.PrimaryHDU(data) if primary else fits.ImageHDU(data, name=name)
    extension.header["CRVAL1"] = 589.0
    extension.header["CDELT1"] = 0.05
    extension.header["CTYPE1"] = "WAVE"
    extension.header["CUNIT1"] = "nm"
    extension.header["SPECSYS"] = "TOPOCENT"
    return extension


@pytest.fixture
def late_spectrum(tmp_path):
    path = tmp_path / "late.fits"
    primary = fits.PrimaryHDU()
    primary.header["OBJECT"] = "Example star"
    metadata = fits.BinTableHDU(Table({"x": [1, 2], "y": [3, 4]}), name="METADATA")
    fits.HDUList([primary, metadata, image_hdu("ERR"), spectrum_hdu(medium="air")]).writeto(path)
    return path


def test_auto_selects_later_spectrum_and_its_metadata(late_spectrum):
    result = load_spectrum(late_spectrum, save_header=False)
    assert result.meta["hdu"] == 3
    assert result.wavelength_unit == "nm"
    assert result.wavelength_medium == "air"
    assert result.observation.metadata["OBJECT"] == "Example star"
    assert result.observation.metadata["AIRMASS"] == 1.6
    np.testing.assert_array_equal(result.flux, np.ones(20))
    assert infer_fits_wavelength_medium(late_spectrum) == "air"


@pytest.mark.parametrize("loader", [load_spectrum, correct, correct_file, select_telluric_regions])
def test_public_hdu_defaults_are_automatic(loader):
    assert inspect.signature(loader).parameters["hdu"].default is None


@pytest.mark.parametrize("primary", [True, False])
def test_auto_selects_wavelength_calibrated_image(tmp_path, primary):
    path = tmp_path / "image.fits"
    image = image_hdu(primary=primary)
    extensions = [image] if primary else [fits.PrimaryHDU(), image_hdu("DQ"), image]
    fits.HDUList(extensions).writeto(path)
    result = load_spectrum(path, save_header=False)
    assert result.meta["hdu"] == (0 if primary else 2)
    assert result.wavelength_medium == "vacuum"
    np.testing.assert_allclose(result.wavelength, 589 + 0.05 * np.arange(20))


def test_2d_image_requires_row_selection(tmp_path):
    path = tmp_path / "orders.fits"
    image = image_hdu(rows=2)
    image.data[1] *= 2
    fits.HDUList([fits.PrimaryHDU(), image]).writeto(path)
    with pytest.raises(SpectrumFormatError, match="image_index"):
        load_spectrum(path, save_header=False)
    result = load_spectrum(path, image_index=1, save_header=False)
    assert result.meta["hdu"] == 1
    np.testing.assert_array_equal(result.flux, np.full(20, 2))


@pytest.mark.parametrize(
    "loader", [load_spectrum, correct, correct_file, infer_fits_wavelength_medium]
)
def test_ambiguous_spectra_report_choices(tmp_path, loader):
    path = tmp_path / "orders.fits"
    fits.HDUList([fits.PrimaryHDU(), spectrum_hdu("ORDER1"), spectrum_hdu("ORDER2")]).writeto(path)
    with pytest.raises(SpectrumFormatError, match=r"1 \(ORDER1\), 2 \(ORDER2\).*hdu="):
        loader(path)


@pytest.mark.parametrize("index", [2, -1, np.int64(2)])
def test_explicit_index_selects_one_of_multiple_spectra(tmp_path, index):
    path = tmp_path / "orders.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), spectrum_hdu("ORDER1"), spectrum_hdu("ORDER2", medium="air")]
    ).writeto(path)
    result = load_spectrum(path, hdu=index, save_header=False)
    assert result.meta["hdu"] == 2
    assert result.wavelength_medium == "air"


@pytest.mark.parametrize("index", [4, -5, True, "auto", 1.5])
def test_invalid_hdu_is_rejected(late_spectrum, index):
    with pytest.raises(SpectrumFormatError, match=r"HDU|hdu must"):
        load_spectrum(late_spectrum, hdu=index, save_header=False)


def test_explicit_empty_hdu_does_not_fall_back_to_primary(tmp_path):
    path = tmp_path / "empty_extension.fits"
    fits.HDUList([image_hdu(primary=True), fits.ImageHDU(name="EMPTY")]).writeto(path)
    with pytest.raises(SpectrumFormatError, match="contains no data"):
        load_spectrum(path, hdu=1, save_header=False)
    assert load_spectrum(path, save_header=False).meta["hdu"] == 0


def test_unknown_table_requires_column_hints_or_explicit_hdu(tmp_path):
    path = tmp_path / "custom.fits"
    fits.HDUList([fits.PrimaryHDU(), spectrum_hdu(wave_name="X", flux_name="Y")]).writeto(path)
    with pytest.raises(SpectrumFormatError, match="No supported spectral HDU"):
        load_spectrum(path, save_header=False)
    by_columns = load_spectrum(path, wavelength_col="X", flux_col="Y", save_header=False)
    by_index = load_spectrum(path, hdu=1, save_header=False)
    np.testing.assert_array_equal(by_columns.flux, by_index.flux)


def test_column_hints_disambiguate_spectra(tmp_path):
    path = tmp_path / "custom.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), spectrum_hdu(), spectrum_hdu("OTHER", flux_name="CUSTOM")]
    ).writeto(path)
    result = load_spectrum(path, flux_col="CUSTOM", save_header=False)
    assert result.meta["hdu"] == 2


def test_auxiliary_table_and_uncalibrated_image_are_not_spectra(tmp_path):
    path = tmp_path / "auxiliary.fits"
    fits.HDUList([fits.PrimaryHDU(np.ones((20, 20))), spectrum_hdu("FLUX_ERR")]).writeto(path)
    with pytest.raises(SpectrumFormatError, match="No supported spectral HDU"):
        load_spectrum(path, save_header=False)
    # Explicit selection still permits non-science data when intentionally requested.
    assert load_spectrum(path, hdu=1, save_header=False).meta["hdu"] == 1


def test_selected_spectrum_still_requires_known_wavelength_medium(tmp_path):
    path = tmp_path / "unknown_medium.fits"
    extension = spectrum_hdu()
    del extension.header["TUCD1"]
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(), extension]).writeto(path)
    with pytest.raises(WavelengthMetadataError, match="could not determine"):
        load_spectrum(path, save_header=False)
    assert load_spectrum(path, wavelength_medium="air", save_header=False).meta["hdu"] == 2


@pytest.mark.parametrize("route", ["file", "correct"])
def test_correction_reads_metadata_from_the_selected_hdu(late_spectrum, monkeypatch, route):
    captured = {}

    def fit(spectrum, **options):
        captured.update(options)
        return spectrum

    monkeypatch.setattr(workflow, "_correct_spectrum_workflow", fit)
    monkeypatch.setattr(workflow, "_finalize_correction", lambda result, **kw: result)
    if route == "file":
        result = correct_file(late_spectrum, report=False)
    else:
        result = correct(late_spectrum, report=False, advanced=AdvancedOptions())
    assert result.meta["hdu"] == 3
    assert result.wavelength_medium == "air"
    assert captured["atmosphere_header"]["AIRMASS"] == 1.6


def test_region_selector_uses_the_same_hdu(late_spectrum, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pymolfit.regions._aer_catalog_for_spectrum",
        lambda *a, **kw: (
            np.array([589.5]),
            np.array(["H2O"]),
            np.array([1.0]),
        ),
    )
    selector = select_telluric_regions(late_spectrum, output_path=tmp_path / "regions.ecsv")
    try:
        assert selector.spectrum.meta["hdu"] == 3
        assert selector.spectrum.observation.metadata["AIRMASS"] == 1.6
    finally:
        selector.close()
