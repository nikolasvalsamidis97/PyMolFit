import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from pymolfit import (
    Spectrum,
    TelluricFitResult,
    air_to_vacuum_wavelength,
    load_fit_product,
    load_spectrum,
    save_fit_product,
    save_spectrum,
    vacuum_to_air_wavelength,
)
from pymolfit.provenance import file_sha256


def test_ascii_roundtrip(tmp_path):
    path = tmp_path / "spectrum.txt"
    spectrum = Spectrum(
        wavelength=np.array([1.0, 1.1, 1.2]),
        flux=np.array([2.0, 2.1, 2.2]),
        uncertainty=np.array([0.1, 0.1, 0.2]),
    )

    save_spectrum(path, spectrum)
    loaded = load_spectrum(path, format="ascii", uncertainty_col=2)

    np.testing.assert_allclose(loaded.wavelength, spectrum.wavelength)
    np.testing.assert_allclose(loaded.flux, spectrum.flux)
    np.testing.assert_allclose(loaded.uncertainty, spectrum.uncertainty)
    assert loaded.meta["source"] == str(path.resolve())
    assert loaded.meta["source_file_sha256"] == file_sha256(path)
    assert loaded.name == "spectrum"


def test_complete_fit_product_roundtrip(tmp_path):
    wavelength = np.linspace(0.68, 0.69, 6)
    spectrum = Spectrum(
        wavelength=wavelength,
        flux=np.linspace(0.8, 1.0, 6),
        uncertainty=np.full(6, 0.01),
        mask=np.array([True, True, False, True, True, True]),
        group_id=np.array([2, 2, 2, 3, 3, 3]),
        wavelength_medium="air",
    )
    transmission = np.linspace(0.9, 1.0, 6)
    corrected = spectrum.with_flux(
        spectrum.flux / transmission,
        uncertainty=spectrum.uncertainty / transmission,
    )
    result = TelluricFitResult(
        spectrum=spectrum,
        corrected=corrected,
        transmission=transmission,
        continuum=np.ones(6),
        model_flux=transmission.copy(),
        species_scales={"O2": 1.2},
        wavelength_shift=0.1,
        wavelength_coefficients=np.array([0.1, 0.02]),
        lsf_sigma_pixels=1.7,
        lsf_box_width_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.3,
        lsf_wavelength_exponent=0.2,
        continuum_coefficients=np.array([1.0, 0.01]),
        metrics={"corrected_scatter": 0.02},
        success=True,
        message="complete",
        cost=1.5,
        nfev=7,
        parameter_names=("scale_O2", "wavelength_shift"),
        parameter_covariance=np.eye(2),
        parameter_standard_errors={"scale_O2": 0.1},
        species_scale_uncertainties={"O2": 0.1},
        transmission_uncertainty=np.full(6, 0.005),
        reduced_chi_square=1.1,
        covariance_rank=2,
        fit_mask=np.array([True, True, False, True, False, True]),
        parameter_bound_status={"scale_O2": "none"},
        wavelength_group_coefficients={2: np.array([0.1])},
        wavelength_group_bounds={2: (0.68, 0.685)},
        provenance={"schema_version": 1, "line_source": "test"},
    )
    path = tmp_path / "fit_product.ecsv"

    assert save_fit_product(result, path) == path
    loaded = load_fit_product(path)

    np.testing.assert_allclose(loaded.spectrum.wavelength, wavelength)
    np.testing.assert_allclose(loaded.corrected.flux, corrected.flux)
    np.testing.assert_allclose(loaded.transmission, transmission)
    np.testing.assert_allclose(loaded.parameter_covariance, np.eye(2))
    np.testing.assert_allclose(
        loaded.wavelength_group_coefficients[2],
        [0.1],
    )
    assert loaded.spectrum.wavelength_medium == "air"
    assert loaded.spectrum.group_id.tolist() == [2, 2, 2, 3, 3, 3]
    assert loaded.metrics == {"corrected_scatter": 0.02}
    assert loaded.species_scale_uncertainties == {"O2": 0.1}
    assert loaded.provenance["line_source"] == "test"


def test_loaded_spectrum_name_survives_conversion(tmp_path):
    path = tmp_path / "beta-pic.fits"
    table = Table()
    table["WAVE"] = [5000.0, 5001.0]
    table["FLUX"] = [1.0, 0.9]
    table.write(path)

    loaded = load_spectrum(
        path,
        wavelength_unit="angstrom",
        wavelength_medium="vacuum",
        save_header=False,
    )

    assert loaded.name == "beta-pic"
    assert loaded.to_unit("micron").name == "beta-pic"


def test_wavelength_unit_conversion():
    spectrum = Spectrum(
        wavelength=np.array([23_100.0, 23_200.0]),
        flux=np.array([1.0, 0.9]),
        wavelength_unit="angstrom",
    )

    converted = spectrum.to_unit("micron")

    np.testing.assert_allclose(converted.wavelength, [2.31, 2.32])
    assert converted.wavelength_unit == "micron"


def test_air_to_vacuum_matches_molecfit_edlen_coefficients():
    wavelength_air = np.array([0.588995, 1.0, 2.3])
    sigma2 = wavelength_air**-2
    refractive_index = 1.0 + 1.0e-8 * (
        8342.13 + 2_406_030.0 / (130.0 - sigma2) + 15_997.0 / (38.9 - sigma2)
    )

    vacuum = air_to_vacuum_wavelength(wavelength_air)

    np.testing.assert_allclose(vacuum, wavelength_air * refractive_index, rtol=0.0, atol=1.0e-15)


def test_air_vacuum_wavelength_conversion_roundtrip():
    air = np.array([6869.0, 7600.0])
    vacuum = air_to_vacuum_wavelength(air, unit="angstrom")
    roundtrip = vacuum_to_air_wavelength(vacuum, unit="angstrom")

    assert np.all(vacuum > air)
    np.testing.assert_allclose(roundtrip, air, rtol=0, atol=1e-8)


def test_spectrum_to_vacuum_preserves_flux_and_unit():
    spectrum = Spectrum(
        wavelength=np.array([6869.0, 6870.0]),
        flux=np.array([1.0, 0.9]),
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )

    converted = spectrum.to_vacuum()

    assert converted.wavelength_unit == "angstrom"
    assert converted.wavelength_medium == "vacuum"
    np.testing.assert_allclose(converted.flux, spectrum.flux)
    assert np.all(converted.wavelength > spectrum.wavelength)


def test_csv_numeric_loading_infers_first_two_columns(tmp_path):
    path = tmp_path / "spectrum.csv"
    data = np.array([[2.31, 1.0], [2.32, 0.9], [2.33, 0.95]])
    np.savetxt(path, data, delimiter=",")

    loaded = load_spectrum(path)

    np.testing.assert_allclose(loaded.wavelength, data[:, 0])
    np.testing.assert_allclose(loaded.flux, data[:, 1])


def test_ecsv_named_columns_are_inferred(tmp_path):
    path = tmp_path / "spectrum.ecsv"
    table = Table()
    table["wave"] = [2310.0, 2320.0, 2330.0]
    table["flux"] = [1.0, 0.9, 0.95]
    table["err"] = [0.01, 0.02, 0.01]
    table["wave"].unit = "nm"
    table.write(path, format="ascii.ecsv")

    loaded = load_spectrum(path)

    np.testing.assert_allclose(loaded.wavelength, table["wave"])
    np.testing.assert_allclose(loaded.uncertainty, table["err"])
    assert loaded.wavelength_unit == "nm"


def test_fits_table_columns_are_inferred(tmp_path):
    path = tmp_path / "spectrum.fits"
    table = Table()
    table["WAVE"] = [2.31, 2.32, 2.33]
    table["FLUX"] = [1.0, 0.9, 0.95]
    table["ERR"] = [0.01, 0.02, 0.01]
    table.write(path)

    loaded = load_spectrum(path, format="fits", wavelength_medium="vacuum")

    np.testing.assert_allclose(loaded.wavelength, table["WAVE"])
    np.testing.assert_allclose(loaded.flux, table["FLUX"])
    np.testing.assert_allclose(loaded.uncertainty, table["ERR"])


def test_fits_load_saves_formatted_header_text(tmp_path, capsys):
    path = tmp_path / "air_spectrum.fits"
    primary = fits.PrimaryHDU()
    primary.header["OBJECT"] = "Example star"
    primary.header["INSTRUME"] = "EXAMPLE"
    columns = [
        fits.Column(name="WAVE_AIR", format="D", unit="Angstrom", array=np.array([5000.0, 5001.0])),
        fits.Column(name="FLUX", format="D", array=np.array([1.0, 0.9])),
    ]
    spectrum_hdu = fits.BinTableHDU.from_columns(columns, name="SPECTRUM")
    spectrum_hdu.header["TUCD1"] = ("em.wl;obs.atmos", "Air wavelength")
    fits.HDUList([primary, spectrum_hdu]).writeto(path)

    loaded = load_spectrum(path, wavelength_col="WAVE_AIR")

    header_path = path.with_suffix(".header.txt")
    output = header_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == ""
    assert "HDU 0: PRIMARY" in output
    assert "HDU 1: SPECTRUM" in output
    assert "OBJECT" in output and "Example star" in output
    assert "TUCD1" in output and "em.wl;obs.atmos" in output and "# Air wavelength" in output
    assert loaded.meta["header_text_path"] == str(header_path.resolve())
    assert loaded.wavelength_medium == "air"
    assert loaded.meta["wavelength_medium_source"] == "fits_header"


def test_fits_load_infers_medium_for_selected_wavelength_column(tmp_path):
    path = tmp_path / "dual_wavelength.fits"
    columns = [
        fits.Column(name="WAVE", format="D", unit="Angstrom", array=np.array([5001.0, 5002.0])),
        fits.Column(name="WAVE_AIR", format="D", unit="Angstrom", array=np.array([5000.0, 5001.0])),
        fits.Column(name="FLUX", format="D", array=np.array([1.0, 0.9])),
    ]
    spectrum_hdu = fits.BinTableHDU.from_columns(columns, name="SPECTRUM")
    spectrum_hdu.header["TUCD1"] = "em.wl;meta.main"
    spectrum_hdu.header["TUCD2"] = "em.wl;obs.atmos"
    fits.HDUList([fits.PrimaryHDU(), spectrum_hdu]).writeto(path)

    vacuum = load_spectrum(path, save_header=False)
    air = load_spectrum(path, wavelength_col="WAVE_AIR", save_header=False)

    assert vacuum.wavelength_medium == "vacuum"
    assert air.wavelength_medium == "air"


def test_fits_load_rejects_unknown_wavelength_medium(tmp_path):
    path = tmp_path / "unknown_medium.fits"
    table = Table()
    table["WAVE"] = [2.31, 2.32]
    table["FLUX"] = [1.0, 0.9]
    table.write(path)

    with pytest.raises(ValueError, match="could not determine"):
        load_spectrum(path, save_header=False)


def test_fits_load_header_text_can_be_disabled(tmp_path):
    path = tmp_path / "spectrum.fits"
    table = Table()
    table["WAVE"] = [2.31, 2.32]
    table["FLUX"] = [1.0, 0.9]
    table.write(path)

    load_spectrum(path, wavelength_medium="vacuum", save_header=False)

    assert not path.with_suffix(".header.txt").exists()


def test_fits_quality_and_physical_group_columns_are_preserved(tmp_path):
    path = tmp_path / "echelle.fits"
    table = Table()
    table["WAVE"] = [2.31, 2.32, 2.41, 2.42]
    table["FLUX"] = [1.0, 0.9, 1.1, 0.8]
    table["ERR"] = [0.01, 0.01, 0.02, 0.02]
    table["QUAL"] = [0, 1, 0, 2]
    table["ORDER"] = [10, 10, 11, 11]
    table["DETEC"] = [1, 1, 2, 2]
    table.write(path)

    loaded = load_spectrum(path, format="fits", wavelength_medium="vacuum")

    np.testing.assert_array_equal(loaded.mask, [True, False, True, False])
    assert loaded.group_id is not None
    assert np.unique(loaded.group_id).size == 2
    assert loaded.meta["quality_columns"] == ("QUAL",)
    assert loaded.meta["physical_group_columns"] == ("ORDER", "DETEC")


def test_fits_table_single_row_vector_columns_are_loaded(tmp_path):
    path = tmp_path / "harps_like.fits"
    wave = np.array([5000.0, 5000.1, 5000.2], dtype=float)
    flux = np.array([1.0, 0.98, 1.01], dtype=np.float32)
    err = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    columns = [
        fits.Column(name="WAVE", format=f"{wave.size}D", unit="Angstrom", array=[wave]),
        fits.Column(name="FLUX", format=f"{flux.size}E", unit="adu", array=[flux]),
        fits.Column(name="ERR", format=f"{err.size}E", unit="adu", array=[err]),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="SPECTRUM")]).writeto(path)

    loaded = load_spectrum(path, format="fits", wavelength_medium="vacuum")

    np.testing.assert_allclose(loaded.wavelength, wave)
    np.testing.assert_allclose(loaded.flux, flux)
    assert loaded.uncertainty is None
    assert loaded.wavelength_unit == "Angstrom"


def test_fits_image_linear_wcs_loading(tmp_path):
    path = tmp_path / "image_spectrum.fits"
    flux = np.array([1.0, 0.9, 0.95, 1.02])
    hdu = fits.PrimaryHDU(flux)
    hdu.header["CRVAL1"] = 2310.0
    hdu.header["CDELT1"] = 0.5
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CUNIT1"] = "nm"
    hdu.writeto(path)

    loaded = load_spectrum(path, format="fits", hdu=0, wavelength_medium="vacuum")

    np.testing.assert_allclose(loaded.wavelength, [2310.0, 2310.5, 2311.0, 2311.5])
    np.testing.assert_allclose(loaded.flux, flux)
    assert loaded.wavelength_unit == "nm"


def test_load_gzip_compressed_fits_infers_format(tmp_path):
    path = tmp_path / "spectrum.fits.gz"
    columns = [
        fits.Column(name="wave", format="D", array=np.array([5000.0, 5001.0])),
        fits.Column(name="flux", format="D", array=np.array([1.0, 0.9])),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)

    loaded = load_spectrum(path, wavelength_unit="angstrom", wavelength_medium="vacuum")

    np.testing.assert_allclose(loaded.wavelength, [5000.0, 5001.0])
    np.testing.assert_allclose(loaded.flux, [1.0, 0.9])
