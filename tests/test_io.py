import numpy as np
from astropy.io import fits
from astropy.table import Table

from pymolfit import Spectrum, air_to_vacuum_wavelength, load_spectrum, save_spectrum, vacuum_to_air_wavelength
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

    loaded = load_spectrum(path, format="fits")

    np.testing.assert_allclose(loaded.wavelength, table["WAVE"])
    np.testing.assert_allclose(loaded.flux, table["FLUX"])
    np.testing.assert_allclose(loaded.uncertainty, table["ERR"])


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

    loaded = load_spectrum(path, format="fits")

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

    loaded = load_spectrum(path, format="fits", hdu=0)

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

    loaded = load_spectrum(path, wavelength_unit="angstrom")

    np.testing.assert_allclose(loaded.wavelength, [5000.0, 5001.0])
    np.testing.assert_allclose(loaded.flux, [1.0, 0.9])
