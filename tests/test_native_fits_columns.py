import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import pymolfit as pm
from pymolfit.io import _spectrum_from_table


def _write_native_fits(tmp_path, count=2, error=True):
    columns = []
    for order in range(3, 3 + count):
        prefix = f"{order:02d}_01"
        columns.append(
            fits.Column(name=prefix + "_SPEC", format="D", array=np.arange(12.0) + 10 * order)
        )
        if error:
            columns.append(
                fits.Column(name=prefix + "_ERR", format="D", array=np.ones(12) * order / 10)
            )
        columns.append(
            fits.Column(
                name=prefix + "_WL",
                format="D",
                unit="nm",
                array=np.arange(12.0) + 1500 + order,
            )
        )
    hdu = fits.BinTableHDU.from_columns(columns)
    for i, name in enumerate(hdu.columns.names, 1):
        if name.endswith("_WL"):
            hdu.header[f"TUCD{i}"] = "em.wl"
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "CRIRES"
    primary.header["SPECSYS"] = "TOPOCENT"
    path = tmp_path / "native.fits"
    fits.HDUList([primary, hdu]).writeto(path)
    return path


@pytest.mark.parametrize("medium", [None, "vacuum"])
def test_multinative_hdu_only_rejects_positional_flux_as_wave(tmp_path, medium):
    path = _write_native_fits(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match=r"[Mm]ultiple.*(physical|native).*orders"):
        pm.load_spectrum(path, hdu=1, wavelength_medium=medium, save_header=False)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.header.txt"))


@pytest.mark.parametrize(
    "selection",
    [
        {"wavelength_col": "03_01_WL"},
        {"flux_col": "03_01_SPEC", "uncertainty_col": "03_01_ERR"},
    ],
)
def test_multinative_partial_triplet_rejected(tmp_path, selection):
    with pytest.raises(ValueError, match=r"[Mm]ultiple.*(physical|native).*orders"):
        pm.load_spectrum(
            _write_native_fits(tmp_path),
            hdu=1,
            wavelength_medium="vacuum",
            save_header=False,
            **selection,
        )


@pytest.mark.parametrize(
    "columns",
    [
        {
            "wavelength_col": "04_01_WL",
            "flux_col": "04_01_SPEC",
            "uncertainty_col": "04_01_ERR",
        },
        {"wavelength_col": 5, "flux_col": 3, "uncertainty_col": 4},
    ],
)
def test_explicit_name_or_index_triplet_exact(tmp_path, columns):
    s = pm.load_spectrum(_write_native_fits(tmp_path), hdu=1, save_header=False, **columns)
    np.testing.assert_array_equal(s.wavelength, np.arange(12.0) + 1504)
    np.testing.assert_array_equal(s.flux, np.arange(12.0) + 40)
    np.testing.assert_array_equal(s.uncertainty, np.ones(12) * 0.4)
    assert s.wavelength_unit == "nm" and s.wavelength_medium == "vacuum"
    assert s.observation.metadata["SPECSYS"] == "TOPOCENT"


@pytest.mark.parametrize("hdu", [None, 1])
def test_single_native_order_resolves_named_triplet_and_medium(tmp_path, hdu):
    s = pm.load_spectrum(_write_native_fits(tmp_path, count=1), hdu=hdu, save_header=False)
    np.testing.assert_array_equal(s.wavelength, np.arange(12.0) + 1503)
    np.testing.assert_array_equal(s.flux, np.arange(12.0) + 30)
    np.testing.assert_array_equal(s.uncertainty, np.ones(12) * 0.3)
    assert s.wavelength_medium == "vacuum"


def test_single_native_missing_error_does_not_use_wave_as_error(tmp_path):
    s = pm.load_spectrum(_write_native_fits(tmp_path, count=1, error=False), save_header=False)
    assert s.uncertainty is None
    np.testing.assert_array_equal(s.flux, np.arange(12.0) + 30)


@pytest.mark.parametrize("explicit", [True, False])
def test_generic_two_column_explicit_hdu_compatibility(tmp_path, explicit):
    path = tmp_path / "two.fits"
    columns = [
        fits.Column(name="coordinate", format="D", unit="nm", array=np.arange(12.0) + 1500),
        fits.Column(name="intensity", format="D", array=np.arange(12.0) + 20),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    options = {"wavelength_col": 0, "flux_col": 1} if explicit else {}
    s = pm.load_spectrum(path, hdu=1, wavelength_medium="vacuum", save_header=False, **options)
    np.testing.assert_array_equal(s.wavelength, np.arange(12.0) + 1500)
    np.testing.assert_array_equal(s.flux, np.arange(12.0) + 20)
    assert s.uncertainty is None


@pytest.mark.parametrize(
    "columns",
    [
        {"wavelength_col": "03_01_WL", "flux_col": "04_01_SPEC"},
        {"wavelength_col": 2, "flux_col": 3, "uncertainty_col": 1},
        {"wavelength_col": "03_01_WL", "flux_col": 3},
        {"wavelength_col": 0, "flux_col": 2},
    ],
)
def test_explicit_mixed_native_prefix_or_roles_rejected(tmp_path, columns):
    with pytest.raises(ValueError, match="matching WL/SPEC"):
        pm.load_spectrum(
            _write_native_fits(tmp_path),
            hdu=1,
            wavelength_medium="vacuum",
            save_header=False,
            **columns,
        )


def test_public_correct_rejects_ambiguous_native_before_fitting(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before any fit")

    monkeypatch.setattr("pymolfit.workflow._correct_spectrum_workflow", forbidden)
    with pytest.raises(ValueError, match=r"[Mm]ultiple.*(physical|native).*orders"):
        pm.correct(_write_native_fits(tmp_path), hdu=1, wavelength_medium="vacuum", report=False)


def test_public_correct_unambiguous_native_mapping(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def capture(spectrum, **options):
        captured.update(spectrum=spectrum, options=options)
        return sentinel

    monkeypatch.setattr("pymolfit.workflow._correct_spectrum_workflow", capture)
    monkeypatch.setattr("pymolfit.workflow._finalize_correction", lambda result, **kwargs: result)
    assert pm.correct(_write_native_fits(tmp_path, count=1), report=False) is sentinel
    np.testing.assert_array_equal(captured["spectrum"].wavelength, np.arange(12.0) + 1503)
    np.testing.assert_array_equal(captured["spectrum"].flux, np.arange(12.0) + 30)
    assert captured["spectrum"].wavelength_medium == "vacuum"
    assert captured["options"]["atmosphere_header"]["SPECSYS"] == "TOPOCENT"


@pytest.mark.parametrize("hdu", [None, 1])
@pytest.mark.parametrize(
    "columns",
    [
        {"wavelength_col": "04_01_WL", "flux_col": "04_01_SPEC"},
        {"wavelength_col": 5, "flux_col": 3},
        {"wavelength_col": "04_01_WL", "flux_col": 3},
        {"wavelength_col": 5, "flux_col": "04_01_SPEC"},
    ],
)
def test_explicit_matching_pair_infers_own_error(tmp_path, columns, hdu):
    s = pm.load_spectrum(_write_native_fits(tmp_path), hdu=hdu, save_header=False, **columns)
    np.testing.assert_array_equal(s.wavelength, np.arange(12.0) + 1504)
    np.testing.assert_array_equal(s.flux, np.arange(12.0) + 40)
    np.testing.assert_array_equal(s.uncertainty, np.ones(12) * 0.4)
    assert s.wavelength_medium == "vacuum" and s.wavelength_unit == "nm"


@pytest.mark.parametrize(
    "columns",
    [
        {"wavelength_col": "04_01_WL", "flux_col": "04_01_SPEC"},
        {"wavelength_col": 3, "flux_col": 2},
    ],
)
def test_multinative_matching_pair_missing_error_preserves_none(tmp_path, columns):
    s = pm.load_spectrum(
        _write_native_fits(tmp_path, error=False), hdu=1, save_header=False, **columns
    )
    np.testing.assert_array_equal(s.wavelength, np.arange(12.0) + 1504)
    np.testing.assert_array_equal(s.flux, np.arange(12.0) + 40)
    assert s.uncertainty is None


@pytest.mark.parametrize("error", ["03_01_ERR", 1, "04_01_WL"])
def test_explicit_native_error_must_match_pair_and_role(tmp_path, error):
    with pytest.raises(ValueError, match="uncertainty column must match"):
        pm.load_spectrum(
            _write_native_fits(tmp_path),
            hdu=1,
            save_header=False,
            wavelength_col="04_01_WL",
            flux_col="04_01_SPEC",
            uncertainty_col=error,
        )


def test_explicit_generic_uncertainty_override_retained(tmp_path):
    table = Table(
        {
            "03_01_WL": np.arange(12.0) + 1500,
            "03_01_SPEC": np.arange(12.0) + 30,
            "03_01_ERR": np.ones(12) * 0.3,
            "my_errors": np.ones(12) * 0.7,
        }
    )
    result = _spectrum_from_table(
        table,
        wavelength_col="03_01_WL",
        flux_col="03_01_SPEC",
        uncertainty_col="my_errors",
        wavelength_unit="nm",
        wavelength_medium="vacuum",
    )
    np.testing.assert_array_equal(result.uncertainty, np.ones(12) * 0.7)


def test_public_correct_explicit_pair_passes_own_error_before_any_fit(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def capture(spectrum, **options):
        captured.update(spectrum=spectrum, options=options)
        return sentinel

    monkeypatch.setattr("pymolfit.workflow._correct_spectrum_workflow", capture)
    monkeypatch.setattr("pymolfit.workflow._finalize_correction", lambda result, **kwargs: result)
    assert (
        pm.correct(
            _write_native_fits(tmp_path),
            hdu=1,
            wavelength_col="04_01_WL",
            flux_col="04_01_SPEC",
            report=False,
        )
        is sentinel
    )
    np.testing.assert_array_equal(captured["spectrum"].wavelength, np.arange(12.0) + 1504)
    np.testing.assert_array_equal(captured["spectrum"].flux, np.arange(12.0) + 40)
    np.testing.assert_array_equal(captured["spectrum"].uncertainty, np.ones(12) * 0.4)
