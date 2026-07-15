import numpy as np
import pytest
from astropy.table import Table
from types import SimpleNamespace

from genmolfit import LineList, ModelConfig, transmission_model
from genmolfit.cli import build_parser, main


def _fixed_decimal(value, width, decimals):
    text = f"{value:.{decimals}f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return f"{text:>{width}}"[-width:]


def _hitran_row():
    row = (
        f"{1:2d}"
        f"{1:1d}"
        f"{4320.0:12.6f}"
        f"{1.0e-24:10.3E}"
        f"{1.0:10.3E}"
        f"{_fixed_decimal(0.07, 5, 4)}"
        f"{_fixed_decimal(0.30, 5, 4)}"
        f"{100.0:10.4f}"
        f"{0.75:4.2f}"
        f"{_fixed_decimal(-0.001, 8, 6)}"
    )
    return row + " " * (160 - len(row))


def test_cli_fit_writes_corrected_and_product(tmp_path):
    wavelength = np.linspace(2.31, 2.36, 300)
    flux = transmission_model(
        wavelength,
        LineList.demo_near_ir(),
        ModelConfig(species_scales={"H2O": 1.2}),
    )
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "corrected.txt"
    product_path = tmp_path / "product.ecsv"
    np.savetxt(input_path, np.column_stack([wavelength, flux]))

    status = main(
        [
            "fit",
            str(input_path),
            str(output_path),
            "--demo-lines",
            "--continuum-order",
            "0",
            "--product",
            str(product_path),
        ]
    )

    corrected = np.loadtxt(output_path)
    product = Table.read(product_path)
    assert status == 0
    assert corrected.shape[0] == wavelength.size
    assert "transmission" in product.colnames


def test_cli_exposes_native_radiative_transfer_controls():
    args = build_parser().parse_args(
        [
            "fit",
            "input.txt",
            "output.txt",
            "--demo-lines",
            "--radiative-transfer-grid",
            "model",
            "--radiative-transfer-step-cm",
            "0.002",
            "--radiative-transfer-max-points",
            "12345",
            "--lblrtm-avmass-amu",
            "35.5",
        ]
    )

    assert args.radiative_transfer_grid == "model"
    assert args.radiative_transfer_step_cm == 0.002
    assert args.radiative_transfer_max_points == 12345
    assert args.lblrtm_avmass_amu == 35.5


def test_cli_refuses_implicit_synthetic_line_data(tmp_path):
    input_path = tmp_path / "input.txt"
    np.savetxt(input_path, np.column_stack([np.linspace(2.31, 2.36, 20), np.ones(20)]))

    with pytest.raises(SystemExit) as exc_info:
        main(["fit", str(input_path), str(tmp_path / "corrected.txt"), "--no-auto-aer"])

    assert exc_info.value.code == 2


def test_cli_install_aer_reports_verified_artifact(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "aer_v_3.9"
    catalog.write_text("catalogue")
    captured = {}

    def fake_install(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            catalog_path=catalog,
            manifest={"catalog_version": "3.9"},
            source="test-archive",
            managed=True,
        )

    monkeypatch.setattr("genmolfit.cli.install_aer_catalog", fake_install)
    status = main(["install-aer", "--source", "test.tar.gz", "--cache-dir", str(tmp_path)])

    assert status == 0
    assert captured["source"] == "test.tar.gz"
    assert captured["cache_dir"] == tmp_path
    assert "version: 3.9" in capsys.readouterr().out


def test_cli_aer_status_is_nonzero_when_catalogue_missing(monkeypatch):
    monkeypatch.setattr("genmolfit.cli.aer_catalog_status", lambda **kwargs: None)
    assert main(["aer-status", "--no-reuse-molecfit"]) == 1


def test_cli_rejects_two_global_wavelength_fit_modes(tmp_path):
    input_path = tmp_path / "input.txt"
    np.savetxt(input_path, np.column_stack([np.linspace(2.31, 2.36, 20), np.ones(20)]))

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "fit",
                str(input_path),
                str(tmp_path / "corrected.txt"),
                "--demo-lines",
                "--fit-wavelength-shift",
                "--fit-wavelength-polynomial",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_convert_hitran(tmp_path):
    input_path = tmp_path / "input.par"
    output_path = tmp_path / "lines.ecsv"
    input_path.write_text(_hitran_row() + "\n")

    status = main(["convert-hitran", str(input_path), str(output_path), "--species", "H2O"])
    table = Table.read(output_path)

    assert status == 0
    assert table["species"][0] == "H2O"


def test_cli_convert_hitran_filters_lines(tmp_path):
    input_path = tmp_path / "input.par"
    output_path = tmp_path / "lines.ecsv"
    input_path.write_text(
        "\n".join(
            [
                _hitran_row(),
                _hitran_row().replace("4320.000000", "4321.000000").replace("1.000E-24", "5.000E-25", 1),
            ]
        )
        + "\n"
    )

    status = main(
        [
            "convert-hitran",
            str(input_path),
            str(output_path),
            "--min-strength",
            "9e-25",
            "--max-lines",
            "1",
        ]
    )
    table = Table.read(output_path)

    assert status == 0
    assert len(table) == 1
    np.testing.assert_allclose(table["strength"], [1.0e-24])


def test_cli_compare_spectra(tmp_path):
    candidate = tmp_path / "candidate.txt"
    reference = tmp_path / "reference.txt"
    wavelength = np.linspace(2.31, 2.36, 10)
    np.savetxt(candidate, np.column_stack([wavelength, np.ones_like(wavelength)]))
    np.savetxt(reference, np.column_stack([wavelength, np.ones_like(wavelength)]))

    status = main(["compare", str(candidate), str(reference)])

    assert status == 0


def test_cli_hitran_filter_respects_wavelength_unit(tmp_path):
    hitran_path = tmp_path / "h2o.par"
    input_path = tmp_path / "spectrum_nm.txt"
    output_path = tmp_path / "corrected.txt"
    hitran_path.write_text(_hitran_row() + "\n")
    center_nm = (1.0e4 / 4320.0) * 1000.0
    wavelength_nm = np.linspace(center_nm - 1.0, center_nm + 1.0, 80)
    flux = np.ones_like(wavelength_nm)
    np.savetxt(input_path, np.column_stack([wavelength_nm, flux]))

    status = main(
        [
            "fit",
            str(input_path),
            str(output_path),
            "--wavelength-unit",
            "nm",
            "--hitran-par",
            str(hitran_path),
            "--continuum-order",
            "0",
            "--mixing-ratio",
            "H2O=1e-5",
            "--allow-default-observatory",
            "--line-wing-mode",
            "lblrtm_panel",
            "--lblrtm-sample",
            "4",
            "--n2-continuum",
            "--lsf-molecfit-voigt",
        ]
    )

    assert status == 0
    assert output_path.exists()
