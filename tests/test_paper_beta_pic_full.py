"""Checks for common-template calibration and unbiased wavelength-bin statistics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

spec = importlib.util.spec_from_file_location(
    "compare_full",
    Path(__file__).resolve().parents[1] / "paper_validation/beta_pic_full/compare_full.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_full_correction_enables_matched_molecular_continua(tmp_path, monkeypatch):
    spectrum = SimpleNamespace(wavelength=np.linspace(0.588, 0.591, 100))
    fit_ranges, excluded = ((0.588, 0.591),), ((0.589, 0.5891),)
    monkeypatch.setattr(
        comparison,
        "prepare",
        lambda *args: (spectrum, {"SPEC_RES": 115000}, fit_ranges, excluded, {}),
    )
    calls = []

    def correct_file(path, *, output_path, **settings):
        assert settings["o2_continuum"] is True
        assert settings["o3_continuum"] is True
        assert settings["n2_continuum"] is True
        assert settings["rayleigh"] is False
        assert settings["h2o_continuum"] == "lblrtm"
        assert settings["co2_continuum"] == "lblrtm"
        assert settings["fixed_species_scales"] == {"O3": 1.0}
        assert settings["fit_ranges"] == fit_ranges
        assert settings["exclude_ranges"] == excluded
        assert "hitran_species" not in settings
        assert settings["minimum_species_peak_optical_depth"] == 5e-3
        calls.append("pymolfit")
        return SimpleNamespace(
            success=True,
            message="converged",
            species_scales={"CO": 1.0, "H2O": 0.9, "O2": 1.1, "O3": 1.0, "N2_continuum": 1.0},
            parameter_names=("log_scale:H2O", "log_scale:O2"),
            provenance={"line_species": ["CO", "H2O", "O2"]},
        )

    monkeypatch.setattr(comparison, "correct_file", correct_file)
    monkeypatch.setattr(
        comparison, "run_molecfit_from_saved", lambda output: calls.append("molecfit")
    )
    comparison.run_corrections(None, None, tmp_path)
    assert calls == ["pymolfit", "molecfit"]
    saved = json.loads((tmp_path / "provenance.json").read_text())
    assert saved["pymolfit_settings"]["o2_continuum"] is True
    assert saved["species_policy"]["included"] == ["CO", "H2O", "N2", "O2", "O3"]
    assert saved["species_policy"]["fixed"] == {"CO": 1.0, "N2": 1.0, "O3": 1.0}


def test_molecular_policy_keeps_unfitted_abundances_at_unity():
    result = SimpleNamespace(
        species_scales={"CO": 0.5, "H2O": 0.9},
        parameter_names=("log_scale:H2O",),
        provenance={"line_species": ["CO", "H2O"]},
    )
    with pytest.raises(ValueError, match="atmospheric-profile abundance"):
        comparison.molecular_policy(result)


def test_molecfit_explicitly_enables_molecular_continua(tmp_path, monkeypatch):
    (tmp_path / "input.fits").touch()
    (tmp_path / "gdas.fits").touch()
    catalogue = tmp_path / "aer_v_3.9"
    catalogue.touch()
    monkeypatch.setattr(
        comparison,
        "aer_catalog_status",
        lambda: SimpleNamespace(catalog_path=catalogue, manifest={"catalog_sha256": "test"}),
    )
    commands = []

    def run_recipe(work, recipe, options, frames):
        commands.append((recipe, options))
        return {"recipe": recipe}

    monkeypatch.setattr(comparison, "run_recipe", run_recipe)
    comparison.fit_molecfit(
        tmp_path,
        SimpleNamespace(wavelength=np.linspace(0.588, 0.591, 100)),
        {"SPEC_RES": 115000},
        [],
        model_options={"LIST_MOLEC": "H2O,O2,O3,N2", "FIT_MOLEC": "1,1,0,0"},
    )
    assert [name for name, _ in commands] == [
        "molecfit_model",
        "molecfit_calctrans",
        "molecfit_correct",
    ]
    assert commands[0][1]["LBLRTM_ICNTNM"] == 5
    assert commands[0][1]["LIST_MOLEC"] == "H2O,O2,O3,N2"
    assert commands[0][1]["FIT_MOLEC"] == "1,1,0,0"


def test_molecular_policy_includes_species_outside_fit_regions():
    result = SimpleNamespace(
        species_scales={"H2O": 0.9, "O2": 1.1},
        parameter_names=("log_scale:H2O", "log_scale:O2"),
        provenance={"line_species": ["CO", "H2O", "O2"]},
    )
    policy = comparison.molecular_policy(result)
    assert policy["included"] == ["CO", "H2O", "O2"]
    assert policy["fixed"] == {"CO": 1.0}


def test_default_reuses_expanded_products_and_shows_plot(tmp_path, monkeypatch, spectra):
    monkeypatch.setattr(comparison, "HERE", tmp_path)
    output = tmp_path / "saved_data"
    monkeypatch.setattr(comparison, "DEFAULT_DATA", output)
    (output / "molecfit").mkdir(parents=True)
    (output / "molecfit/SCIENCE_TELLURIC_CORR_input.fits").touch()
    (output / "pymolfit.ecsv").touch()
    (output / "provenance.json").write_text(
        json.dumps({"comparison_protocol": comparison.COMPARISON_PROTOCOL})
    )

    def unexpected_fit(*args):
        raise AssertionError("Completed corrections were refitted")

    calls = []
    monkeypatch.setattr(comparison, "run_corrections", unexpected_fit)
    monkeypatch.setattr(comparison, "collect_results", lambda path: spectra)
    monkeypatch.setattr(
        comparison, "plot_transmissions", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    comparison.main([])
    assert calls[0][0][0] == output
    assert calls[0][0][-2] == 0.1
    assert calls[0][0][-1] is True
    assert calls[0][1]["figure_path"] == tmp_path / "mean_absolute_deviation.png"


def test_transmission_ratio_is_native_signed_and_independent_of_reference(spectra):
    spectra["pymolfit_transmission"][:4] = [0.4, 0.9, 0.1, 0.01]
    spectra["molecfit_transmission"][:4] = [0.5, 0.6, 0.1, 0.02]
    spectra["stellar_reference"] = np.nan
    spectra["pymolfit_corrected"] = np.nan
    result = comparison.compare_transmissions(spectra)
    np.testing.assert_allclose(result["relative_transmission_difference"][:3], [-0.2, 0.5, 0])
    assert len(result) == len(spectra)
    assert result["transmission_valid"].all()
    assert not result["ratio_valid"][3]
    assert np.isnan(result["relative_transmission_difference"][3])
    assert "ratio_valid" not in spectra.colnames


def test_transmission_masks_zero_invalid_and_masked_samples_without_mutating_input(spectra):
    from astropy.table import MaskedColumn

    spectra["molecfit_transmission"][:4] = [0, -0.1, np.nan, np.inf]
    spectra["valid"][4] = False
    spectra.replace_column("pymolfit_transmission", MaskedColumn(spectra["pymolfit_transmission"]))
    spectra["pymolfit_transmission"].mask[5] = True
    original = spectra["valid"].copy()
    with np.errstate(divide="raise", invalid="raise"):
        result = comparison.compare_transmissions(spectra)
    assert not result["ratio_valid"][:6].any()
    assert np.isnan(result["relative_transmission_difference"][:6]).all()
    np.testing.assert_array_equal(spectra["valid"], original)


@pytest.mark.parametrize("minimum", [0, -1, 1.1, np.nan, np.inf])
def test_transmission_threshold_validation(spectra, minimum):
    with pytest.raises(ValueError, match="minimum transmission"):
        comparison.compare_transmissions(spectra, minimum)


def test_transmission_plot_preserves_gaps_and_deep_cores(spectra):
    import matplotlib.pyplot as plt

    spectra["wavelength_nm"][1000:] += 3
    spectra["pymolfit_transmission"][10] = 0.01
    result = comparison.compare_transmissions(spectra)
    fig, axes = plt.subplots(2, 1)
    try:
        comparison.plot_transmission_curves(axes, result)
        top = axes[0].lines[0]
        ratio = axes[1].lines[0]
        assert top.get_ydata()[10] == 0.01
        assert np.isnan(ratio.get_ydata()[10])
        assert np.isnan(top.get_xdata()[1000])
        assert np.isnan(top.get_ydata()[1000])
        np.testing.assert_array_equal(
            top.get_xdata()[np.isfinite(top.get_xdata())], spectra["wavelength_nm"]
        )
        assert len(top.get_xdata()) == len(spectra) + 1
    finally:
        plt.close(fig)


def test_transmission_plot_does_not_calibrate_or_refit(spectra, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Transmission plotting must not fit a source or atmosphere")

    monkeypatch.setattr(comparison, "run_corrections", unexpected)
    spectra["fit"] = np.arange(len(spectra)) % 2 == 0
    spectra["holdout"] = ~spectra["fit"]
    destination = tmp_path / "transmission.png"
    comparison.plot_transmissions(
        tmp_path, spectra, 0.1, False, figure_path=destination, wavelength_range=None
    )
    assert destination.exists()
    metrics = json.loads((tmp_path / "transmission_metrics.json").read_text())
    assert metrics["ratio_pixels"] == len(spectra)
    assert metrics["statistics"]["fit"]["pixels"] == len(spectra) // 2
    assert metrics["statistics"]["holdout"]["mean_absolute"] == 0
    assert not (tmp_path / "stellar_comparison.ecsv").exists()


def test_default_transmission_display_crops_but_retains_full_products(
    spectra, tmp_path, monkeypatch
):
    import matplotlib.figure

    spectra["wavelength_nm"] = np.linspace(550, 620, len(spectra))
    recorded = []

    def inspect(figure, *args, **kwargs):
        recorded.append(figure.axes[0].get_xlim())
        x = figure.axes[0].lines[0].get_xdata()
        assert np.nanmin(x) >= 560 and np.nanmax(x) <= 610

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", inspect)
    comparison.plot_transmissions(tmp_path, spectra, 0.1, False, figure_path=tmp_path / "plot.png")
    assert recorded == [(560, 610)]
    assert len(Table.read(tmp_path / "transmission_comparison.ecsv")) == len(spectra)


def test_transmission_statistics_with_no_usable_ratios(spectra, tmp_path):
    spectra["molecfit_transmission"] = 0
    result = comparison.save_transmission_comparison(tmp_path, spectra)
    metrics = json.loads((tmp_path / "transmission_metrics.json").read_text())
    assert metrics["ratio_pixels"] == 0
    assert metrics["statistics"]["all"]["mean"] is None
    assert result["transmission_valid"].all()


def test_old_species_cache_is_not_silently_reused(tmp_path, monkeypatch):
    (tmp_path / "molecfit").mkdir()
    (tmp_path / "molecfit/SCIENCE_TELLURIC_CORR_input.fits").touch()
    (tmp_path / "pymolfit.ecsv").touch()
    (tmp_path / "provenance.json").write_text(json.dumps({"species": ["H2O", "O2"]}))
    with pytest.raises(SystemExit) as error:
        comparison.main(["--data-dir", str(tmp_path)])
    assert error.value.code == 2


@pytest.fixture
def spectra():
    wave = np.linspace(400.0025, 409.9975, 2000)
    return Table(
        {
            "wavelength_nm": wave,
            "stellar_reference": np.ones(len(wave)),
            "valid": np.ones(len(wave), bool),
            "nonphotospheric_core": np.zeros(len(wave), bool),
            "pymolfit_transmission": np.ones(len(wave)),
            "molecfit_transmission": np.ones(len(wave)),
            "pymolfit_corrected": 1 + 0.02 * np.tile([-1, 1], 1000),
            "molecfit_corrected": np.full(len(wave), 1.01),
        }
    )


def test_molecfit_receives_explicit_good_pixel_mask(tmp_path, monkeypatch):
    wave = np.linspace(0.588, 0.591, 5)
    good = np.array([True, True, False, True, True])
    header = fits.Header({"SPEC_RES": 115000})
    columns = [
        fits.Column(name="lambda", format="D", unit="um", array=wave),
        fits.Column(name="flux", format="D", array=np.ones(5)),
        fits.Column(name="dflux", format="D", array=np.full(5, 0.01)),
        fits.Column(name="mask", format="J", array=(~good).astype(int)),
    ]
    fits.HDUList([fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns)]).writeto(
        tmp_path / "input.fits"
    )
    Table({"valid": good}).write(tmp_path / "input.ecsv")
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "fit_windows_nm": [[588, 591]],
                "excluded_ranges_micron": [],
                "species_policy": {
                    "included": ["CO", "H2O", "O2"],
                    "fitted": ["H2O", "O2"],
                    "fixed": {"CO": 1.0},
                },
            }
        )
    )
    calls = []

    def fit(output, spectrum, header, excluded, *, model_options):
        with fits.open(output / "input.fits") as hdus:
            np.testing.assert_array_equal(hdus[1].data["valid"], good.astype(int))
            np.testing.assert_array_equal(hdus[1].data["mask"], (~good).astype(int))
        assert model_options["COLUMN_MASK"] == "valid"
        assert model_options["LIST_MOLEC"] == "CO,H2O,O2"
        assert model_options["FIT_MOLEC"] == "0,1,1"
        assert model_options["REL_COL"] == "1,1,1"
        np.testing.assert_array_equal(spectrum.valid, good)
        calls.append(True)
        return [], "test-catalog"

    monkeypatch.setattr(comparison, "fit_molecfit", fit)
    comparison.run_molecfit_from_saved(tmp_path)
    comparison.run_molecfit_from_saved(tmp_path)
    assert calls == [True, True]


def test_collection_uses_effective_molecfit_weights(tmp_path):
    output = tmp_path / "molecfit"
    output.mkdir()
    wave = np.linspace(588, 591, 5)
    selected = np.array([False, True, True, False, False])
    flux, uncertainty, transmission = np.ones(5), np.full(5, 0.01), np.full(5, 0.9)
    Table(
        {
            "wavelength_nm": wave,
            "flux": flux,
            "uncertainty": uncertainty,
            "valid": np.ones(5, bool),
            "fit": selected,
        }
    ).write(tmp_path / "input.ecsv")
    Table(
        {
            "wavelength": wave / 1000,
            "flux": flux,
            "uncertainty": uncertainty,
            "fit_mask": selected,
            "transmission": transmission,
            "corrected_flux": flux / transmission,
        },
        meta={"fit_success": True},
    ).write(tmp_path / "pymolfit.ecsv")
    Table({"lambda": wave / 1000, "flux": flux / transmission}).write(
        output / "SCIENCE_TELLURIC_CORR_input.fits"
    )
    Table({"parameter": ["status"], "value": [2.0]}).write(output / "BEST_FIT_PARAMETERS.fits")
    model = Table(
        {
            "lambda": wave / 1000,
            "flux": flux,
            "weight": 1 / uncertainty,
            "mweight": selected.astype(float),
        }
    )
    model.write(output / "BEST_FIT_MODEL.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(transmission)]).writeto(
        output / "TELLURIC_CORR.fits"
    )
    result = comparison.collect_results(tmp_path)
    np.testing.assert_allclose(result["molecfit_corrected"], flux / transmission)
    model["mweight"][0] = 1
    model.write(output / "BEST_FIT_MODEL.fits", overwrite=True)
    with pytest.raises(ValueError, match="fit mask differs"):
        comparison.collect_results(tmp_path)
