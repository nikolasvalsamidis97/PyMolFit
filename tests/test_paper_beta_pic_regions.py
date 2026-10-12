"""Regional plots must reuse the all-species results without fitting or rescaling."""

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "compare_regions",
    Path(__file__).resolve().parents[1] / "paper_validation/beta_pic_regions/compare_regions.py",
)
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)


def test_default_region_is_only_na_d():
    assert list(regions.REGIONS) == ["nad"]
    assert regions.REGIONS["nad"][1] == (588.0, 591.0)


@pytest.fixture
def products(tmp_path):
    source = tmp_path / "source"
    (source / "molecfit").mkdir(parents=True)
    wave = np.concatenate([np.linspace(*limits, 401) for _, limits in regions.REGIONS.values()])
    table = Table(
        {
            "wavelength_nm": wave,
            "flux": np.ones(len(wave)),
            "valid": np.ones(len(wave), dtype=bool),
            "pymolfit_corrected": np.full(len(wave), 1.02),
            "molecfit_corrected": np.full(len(wave), 1.01),
            "pymolfit_transmission": np.full(len(wave), 1 / 1.02),
            "molecfit_transmission": np.full(len(wave), 1 / 1.01),
        }
    )
    table.write(source / "comparison.ecsv")
    table["stellar_reference"] = 1.0
    table["score_valid"] = True
    table.write(source / "stellar_comparison.ecsv")
    policy = {
        "included": ["CO", "H2O", "O2"],
        "fitted": ["H2O", "O2"],
        "fixed": {"CO": 1.0},
    }
    provenance = {
        "comparison_protocol": regions.full.COMPARISON_PROTOCOL,
        "date_obs": "2017-04-06T00:00:00",
        "pymolfit_settings": {},
        "species_policy": policy,
    }
    (source / "provenance.json").write_text(json.dumps(provenance))
    (source / "stellar_reference.json").write_text(
        json.dumps(
            {
                "input_table_sha256": regions.full.sha256(source / "comparison.ecsv"),
            }
        )
    )
    Table(
        {"sample": [1]},
        meta={
            "fit_success": True,
            "species_scales": json.dumps({"CO": 1, "H2O": 0.9, "O2": 1.1}),
            "parameter_names": json.dumps(["log_scale:H2O", "log_scale:O2"]),
            "provenance_json": json.dumps({"line_species": policy["included"]}),
        },
    ).write(source / "pymolfit.ecsv")
    Table(
        {
            "LIST_MOLEC": policy["included"],
            "FIT_MOLEC": [0, 1, 1],
            "REL_COL": [1, 1, 1],
        }
    ).write(source / "molecfit/MODEL_MOLECULES.fits")
    return source


def test_loads_all_species_without_requiring_stellar_reference(products):
    (products / "stellar_reference.json").unlink()
    (products / "stellar_comparison.ecsv").unlink()
    table, provenance = regions.load_products(products)
    assert len(table) == 401
    assert provenance["species_policy"]["included"] == ["CO", "H2O", "O2"]


@pytest.mark.parametrize(
    "change",
    ["old_protocol", "whitelist", "different_molecules"],
)
def test_rejects_incompatible_source_products(products, change):
    if change in ("old_protocol", "whitelist"):
        path = products / "provenance.json"
        provenance = json.loads(path.read_text())
        if change == "old_protocol":
            provenance["comparison_protocol"] = "water_only"
        else:
            provenance["pymolfit_settings"]["hitran_species"] = ["H2O"]
        path.write_text(json.dumps(provenance))
    elif change == "different_molecules":
        path = products / "molecfit/MODEL_MOLECULES.fits"
        Table.read(path)[1:].write(path, overwrite=True)
    with pytest.raises(ValueError):
        regions.load_products(products)


def test_crops_preserve_native_pixels_and_flux(products):
    table, _ = regions.load_products(products)
    for name, (_, (lower, upper)) in regions.REGIONS.items():
        crop = regions.crop_region(table, name)
        expected = table[(table["wavelength_nm"] >= lower) & (table["wavelength_nm"] <= upper)]
        assert len(crop) == 401
        for column in expected.colnames:
            np.testing.assert_array_equal(crop[column], expected[column])
    with pytest.raises(ValueError, match="complete"):
        regions.crop_region(table[1:], "nad")


def test_two_stars_show_original_flux_and_preserve_excluded_stellar_cores(products):
    table, provenance = regions.load_products(products)
    crops = {name: regions.crop_region(table, name) for name in regions.REGIONS}
    for crop in crops.values():
        crop["fit"] = True
        crop["fit"][10] = False
        crop["valid"][20] = False
        crop["pymolfit_transmission"][30] = 0.001
    aumic = (crops["nad"].copy(), {"date_obs": "2023-07-08"})
    fig = regions.make_figure(crops, provenance, aumic)
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 2)
        np.testing.assert_allclose(fig.get_size_inches(), [13, 4.3])
        assert "HARPS" in fig.axes[0].get_title()
        assert "ESPRESSO" in fig.axes[1].get_title()
        assert [text.get_text() for text in fig.legends[0].texts] == [
            "Uncorrected",
            "PyMolFit",
            "Molecfit",
        ]
        assert all(text.get_fontsize() == 12.55 for text in fig.legends[0].texts)
        for ax, crop in zip(fig.axes, [crops["nad"], aumic[0]], strict=True):
            assert ax.title.get_fontsize() == 13.7
            assert all(text.get_fontsize() == 11.7 for text in ax.get_xticklabels())
            assert all(text.get_fontsize() == 11.7 for text in ax.get_yticklabels())
            assert len(ax.lines) == 3
            for line, column in zip(
                ax.lines,
                ["flux", "pymolfit_corrected", "molecfit_corrected"],
                strict=True,
            ):
                np.testing.assert_array_equal(line.get_xdata(), crop["wavelength_nm"])
                expected_valid = np.asarray(crop["valid"]).copy()
                if column in ("pymolfit_corrected", "molecfit_corrected"):
                    expected_valid[30] = False
                np.testing.assert_allclose(
                    line.get_ydata()[expected_valid], crop[column][expected_valid]
                )
                np.testing.assert_array_equal(np.isfinite(line.get_ydata()), expected_valid)
                assert np.isfinite(line.get_ydata()[10])
                assert np.isnan(line.get_ydata()[20])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("with_aumic", [False, True])
def test_larger_text_fits_in_the_figure(products, with_aumic):
    table, provenance = regions.load_products(products)
    fig = regions.make_figure(
        {"nad": regions.crop_region(table, "nad")},
        provenance,
        (table, provenance) if with_aumic else None,
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.get_tightbbox(renderer)
        width, height = fig.get_size_inches()
        assert 0 <= bounds.x0 < bounds.x1 <= width
        assert 0 <= bounds.y0 < bounds.y1 <= height
        ylabel = next(text for text in fig.texts if text.get_text().startswith("Flux"))
        label_bounds = ylabel.get_window_extent(renderer)
        for ax in fig.axes:
            for tick in ax.get_yticklabels():
                assert not label_bounds.overlaps(tick.get_window_extent(renderer))
    finally:
        plt.close(fig)


def test_saves_only_one_png_and_opens_only_the_mosaic(products, tmp_path, monkeypatch):
    plt.close("all")
    saved, shown = [], []
    before = {path: regions.full.sha256(path) for path in products.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        matplotlib.figure.Figure, "savefig", lambda self, path, **kwargs: saved.append(path.name)
    )
    monkeypatch.setattr(plt, "show", lambda: shown.extend(plt.get_fignums()))
    monkeypatch.setattr(regions, "ensure_aumic", lambda *a, **kw: regions.load_products(products))
    output = tmp_path / "plots/regions_mosaic.png"
    regions.save_regions(products, output, show=True)
    assert len(shown) == 1
    assert saved == ["regions_mosaic.png"]
    assert not list(output.parent.iterdir())  # savefig is mocked; no auxiliary files.
    assert all(regions.full.sha256(path) == digest for path, digest in before.items())
    assert not plt.get_fignums()


def test_cli_opens_mosaic_by_default_and_can_save_without_showing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        regions, "save_regions", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    regions.main([])
    assert calls[0][0] == (regions.DEFAULT_SOURCE, regions.HERE / "regions_mosaic.png")
    assert calls[0][1]["show"] is True
    regions.main(["--no-show"])
    assert calls[1][0] == calls[0][0]
    assert calls[1][1]["show"] is False


def test_aumic_plot_only_requires_saved_fit(tmp_path):
    with pytest.raises(FileNotFoundError, match="not been fitted"):
        regions.ensure_aumic(tmp_path, tmp_path / "input", tmp_path / "template", plot_only=True)


def test_aumic_cached_fit_is_reused_and_input_changes_are_rejected(products, tmp_path):
    source_input, template = tmp_path / "raw.fits", tmp_path / "template.dat"
    source_input.write_bytes(b"unchanged observation")
    template.write_bytes(b"exclusion template")
    path = products / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance.update(
        comparison_protocol=regions.AUMIC_PROTOCOL,
        input_sha256=regions.full.sha256(source_input),
        template_sha256=regions.full.sha256(template),
        fit_windows_nm=[[592.05, 592.22]],
    )
    path.write_text(json.dumps(provenance))
    result, _ = regions.ensure_aumic(products, source_input, template, plot_only=True)
    assert len(result) == 401
    source_input.write_bytes(b"different observation")
    with pytest.raises(ValueError, match="input changed"):
        regions.ensure_aumic(products, source_input, template, plot_only=True)
