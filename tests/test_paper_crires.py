"""Common-pixel and held-out checks for the second-instrument comparison."""

import importlib.util
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "compare_crires",
    Path(__file__).resolve().parents[1] / "paper_validation/crires_comparison/compare_crires.py",
)
crires = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crires)


def test_direct_run_defaults_to_updated_mask_and_separate_cache():
    assert crires.DEFAULT_FIT_EDGE_PIXELS == 30
    assert crires.DEFAULT_DATA == crires.REPO / "validation_results/crires_full_matched_continua"
    assert crires.DEFAULT_DATA != crires.REPO / "validation_results/crires_full"


@pytest.fixture
def table():
    wave = np.linspace(3513.2, 3535.8, 1000)
    holdout = crires.full.interval_mask(wave, crires.heldout_intervals(wave))
    return Table(
        {
            "wavelength_nm": wave,
            "flux": np.ones(1000),
            "uncertainty": np.full(1000, 0.01),
            "valid": np.ones(1000, bool),
            "fit": ~holdout,
            "holdout": holdout,
            "pymolfit_transmission": np.full(1000, 0.8),
            "molecfit_transmission": np.full(1000, 0.79),
            "pymolfit_corrected": np.full(1000, 1.25),
            "molecfit_corrected": np.full(1000, 1 / 0.79),
            "pymolfit_model": np.full(1000, 0.99),
            "molecfit_model": np.full(1000, 0.98),
        }
    )


def test_fractional_holdouts_are_disjoint_and_independent_of_flux(table):
    assert not np.any(table["fit"] & table["holdout"])
    assert np.all(table["fit"] | table["holdout"])
    assert table["holdout"].sum() == 150
    intervals = crires.heldout_intervals(table["wavelength_nm"])
    assert len(intervals) == 3
    assert all(b < c for (_, b), (c, _) in pairwise(intervals))


def test_continuum_component_is_mapped_to_molecfit_molecule_name():
    result = SimpleNamespace(
        provenance={"line_species": ["CH4", "H2O"]},
        species_scales={"CH4": 1.1, "H2O": 0.7, "N2_continuum": 1.0},
        parameter_names=["log_scale:CH4", "log_scale:H2O"],
    )
    policy = crires.molecular_policy(result)
    assert policy["included"] == ["CH4", "H2O", "N2"]
    assert policy["fitted"] == ["CH4", "H2O"]
    assert policy["fixed"] == {"N2": 1.0}


@pytest.mark.parametrize("edge_pixels", (0, 30))
@pytest.mark.parametrize("matched", (False, True))
def test_prepare_preserves_native_samples_errors_and_common_masks(
    tmp_path, monkeypatch, edge_pixels, matched
):
    wave = np.linspace(4043.3, 4068.7, 2048) if matched else np.linspace(3512.8, 3536.2, 2048)
    flux = np.linspace(100, 200, len(wave))
    error = np.full(len(wave), 2.0)
    error[1000] = 0
    header = fits.Header(
        {
            "INSTRUME": "CRIRES",
            "DATE-OBS": "2022-04-05T00:02:38",
            "HIERARCH ESO INS SLIT1 WID": 0.2,
            "OBJECT": "Unknown extraction",
        }
    )
    columns = [
        fits.Column(name=name, format="D", array=values)
        for name, values in (("08_01_WL", wave), ("08_01_SPEC", flux), ("08_01_ERR", error))
    ]
    path = tmp_path / "source.fits"
    fits.HDUList([fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns)]).writeto(
        path
    )
    gdas = tmp_path / "source_gdas.fits"
    fits.PrimaryHDU().writeto(gdas)
    metadata = {"gdas_profile": str(gdas), "airmass": 1.2765}
    monkeypatch.setattr(
        crires.full.AtmosphereProfile,
        "from_fits_header_mipas_gdas",
        lambda *a, **kw: SimpleNamespace(metadata=metadata),
    )
    output = tmp_path / "products"
    output.mkdir()
    saved_header, provenance = crires.prepare(
        path, output, "chip1_08_01", edge_pixels, matched=matched
    )
    table = Table.read(output / "input.ecsv")
    scale = provenance["constant_flux_scale"]
    np.testing.assert_array_equal(table["wavelength_nm"], wave)
    valid = error > 0
    np.testing.assert_allclose(table["flux"][valid] * scale, flux[valid])
    np.testing.assert_allclose(table["uncertainty"][valid] * scale, error[valid])
    np.testing.assert_allclose(table["original_flux"] * scale, flux)
    np.testing.assert_allclose(table["original_uncertainty"] * scale, error)
    assert np.all(np.isfinite(table["flux"]))
    assert np.all(np.asarray(table["uncertainty"]) > 0)
    assert provenance["native_pixels"] == 2048
    assert not np.any(table["fit"] & table["holdout"])
    edge_mask = crires.full.interval_mask(wave / 1000, crires.edge_intervals(wave, edge_pixels))
    source_mask = ((wave >= 4050) & (wave <= 4056)) if matched else np.zeros(len(wave), bool)
    np.testing.assert_array_equal(table["source_excluded"], source_mask)
    np.testing.assert_array_equal(
        table["fit"] | table["holdout"], valid & ~edge_mask & (~source_mask | table["holdout"])
    )
    np.testing.assert_array_equal(
        table["holdout"], valid & crires.full.interval_mask(wave, crires.heldout_intervals(wave))
    )
    np.testing.assert_array_equal(
        table["fit"],
        valid & ~crires.full.interval_mask(wave / 1000, provenance["excluded_ranges_micron"]),
    )
    assert np.count_nonzero(edge_mask) == 2 * edge_pixels
    assert provenance["fit_edge_pixels"] == edge_pixels
    with fits.open(output / "input.fits") as hdus:
        assert saved_header["SPECSYS"] == "TOPOCENT"
        assert saved_header["SPEC_RES"] == 86000
        np.testing.assert_array_equal(hdus[1].data["mask"], ~table["valid"])
        np.testing.assert_array_equal(hdus[1].data["valid"], table["valid"])
    assert "Nominal" in provenance["resolution_source"]
    assert provenance["object_header"] == "Unknown extraction"


def test_discovers_every_detector_and_order_without_a_wavelength_whitelist(tmp_path):
    hdus = [fits.PrimaryHDU()]
    for chip in range(3):
        columns = []
        for order in ("03_01", "04_01", "06_01", "07_01", "08_01"):
            lo = 3500 + 100 * int(order[:2]) + chip * 10
            for suffix, values in (("_WL", [lo, lo + 5]), ("_SPEC", [1, 2]), ("_ERR", [0.1, 0.1])):
                columns.append(fits.Column(name=order + suffix, format="D", array=values))
        hdus.append(fits.BinTableHDU.from_columns(columns))
    path = tmp_path / "all.fits"
    fits.HDUList(hdus).writeto(path)
    orders = crires.discover_orders(path)
    assert len(orders) == 15
    assert {row[0] for row in orders.values()} == {1, 2, 3}
    assert {row[1] for row in orders.values()} == {"03_01", "04_01", "06_01", "07_01", "08_01"}
    assert [row[2][0] for row in orders.values()] == sorted(row[2][0] for row in orders.values())


def test_full_plot_cannot_silently_reuse_only_a_subset(tmp_path):
    source = tmp_path / "input.fits"
    source.write_bytes(b"input checksum")
    orders = {"chip1_08_01": (1, "08_01", (3512, 3536)), "chip3_03_01": (3, "03_01", (5180, 5210))}
    with pytest.raises(FileNotFoundError, match="Missing 2 of 2 orders"):
        crires.ensure_corrections(source, tmp_path / "fits", orders, plot_only=True)


@pytest.mark.parametrize("saved_edges", (None, 30))
def test_edge_setting_cannot_silently_reuse_original_fits(tmp_path, saved_edges):
    source = tmp_path / "input.fits"
    source.write_bytes(b"input checksum")
    data = tmp_path / "fits"
    output = data / "chip1_08_01"
    output.mkdir(parents=True)
    (output / "comparison.ecsv").write_bytes(b"already fitted")
    saved = {"input_sha256": crires.full.sha256(source), "comparison_protocol": crires.PROTOCOL}
    if saved_edges is not None:
        saved["fit_edge_pixels"] = saved_edges
    path = output / "provenance.json"
    path.write_text(json.dumps(saved))
    orders = {"chip1_08_01": (1, "08_01", (3512, 3536))}
    current = saved_edges or 0
    inventory = crires.ensure_corrections(
        source, data, orders, plot_only=True, fit_edge_pixels=current
    )
    assert inventory["orders"]["chip1_08_01"]["status"] == "complete"
    with pytest.raises(ValueError, match="another --data-dir"):
        crires.ensure_corrections(source, data, orders, plot_only=True, fit_edge_pixels=current + 1)
    assert json.loads(path.read_text()) == saved


def test_combined_products_preserve_order_identity_and_native_samples(table):
    second = table.copy()
    second["wavelength_nm"] += 1000
    products = [
        (table, {"detector_hdu": 1, "column_prefix": "08_01", "constant_flux_scale": 20.0}),
        (second, {"detector_hdu": 3, "column_prefix": "04_01", "constant_flux_scale": 5.0}),
    ]
    combined = crires.combine_order_tables(products)
    assert len(combined) == 2 * len(table)
    np.testing.assert_array_equal(combined["wavelength_nm"][: len(table)], table["wavelength_nm"])
    np.testing.assert_array_equal(combined["flux"][len(table) :], second["flux"])
    assert set(combined["detector_order"][: len(table)]) == {"chip1_08_01"}
    assert set(combined["detector_order"][len(table) :]) == {"chip3_04_01"}
    assert np.all(combined["constant_flux_scale"][: len(table)] == 20)
    assert np.all(combined["constant_flux_scale"][len(table) :] == 5)


def test_adjacent_detectors_keep_physical_gaps_and_only_large_gaps_are_compressed(table):
    products = [
        (table, {"range_nm": [3513, 3536]}),
        (table, {"range_nm": [3538, 3560]}),
        (table, {"range_nm": [3747, 3772]}),
    ]
    fig, axes = plt.subplots(2, 1, sharex=True)
    try:
        offsets = crires.configure_wavelength_axes(axes, products)
        assert offsets[0] == offsets[1]
        assert offsets[2] != offsets[1]
        assert len([line for line in axes[0].lines if line.get_gid() == "wavelength-gap"]) == 1
        assert (3538 + offsets[1]) - (3536 + offsets[0]) == 2
    finally:
        plt.close(fig)


def test_display_ranges_crop_native_samples_without_changing_products(table):
    products = []
    for lo, hi in ((3513, 3536), (3747, 3771), (4070, 4095), (5150, 5177)):
        part = table.copy()
        part["wavelength_nm"] = np.linspace(lo, hi, len(part))
        products.append((part, {"range_nm": [lo, hi]}))
    cropped = crires.crop_display_products(products, crires.DISPLAY_RANGES_NM)
    assert len(cropped) == 2
    np.testing.assert_array_equal(cropped[0][0]["flux"], products[1][0]["flux"])
    np.testing.assert_array_equal(cropped[1][0]["flux"], products[2][0]["flux"])
    assert products[2][1]["range_nm"] == [4070, 4095]
    assert len(products[2][0]) == len(table)
    fig = crires.make_transmission_figure(products, display_ranges=crires.DISPLAY_RANGES_NM)
    try:
        fig.canvas.draw()
        labels = [t for t in fig.axes[-1].get_xticklabels() if t.get_visible()]
        boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in labels]
        assert all(not a.overlaps(b) for a, b in pairwise(boxes))
        assert sum(x.get_gid() == "wavelength-gap" for x in fig.axes[0].lines) == 1
    finally:
        plt.close(fig)


def test_transmission_figure_uses_native_ratios_and_preserves_order_gaps(table):
    second = table.copy()
    second["wavelength_nm"] += 500
    table["pymolfit_transmission"][5] = 0.01
    fig = crires.make_transmission_figure(
        [
            (table, {"range_nm": [3513.2, 3535.8]}),
            (second, {"range_nm": [4013.2, 4035.8]}),
        ]
    )
    try:
        assert len(fig.axes) == 2
        top, bottom = fig.axes
        curves = {line.get_gid(): line for line in top.lines}
        ratios = {line.get_gid(): line for line in bottom.lines}
        left = curves["transmission:pymolfit:0"]
        right = curves["transmission:pymolfit:1"]
        assert len(left.get_xdata()) == len(table)
        assert left.get_ydata()[5] == 0.01
        assert np.isnan(ratios["ratio:0"].get_ydata()[5])
        np.testing.assert_allclose(ratios["ratio:1"].get_ydata(), 0.8 / 0.79 - 1)
        divider = curves["wavelength-gap"].get_xdata()[0]
        assert left.get_xdata().max() < divider < right.get_xdata().min()
        for ax in fig.axes:
            assert sum(line.get_gid() == "axis-break" for line in ax.lines) == 1
        assert [text.get_text() for text in top.get_legend().texts] == ["PyMolFit", "Molecfit"]
        assert not fig._supxlabel
    finally:
        plt.close(fig)


@pytest.mark.parametrize("edge_pixels", (None, 0, 30))
def test_default_transmission_view_uses_saved_fits_without_reference(
    table, tmp_path, monkeypatch, edge_pixels
):
    table["wavelength_nm"] = np.linspace(3750, 3775, len(table))
    monkeypatch.setattr(crires, "DEFAULT_INPUT", tmp_path / "missing.fits")
    monkeypatch.setattr(crires, "DEFAULT_DATA", tmp_path)
    monkeypatch.setattr(crires, "HERE", tmp_path)
    monkeypatch.setattr(crires, "discover_orders", lambda path: {"order": None})

    def saved_only(*args, **kwargs):
        assert kwargs["plot_only"] and not kwargs["refit"]
        assert kwargs["fit_protocol"] == "matched"
        assert kwargs["fit_edge_pixels"] == (30 if edge_pixels is None else edge_pixels)
        return {"orders": {"order": {}}}

    monkeypatch.setattr(crires, "ensure_corrections", saved_only)
    monkeypatch.setattr(
        crires,
        "load_comparison",
        lambda path: (
            table,
            {
                "native_pixels": len(table),
                "median_input_snr": 100,
                "detector_hdu": 1,
                "column_prefix": "01_01",
                "constant_flux_scale": 1,
                "range_nm": [3750, 3775],
            },
        ),
    )
    args = ["--plot-only", "--no-show"]
    if edge_pixels is not None:
        args += ["--fit-edge-pixels", str(edge_pixels)]
    crires.main(args)
    assert (tmp_path / "crires_comparison.png").is_file()
    assert (tmp_path / "transmission_comparison.ecsv").is_file()
    assert (tmp_path / "transmission_metrics.json").is_file()
    assert not (tmp_path / "reference_comparison.ecsv").exists()


def test_fit_quality_keeps_fixed_pixels_and_reports_source_holdouts(table):
    table["source_excluded"] = False
    selected = np.flatnonzero(table["holdout"])[0]
    table["source_excluded"][selected] = True
    table["pymolfit_model"][selected] = 0.5
    table["pymolfit_transmission"][selected] = 0.01
    quality = crires.fit_quality(table)
    assert quality["pymolfit"]["heldout"]["rms_sigma"] == pytest.approx(1)
    assert quality["pymolfit"]["heldout_including_source"]["rms_sigma"] > 4
    assert quality["pymolfit"]["all"]["pixels"] == len(table)
    for subset in ("fit", "heldout", "heldout_including_source", "all"):
        assert quality["pymolfit"][subset]["pixels"] == quality["molecfit"][subset]["pixels"]


def test_cache_rejects_legacy_results_for_matched_fit(tmp_path):
    source = tmp_path / "input.fits"
    source.write_bytes(b"common input")
    directory = tmp_path / "fits" / "order"
    directory.mkdir(parents=True)
    (directory / "comparison.ecsv").write_text("saved legacy result")
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "input_sha256": crires.full.sha256(source),
                "comparison_protocol": crires.PROTOCOL,
                "fit_edge_pixels": 30,
            }
        )
    )
    with pytest.raises(ValueError, match="Incompatible saved"):
        crires.ensure_corrections(
            source,
            directory.parent,
            {"order": (1, "06_01", (4043, 4069))},
            plot_only=True,
            fit_edge_pixels=30,
            fit_protocol="matched",
        )
