from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import matplotlib
import numpy as np
import pytest

from pymolfit import (
    AdvancedOptions,
    Observation,
    Spectrum,
    TelluricFitResult,
    TheoreticalSpectrum,
    correct,
    load_fit_product,
    load_spectrum,
    select_telluric_regions,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@pytest.mark.parametrize(
    ("api", "page"),
    [
        (correct, "correct.md"),
        (AdvancedOptions, "correct.md"),
        (Spectrum, "Spectrum.md"),
        (Observation, "Observation.md"),
        (TheoreticalSpectrum, "TheoreticalSpectrum.md"),
    ],
)
def test_parameter_reference_covers_current_signature(api, page):
    text = (DOCS / "parameter_descriptions" / page).read_text()
    for name in inspect.signature(api).parameters:
        assert re.search(rf"`(?:\*\*)?{name}`", text), (api.__name__, name)


@pytest.mark.parametrize(
    "api",
    [
        correct,
        AdvancedOptions,
        Spectrum,
        Observation,
        TheoreticalSpectrum,
        load_spectrum,
        load_fit_product,
        select_telluric_regions,
    ],
)
def test_user_entry_points_have_parameter_hover_help(api):
    doc = inspect.getdoc(api)
    for name in inspect.signature(api).parameters:
        assert f":param {name}:" in doc, (api.__name__, name)


def test_documentation_has_only_requested_parameter_pages():
    pages = {p.name for p in (DOCS / "parameter_descriptions").glob("*.md")}
    assert pages == {"correct.md", "Spectrum.md", "Observation.md", "TheoreticalSpectrum.md"}


def test_local_documentation_links_resolve():
    pages = [(path, path.read_text()) for path in [ROOT / "README.md", *DOCS.rglob("*.md")]]
    for path in DOCS.rglob("*.ipynb"):
        book = json.loads(path.read_text())
        pages.append(
            (
                path,
                "\n".join(
                    "".join(c["source"]) for c in book["cells"] if c["cell_type"] == "markdown"
                ),
            )
        )
    for path, text in pages:
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if link.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = unquote(link.split("#", 1)[0])
            assert (path.parent / target).exists(), (path, link)


@pytest.mark.parametrize("with_uncertainty", [False, True])
def test_output_handling_examples_load_plot_and_export_without_refitting(
    tmp_path, monkeypatch, with_uncertainty
):
    import pymolfit

    wave = np.array([0.588, 0.589, 0.590])
    transmission = np.array([0.9, 0.8, 0.95])
    continuum = np.full(3, 2.0)
    valid = np.array([True, False, True])
    error = np.full(3, 0.1) if with_uncertainty else None
    observed = Spectrum(wave, continuum * transmission, uncertainty=error, mask=valid)
    corrected = Spectrum(wave, continuum, uncertainty=error, mask=valid)
    result = TelluricFitResult(
        spectrum=observed,
        corrected=corrected,
        transmission=transmission,
        continuum=continuum,
        model_flux=observed.flux,
        species_scales={"H2O": 1.0},
        wavelength_shift=0.0,
        wavelength_coefficients=np.zeros(1),
        lsf_sigma_pixels=1.0,
        lsf_box_width_pixels=0.0,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_wavelength_exponent=0.0,
        continuum_coefficients=np.array([2.0]),
        metrics={},
        success=True,
        message="Test fixture",
        cost=0.0,
        nfev=1,
        fit_mask=np.array([True, False, False]),
    )
    output = tmp_path / "outputs/beta_pic_fits.ecsv"
    output.parent.mkdir()
    result.write(output)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pymolfit, "correct", lambda *a, **kw: pytest.fail("Must not refit"))
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    monkeypatch.setattr(plt.Figure, "savefig", lambda *a, **kw: pytest.fail("Must not save images"))
    page = DOCS / "user_guide/output_handling.md"
    namespace = {}
    try:
        for source in re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL):
            exec(compile(source, str(page), "exec"), namespace)
        np.testing.assert_allclose(namespace["wavelength"], wave * 1000)
        np.testing.assert_array_equal(namespace["uncorrected_flux"], observed.flux)
        np.testing.assert_array_equal(namespace["corrected_flux"], corrected.flux)
        np.testing.assert_array_equal(namespace["valid"], valid)
        if with_uncertainty:
            np.testing.assert_array_equal(namespace["uncertainty"], error)
        else:
            assert namespace["uncertainty"] is None
        exported = np.loadtxt(tmp_path / "corrected_spectrum.txt")
        np.testing.assert_allclose(exported[:, 0], wave[valid] * 1000)
        np.testing.assert_array_equal(exported[:, 1], corrected.flux[valid])
        assert len(shown) == 1
        assert len(shown[0].axes) == 2
    finally:
        for fig in shown:
            plt.close(fig)


@pytest.mark.parametrize(
    "name",
    [
        "user_guide/01_fits_correction.ipynb",
        "user_guide/02_fits_with_stellar_template.ipynb",
        "user_guide/03_wavelength_flux_input.ipynb",
        "user_guide/04_interactive_selector.ipynb",
        "troubleshooting/faster_correction.ipynb",
    ],
)
def test_notebooks_are_clean_and_use_current_api(name):
    notebook = json.loads((DOCS / name).read_text())
    assert notebook["nbformat"] == 4
    api = {
        obj.__name__: obj
        for obj in (correct, Spectrum, Observation, TheoreticalSpectrum, select_telluric_regions)
    }
    found_correction = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        source = "".join(cell["source"])
        assert "/Users/" not in source
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in api:
                signature = inspect.signature(api[node.func.id])
                for keyword in node.keywords:
                    assert keyword.arg in signature.parameters
                found_correction |= node.func.id == "correct"
    assert found_correction


def test_selector_notebook_opens_then_saves_then_corrects():
    book = json.loads((DOCS / "user_guide/04_interactive_selector.ipynb").read_text())
    sources = ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]
    assert "%matplotlib widget" in sources[0]
    editor = next(i for i, s in enumerate(sources) if "selector = select_telluric_regions(" in s)
    saved = next(i for i, s in enumerate(sources) if "selector.save(region_file)" in s)
    corrected = next(i for i, s in enumerate(sources) if "result = correct(" in s)
    assert editor < saved < corrected
    assert "show_interactive_window=True" in sources[editor]
    assert "region_file=region_file" in sources[corrected]


def test_pipeline_links_each_detailed_stage():
    pipeline = (DOCS / "technical_details/pipeline.md").read_text()
    for page in (
        "atmosphere_and_absorption.md",
        "segmentation_and_regions.md",
        "spectral_model.md",
        "parameter_estimation.md",
        "correction_and_uncertainties.md",
    ):
        assert f"]({page})" in pipeline
    assert "evaluation limit" in pipeline
    assert "not another fit" in pipeline


def test_troubleshooting_items_are_self_contained_notebooks():
    files = list((DOCS / "troubleshooting").iterdir())
    assert files
    assert all(path.suffix == ".ipynb" for path in files)
    for path in files:
        book = json.loads(path.read_text())
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                assert "savefig" not in source
                assert "output_path=" not in source


@pytest.mark.parametrize("working_directory", [ROOT, DOCS / "troubleshooting"])
@pytest.mark.parametrize("with_uncertainty", [False, True])
def test_speed_notebook_compares_full_input_on_common_fitting_pixels(
    monkeypatch, working_directory, with_uncertainty
):
    import pymolfit

    assert not (DOCS / "troubleshooting/zigzag_residuals.ipynb").exists()
    path = DOCS / "troubleshooting/faster_correction.ipynb"
    book = json.loads(path.read_text())
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 3
    sources = ["".join(c["source"]) for c in cells]
    assert "TheoreticalSpectrum" not in "".join(sources)
    assert "replace(" not in "".join(sources)
    wave = np.linspace(3781.75, 6912.64, 61)
    valid = np.ones(wave.size, dtype=bool)
    valid[25] = False
    spectrum = Spectrum(
        wave,
        np.full(wave.size, 1.96),
        uncertainty=np.full(wave.size, 0.01) if with_uncertainty else None,
        mask=valid,
        group_id=np.where(wave < 6890, "a", "b"),
        wavelength_unit="angstrom",
        wavelength_medium="air",
        observation=Observation(metadata={"AIRMASS": 1.2}),
        meta={"name": "Example"},
    )
    calls = []
    results = []

    def load_example(input_path, **kwargs):
        assert input_path.resolve() == DOCS / "user_guide/data/ADP.2017-04-07T01_04_41.632.fits"
        assert input_path.is_file()
        assert kwargs == {"wavelength_medium": "air", "save_header": False}
        return spectrum

    def correct_example(input_path, **kwargs):
        calls.append((input_path, kwargs))
        observed = input_path.to_vacuum().to_unit("micron")
        observed = observed.with_flux(observed.flux, uncertainty=np.full(wave.size, 0.01))
        corrected = Spectrum(
            observed.wavelength,
            np.full(observed.wavelength.size, 2.02 if len(calls) == 1 else 2.0),
            mask=observed.valid & (observed.wavelength != observed.wavelength[10]),
            wavelength_unit="micron",
        )
        fit_mask = np.arange(wave.size) % (2 if len(calls) == 1 else 3) == 0
        # Different error estimates must not change the residual normalisation.
        model_flux = observed.flux - (0.01 if len(calls) == 1 else 0.005)
        model_flux[0 if len(calls) == 1 else 6] = np.nan
        if len(calls) == 2:
            observed = observed.with_flux(observed.flux, uncertainty=np.full(wave.size, 0.02))
        result = SimpleNamespace(
            spectrum=observed,
            corrected=corrected,
            fit_mask=fit_mask,
            model_flux=model_flux,
            success=True,
        )
        results.append(result)
        return result

    monkeypatch.chdir(working_directory)
    monkeypatch.setattr(pymolfit, "load_spectrum", load_example)
    monkeypatch.setattr(pymolfit, "correct", correct_example)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda *a, **kw: pytest.fail("Example must not save images")
    )
    namespace = {}
    try:
        for source in sources:
            source = "\n".join(s for s in source.splitlines() if not s.startswith("%"))
            exec(compile(source, str(path), "exec"), namespace)
        assert len(calls) == 2
        assert len(shown) == 1
        assert calls[0][0] is calls[1][0] is spectrum
        assert calls[0][1] == {"report": False}
        assert calls[1][1]["report"] is False
        options = calls[1][1]["advanced"]
        assert options.continuum_order == 2
        assert options.wavelength_order == 1
        assert options.fit_lsf_lorentz_fwhm is False
        assert options.lsf_variable_width is True
        assert options.f_scale == 1.0
        assert options.fit_ranges is None
        assert options.automatic_fit_regions is True
        assert options.fit_lsf_sigma == "auto"
        assert options.ftol == options.xtol == options.gtol == 1e-10
        axes = shown[0].axes
        assert len(axes) == 3
        assert axes[1].get_ylim() == axes[2].get_ylim()
        assert axes[0].get_xlim() == axes[1].get_xlim() == axes[2].get_xlim()
        assert axes[2].get_xlabel() == "Observatory-frame vacuum wavelength [nm]"
        assert axes[0].get_ylabel() == "Flux"
        assert len(axes[0].lines) == 3
        assert axes[0].lines[0].get_label() == "Uncorrected"
        expected_wave = results[0].spectrum.to_unit("nm").wavelength
        for line in axes[0].lines:
            np.testing.assert_allclose(line.get_xdata(), expected_wave)
        np.testing.assert_allclose(
            axes[0].lines[0].get_ydata(), np.where(spectrum.valid, spectrum.flux, np.nan)
        )
        for line, result in zip(axes[0].lines[1:], results, strict=True):
            np.testing.assert_allclose(
                line.get_ydata(), np.where(result.corrected.valid, result.corrected.flux, np.nan)
            )
        common = spectrum.valid & results[0].fit_mask & results[1].fit_mask
        common &= np.isfinite(results[0].model_flux) & np.isfinite(results[1].model_flux)
        assert common.any()
        for ax, expected in zip(axes[1:], (1.0, 0.5), strict=True):
            assert len(ax.lines) == 2  # Residual samples and a zero reference line.
            assert ax.get_ylabel() == "Residual / uncertainty"
            np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_wave)
            np.testing.assert_allclose(ax.lines[0].get_ydata(), np.where(common, expected, np.nan))
        assert all(ax.get_legend() is not None for ax in axes)
    finally:
        for fig in shown:
            plt.close(fig)


def test_oxygen_example_preserves_original_observed_samples():
    data = DOCS / "user_guide/data"
    full = load_spectrum(data / "ADP.2017-04-07T01_04_41.632.fits", save_header=False)
    crop = load_spectrum(data / "harps_oxygen_crop_air.fits", save_header=False)
    keep = (full.wavelength > 6270) & (full.wavelength < 6310)
    np.testing.assert_array_equal(crop.wavelength, full.wavelength[keep])
    np.testing.assert_array_equal(crop.flux, full.flux[keep])
    if full.uncertainty is None:
        assert crop.uncertainty is None
    else:
        np.testing.assert_array_equal(crop.uncertainty, full.uncertainty[keep])
    assert crop.wavelength_unit == full.wavelength_unit
    assert crop.wavelength_medium == full.wavelength_medium


def test_array_example_retains_the_fits_samples_and_metadata():
    data = DOCS / "user_guide/data"
    spectrum = load_spectrum(data / "harps_nad_crop_air.fits", save_header=False)
    wavelength, flux = np.loadtxt(data / "beta_pic_nad.txt", unpack=True)
    np.testing.assert_allclose(wavelength, spectrum.to_unit("nm").wavelength, rtol=1e-12)
    np.testing.assert_allclose(flux, spectrum.flux, rtol=1e-12)
    assert spectrum.wavelength_medium == "air"
    template = TheoreticalSpectrum(
        data / "beta_pic_template.txt",
        radial_velocity_kms=20,
        vsini_kms=130,
        wavelength_unit="angstrom",
        wavelength_medium="air",
    )
    assert template.wavelength.min() < wavelength.min() * 10
    assert template.wavelength.max() > wavelength.max() * 10


@pytest.mark.parametrize("kind", ("nad_correction", "transmission"))
def test_figure_examples_use_saved_arrays_without_refitting(kind, tmp_path, monkeypatch):
    path = DOCS / "figure_examples" / f"plot_{kind}.py"
    spec = importlib.util.spec_from_file_location(f"_doc_{kind}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wave = np.linspace(588.4, 590.8, 101)
    transmission = 1 - 0.1 * np.exp(-(((wave - 589.5) / 0.02) ** 2))
    mask = np.ones(wave.size, bool)
    mask[3] = False
    observed = Spectrum(wave, 2 * transmission, mask=mask, wavelength_unit="nm")
    corrected = Spectrum(wave, np.full(wave.size, 2.0), mask=mask, wavelength_unit="nm")
    result = SimpleNamespace(spectrum=observed, corrected=corrected, transmission=transmission)
    monkeypatch.setattr(module, "load_fit_product", lambda _: result)
    destination = tmp_path / f"{kind}.png"
    fig = getattr(module, f"plot_{kind}")(
        tmp_path / "saved.ecsv", output_path=destination, show=False
    )
    try:
        assert destination.is_file()
        assert len(fig.axes) == 1
        if kind == "nad_correction":
            before, after = fig.axes[0].lines
            np.testing.assert_allclose(before.get_ydata()[mask], observed.flux[mask])
            np.testing.assert_allclose(after.get_ydata()[mask], corrected.flux[mask])
            assert np.isnan(before.get_ydata()[3])
            assert np.isnan(after.get_ydata()[3])
        else:
            np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata()[mask], transmission[mask])
    finally:
        plt.close(fig)
