import ast
import inspect
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from pymolfit import (
    AdvancedOptions,
    CacheConfig,
    Observation,
    RegionSelection,
    Spectrum,
    correct,
    correct_arrays,
    correct_file,
    workflow,
)
from pymolfit.provenance import provenance_json
from tests.helpers import demo_lines

REMOVED = {
    "species",
    "line_list",
    "hitran_par",
    "partition_table",
    "demo_line_list",
    "hitran_min_strength",
    "hitran_max_lines",
    "min_line_strength",
    "max_lines",
    "atmosphere",
    "atmosphere_mode",
    "mipas_profile",
    "gdas_profile",
    "airmass",
    "pressure_atm",
    "temperature_k",
    "mixing_ratios",
    "components",
    "physical",
    "h2o_continuum",
    "co2_continuum",
    "h2o_continuum_foreign_closure",
    "o2_cia",
    "n2_cia",
    "cia_tables",
    "rayleigh_xrayl",
    "n2_continuum",
    "o2_continuum",
    "o3_continuum",
    "n2_continuum_xn2cn",
    "o2_continuum_xo2cn",
    "o3_continuum_xo3cn",
    "high_resolution_grid",
    "line_wing_mode",
    "joint_stellar_model",
    "stellar_mask_path",
    "output_format",
    "fit_wavelength_shift",
    "fit_wavelength_polynomial",
    "fit_segment_wavelength_polynomial",
    "fit_segment_wavelength_shifts",
}


def test_fit_options_have_defaults_and_editor_descriptions():
    lower = inspect.signature(correct_file).parameters
    main = inspect.signature(correct).parameters
    for cls in (AdvancedOptions, CacheConfig):
        for field in fields(cls):
            assert field.name not in main
            if field.name != "wavelength_order":
                assert getattr(cls(), field.name) == lower[field.name].default
            assert f":param {field.name}:" in inspect.getdoc(cls)
    assert AdvancedOptions().wavelength_order == "auto"
    body = next(
        node
        for node in ast.parse(inspect.getsource(AdvancedOptions)).body
        if isinstance(node, ast.ClassDef)
    ).body
    for index, node in enumerate(body):
        if isinstance(node, ast.AnnAssign):
            following = body[index + 1]
            assert isinstance(following, ast.Expr)
            assert isinstance(following.value, ast.Constant)
            assert isinstance(following.value.value, str)
    for name in ("min_transmission", "estimate_uncertainties"):
        assert name in main
        assert name not in inspect.signature(AdvancedOptions).parameters


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_physics_and_removed_controls_are_not_public(name):
    assert name not in inspect.signature(AdvancedOptions).parameters
    assert f":param {name}:" not in inspect.getdoc(AdvancedOptions)
    with pytest.raises(TypeError, match=name):
        AdvancedOptions(**{name: None})


def test_option_objects_are_keyword_only_and_immutable():
    with pytest.raises(TypeError):
        AdvancedOptions(1e-8)
    with pytest.raises(FrozenInstanceError):
        AdvancedOptions().ftol = 1e-8
    with pytest.raises(FrozenInstanceError):
        CacheConfig().aer_offline = True


def test_reduced_interface_preserves_internal_defaults(monkeypatch):
    monkeypatch.setattr(workflow, "correct_file", lambda **kw: kw)
    plain = correct("input.fits")
    grouped = correct("input.fits", advanced=AdvancedOptions())
    assert plain == grouped
    lower = inspect.signature(correct_file).parameters
    for name, value in plain.items():
        if name in lower and name not in {"input_path", "automatic_fit_regions"}:
            assert value == lower[name].default
    assert plain["hitran_species"] is None
    assert plain["automatic_fit_regions"] is True
    assert plain["rayleigh"] is False
    for species in ("n2", "o2", "o3"):
        assert plain[f"{species}_continuum"] is False
    for name in (
        "rayleigh_xrayl",
        "n2_continuum_xn2cn",
        "o2_continuum_xo2cn",
        "o3_continuum_xo3cn",
    ):
        assert plain[name] == 1.0


@pytest.mark.parametrize("scope", ["group", "global"])
@pytest.mark.parametrize("order", [None, "auto", 0, 1, 2, 3])
def test_wavelength_model_maps_to_exclusive_flags(monkeypatch, scope, order):
    monkeypatch.setattr(workflow, "correct_file", lambda **kw: kw)
    result = correct(
        "input.fits",
        advanced=AdvancedOptions(
            wavelength_order=order,
            wavelength_scope=scope,
            initial_wavelength_shift=1e-6,
            wavelength_shift_bounds=(-2e-5, 2e-5),
        ),
    )
    assert result["wavelength_scope"] == scope
    assert result["initial_wavelength_shift"] == 1e-6
    assert result["wavelength_shift_bounds"] == (-2e-5, 2e-5)
    flags = {
        name: result[name]
        for name in (
            "fit_wavelength_shift",
            "fit_wavelength_polynomial",
            "fit_segment_wavelength_shifts",
            "fit_segment_wavelength_polynomial",
        )
    }
    if order == "auto":
        assert flags["fit_wavelength_shift"] == "auto"
        assert sum(bool(value) for value in flags.values()) == 1
    elif order is None:
        assert not any(flags.values())
    else:
        expected = (
            ("fit_segment_wavelength_shifts" if order == 0 else "fit_segment_wavelength_polynomial")
            if scope == "group"
            else ("fit_wavelength_shift" if order == 0 else "fit_wavelength_polynomial")
        )
        assert flags[expected] is True
        assert sum(flags.values()) == 1
        assert result["wavelength_polynomial_order"] == order
        assert result["segment_wavelength_polynomial_order"] == order


@pytest.mark.parametrize("bad", [-1, True, 1.5, "linear"])
def test_invalid_wavelength_order_fails_before_loading(bad):
    with pytest.raises(ValueError, match="wavelength_order"):
        correct("missing.fits", advanced=AdvancedOptions(wavelength_order=bad))


def test_invalid_scope_fails_before_loading():
    with pytest.raises(ValueError, match="wavelength_scope"):
        correct("missing.fits", advanced=AdvancedOptions(wavelength_scope="segment"))


@pytest.mark.parametrize(
    "name",
    sorted(
        REMOVED
        | {
            "spectrum",
            "wavelength",
            "flux",
            "uncertainty",
            "mask",
            "group_id",
            "observation",
            "continuum_order",
            "ftol",
            "aer_offline",
            "line_list_path",
            "plot_path",
        }
    ),
)
def test_removed_flat_keywords_are_rejected(name):
    with pytest.raises(TypeError, match=name):
        correct("missing.fits", **{name: None})


def test_correct_has_no_catch_all_keywords():
    assert all(
        p.kind is not inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(correct).parameters.values()
    )


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_evaluation_limit_is_validated(monkeypatch, bad):
    spectrum = Spectrum(
        np.linspace(0.58, 0.59, 30),
        np.ones(30),
        observation=Observation(wavelength_frame="observatory"),
    )
    monkeypatch.setattr(
        workflow, "_resolve_line_list", lambda *a, **kw: pytest.fail("late validation")
    )
    with pytest.raises(ValueError, match="max_nfev"):
        correct(spectrum, advanced=AdvancedOptions(max_nfev=bad), report=False)


def test_column_guesses_and_bounds_reach_the_optimizer():
    from pymolfit.fit import FitConfig, _species_scale_setup

    names, fixed, start, lower, upper = _species_scale_setup(
        ("H2O", "O2"),
        FitConfig(
            initial_species_scales={"H2O": 1.7},
            fixed_species_scales={"O2": 0.9},
            scale_bounds=(0.5, 2.0),
        ),
    )
    assert names == ("H2O",)
    assert fixed == {"O2": 0.9}
    np.testing.assert_allclose(np.exp(start), [1.7])
    np.testing.assert_allclose(np.exp(lower), [0.5])
    np.testing.assert_allclose(np.exp(upper), [2.0])
    for bad in (0.0, -1.0, np.nan, np.inf, 3.0):
        with pytest.raises(ValueError, match="initial scale"):
            _species_scale_setup(
                ("H2O",), FitConfig(initial_species_scales={"H2O": bad}, scale_bounds=(0.5, 2.0))
            )


@pytest.mark.parametrize("route", ["file", "spectrum"])
def test_fit_controls_reach_each_route(monkeypatch, route):
    observation = Observation(wavelength_frame="observatory", pwv_mm=2.3)
    spectrum = Spectrum(np.linspace(0.58, 0.59, 30), np.ones(30), observation=observation)
    initial = {"H2O": 1.2}
    fixed = {"CO2": 0.9}
    options = AdvancedOptions(
        automatic_fit_regions=False,
        fit_ranges=((0.581, 0.589),),
        exclude_ranges=((0.584, 0.585),),
        initial_species_scales=initial,
        fixed_species_scales=fixed,
        scale_bounds=(0.1, 3.0),
        max_nfev=45,
        segment_size=0.02,
        lsf_sigma_pixels=1.25,
        continuum_order=3,
        wavelength_order=1,
        fit_lsf_sigma=True,
    )
    captured = {}

    def record(*args, **kwargs):
        captured.update(kwargs)
        return spectrum

    monkeypatch.setattr(workflow, "correct_file", record)
    monkeypatch.setattr(workflow, "correct_arrays", record)
    monkeypatch.setattr(workflow, "_correct_spectrum_workflow", record)
    monkeypatch.setattr(workflow, "_finalize_correction", lambda result, **kw: result)
    correct("input.fits" if route == "file" else spectrum, advanced=options, report=False)
    assert captured["initial_species_scales"] is initial
    assert captured["fixed_species_scales"] is fixed
    for name in (
        "automatic_fit_regions",
        "fit_ranges",
        "exclude_ranges",
        "segment_size",
        "lsf_sigma_pixels",
        "continuum_order",
        "fit_lsf_sigma",
        "scale_bounds",
        "max_nfev",
    ):
        assert captured[name] == getattr(options, name)
    assert captured["fit_segment_wavelength_polynomial"] is True
    assert "joint_stellar_model" not in captured


def test_file_and_cache_controls_still_forward(monkeypatch):
    monkeypatch.setattr(workflow, "correct_file", lambda **kw: kw)
    result = correct(
        "input.fits",
        output_path="result.ecsv",
        hdu=3,
        image_index=2,
        advanced=AdvancedOptions(ftol=1e-8),
        cache=CacheConfig(aer_cache_dir="cache", aer_offline=True),
    )
    assert result["hdu"] == 3
    assert result["image_index"] == 2
    assert result["aer_offline"] is True
    assert result["aer_cache_dir"] == "cache"
    assert result["ftol"] == 1e-8


@pytest.mark.parametrize("range_name", ["fit_ranges", "exclude_ranges"])
def test_advanced_ranges_conflict_with_region_file(range_name, tmp_path):
    spectrum = Spectrum(
        np.linspace(0.58, 0.59, 30),
        np.ones(30),
        observation=Observation(wavelength_frame="observatory"),
    )
    path = tmp_path / "regions.ecsv"
    RegionSelection(fit_ranges=((0.581, 0.589),)).save(path)
    with pytest.raises(ValueError, match="region_file cannot be combined"):
        correct(
            spectrum, region_file=path, advanced=AdvancedOptions(**{range_name: ((0.582, 0.584),)})
        )


@pytest.mark.parametrize("bad", ["ftoll", "product_path", "product_format", "joint_stellar_model"])
def test_removed_or_unknown_keywords_fail_before_loading(bad):
    with pytest.raises(TypeError, match=bad):
        correct("missing.fits", **{bad: True})


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"advanced": {}}, "advanced must be"),
        ({"cache": {}}, "cache must be"),
        ({"advanced": AdvancedOptions(), "ftol": 1e-8}, "unexpected keyword"),
        ({"advanced": AdvancedOptions(), "atmosphere": None}, "unexpected keyword"),
        ({"cache": CacheConfig(), "aer_offline": True}, "unexpected keyword"),
        ({"wavelength_order": 1, "fit_wavelength_shift": True}, "unexpected keyword"),
    ],
)
def test_ambiguous_options_are_rejected(kwargs, message):
    with pytest.raises(TypeError, match=message):
        correct("missing.fits", **kwargs)


@pytest.mark.parametrize("route", ["spectrum", "file"])
def test_interface_is_numerically_identical_to_lower_level(monkeypatch, tmp_path, route):
    wave = np.linspace(2.31, 2.36, 120)
    spectrum = Spectrum(
        wave,
        np.ones(wave.size),
        uncertainty=np.full(wave.size, 0.01),
        observation=Observation(wavelength_frame="observatory"),
    )
    lines = demo_lines()
    monkeypatch.setattr(workflow, "_resolve_line_list", lambda *a, **kw: lines)
    expert = AdvancedOptions(
        segment_size=0.1,
        automatic_fit_regions=False,
        lsf_lorentz_fwhm_pixels=0.0,
        lsf_variable_width=False,
        fit_lsf_lorentz_fwhm=False,
        lsf_sigma_pixels=0.0,
        continuum_order=0,
        wavelength_order=None,
        fit_lsf_sigma=False,
        initial_species_scales={"H2O": 1.2},
        scale_bounds=(0.01, 5.0),
        max_nfev=60,
    )
    flat = {field.name: getattr(expert, field.name) for field in fields(expert)}
    flat.pop("wavelength_order")
    flat["fit_wavelength_shift"] = False
    expected = correct_arrays(
        wave,
        spectrum.flux,
        uncertainty=spectrum.uncertainty,
        observation=spectrum.observation,
        wavelength_medium="vacuum",
        **flat,
    )
    if route == "file":
        from tests.helpers import write_spectrum_fits

        path = write_spectrum_fits(spectrum, tmp_path / "input.fits")
        actual = correct(path, advanced=expert, report=False)
    else:
        actual = correct(spectrum, advanced=expert, report=False)
    np.testing.assert_allclose(actual.transmission, expected.transmission)
    np.testing.assert_allclose(actual.corrected.flux, expected.corrected.flux)
    np.testing.assert_array_equal(actual.fit_mask, expected.fit_mask)
    assert actual.species_scales == expected.species_scales
    assert provenance_json(actual.provenance["fit_config"]) == provenance_json(
        expected.provenance["fit_config"]
    )
