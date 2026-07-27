from __future__ import annotations

import inspect
from typing import get_args, get_type_hints

from pymolfit import (
    Observation,
    correct,
    correct_file,
    save_corrected_txt,
    save_fit_product_ecsv,
)


def test_unified_correct_documents_every_explicit_parameter_for_editor_hover() -> None:
    docstring = inspect.getdoc(correct)
    assert docstring is not None

    undocumented = [
        name
        for name in inspect.signature(correct).parameters
        if f":param {name}:" not in docstring
    ]

    assert undocumented == []


def test_observation_hover_text_explains_array_metadata_and_frames() -> None:
    docstring = inspect.getdoc(Observation)
    assert docstring is not None

    assert "wavelength/flux arrays" in docstring
    assert "wavelength_frame" in docstring
    assert "Barycentric arrays" in docstring
    assert "Heliocentric arrays" in docstring


def test_correct_file_documents_every_parameter_for_editor_hover() -> None:
    docstring = inspect.getdoc(correct_file)
    assert docstring is not None

    undocumented = [
        name
        for name in inspect.signature(correct_file).parameters
        if f":param {name}:" not in docstring
    ]

    assert undocumented == []


def test_correct_file_exposes_canonical_string_choices() -> None:
    hints = get_type_hints(correct_file)

    assert set(get_args(hints["wavelength_medium"])) == {
        "vacuum",
        "vac",
        "air",
        None,
    }
    assert set(get_args(hints["atmosphere_mode"])) == {
        "mipas_gdas",
        "mipas",
        "gdas",
        "single",
        "standard",
    }
    assert set(get_args(hints["gdas_mode"])) == {"auto", "online", "cache", "average"}
    assert set(get_args(hints["loss"])) == {
        "linear",
        "soft_l1",
        "huber",
        "cauchy",
        "arctan",
    }
    assert (
        inspect.signature(correct_file)
        .parameters["solve_continuum_linear"]
        .default
        == "auto"
    )
    assert (
        inspect.signature(correct_file)
        .parameters["lsf_sigma_pixels"]
        .default
        == "auto"
    )
    assert (
        inspect.signature(correct_file)
        .parameters["fit_lsf_sigma"]
        .default
        == "auto"
    )
    assert (
        inspect.signature(correct_file)
        .parameters["lsf_sigma_bounds"]
        .default
        is None
    )
    assert (
        inspect.signature(correct_file)
        .parameters["lsf_lorentz_fwhm_pixels"]
        .default
        == "auto"
    )
    assert (
        inspect.signature(correct_file)
        .parameters["fit_lsf_lorentz_fwhm"]
        .default
        == "auto"
    )
    assert (
        inspect.signature(correct_file)
        .parameters["lsf_lorentz_fwhm_bounds"]
        .default
        is None
    )


def test_atmosphere_hover_text_defines_domain_terms() -> None:
    docstring = inspect.getdoc(correct_file)
    assert docstring is not None

    assert "MIPAS (Michelson Interferometer for Passive Atmospheric Sounding)" in docstring
    assert "GDAS (NOAA Global Data Assimilation System)" in docstring
    assert "lower part of the MIPAS climatology" in docstring


def test_loss_hover_text_explains_when_robust_fitting_is_appropriate() -> None:
    docstring = inspect.getdoc(correct_file)
    assert docstring is not None

    assert "clean, well-masked spectrum" in docstring
    assert "mostly clean spectrum containing a limited number" in docstring
    assert "Robust loss cannot repair generally poor calibration" in docstring
    assert "soft_l1`` select nonlinear continuum fitting immediately" in docstring


def test_lsf_hover_text_explains_automatic_estimation_and_refinement() -> None:
    docstring = inspect.getdoc(correct_file)
    assert docstring is not None

    assert "derives Gaussian sigma in detector pixels from FITS" in docstring
    assert "estimates narrow observed-feature widths" in docstring
    assert "refines an automatically estimated" in docstring
    assert "generates broad non-negative bounds" in docstring
    assert "telluric-rich pilot regions distributed over the spectrum" in docstring
    assert "penalized pilot-model selection" in docstring


def test_output_helpers_explain_their_contents_for_editor_hover() -> None:
    txt_docstring = inspect.getdoc(save_corrected_txt)
    ecsv_docstring = inspect.getdoc(save_fit_product_ecsv)

    assert txt_docstring is not None
    assert "wavelength and corrected" in txt_docstring
    assert "atmospheric transmission" in txt_docstring
    assert ecsv_docstring is not None
    assert "Enhanced Character-Separated Values" in ecsv_docstring
    assert "fitted transmission" in ecsv_docstring
    assert "provenance" in ecsv_docstring
