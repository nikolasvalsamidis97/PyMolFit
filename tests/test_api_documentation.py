from __future__ import annotations

import inspect
from typing import get_args, get_type_hints

from pymolfit import correct_file


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

    assert set(get_args(hints["wavelength_medium"])) == {"vacuum", "vac", "air"}
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


def test_atmosphere_hover_text_defines_domain_terms() -> None:
    docstring = inspect.getdoc(correct_file)
    assert docstring is not None

    assert "MIPAS (Michelson Interferometer for Passive Atmospheric Sounding)" in docstring
    assert "GDAS (NOAA Global Data Assimilation System)" in docstring
    assert "lower part of the MIPAS climatology" in docstring
