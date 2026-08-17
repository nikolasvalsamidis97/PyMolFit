from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from pymolfit.plotting import plot_fit


def test_plot_fit_draws_full_spectrum_as_disconnected_lines():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = np.linspace(0.50, 0.505, 12)
    second = np.linspace(0.52, 0.525, 12)
    wavelength = np.concatenate((first, second))
    observed = 1.0 - 0.2 * np.exp(-(((wavelength - 0.5025) / 0.0004) ** 2))
    transmission = 1.0 - 0.15 * np.exp(-(((wavelength - 0.5025) / 0.0003) ** 2))
    corrected = observed / transmission

    result = SimpleNamespace(
        spectrum=SimpleNamespace(
            wavelength=wavelength,
            flux=observed,
            wavelength_unit="micron",
        ),
        corrected=SimpleNamespace(flux=corrected),
        transmission=transmission,
    )

    figure = plot_fit(result, show=False)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xlim() == pytest.approx((first[0], second[-1]))
    assert [line.get_label() for line in figure.axes[0].lines] == [
        "Observed",
        "Telluric corrected",
        "_nolegend_",
        "_nolegend_",
    ]
    np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), first)
    np.testing.assert_allclose(figure.axes[0].lines[2].get_xdata(), second)
    assert all(line.get_linestyle() == "-" for line in figure.axes[0].lines)
    assert len(figure.axes[1].lines) == 2

    plt.close(figure)


def test_plot_fit_reads_saved_product_ecsv(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = np.linspace(1.55, 1.56, 20)
    observed = np.linspace(0.8, 1.0, wavelength.size)
    transmission = np.linspace(0.9, 1.0, wavelength.size)
    corrected = observed / transmission
    table = Table()
    table["wavelength"] = wavelength
    table["wavelength"].unit = "micron"
    table["flux"] = observed
    table["corrected_flux"] = corrected
    table["transmission"] = transmission
    product_path = tmp_path / "fit_product.ecsv"
    table.write(product_path, format="ascii.ecsv")

    figure = plot_fit(product_path, show=False)

    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), observed)
    np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), corrected)
    np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), transmission)
    assert figure.axes[1].get_xlabel() == "Wavelength [micron]"
    plt.close(figure)


def test_plot_fit_flux_limits_include_isolated_corrected_extrema():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = np.linspace(0.50, 0.51, 1000)
    observed = np.ones_like(wavelength)
    corrected = np.ones_like(wavelength)
    corrected[100] = -2.0
    corrected[900] = 8.0
    result = SimpleNamespace(
        spectrum=SimpleNamespace(
            wavelength=wavelength,
            flux=observed,
            wavelength_unit="micron",
        ),
        corrected=SimpleNamespace(flux=corrected),
        transmission=np.ones_like(wavelength),
    )

    figure = plot_fit(result, show=False)

    lower, upper = figure.axes[0].get_ylim()
    assert lower < np.nanmin(corrected)
    assert upper > np.nanmax(corrected)
    plt.close(figure)


def test_plot_fit_rejects_compact_output_file(tmp_path):
    compact_path = tmp_path / "corrected.ecsv"
    np.savetxt(compact_path, np.ones((3, 2)))

    with pytest.raises(ValueError, match="product_path"):
        plot_fit(compact_path, show=False)


@pytest.mark.parametrize(
    ("notebook", "expected_show_calls"),
    ((True, 0), (False, 1)),
)
def test_plot_fit_uses_only_the_environment_display_path(
    monkeypatch,
    notebook,
    expected_show_calls,
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pymolfit.plotting as plotting

    wavelength = np.linspace(0.50, 0.51, 20)
    result = SimpleNamespace(
        spectrum=SimpleNamespace(
            wavelength=wavelength,
            flux=np.ones_like(wavelength),
            wavelength_unit="micron",
        ),
        corrected=SimpleNamespace(flux=np.ones_like(wavelength)),
        transmission=np.ones_like(wavelength),
    )
    show_calls: list[None] = []
    monkeypatch.setattr(
        plotting,
        "_is_notebook_environment",
        lambda: notebook,
    )
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(None))

    figure = plotting.plot_fit(result)

    assert len(show_calls) == expected_show_calls
    plt.close(figure)
