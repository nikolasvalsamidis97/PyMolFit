from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from .errors import ProductFormatError
from .fit import TelluricFitResult
from .io import load_fit_product


def plot_fit(
    result: TelluricFitResult | str | Path,
    *,
    path: str | Path | None = None,
    show: bool = True,
):
    """Plot a live fit result or a saved PyMolFit product ECSV.

    The first panel overlays the observed and telluric-corrected flux. The
    second panel shows atmospheric transmission. Every valid sample is drawn
    at full resolution. Separate echelle orders and other genuine wavelength
    discontinuities remain disconnected instead of being joined by artificial
    lines. Pass the ``product_path`` ECSV written by :func:`pymolfit.correct`
    to recreate the same diagnostic plot without rerunning the correction.
    Compact files written through ``output_path`` do not contain the observed
    flux or transmission and therefore cannot be used here.

    :param result: Live ``TelluricFitResult`` or path to a full PyMolFit
        product ECSV created with ``product_path``.
    :param path: Optional image destination for the generated figure.
    :param show: Display the figure when ``True``. In Jupyter, PyMolFit leaves
        display to the active notebook backend so an interactive canvas is not
        followed by a duplicate static rendering. In a regular Python process,
        it calls ``matplotlib.pyplot.show`` once.
    :return: Matplotlib figure containing flux and transmission panels.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib; install with `pip install pymolfit[plot]`"
        ) from exc

    wavelength, observed, corrected, transmission, wavelength_unit = (
        _fit_plot_data(result)
    )
    sections = _continuous_wavelength_sections(wavelength)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    for index, section in enumerate(sections):
        axes[0].plot(
            wavelength[section],
            observed[section],
            color="0.55",
            linewidth=0.5,
            label="Observed" if index == 0 else "_nolegend_",
        )
        axes[0].plot(
            wavelength[section],
            corrected[section],
            color="tab:blue",
            linewidth=0.5,
            label="Telluric corrected" if index == 0 else "_nolegend_",
        )
        axes[1].plot(
            wavelength[section],
            transmission[section],
            color="tab:green",
            linewidth=0.5,
            label="Atmospheric transmission" if index == 0 else "_nolegend_",
        )

    finite_wavelength = wavelength[np.isfinite(wavelength)]
    axes[0].set_xlim(
        float(np.nanmin(finite_wavelength)),
        float(np.nanmax(finite_wavelength)),
    )
    _set_complete_flux_limits(axes[0], observed, corrected)
    axes[0].set_ylabel("Flux")
    axes[0].legend(loc="best")

    axes[1].set_xlabel(f"Wavelength [{wavelength_unit}]")
    axes[1].set_ylabel("Transmission")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="best")

    if path is not None:
        figure.savefig(path, dpi=180)
    if show and not _is_notebook_environment():
        plt.show()
    return figure


def _is_notebook_environment() -> bool:
    """Return whether Matplotlib output is managed by a Jupyter kernel."""

    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _fit_plot_data(
    result: TelluricFitResult | str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not isinstance(result, (str, Path)):
        return (
            np.asarray(result.spectrum.wavelength, dtype=float),
            np.asarray(result.spectrum.flux, dtype=float),
            np.asarray(result.corrected.flux, dtype=float),
            np.asarray(result.transmission, dtype=float),
            str(result.spectrum.wavelength_unit),
        )

    try:
        loaded = load_fit_product(result)
    except ProductFormatError:
        return _legacy_product_plot_data(Path(result))
    return (
        np.asarray(loaded.spectrum.wavelength, dtype=float),
        np.asarray(loaded.spectrum.flux, dtype=float),
        np.asarray(loaded.corrected.flux, dtype=float),
        np.asarray(loaded.transmission, dtype=float),
        str(loaded.spectrum.wavelength_unit),
    )


def _legacy_product_plot_data(
    product_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Read plotting columns from an unversioned pre-0.6 product."""

    try:
        table = Table.read(product_path, format="ascii.ecsv")
    except Exception as exc:
        raise ProductFormatError(
            f"{product_path} is not a readable PyMolFit product ECSV; save "
            "the correction with product_path=..., not output_path=..."
        ) from exc
    required = {"wavelength", "flux", "corrected_flux", "transmission"}
    missing = sorted(required.difference(table.colnames))
    if missing:
        raise ProductFormatError(
            f"{product_path} is not a full PyMolFit product ECSV; missing "
            f"columns: {', '.join(missing)}; use product_path=... when saving"
        )
    wavelength_column = table["wavelength"]
    wavelength_unit = getattr(wavelength_column, "unit", None)
    if wavelength_unit is None:
        wavelength_unit = table.meta.get("wavelength_unit", "micron")
    return (
        np.asarray(wavelength_column, dtype=float),
        np.asarray(table["flux"], dtype=float),
        np.asarray(table["corrected_flux"], dtype=float),
        np.asarray(table["transmission"], dtype=float),
        str(wavelength_unit),
    )


def _continuous_wavelength_sections(
    wavelength: np.ndarray,
) -> tuple[np.ndarray, ...]:
    finite_indices = np.flatnonzero(np.isfinite(wavelength))
    if finite_indices.size == 0:
        raise ValueError("cannot plot a spectrum without finite wavelengths")
    if finite_indices.size == 1:
        return (finite_indices,)

    spacing = np.diff(wavelength[finite_indices])
    adjacent = np.diff(finite_indices) == 1
    positive = spacing[adjacent & np.isfinite(spacing) & (spacing > 0)]
    gap_limit = (
        20.0 * float(np.nanmedian(positive))
        if positive.size
        else np.inf
    )
    discontinuity = (
        ~adjacent
        | ~np.isfinite(spacing)
        | (spacing <= 0)
        | (spacing > gap_limit)
    )
    breaks = np.flatnonzero(discontinuity) + 1
    return tuple(
        section
        for section in np.split(finite_indices, breaks)
        if section.size
    )


def _set_complete_flux_limits(axis, *values: np.ndarray) -> None:
    """Set limits that include every finite plotted flux sample."""

    finite_parts = [
        np.asarray(value, dtype=float)[np.isfinite(value)]
        for value in values
        if np.asarray(value).size
    ]
    finite_parts = [value for value in finite_parts if value.size]
    if not finite_parts:
        return

    finite = np.concatenate(finite_parts)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return
    if upper <= lower:
        padding = max(abs(lower), 1.0) * 0.05
    else:
        padding = 0.08 * (upper - lower)
    axis.set_ylim(lower - padding, upper + padding)
