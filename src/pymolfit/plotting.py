from __future__ import annotations

from pathlib import Path

from .fit import TelluricFitResult


def plot_fit(
    result: TelluricFitResult,
    *,
    path: str | Path | None = None,
    show: bool = True,
):
    """Plot observed spectrum, fitted model, transmission, and correction."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib; install with `pip install pymolfit[plot]`") from exc

    wave = result.spectrum.wavelength
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))

    axes[0].plot(wave, result.spectrum.flux, color="0.35", lw=0.9, label="observed")
    axes[0].plot(wave, result.model_flux, color="tab:red", lw=1.1, label="model")
    axes[0].set_ylabel("flux")
    axes[0].legend(loc="best")

    axes[1].plot(wave, result.transmission, color="tab:green", lw=1.0)
    axes[1].set_ylabel("transmission")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(wave, result.corrected.flux, color="tab:blue", lw=0.9, label="corrected")
    axes[2].plot(wave, result.continuum, color="0.25", lw=1.0, ls="--", label="continuum")
    axes[2].set_xlabel(f"wavelength [{result.spectrum.wavelength_unit}]")
    axes[2].set_ylabel("corrected flux")
    axes[2].legend(loc="best")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180)
    if show:
        plt.show()
    return fig
