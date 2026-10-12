"""Plot Beta Pictoris Na D before/after correction from a saved PyMolFit product.

Run notebook 01 first. Optionally pass another product as the first argument.
The script saves one PNG beside itself; --no-show suppresses the window.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pymolfit import load_fit_product

HERE = Path(__file__).resolve().parent
DEFAULT_PRODUCT = HERE.parent / "user_guide/outputs/beta_pic_fits.ecsv"


def plot_nad_correction(product_path=DEFAULT_PRODUCT, *, output_path=None, show=True):
    """Overlay native fluxes; product_path is ECSV, output_path an image, show opens it."""
    result = load_fit_product(product_path)
    observed = result.spectrum.to_unit("nm")
    corrected = result.corrected.to_unit("nm")
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for spectrum, colour, label in (
        (observed, "0.55", "Uncorrected"),
        (corrected, "tab:blue", "PyMolFit corrected"),
    ):
        ax.plot(
            spectrum.wavelength,
            np.where(spectrum.valid, spectrum.flux, np.nan),
            color=colour,
            lw=0.9,
            label=label,
        )
    ax.set(
        xlim=(
            max(588.0, np.nanmin(observed.wavelength)),
            min(591.0, np.nanmax(observed.wavelength)),
        ),
        xlabel=f"Wavelength [nm] ({observed.wavelength_medium})",
        ylabel="Flux [ADU]",
        title="Beta Pictoris / HARPS: Na I D",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    destination = Path(output_path) if output_path is not None else HERE / "nad_correction.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "product",
        nargs="?",
        type=Path,
        default=DEFAULT_PRODUCT,
        help="Full correction product ECSV, not a flux-only text export",
    )
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window")
    args = parser.parse_args()
    plot_nad_correction(args.product, show=not args.no_show)
