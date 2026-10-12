"""Plot the fitted transmission from the same product used for the Na D example.

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


def plot_transmission(product_path=DEFAULT_PRODUCT, *, output_path=None, show=True):
    """Plot fitted transmission; product_path is ECSV, output_path an image, show opens it."""
    result = load_fit_product(product_path)
    spectrum = result.spectrum.to_unit("nm")
    values = np.where(spectrum.valid, result.transmission, np.nan)
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    ax.plot(spectrum.wavelength, values, color="tab:blue", lw=0.9)
    ax.set(
        xlabel=f"Wavelength [nm] ({spectrum.wavelength_medium})",
        ylabel="Transmission",
        title="Beta Pictoris / HARPS: fitted atmospheric transmission",
    )
    ax.grid(alpha=0.15)
    destination = Path(output_path) if output_path is not None else HERE / "transmission.png"
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
    plot_transmission(args.product, show=not args.no_show)
