from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from genmolfit import FitConfig, LineList, ModelConfig, Spectrum, fit_tellurics, transmission_model


def main() -> None:
    rng = np.random.default_rng(10)
    wavelength = np.linspace(2.31, 2.36, 1200)
    line_list = LineList.demo_near_ir()

    true_transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(airmass=1.2, species_scales={"H2O": 2.0, "CO2": 0.7, "CH4": 1.5}),
    )
    continuum = 1.0 + 0.08 * (wavelength - wavelength.mean()) / np.ptp(wavelength)
    flux = continuum * true_transmission + rng.normal(0.0, 0.004, wavelength.size)

    result = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux, uncertainty=np.full_like(flux, 0.004)),
        line_list=line_list,
        config=FitConfig(airmass=1.2, continuum_order=1),
    )

    print("Fit success:", result.success)
    print("Species scales:", result.species_scales)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(wavelength, flux, color="0.3", lw=1, label="observed synthetic spectrum")
    axes[0].plot(wavelength, result.model_flux, color="tab:red", lw=1.2, label="fitted model")
    axes[0].set_ylabel("flux")
    axes[0].legend()

    axes[1].plot(wavelength, result.corrected.flux / result.continuum, color="tab:blue", lw=1)
    axes[1].axhline(1.0, color="0.2", lw=0.8, ls="--")
    axes[1].set_xlabel("wavelength [micron]")
    axes[1].set_ylabel("corrected / continuum")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
