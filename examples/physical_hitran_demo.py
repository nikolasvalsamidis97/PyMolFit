from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from genmolfit import (
    AtmosphereProfile,
    LineList,
    PhysicalModelConfig,
    correct_arrays,
    physical_transmission_model,
)


def _fixed_decimal(value, width, decimals):
    text = f"{value:.{decimals}f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return f"{text:>{width}}"[-width:]


def _hitran_row(
    *,
    mol_id=1,
    iso_id=1,
    wavenumber=4320.0,
    intensity=6.0e-25,
    air_width=0.07,
    self_width=0.30,
    lower_energy=100.0,
    n_air=0.75,
    pressure_shift=-0.001,
):
    row = (
        f"{mol_id:2d}"
        f"{iso_id:1d}"
        f"{wavenumber:12.6f}"
        f"{intensity:10.3E}"
        f"{1.0:10.3E}"
        f"{_fixed_decimal(air_width, 5, 4)}"
        f"{_fixed_decimal(self_width, 5, 4)}"
        f"{lower_energy:10.4f}"
        f"{n_air:4.2f}"
        f"{_fixed_decimal(pressure_shift, 8, 6)}"
    )
    return row + " " * (160 - len(row))


def main() -> None:
    rng = np.random.default_rng(5)
    with TemporaryDirectory() as tmp:
        hitran_path = Path(tmp) / "demo_h2o.par"
        hitran_path.write_text(
            "\n".join(
                [
                    _hitran_row(wavenumber=4320.0, intensity=8.0e-25),
                    _hitran_row(wavenumber=4324.0, intensity=5.0e-25, lower_energy=250.0),
                    _hitran_row(wavenumber=4328.0, intensity=4.0e-25, lower_energy=80.0),
                ]
            )
            + "\n"
        )

        wavelength = np.linspace(1.0e4 / 4330.0, 1.0e4 / 4318.0, 1000)
        line_list = LineList.from_hitran_par(hitran_path)
        atmosphere = AtmosphereProfile.single_layer(
            pressure_atm=0.75,
            temperature_k=280.0,
            path_length_m=5_000.0,
            mixing_ratios={"H2O": 2.0e-5},
        )
        transmission = physical_transmission_model(
            wavelength,
            line_list,
            atmosphere,
            PhysicalModelConfig(species_scales={"H2O": 1.8}),
        )
        continuum = 1.0 + 0.05 * (wavelength - wavelength.mean()) / np.ptp(wavelength)
        flux = continuum * transmission + rng.normal(0.0, 0.002, wavelength.size)

        result = correct_arrays(
            wavelength,
            flux,
            hitran_par=hitran_path,
            mixing_ratios={"H2O": 2.0e-5},
            pressure_atm=0.75,
            temperature_k=280.0,
            path_length_m=5_000.0,
            continuum_order=1,
            fit_wavelength_shift=True,
            fit_lsf_sigma=True,
        )

    print("Fit success:", result.success)
    print("Species scales:", result.species_scales)
    print("Metrics:", result.metrics)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(result.spectrum.wavelength, result.spectrum.flux, color="0.35", lw=0.9, label="synthetic observed")
    axes[0].plot(result.spectrum.wavelength, result.model_flux, color="tab:red", lw=1.0, label="fit")
    axes[0].set_ylabel("flux")
    axes[0].legend()

    axes[1].plot(result.spectrum.wavelength, result.transmission, color="tab:green", lw=1.0)
    axes[1].set_ylabel("transmission")

    axes[2].plot(result.spectrum.wavelength, result.corrected.flux, color="tab:blue", lw=0.9)
    axes[2].plot(result.spectrum.wavelength, result.continuum, color="0.2", ls="--", lw=1.0)
    axes[2].set_xlabel("wavelength [micron]")
    axes[2].set_ylabel("corrected")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
