from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from genmolfit.atmosphere import AtmosphereProfile, BOLTZMANN_J_PER_K, CM_PER_M, PA_PER_ATM
from genmolfit.components import hitran_line_optical_depth_basis
from genmolfit.linelist import LineList
from genmolfit.physics import (
    AMU_KG,
    SECOND_RADIATION_CONSTANT_CM_K,
    SPEED_OF_LIGHT_M_PER_S,
    lblrtm_panel_voigt_profile_wavenumber,
)


def synthetic_line() -> LineList:
    center_cm = 4320.0
    return LineList(
        wavelength=np.array([1.0e4 / center_cm]),
        strength=np.array([2.0e-24]),
        sigma=np.array([0.01]),
        gamma=np.array([0.02]),
        species=np.array(["H2O"]),
        wavenumber=np.array([center_cm]),
        mol_id=np.array([1]),
        iso_id=np.array([1]),
        air_width=np.array([0.07]),
        self_width=np.array([0.30]),
        lower_state_energy=np.array([100.0]),
        temperature_exponent=np.array([0.75]),
        pressure_shift=np.array([-0.001]),
        molecular_mass_amu=np.array([18.010565]),
        line_source="hitran_par",
    )


def synthetic_atmosphere() -> AtmosphereProfile:
    return AtmosphereProfile.single_layer(
        pressure_atm=0.72,
        temperature_k=278.0,
        path_length_m=1500.0,
        mixing_ratios={"H2O": 1.2e-5},
    )


def reference_panel_tau(wavenumber: np.ndarray, line_list: LineList, atmosphere: AtmosphereProfile) -> np.ndarray:
    layer = atmosphere.layers[0]
    ref_t = line_list.reference_temperature
    vmr = layer.mixing_ratios["H2O"]
    center = line_list.wavenumber[0] + line_list.pressure_shift[0] * layer.pressure_atm
    c2 = SECOND_RADIATION_CONSTANT_CM_K
    strength = (
        line_list.strength[0]
        * (ref_t / layer.temperature_k) ** 1.5
        * np.exp(-c2 * line_list.lower_state_energy[0] * (1.0 / layer.temperature_k - 1.0 / ref_t))
        * (1.0 - np.exp(-c2 * line_list.wavenumber[0] / layer.temperature_k))
        / (1.0 - np.exp(-c2 * line_list.wavenumber[0] / ref_t))
    )
    mass_kg = line_list.molecular_mass_amu[0] * AMU_KG
    sigma = center * np.sqrt(
        BOLTZMANN_J_PER_K * layer.temperature_k / (mass_kg * SPEED_OF_LIGHT_M_PER_S**2)
    )
    gamma = (
        ((1.0 - vmr) * line_list.air_width[0] + vmr * line_list.self_width[0])
        * layer.pressure_atm
        * (ref_t / layer.temperature_k) ** line_list.temperature_exponent[0]
    )
    column = layer.pressure_atm * PA_PER_ATM * layer.path_length_m / (
        BOLTZMANN_J_PER_K * layer.temperature_k * CM_PER_M**2
    ) * vmr
    profile = lblrtm_panel_voigt_profile_wavenumber(
        wavenumber,
        np.array([center]),
        np.array([sigma]),
        np.array([gamma]),
    )[0]
    return profile * strength * column


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    line_list = synthetic_line()
    atmosphere = synthetic_atmosphere()
    wavenumber = np.linspace(4319.5, 4320.5, 301)
    wavelength = 1.0e4 / wavenumber

    modes = {}
    for mode in ("full", "lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"):
        _, basis = hitran_line_optical_depth_basis(
            wavelength,
            line_list,
            atmosphere,
            species=("H2O",),
            chunk_size=1,
            line_wing_mode=mode,
        )
        modes[mode] = basis[0]
    modes["reference_panel"] = reference_panel_tau(wavenumber, line_list, atmosphere)

    table = Table()
    table["wavenumber_cm-1"] = wavenumber
    table["wavelength_um"] = wavelength
    for name, tau in modes.items():
        table[f"tau_{name}"] = tau
        table[f"transmission_{name}"] = np.exp(-tau)
    table.write(out_dir / "single_line_audit.ecsv", format="ascii.ecsv", overwrite=True)

    summary = [
        "Synthetic one-line LBLRTM audit",
        "No observed spectrum and no fitted coefficients are used.",
        "The production accumulated-panel path is compared with an independent per-line panel evaluation.",
        f"max_abs_tau_panel_minus_reference = {np.max(np.abs(modes['lblrtm_panel'] - modes['reference_panel'])):.6e}",
        f"max_tau_full = {np.max(modes['full']):.6e}",
        f"max_tau_lblrtm_panel = {np.max(modes['lblrtm_panel']):.6e}",
    ]
    (out_dir / "single_line_audit_summary.txt").write_text("\n".join(summary) + "\n")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(wavenumber, modes["full"], label="exact Voigt full", lw=1.4)
    axes[0].plot(wavenumber, modes["lblrtm_dynamic"], label="LBLRTM dynamic", lw=1.2)
    axes[0].plot(wavenumber, modes["lblrtm_panel"], label="LBLRTM panel", lw=1.2)
    axes[0].plot(wavenumber, modes["reference_panel"], "--", label="independent per-line reference", lw=1.0)
    axes[0].set_ylabel("Optical depth")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(wavenumber, modes["lblrtm_panel"] - modes["reference_panel"], color="black", lw=1.0)
    axes[1].set_xlabel("Wavenumber [cm-1]")
    axes[1].set_ylabel("Panel - reference")
    fig.tight_layout()
    fig.savefig(out_dir / "single_line_audit.png", dpi=180)


if __name__ == "__main__":
    main()
