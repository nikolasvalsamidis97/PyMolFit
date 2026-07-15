"""Isolate X-shooter radiative-transfer parity from fitting differences.

This diagnostic evaluates PyMolFit at the molecule columns, wavelength
solution, and instrumental widths fitted independently by Molecfit.  It does
not tune any PyMolFit coefficient and is not part of the public package API.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from pymolfit.atmosphere import AtmosphereProfile
from pymolfit.continuum import LBLRTMCO2Continuum, LBLRTMH2OContinuum
from pymolfit.fit import FitConfig, fit_tellurics
from pymolfit.linelist import LineList
from pymolfit.partition import PartitionTable
from pymolfit.physics import (
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_SAMPLE,
    LBLRTM_VOIGT_DOMAIN_HWF3,
)
from pymolfit.spectrum import Spectrum
from pymolfit.workflow import _build_components

from local_tests.run_science_readiness_validation import (
    CACHE_PATH,
    OUTPUT_DIR,
    REAL_BANDS,
    _molecfit_parameters,
)


OUTPUT = OUTPUT_DIR / "xshooter_fixed_rt_parity"
GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493
MOLECFIT_DEFAULT_PIXEL_SCALE_ARCSEC = 0.086


def _combined_header(path: Path) -> dict[str, object]:
    with fits.open(path) as hdul:
        header = dict(hdul[0].header)
        for key, value in hdul[1].header.items():
            header.setdefault(key, value)
    return header


def _fixed_transmission(case, all_lines: LineList) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_dir = OUTPUT_DIR / "xshooter" / case.name
    input_path = case_dir / "input_air.fits"
    model_path = case_dir / "BEST_FIT_MODEL.fits"
    parameter_path = case_dir / "BEST_FIT_PARAMETERS.fits"
    header = _combined_header(input_path)

    with fits.open(input_path) as hdul:
        input_data = hdul[1].data
        uncertainty = np.asarray(input_data["dflux"], dtype=float)
    with fits.open(model_path) as hdul:
        model_data = hdul[1].data
        wavelength_vacuum = np.asarray(model_data["lambda"], dtype=float)
        flux = np.asarray(model_data["flux"], dtype=float)
        reference = np.asarray(model_data["mtrans"], dtype=float)

    # Molecfit converts AIR inputs to topocentric vacuum before fitting. Its
    # mtrans values live on this internal detector lambda grid; mlambda is a
    # back-projected diagnostic coordinate that is not reliable across gaps.
    spectrum = Spectrum(
        wavelength=wavelength_vacuum,
        flux=flux,
        uncertainty=uncertainty,
        wavelength_unit="micron",
        wavelength_medium="vacuum",
    )
    atmosphere = AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        mipas_profile="equ",
        gdas_mode="auto",
        gdas_download_timeout_s=30.0,
        reference_wavenumber_cm=float(1.0e4 / np.nanmedian(spectrum.wavelength)),
    )
    lines = all_lines.select_species(case.species).select_range(
        float(np.nanmin(spectrum.wavelength)),
        float(np.nanmax(spectrum.wavelength)),
        margin=0.01,
    )
    partition = PartitionTable.from_lblrtm_package_data()
    h2o_continuum = (
        LBLRTMH2OContinuum.from_package_data() if "H2O" in case.species else None
    )
    co2_continuum = (
        LBLRTMCO2Continuum.from_package_data() if "CO2" in case.species else None
    )
    components = _build_components(
        extra_components=None,
        line_list=lines,
        chunk_size=0,
        partition_table=partition,
        line_cutoff_cm=None,
        subtract_cutoff_profile=False,
        line_taper_cm=0.0,
        line_wing_mode="lblrtm_panel",
        lblrtm_sample=LBLRTM_DEFAULT_SAMPLE,
        lblrtm_alfal0=LBLRTM_DEFAULT_ALFAL0,
        lblrtm_hwf3=LBLRTM_VOIGT_DOMAIN_HWF3,
        lblrtm_avmass_amu=36.0,
        rayleigh=False,
        rayleigh_xrayl=1.0,
        n2_continuum=True,
        n2_continuum_xn2cn=1.0,
        o2_continuum=True,
        o2_continuum_xo2cn=1.0,
        h2o_continuum=h2o_continuum,
        h2o_continuum_foreign_closure=False,
        co2_continuum=co2_continuum,
        o2_cia=None,
        n2_cia=None,
        cia_tables=None,
    )
    parameters = _molecfit_parameters(parameter_path)
    wavelength_linear = parameters["chip 1, coef 1"]
    if not np.isclose(wavelength_linear, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("fixed parity audit currently requires a unit linear wavelength term")
    wavelength_span = float(np.nanmax(wavelength_vacuum) - np.nanmin(wavelength_vacuum))
    fixed_wavelength_shift = parameters["chip 1, coef 0"] * wavelength_span / 2.0
    wavelength_epsilon = max(1.0e-12, abs(fixed_wavelength_shift) * 1.0e-8)
    fixed_scales = {
        species: parameters[f"rel_mol_col_{species}"] for species in case.species
    }
    fixed_scales.update({"N2_continuum": 1.0, "O2_continuum": 1.0})
    slit_width_arcsec = float(header["ESO INS SLIT1 WID"])
    box_width_pixels = (
        slit_width_arcsec
        / MOLECFIT_DEFAULT_PIXEL_SCALE_ARCSEC
        * parameters["boxfwhm"]
    )
    config = FitConfig(
        airmass=1.0,
        continuum_order=2,
        fixed_species_scales=fixed_scales,
        solve_continuum_linear=False,
        lsf_sigma_pixels=parameters["gaussfwhm"] / GAUSSIAN_FWHM_TO_SIGMA,
        lsf_box_width_pixels=box_width_pixels,
        high_resolution_grid=True,
        high_resolution_oversampling=5.0,
        high_resolution_rebin_mode="molecfit_overlap",
        line_wing_mode="lblrtm_panel",
        atmosphere=atmosphere,
        partition_table=partition,
        h2o_continuum=h2o_continuum,
        n2_continuum=True,
        o2_continuum=True,
        components=components,
        fit_wavelength_shift=True,
        initial_wavelength_shift=fixed_wavelength_shift,
        wavelength_shift_bounds=(
            fixed_wavelength_shift - wavelength_epsilon,
            fixed_wavelength_shift + wavelength_epsilon,
        ),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    result = fit_tellurics(spectrum, line_list=lines, config=config)
    return wavelength_vacuum, result.transmission, reference


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_lines = LineList.from_table(CACHE_PATH)
    rows: list[dict[str, object]] = []
    for case in REAL_BANDS:
        wavelength, pymolfit, molecfit = _fixed_transmission(case, all_lines)
        reliable = np.isfinite(pymolfit) & np.isfinite(molecfit) & (molecfit > 0.2)
        telluric = reliable & (molecfit < 0.995)
        delta = pymolfit - molecfit
        optical_depth_pymolfit = -np.log(np.clip(pymolfit[telluric], 1.0e-12, 1.0))
        optical_depth_molecfit = -np.log(np.clip(molecfit[telluric], 1.0e-12, 1.0))
        optical_depth_scale = float(
            np.dot(optical_depth_pymolfit, optical_depth_molecfit)
            / np.dot(optical_depth_pymolfit, optical_depth_pymolfit)
        )
        row = {
            "case": case.name,
            "all_reliable_rms": float(np.sqrt(np.mean(delta[reliable] ** 2))),
            "telluric_rms": float(np.sqrt(np.mean(delta[telluric] ** 2))),
            "maximum_absolute_difference": float(np.max(np.abs(delta[reliable]))),
            "optical_depth_correlation": float(
                np.corrcoef(optical_depth_pymolfit, optical_depth_molecfit)[0, 1]
            ),
            "molecfit_to_pymolfit_optical_depth_scale": optical_depth_scale,
            "reliable_pixels": int(np.count_nonzero(reliable)),
            "telluric_pixels": int(np.count_nonzero(telluric)),
        }
        rows.append(row)

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(wavelength, molecfit, lw=0.9, label="Molecfit")
        axes[0].plot(wavelength, pymolfit, lw=0.8, label="PyMolFit at Molecfit parameters")
        axes[0].set_ylabel("Transmission")
        axes[0].legend(loc="best")
        axes[1].plot(wavelength[reliable], delta[reliable], color="black", lw=0.7)
        axes[1].axhline(0.0, color="0.5", lw=0.7)
        axes[1].set_ylabel("PyMolFit - Molecfit")
        axes[1].set_xlabel("Vacuum wavelength [micron]")
        fig.suptitle(case.name)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{case.name}.png", dpi=180)
        plt.close(fig)

    with (OUTPUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
