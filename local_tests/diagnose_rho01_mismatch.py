from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from genmolfit import (
    CO2ContinuumAbsorption,
    FitConfig,
    H2OContinuumAbsorption,
    HitranCIATable,
    HitranLineAbsorption,
    IsotopologueMetadata,
    LBLRTMCO2Continuum,
    LBLRTMH2OContinuum,
    LineList,
    PairCIAAbsorption,
    PartitionTable,
    Spectrum,
    fit_telluric_segments,
)

import compare_rho01_genmolfit_molecfit_lband as base


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "local_tests" / "rho01_mismatch_diagnostics"
CHIPS = (1, 4, 5, 6, 10, 11, 12, 13, 17, 18)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    line_list = LineList.from_table(base.LINE_LIST)
    isotopologues = IsotopologueMetadata.from_table(base.ISO_METADATA)
    line_list = line_list.with_isotopologue_metadata(isotopologues)
    partition_table = PartitionTable.from_hitran_q_directory(base.HITRAN_Q_DIR, isotopologues)
    atmosphere, fit_airmass = base.comparison_atmosphere(
        source=base.ATMOSPHERE_SOURCE,
        airmass=base.representative_airmass(base.MOLECFIT / "molecfit_input" / "SCIENCE_A.fits"),
        h2o_col_mm=base.best_fit_parameter(
            base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits",
            "h2o_col_mm",
        ),
    )
    scale_table = Table.read(base.OUTPUT / "summary.ecsv")
    species_scales = {
        name.removeprefix("scale_"): float(scale_table[name][0])
        for name in scale_table.colnames
        if name.startswith("scale_")
    }

    with fits.open(base.MOLECFIT / "molecfit_model" / "MOLECFIT_DATA.fits") as hdul:
        data = hdul[1].data.copy()

    spectra, masks, molecfit_trans, priors = _load_segments(data)
    variants = _variants(line_list, partition_table)
    rows = []
    for name, variant_line_list, components, config_updates in variants:
        print(f"running {name}")
        config = _fit_config(
            atmosphere=atmosphere,
            fit_airmass=fit_airmass,
            partition_table=partition_table,
            components=components,
            species_scales=species_scales,
            **config_updates,
        )
        result = fit_telluric_segments(
            spectra,
            line_list=variant_line_list,
            config=config,
            fit_masks=masks,
            continuum_priors=priors,
        )
        for chip, segment, reference in zip(CHIPS, result.segment_results, molecfit_trans, strict=True):
            transmission = np.asarray(segment.transmission, dtype=float)
            reference = np.asarray(reference, dtype=float)
            valid = np.isfinite(transmission) & np.isfinite(reference)
            diff = transmission[valid] - reference[valid]
            reference_valid = reference[valid]
            gen_tau = -np.log(np.clip(transmission[valid], 1.0e-300, 1.0))
            mol_tau = -np.log(np.clip(reference_valid, 1.0e-300, 1.0))
            tau_ok = np.isfinite(gen_tau) & np.isfinite(mol_tau) & (mol_tau > 0)
            rows.append(
                {
                    "variant": name,
                    "chip": chip,
                    "rms": float(np.sqrt(np.nanmean(diff**2))),
                    "median_abs": float(np.nanmedian(np.abs(diff))),
                    "mean_diff": float(np.nanmean(diff)),
                    "tau_ratio_median": float(np.nanmedian(gen_tau[tau_ok] / mol_tau[tau_ok])),
                    "rms_reference_lt_0p2": _bin_rms(diff, reference_valid < 0.2),
                    "rms_reference_0p2_0p8": _bin_rms(diff, (reference_valid >= 0.2) & (reference_valid < 0.8)),
                    "rms_reference_gt_0p8": _bin_rms(diff, reference_valid >= 0.8),
                    "nfev": int(result.nfev),
                    **{f"scale_{key}": value for key, value in result.species_scales.items()},
                }
            )
    table = Table(rows=rows)
    table.write(OUTPUT / "variant_summary.ecsv", format="ascii.ecsv", overwrite=True)
    table.write(OUTPUT / "variant_summary.csv", format="ascii.csv", overwrite=True)
    _write_pivot(table)
    print(f"wrote {OUTPUT}")


def _load_segments(data):
    spectra = []
    masks = []
    references = []
    priors = []
    for chip in CHIPS:
        keep = data["chip"] == chip
        wavelength = np.asarray(data["mlambda"][keep], dtype=float)
        flux = np.asarray(data["flux"][keep], dtype=float)
        reference = np.asarray(data["mtrans"][keep], dtype=float)
        continuum = np.asarray(data["mscal"][keep], dtype=float)
        weight = np.asarray(data["weight"][keep], dtype=float)
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        spectra.append(Spectrum(wavelength=wavelength, flux=flux[order], wavelength_unit="micron"))
        masks.append(np.isfinite(wavelength) & np.isfinite(flux[order]) & (flux[order] > 0) & (weight[order] > 0))
        references.append(reference[order])
        priors.append(continuum[order])
    return spectra, masks, references, priors


def _variants(line_list: LineList, partition_table: PartitionTable):
    h2o = LBLRTMH2OContinuum.from_package_data()
    co2 = LBLRTMCO2Continuum.from_package_data()
    cia_tables = {name: HitranCIATable.from_hitran_cia(path) for name, path in base.CIA_TABLES.items()}

    def components(lines: LineList, *, include_continuum=True, line_wing_mode="lblrtm_dynamic"):
        built = [
            HitranLineAbsorption(
                lines,
                chunk_size=128,
                partition_table=partition_table,
                line_wing_mode=line_wing_mode,
                line_cutoff_cm=base.LINE_CUTOFF_CM,
                subtract_cutoff_profile=base.SUBTRACT_CUTOFF_PROFILE,
                line_taper_cm=base.LINE_TAPER_CM,
                lblrtm_sample=base.LBLRTM_SAMPLE,
                lblrtm_alfal0=base.LBLRTM_ALFAL0,
                lblrtm_hwf3=base.LBLRTM_HWF3,
            )
        ]
        if include_continuum:
            built.extend([H2OContinuumAbsorption(h2o), CO2ContinuumAbsorption(co2)])
        built.extend(PairCIAAbsorption(table, basis_name=name) for name, table in cia_tables.items())
        return tuple(built)

    no_coupling = replace(
        line_list,
        line_flags=np.zeros_like(line_list.line_flags),
        line_coupling_a=np.zeros_like(line_list.line_coupling_a),
        line_coupling_b=np.zeros_like(line_list.line_coupling_b),
    )
    no_broadener = replace(
        line_list,
        broadener_flags=np.zeros_like(line_list.broadener_flags),
        broadener_widths=np.zeros_like(line_list.broadener_widths),
        broadener_temperature_exponents=np.zeros_like(line_list.broadener_temperature_exponents),
        broadener_pressure_shifts=np.zeros_like(line_list.broadener_pressure_shifts),
    )
    return (
        ("base", line_list, components(line_list), {}),
        ("no_line_coupling", no_coupling, components(no_coupling), {}),
        ("no_extra_broadener", no_broadener, components(no_broadener), {}),
        ("no_h2o_co2_continuum", line_list, components(line_list, include_continuum=False), {}),
        ("no_lsf", line_list, components(line_list), {"lsf_box_width_pixels": 0.0, "lsf_lorentz_fwhm_pixels": 0.0}),
        ("molecfit_voigt_lsf", line_list, components(line_list), {"lsf_molecfit_voigt": True}),
        ("center_rebin", line_list, components(line_list), {"high_resolution_rebin_mode": "center"}),
        ("oversampling_10", line_list, components(line_list), {"high_resolution_oversampling": 10.0}),
        ("line_wing_panel", line_list, components(line_list, line_wing_mode="lblrtm_panel"), {"line_wing_mode": "lblrtm_panel"}),
    )


def _fit_config(
    *,
    atmosphere,
    fit_airmass,
    partition_table,
    components,
    species_scales,
    **updates,
):
    boxfwhm_pix = base.molecfit_box_width_pixels(base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits")
    lorentzfwhm_pix = base.best_fit_parameter(
        base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits",
        "lorentzfwhm",
    )
    lsf_reference = base.molecfit_reference_wavelength(
        base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits"
    )
    defaults = {
        "airmass": fit_airmass,
        "continuum_order": 2,
        "continuum_prior_weight": 1.0,
        "continuum_prior_fractional_sigma": 0.03,
        "atmosphere": atmosphere,
        "components": components,
        "partition_table": partition_table,
        "fixed_species_scales": species_scales,
        "line_wing_mode": base.LINE_WING_MODE,
        "line_cutoff_cm": base.LINE_CUTOFF_CM,
        "subtract_cutoff_profile": base.SUBTRACT_CUTOFF_PROFILE,
        "line_taper_cm": base.LINE_TAPER_CM,
        "lblrtm_sample": base.LBLRTM_SAMPLE,
        "lblrtm_alfal0": base.LBLRTM_ALFAL0,
        "lblrtm_hwf3": base.LBLRTM_HWF3,
        "lsf_sigma_pixels": 0.0,
        "lsf_box_width_pixels": boxfwhm_pix,
        "lsf_lorentz_fwhm_pixels": lorentzfwhm_pix,
        "lsf_variable_width": base.USE_VARIABLE_LSF,
        "lsf_reference_wavelength_micron": lsf_reference,
        "lsf_kernel_width_fwhm": base.LSF_KERNEL_WIDTH_FWHM,
        "lsf_molecfit_voigt": base.USE_MOLECFIT_VOIGT_LSF,
        "high_resolution_grid": base.HIGH_RESOLUTION_GRID,
        "high_resolution_oversampling": base.HIGH_RESOLUTION_OVERSAMPLING,
        "high_resolution_margin_pixels": base.HIGH_RESOLUTION_MARGIN_PIXELS,
        "high_resolution_rebin_mode": base.HIGH_RESOLUTION_REBIN_MODE,
        "fit_wavelength_shift": False,
        "fit_lsf_sigma": False,
        "loss": "soft_l1",
        "f_scale": 2.0,
        "min_transmission": 0.03,
    }
    defaults.update(updates)
    return FitConfig(**defaults)


def _bin_rms(diff: np.ndarray, keep: np.ndarray) -> float:
    if not np.any(keep):
        return np.nan
    return float(np.sqrt(np.nanmean(diff[keep] ** 2)))


def _write_pivot(table: Table) -> None:
    variants = list(dict.fromkeys(str(value) for value in table["variant"]))
    chips = sorted(set(int(value) for value in table["chip"]))
    rows = []
    for chip in chips:
        row = {"chip": chip}
        for variant in variants:
            keep = (np.asarray(table["chip"], int) == chip) & (np.asarray(table["variant"], str) == variant)
            if np.any(keep):
                row[f"rms_{variant}"] = float(np.asarray(table["rms"], float)[keep][0])
                row[f"tau_ratio_{variant}"] = float(np.asarray(table["tau_ratio_median"], float)[keep][0])
        rows.append(row)
    Table(rows=rows).write(OUTPUT / "variant_pivot.ecsv", format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    main()
