from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from genmolfit import (
    CO2ContinuumAbsorption,
    H2OContinuumAbsorption,
    HitranLineAbsorption,
    IsotopologueMetadata,
    LBLRTMCO2Continuum,
    LBLRTMH2OContinuum,
    LineList,
    PartitionTable,
    combine_optical_depth_components,
    high_resolution_wavelength_grid,
    transmission_from_high_resolution_basis,
)

import compare_rho01_genmolfit_molecfit_lband as base


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "local_tests" / "rho01_mismatch_diagnostics"
CHIPS = (1, 4, 5, 6, 10, 11, 12, 13, 17, 18)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    isotopologues = IsotopologueMetadata.from_table(base.ISO_METADATA)
    partition_table = PartitionTable.from_hitran_q_directory(base.HITRAN_Q_DIR, isotopologues)
    line_list = LineList.from_table(base.LINE_LIST).with_isotopologue_metadata(isotopologues)
    atmosphere, fit_airmass = base.comparison_atmosphere(
        source=base.ATMOSPHERE_SOURCE,
        airmass=base.representative_airmass(base.MOLECFIT / "molecfit_input" / "SCIENCE_A.fits"),
        h2o_col_mm=base.best_fit_parameter(
            base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits",
            "h2o_col_mm",
        ),
    )
    scales = _scales()
    with fits.open(base.MOLECFIT / "molecfit_model" / "MOLECFIT_DATA.fits") as hdul:
        data = hdul[1].data.copy()

    variants = _variants(line_list, partition_table)
    rows = []
    for variant_name, variant_lines, variant_components, variant_options in variants:
        print(f"variant {variant_name}", flush=True)
        for chip in CHIPS:
            wavelength, reference = _chip_data(data, chip)
            selected_lines = variant_lines.select_range(
                float(np.nanmin(wavelength)),
                float(np.nanmax(wavelength)),
                margin=base.line_margin_micron(wavelength),
            )
            highres_wavelength, highres_per_pixel = high_resolution_wavelength_grid(
                wavelength,
                oversampling=float(variant_options.get("oversampling", base.HIGH_RESOLUTION_OVERSAMPLING)),
                margin_pixels=base.HIGH_RESOLUTION_MARGIN_PIXELS,
            )
            components = _components_for_selected_lines(variant_components, selected_lines)
            species, basis = combine_optical_depth_components(
                highres_wavelength,
                atmosphere,
                components,
            )
            transmission = transmission_from_high_resolution_basis(
                wavelength,
                highres_wavelength,
                species,
                basis,
                highres_pixels_per_observed_pixel=highres_per_pixel,
                species_scales=scales,
                airmass=fit_airmass,
                lsf_box_width_pixels=float(variant_options.get("box", base.molecfit_box_width_pixels(base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits"))),
                lsf_lorentz_fwhm_pixels=float(variant_options.get("lorentz", base.best_fit_parameter(base.MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits", "lorentzfwhm"))),
                lsf_molecfit_voigt=bool(variant_options.get("molecfit_voigt", False)),
                lsf_kernel_width_fwhm=base.LSF_KERNEL_WIDTH_FWHM,
                rebin_mode=str(variant_options.get("rebin", base.HIGH_RESOLUTION_REBIN_MODE)),
            )
            rows.append(_metrics(variant_name, chip, transmission, reference, len(selected_lines.wavelength)))
    table = Table(rows=rows)
    table.write(OUTPUT / "direct_variant_summary.ecsv", format="ascii.ecsv", overwrite=True)
    table.write(OUTPUT / "direct_variant_summary.csv", format="ascii.csv", overwrite=True)
    _write_pivot(table)
    print(f"wrote {OUTPUT}")


def _scales() -> dict[str, float]:
    summary = Table.read(base.OUTPUT / "summary.ecsv")
    scales = {}
    for name in summary.colnames:
        if name.startswith("scale_"):
            key = name.removeprefix("scale_")
            if key.endswith("_CIA"):
                continue
            scales[key] = float(summary[name][0])
    return scales


def _chip_data(data, chip: int) -> tuple[np.ndarray, np.ndarray]:
    keep = data["chip"] == chip
    wavelength = np.asarray(data["mlambda"][keep], dtype=float)
    reference = np.asarray(data["mtrans"][keep], dtype=float)
    order = np.argsort(wavelength)
    return wavelength[order], reference[order]


def _variants(line_list: LineList, partition_table: PartitionTable):
    h2o = LBLRTMH2OContinuum.from_package_data()
    co2 = LBLRTMCO2Continuum.from_package_data()

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
        ("base_no_cia", line_list, components(line_list), {}),
        ("no_line_coupling", no_coupling, components(no_coupling), {}),
        ("no_extra_broadener", no_broadener, components(no_broadener), {}),
        ("no_h2o_co2_continuum", line_list, components(line_list, include_continuum=False), {}),
        ("no_lsf", line_list, components(line_list), {"box": 0.0, "lorentz": 0.0}),
        ("molecfit_voigt_lsf", line_list, components(line_list), {"molecfit_voigt": True}),
        ("center_rebin", line_list, components(line_list), {"rebin": "center"}),
        ("oversampling_10", line_list, components(line_list), {"oversampling": 10.0}),
        ("line_wing_panel", line_list, components(line_list, line_wing_mode="lblrtm_panel"), {}),
    )


def _components_for_selected_lines(components, selected_lines: LineList):
    selected = []
    for component in components:
        if isinstance(component, HitranLineAbsorption):
            selected.append(replace(component, line_list=selected_lines))
        else:
            selected.append(component)
    return tuple(selected)


def _metrics(variant: str, chip: int, transmission: np.ndarray, reference: np.ndarray, n_lines: int) -> dict:
    valid = np.isfinite(transmission) & np.isfinite(reference)
    diff = transmission[valid] - reference[valid]
    ref = reference[valid]
    gen_tau = -np.log(np.clip(transmission[valid], 1.0e-300, 1.0))
    mol_tau = -np.log(np.clip(ref, 1.0e-300, 1.0))
    tau_ok = np.isfinite(gen_tau) & np.isfinite(mol_tau) & (mol_tau > 0)
    return {
        "variant": variant,
        "chip": chip,
        "n_lines": n_lines,
        "rms": float(np.sqrt(np.nanmean(diff**2))),
        "median_abs": float(np.nanmedian(np.abs(diff))),
        "mean_diff": float(np.nanmean(diff)),
        "tau_ratio_median": float(np.nanmedian(gen_tau[tau_ok] / mol_tau[tau_ok])),
        "rms_reference_lt_0p2": _bin_rms(diff, ref < 0.2),
        "rms_reference_0p2_0p8": _bin_rms(diff, (ref >= 0.2) & (ref < 0.8)),
        "rms_reference_gt_0p8": _bin_rms(diff, ref >= 0.8),
    }


def _bin_rms(diff: np.ndarray, keep: np.ndarray) -> float:
    if not np.any(keep):
        return np.nan
    return float(np.sqrt(np.nanmean(diff[keep] ** 2)))


def _write_pivot(table: Table) -> None:
    variants = list(dict.fromkeys(str(value) for value in table["variant"]))
    rows = []
    for chip in sorted(set(int(value) for value in table["chip"])):
        row = {"chip": chip}
        for variant in variants:
            keep = (np.asarray(table["chip"], int) == chip) & (np.asarray(table["variant"], str) == variant)
            if not np.any(keep):
                continue
            row[f"rms_{variant}"] = float(np.asarray(table["rms"], float)[keep][0])
            row[f"tau_ratio_{variant}"] = float(np.asarray(table["tau_ratio_median"], float)[keep][0])
        rows.append(row)
    Table(rows=rows).write(OUTPUT / "direct_variant_pivot.ecsv", format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    main()
