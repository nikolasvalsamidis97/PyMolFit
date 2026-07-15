from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from pymolfit import (
    AtmosphereProfile,
    CO2ContinuumAbsorption,
    FitConfig,
    H2OContinuumAbsorption,
    HitranCIATable,
    HitranLineAbsorption,
    IsotopologueMetadata,
    LineList,
    LBLRTMCO2Continuum,
    LBLRTMH2OContinuum,
    N2ContinuumAbsorption,
    PairCIAAbsorption,
    PartitionTable,
    Spectrum,
    fit_telluric_segments,
)


PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = PROJECT / "local_tests" / "external_absorption"
RHO01_INPUT = Path(
    os.environ.get(
        "PYMOLFIT_RHO01_MOLECFIT_INPUT",
        PROJECT / "local_tests" / "data" / "rho01" / "molecfit_input",
    )
)
MOLECFIT_SPEED_CSV = Path("/tmp/pymolfit_molecfit_speed_benchmark/molecfit_speed_summary.csv")
ASSISTED_PYMOLFIT_OUTPUT = PROJECT / "local_tests" / "speed_benchmark_pymolfit_dynamic_noplot"
OUTPUT = PROJECT / "local_tests" / "fair_runtime_benchmark"

SCIENCE = RHO01_INPUT / "SCIENCE_A.fits"
LINE_LIST = EXTERNAL / "aer_lband_h2o_co2_co_ch4_o2_strength1e-32.ecsv"
ISO_METADATA = EXTERNAL / "hitran_iso_metadata_lband.ecsv"
HITRAN_Q_DIR = EXTERNAL / "hitran_q"
H2O_CONTINUUM = EXTERNAL / "absco-ref_wv-mt-ckd.nc"
CIA_TABLES = {
    "CO2-CO2_CIA": EXTERNAL / "CO2-CO2_2024.cia",
    "O2-O2_CIA": EXTERNAL / "O2-O2_2024.cia",
}
HIGH_RESOLUTION_REBIN_MODE = os.environ.get("PYMOLFIT_HIGH_RESOLUTION_REBIN_MODE", "molecfit_overlap")
SUBTRACT_CUTOFF_PROFILE = os.environ.get("PYMOLFIT_SUBTRACT_CUTOFF_PROFILE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
LINE_WING_MODE = os.environ.get("PYMOLFIT_BENCHMARK_LINE_WING_MODE", "lblrtm_panel")
FIT_FTOL = float(os.environ.get("PYMOLFIT_BENCHMARK_FTOL", "1e-10"))
FIT_XTOL = float(os.environ.get("PYMOLFIT_BENCHMARK_XTOL", "1e-10"))
BASIS_WORKERS = int(os.environ.get("PYMOLFIT_BENCHMARK_BASIS_WORKERS", "0"))


def representative_airmass(header: fits.Header) -> float:
    start = header.get("ESO TEL AIRM START", header.get("HIERARCH ESO TEL AIRM START", 1.0))
    end = header.get("ESO TEL AIRM END", header.get("HIERARCH ESO TEL AIRM END", start))
    return 0.5 * (float(start) + float(end))


def observatory_altitude_m(header: fits.Header) -> float:
    return float(header.get("ESO TEL GEOELEV", header.get("HIERARCH ESO TEL GEOELEV", 2648.0)))


def input_lsf_width_pixels(header: fits.Header) -> float:
    keys = (
        "ESO QC SLITFWHM MED AVG",
        "HIERARCH ESO QC SLITFWHM MED AVG",
        "ESO QC SLITFWHM2 AVG",
        "HIERARCH ESO QC SLITFWHM2 AVG",
    )
    for key in keys:
        if key in header and np.isfinite(float(header[key])) and float(header[key]) > 0:
            return float(header[key])
    return 1.0


def load_science_segments(path: Path) -> tuple[list[Spectrum], list[np.ndarray], fits.Header]:
    spectra: list[Spectrum] = []
    masks: list[np.ndarray] = []
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        for hdu in hdul[1:]:
            data = hdu.data
            wavelength = np.asarray(data["WAVE"], dtype=float)
            flux = np.asarray(data["SPEC"], dtype=float)
            uncertainty = np.asarray(data["ERR"], dtype=float)
            order = np.argsort(wavelength)
            wavelength = wavelength[order]
            flux = flux[order]
            uncertainty = uncertainty[order]
            mask = (
                np.isfinite(wavelength)
                & np.isfinite(flux)
                & np.isfinite(uncertainty)
                & (flux > 0)
                & (uncertainty > 0)
            )
            spectra.append(
                Spectrum(
                    wavelength=wavelength,
                    flux=flux,
                    uncertainty=uncertainty,
                    wavelength_unit="micron",
                    wavelength_medium="vacuum",
                )
            )
            masks.append(mask)
    return spectra, masks, header


def build_absorption_inputs():
    isotopologues = IsotopologueMetadata.from_table(ISO_METADATA)
    partition_table = PartitionTable.from_lblrtm_package_data()
    line_list = LineList.from_table(LINE_LIST).with_isotopologue_metadata(isotopologues)
    h2o_continuum = LBLRTMH2OContinuum.from_package_data()
    co2_continuum = LBLRTMCO2Continuum.from_package_data()
    cia_tables = {name: HitranCIATable.from_hitran_cia(path) for name, path in CIA_TABLES.items()}
    components = [
        HitranLineAbsorption(
            line_list,
            chunk_size=0,
            partition_table=partition_table,
            line_wing_mode=LINE_WING_MODE,
            subtract_cutoff_profile=SUBTRACT_CUTOFF_PROFILE,
        ),
        H2OContinuumAbsorption(h2o_continuum),
        CO2ContinuumAbsorption(co2_continuum),
        N2ContinuumAbsorption(),
    ]
    for name, cia in cia_tables.items():
        components.append(PairCIAAbsorption(cia, basis_name=name))
    return line_list, partition_table, tuple(components)


def benchmark_pymolfit_from_science() -> tuple[float, dict[str, object]]:
    t0 = time.perf_counter()
    spectra, masks, header = load_science_segments(SCIENCE)
    line_list, partition_table, components = build_absorption_inputs()
    setup_seconds = time.perf_counter() - t0

    airmass = representative_airmass(header)
    atmosphere = AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        airmass=airmass,
        observatory_altitude_m=observatory_altitude_m(header),
        gdas_mode="auto",
        reference_wavenumber_cm=float(
            np.nanmedian(
                np.concatenate(
                    [1.0e4 / spectrum.to_unit("micron").wavelength for spectrum in spectra]
                )
            )
        ),
    )
    lsf_width = input_lsf_width_pixels(header)

    t1 = time.perf_counter()
    result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            atmosphere=atmosphere,
            airmass=1.0,
            continuum_order=2,
            components=components,
            partition_table=partition_table,
            fixed_species_scales={
                **{name: 1.0 for name in CIA_TABLES},
                "N2_continuum": 1.0,
            },
            scale_bounds=(1.0e-5, 1.0e5),
            lsf_box_width_pixels=lsf_width,
            fit_lsf_sigma=True,
            lsf_sigma_pixels=1.0,
            lsf_sigma_bounds=(0.0, 4.0),
            high_resolution_grid=True,
            high_resolution_oversampling=5.0,
            high_resolution_margin_pixels=2.0,
            high_resolution_rebin_mode=HIGH_RESOLUTION_REBIN_MODE,
            line_wing_mode=LINE_WING_MODE,
            lsf_kernel_width_fwhm=3.0,
            subtract_cutoff_profile=SUBTRACT_CUTOFF_PROFILE,
            loss="soft_l1",
            f_scale=2.0,
            min_transmission=0.03,
            ftol=FIT_FTOL,
            xtol=FIT_XTOL,
            basis_workers=BASIS_WORKERS,
        ),
        fit_masks=masks,
    )
    fit_seconds = time.perf_counter() - t1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    t2 = time.perf_counter()
    for chip, segment_result, mask in zip(range(1, 19), result.segment_results, masks, strict=True):
        table = segment_result.to_table()
        table["fit_mask"] = mask
        table.write(OUTPUT / f"chip_{chip:02d}_science_input_product.ecsv", format="ascii.ecsv", overwrite=True)
        summaries.append(
            {
                "chip": chip,
                "n_pixels": len(segment_result.spectrum.wavelength),
                "n_fit_pixels": int(np.count_nonzero(mask)),
                "corrected_scatter": segment_result.metrics.get("corrected_scatter", np.nan),
                "min_transmission": float(np.nanmin(segment_result.transmission)),
                "median_transmission": float(np.nanmedian(segment_result.transmission)),
            }
        )
    summary_table = Table(rows=summaries)
    summary_table.meta["source"] = str(SCIENCE)
    summary_table.meta["airmass"] = airmass
    summary_table.meta["lsf_box_width_pixels_from_input_header"] = lsf_width
    summary_table.meta["success"] = bool(result.success)
    summary_table.meta["nfev"] = int(result.nfev)
    summary_table.meta["fit_cost"] = float(result.cost)
    summary_table.meta["species_scales"] = repr(result.species_scales)
    summary_table.meta["lsf_sigma_pixels"] = float(result.lsf_sigma_pixels)
    summary_table.write(OUTPUT / "pymolfit_from_science_summary.ecsv", format="ascii.ecsv", overwrite=True)
    write_seconds = time.perf_counter() - t2

    total = time.perf_counter() - t0
    return total, {
        "setup_seconds": setup_seconds,
        "fit_seconds": fit_seconds,
        "write_seconds": write_seconds,
        "success": result.success,
        "nfev": result.nfev,
        "cost": result.cost,
        "airmass": airmass,
        "lsf_box_width_pixels": lsf_width,
        "lsf_sigma_pixels": result.lsf_sigma_pixels,
        "high_resolution_grid": True,
        "high_resolution_oversampling": 5.0,
        "high_resolution_rebin_mode": HIGH_RESOLUTION_REBIN_MODE,
        "subtract_cutoff_profile": SUBTRACT_CUTOFF_PROFILE,
        "line_wing_mode": LINE_WING_MODE,
        "ftol": FIT_FTOL,
        "xtol": FIT_XTOL,
        "basis_workers": BASIS_WORKERS,
        "line_list_strength_floor": 1.0e-32,
        "continuum_components": "LBLRTM H2O, CO2, and N2 source continua",
        "cia_components": tuple(CIA_TABLES),
        "species_scales": result.species_scales,
    }


def read_molecfit_timing() -> dict[str, float]:
    if not MOLECFIT_SPEED_CSV.exists():
        return {}
    timings: dict[str, float] = {}
    with MOLECFIT_SPEED_CSV.open() as handle:
        for row in csv.DictReader(handle):
            timings[row["stage"]] = float(row["seconds"])
    return timings


def assisted_pymolfit_runtime_note() -> float | None:
    summary = ASSISTED_PYMOLFIT_OUTPUT / "summary.ecsv"
    if not summary.exists():
        return None
    return 61.19


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total, details = benchmark_pymolfit_from_science()
    molecfit = read_molecfit_timing()
    rows = []
    if "TOTAL" in molecfit:
        rows.append(
            {
                "run": "molecfit_official_full_chain",
                "seconds": molecfit.get("TOTAL", np.nan),
                "notes": "molecfit_model + molecfit_calctrans + molecfit_correct from SCIENCE_A.fits",
            }
        )
    if "molecfit_model" in molecfit:
        rows.append(
            {
                "run": "molecfit_model_only",
                "seconds": molecfit.get("molecfit_model", np.nan),
                "notes": "official fitted-model recipe only",
            }
        )
    assisted = assisted_pymolfit_runtime_note()
    if assisted is not None:
        rows.append(
            {
                "run": "pymolfit_assisted_by_molecfit_products",
                "seconds": assisted,
                "notes": "old benchmark; used MOLECFIT_DATA, ATM_PROFILE_COMBINED, BEST_FIT_PARAMETERS",
            }
        )
    rows.append(
        {
            "run": "pymolfit_from_science_input",
            "seconds": total,
            "notes": (
                "loads SCIENCE_A.fits directly; self-contained MIPAS+GDAS atmosphere; "
                "full 1e-32 line list; LBLRTM H2O/CO2/N2 continua; non-overlapping "
                "CO2-CO2/O2-O2 CIA; input-header LSF width; no Molecfit model "
                "products used for fitting"
            ),
        }
    )
    timing_table = Table(rows=rows)
    for key, value in details.items():
        timing_table.meta[f"pymolfit_{key}"] = repr(value)
    timing_table.write(OUTPUT / "runtime_summary.ecsv", format="ascii.ecsv", overwrite=True)
    timing_table.write(OUTPUT / "runtime_summary.csv", format="ascii.csv", overwrite=True)

    print(f"PyMolFit from science input: {total:.3f} s")
    for key, value in details.items():
        print(f"  {key}: {value}")
    if molecfit:
        if "TOTAL" in molecfit:
            print(f"Molecfit full chain: {molecfit['TOTAL']:.3f} s")
        if "molecfit_model" in molecfit:
            print(f"Molecfit model only: {molecfit['molecfit_model']:.3f} s")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
