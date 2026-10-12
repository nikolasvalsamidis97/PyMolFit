"""Full HARPS Beta Pic comparison of fitted atmospheric transmissions.

Run directly to reuse saved fits and show mean_absolute_deviation.png (the legacy
filename is retained for the paper), displaying 560--610 nm. Full-spectrum
products and statistics are retained. --refit
repeats both fits; --no-show saves the PNG without opening a window. Intermediate
data are kept in validation_results/beta_pic_full_continua.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pymolfit import (  # noqa: E402
    AtmosphereProfile,
    TheoreticalSpectrum,
    aer_catalog_status,
    correct_file,
    load_spectrum,
)
from pymolfit._automatic import estimate_missing_uncertainty  # noqa: E402
from pymolfit.workflow import _spectrum_to_observatory_vacuum  # noqa: E402

DEFAULT_DATA = REPO / "validation_results/beta_pic_full_continua"
DEFAULT_INPUT = REPO / "docs/user_guide/data/ADP.2017-04-07T01_04_41.632.fits"
DEFAULT_TEMPLATE = REPO.parent / "theoretical_spectra/betapic_theoretical.dat.txt"
BACKEND = Path.home() / ".criresflow/molecfit"
RANGE_NM = (588.0, 591.0)
NA_CORE_NM = ((589.19, 589.23), (589.79, 589.83))
RATIO_MIN_TRANSMISSION = 0.1

# Observatory-frame vacuum nm. Chosen before either full-spectrum fit.
FIT_WINDOWS_NM = (
    (588.0, 588.9),
    (590.1, 591.0),
    (592.8, 593.6),
    (594.0, 594.8),
    (627.5, 629.0),
    (629.5, 630.5),
    (646.0, 647.0),
    (650.0, 651.0),
    (686.4, 688.4),
    (689.0, 690.8),
)
HOLDOUT_NM = (
    (588.745, 588.805),
    (590.285, 590.335),
    (628.05, 628.15),
    (687.15, 687.25),
    (689.5, 689.6),
)
COMPARISON_PROTOCOL = "all_aer_species_continua_v2"
CONTINUUM_SETTINGS = {
    "h2o_continuum": "lblrtm",
    "co2_continuum": "lblrtm",
    "o2_continuum": True,
    "o3_continuum": True,
    "n2_continuum": True,
    "rayleigh": False,
}
SPECIES_PEAK_THRESHOLD = 5e-3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval_mask(wavelength_nm, intervals):
    mask = np.zeros(len(wavelength_nm), dtype=bool)
    for lower, upper in intervals:
        mask |= (wavelength_nm >= lower) & (wavelength_nm <= upper)
    return mask


def run_recipe(work, recipe, options, frames):
    sof = work / f"{recipe}.sof"
    sof.write_text("".join(f"{path} {tag}\n" for path, tag in frames))
    command = [
        str(BACKEND / "bin/esorex"),
        f"--recipe-dir={BACKEND / 'lib/esopipes-plugins/molecfit-4.4.4'}",
        "--output-dir=.",
        f"--log-file={recipe}.esorex.log",
        recipe,
    ]
    command += [f"--{key}={value}" for key, value in options.items()]
    command.append(sof.name)
    start = time.perf_counter()
    with (work / f"{recipe}.stdout.log").open("w") as log:
        completed = subprocess.run(
            command,
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            timeout=900,
        )
    record = {
        "command": command,
        "seconds": time.perf_counter() - start,
        "returncode": completed.returncode,
    }
    (work / f"{recipe}.command.json").write_text(json.dumps(record, indent=2))
    if completed.returncode:
        raise RuntimeError(f"{recipe} failed; inspect {recipe}.stdout.log")
    return record


def fit_molecfit(output, spectrum, header, excluded, *, model_options=None):
    destination = output / "molecfit"
    destination.mkdir(exist_ok=True)
    catalog = aer_catalog_status()
    if catalog is None:
        raise RuntimeError("No local AER catalogue")
    # Molecfit serialises some executable/data paths into 30-character fields.
    with tempfile.TemporaryDirectory(prefix="bp_", dir="/tmp") as temporary:
        work = Path(temporary)
        (work / "b").symlink_to(BACKEND)
        shutil.copy2(output / "input.fits", work / "input.fits")
        shutil.copy2(output / "gdas.fits", work / "gdas.fits")
        data = work / "data"
        data.mkdir()
        (data / "profiles").symlink_to(BACKEND / "share/molecfit/data/profiles")
        (data / "hitran").symlink_to(catalog.catalog_path.parent)
        (work / "tmp").mkdir()
        width = (
            np.mean(spectrum.wavelength)
            / header["SPEC_RES"]
            / np.median(np.diff(spectrum.wavelength))
        )
        options = {
            "TELLURICCORR_PATH": work / "b",
            "TELLURICCORR_DATA_PATH": data,
            "TMP_PATH": work / "tmp",
            "COLUMN_LAMBDA": "lambda",
            "COLUMN_FLUX": "flux",
            "COLUMN_DFLUX": "dflux",
            "COLUMN_MASK": "mask",
            "WLG_TO_MICRON": 1,
            "WAVELENGTH_FRAME": "VAC",
            "LIST_MOLEC": "H2O",
            "FIT_MOLEC": "1",
            "REL_COL": "1",
            "WAVE_INCLUDE": f"{spectrum.wavelength[0]},{spectrum.wavelength[-1]}",
            "WAVE_EXCLUDE": ",".join(str(x) for pair in excluded for x in pair),
            "FIT_CONTINUUM": "1",
            "CONTINUUM_N": "2",
            "CONTINUUM_CONST": 1,
            "FIT_WLC": "1",
            "WLC_N": 1,
            "WLC_CONST": 0,
            "FIT_RES_GAUSS": "TRUE",
            "RES_GAUSS": float(width),
            "FIT_RES_BOX": "FALSE",
            "RES_BOX": 0,
            "FIT_RES_LORENTZ": "FALSE",
            "RES_LORENTZ": 0,
            "VARKERN": "FALSE",
            "KERNMODE": "FALSE",
            "KERNFAC": 3,
            "SLIT_WIDTH_KEYWORD": "NONE",
            "SLIT_WIDTH_VALUE": 0.4,
            "GDAS_PROFILE": work / "gdas.fits",
            "REFERENCE_ATMOSPHERIC": "equ.fits",
            "LNFL_LINE_DB": catalog.catalog_path.name,
            "LBLRTM_ICNTNM": 5,  # Molecular continua enabled; Rayleigh disabled.
            "FTOL": 1e-10,
            "XTOL": 1e-10,
        }
        options.update(model_options or {})
        records = []
        try:
            print("Running independent Molecfit model fit...", flush=True)
            records.append(run_recipe(work, "molecfit_model", options, [("input.fits", "SCIENCE")]))
            frames = [("input.fits", "SCIENCE")]
            frames += [
                (f"{tag}.fits", tag)
                for tag in ("MODEL_MOLECULES", "ATM_PARAMETERS", "BEST_FIT_PARAMETERS")
            ]
            print("Evaluating Molecfit over the entire input spectrum...", flush=True)
            records.append(
                run_recipe(
                    work,
                    "molecfit_calctrans",
                    {
                        "MAPPING_ATMOSPHERIC": "0,1",
                        "MAPPING_CONVOLVE": "0,1",
                    },
                    frames,
                )
            )
            records.append(
                run_recipe(
                    work,
                    "molecfit_correct",
                    {
                        "MAPPING_CORRECT": "0,1",
                        "WLC_REF": "DATA",
                    },
                    [("input.fits", "SCIENCE"), ("TELLURIC_CORR.fits", "TELLURIC_CORR")],
                )
            )
        finally:
            for file in work.iterdir():
                if file.is_file():
                    shutil.copy2(file, destination / file.name)
        return records, catalog.manifest.get("catalog_sha256")


def prepare(
    input_path,
    template_path,
    output,
    *,
    wavelength_range_nm=None,
    fit_windows_nm=FIT_WINDOWS_NM,
    holdout_nm=HOLDOUT_NM,
    core_nm=NA_CORE_NM,
    stellar_options=None,
    comparison_protocol=COMPARISON_PROTOCOL,
    stellar_frame_from_metadata=False,
):
    header = fits.getheader(input_path)
    native = load_spectrum(input_path, save_header=False)
    observed = _spectrum_to_observatory_vacuum(native, header).to_unit("micron")
    if wavelength_range_nm is not None:
        lo, hi = wavelength_range_nm
        keep = (observed.wavelength * 1000 >= lo) & (observed.wavelength * 1000 <= hi)
        observed = replace(
            observed,
            wavelength=observed.wavelength[keep],
            flux=observed.flux[keep],
            uncertainty=None if observed.uncertainty is None else observed.uncertainty[keep],
            mask=observed.valid[keep],
            group_id=None,
        )
    valid = observed.valid & (observed.flux > 0)
    flux_scale = float(np.median(observed.flux[valid]))
    spectrum = replace(
        observed,
        flux=observed.flux / flux_scale,
        uncertainty=None if observed.uncertainty is None else observed.uncertainty / flux_scale,
        mask=valid,
        group_id=None,
    )
    template_settings = {
        "radial_velocity_kms": 29.1,
        "vsini_kms": 125,
        "resolving_power": float(header["SPEC_RES"]),
        "mask_depth": 0.015,
        "fit_velocity_offset": False,
        "mask_padding_kms": 5,
    }
    template_settings.update(stellar_options or {})
    theory = TheoreticalSpectrum(template_path, **template_settings)
    frame_factor = (
        observed.meta.get("observatory_erf_factor", 1) if stellar_frame_from_metadata else 1
    )
    stellar = theory.build_mask(spectrum, frame_correction_factor=frame_factor)
    excluded = tuple(stellar.selection_for_spectrum(spectrum).exclude_ranges)
    excluded += tuple((a / 1000, b / 1000) for a, b in holdout_nm + core_nm)
    fit_ranges = tuple((a / 1000, b / 1000) for a, b in fit_windows_nm)
    spectrum, noise = estimate_missing_uncertainty(spectrum, fit_ranges)
    wave_nm = spectrum.wavelength * 1000
    selected = interval_mask(wave_nm, fit_windows_nm)
    fit_mask = spectrum.valid & selected & ~interval_mask(spectrum.wavelength, excluded)
    if fit_mask.sum() < 100:
        raise ValueError("Too few common fitting pixels")
    atmosphere = AtmosphereProfile.from_fits_header_mipas_gdas(
        header, reference_wavenumber_cm=1e4 / np.median(spectrum.wavelength)
    )
    if "gdas_profile" not in atmosphere.metadata:
        raise RuntimeError("Time-local GDAS is required")
    shutil.copy2(atmosphere.metadata["gdas_profile"], output / "gdas.fits")
    header["SPECSYS"] = "TOPOCENT"
    header["AIRORVAC"] = "vacuum"
    header["AIRMASS"] = atmosphere.metadata["airmass"]
    header["HIERARCH ESO TEL ALT"] = float(np.degrees(np.arcsin(1 / header["AIRMASS"])))
    # ESPRESSO identifies its active telescope as TEL1--TEL4; Molecfit expects TEL.
    for suffix in (
        "AMBI RHUM",
        "AMBI PRES START",
        "AMBI TEMP",
        "TH M1 TEMP",
        "GEOELEV",
        "GEOLON",
        "GEOLAT",
    ):
        key = f"ESO TEL {suffix}"
        if key not in header:
            candidates = [
                f"ESO TEL{i} {suffix}" for i in range(1, 5) if f"ESO TEL{i} {suffix}" in header
            ]
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify one active telescope for {key}")
            header[f"HIERARCH {key}"] = header[candidates[0]]
    columns = [
        fits.Column(name="lambda", format="D", unit="um", array=spectrum.wavelength),
        fits.Column(name="flux", format="D", array=spectrum.flux),
        fits.Column(name="dflux", format="D", array=spectrum.uncertainty),
        fits.Column(name="mask", format="J", array=(~spectrum.valid).astype(int)),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns, header=header)]
    ).writeto(output / "input.fits", overwrite=True)
    Table(
        {
            "wavelength_nm": wave_nm,
            "flux": spectrum.flux,
            "uncertainty": spectrum.uncertainty,
            "valid": spectrum.valid,
            "fit": fit_mask,
            "holdout": spectrum.valid & interval_mask(wave_nm, holdout_nm),
            "stellar": interval_mask(
                spectrum.wavelength, stellar.selection_for_spectrum(spectrum).exclude_ranges
            ),
        }
    ).write(output / "input.ecsv", overwrite=True)
    provenance = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "template": str(template_path),
        "template_sha256": sha256(template_path),
        "date_obs": header["DATE-OBS"],
        "instrument": header["INSTRUME"],
        "range_nm": [float(wave_nm[0]), float(wave_nm[-1])],
        "wavelength_medium": "vacuum",
        "wavelength_frame": "observatory",
        "comparison_protocol": comparison_protocol,
        "constant_flux_scale": flux_scale,
        "fit_windows_nm": fit_windows_nm,
        "heldout_nm": holdout_nm,
        "excluded_ranges_micron": excluded,
        "fit_pixels": int(fit_mask.sum()),
        "invalid_pixels": int((~spectrum.valid).sum()),
        "zero_flux_policy": "Zero/nonpositive input samples treated as missing, not stellar absorption",
        "noise": noise,
        "atmosphere": atmosphere.metadata,
        "gdas_sha256": sha256(output / "gdas.fits"),
        "stellar_exclusion_settings": {
            "observer_velocity_kms": template_settings["radial_velocity_kms"],
            "vsini_kms": template_settings["vsini_kms"],
            "mask_depth": template_settings["mask_depth"],
            "padding_kms": template_settings["mask_padding_kms"],
            "frame_correction_factor": frame_factor,
        },
        "script_sha256": sha256(__file__),
        "source_hashes": {
            str(p.relative_to(REPO)): sha256(p)
            for p in sorted((REPO / "src/pymolfit").glob("*.py"))
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return spectrum, header, fit_ranges, excluded, provenance


def molecular_policy(result):
    """Share the included molecules and observability decision, not fitted values."""
    included = sorted(set(result.provenance["line_species"]) | set(result.species_scales))
    fitted = [name for name in included if f"log_scale:{name}" in result.parameter_names]
    fixed = {
        name: float(result.species_scales.get(name, 1.0)) for name in included if name not in fitted
    }
    if any(not np.isclose(value, 1.0) for value in fixed.values()):
        raise ValueError("Unconstrained species must retain their atmospheric-profile abundance")
    if not included or not fitted:
        raise ValueError("No molecular columns were fitted")
    # The separately named nitrogen continuum uses N2 in Molecfit's molecule list.
    aliases = {"N2_continuum": "N2"}
    return {
        "included": sorted({aliases.get(name, name) for name in included}),
        "fitted": sorted({aliases.get(name, name) for name in fitted}),
        "fixed": {aliases.get(name, name): value for name, value in fixed.items()},
        "component_aliases": aliases,
        "minimum_peak_optical_depth": SPECIES_PEAK_THRESHOLD,
        "selection": "All AER species passing the wavelength and line-strength filters, plus enabled molecular continua; no species whitelist",
        "fitting": "The same observability decision is used by both engines; weak species stay at unity. Fits start independently at unity.",
    }


def run_corrections(
    input_path,
    template_path,
    output,
    *,
    prepare_options=None,
    settings_overrides=None,
    molecfit_options=None,
):
    spectrum, header, fit_ranges, excluded, provenance = prepare(
        input_path, template_path, output, **(prepare_options or {})
    )
    reference_wave = float(np.median(spectrum.wavelength))
    sigma = (
        reference_wave
        / header["SPEC_RES"]
        / np.median(np.diff(spectrum.wavelength))
        / np.sqrt(8 * np.log(2))
    )
    settings = {
        "wavelength_medium": "vacuum",
        "wavelength_unit": "micron",
        "wavelength_col": "lambda",
        "flux_col": "flux",
        "uncertainty_col": "dflux",
        "min_line_strength": 1e-32,
        "minimum_species_peak_optical_depth": SPECIES_PEAK_THRESHOLD,
        # Short optical windows cannot separate broad ozone from a local continuum.
        # Keep its profile column in both engines; this does not omit its opacity.
        "fixed_species_scales": {"O3": 1.0},
        "gdas_profile": str(output / "gdas.fits"),
        "continuum_order": 2,
        "auto_segment": True,
        "segment_size": 0.005,
        "automatic_fit_regions": False,
        "fit_ranges": fit_ranges,
        "exclude_ranges": excluded,
        "fit_wavelength_shift": False,
        "fit_wavelength_polynomial": True,
        "wavelength_polynomial_order": 1,
        "lsf_sigma_pixels": float(sigma),
        "fit_lsf_sigma": True,
        "lsf_variable_width": True,
        "lsf_reference_wavelength_micron": reference_wave,
        "lsf_lorentz_fwhm_pixels": 0.0,
        "fit_lsf_lorentz_fwhm": False,
        "loss": "soft_l1",
        "f_scale": 1.0,
        "min_transmission": 0.01,
        **CONTINUUM_SETTINGS,
        "report": True,
    }
    settings.update(settings_overrides or {})
    provenance["pymolfit_settings"] = settings
    if molecfit_options:
        provenance["molecfit_model_options"] = molecfit_options
    start = time.perf_counter()
    print(
        f"Fitting PyMolFit and evaluating the {header.get('INSTRUME', 'spectral')} input...",
        flush=True,
    )
    result = correct_file(output / "input.fits", output_path=output / "pymolfit.ecsv", **settings)
    provenance.update(
        pymolfit_seconds=time.perf_counter() - start,
        pymolfit_success=bool(result.success),
        pymolfit_message=result.message,
    )
    if result.success:
        provenance["species_policy"] = molecular_policy(result)
        provenance["species"] = provenance["species_policy"]["included"]
        print(f"Molecular policy: {provenance['species_policy']}", flush=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if not result.success:
        raise RuntimeError(f"PyMolFit did not converge: {result.message}")
    run_molecfit_from_saved(output)


def run_molecfit_from_saved(output):
    """Run ESO recipes on exactly the saved PyMolFit input, including its quality mask."""
    provenance = json.loads((output / "provenance.json").read_text())
    spectrum = load_spectrum(
        output / "input.fits",
        wavelength_col="lambda",
        flux_col="flux",
        uncertainty_col="dflux",
        wavelength_unit="micron",
        wavelength_medium="vacuum",
        save_header=False,
    )
    header = fits.getheader(output / "input.fits")
    common = Table.read(output / "input.ecsv")
    if not np.array_equal(spectrum.valid, common["valid"]):
        raise ValueError("Saved input quality mask differs from the common pixels")
    # Molecfit interprets mixed binary masks as 1=good, whereas the original mask
    # is 0=good. Keep both explicitly named columns without changing flux/errors.
    with fits.open(output / "input.fits") as hdus:
        if "valid" not in hdus[1].columns.names:
            columns = hdus[1].columns + fits.ColDefs(
                [fits.Column(name="valid", format="J", array=spectrum.valid.astype(int))]
            )
            updated = fits.HDUList(
                [hdus[0].copy(), fits.BinTableHDU.from_columns(columns, header=hdus[1].header)]
            )
        else:
            updated = None
    if updated is not None:
        updated.writeto(output / "input.fits", overwrite=True)
        updated.close()
    fit_ranges = [(a / 1000, b / 1000) for a, b in provenance["fit_windows_nm"]]
    policy = provenance["species_policy"]
    included, fitted = policy["included"], policy["fitted"]
    options = {
        "LIST_MOLEC": ",".join(included),
        "FIT_MOLEC": ",".join("1" if name in fitted else "0" for name in included),
        "REL_COL": ",".join("1" for _ in included),
        "WAVE_INCLUDE": ",".join(str(x) for pair in fit_ranges for x in pair),
        "VARKERN": "TRUE",
        "COLUMN_MASK": "valid",
    }
    options.update(provenance.get("molecfit_model_options", {}))
    records, catalog_sha = fit_molecfit(
        output, spectrum, header, provenance["excluded_ranges_micron"], model_options=options
    )
    provenance.update(
        molecfit_runs=records,
        aer_sha256=catalog_sha,
        mask_convention="mask: 0=good for PyMolFit; valid: 1=good for Molecfit; identical pixels",
        shared_input_sha256=sha256(output / "input.fits"),
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))


def collect_results(output):
    table = Table.read(output / "input.ecsv")
    py = Table.read(output / "pymolfit.ecsv")
    mf = Table.read(output / "molecfit/SCIENCE_TELLURIC_CORR_input.fits", hdu=1)
    parameters = Table.read(output / "molecfit/BEST_FIT_PARAMETERS.fits", hdu=1)
    values = {
        str(row["parameter"]).strip(): float(row["value"])
        for row in parameters
        if not np.ma.is_masked(row["value"])
    }
    if not py.meta["fit_success"] or values["status"] not in (1, 2, 3, 4):
        raise RuntimeError("Both atmospheric fits must converge")
    wave = np.asarray(table["wavelength_nm"])
    for other in (py["wavelength"] * 1000, mf["lambda"] * 1000):
        if len(other) != len(wave) or not np.allclose(wave, other, rtol=0, atol=1e-8):
            raise ValueError("Correction changed the observed wavelength grid")
    if not np.allclose(py["flux"], table["flux"], rtol=0, atol=1e-12):
        raise ValueError("PyMolFit did not use the common flux")
    if not np.allclose(py["uncertainty"], table["uncertainty"], rtol=1e-9):
        raise ValueError("PyMolFit did not use the common uncertainties")
    if not np.array_equal(py["fit_mask"], table["fit"]):
        raise ValueError("PyMolFit fit mask differs from the common mask")
    fitted = Table.read(output / "molecfit/BEST_FIT_MODEL.fits", hdu=1)
    indices = np.searchsorted(wave, fitted["lambda"] * 1000 - 1e-8)
    if np.any(indices >= len(wave)) or not np.allclose(
        wave[indices], fitted["lambda"] * 1000, atol=1e-8, rtol=0
    ):
        raise ValueError("Molecfit fitting wavelengths differ from the input")
    expected = np.asarray(table["fit"])
    # Molecfit residuals use measurement weight * model weight. Measurement
    # weights alone can remain nonzero outside the selected fitting ranges.
    effective_weight = np.asarray(fitted["weight"] * fitted["mweight"])
    actual = np.zeros(len(wave), bool)
    actual[indices] = effective_weight > 0
    if not np.array_equal(actual, expected):
        raise ValueError(
            f"Molecfit fit mask differs at {np.count_nonzero(actual != expected)} pixels"
        )
    if not np.allclose(fitted["flux"], table["flux"][indices], rtol=0, atol=1e-12):
        raise ValueError("Molecfit did not use the common flux")
    weights = np.where(expected[indices], 1 / table["uncertainty"][indices], 0)
    if not np.allclose(effective_weight, weights, rtol=1e-9):
        raise ValueError("Molecfit weights differ from the common uncertainties")
    with fits.open(output / "molecfit/TELLURIC_CORR.fits") as hdus:
        transmission = np.asarray(hdus[1].data, dtype=float).reshape(-1)
    table["pymolfit_transmission"] = py["transmission"]
    table["molecfit_transmission"] = transmission
    table["pymolfit_corrected"] = py["corrected_flux"]
    table["molecfit_corrected"] = mf["flux"]
    valid = np.asarray(table["valid"])
    for name in ("pymolfit", "molecfit"):
        trans = np.asarray(table[f"{name}_transmission"])
        usable = valid & np.isfinite(trans) & (trans > 0.01)
        if not np.allclose(
            table[f"{name}_corrected"][usable], table["flux"][usable] / trans[usable], rtol=1e-9
        ):
            raise ValueError(f"{name} correction does not equal observed flux / transmission")
    table.meta.update(
        wavelength_frame="observatory",
        wavelength_medium="vacuum",
        molecfit_parameters=json.dumps(values),
    )
    table.write(output / "comparison.ecsv", overwrite=True)
    return table


def compare_transmissions(table, minimum_transmission=RATIO_MIN_TRANSMISSION):
    """Compare native detector-grid transmissions, without a source reference."""
    if not np.isfinite(minimum_transmission) or not 0 < minimum_transmission <= 1:
        raise ValueError("The minimum transmission must be finite and in (0, 1]")
    result = table.copy()
    wave, py, mf = (
        np.ma.asarray(table[name], dtype=float).filled(np.nan)
        for name in ("wavelength_nm", "pymolfit_transmission", "molecfit_transmission")
    )
    valid = np.ma.asarray(table["valid"], dtype=bool).filled(False).copy()
    valid &= np.isfinite(wave) & np.isfinite(py) & np.isfinite(mf)
    ratio_valid = valid & (py >= minimum_transmission) & (mf >= minimum_transmission)
    difference = np.full(len(table), np.nan)
    difference[ratio_valid] = py[ratio_valid] / mf[ratio_valid] - 1
    result["transmission_valid"] = valid
    result["ratio_valid"] = ratio_valid
    result["relative_transmission_difference"] = difference
    result.meta.update(
        relative_difference_definition="pymolfit_transmission / molecfit_transmission - 1",
        relative_difference_units="dimensionless",
        relative_difference_sign="negative: PyMolFit lower transmission; positive: PyMolFit higher transmission",
        ratio_minimum_transmission=minimum_transmission,
        sampling="Native pixels; no binning, smoothing, interpolation or reference calibration",
    )
    return result


def save_transmission_comparison(output, table, minimum_transmission=RATIO_MIN_TRANSMISSION):
    table = compare_transmissions(table, minimum_transmission)
    good = np.asarray(table["ratio_valid"])
    difference = np.asarray(table["relative_transmission_difference"])
    subsets = {"all": good}
    for name in ("fit", "holdout"):
        if name in table.colnames:
            subsets[name] = good & np.ma.asarray(table[name], dtype=bool).filled(False)
    summary = {
        "definition": table.meta["relative_difference_definition"],
        "units": "dimensionless",
        "minimum_transmission_both_engines": minimum_transmission,
        "native_pixels": len(table),
        "transmission_pixels": int(np.count_nonzero(table["transmission_valid"])),
        "ratio_pixels": int(good.sum()),
        "statistics": {
            name: {
                "pixels": int(mask.sum()),
                "mean": float(np.mean(difference[mask])) if mask.any() else None,
                "mean_absolute": float(np.mean(np.abs(difference[mask]))) if mask.any() else None,
                "rms": float(np.sqrt(np.mean(difference[mask] ** 2))) if mask.any() else None,
            }
            for name, mask in subsets.items()
        },
        "caution": "Transmission agreement is not an independent measure of correction accuracy.",
    }
    table.write(output / "transmission_comparison.ecsv", overwrite=True)
    (output / "transmission_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    return table


def plot_transmission_curves(axes, table, *, offset=0, index=0):
    """Insert plot-only breaks without removing samples on either side of a gap."""
    top, bottom = axes
    wave = np.ma.asarray(table["wavelength_nm"], dtype=float).filled(np.nan)
    steps = np.diff(wave)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    gap = 10 * np.median(positive) if len(positive) else np.inf
    breaks = np.flatnonzero((steps > gap) | (steps <= 0)) + 1
    x = np.insert(wave + offset, breaks, np.nan)
    valid = np.asarray(table["transmission_valid"])
    for name, label, color, style in (
        ("pymolfit", "PyMolFit", "#0072B2", "-"),
        ("molecfit", "Molecfit", "#D55E00", "--"),
    ):
        values = np.ma.asarray(table[f"{name}_transmission"], dtype=float).filled(np.nan)
        top.plot(
            x,
            np.insert(np.where(valid, values, np.nan), breaks, np.nan),
            color=color,
            ls=style,
            lw=0.9,
            label=label if index == 0 else "_nolegend_",
            gid=f"transmission:{name}:{index}",
        )
    values = np.ma.asarray(table["relative_transmission_difference"], dtype=float).filled(np.nan)
    bottom.plot(x, np.insert(values, breaks, np.nan), color="0.25", lw=0.8, gid=f"ratio:{index}")


def style_transmission_axes(axes):
    top, bottom = axes
    top.set_ylabel("Transmission", fontsize=13.4)
    top.margins(y=0.05)
    top.legend(fontsize=12.55, loc="lower left")
    bottom.set_ylabel(r"$T_{\mathrm{PyMolFit}}/T_{\mathrm{Molecfit}} - 1$", fontsize=13.4)
    bottom.axhline(0, color="0.5", lw=0.8, ls="--")
    bottom.set_xlabel("Observatory-frame vacuum wavelength [nm]", fontsize=13.4)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=11.7)


def plot_transmissions(
    output, table, minimum_transmission, show, *, figure_path, wavelength_range=(560.0, 610.0)
):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = save_transmission_comparison(output, table, minimum_transmission)
    if wavelength_range is not None:
        lo, hi = wavelength_range
        table = table[(table["wavelength_nm"] >= lo) & (table["wavelength_nm"] <= hi)]
        if not len(table):
            raise ValueError("No transmission samples in the requested display range")
    fig, axes = plt.subplots(2, 1, figsize=(13, 4), sharex=True, layout="constrained")
    plot_transmission_curves(axes, table)
    style_transmission_axes(axes)
    axes[-1].set_xlim(
        wavelength_range or (np.nanmin(table["wavelength_nm"]), np.nanmax(table["wavelength_nm"]))
    )
    fig.suptitle("Beta Pictoris | HARPS | Atmospheric transmission", fontsize=15.7)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    print(f"Saved {figure_path}\nMetrics: {output / 'transmission_metrics.json'}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Fit a different exposure")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="Saved fit products")
    parser.add_argument("--output", type=Path, default=HERE / "mean_absolute_deviation.png")
    parser.add_argument(
        "--min-transmission",
        type=float,
        default=RATIO_MIN_TRANSMISSION,
        help="Minimum transmission from both engines for the ratio panel only",
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refit", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.min_transmission) or not 0 < args.min_transmission <= 1:
        parser.error("--min-transmission must be finite and in (0, 1]")
    if args.plot_only and args.input is not None:
        parser.error("--plot-only uses saved corrections; omit --input")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must name a PNG file; use --data-dir for fit products")
    output = args.data_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed = (output / "molecfit/SCIENCE_TELLURIC_CORR_input.fits").is_file() and (
        output / "pymolfit.ecsv"
    ).is_file()
    if completed and not args.refit and not args.plot_only:
        provenance = json.loads((output / "provenance.json").read_text())
        if provenance.get("comparison_protocol") != COMPARISON_PROTOCOL:
            parser.error("Saved fits use an older species selection; use a separate output folder")
    if args.refit or args.input is not None or (not args.plot_only and not completed):
        run_corrections(
            (args.input or DEFAULT_INPUT).expanduser().resolve(),
            args.template.expanduser().resolve(),
            output,
        )
    table = collect_results(output)
    if args.fit_only:
        return
    plot_transmissions(
        output,
        table,
        args.min_transmission,
        args.show,
        figure_path=args.output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
