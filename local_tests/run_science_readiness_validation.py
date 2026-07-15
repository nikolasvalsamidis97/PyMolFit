from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import astropy
from astropy.io import fits
from astropy.table import Table
import pymolfit.atmosphere as atmosphere_impl

from pymolfit import (
    __version__,
    AtmosphereProfile,
    FitConfig,
    LineList,
    ModelConfig,
    PhysicalModelConfig,
    ScienceReadinessReport,
    Spectrum,
    ValidationCheck,
    correct_spectrum,
    fit_telluric_segments,
    fit_tellurics,
    fit_tellurics_with_systematics,
    physical_optical_depth_basis,
    transmission_model,
)
from pymolfit.aer_data import (
    AER_CATALOG_FILENAME,
    AER_CATALOG_SHA256,
    AER_CATALOG_VERSION,
    AERCatalogArtifact,
    install_aer_catalog,
)
from pymolfit.model import transmission_from_basis

try:
    from local_tests.molecfit_reference_data import stage_aer_molecfit_data
except ModuleNotFoundError:  # Direct script execution.
    from molecfit_reference_data import stage_aer_molecfit_data


ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_DIR = ROOT / "local_tests" / "science_readiness"
DATA_DIR = CAMPAIGN_DIR / "data" / "xshooter_hd53123"
UVES_DATA_DIR = CAMPAIGN_DIR / "data" / "uves_demo"
UVES_DATASET = (
    "ADP.2020-06-08T15_07_14.471.fits",
    "ADP.2020-06-08T15:07:14.471",
)
UVES_SUMMARY = ROOT / "local_tests" / "uves_official_demo_comparison" / "summary.csv"
HIRES_DATA_DIR = CAMPAIGN_DIR / "data" / "keck_hires_bd17"
HIRES_SUMMARY = ROOT / "local_tests" / "keck_hires_bd17_o2_comparison" / "summary.csv"
HIRES_KOAID = "HI.20040824.18925.fits"
HIRES_ORDER_FILES = (
    "binaryfits/ccd3/flux/HI.20040824.18925_3_04_flux.fits.gz",
    "binaryfits/ccd3/flux/HI.20040824.18925_3_09_flux.fits.gz",
)
HIRES_LEVEL1_PREFIX = "/koadata1/HIRES/20040824/lev1"
KPF_DATA_DIR = CAMPAIGN_DIR / "data" / "kpf_vega"
KPF_FILENAME = "KP.20250519.55029.51_L1.fits"
KPF_SUMMARY = ROOT / "local_tests" / "keck_kpf_vega_o2_comparison" / "summary.csv"
KPF_KOAID = "KP.20250519.55029.51.fits"
KPF_FILEHANDLE = "/KPF/2025/20250519/lev2/KP.20250519.55029.51/KP.20250519.55029.51_L1.fits"
CACHE_PATH = CAMPAIGN_DIR / "cache" / "aer_science_readiness_windows.fits"
CACHE_MANIFEST_PATH = CACHE_PATH.with_suffix(".manifest.json")
CACHE_LAYOUT_VERSION = 3
CACHE_WAVENUMBER_MARGIN_CM = 30.0
OUTPUT_DIR = CAMPAIGN_DIR / "results"
REVIEW_PACKET = OUTPUT_DIR / "independent_review"
REVIEW_ARCHIVE = OUTPUT_DIR / "independent_review_packet.zip"
HITRAN_RECEIPT = OUTPUT_DIR / "authenticated_hitran_receipt.json"
MOLECFIT_ESOREX = Path.home() / ".criresflow" / "molecfit" / "bin" / "esorex"
MOLECFIT_DATA_ROOT = Path.home() / ".criresflow" / "molecfit" / "share" / "molecfit" / "data"

XSHOOTER_DATASETS = {
    "VIS": (
        "ADP.2026-03-26T15_48_39.243.fits",
        "ADP.2026-03-26T15:48:39.243",
    ),
    "NIR": (
        "ADP.2026-03-26T15_48_39.204.fits",
        "ADP.2026-03-26T15:48:39.204",
    ),
}


@dataclass(frozen=True)
class BandCase:
    name: str
    arm: str
    wavelength_min: float
    wavelength_max: float
    species: tuple[str, ...]
    band: str


REAL_BANDS = (
    BandCase("xshooter_o2_a", "VIS", 0.7580, 0.7700, ("O2",), "optical"),
    BandCase("xshooter_h2o_j", "NIR", 1.1000, 1.1700, ("H2O",), "J"),
    BandCase("xshooter_h2o_h", "NIR", 1.5000, 1.5600, ("H2O",), "H"),
    BandCase("xshooter_h2o_co2_k", "NIR", 2.0000, 2.0800, ("H2O", "CO2"), "K"),
)

SYNTHETIC_BANDS = {
    "O2": (0.758, 0.770, "optical", 0.0015),
    "H2O": (1.10, 1.17, "J", 0.0015),
    "CO2": (2.00, 2.08, "K", 0.0025),
    "CH4": (3.28, 3.36, "L", 0.0025),
    "N2O": (4.45, 4.55, "M", 0.0025),
    "CO": (4.60, 4.75, "M", 0.0025),
    "O3": (9.45, 9.75, "N", 0.0050),
}

ALL_WINDOWS = (
    (0.758, 0.770),
    (1.10, 1.17),
    (1.50, 1.56),
    (2.00, 2.08),
    (2.28, 2.36),
    (3.28, 3.36),
    (4.20, 4.35),
    (4.45, 4.55),
    (4.60, 4.75),
    (9.45, 9.75),
)
ALL_SPECIES = ("H2O", "O2", "CO2", "CH4", "N2O", "CO", "O3")


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values, coordinates))


def _ensure_inputs() -> AERCatalogArtifact:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, dataset_id in XSHOOTER_DATASETS.values():
        path = DATA_DIR / filename
        if path.exists():
            continue
        url = f"https://dataportal.eso.org/dataPortal/file/{dataset_id}"
        urllib.request.urlretrieve(url, path)

    UVES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    uves_path = UVES_DATA_DIR / UVES_DATASET[0]
    if not uves_path.exists():
        url = f"https://dataportal.eso.org/dataPortal/file/{UVES_DATASET[1]}"
        urllib.request.urlretrieve(url, uves_path)

    for relative in HIRES_ORDER_FILES:
        path = HIRES_DATA_DIR / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        query = urllib.parse.urlencode(
            {
                "instrument": "HIRES",
                "koaid": HIRES_KOAID,
                "filehand": f"{HIRES_LEVEL1_PREFIX}/{relative}",
            }
        )
        url = f"https://koa.ipac.caltech.edu/cgi-bin/KoaAPI/nph-dnloadL1data?{query}"
        urllib.request.urlretrieve(url, path)

    KPF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    kpf_path = KPF_DATA_DIR / KPF_FILENAME
    if not kpf_path.exists():
        query = urllib.parse.urlencode(
            {
                "instrument": "KPF",
                "koaid": KPF_KOAID,
                "filehand": KPF_FILEHANDLE,
            }
        )
        url = f"https://koa.ipac.caltech.edu/cgi-bin/KoaAPI/nph-dnloadL1data?{query}"
        urllib.request.urlretrieve(url, kpf_path)

    aer_catalog = install_aer_catalog(reuse_molecfit=False)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranges = tuple(
        (
            max(0.0, 1.0e4 / upper - CACHE_WAVENUMBER_MARGIN_CM),
            1.0e4 / lower + CACHE_WAVENUMBER_MARGIN_CM,
        )
        for lower, upper in ALL_WINDOWS
    )
    cache_manifest = {
        "layout_version": CACHE_LAYOUT_VERSION,
        "catalog_version": aer_catalog.manifest["catalog_version"],
        "catalog_sha256": aer_catalog.manifest["catalog_sha256"],
        "catalog_source_page": aer_catalog.manifest["source_page"],
        "source_archive_sha256": aer_catalog.manifest["source_archive_sha256"],
        "wavenumber_margin_cm": CACHE_WAVENUMBER_MARGIN_CM,
        "wavenumber_ranges_cm": [list(values) for values in ranges],
        "species": list(ALL_SPECIES),
    }
    if CACHE_PATH.exists() and CACHE_MANIFEST_PATH.exists():
        try:
            existing_manifest = json.loads(CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = None
        if existing_manifest == cache_manifest:
            return aer_catalog
    lines = LineList.from_aer_line_file(
        aer_catalog.catalog_path,
        wavenumber_ranges=ranges,
        species=ALL_SPECIES,
        extra_broadener_dir=aer_catalog.extra_broadener_dir,
        # AER 3.9 is grouped by molecule rather than globally by wavenumber.
        assume_sorted=False,
    )
    temporary_cache = CACHE_PATH.with_name(f".{CACHE_PATH.name}.tmp")
    lines.write(temporary_cache, format="fits")
    temporary_cache.replace(CACHE_PATH)
    CACHE_MANIFEST_PATH.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aer_catalog


def _strong_local_lines(
    all_lines: LineList,
    species: str,
    lower: float,
    upper: float,
    half_width: float,
    *,
    max_lines: int = 500,
) -> tuple[LineList, float]:
    band = all_lines.select_species((species,)).select_range(lower, upper)
    center = float(band.wavelength[np.nanargmax(band.strength)])
    local = band.select_range(center - half_width, center + half_width, margin=half_width)
    if local.wavelength.size > max_lines:
        indices = np.argpartition(local.strength, -max_lines)[-max_lines:]
        keep = np.zeros(local.wavelength.size, dtype=bool)
        keep[indices] = True
        local = local.select(keep)
    return local, center


def _synthetic_recovery_checks(
    all_lines: LineList,
    rng: np.random.Generator,
) -> tuple[list[ValidationCheck], list[dict[str, object]]]:
    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []
    atmosphere = AtmosphereProfile.standard_midlatitude(airmass=1.0, n_layers=24)
    for species, (lower, upper, band, half_width) in SYNTHETIC_BANDS.items():
        lines, center = _strong_local_lines(
            all_lines,
            species,
            lower,
            upper,
            half_width,
        )
        wavelength = np.linspace(center - half_width, center + half_width, 600)
        names, basis = physical_optical_depth_basis(
            wavelength,
            lines,
            atmosphere,
            PhysicalModelConfig(line_wing_mode="lblrtm_panel"),
            species=(species,),
        )
        max_tau = float(np.nanmax(np.maximum(basis[0], 0.0)))
        if not np.isfinite(max_tau) or max_tau <= 0:
            checks.append(
                ValidationCheck(
                    "synthetic_recovery",
                    species,
                    "FAIL",
                    details="No positive optical depth in selected physical line window",
                )
            )
            continue
        true_scale = float(np.clip(-np.log(0.55) / max_tau, 0.02, 20.0))
        lsf_sigma = 1.1
        true_transmission = transmission_from_basis(
            names,
            basis,
            species_scales={species: true_scale},
            lsf_sigma_pixels=lsf_sigma,
            wavelength_micron=wavelength,
        )
        x = 2.0 * (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
        continuum = 1.0 + 0.025 * x
        noise_sigma = 0.002
        flux = continuum * true_transmission + rng.normal(0.0, noise_sigma, wavelength.size)
        result = fit_tellurics(
            Spectrum(
                wavelength=wavelength,
                flux=flux,
                uncertainty=np.full(wavelength.shape, noise_sigma),
            ),
            line_list=lines,
            config=FitConfig(
                atmosphere=atmosphere,
                species=(species,),
                continuum_order=1,
                solve_continuum_linear=True,
                lsf_sigma_pixels=lsf_sigma,
                line_wing_mode="lblrtm_panel",
                estimate_uncertainties=True,
            ),
        )
        fitted_scale = float(result.species_scales[species])
        scale_error = float(result.species_scale_uncertainties.get(species, np.nan))
        relative_error = abs(fitted_scale - true_scale) / true_scale
        z_score = (
            (fitted_scale - true_scale) / scale_error
            if np.isfinite(scale_error) and scale_error > 0
            else np.nan
        )
        reliable = true_transmission > 0.2
        normalized_corrected = result.corrected.flux / continuum
        corrected_rms = float(
            np.sqrt(np.nanmean((normalized_corrected[reliable] - 1.0) ** 2))
        )
        expected_noise = float(
            np.nanmedian(noise_sigma / true_transmission[reliable])
        )
        passed = (
            result.success
            and relative_error <= 0.03
            and corrected_rms <= 1.5 * expected_noise
            and np.isfinite(z_score)
            and abs(z_score) <= 3.0
        )
        checks.append(
            ValidationCheck(
                "synthetic_recovery",
                f"{species} ({band}-band)",
                "PASS" if passed else "FAIL",
                value=relative_error,
                threshold="relative scale error <= 0.03; |z| <= 3; corrected RMS <= 1.5x propagated noise",
                details=(
                    f"true={true_scale:.6g}, fitted={fitted_scale:.6g}, "
                    f"stderr={scale_error:.3g}, z={z_score:.3f}, "
                    f"corrected_rms={corrected_rms:.4g}, lines={lines.wavelength.size}"
                ),
            )
        )
        rows.append(
            {
                "group": "synthetic_recovery",
                "case": species,
                "band": band,
                "true_scale": true_scale,
                "fitted_scale": fitted_scale,
                "scale_stderr": scale_error,
                "relative_error": relative_error,
                "z_score": z_score,
                "corrected_rms": corrected_rms,
                "expected_noise": expected_noise,
                "n_lines": lines.wavelength.size,
            }
        )
    return checks, rows


def _uncertainty_coverage_check(
    rng: np.random.Generator,
    *,
    n_trials: int = 30,
) -> tuple[ValidationCheck, dict[str, object]]:
    wavelength = np.linspace(2.31, 2.36, 300)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    true_scale = 1.4
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": true_scale}),
    )
    noise = 0.01
    z_scores = []
    for _ in range(n_trials):
        spectrum = Spectrum(
            wavelength=wavelength,
            flux=transmission + rng.normal(0.0, noise, wavelength.size),
            uncertainty=np.full(wavelength.shape, noise),
        )
        result = fit_tellurics(
            spectrum,
            line_list=line_list,
            config=FitConfig(
                species=("H2O",),
                continuum_order=0,
                estimate_uncertainties=True,
            ),
        )
        error = result.species_scale_uncertainties["H2O"]
        z_scores.append((result.species_scales["H2O"] - true_scale) / error)
    z_scores = np.asarray(z_scores)
    coverage = float(np.mean(np.abs(z_scores) <= 1.0))
    z_mean = float(np.mean(z_scores))
    z_std = float(np.std(z_scores, ddof=1))
    passed = 0.50 <= coverage <= 0.82 and abs(z_mean) <= 0.5 and 0.7 <= z_std <= 1.3
    return (
        ValidationCheck(
            "uncertainty",
            "Monte Carlo 68% scale coverage",
            "PASS" if passed else "FAIL",
            value=coverage,
            threshold="0.50 <= coverage <= 0.82, |mean z| <= 0.5, 0.7 <= std(z) <= 1.3",
            details=f"trials={n_trials}, mean_z={z_mean:.3f}, std_z={z_std:.3f}",
        ),
        {
            "group": "uncertainty",
            "case": "H2O_monte_carlo",
            "coverage": coverage,
            "z_mean": z_mean,
            "z_std": z_std,
            "n_trials": n_trials,
        },
    )


def _model_systematics_check(
    rng: np.random.Generator,
) -> tuple[ValidationCheck, dict[str, object]]:
    wavelength = np.linspace(2.31, 2.36, 420)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    noise = 0.002
    true_transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": 1.35}, lsf_sigma_pixels=1.0),
    )
    spectrum = Spectrum(
        wavelength=wavelength,
        flux=true_transmission + rng.normal(0.0, noise, wavelength.size),
        uncertainty=np.full(wavelength.shape, noise),
    )
    baseline_config = FitConfig(
        species=("H2O",),
        continuum_order=0,
        lsf_sigma_pixels=0.8,
        estimate_uncertainties=True,
    )
    result = fit_tellurics_with_systematics(
        spectrum,
        line_list,
        baseline_config,
        {
            "narrower_lsf": replace(baseline_config, lsf_sigma_pixels=0.6),
            "broader_lsf": replace(baseline_config, lsf_sigma_pixels=1.2),
        },
    )
    finite = (
        np.isfinite(result.transmission_systematic_uncertainty)
        & np.isfinite(result.combined_transmission_uncertainty)
    )
    baseline_statistical = result.baseline.transmission_uncertainty
    propagated = (
        baseline_statistical is not None
        and np.all(
            result.combined_transmission_uncertainty[finite]
            >= np.asarray(baseline_statistical)[finite]
        )
        and result.corrected.uncertainty is not None
        and bool(result.corrected.meta.get("model_systematic_uncertainty_propagated"))
    )
    p95 = float(result.metrics["transmission_systematic_rms_p95"])
    passed = (
        result.baseline.success
        and all(variant.success for variant in result.variants.values())
        and np.any(finite)
        and np.isfinite(p95)
        and p95 > 0
        and propagated
    )
    return (
        ValidationCheck(
            "uncertainty",
            "refitted model-systematics ensemble propagation",
            "PASS" if passed else "FAIL",
            value=p95,
            threshold="named model variants refit successfully and add finite nonzero uncertainty in quadrature",
            details=(
                f"variants={list(result.variants)}, "
                f"p95_transmission_systematic={p95:.5g}, propagated={propagated}"
            ),
        ),
        {
            "group": "uncertainty",
            "case": "model_systematics_lsf_ensemble",
            **dict(result.metrics),
        },
    )


def _shared_uncertainty_and_output_checks(
    rng: np.random.Generator,
    *,
    n_trials: int = 24,
) -> tuple[list[ValidationCheck], dict[str, object]]:
    wavelength = np.linspace(2.31, 2.36, 250)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    true_scale = 1.4
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(species_scales={"H2O": true_scale}),
    )
    noise = 0.008
    z_scores = []
    full_rank = True
    propagated = True
    final_result = None
    for _ in range(n_trials):
        spectra = tuple(
            Spectrum(
                wavelength=wavelength,
                flux=continuum * transmission + rng.normal(0.0, noise, wavelength.size),
                uncertainty=np.full(wavelength.shape, noise),
            )
            for continuum in (0.9, 1.1)
        )
        final_result = fit_telluric_segments(
            spectra,
            line_list=line_list,
            config=FitConfig(
                species=("H2O",),
                continuum_order=0,
                estimate_uncertainties=True,
            ),
        )
        scale_error = final_result.species_scale_uncertainties.get("H2O", np.nan)
        z_scores.append((final_result.species_scales["H2O"] - true_scale) / scale_error)
        full_rank &= (
            final_result.parameter_covariance is not None
            and final_result.covariance_rank == len(final_result.parameter_names)
        )
        for segment in final_result.segment_results:
            propagated &= (
                segment.transmission_uncertainty is not None
                and segment.corrected.uncertainty is not None
                and np.all(np.isfinite(segment.transmission_uncertainty))
                and np.all(np.isfinite(segment.corrected.uncertainty[segment.corrected.valid]))
            )

    z_scores = np.asarray(z_scores, dtype=float)
    coverage = float(np.mean(np.abs(z_scores) <= 1.0))
    z_mean = float(np.mean(z_scores))
    z_std = float(np.std(z_scores, ddof=1))
    uncertainty_pass = (
        full_rank
        and propagated
        and 0.50 <= coverage <= 0.82
        and abs(z_mean) <= 0.5
        and 0.7 <= z_std <= 1.3
    )

    assert final_result is not None
    with tempfile.TemporaryDirectory(prefix="pymolfit_product_audit_") as temporary:
        product_path = Path(temporary) / "segment.ecsv"
        final_result.segment_results[0].write(product_path)
        product = Table.read(product_path, format="ascii.ecsv")
    required_columns = {
        "wavelength",
        "flux",
        "model_flux",
        "continuum",
        "transmission",
        "transmission_uncertainty",
        "corrected_flux",
        "uncertainty",
        "corrected_uncertainty",
        "input_mask",
        "fit_mask",
        "corrected_mask",
    }
    try:
        product_provenance = json.loads(str(product.meta["provenance_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        product_provenance = {}
    expected_fit_pixel_counts = [
        int(np.count_nonzero(segment.fit_mask)) for segment in final_result.segment_results
    ]
    provenance_pass = (
        product_provenance.get("schema_version") == 1
        and len(str(product_provenance.get("line_list_sha256", ""))) == 64
        and len(str(product_provenance.get("selected_line_list_sha256", ""))) == 64
        and len(str(product_provenance.get("fit_config_sha256", ""))) == 64
        and product_provenance.get("fit_pixel_counts")
        == expected_fit_pixel_counts
        and expected_fit_pixel_counts[0] == int(np.count_nonzero(product["fit_mask"]))
        and "atmosphere_metadata" in product_provenance
        and "fit_config" in product_provenance
    )
    output_pass = (
        required_columns.issubset(product.colnames)
        and bool(product.meta.get("covariance_full_rank", False))
        and product.meta.get("wavelength_medium") == "vacuum"
        and "species_scales" in product.meta
        and "parameter_standard_errors" in product.meta
        and "parameter_bound_status" in product.meta
        and provenance_pass
    )

    return (
        [
            ValidationCheck(
                "uncertainty",
                "shared multi-segment 68% scale coverage",
                "PASS" if uncertainty_pass else "FAIL",
                value=coverage,
                threshold=(
                    "0.50 <= coverage <= 0.82, |mean z| <= 0.5, "
                    "0.7 <= std(z) <= 1.3, full-rank covariance"
                ),
                details=(
                    f"trials={n_trials}, mean_z={z_mean:.3f}, std_z={z_std:.3f}, "
                    f"full_rank={full_rank}, propagated={propagated}"
                ),
            ),
            ValidationCheck(
                "scientific_output",
                "fit-product masks, uncertainties, and provenance round-trip",
                "PASS" if output_pass else "FAIL",
                threshold="required science columns and fit provenance survive ECSV round-trip",
                details=f"columns={sorted(product.colnames)}, metadata_keys={sorted(product.meta)}",
            ),
        ],
        {
            "group": "uncertainty",
            "case": "shared_multi_segment_H2O",
            "coverage": coverage,
            "z_mean": z_mean,
            "z_std": z_std,
            "n_trials": n_trials,
            "full_rank": full_rank,
            "propagated": propagated,
            "output_roundtrip": output_pass,
        },
    )


def _intrinsic_line_preservation_check(
    rng: np.random.Generator,
) -> tuple[ValidationCheck, dict[str, object]]:
    wavelength = np.linspace(2.31, 2.36, 1200)
    line_list = LineList.demo_near_ir()
    true_transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(
            species_scales={"H2O": 1.5, "CO2": 0.8, "CH4": 1.2},
            lsf_sigma_pixels=1.2,
        ),
    )
    x = 2.0 * (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    continuum = 1.03 + 0.025 * x
    line_center = 2.345
    line_sigma = 1.8e-4
    line_depth = 0.22
    intrinsic = 1.0 - line_depth * np.exp(
        -0.5 * ((wavelength - line_center) / line_sigma) ** 2
    )
    noise = 0.0015
    flux = continuum * intrinsic * true_transmission + rng.normal(0.0, noise, wavelength.size)
    fit_mask = np.abs(wavelength - line_center) > 5.0 * line_sigma
    result = fit_tellurics(
        Spectrum(wavelength, flux, np.full(wavelength.shape, noise)),
        line_list=line_list,
        config=FitConfig(
            continuum_order=1,
            solve_continuum_linear=True,
            lsf_sigma_pixels=1.2,
        ),
        fit_mask=fit_mask,
    )
    recovered = result.corrected.flux / result.continuum
    line_region = np.abs(wavelength - line_center) < 5.0 * line_sigma
    true_equivalent_width = float(
        _trapezoid(1.0 - intrinsic[line_region], wavelength[line_region])
    )
    recovered_equivalent_width = float(
        _trapezoid(1.0 - recovered[line_region], wavelength[line_region])
    )
    equivalent_width_error = abs(recovered_equivalent_width - true_equivalent_width) / true_equivalent_width
    recovered_depth = 1.0 - float(np.nanmin(recovered[line_region]))
    depth_error = abs(recovered_depth - line_depth)
    reliable_outside = fit_mask & (result.transmission > 0.2)
    outside_rms = float(
        np.sqrt(np.mean((recovered[reliable_outside] - intrinsic[reliable_outside]) ** 2))
    )
    passed = (
        result.success
        and equivalent_width_error <= 0.02
        and depth_error <= 0.01
        and outside_rms <= 0.004
    )
    return (
        ValidationCheck(
            "line_preservation",
            "masked intrinsic absorption-line preservation",
            "PASS" if passed else "FAIL",
            value=equivalent_width_error,
            threshold="equivalent-width error <= 2%, depth error <= 0.01, outside RMS <= 0.004",
            details=(
                f"depth_error={depth_error:.5g}, outside_rms={outside_rms:.5g}, "
                f"true_EW={true_equivalent_width:.6g}, recovered_EW={recovered_equivalent_width:.6g}"
            ),
        ),
        {
            "group": "line_preservation",
            "case": "masked_intrinsic_absorption",
            "equivalent_width_relative_error": equivalent_width_error,
            "depth_error": depth_error,
            "outside_rms": outside_rms,
        },
    )


def _instrument_parameter_recovery_check(
    rng: np.random.Generator,
) -> tuple[ValidationCheck, dict[str, object]]:
    wavelength = np.linspace(2.31, 2.36, 900)
    line_list = LineList.demo_near_ir().select_species(("H2O",))
    true_scale = 1.3
    true_shift = 2.0e-5
    true_lsf_sigma = 1.4
    transmission = transmission_model(
        wavelength,
        line_list,
        ModelConfig(
            species_scales={"H2O": true_scale},
            lsf_sigma_pixels=true_lsf_sigma,
        ),
    )
    transmission = np.interp(
        wavelength - true_shift,
        wavelength,
        transmission,
        left=1.0,
        right=1.0,
    )
    x = (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    noise = 0.002
    flux = (1.0 + 0.01 * x) * transmission + rng.normal(0.0, noise, wavelength.size)
    result = fit_tellurics(
        Spectrum(wavelength, flux, np.full(wavelength.shape, noise)),
        line_list=line_list,
        config=FitConfig(
            species=("H2O",),
            continuum_order=1,
            solve_continuum_linear=True,
            fit_wavelength_shift=True,
            wavelength_shift_bounds=(-1.0e-4, 1.0e-4),
            fit_lsf_sigma=True,
            lsf_sigma_pixels=1.0,
            lsf_sigma_bounds=(0.1, 3.0),
            estimate_uncertainties=True,
        ),
    )
    scale_error = abs(result.species_scales["H2O"] - true_scale) / true_scale
    shift_error_pixels = abs(result.wavelength_shift - true_shift) / np.median(np.diff(wavelength))
    lsf_error = abs(result.lsf_sigma_pixels - true_lsf_sigma)
    lsf_relative_error = lsf_error / true_lsf_sigma
    passed = (
        scale_error <= 0.02
        and shift_error_pixels <= 0.1
        and lsf_relative_error <= 0.15
    )
    return (
        ValidationCheck(
            "synthetic_recovery",
            "joint molecular/wavelength/LSF recovery",
            "PASS" if passed else "FAIL",
            value=scale_error,
            threshold="scale error <= 2%, shift error <= 0.1 pixel, LSF sigma error <= 15%",
            details=(
                f"scale_error={scale_error:.4g}, shift_error_pixels={shift_error_pixels:.4g}, "
                f"lsf_sigma_error={lsf_error:.4g} ({lsf_relative_error:.2%})"
            ),
        ),
        {
            "group": "synthetic_recovery",
            "case": "joint_shift_lsf",
            "relative_error": scale_error,
            "shift_error_pixels": shift_error_pixels,
            "lsf_sigma_error_pixels": lsf_error,
            "lsf_relative_error": lsf_relative_error,
        },
    )


def _atmosphere_and_convergence_checks(
    all_lines: LineList,
) -> tuple[list[ValidationCheck], list[dict[str, object]]]:
    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []
    lines, center = _strong_local_lines(all_lines, "H2O", 1.10, 1.17, 0.0012, max_lines=300)
    wavelength = np.linspace(center - 0.0012, center + 0.0012, 500)

    airmasses = (1.0, 1.5, 2.0)
    airmass_bases = []
    for airmass in airmasses:
        atmosphere = AtmosphereProfile.standard_midlatitude(airmass=airmass, n_layers=40)
        names, basis = physical_optical_depth_basis(
            wavelength,
            lines,
            atmosphere,
            PhysicalModelConfig(line_wing_mode="lblrtm_panel"),
            species=("H2O",),
        )
        airmass_bases.append((names, basis))
    response_scale = -np.log(0.55) / float(np.nanmax(airmass_bases[0][1][0]))
    minima = [
        float(
            np.nanmin(
                transmission_from_basis(
                    names,
                    basis,
                    species_scales={"H2O": response_scale},
                )
            )
        )
        for names, basis in airmass_bases
    ]
    monotonic = minima[0] > minima[1] > minima[2]
    checks.append(
        ValidationCheck(
            "atmosphere",
            "airmass response",
            "PASS" if monotonic else "FAIL",
            value=minima[-1],
            threshold="minimum transmission decreases monotonically from airmass 1.0 to 2.0",
            details=f"minima={minima}",
        )
    )
    rows.append({"group": "atmosphere", "case": "airmass", "minima": str(minima)})

    layer_transmissions = []
    for n_layers in (40, 80, 160):
        atmosphere = AtmosphereProfile.standard_midlatitude(airmass=1.3, n_layers=n_layers)
        names, basis = physical_optical_depth_basis(
            wavelength,
            lines,
            atmosphere,
            PhysicalModelConfig(line_wing_mode="lblrtm_panel", chunk_size=512),
            species=("H2O",),
        )
        layer_transmissions.append(
            transmission_from_basis(names, basis, species_scales={"H2O": 1.0})
        )
    rms_40_80 = float(
        np.sqrt(np.mean((layer_transmissions[0] - layer_transmissions[1]) ** 2))
    )
    rms_80_160 = float(
        np.sqrt(np.mean((layer_transmissions[1] - layer_transmissions[2]) ** 2))
    )
    checks.append(
        ValidationCheck(
            "convergence",
            "atmospheric layer resolution",
            "PASS" if rms_80_160 <= 0.002 else "FAIL",
            value=rms_80_160,
            threshold="RMS transmission difference (80 vs 160 layers) <= 0.002",
            details=f"40_vs_80={rms_40_80:.5g}",
        )
    )
    rows.append(
        {
            "group": "convergence",
            "case": "atmosphere_layers_80_160",
            "rms_difference": rms_80_160,
            "rms_40_80": rms_40_80,
        }
    )

    chunk_transmissions = []
    atmosphere = AtmosphereProfile.standard_midlatitude(airmass=1.3, n_layers=40)
    for chunk_size in (64, 512, 0):
        names, basis = physical_optical_depth_basis(
            wavelength,
            lines,
            atmosphere,
            PhysicalModelConfig(line_wing_mode="lblrtm_panel", chunk_size=chunk_size),
            species=("H2O",),
        )
        chunk_transmissions.append(
            transmission_from_basis(names, basis, species_scales={"H2O": 1.0})
        )
    chunk_max = float(
        max(
            np.max(np.abs(chunk_transmissions[0] - chunk_transmissions[1])),
            np.max(np.abs(chunk_transmissions[1] - chunk_transmissions[2])),
        )
    )
    checks.append(
        ValidationCheck(
            "convergence",
            "line chunk invariance",
            "PASS" if chunk_max <= 1.0e-12 else "FAIL",
            value=chunk_max,
            threshold="maximum transmission difference <= 1e-12",
        )
    )
    rows.append(
        {
            "group": "convergence",
            "case": "chunk_invariance",
            "max_difference": chunk_max,
        }
    )

    dry = AtmosphereProfile.standard_midlatitude(airmass=1.3, n_layers=80).with_pwv_mm(1.0)
    humid = AtmosphereProfile.standard_midlatitude(airmass=1.3, n_layers=80).with_pwv_mm(5.0)
    humidity_bases = []
    for atmosphere in (dry, humid):
        names, basis = physical_optical_depth_basis(
            wavelength,
            lines,
            atmosphere,
            PhysicalModelConfig(line_wing_mode="lblrtm_panel"),
            species=("H2O",),
        )
        humidity_bases.append((names, basis))
    humidity_scale = -np.log(0.70) / float(np.nanmax(humidity_bases[0][1][0]))
    humidity_minima = [
        float(
            np.nanmin(
                transmission_from_basis(
                    names,
                    basis,
                    species_scales={"H2O": humidity_scale},
                )
            )
        )
        for names, basis in humidity_bases
    ]
    checks.append(
        ValidationCheck(
            "atmosphere",
            "dry versus humid water column response",
            "PASS" if humidity_minima[0] > humidity_minima[1] else "FAIL",
            threshold="5 mm PWV produces deeper H2O absorption than 1 mm PWV",
            details=f"minima={humidity_minima}",
        )
    )
    rows.append({"group": "atmosphere", "case": "pwv_1_vs_5_mm", "minima": str(humidity_minima)})
    return checks, rows


def _failure_mode_checks() -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    spectrum = Spectrum(
        wavelength=np.array([1.0, 1.1, 1.2]),
        flux=np.array([1.0, 0.5, 0.1]),
        uncertainty=np.array([0.01, 0.01, 0.01]),
    )
    transmission = np.array([1.0, 0.5, 0.01])
    corrected = correct_spectrum(
        spectrum,
        transmission,
        transmission_uncertainty=np.array([0.001, 0.01, 0.01]),
        min_transmission=0.03,
    )
    saturated_ok = (
        np.isfinite(corrected.flux[0])
        and np.isfinite(corrected.flux[1])
        and np.isnan(corrected.flux[2])
        and corrected.uncertainty is not None
    )
    checks.append(
        ValidationCheck(
            "failure_modes",
            "saturated-line masking and uncertainty propagation",
            "PASS" if saturated_ok else "FAIL",
            threshold="transmission < 0.03 is masked; surviving errors are finite",
        )
    )

    wavelength = np.linspace(2.31, 2.36, 300)
    lines = LineList.demo_near_ir().select_species(("H2O",))
    truth = transmission_model(
        wavelength,
        lines,
        ModelConfig(species_scales={"H2O": 1.3}),
    )
    rng = np.random.default_rng(711)
    scale_errors = []
    for noise in (0.005, 0.10):
        flux = truth + rng.normal(0.0, noise, wavelength.size)
        flux[90:120] = np.nan
        mask = np.ones(wavelength.shape, dtype=bool)
        mask[180:200] = False
        result = fit_tellurics(
            Spectrum(
                wavelength=wavelength,
                flux=flux,
                uncertainty=np.full(wavelength.shape, noise),
                mask=mask,
            ),
            line_list=lines,
            config=FitConfig(
                species=("H2O",),
                continuum_order=0,
                estimate_uncertainties=True,
            ),
        )
        scale_errors.append(result.species_scale_uncertainties["H2O"])
    low_snr_ok = np.isfinite(scale_errors).all() and scale_errors[1] > 5.0 * scale_errors[0]
    checks.append(
        ValidationCheck(
            "failure_modes",
            "low-S/N and data-gap behavior",
            "PASS" if low_snr_ok else "FAIL",
            value=float(scale_errors[1] / scale_errors[0]),
            threshold="fit remains finite and low-S/N scale error grows by >5x",
            details=f"high_SNR_error={scale_errors[0]:.4g}, low_SNR_error={scale_errors[1]:.4g}",
        )
    )

    degenerate_lines = LineList(
        wavelength=np.array([2.33, 2.33]),
        strength=np.array([0.03, 0.03]),
        sigma=np.array([2.0e-5, 2.0e-5]),
        gamma=np.array([1.0e-5, 1.0e-5]),
        species=np.array(["H2O", "CO2"]),
    )
    degenerate_wavelength = np.linspace(2.32, 2.34, 300)
    degenerate_flux = transmission_model(
        degenerate_wavelength,
        degenerate_lines,
        ModelConfig(species_scales={"H2O": 1.0, "CO2": 1.0}),
    )
    degenerate_result = fit_tellurics(
        Spectrum(
            degenerate_wavelength,
            degenerate_flux,
            np.full(degenerate_wavelength.shape, 0.003),
        ),
        line_list=degenerate_lines,
        config=FitConfig(continuum_order=0, estimate_uncertainties=True),
    )
    rank_deficiency_exposed = (
        degenerate_result.parameter_covariance is not None
        and degenerate_result.covariance_rank < len(degenerate_result.parameter_names)
        and np.all(np.isnan(degenerate_result.parameter_covariance))
        and degenerate_result.transmission_uncertainty is not None
        and np.all(np.isnan(degenerate_result.transmission_uncertainty))
    )
    checks.append(
        ValidationCheck(
            "failure_modes",
            "non-identifiable covariance reports no false precision",
            "PASS" if rank_deficiency_exposed else "FAIL",
            value=float(degenerate_result.covariance_rank),
            threshold="rank < parameter count and covariance/output uncertainty are NaN",
            details=(
                f"rank={degenerate_result.covariance_rank}, "
                f"parameters={len(degenerate_result.parameter_names)}"
            ),
        )
    )

    try:
        fallback_profile = AtmosphereProfile.from_mipas_gdas(
            observation_time="2022-01-02T05:17:35",
            observatory_altitude_m=2635.0,
            airmass=1.2,
            gdas_mode="average",
        )
        fallback_ok = (
            fallback_profile.metadata.get("gdas_source") == "average"
            and len(fallback_profile.layers) > 40
            and np.isfinite(fallback_profile.total_column_cm2("H2O"))
            and fallback_profile.total_column_cm2("H2O") > 0
        )
        fallback_details = (
            f"source={fallback_profile.metadata.get('gdas_source')}, "
            f"layers={len(fallback_profile.layers)}"
        )
    except Exception as exc:
        fallback_ok = False
        fallback_details = f"raised {type(exc).__name__}: {exc}"
    checks.append(
        ValidationCheck(
            "failure_modes",
            "offline GDAS-average atmosphere fallback",
            "PASS" if fallback_ok else "FAIL",
            threshold="packaged seasonal GDAS average yields a finite MIPAS/GDAS atmosphere",
            details=fallback_details,
        )
    )

    try:
        AtmosphereProfile.from_fits_header_mipas_gdas(
            {
                "OBSERVAT": "UNREGISTERED VALIDATION SITE",
                "DATE-OBS": "2025-01-01T00:00:00",
                "AIRMASS": 1.2,
            },
            gdas_mode="average",
        )
        unknown_site_rejected = False
        unknown_site_details = "constructor silently accepted unresolved observatory geometry"
    except ValueError as exc:
        unknown_site_rejected = "cannot resolve latitude_deg" in str(exc)
        unknown_site_details = str(exc)
    checks.append(
        ValidationCheck(
            "failure_modes",
            "unresolved observatory metadata fails closed",
            "PASS" if unknown_site_rejected else "FAIL",
            threshold="unknown site without coordinates is rejected unless fallback is explicitly enabled",
            details=unknown_site_details,
        )
    )
    return checks


def _write_xshooter_crop(case: BandCase, output: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    source = DATA_DIR / XSHOOTER_DATASETS[case.arm][0]
    with fits.open(source) as hdul:
        header = hdul[0].header.copy()
        data = hdul[1].data
        wavelength = np.asarray(data["WAVE"][0], dtype=float) * 1.0e-3
        flux = np.asarray(data["FLUX_REDUCED"][0], dtype=float)
        uncertainty = np.asarray(data["ERR_REDUCED"][0], dtype=float)
        quality = np.asarray(data["QUAL"][0], dtype=int)
    keep = (
        (wavelength >= case.wavelength_min)
        & (wavelength <= case.wavelength_max)
        & np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
        & (quality == 0)
    )
    wavelength = wavelength[keep]
    flux = flux[keep]
    uncertainty = uncertainty[keep]
    slit_width = float(header.get("ESO QC SLIT WIDTH", 0.9))
    header["HIERARCH ESO INS SLIT1 WID"] = (slit_width, "slit width for Molecfit")
    columns = [
        fits.Column(name="lambda", array=wavelength, format="D", unit="um"),
        fits.Column(name="flux", array=flux, format="D", unit="adu"),
        fits.Column(name="dflux", array=uncertainty, format="D", unit="adu"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns, name="SCIENCE")]
    ).writeto(output, overwrite=True)
    return wavelength, flux, uncertainty, header


def _run_molecfit_xshooter(
    crop: Path,
    case: BandCase,
    output_dir: Path,
    gaussian_fwhm_pixels: float,
) -> tuple[Path | None, float]:
    existing_model = output_dir / "BEST_FIT_MODEL.fits"
    existing_parameters = output_dir / "BEST_FIT_PARAMETERS.fits"
    manifest_path = output_dir / "molecfit_validation_manifest.json"
    recipe_options = {
        "LIST_MOLEC": list(case.species),
        "FIT_MOLEC": [1] * len(case.species),
        "REL_COL": [1] * len(case.species),
        "WAVE_INCLUDE": [case.wavelength_min, case.wavelength_max],
        "WAVELENGTH_FRAME": "AIR",
        "FIT_WLC": 1,
        "WLC_N": 0,
        "WLC_CONST": 0.0,
        "FIT_RES_BOX": False,
        "RES_BOX": 1.0,
        "FIT_RES_GAUSS": True,
        "RES_GAUSS": float(gaussian_fwhm_pixels),
        "FIT_RES_LORENTZ": False,
        "RES_LORENTZ": 0.0,
        "FIT_CONTINUUM": 1,
        "CONTINUUM_N": 2,
        "GDAS_PROFILE": "auto",
        "FTOL": 1.0e-10,
        "XTOL": 1.0e-10,
    }
    cache_payload = {
        "schema": 3,
        "input_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
        "recipe": "molecfit_model",
        "line_catalog_version": AER_CATALOG_VERSION,
        "line_catalog_sha256": AER_CATALOG_SHA256,
        "recipe_options": recipe_options,
    }
    signature = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if existing_model.exists() and existing_parameters.exists() and manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("signature") == signature:
            elapsed = float(cached.get("elapsed_seconds", np.nan))
            return existing_model, elapsed
    if not MOLECFIT_ESOREX.exists():
        return None, np.nan
    with tempfile.TemporaryDirectory(prefix="gmf_science_ready_") as temporary:
        stage = Path(temporary)
        staged_input = stage / "input.fits"
        staged_output = stage / "out"
        staged_tmp = stage / "tmp"
        staged_output.mkdir()
        staged_tmp.mkdir()
        molecfit_data, _ = stage_aer_molecfit_data(
            MOLECFIT_DATA_ROOT,
            stage / "molecfit-data",
        )
        shutil.copy2(crop, staged_input)
        sof = stage / "model.sof"
        sof.write_text(f"{staged_input} SCIENCE\n", encoding="utf-8")
        molecule_flags = ",".join("1" for _ in case.species)
        command = [
            str(MOLECFIT_ESOREX),
            f"--output-dir={staged_output}",
            "molecfit_model",
            f"--TELLURICCORR_DATA_PATH={molecfit_data}",
            f"--LNFL_LINE_DB={AER_CATALOG_FILENAME}",
            f"--LIST_MOLEC={','.join(case.species)}",
            f"--FIT_MOLEC={molecule_flags}",
            f"--REL_COL={molecule_flags}",
            f"--WAVE_INCLUDE={case.wavelength_min},{case.wavelength_max}",
            "--WAVELENGTH_FRAME=AIR",
            "--COLUMN_LAMBDA=lambda",
            "--COLUMN_FLUX=flux",
            "--COLUMN_DFLUX=dflux",
            "--WLG_TO_MICRON=1.0",
            "--FIT_WLC=1",
            "--WLC_N=0",
            "--WLC_CONST=0.0",
            "--FIT_RES_BOX=FALSE",
            "--RES_BOX=1.0",
            "--FIT_RES_GAUSS=TRUE",
            f"--RES_GAUSS={gaussian_fwhm_pixels}",
            "--FIT_RES_LORENTZ=FALSE",
            "--RES_LORENTZ=0.0",
            "--FIT_CONTINUUM=1",
            "--CONTINUUM_N=2",
            "--GDAS_PROFILE=auto",
            "--UTC_KEYWORD=UTC",
            "--MIRROR_TEMPERATURE_KEYWORD=ESO TEL TH M1 TEMP",
            "--SLIT_WIDTH_KEYWORD=ESO INS SLIT1 WID",
            "--FTOL=1e-10",
            "--XTOL=1e-10",
            f"--TMP_PATH={staged_tmp}",
            str(sof),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=stage,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        (output_dir / "molecfit.log").write_text(
            completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return None, elapsed
        for product in staged_output.iterdir():
            if product.is_file():
                shutil.copy2(product, output_dir / product.name)
    manifest_path.write_text(
        json.dumps(
            {
                **cache_payload,
                "signature": signature,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir / "BEST_FIT_MODEL.fits", elapsed


def _molecfit_parameters(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with fits.open(path) as hdul:
        data = hdul[1].data
        parameters = {}
        for name, value in zip(data["parameter"], data["value"], strict=True):
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore")
            normalized = str(name).strip().strip("\x00")
            if normalized:
                parameters[normalized] = float(value)
    return parameters


def _normalise(values: np.ndarray) -> np.ndarray:
    scale = np.nanmedian(values)
    return values / scale if np.isfinite(scale) and scale != 0 else values


def _compare_gdas_profiles(
    pymolfit_path: Path,
    molecfit_path: Path,
) -> tuple[bool, float, str]:
    columns = ("press", "height", "temp", "relhum")
    pymolfit = Table.read(pymolfit_path, hdu=1)
    molecfit = Table.read(molecfit_path, hdu=1)
    if len(pymolfit) != len(molecfit):
        return False, np.inf, f"level counts differ ({len(pymolfit)} vs {len(molecfit)})"
    maxima = {}
    for column in columns:
        delta = np.asarray(pymolfit[column], dtype=float) - np.asarray(
            molecfit[column], dtype=float
        )
        maxima[column] = float(np.nanmax(np.abs(delta)))
    maximum = max(maxima.values())
    return maximum <= 1.0e-8, maximum, f"column maxima={maxima}"


def _compare_combined_atmosphere_profiles(
    atmosphere: AtmosphereProfile,
    gdas_path: Path,
    molecfit_path: Path,
) -> tuple[bool, float, str]:
    """Compare PyMolFit's merged MIPAS/GDAS levels to Molecfit's product."""

    metadata = dict(atmosphere.metadata)
    altitude_m = float(metadata["observatory_altitude_m"])
    levels_m = atmosphere_impl._molecfit_fixed_height_levels_m(
        observatory_altitude_m=altitude_m,
        top_altitude_m=120_000.0,
    )
    mipas = atmosphere_impl._load_mipas_profile(str(metadata["mipas_profile"]))
    gdas = atmosphere_impl._load_gdas_profile(
        gdas_path,
        None,
        latitude_deg=float(metadata["latitude_deg"]),
        longitude_deg=float(metadata["longitude_deg"]),
        gdas_mode="auto",
        gdas_cache_dir=None,
        gdas_download_timeout_s=1.0,
    )
    pressure_hpa, temperature_k, mixing_ratios = (
        atmosphere_impl._merge_mipas_gdas_fixed_levels(
            levels_m,
            mipas=mipas,
            gdas=gdas,
        )
    )
    pressure_hpa, temperature_k, mixing_ratios = (
        atmosphere_impl._adapt_profile_to_local_meteo(
            levels_m,
            pressure_hpa,
            temperature_k,
            mixing_ratios,
            observatory_altitude_m=altitude_m,
            meteo_mixing_height_m=atmosphere_impl.DEFAULT_METEO_MIXING_HEIGHT_M,
            pressure_at_observatory_atm=metadata.get("pressure_at_observatory_atm"),
            temperature_at_observatory_k=metadata.get("temperature_at_observatory_k"),
            relative_humidity_percent=metadata.get("relative_humidity_percent"),
        )
    )
    molecfit = Table.read(molecfit_path, hdu=1)
    if len(molecfit) != levels_m.size:
        return (
            False,
            np.inf,
            f"level counts differ ({levels_m.size} vs {len(molecfit)})",
        )

    expected = {
        "HGT": levels_m / 1_000.0,
        "PRE": pressure_hpa,
        "TEM": temperature_k,
    }
    for column in molecfit.colnames:
        upper = column.upper()
        if upper in expected or upper not in mixing_ratios:
            continue
        expected[upper] = np.asarray(mixing_ratios[upper], dtype=float) * 1.0e6

    maxima: dict[str, float] = {}
    for column, values in expected.items():
        if column not in molecfit.colnames:
            return False, np.inf, f"Molecfit profile is missing {column}"
        delta = np.asarray(values, dtype=float) - np.asarray(molecfit[column], dtype=float)
        maxima[column] = float(np.nanmax(np.abs(delta)))
    maximum = max(maxima.values())
    return maximum <= 1.0e-8, maximum, f"column maxima={maxima}"


def _run_xshooter_case(
    case: BandCase,
    all_lines: LineList,
    *,
    run_molecfit: bool,
) -> tuple[list[ValidationCheck], dict[str, object]]:
    from pymolfit import correct_file

    case_dir = OUTPUT_DIR / "xshooter" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    crop = case_dir / "input_air.fits"
    wavelength, flux, uncertainty, header = _write_xshooter_crop(case, crop)
    lines = all_lines.select_species(case.species).select_range(
        case.wavelength_min,
        case.wavelength_max,
        margin=0.01,
    )
    resolution = float(header["SPEC_RES"])
    pixel_spacing = float(np.nanmedian(np.diff(wavelength)))
    fwhm_pixels = float(np.nanmedian(wavelength) / resolution / pixel_spacing)
    gaussian_sigma = max(0.2, fwhm_pixels / 2.354820045)
    started = time.perf_counter()
    result = correct_file(
        crop,
        case_dir / "pymolfit_corrected.txt",
        input_format="fits",
        wavelength_col="lambda",
        flux_col="flux",
        uncertainty_col="dflux",
        wavelength_unit="micron",
        wavelength_medium="air",
        line_list=lines,
        physical=True,
        atmosphere_mode="mipas_gdas",
        mipas_profile="equ",
        gdas_mode="auto",
        gdas_download_timeout_s=30.0,
        continuum_order=2,
        solve_continuum_linear=True,
        lsf_box_width_pixels=1.0,
        lsf_sigma_pixels=gaussian_sigma,
        high_resolution_grid=True,
        high_resolution_oversampling=5.0,
        high_resolution_rebin_mode="molecfit_overlap",
        line_wing_mode="lblrtm_panel",
        n2_continuum=True,
        o2_continuum=True,
        fit_wavelength_shift=True,
        wavelength_shift_bounds=(-2.0e-4, 2.0e-4),
        fit_lsf_sigma=True,
        lsf_sigma_bounds=(0.0, 8.0),
        estimate_uncertainties=True,
        product_path=case_dir / "pymolfit_product.ecsv",
        plot_path=case_dir / "pymolfit_fit.png",
    )
    gen_seconds = time.perf_counter() - started
    atmosphere = AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        mipas_profile="equ",
        gdas_mode="auto",
        gdas_download_timeout_s=30.0,
        reference_wavenumber_cm=float(1.0e4 / np.nanmedian(wavelength)),
    )
    molecfit_model = None
    molecfit_seconds = np.nan
    if run_molecfit:
        molecfit_model, molecfit_seconds = _run_molecfit_xshooter(
            crop,
            case,
            case_dir,
            fwhm_pixels,
        )

    checks: list[ValidationCheck] = []
    raw_relative = flux / result.continuum
    reliable_gen = np.isfinite(result.corrected.flux) & (result.transmission > 0.2)
    corrected_relative = result.corrected.flux / result.continuum
    raw_scatter = float(np.nanstd(raw_relative[reliable_gen] - 1.0))
    corrected_scatter = float(np.nanstd(corrected_relative[reliable_gen] - 1.0))
    improvement = raw_scatter / corrected_scatter if corrected_scatter > 0 else np.inf
    checks.append(
        ValidationCheck(
            "real_correction",
            f"X-shooter {case.band} correction reduces telluric scatter",
            "PASS" if result.success and improvement >= 1.10 else "FAIL",
            value=improvement,
            threshold="raw/corrected normalized scatter >= 1.10",
            details=f"raw={raw_scatter:.4g}, corrected={corrected_scatter:.4g}",
        )
    )

    transmission_rms = np.nan
    telluric_rms = np.nan
    molecfit_best_chi2 = np.nan
    molecfit_reduced_chi2 = np.nan
    molecfit_raw_scatter = np.nan
    molecfit_corrected_scatter = np.nan
    molecfit_improvement = np.nan
    molecfit_transmission = np.full(wavelength.shape, np.nan)
    molecfit_corrected_relative = np.full(wavelength.shape, np.nan)
    if molecfit_model is not None and molecfit_model.exists():
        parameters = _molecfit_parameters(case_dir / "BEST_FIT_PARAMETERS.fits")
        molecfit_best_chi2 = parameters.get("best_chi2", np.nan)
        molecfit_reduced_chi2 = parameters.get("reduced_chi2", np.nan)
        with fits.open(molecfit_model) as hdul:
            model_data = hdul[1].data
            n = min(wavelength.size, len(model_data))
            molecfit_transmission[:n] = np.asarray(model_data["mtrans"][:n], dtype=float)
            molecfit_flux = np.asarray(model_data["flux"][:n], dtype=float)
            molecfit_continuum = np.asarray(model_data["mscal"][:n], dtype=float)
            molecfit_corrected_relative[:n] = (
                molecfit_flux
                / np.where(molecfit_transmission[:n] > 0.2, molecfit_transmission[:n], np.nan)
                / molecfit_continuum
            )
            molecfit_raw_relative = np.full(wavelength.shape, np.nan)
            molecfit_raw_relative[:n] = molecfit_flux / molecfit_continuum
        reliable_molecfit = (
            np.isfinite(molecfit_corrected_relative)
            & np.isfinite(molecfit_raw_relative)
            & (molecfit_transmission > 0.2)
        )
        molecfit_raw_scatter = float(
            np.nanstd(molecfit_raw_relative[reliable_molecfit] - 1.0)
        )
        molecfit_corrected_scatter = float(
            np.nanstd(molecfit_corrected_relative[reliable_molecfit] - 1.0)
        )
        molecfit_improvement = (
            molecfit_raw_scatter / molecfit_corrected_scatter
            if molecfit_corrected_scatter > 0
            else np.inf
        )
        reliable = (
            np.isfinite(molecfit_transmission)
            & np.isfinite(result.transmission)
            & (molecfit_transmission > 0.2)
        )
        transmission_rms = float(
            np.sqrt(np.mean((result.transmission[reliable] - molecfit_transmission[reliable]) ** 2))
        )
        telluric = reliable & (molecfit_transmission < 0.995)
        telluric_rms = float(
            np.sqrt(np.mean((result.transmission[telluric] - molecfit_transmission[telluric]) ** 2))
        ) if np.any(telluric) else np.nan
        agreement_passed = transmission_rms <= 0.02 and (
            not np.isfinite(telluric_rms) or telluric_rms <= 0.03
        )
        pymolfit_chi2 = 2.0 * float(result.cost)
        pymolfit_fits_better = (
            np.isfinite(molecfit_best_chi2)
            and pymolfit_chi2 <= 1.05 * molecfit_best_chi2
            and np.isfinite(molecfit_corrected_scatter)
            and corrected_scatter <= molecfit_corrected_scatter
        )
        if agreement_passed:
            agreement_status = "PASS"
            agreement_details = f"telluric_rms={telluric_rms:.5g}"
        elif pymolfit_fits_better:
            agreement_status = "WARN"
            agreement_details = (
                f"telluric_rms={telluric_rms:.5g}; models disagree, but PyMolFit has lower "
                f"weighted objective ({pymolfit_chi2:.5g} vs {molecfit_best_chi2:.5g}) and "
                f"corrected scatter ({corrected_scatter:.5g} vs {molecfit_corrected_scatter:.5g})"
            )
        else:
            agreement_status = "FAIL"
            agreement_details = f"telluric_rms={telluric_rms:.5g}"
        checks.append(
            ValidationCheck(
                "external_reference",
                f"X-shooter {case.band} transmission versus Molecfit",
                agreement_status,
                value=transmission_rms,
                threshold="all reliable RMS <= 0.02 and telluric-pixel RMS <= 0.03",
                details=agreement_details,
                required=agreement_status != "WARN",
            )
        )
        pymolfit_gdas = Path(str(atmosphere.metadata.get("gdas_profile", "")))
        molecfit_gdas = case_dir / "GDAS.fits"
        molecfit_combined = case_dir / "ATM_PROFILE_COMBINED.fits"
        if pymolfit_gdas.exists() and molecfit_gdas.exists():
            gdas_equal, gdas_difference, gdas_details = _compare_gdas_profiles(
                pymolfit_gdas,
                molecfit_gdas,
            )
            checks.append(
                ValidationCheck(
                    "atmosphere",
                    f"X-shooter {case.band} time-local GDAS input matches Molecfit",
                    "PASS" if gdas_equal else "FAIL",
                    value=gdas_difference,
                    threshold="maximum absolute source-profile difference <= 1e-8",
                    details=gdas_details,
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    "atmosphere",
                    f"X-shooter {case.band} time-local GDAS input matches Molecfit",
                    "SKIP",
                    details="one or both GDAS source products are unavailable",
                )
            )
        if molecfit_gdas.exists() and molecfit_combined.exists():
            combined_equal, combined_difference, combined_details = (
                _compare_combined_atmosphere_profiles(
                    atmosphere,
                    molecfit_gdas,
                    molecfit_combined,
                )
            )
            checks.append(
                ValidationCheck(
                    "atmosphere",
                    f"X-shooter {case.band} merged MIPAS/GDAS profile matches Molecfit",
                    "PASS" if combined_equal else "FAIL",
                    value=combined_difference,
                    threshold="maximum absolute merged-level difference <= 1e-8 in native units",
                    details=combined_details,
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    "atmosphere",
                    f"X-shooter {case.band} merged MIPAS/GDAS profile matches Molecfit",
                    "FAIL",
                    threshold="Molecfit GDAS and ATM_PROFILE_COMBINED products exist",
                    details="one or both Molecfit atmosphere products are unavailable",
                )
            )
        objective_ratio = pymolfit_chi2 / molecfit_best_chi2
        checks.append(
            ValidationCheck(
                "real_correction",
                f"X-shooter {case.band} weighted fit quality versus Molecfit",
                "PASS" if np.isfinite(objective_ratio) and objective_ratio <= 1.05 else "FAIL",
                value=objective_ratio,
                threshold="PyMolFit weighted objective / Molecfit best chi-square <= 1.05",
                details=(
                    f"PyMolFit={pymolfit_chi2:.5g}, Molecfit={molecfit_best_chi2:.5g}; "
                    "screening and optimizer implementations are not identical"
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                "external_reference",
                f"X-shooter {case.band} transmission versus Molecfit",
                "SKIP",
                details="Molecfit executable unavailable or model run failed",
            )
        )

    table = Table()
    table["wavelength_air_micron"] = wavelength
    table["raw_flux"] = flux
    table["pymolfit_transmission"] = result.transmission
    table["pymolfit_transmission_uncertainty"] = result.transmission_uncertainty
    table["pymolfit_corrected_relative"] = corrected_relative
    table["molecfit_transmission"] = molecfit_transmission
    table["molecfit_corrected_relative"] = molecfit_corrected_relative
    table.write(case_dir / "comparison.ecsv", overwrite=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(wavelength, _normalise(flux), color="black", lw=0.8, label="raw")
    axes[0].set_ylabel("Normalized raw flux")
    axes[0].legend(loc="best")
    axes[1].plot(wavelength, result.transmission, color="C1", lw=0.9, label="PyMolFit")
    if np.any(np.isfinite(molecfit_transmission)):
        axes[1].plot(wavelength, molecfit_transmission, color="C0", lw=0.8, alpha=0.8, label="Molecfit")
    axes[1].set_ylabel("Transmission")
    axes[1].legend(loc="best")
    axes[2].plot(wavelength[reliable_gen], corrected_relative[reliable_gen], color="C1", lw=0.8, label="PyMolFit")
    if np.any(np.isfinite(molecfit_corrected_relative)):
        axes[2].plot(wavelength, molecfit_corrected_relative, color="C0", lw=0.8, alpha=0.8, label="Molecfit")
    axes[2].axhline(1.0, color="0.4", lw=0.7)
    axes[2].set_ylabel("Corrected / continuum")
    axes[2].set_xlabel("Air wavelength [micron]")
    axes[2].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"HD 53123 X-shooter: {case.band} ({', '.join(case.species)})")
    fig.tight_layout()
    fig.savefig(case_dir / "comparison.png", dpi=170)
    plt.close(fig)

    return checks, {
        "group": "xshooter",
        "case": case.name,
        "band": case.band,
        "species": ",".join(case.species),
        "n_pixels": wavelength.size,
        "n_lines": lines.wavelength.size,
        "pymolfit_seconds": gen_seconds,
        "molecfit_seconds": molecfit_seconds,
        "molecfit_best_chi2": molecfit_best_chi2,
        "molecfit_reduced_chi2": molecfit_reduced_chi2,
        "molecfit_raw_scatter": molecfit_raw_scatter,
        "molecfit_corrected_scatter": molecfit_corrected_scatter,
        "molecfit_improvement": molecfit_improvement,
        "raw_scatter": raw_scatter,
        "corrected_scatter": corrected_scatter,
        "improvement": improvement,
        "transmission_rms": transmission_rms,
        "telluric_rms": telluric_rms,
        "pymolfit_cost": result.cost,
        "pymolfit_reduced_chi_square": result.reduced_chi_square,
        "pymolfit_nfev": result.nfev,
        "pymolfit_lsf_sigma_pixels": result.lsf_sigma_pixels,
        "pymolfit_wavelength_shift_micron": result.wavelength_shift,
        "species_scales": str(result.species_scales),
        "species_scale_uncertainties": str(result.species_scale_uncertainties),
        "gdas_source": atmosphere.metadata.get("gdas_source", "unknown"),
        "gdas_profile": atmosphere.metadata.get("gdas_profile", ""),
        "vertical_h2o_column_cm2": atmosphere.total_vertical_column_cm2("H2O"),
        "vertical_o2_column_cm2": atmosphere.total_vertical_column_cm2("O2"),
    }


def _refresh_uves_reference() -> str | None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "local_tests" / "compare_uves_official_demo.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return None
    return (completed.stdout + "\n" + completed.stderr).strip()[-4000:]


def _refresh_hires_reference() -> str | None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "local_tests" / "compare_keck_hires_bd17_o2.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return None
    return (completed.stdout + "\n" + completed.stderr).strip()[-4000:]


def _refresh_kpf_reference() -> str | None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "local_tests" / "compare_keck_kpf_vega_o2.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return None
    return (completed.stdout + "\n" + completed.stderr).strip()[-4000:]


def _existing_external_checks(
    *,
    include_uves: bool,
    uves_error: str | None,
    include_hires: bool,
    hires_error: str | None,
    include_kpf: bool,
    kpf_error: str | None,
) -> tuple[list[ValidationCheck], list[dict[str, object]]]:
    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []
    harps_path = ROOT / "local_tests" / "betapic_harps_o2_comparison" / "summary.csv"
    if harps_path.exists():
        table = Table.read(harps_path, format="ascii.csv")
        values = np.asarray(table["transmission_rms"], dtype=float)
        maximum = float(np.nanmax(values))
        checks.append(
            ValidationCheck(
                "external_reference",
                "HARPS O2 across four epochs",
                "PASS" if maximum <= 0.005 else "FAIL",
                value=maximum,
                threshold="maximum epoch RMS versus Molecfit <= 0.005",
                details=f"mean={np.nanmean(values):.5g}, epochs={len(values)}",
            )
        )
        rows.append({"group": "external", "case": "HARPS_O2", "mean_rms": float(np.mean(values)), "max_rms": maximum})
    else:
        checks.append(ValidationCheck("external_reference", "HARPS O2 across four epochs", "SKIP"))

    if not include_uves:
        checks.append(
            ValidationCheck(
                "external_reference",
                "official ESO UVES demo transmission",
                "SKIP",
                details="Molecfit comparisons disabled by --skip-molecfit",
            )
        )
    elif uves_error is not None:
        checks.append(
            ValidationCheck(
                "external_reference",
                "official ESO UVES demo transmission",
                "FAIL",
                threshold="held-out UVES comparison completes reproducibly",
                details=uves_error,
            )
        )
    elif UVES_SUMMARY.exists():
        table = Table.read(UVES_SUMMARY, format="ascii.csv")
        row = table[0]
        rms = float(row["transmission_rms"])
        telluric_rms = float(row["telluric_transmission_rms"])
        maximum = float(row["transmission_max_abs"])
        objective_ratio = float(row["weighted_objective_ratio"])
        scatter_ratio = float(row["relative_scatter_ratio"])
        covariance_rank = int(float(row["pymolfit_covariance_rank"]))
        parameter_count = int(float(row["pymolfit_parameter_count"]))
        direct_pass = rms <= 0.005 and telluric_rms <= 0.01 and maximum <= 0.02
        checks.extend(
            [
                ValidationCheck(
                    "external_reference",
                    "official ESO UVES demo transmission",
                    "PASS" if direct_pass else "FAIL",
                    value=rms,
                    threshold="RMS <= 0.005, telluric RMS <= 0.01, max <= 0.02",
                    details=f"telluric_rms={telluric_rms:.5g}, max={maximum:.5g}",
                ),
                ValidationCheck(
                    "real_correction",
                    "official ESO UVES demo fit quality",
                    "PASS" if objective_ratio <= 1.05 and scatter_ratio <= 1.05 else "FAIL",
                    value=objective_ratio,
                    threshold="weighted objective and relative-scatter ratios versus Molecfit <= 1.05",
                    details=f"objective_ratio={objective_ratio:.5g}, scatter_ratio={scatter_ratio:.5g}",
                ),
                ValidationCheck(
                    "uncertainty",
                    "UVES wavelength/LSF fit is locally identifiable",
                    "PASS" if covariance_rank == parameter_count else "FAIL",
                    value=float(covariance_rank),
                    threshold="covariance rank equals nonlinear parameter count",
                    details=f"rank={covariance_rank}, parameters={parameter_count}",
                ),
            ]
        )
        rows.append(
            {
                "group": "external",
                "case": "UVES_official_demo",
                "band": "optical",
                "transmission_rms": rms,
                "telluric_rms": telluric_rms,
                "max_rms": maximum,
                "objective_ratio": objective_ratio,
                "corrected_scatter_ratio": scatter_ratio,
                "covariance_rank": covariance_rank,
                "parameter_count": parameter_count,
                "pymolfit_seconds": float(row["pymolfit_seconds"]),
                "molecfit_seconds": float(row["molecfit_seconds"]),
            }
        )
    else:
        checks.append(
            ValidationCheck(
                "external_reference",
                "official ESO UVES demo transmission",
                "FAIL",
                threshold="held-out UVES comparison summary exists",
                details=f"missing {UVES_SUMMARY}",
            )
        )

    if not include_hires:
        checks.append(
            ValidationCheck(
                "external_reference",
                "Keck HIRES O2 transmission",
                "SKIP",
                details="Molecfit comparisons disabled by --skip-molecfit",
            )
        )
    elif hires_error is not None:
        checks.append(
            ValidationCheck(
                "external_reference",
                "Keck HIRES O2 transmission",
                "FAIL",
                threshold="non-ESO HIRES comparison completes reproducibly",
                details=hires_error,
            )
        )
    elif HIRES_SUMMARY.exists():
        table = Table.read(HIRES_SUMMARY, format="ascii.csv")
        row = table[0]
        rms = float(row["transmission_rms"])
        telluric_rms = float(row["telluric_transmission_rms"])
        maximum = float(row["transmission_max_abs"])
        objective_ratio = float(row["weighted_objective_ratio"])
        gen_corrected = float(row["pymolfit_telluric_rms_from_unity"])
        mol_corrected = float(row["molecfit_telluric_rms_from_unity"])
        improvement = float(row["pymolfit_telluric_rms_improvement"])
        covariance_rank = int(float(row["pymolfit_covariance_rank"]))
        parameter_count = int(float(row["pymolfit_parameter_count"]))
        cv_prediction = float(row["cross_validation_prediction_coverage"])
        cv_reliable = float(row["cross_validation_reliable_correction_coverage"])
        cv_relative = float(row["cross_validation_relative_rms_improvement"])
        cv_weighted = float(row["cross_validation_weighted_rms_improvement"])
        cv_success = float(row["cross_validation_all_folds_successful"])
        direct_pass = rms <= 0.01 and telluric_rms <= 0.01 and maximum <= 0.03
        correction_pass = (
            objective_ratio <= 1.05
            and gen_corrected <= 1.10 * mol_corrected
            and improvement >= 5.0
        )
        cross_validation_pass = (
            cv_prediction >= 0.999
            and cv_reliable >= 0.80
            and cv_relative >= 1.5
            and cv_weighted >= 1.5
            and cv_success == 1.0
        )
        checks.extend(
            [
                ValidationCheck(
                    "external_reference",
                    "Keck HIRES O2 transmission",
                    "PASS" if direct_pass else "FAIL",
                    value=rms,
                    threshold="RMS and telluric RMS <= 0.01, max <= 0.03",
                    details=f"telluric_rms={telluric_rms:.5g}, max={maximum:.5g}",
                ),
                ValidationCheck(
                    "real_correction",
                    "Keck HIRES O2 correction quality",
                    "PASS" if correction_pass else "FAIL",
                    value=improvement,
                    threshold=(
                        "objective ratio <= 1.05, corrected RMS <= 1.10x Molecfit, "
                        "raw-to-corrected improvement >= 5"
                    ),
                    details=(
                        f"objective_ratio={objective_ratio:.5g}, Gen_RMS={gen_corrected:.5g}, "
                        f"Molecfit_RMS={mol_corrected:.5g}"
                    ),
                ),
                ValidationCheck(
                    "uncertainty",
                    "HIRES O2 fit is locally identifiable",
                    "PASS" if covariance_rank == parameter_count else "FAIL",
                    value=float(covariance_rank),
                    threshold="covariance rank equals nonlinear parameter count",
                    details=f"rank={covariance_rank}, parameters={parameter_count}",
                ),
                ValidationCheck(
                    "generalization",
                    "Keck HIRES O2 blocked out-of-fold prediction",
                    "PASS" if cross_validation_pass else "FAIL",
                    value=cv_relative,
                    threshold=(
                        "prediction coverage >= 0.999, reliable correction coverage >= 0.80, "
                        "relative and weighted RMS improvement >= 1.5, all folds converge"
                    ),
                    details=(
                        f"block_size=64, folds=2, prediction={cv_prediction:.5g}, "
                        f"reliable={cv_reliable:.5g}, weighted_improvement={cv_weighted:.5g}"
                    ),
                ),
            ]
        )
        rows.append(
            {
                "group": "external",
                "case": "Keck_HIRES_BD17_O2",
                "band": "optical",
                "transmission_rms": rms,
                "telluric_rms": telluric_rms,
                "max_rms": maximum,
                "objective_ratio": objective_ratio,
                "corrected_rms_ratio": gen_corrected / mol_corrected,
                "telluric_rms_improvement": improvement,
                "covariance_rank": covariance_rank,
                "parameter_count": parameter_count,
                "cross_validation_prediction_coverage": cv_prediction,
                "cross_validation_reliable_correction_coverage": cv_reliable,
                "cross_validation_relative_rms_improvement": cv_relative,
                "cross_validation_weighted_rms_improvement": cv_weighted,
                "pymolfit_seconds": float(row["pymolfit_seconds"]),
                "pymolfit_cross_validation_seconds": float(
                    row["pymolfit_cross_validation_seconds"]
                ),
                "molecfit_seconds": float(row["molecfit_seconds"]),
            }
        )
    else:
        checks.append(
            ValidationCheck(
                "external_reference",
                "Keck HIRES O2 transmission",
                "FAIL",
                threshold="non-ESO HIRES comparison summary exists",
                details=f"missing {HIRES_SUMMARY}",
            )
        )

    if not include_kpf:
        checks.append(
            ValidationCheck(
                "external_reference",
                "Keck KPF Vega O2 transmission",
                "SKIP",
                details="Molecfit comparisons disabled by --skip-molecfit",
            )
        )
    elif kpf_error is not None:
        checks.append(
            ValidationCheck(
                "external_reference",
                "Keck KPF Vega O2 transmission",
                "FAIL",
                threshold="independently reduced KPF comparison completes reproducibly",
                details=kpf_error,
            )
        )
    elif KPF_SUMMARY.exists():
        table = Table.read(KPF_SUMMARY, format="ascii.csv")
        row = table[0]
        rms = float(row["transmission_rms"])
        telluric_rms = float(row["telluric_transmission_rms"])
        maximum = float(row["transmission_max_abs"])
        objective_ratio = float(row["weighted_objective_ratio"])
        gen_corrected = float(row["pymolfit_telluric_rms_from_unity"])
        mol_corrected = float(row["molecfit_telluric_rms_from_unity"])
        improvement = float(row["pymolfit_telluric_rms_improvement"])
        covariance_rank = int(float(row["pymolfit_covariance_rank"]))
        parameter_count = int(float(row["pymolfit_parameter_count"]))
        systematics_p95 = float(row["systematics_transmission_rms_p95"])
        systematics_max = float(row["systematics_transmission_envelope_max"])
        systematics_finite = float(row["systematics_finite_fraction"])
        cv_prediction = float(row["cross_validation_prediction_coverage"])
        cv_reliable = float(row["cross_validation_reliable_correction_coverage"])
        cv_relative = float(row["cross_validation_relative_rms_improvement"])
        cv_weighted = float(row["cross_validation_weighted_rms_improvement"])
        cv_success = float(row["cross_validation_all_folds_successful"])
        direct_pass = rms <= 0.01 and telluric_rms <= 0.01 and maximum <= 0.03
        correction_pass = (
            objective_ratio <= 1.05
            and gen_corrected <= 1.10 * mol_corrected
            and improvement >= 5.0
        )
        systematics_pass = (
            systematics_p95 <= 0.02
            and systematics_max <= 0.05
            and systematics_finite >= 0.999
        )
        cross_validation_pass = (
            cv_prediction >= 0.999
            and cv_reliable >= 0.80
            and cv_relative >= 1.5
            and cv_weighted >= 1.5
            and cv_success == 1.0
        )
        checks.extend(
            [
                ValidationCheck(
                    "external_reference",
                    "Keck KPF Vega O2 transmission",
                    "PASS" if direct_pass else "FAIL",
                    value=rms,
                    threshold="RMS and telluric RMS <= 0.01, max <= 0.03",
                    details=f"telluric_rms={telluric_rms:.5g}, max={maximum:.5g}",
                ),
                ValidationCheck(
                    "real_correction",
                    "Keck KPF Vega O2 correction quality",
                    "PASS" if correction_pass else "FAIL",
                    value=improvement,
                    threshold=(
                        "objective ratio <= 1.05, corrected RMS <= 1.10x Molecfit, "
                        "raw-to-corrected improvement >= 5"
                    ),
                    details=(
                        f"objective_ratio={objective_ratio:.5g}, Gen_RMS={gen_corrected:.5g}, "
                        f"Molecfit_RMS={mol_corrected:.5g}"
                    ),
                ),
                ValidationCheck(
                    "uncertainty",
                    "KPF O2 fit is locally identifiable",
                    "PASS" if covariance_rank == parameter_count else "FAIL",
                    value=float(covariance_rank),
                    threshold="covariance rank equals nonlinear parameter count",
                    details=f"rank={covariance_rank}, parameters={parameter_count}",
                ),
                ValidationCheck(
                    "uncertainty",
                    "KPF O2 atmosphere/continuum model-systematics envelope",
                    "PASS" if systematics_pass else "FAIL",
                    value=systematics_p95,
                    threshold="p95 transmission RMS <= 0.02, envelope max <= 0.05, finite fraction >= 0.999",
                    details=(
                        f"variants=seasonal_gdas,continuum_order_2,continuum_order_4; "
                        f"envelope_max={systematics_max:.5g}, finite={systematics_finite:.5g}"
                    ),
                ),
                ValidationCheck(
                    "generalization",
                    "Keck KPF Vega O2 blocked out-of-fold prediction",
                    "PASS" if cross_validation_pass else "FAIL",
                    value=cv_relative,
                    threshold=(
                        "prediction coverage >= 0.999, reliable correction coverage >= 0.80, "
                        "relative and weighted RMS improvement >= 1.5, all folds converge"
                    ),
                    details=(
                        f"block_size=64, folds=2, prediction={cv_prediction:.5g}, "
                        f"reliable={cv_reliable:.5g}, weighted_improvement={cv_weighted:.5g}"
                    ),
                ),
            ]
        )
        rows.append(
            {
                "group": "external",
                "case": "Keck_KPF_Vega_O2",
                "band": "optical",
                "transmission_rms": rms,
                "telluric_rms": telluric_rms,
                "max_rms": maximum,
                "objective_ratio": objective_ratio,
                "corrected_rms_ratio": gen_corrected / mol_corrected,
                "telluric_rms_improvement": improvement,
                "covariance_rank": covariance_rank,
                "parameter_count": parameter_count,
                "transmission_systematic_rms_p95": systematics_p95,
                "transmission_systematic_envelope_max": systematics_max,
                "finite_systematic_fraction": systematics_finite,
                "cross_validation_prediction_coverage": cv_prediction,
                "cross_validation_reliable_correction_coverage": cv_reliable,
                "cross_validation_relative_rms_improvement": cv_relative,
                "cross_validation_weighted_rms_improvement": cv_weighted,
                "pymolfit_seconds": float(row["pymolfit_seconds"]),
                "pymolfit_cross_validation_seconds": float(
                    row["pymolfit_cross_validation_seconds"]
                ),
                "molecfit_seconds": float(row["molecfit_seconds"]),
            }
        )
    else:
        checks.append(
            ValidationCheck(
                "external_reference",
                "Keck KPF Vega O2 transmission",
                "FAIL",
                threshold="independently reduced KPF comparison summary exists",
                details=f"missing {KPF_SUMMARY}",
            )
        )

    crires_path = ROOT / "local_tests" / "rho01_molecfit_vs_pymolfit_lband" / "summary.csv"
    if crires_path.exists():
        table = Table.read(crires_path, format="ascii.csv")
        values = np.asarray(table["transmission_rms_difference"], dtype=float)
        mean = float(np.nanmean(values))
        maximum = float(np.nanmax(values))
        direct_pass = mean <= 0.01 and maximum <= 0.03
        has_shape_metrics = all(
            name in table.colnames
            for name in (
                "continuum_invariant_shape_rms",
                "pymolfit_weighted_objective",
                "molecfit_weighted_objective",
                "pymolfit_corrected_scatter",
                "molecfit_corrected_scatter",
            )
        )
        shape_mean = shape_max = objective_ratio = scatter_ratio = np.nan
        shape_pass = False
        if has_shape_metrics:
            shape = np.asarray(table["continuum_invariant_shape_rms"], dtype=float)
            shape_mean = float(np.nanmean(shape))
            shape_max = float(np.nanmax(shape))
            gen_objective = float(np.nansum(table["pymolfit_weighted_objective"]))
            molecfit_objective = float(np.nansum(table["molecfit_weighted_objective"]))
            objective_ratio = gen_objective / molecfit_objective
            gen_scatter = float(np.nanmedian(table["pymolfit_corrected_scatter"]))
            molecfit_scatter = float(np.nanmedian(table["molecfit_corrected_scatter"]))
            scatter_ratio = gen_scatter / molecfit_scatter
            shape_pass = (
                shape_mean <= 0.015
                and shape_max <= 0.03
                and objective_ratio <= 1.05
                and scatter_ratio <= 1.05
            )
        direct_status = "PASS" if direct_pass else ("WARN" if shape_pass else "FAIL")
        checks.append(
            ValidationCheck(
                "external_reference",
                "CRIRES+ L-band 18-chip transmission",
                direct_status,
                value=mean,
                threshold="mean RMS <= 0.01 and maximum chip RMS <= 0.03",
                details=(
                    f"max={maximum:.5g}, median={np.nanmedian(values):.5g}; "
                    "direct transmission includes smooth continuum-model degeneracy"
                ),
                required=direct_status != "WARN",
            )
        )
        if has_shape_metrics:
            checks.append(
                ValidationCheck(
                    "real_correction",
                    "CRIRES+ continuum-invariant line shape and corrected fit",
                    "PASS" if shape_pass else "FAIL",
                    value=shape_mean,
                    threshold=(
                        "shape mean RMS <= 0.015, max <= 0.03, weighted objective and "
                        "corrected scatter ratios <= 1.05"
                    ),
                    details=(
                        f"shape_max={shape_max:.5g}, objective_ratio={objective_ratio:.4g}, "
                        f"scatter_ratio={scatter_ratio:.4g}"
                    ),
                )
            )
        rows.append(
            {
                "group": "external",
                "case": "CRIRES_L",
                "mean_rms": mean,
                "max_rms": maximum,
                "shape_mean_rms": shape_mean,
                "shape_max_rms": shape_max,
                "objective_ratio": objective_ratio,
                "corrected_scatter_ratio": scatter_ratio,
            }
        )
    else:
        checks.append(ValidationCheck("external_reference", "CRIRES+ L-band 18-chip transmission", "SKIP"))
    return checks, rows


def _external_physics_golden_checks() -> tuple[list[ValidationCheck], list[dict[str, object]]]:
    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []
    cases = (
        (
            "LBLRTM combined H2O/N2 continuum window",
            ROOT / "local_tests" / "lblrtm_external_golden" / "summary.csv",
            "h2o_lines_and_continuum",
            0.005,
            0.015,
            0.03,
            0.05,
        ),
        (
            "LBLRTM N2 overtone continuum differential",
            ROOT / "local_tests" / "lblrtm_external_golden_n2_overtone" / "summary.csv",
            "n2_continuum_differential",
            2.0e-5,
            2.0e-4,
            0.01,
            0.01,
        ),
        (
            "LBLRTM O2 1.27 micron continuum differential",
            ROOT / "local_tests" / "lblrtm_external_golden_o2_127" / "summary.csv",
            "o2_continuum_differential",
            3.0e-4,
            0.002,
            0.01,
            0.01,
        ),
        (
            "LBLRTM O2 visible continuum differential",
            ROOT / "local_tests" / "lblrtm_external_golden_o2_visible" / "summary.csv",
            "o2_continuum_differential",
            1.0e-4,
            0.002,
            0.01,
            0.05,
        ),
    )
    for name, path, case, rms_limit, maximum_limit, integral_limit, peak_limit in cases:
        if not path.exists():
            checks.append(
                ValidationCheck(
                    "external_physics",
                    name,
                    "SKIP",
                    details=f"missing {path.relative_to(ROOT)}",
                )
            )
            continue
        table = Table.read(path, format="ascii.csv")
        matching = table[np.asarray(table["case"], dtype=str) == case]
        if len(matching) != 1:
            checks.append(
                ValidationCheck(
                    "external_physics",
                    name,
                    "FAIL",
                    details=f"expected one {case!r} row in {path.relative_to(ROOT)}",
                )
            )
            continue
        row = matching[0]
        rms = float(row["transmission_rms"])
        maximum = float(row["transmission_max_abs"])
        integral_ratio = float(row["tau_integral_ratio"])
        peak_ratio = float(row["tau_peak_ratio"])
        passed = (
            rms <= rms_limit
            and maximum <= maximum_limit
            and abs(integral_ratio - 1.0) <= integral_limit
            and abs(peak_ratio - 1.0) <= peak_limit
        )
        checks.append(
            ValidationCheck(
                "external_physics",
                name,
                "PASS" if passed else "FAIL",
                value=rms,
                threshold=(
                    f"RMS <= {rms_limit:g}, max <= {maximum_limit:g}, "
                    f"|integral ratio - 1| <= {integral_limit:g}, "
                    f"|peak ratio - 1| <= {peak_limit:g}"
                ),
                details=(
                    f"max={maximum:.6g}, integral_ratio={integral_ratio:.6g}, "
                    f"peak_ratio={peak_ratio:.6g}; external LBLRTM is audit-only"
                ),
            )
        )
        rows.append(
            {
                "group": "external_physics",
                "case": case,
                "transmission_rms": rms,
                "transmission_max_abs": maximum,
                "tau_integral_ratio": integral_ratio,
                "tau_peak_ratio": peak_ratio,
            }
        )

    single_line_path = ROOT / "local_tests" / "lblrtm_audit" / "single_line_audit_summary.txt"
    single_line_max = np.nan
    if single_line_path.exists():
        for line in single_line_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("max_abs_tau_panel_minus_reference"):
                single_line_max = float(line.split("=", 1)[1])
                break
    single_line_pass = np.isfinite(single_line_max) and single_line_max <= 1.0e-12
    checks.append(
        ValidationCheck(
            "external_physics",
            "independent accumulated-panel single-line audit",
            "PASS" if single_line_pass else ("SKIP" if not single_line_path.exists() else "FAIL"),
            value=float(single_line_max) if np.isfinite(single_line_max) else None,
            threshold="maximum optical-depth difference <= 1e-12",
            details="synthetic source-equivalence audit; no observed-spectrum coefficients",
        )
    )
    if np.isfinite(single_line_max):
        rows.append(
            {
                "group": "external_physics",
                "case": "single_line_accumulated_panel",
                "max_tau_difference": float(single_line_max),
            }
        )

    o2_aband_path = ROOT / "local_tests" / "lblrtm_external_golden_o2_aband" / "summary.csv"
    if o2_aband_path.exists():
        table = Table.read(o2_aband_path, format="ascii.csv")
        matching = table[np.asarray(table["case"], dtype=str) == "o2_continuum_differential"]
        if len(matching) == 1:
            rms = float(matching[0]["transmission_rms"])
            checks.append(
                ValidationCheck(
                    "external_physics",
                    "O2 A-band continuum plotting-grid diagnostic",
                    "WARN",
                    value=rms,
                    threshold="diagnostic only: differential division is affected by convolved saturated O2 lines",
                    details=(
                        "TAPE28 plotting convolution does not commute with line-only division; "
                        "the source O2 continuum branch is tested separately and this check is non-gating"
                    ),
                    required=False,
                )
            )
    return checks, rows


def _runtime_check() -> tuple[ValidationCheck, dict[str, object]]:
    path = ROOT / "local_tests" / "fair_runtime_benchmark" / "runtime_summary.csv"
    if not path.exists():
        return ValidationCheck("runtime", "matched CRIRES benchmark", "SKIP"), {}
    table = Table.read(path, format="ascii.csv")
    mapping = {str(row["run"]): float(row["seconds"]) for row in table}
    gen = mapping["pymolfit_from_science_input"]
    molecfit = mapping["molecfit_model_only"]
    ratio = gen / molecfit
    return (
        ValidationCheck(
            "runtime",
            "matched CRIRES benchmark",
            "PASS" if ratio <= 1.25 else "FAIL",
            value=ratio,
            threshold="PyMolFit/Molecfit wall time <= 1.25",
            details=f"PyMolFit={gen:.3f}s, Molecfit={molecfit:.3f}s",
        ),
        {"group": "runtime", "case": "CRIRES_18_segment", "gen_seconds": gen, "molecfit_seconds": molecfit, "ratio": ratio},
    )


def _fixed_rt_parity_checks(
    *,
    run_reference: bool,
) -> tuple[list[ValidationCheck], list[dict[str, object]]]:
    """Run the fixed-parameter X-shooter radiative-transfer parity audit."""

    if not run_reference:
        return [
            ValidationCheck(
                "external_physics",
                "fixed-parameter X-shooter radiative-transfer parity",
                "SKIP",
                required=False,
                details="Molecfit reference checks were explicitly skipped",
            )
        ], []

    summary = OUTPUT_DIR / "xshooter_fixed_rt_parity" / "summary.csv"
    completed = subprocess.run(
        [sys.executable, "-m", "local_tests.audit_xshooter_fixed_rt_parity"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not summary.exists():
        details = (completed.stdout + "\n" + completed.stderr).strip()[-2000:]
        return [
            ValidationCheck(
                "external_physics",
                "fixed-parameter X-shooter radiative-transfer parity",
                "FAIL",
                threshold="audit executes and writes four case metrics",
                details=details or "fixed-parameter audit produced no summary",
            )
        ], []

    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []
    with summary.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != len(REAL_BANDS):
        return [
            ValidationCheck(
                "external_physics",
                "fixed-parameter X-shooter radiative-transfer parity",
                "FAIL",
                value=float(len(records)),
                threshold=f"exactly {len(REAL_BANDS)} fixed-parameter cases",
                details=str(summary.relative_to(ROOT)),
            )
        ], []

    for record in records:
        rms = float(record["all_reliable_rms"])
        telluric_rms = float(record["telluric_rms"])
        maximum = float(record["maximum_absolute_difference"])
        correlation = float(record["optical_depth_correlation"])
        scale = float(record["molecfit_to_pymolfit_optical_depth_scale"])
        passed = (
            rms <= 0.002
            and telluric_rms <= 0.002
            and maximum <= 0.005
            and correlation >= 0.9999
            and abs(scale - 1.0) <= 0.01
        )
        case = str(record["case"])
        checks.append(
            ValidationCheck(
                "external_physics",
                f"{case} fixed-parameter radiative-transfer parity",
                "PASS" if passed else "FAIL",
                value=rms,
                threshold=(
                    "RMS/telluric RMS <= 0.002; max <= 0.005; "
                    "tau correlation >= 0.9999; |tau scale - 1| <= 0.01"
                ),
                details=(
                    f"telluric_rms={telluric_rms:.6g}, max={maximum:.6g}, "
                    f"tau_corr={correlation:.8f}, tau_scale={scale:.8f}"
                ),
            )
        )
        rows.append(
            {
                "group": "fixed_rt_parity",
                "case": case,
                "transmission_rms": rms,
                "telluric_rms": telluric_rms,
                "transmission_max_abs": maximum,
                "optical_depth_correlation": correlation,
                "optical_depth_scale_ratio": scale,
            }
        )
    return checks, rows


def _package_checks() -> tuple[list[ValidationCheck], list[dict[str, object]]]:
    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []

    started = time.perf_counter()
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    test_seconds = time.perf_counter() - started
    test_tail = " | ".join((tests.stdout + tests.stderr).splitlines()[-3:])
    checks.append(
        ValidationCheck(
            "package",
            "complete automated test suite",
            "PASS" if tests.returncode == 0 else "FAIL",
            value=test_seconds,
            units="s",
            threshold="pytest exits successfully",
            details=test_tail,
        )
    )
    rows.append(
        {
            "group": "package",
            "case": "pytest",
            "seconds": test_seconds,
            "returncode": tests.returncode,
        }
    )

    wheels = sorted((ROOT / "dist").glob("pymolfit-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        checks.append(ValidationCheck("package", "wheel contents and freshness", "FAIL", details="no wheel in dist/"))
        checks.append(ValidationCheck("package", "isolated wheel install and smoke test", "SKIP"))
        return checks, rows

    wheel = wheels[-1]
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    source_files = [
        *list((ROOT / "src" / "pymolfit").rglob("*.py")),
        *list((ROOT / "src" / "pymolfit" / "data").rglob("*")),
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]
    source_files = [path for path in source_files if path.is_file()]
    fresh = wheel.stat().st_mtime >= max(path.stat().st_mtime for path in source_files)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    expected_suffixes = (
        "pymolfit/__init__.py",
        "pymolfit/aer_data.py",
        "pymolfit/provenance.py",
        "pymolfit/systematics.py",
        "pymolfit/validation.py",
        "pymolfit/data/lblrtm_v12_11_n2_fundamental.npz",
        "pymolfit/data/lblrtm_v12_11_o2_continuum.npz",
        "pymolfit/data/lblrtm_v12_11_h2o_continuum.npz",
        "pymolfit/data/lblrtm_v12_11_co2_continuum.npz",
    )
    missing = [suffix for suffix in expected_suffixes if not any(name.endswith(suffix) for name in names)]
    notice_present = any(
        name.endswith(".dist-info/THIRD_PARTY_NOTICES.md")
        or name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md")
        for name in names
    )
    if not notice_present:
        missing.append(".dist-info[/licenses]/THIRD_PARTY_NOTICES.md")
    wheel_ok = fresh and not missing
    checks.append(
        ValidationCheck(
            "package",
            "wheel contents and freshness",
            "PASS" if wheel_ok else "FAIL",
            value=float(wheel.stat().st_size),
            units="bytes",
            threshold="wheel newer than package sources and contains required physics data/notices",
            details=(
                f"wheel={wheel.name}, sha256={wheel_sha256}, "
                f"fresh={fresh}, missing={missing}"
            ),
        )
    )

    with tempfile.TemporaryDirectory(prefix="pymolfit_wheel_campaign_") as temporary:
        temporary_path = Path(temporary)
        environment = temporary_path / "venv"
        create = subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            text=True,
            capture_output=True,
            check=False,
        )
        python = environment / "bin" / "python"
        install = subprocess.CompletedProcess([], 1, "", "virtual environment creation failed")
        smoke = subprocess.CompletedProcess([], 1, "", "wheel installation was not attempted")
        if create.returncode == 0:
            install = subprocess.run(
                [str(python), "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)],
                cwd=temporary_path,
                text=True,
                capture_output=True,
                check=False,
            )
        if install.returncode == 0:
            smoke_code = """
import pathlib
import numpy as np
import pymolfit
from pymolfit import LBLRTMN2OvertoneContinuum, LBLRTMO2Continuum, correct_arrays
assert 'site-packages' in pathlib.Path(pymolfit.__file__).parts
assert LBLRTMN2OvertoneContinuum.from_package_data().wavenumber_cm.size == 191
assert LBLRTMO2Continuum.from_package_data().visible_wavenumber_cm.size == 1474
try:
    correct_arrays(np.linspace(2.31, 2.36, 20), np.ones(20), aer_catalog=None)
except ValueError as exc:
    assert 'no molecular line data supplied' in str(exc)
else:
    raise AssertionError('implicit demo line data were accepted')
result = correct_arrays(
    np.linspace(2.31, 2.36, 200),
    np.ones(200),
    demo_line_list=True,
    continuum_order=0,
)
assert result.success
"""
            environment_variables = dict(os.environ)
            environment_variables.pop("PYTHONPATH", None)
            smoke = subprocess.run(
                [str(python), "-c", smoke_code],
                cwd=temporary_path,
                env=environment_variables,
                text=True,
                capture_output=True,
                check=False,
            )
    smoke_ok = create.returncode == 0 and install.returncode == 0 and smoke.returncode == 0
    smoke_details = " | ".join(
        part.strip()
        for part in (create.stderr, install.stderr, smoke.stdout, smoke.stderr)
        if part.strip()
    )
    checks.append(
        ValidationCheck(
            "package",
            "isolated wheel install and smoke test",
            "PASS" if smoke_ok else "FAIL",
            threshold="fresh venv imports wheel data, rejects implicit demo physics, and completes a fit",
            details=smoke_details[-1000:],
        )
    )
    rows.append(
        {
            "group": "package",
            "case": "wheel",
            "wheel": wheel.name,
            "wheel_sha256": wheel_sha256,
            "wheel_bytes": wheel.stat().st_size,
            "wheel_fresh": fresh,
            "missing_contents": str(missing),
            "isolated_smoke": smoke_ok,
        }
    )
    return checks, rows


def _write_metrics(rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (OUTPUT_DIR / "campaign_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_plot(report: ScienceReadinessReport, rows: list[dict[str, object]]) -> None:
    counts = report.status_counts()
    statuses = ["PASS", "WARN", "FAIL", "MANUAL", "SKIP"]
    colors = {"PASS": "#2b8c4b", "WARN": "#d49a22", "FAIL": "#c43d3d", "MANUAL": "#476a9e", "SKIP": "#888888"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(statuses, [counts.get(status, 0) for status in statuses], color=[colors[s] for s in statuses])
    axes[0].set_ylabel("Number of checks")
    axes[0].set_title(report.verdict.replace("_", " ").title())
    axes[0].grid(axis="y", alpha=0.25)

    synthetic = [row for row in rows if row.get("group") == "synthetic_recovery"]
    axes[1].bar(
        [str(row["case"]) for row in synthetic],
        [100.0 * float(row["relative_error"]) for row in synthetic],
        color="#476a9e",
    )
    axes[1].axhline(3.0, color="#c43d3d", ls="--", lw=1.0, label="3% criterion")
    axes[1].set_ylabel("Molecular scale error [%]")
    axes[1].set_title("Synthetic physical recovery")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "science_readiness_summary.png", dpi=170)
    plt.close(fig)


def _write_review_packet() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "local_tests" / "make_blind_review_packet.py")],
        cwd=ROOT,
        check=True,
    )


def _display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _review_packet_archive_check() -> ValidationCheck:
    manifest_member = "independent_review/packet_manifest.json"
    try:
        with zipfile.ZipFile(REVIEW_ARCHIVE) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError("archive contains duplicate member names")
            if any(
                not name.startswith("independent_review/")
                or name.startswith("/")
                or ".." in Path(name).parts
                or "answer_key" in name.lower()
                for name in members
            ):
                raise ValueError("archive contains an unsafe or private member name")
            if manifest_member not in members:
                raise ValueError("archive manifest is missing")
            manifest = json.loads(archive.read(manifest_member))
            if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
                raise ValueError("archive manifest has an unsupported schema")

            entries = manifest["files"]
            paths = [str(entry.get("path", "")) for entry in entries]
            if len(paths) != len(set(paths)) or any(
                not path or Path(path).is_absolute() or ".." in Path(path).parts for path in paths
            ):
                raise ValueError("archive manifest contains unsafe or duplicate paths")
            expected_members = {manifest_member} | {
                f"independent_review/{path}" for path in paths
            }
            if set(members) != expected_members:
                raise ValueError("archive contents do not exactly match its manifest")
            required = {"README.md", "review.csv", "held_out_review.csv"}
            if not required.issubset(paths):
                raise ValueError("archive is missing reviewer instructions or forms")
            case_tables = {path for path in paths if path.startswith("case_") and path.endswith(".ecsv")}
            case_plots = {path for path in paths if path.startswith("case_") and path.endswith(".png")}
            if len(case_tables) != 9 or {Path(path).stem for path in case_tables} != {
                Path(path).stem for path in case_plots
            }:
                raise ValueError("archive does not contain nine matched case table/plot pairs")

            for entry in entries:
                relative = str(entry["path"])
                payload = archive.read(f"independent_review/{relative}")
                digest = hashlib.sha256(payload).hexdigest()
                if len(payload) != int(entry["size_bytes"]) or digest != entry["sha256"]:
                    raise ValueError(f"manifest hash mismatch for {relative}")
                source = REVIEW_PACKET / relative
                if not source.is_file() or source.read_bytes() != payload:
                    raise ValueError(f"archive member is stale relative to {relative}")
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        return ValidationCheck(
            "independent_review",
            "reviewer-safe archive integrity",
            "FAIL",
            threshold="manifest-valid archive with no private answer key and nine matched cases",
            details=str(exc),
        )

    return ValidationCheck(
        "independent_review",
        "reviewer-safe archive integrity",
        "PASS",
        threshold="manifest-valid archive with no private answer key and nine matched cases",
        details=(
            f"{_display_path(REVIEW_ARCHIVE)}, "
            f"sha256={hashlib.sha256(REVIEW_ARCHIVE.read_bytes()).hexdigest()}"
        ),
    )


def _is_sha256(value: object) -> bool:
    text = str(value)
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _authenticated_hitran_check() -> ValidationCheck:
    command = "PYTHONPATH=src python local_tests/validate_authenticated_hitran.py"
    if not HITRAN_RECEIPT.is_file():
        return ValidationCheck(
            "line_data",
            "authenticated HITRAN API acquisition",
            "MANUAL",
            required=False,
            threshold="forced API v2 request succeeds with a real account and stores no credential",
            details=f"live receipt pending; run `{command}` with HITRAN_API_KEY set",
        )

    try:
        receipt = json.loads(HITRAN_RECEIPT.read_text(encoding="utf-8"))
        request = receipt["request"]
        encoded = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        request_sha256 = hashlib.sha256(encoded).hexdigest()
        source_sha256 = hashlib.sha256(
            (ROOT / "src" / "pymolfit" / "line_data.py").read_bytes()
        ).hexdigest()
        validator_sha256 = hashlib.sha256(
            (ROOT / "local_tests" / "validate_authenticated_hitran.py").read_bytes()
        ).hexdigest()
        artifacts = receipt["artifact_sha256"]
        valid = (
            receipt.get("schema_version") == 1
            and receipt.get("status") == "PASS"
            and receipt.get("online_request_completed") is True
            and receipt.get("cache_hit") is False
            and receipt.get("credential_environment_variable") == "HITRAN_API_KEY"
            and receipt.get("credential_persisted") is False
            and receipt.get("api_version") == "v2"
            and receipt.get("api_base_url") == "https://hitran.org"
            and receipt.get("pymolfit_version") == __version__
            and receipt.get("client_source_sha256") == source_sha256
            and receipt.get("validator_source_sha256") == validator_sha256
            and request.get("source") == "hitran_api"
            and request.get("species") == ["O2"]
            and request.get("molecule_ids") == [7]
            and np.isclose(float(request.get("wavenumber_min_cm")), 13160.0)
            and np.isclose(float(request.get("wavenumber_max_cm")), 13161.0)
            and receipt.get("request_sha256") == request_sha256
            and int(receipt.get("line_count", 0)) > 0
            and "O2" in receipt.get("species_with_lines", [])
            and isinstance(artifacts, dict)
            and all(_is_sha256(artifacts.get(name)) for name in ("par", "table", "manifest"))
        )
        if not valid:
            raise ValueError("receipt does not match the fixed live request or current client source")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return ValidationCheck(
            "line_data",
            "authenticated HITRAN API acquisition",
            "FAIL",
            required=False,
            threshold="forced API v2 request succeeds with a real account and stores no credential",
            details=f"invalid live receipt: {exc}",
        )

    return ValidationCheck(
        "line_data",
        "authenticated HITRAN API acquisition",
        "PASS",
        required=False,
        value=float(receipt["line_count"]),
        units="lines",
        threshold="forced API v2 request succeeds with a real account and stores no credential",
        details=(
            f"O2 13160--13161 cm^-1, database={receipt.get('database_edition')}, "
            f"receipt={_display_path(HITRAN_RECEIPT)}"
        ),
    )


def _read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _independent_review_check() -> ValidationCheck:
    packet = OUTPUT_DIR / "independent_review"
    answer_path = OUTPUT_DIR / "independent_review_answer_key.csv"
    review_path = packet / "review.csv"
    held_out_path = packet / "held_out_review.csv"
    if not all(path.exists() for path in (answer_path, review_path, held_out_path)):
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "FAIL",
            threshold="independent reviewer finds no scientifically material regression",
            details="review packet or answer key is missing",
        )

    _, answer_rows = _read_csv_rows(answer_path)
    review_fields, review_rows = _read_csv_rows(review_path)
    answers = {
        (row.get("case", ""), row.get("case_sha256", "")): row for row in answer_rows
    }
    reviews = {
        (row.get("case", ""), row.get("case_sha256", "")): row for row in review_rows
    }
    required_review_fields = {
        "case",
        "case_sha256",
        "preferred_candidate",
        "candidate_A_material_artifact",
        "candidate_B_material_artifact",
        "candidate_A_intrinsic_lines_preserved",
        "candidate_B_intrinsic_lines_preserved",
        "notes",
    }
    if not answers or set(reviews) != set(answers) or not required_review_fields.issubset(review_fields):
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "FAIL",
            threshold="independent reviewer finds no scientifically material regression",
            details="review form does not match the hashed blind cases",
        )

    response_fields = required_review_fields - {"case", "case_sha256", "notes"}
    completed = [
        row for row in review_rows if all(str(row.get(field, "")).strip() for field in response_fields)
    ]
    if len(completed) != len(review_rows):
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "MANUAL",
            threshold="independent reviewer finds no scientifically material regression",
            details=f"blind review progress: {len(completed)}/{len(review_rows)} hashed cases complete",
        )

    invalid = []
    regressions = []
    for row in review_rows:
        case = row["case"]
        preference = row["preferred_candidate"].strip().upper()
        values = {
            field: row[field].strip().upper()
            for field in response_fields
            if field != "preferred_candidate"
        }
        if preference not in {"A", "B", "EQUIVALENT"} or any(
            value not in {"YES", "NO"} for value in values.values()
        ):
            invalid.append(case)
            continue
        answer = answers[(case, row["case_sha256"])]
        gen_candidate = "A" if answer.get("candidate_A") == "PyMolFit" else "B"
        if (
            values[f"candidate_{gen_candidate}_material_artifact"] != "NO"
            or values[f"candidate_{gen_candidate}_intrinsic_lines_preserved"] != "YES"
        ):
            regressions.append(case)
    if invalid:
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "FAIL",
            threshold="independent reviewer finds no scientifically material regression",
            details=f"invalid blind-review values in {invalid}",
        )

    held_fields, held_rows = _read_csv_rows(held_out_path)
    required_held_fields = {
        "reviewer",
        "date",
        "dataset",
        "instrument",
        "molecfit_version",
        "pymolfit_version",
        "decision",
        "intrinsic_lines_preserved",
        "masks_usable",
        "uncertainties_usable",
        "settings_and_notes",
    }
    if not required_held_fields.issubset(held_fields):
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "FAIL",
            threshold="independent reviewer finds no scientifically material regression",
            details="held-out review form has an unsupported schema",
        )
    if not held_rows:
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "MANUAL",
            threshold="independent reviewer finds no scientifically material regression",
            details=f"{len(completed)}/{len(review_rows)} blind cases complete; held-out review pending",
        )

    held = held_rows[-1]
    if any(not str(held.get(field, "")).strip() for field in required_held_fields):
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "MANUAL",
            threshold="independent reviewer finds no scientifically material regression",
            details="latest held-out review is incomplete",
        )
    decision = held["decision"].strip().upper()
    held_assessments = {
        held[field].strip().upper()
        for field in ("intrinsic_lines_preserved", "masks_usable", "uncertainties_usable")
    }
    if decision not in {"PASS", "FAIL"} or not held_assessments.issubset({"YES", "NO"}):
        return ValidationCheck(
            "independent_review",
            "blind experienced-user comparison",
            "FAIL",
            threshold="independent reviewer finds no scientifically material regression",
            details="latest held-out review contains invalid decision values",
        )
    passed = not regressions and decision == "PASS" and held_assessments == {"YES"}
    return ValidationCheck(
        "independent_review",
        "blind experienced-user comparison",
        "PASS" if passed else "FAIL",
        threshold="independent reviewer finds no scientifically material regression",
        details=(
            f"reviewer={held['reviewer']}, held-out={held['dataset']}, "
            f"decision={decision}, blind PyMolFit regressions={regressions}"
        ),
    )


def run(*, skip_molecfit: bool = False) -> ScienceReadinessReport:
    started = time.perf_counter()
    aer_catalog = _ensure_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_lines = LineList.from_table(CACHE_PATH)
    rng = np.random.default_rng(20260713)
    checks: list[ValidationCheck] = []
    rows: list[dict[str, object]] = []

    synthetic_checks, synthetic_rows = _synthetic_recovery_checks(all_lines, rng)
    checks.extend(synthetic_checks)
    rows.extend(synthetic_rows)

    uncertainty_check, uncertainty_row = _uncertainty_coverage_check(rng)
    checks.append(uncertainty_check)
    rows.append(uncertainty_row)

    systematics_check, systematics_row = _model_systematics_check(rng)
    checks.append(systematics_check)
    rows.append(systematics_row)

    shared_uncertainty_checks, shared_uncertainty_row = _shared_uncertainty_and_output_checks(rng)
    checks.extend(shared_uncertainty_checks)
    rows.append(shared_uncertainty_row)

    line_preservation_check, line_preservation_row = _intrinsic_line_preservation_check(rng)
    checks.append(line_preservation_check)
    rows.append(line_preservation_row)

    instrument_check, instrument_row = _instrument_parameter_recovery_check(rng)
    checks.append(instrument_check)
    rows.append(instrument_row)

    atmosphere_checks, atmosphere_rows = _atmosphere_and_convergence_checks(all_lines)
    checks.extend(atmosphere_checks)
    rows.extend(atmosphere_rows)
    checks.extend(_failure_mode_checks())

    for case in REAL_BANDS:
        case_checks, row = _run_xshooter_case(
            case,
            all_lines,
            run_molecfit=not skip_molecfit,
        )
        checks.extend(case_checks)
        rows.append(row)

    uves_error = None if skip_molecfit else _refresh_uves_reference()
    hires_error = None if skip_molecfit else _refresh_hires_reference()
    kpf_error = None if skip_molecfit else _refresh_kpf_reference()
    external_checks, external_rows = _existing_external_checks(
        include_uves=not skip_molecfit,
        uves_error=uves_error,
        include_hires=not skip_molecfit,
        hires_error=hires_error,
        include_kpf=not skip_molecfit,
        kpf_error=kpf_error,
    )
    checks.extend(external_checks)
    rows.extend(external_rows)

    physics_checks, physics_rows = _external_physics_golden_checks()
    checks.extend(physics_checks)
    rows.extend(physics_rows)

    fixed_rt_checks, fixed_rt_rows = _fixed_rt_parity_checks(
        run_reference=not skip_molecfit,
    )
    checks.extend(fixed_rt_checks)
    rows.extend(fixed_rt_rows)

    instrument_evidence = {
        "HARPS high-resolution optical": ROOT / "local_tests" / "betapic_harps_o2_comparison" / "summary.csv",
        "UVES high-resolution optical official demo": UVES_SUMMARY,
        "Keck HIRES high-resolution optical": HIRES_SUMMARY,
        "Keck KPF high-resolution optical": KPF_SUMMARY,
        "X-shooter medium-resolution optical/NIR": DATA_DIR / XSHOOTER_DATASETS["VIS"][0],
        "CRIRES+ high-resolution infrared": ROOT / "local_tests" / "rho01_molecfit_vs_pymolfit_lband" / "summary.csv",
    }
    for name, path in instrument_evidence.items():
        checks.append(
            ValidationCheck(
                "instrument_coverage",
                name,
                "PASS" if path.exists() else "FAIL",
                threshold="real observed spectrum processed and retained as reproducible evidence",
                details=str(path.relative_to(ROOT)) if path.exists() else f"missing {path}",
            )
        )

    runtime_check, runtime_row = _runtime_check()
    checks.append(runtime_check)
    if runtime_row:
        rows.append(runtime_row)

    package_checks, package_rows = _package_checks()
    checks.extend(package_checks)
    rows.extend(package_rows)

    covered_bands = {str(row.get("band")) for row in rows if row.get("band")}
    for band in ("optical", "J", "H", "K", "L", "M", "N"):
        checks.append(
            ValidationCheck(
                "wavelength_coverage",
                f"{band}-band exercised",
                "PASS" if band in covered_bands else "FAIL",
                details="real observed spectrum" if any(
                    row.get("group") == "xshooter" and row.get("band") == band for row in rows
                ) else "physical synthetic spectrum",
            )
        )

    checks.append(_authenticated_hitran_check())
    _write_review_packet()
    checks.append(_review_packet_archive_check())
    checks.append(_independent_review_check())

    elapsed = time.perf_counter() - started
    source_hash = hashlib.sha256(CACHE_PATH.read_bytes()).hexdigest()
    dataset_hashes = {
        filename: hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest()
        for filename, _ in XSHOOTER_DATASETS.values()
    }
    dataset_hashes[UVES_DATASET[0]] = hashlib.sha256(
        (UVES_DATA_DIR / UVES_DATASET[0]).read_bytes()
    ).hexdigest()
    for relative in HIRES_ORDER_FILES:
        path = HIRES_DATA_DIR / relative
        dataset_hashes[f"KOA/{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    kpf_path = KPF_DATA_DIR / KPF_FILENAME
    dataset_hashes[f"KOA/{KPF_FILENAME}"] = hashlib.sha256(kpf_path.read_bytes()).hexdigest()
    wheels = sorted((ROOT / "dist").glob("pymolfit-*.whl"), key=lambda path: path.stat().st_mtime)
    wheel_sha256 = hashlib.sha256(wheels[-1].read_bytes()).hexdigest() if wheels else None
    report = ScienceReadinessReport.create(
        checks,
        metadata={
            "campaign_version": 12,
            "elapsed_seconds": elapsed,
            "line_cache": str(CACHE_PATH),
            "line_cache_sha256": source_hash,
            "line_catalog_version": aer_catalog.manifest["catalog_version"],
            "line_catalog_sha256": aer_catalog.manifest["catalog_sha256"],
            "line_catalog_source_page": aer_catalog.manifest["source_page"],
            "line_catalog_source_archive_sha256": aer_catalog.manifest[
                "source_archive_sha256"
            ],
            "dataset_sha256": dataset_hashes,
            "wheel_sha256": wheel_sha256,
            "xshooter_target": "HD 53123",
            "keck_hires_target": "BD+17 3248",
            "keck_hires_koaid": HIRES_KOAID,
            "keck_kpf_target": "Vega (HR 7001)",
            "keck_kpf_koaid": KPF_KOAID,
            "molecfit_executable": str(MOLECFIT_ESOREX),
            "molecfit_skipped": skip_molecfit,
            "pymolfit_version": __version__,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "astropy_version": astropy.__version__,
            "platform": platform.platform(),
        },
    )
    report.write(OUTPUT_DIR)
    _write_metrics(rows)
    _write_summary_plot(report, rows)
    print(f"verdict: {report.verdict}")
    print(f"status counts: {report.status_counts()}")
    print(f"elapsed: {elapsed:.2f} s")
    print(f"wrote: {OUTPUT_DIR}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PyMolFit science-readiness campaign.")
    parser.add_argument(
        "--skip-molecfit",
        action="store_true",
        help="run self-contained checks but mark new Molecfit comparisons skipped",
    )
    args = parser.parse_args()
    run(skip_molecfit=args.skip_molecfit)


if __name__ == "__main__":
    main()
