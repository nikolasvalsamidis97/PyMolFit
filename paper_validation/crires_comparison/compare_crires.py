"""Compare every CRIRES+ detector/order with independent atmospheric fits.

Run directly to fit missing products and display the comparison. The matched
protocol uses weighted least squares for both engines, excludes 30 pixels at
each edge and the broad 4050--4056 nm source feature from fitting, and selects
each engine's best fit from three initial Gaussian widths. Native data are
retained. Updated fits use their own cache; subsequent runs reuse saved fits.
The default figure shows both transmissions and PyMolFit/Molecfit - 1 within
3740--3830 and 4007.7--4103 nm; full-coverage products/statistics are retained.
--refit repeats the atmospheric fits; --plot-only never runs a correction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "_paper_full_comparison", HERE.parent / "beta_pic_full/compare_full.py"
)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

DEFAULT_INPUT = (
    REPO.parent
    / "Alexandra_help_folder/correct_function_err"
    / "cr2res_obs_nodding_extracted_combined.fits"
)
DEFAULT_DATA = REPO / "validation_results/crires_full_matched_continua"
DEFAULT_FIT_EDGE_PIXELS = 30
PROTOCOL = "crires_all_orders_v2"
MATCHED_PROTOCOL = "crires_matched_least_squares_continua_v2"
WIDTH_START_FACTORS = (1.0, 0.7, 1.3)
SOURCE_EXCLUSIONS_NM = ((4050.0, 4056.0),)
# Match the left window's approximately 7.1/8.4 nm leading/trailing margins.
DISPLAY_RANGES_NM = ((3740.0, 3830.0), (4007.7, 4103.0))
HOLDOUT_FRACTIONS = ((0.25, 0.30), (0.55, 0.60), (0.80, 0.85))
RESOLVING_POWER = 86000.0


def discover_orders(input_path):
    """Inventory all extracted wavelength/flux/error triplets, without a whitelist."""
    orders = {}
    with fits.open(input_path) as hdus:
        for detector, hdu in enumerate(hdus[1:], start=1):
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            for name in hdu.columns.names:
                if not name.endswith("_WL"):
                    continue
                prefix = name[:-3]
                if any(prefix + suffix not in hdu.columns.names for suffix in ("_SPEC", "_ERR")):
                    raise ValueError(f"Missing flux/error column for detector {detector}: {prefix}")
                wave = np.asarray(hdu.data[name], float).ravel()
                finite = np.isfinite(wave)
                if not finite.any():
                    raise ValueError(f"No wavelength coverage for detector {detector}: {prefix}")
                orders[f"chip{detector}_{prefix}"] = (
                    detector,
                    prefix,
                    (float(wave[finite].min()), float(wave[finite].max())),
                )
    if not orders:
        raise ValueError("No extracted detector/order spectra found")
    return dict(sorted(orders.items(), key=lambda item: item[1][2][0]))


def heldout_intervals(wavelength_nm):
    lo, hi = float(wavelength_nm[0]), float(wavelength_nm[-1])
    return [(lo + a * (hi - lo), lo + b * (hi - lo)) for a, b in HOLDOUT_FRACTIONS]


def edge_intervals(wavelength_nm, pixels):
    """Return fit-only edge exclusions in micron, without removing native data."""
    wave = np.asarray(wavelength_nm, float) / 1000
    if pixels < 0 or 2 * pixels >= len(wave):
        raise ValueError("Edge exclusion must leave interior pixels")
    if pixels == 0:
        return []
    return [
        [float(wave[0]), float((wave[pixels - 1] + wave[pixels]) / 2)],
        [float((wave[-pixels - 1] + wave[-pixels]) / 2), float(wave[-1])],
    ]


def molecular_policy(result):
    return full.molecular_policy(result)


def prepare(input_path, output, order, fit_edge_pixels=0, *, matched=False):
    detector, prefix, _ = discover_orders(input_path)[order]
    with fits.open(input_path) as hdus:
        header = hdus[0].header.copy()
        data = hdus[detector].data
        wave = np.asarray(data[prefix + "_WL"], dtype=float).copy()
        flux = np.asarray(data[prefix + "_SPEC"], dtype=float).copy()
        error = np.asarray(data[prefix + "_ERR"], dtype=float).copy()
    selected = np.isfinite(wave)
    wave, flux, error = wave[selected], flux[selected], error[selected]
    if len(wave) < 100 or np.any(np.diff(wave) <= 0):
        raise ValueError("The selected order must have increasing, well-sampled wavelengths")
    if header.get("INSTRUME") != "CRIRES" or header.get("ESO INS SLIT1 WID") != 0.2:
        raise ValueError("This protocol requires the CRIRES+ 0.2-arcsec extraction")
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    scale = float(np.median(flux[valid]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("No positive common flux scale")
    flux, error = flux / scale, error / scale
    original_flux, original_error = flux.copy(), error.copy()
    # Keep native wavelengths. Masked finite placeholders are needed by CPL;
    # no valid measurement or uncertainty is changed by this conversion.
    flux = np.where(valid, flux, 0.0)
    error = np.where(valid, error, 1.0)
    heldout = heldout_intervals(wave)
    holdout_mask = valid & full.interval_mask(wave, heldout)
    edges = edge_intervals(wave, fit_edge_pixels)
    source = [
        [max(a, float(wave[0])) / 1000, min(b, float(wave[-1])) / 1000]
        for a, b in SOURCE_EXCLUSIONS_NM
        if matched and a < wave[-1] and b > wave[0]
    ]
    source_mask = full.interval_mask(wave / 1000, source)
    excluded = [(a / 1000, b / 1000) for a, b in heldout] + edges + source
    fit_mask = valid & ~full.interval_mask(wave / 1000, excluded)
    atmosphere = full.AtmosphereProfile.from_fits_header_mipas_gdas(
        header, reference_wavenumber_cm=1e7 / np.median(wave)
    )
    if "gdas_profile" not in atmosphere.metadata:
        raise RuntimeError("A time-local GDAS profile is required for this comparison")
    shutil.copy2(atmosphere.metadata["gdas_profile"], output / "gdas.fits")
    # CR2RES extraction wavelengths are topocentric vacuum nm. R is a nominal
    # starting estimate for this slit, not a resolution measured in the header.
    header["SPECSYS"] = "TOPOCENT"
    header["AIRORVAC"] = "vacuum"
    header["SPEC_RES"] = RESOLVING_POWER
    header["AIRMASS"] = atmosphere.metadata["airmass"]
    header["HIERARCH ESO TEL ALT"] = float(np.degrees(np.arcsin(1 / header["AIRMASS"])))
    columns = [
        fits.Column(name="lambda", format="D", unit="um", array=wave / 1000),
        fits.Column(name="flux", format="D", array=flux),
        fits.Column(name="dflux", format="D", array=error),
        fits.Column(name="mask", format="J", array=(~valid).astype(int)),
        fits.Column(name="valid", format="J", array=valid.astype(int)),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns, header=header)]
    ).writeto(output / "input.fits", overwrite=True)
    Table(
        {
            "wavelength_nm": wave,
            "flux": flux,
            "uncertainty": error,
            "valid": valid,
            "fit": fit_mask,
            "holdout": holdout_mask,
            "source_excluded": source_mask,
            "original_flux": original_flux,
            "original_uncertainty": original_error,
        }
    ).write(output / "input.ecsv", overwrite=True)
    provenance = {
        "comparison_protocol": MATCHED_PROTOCOL if matched else PROTOCOL,
        "input": str(input_path),
        "input_sha256": full.sha256(input_path),
        "object_header": header.get("OBJECT"),
        "date_obs": header["DATE-OBS"],
        "instrument": "CRIRES+",
        "detector_hdu": detector,
        "column_prefix": prefix,
        "range_nm": [float(wave[0]), float(wave[-1])],
        "fit_windows_nm": [[float(wave[0]), float(wave[-1])]],
        "heldout_nm": heldout,
        "excluded_ranges_micron": excluded,
        "fit_edge_pixels": fit_edge_pixels,
        "fit_edge_intervals_micron": edges,
        "source_exclusions_micron": source,
        "constant_flux_scale": scale,
        "median_input_snr": float(np.median(flux[valid] / error[valid])),
        "fit_pixels": int(fit_mask.sum()),
        "holdout_pixels": int(holdout_mask.sum()),
        "atmosphere": atmosphere.metadata,
        "gdas_sha256": full.sha256(output / "gdas.fits"),
        "wavelength_medium": "vacuum",
        "wavelength_frame": "observatory",
        "initial_resolving_power": RESOLVING_POWER,
        "resolution_source": "Nominal lower-limit R for the 0.2-arcsec slit; Gaussian width fitted independently by both engines",
        "resolution_reference": "https://www.eso.org/sci/facilities/paranal/instruments/crires/overview.html",
        "selection": f"Every extracted detector/order; full finite wavelength coverage retained without data or S/N cuts; {fit_edge_pixels} native pixels per edge excluded from fitting only; three fractional wavelength blocks withheld from both fits",
        "native_pixels": len(selected),
        "nonfinite_wavelength_pixels": int((~selected).sum()),
        "invalid_pixels": int((~valid).sum()),
        "invalid_values": "Original normalised values retained in input.ecsv; finite placeholders supplied identically to both engines, always masked",
        "reference": "No assumed stellar/planetary template or absolute correction-accuracy claim",
        "script_sha256": full.sha256(__file__),
        "backend_helper_sha256": full.sha256(full.__file__),
        "source_hashes": {
            str(p.relative_to(REPO)): full.sha256(p)
            for p in sorted((REPO / "src/pymolfit").glob("*.py"))
        },
    }
    return header, provenance


def run_order(
    input_path,
    output,
    order,
    fit_edge_pixels=0,
    *,
    matched=False,
    width_factor=1.0,
    frozen_policy=None,
):
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.ecsv").unlink(missing_ok=True)
    header, provenance = prepare(input_path, output, order, fit_edge_pixels, matched=matched)
    spectrum = full.load_spectrum(
        output / "input.fits",
        wavelength_col="lambda",
        flux_col="flux",
        uncertainty_col="dflux",
        wavelength_unit="micron",
        wavelength_medium="vacuum",
        save_header=False,
    )
    fwhm = (
        width_factor
        * np.mean(spectrum.wavelength)
        / RESOLVING_POWER
        / np.median(np.diff(spectrum.wavelength))
    )
    settings = {
        "wavelength_unit": "micron",
        "wavelength_medium": "vacuum",
        "wavelength_col": "lambda",
        "flux_col": "flux",
        "uncertainty_col": "dflux",
        "gdas_profile": str(output / "gdas.fits"),
        "min_line_strength": 1e-32,
        "minimum_species_peak_optical_depth": full.SPECIES_PEAK_THRESHOLD,
        "auto_segment": False,
        "automatic_fit_regions": False,
        "fit_ranges": [(spectrum.wavelength[0], spectrum.wavelength[-1])],
        "exclude_ranges": provenance["excluded_ranges_micron"],
        "continuum_order": 2,
        "solve_continuum_linear": True,
        "fit_wavelength_shift": False,
        "fit_wavelength_polynomial": True,
        "wavelength_polynomial_order": 2,
        "wavelength_shift_bounds": (-0.0002, 0.0002),
        "lsf_sigma_pixels": float(fwhm / np.sqrt(8 * np.log(2))),
        "fit_lsf_sigma": True,
        "lsf_variable_width": True,
        "lsf_reference_wavelength_micron": float(np.mean(spectrum.wavelength)),
        "lsf_lorentz_fwhm_pixels": 0.0,
        "fit_lsf_lorentz_fwhm": False,
        "loss": "linear" if matched else "soft_l1",
        "f_scale": 1.0,
        "min_transmission": 0.01,
        "high_resolution_grid": True,
        "high_resolution_oversampling": 5,
        **full.CONTINUUM_SETTINGS,
        "report": True,
    }
    if frozen_policy is not None:
        settings["minimum_species_peak_optical_depth"] = 0.0
        settings["fixed_species_scales"] = {
            ("N2_continuum" if name == "N2" else name): value
            for name, value in frozen_policy["fixed"].items()
        }
    provenance["initial_width_factor"] = width_factor
    provenance["pymolfit_settings"] = settings
    path = output / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2))
    print(f"Fitting PyMolFit: {order}", flush=True)
    start = time.perf_counter()
    result = full.correct_file(
        output / "input.fits", output_path=output / "pymolfit.ecsv", **settings
    )
    provenance.update(
        pymolfit_seconds=time.perf_counter() - start,
        pymolfit_success=bool(result.success),
        pymolfit_message=result.message,
    )
    path.write_text(json.dumps(provenance, indent=2))
    if not result.success:
        raise RuntimeError(f"PyMolFit did not converge: {result.message}")
    policy = molecular_policy(result)
    if frozen_policy is not None and any(
        policy[key] != frozen_policy[key] for key in ("included", "fitted", "fixed")
    ):
        raise ValueError("The molecular policy changed between starting points")
    provenance["species_policy"] = policy
    path.write_text(json.dumps(provenance, indent=2))
    options = {
        "LIST_MOLEC": ",".join(policy["included"]),
        "FIT_MOLEC": ",".join("1" if s in policy["fitted"] else "0" for s in policy["included"]),
        "REL_COL": ",".join("1" for _ in policy["included"]),
        "WLC_N": 2,
        "VARKERN": "TRUE",
        "COLUMN_MASK": "valid",
        "SLIT_WIDTH_VALUE": 0.2,
        "RES_GAUSS": float(fwhm),
    }
    records, catalog_sha = full.fit_molecfit(
        output, spectrum, header, provenance["excluded_ranges_micron"], model_options=options
    )
    provenance.update(
        molecfit_runs=records,
        aer_sha256=catalog_sha,
        shared_input_sha256=full.sha256(output / "input.fits"),
    )
    path.write_text(json.dumps(provenance, indent=2))
    full.collect_results(output)


def fit_quality(table):
    """Score observed flux on fixed pixels, independent of either transmission."""
    valid = np.asarray(table["valid"], bool)
    source = (
        np.asarray(table["source_excluded"], bool)
        if "source_excluded" in table.colnames
        else np.zeros(len(table), bool)
    )
    subsets = {
        "fit": valid & np.asarray(table["fit"], bool),
        "heldout": valid & np.asarray(table["holdout"], bool) & ~source,
        "heldout_including_source": valid & np.asarray(table["holdout"], bool),
        "all": valid,
    }
    quality = {}
    for name in ("pymolfit", "molecfit"):
        residual = np.asarray((table["flux"] - table[f"{name}_model"]) / table["uncertainty"])
        if not np.all(np.isfinite(residual[valid])):
            raise ValueError(f"Nonfinite {name} forward-model residuals")
        quality[name] = {
            key: {
                "pixels": int(mask.sum()),
                "chi2": float(np.sum(residual[mask] ** 2)),
                "rms_sigma": float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else None,
            }
            for key, mask in subsets.items()
        }
    return quality


def run_matched_order(input_path, output, order, fit_edge_pixels):
    """Give each engine the same starts and select by its own training chi-square."""
    candidates, failures, policy = [], [], None
    for factor in WIDTH_START_FACTORS:
        directory = output / "starts" / f"width_{factor:g}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (
                (directory / "run.log").open("w") as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                run_order(
                    input_path,
                    directory,
                    order,
                    fit_edge_pixels,
                    matched=True,
                    width_factor=factor,
                    frozen_policy=policy,
                )
            table, provenance = load_comparison(directory)
            if policy is None:
                policy = provenance["species_policy"]
            candidates.append(
                {"factor": factor, "quality": fit_quality(table), "directory": str(directory)}
            )
            print(f"{order}: completed width start {factor:g}", flush=True)
        except Exception as exc:
            failures.append({"factor": factor, "error": str(exc)})
            print(f"{order}: width start {factor:g} failed: {exc}", flush=True)
    if not candidates:
        raise RuntimeError(f"Every matched starting point failed: {failures}")
    winners = {
        name: min(candidates, key=lambda item: item["quality"][name]["fit"]["chi2"])
        for name in ("pymolfit", "molecfit")
    }
    py_dir = Path(winners["pymolfit"]["directory"])
    mf_dir = Path(winners["molecfit"]["directory"])
    py_record = json.loads((py_dir / "provenance.json").read_text())
    mf_record = json.loads((mf_dir / "provenance.json").read_text())
    if py_record["shared_input_sha256"] != mf_record["shared_input_sha256"]:
        raise ValueError("Selected engines received different inputs")
    for name in ("input.fits", "input.ecsv", "gdas.fits", "pymolfit.ecsv"):
        shutil.copy2(py_dir / name, output / name)
    shutil.copytree(mf_dir / "molecfit", output / "molecfit", dirs_exist_ok=True)
    py_record["molecfit_runs"] = mf_record["molecfit_runs"]
    py_record["multistart"] = {
        "selection": "Lowest training chi-square independently for each engine; withheld pixels and transmission agreement are not selection criteria",
        "width_factors": list(WIDTH_START_FACTORS),
        "candidates": candidates,
        "failures": failures,
        "selected_width_factors": {name: item["factor"] for name, item in winners.items()},
        "objective": "sum(((observed_flux - forward_model_flux) / supplied_uncertainty)**2)",
    }
    (output / "provenance.json").write_text(json.dumps(py_record, indent=2))
    table, _ = load_comparison(output)
    (output / "fit_quality.json").write_text(json.dumps(fit_quality(table), indent=2))


def run_order_job(input_path, output, order, fit_edge_pixels=0, fit_protocol="legacy"):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        if fit_protocol == "matched":
            run_matched_order(input_path, output, order, fit_edge_pixels)
        else:
            run_order(input_path, output, order, fit_edge_pixels)


def ensure_corrections(
    input_path,
    data_dir,
    orders,
    *,
    refit=False,
    plot_only=False,
    jobs=2,
    fit_edge_pixels=0,
    fit_protocol="legacy",
):
    """Complete every order or fail explicitly; never plot a successful subset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    source_sha = full.sha256(input_path)
    protocol = MATCHED_PROTOCOL if fit_protocol == "matched" else PROTOCOL
    inventory = {
        "input_sha256": source_sha,
        "comparison_protocol": protocol,
        "fit_edge_pixels": fit_edge_pixels,
        "orders": {},
    }
    pending = []
    for order, (detector, prefix, limits) in orders.items():
        output = data_dir / order
        complete = (output / "comparison.ecsv").exists() and (output / "provenance.json").exists()
        if complete:
            saved = json.loads((output / "provenance.json").read_text())
            if (
                saved["input_sha256"] != source_sha
                or saved["comparison_protocol"] != protocol
                or saved.get("fit_edge_pixels", 0) != fit_edge_pixels
            ):
                if not refit:
                    raise ValueError(
                        f"Incompatible saved data/settings for {order}; "
                        "choose another --data-dir to preserve originals, or use --refit"
                    )
                complete = False
        inventory["orders"][order] = {
            "detector_hdu": detector,
            "column_prefix": prefix,
            "range_nm": limits,
            "status": "complete" if complete and not refit else "pending",
        }
        if refit or not complete:
            pending.append(order)
    path = data_dir / "coverage.json"
    path.write_text(json.dumps(inventory, indent=2))
    print(
        f"Saved fits: {len(orders) - len(pending)}/{len(orders)} compatible orders; "
        f"{len(pending)} require fitting.",
        flush=True,
    )
    if pending and plot_only:
        raise FileNotFoundError(
            f"Missing {len(pending)} of {len(orders)} orders: {', '.join(pending)}"
        )
    if pending:
        print(f"Fitting {len(pending)} orders; {len(orders)} total in the input file.", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    run_order_job,
                    input_path,
                    data_dir / order,
                    order,
                    fit_edge_pixels,
                    fit_protocol,
                ): order
                for order in pending
            }
            for future in as_completed(futures):
                order = futures[future]
                try:
                    future.result()
                    inventory["orders"][order]["status"] = "complete"
                    print(f"Completed {order}", flush=True)
                except Exception as exc:
                    inventory["orders"][order].update(status="failed", error=str(exc))
                    print(f"FAILED {order}: {exc}; see {data_dir / order / 'run.log'}", flush=True)
                path.write_text(json.dumps(inventory, indent=2))
    failed = [
        name for name, record in inventory["orders"].items() if record["status"] != "complete"
    ]
    if failed:
        raise RuntimeError(f"Full-spectrum comparison incomplete: {', '.join(failed)}")
    return inventory


def load_comparison(output):
    provenance = json.loads((output / "provenance.json").read_text())
    if provenance["comparison_protocol"] not in (PROTOCOL, MATCHED_PROTOCOL):
        raise ValueError("Saved fits use another protocol; use --refit")
    if full.sha256(output / "input.fits") != provenance["shared_input_sha256"]:
        raise ValueError("The common input changed after fitting")
    # Recheck identical grids, flux, errors, masks, convergence and F_corrected=F/T.
    table = full.collect_results(output)
    py = Table.read(output / "pymolfit.ecsv")
    mf = Table.read(output / "molecfit/BEST_FIT_MODEL.fits", hdu=1)
    if len(mf) != len(table) or not np.allclose(mf["lambda"] * 1000, table["wavelength_nm"]):
        raise ValueError("Molecfit forward model must cover this whole order")
    table["pymolfit_model"] = py["model_flux"]
    table["molecfit_model"] = mf["mflux"]
    molecules = Table.read(output / "molecfit/MODEL_MOLECULES.fits", hdu=1)
    policy = provenance["species_policy"]
    mf_policy = {str(r["LIST_MOLEC"]).strip(): bool(r["FIT_MOLEC"]) for r in molecules}
    if mf_policy != {s: s in policy["fitted"] for s in policy["included"]}:
        raise ValueError("Molecular inclusion/fitting policies differ")
    return table, provenance


def combine_order_tables(products):
    """Keep order identity and flux normalisation when collecting native tables."""
    tables = []
    for table, provenance in products:
        tagged = table.copy()
        tagged.meta.clear()
        tagged["detector_order"] = f"chip{provenance['detector_hdu']}_{provenance['column_prefix']}"
        tagged["constant_flux_scale"] = provenance["constant_flux_scale"]
        tables.append(tagged)
    combined = vstack(tables, join_type="exact")
    combined.meta["comparison_protocol"] = PROTOCOL
    combined.meta["sampling"] = "Orders kept separate; no resampling or gap interpolation"
    combined.meta["flux_units"] = "Each order divided by its constant_flux_scale"
    return combined


def configure_wavelength_axes(axes, products, display_ranges=None):
    """Compress large gaps only; adjacent detectors retain their native spacing."""
    from matplotlib.ticker import MaxNLocator

    groups = [] if display_ranges is None else [list(pair) for pair in display_ranges]
    membership = []
    for _, provenance in products:
        lo, hi = provenance["range_nm"]
        if display_ranges is not None:
            membership.append(next(i for i, (a, b) in enumerate(groups) if a <= lo <= hi <= b))
            continue
        if not groups or lo - groups[-1][1] > 10:
            groups.append([lo, hi])
        else:
            groups[-1][1] = max(hi, groups[-1][1])
        membership.append(len(groups) - 1)
    spans = np.array([hi - lo for lo, hi in groups])
    gap_width = (0.05 if display_ranges is not None else 0.025) * spans.sum()
    starts = np.r_[0, np.cumsum(spans[:-1] + gap_width)]
    ticks, tick_labels = [], []

    def format_coord(x, y):
        for (lo, _), start, span in zip(groups, starts, spans, strict=True):
            if start <= x <= start + span:
                wavelength = lo + x - start
                return f"wavelength={wavelength:.3f} nm, y={y:.4g}"
        return f"Wavelength gap, y={y:.4g}"

    for (lo, hi), start in zip(groups, starts, strict=True):
        range_ticks = MaxNLocator(
            nbins=6 if len(groups) <= 2 else 3, steps=[1, 2, 2.5, 5, 10]
        ).tick_values(lo, hi)
        range_ticks = range_ticks[(range_ticks >= lo) & (range_ticks <= hi)]
        ticks.extend(range_ticks - lo + start)
        tick_labels.extend(f"{value:g}" for value in range_ticks)
    for divider in starts[1:] - gap_width / 2:
        for ax in axes:
            ax.axvline(divider, color="0.45", lw=1, ls="--", gid="wavelength-gap")
            ax.plot(
                [divider - gap_width / 8, divider + gap_width / 8],
                [0, 0],
                transform=ax.get_xaxis_transform(),
                linestyle="none",
                marker=[(-1, -0.5), (1, 0.5)],
                markersize=9,
                markeredgewidth=1.3,
                color="0.2",
                clip_on=False,
                zorder=5,
                gid="axis-break",
            )
    axes[-1].set_xticks(ticks, tick_labels)
    axes[-1].set_xlabel("Observatory-frame vacuum wavelength [nm]", fontsize=13.4)
    for ax in axes:
        ax.set_xlim(0, starts[-1] + spans[-1])
        ax.format_coord = format_coord
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=11.7)
    return [starts[index] - groups[index][0] for index in membership]


def crop_display_products(products, display_ranges):
    """Crop native samples for display only, keeping separate detector curves."""
    cropped = []
    for lo, hi in display_ranges:
        for table, provenance in products:
            wave = np.asarray(table["wavelength_nm"])
            part = table[(wave >= lo) & (wave <= hi)].copy()
            if len(part):
                record = dict(provenance)
                record["range_nm"] = [
                    float(part["wavelength_nm"][0]),
                    float(part["wavelength_nm"][-1]),
                ]
                cropped.append((part, record))
    if not cropped:
        raise ValueError("No CRIRES+ samples in the requested display ranges")
    return cropped


def make_transmission_figure(
    products, minimum_transmission=full.RATIO_MIN_TRANSMISSION, display_ranges=None
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 4), sharex=True, layout="constrained")
    if display_ranges is not None:
        products = crop_display_products(products, display_ranges)
    offsets = configure_wavelength_axes(axes, products, display_ranges)
    for index, ((table, _), offset) in enumerate(zip(products, offsets, strict=True)):
        full.plot_transmission_curves(
            axes,
            full.compare_transmissions(table, minimum_transmission),
            offset=offset,
            index=index,
        )
    full.style_transmission_axes(axes)
    axes[1].set_ylim(top=0.005)
    fig.suptitle("CRIRES+ | Atmospheric transmission", fontsize=15.7)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=HERE / "crires_comparison.png")
    parser.add_argument(
        "--min-transmission",
        type=float,
        default=full.RATIO_MIN_TRANSMISSION,
        help="Minimum transmission from both engines for the ratio panel only",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refit", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent independent order fits")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument(
        "--fit-edge-pixels",
        type=int,
        default=DEFAULT_FIT_EDGE_PIXELS,
        help="Native pixels at each edge excluded from fitting only (default: 30)",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.fit_edge_pixels < 0:
        parser.error("--fit-edge-pixels must be nonnegative")
    if not np.isfinite(args.min_transmission) or not 0 < args.min_transmission <= 1:
        parser.error("--min-transmission must be finite and in (0, 1]")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must name a PNG file")
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    print(
        f"Fit products: {args.data_dir.resolve()}\n"
        f"Fitting mask: exclude {args.fit_edge_pixels} pixels at each order edge; "
        "native spectra retained. Both engines use the matched protocol.",
        flush=True,
    )
    orders = discover_orders(args.input)
    inventory = ensure_corrections(
        args.input,
        args.data_dir,
        orders,
        refit=args.refit,
        plot_only=args.plot_only,
        jobs=args.jobs,
        fit_edge_pixels=args.fit_edge_pixels,
        fit_protocol="matched",
    )
    if args.fit_only:
        return
    products = []
    for order in orders:
        table, provenance = load_comparison(args.data_dir / order)
        if args.input.exists() and full.sha256(args.input) != provenance["input_sha256"]:
            raise ValueError(
                "Saved fits belong to a different input; use --refit or another --data-dir"
            )
        products.append((table, provenance))
        scored = full.compare_transmissions(table, args.min_transmission)
        inventory["orders"][order].update(
            native_pixels=provenance["native_pixels"],
            wavelength_pixels=len(table),
            valid_input_pixels=int(np.count_nonzero(table["valid"])),
            transmission_pixels=int(np.count_nonzero(scored["transmission_valid"])),
            evaluation_pixels=int(np.count_nonzero(scored["ratio_valid"])),
            median_input_snr=provenance["median_input_snr"],
        )
    inventory["evaluation_view"] = "transmission"
    (args.data_dir / "coverage.json").write_text(json.dumps(inventory, indent=2))
    combined = combine_order_tables(products)
    combined.write(args.data_dir / "comparison.ecsv", overwrite=True)
    full.save_transmission_comparison(args.data_dir, combined, args.min_transmission)
    figure = make_transmission_figure(products, args.min_transmission, DISPLAY_RANGES_NM)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(
        f"Saved {args.output}\nMetrics: {args.data_dir / 'transmission_metrics.json'}",
        flush=True,
    )
    import matplotlib.pyplot as plt

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
