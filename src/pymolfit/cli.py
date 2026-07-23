from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .aer_data import AERDataError, aer_catalog_status, install_aer_catalog
from .io import load_spectrum
from .linelist import LineList
from .line_data import HitranAcquisitionError, cache_hitran_par, fetch_hitran_lines
from .physics import LBLRTM_DEFAULT_AVMASS_AMU
from .validation import compare_spectra
from .workflow import DEFAULT_SEGMENT_SIZE_MICRON, correct_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pymolfit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit and remove telluric absorption")
    fit_parser.add_argument("input", type=Path, help="input spectrum")
    fit_parser.add_argument("output", type=Path, help="output corrected ASCII spectrum")
    fit_parser.add_argument("--product", type=Path, help="optional full fit-product table")
    fit_parser.add_argument("--product-format", default="ascii.ecsv", help="Astropy table format for --product")
    fit_parser.add_argument("--plot", type=Path, help="optional diagnostic plot path")
    fit_parser.add_argument("--line-list", type=Path, help="Astropy-readable line-list table")
    fit_parser.add_argument("--hitran-par", type=Path, help="HITRAN .par line-list file")
    fit_parser.add_argument("--hitran-species", action="append", default=[], help="species to keep from HITRAN .par")
    fit_parser.add_argument("--hitran-min-strength", type=float, help="minimum HITRAN line intensity to keep")
    fit_parser.add_argument("--hitran-max-lines", type=int, help="maximum strongest HITRAN lines to keep")
    fit_parser.add_argument(
        "--aer-catalog",
        type=Path,
        help="exact AER 3.9 catalogue path; otherwise discover/download it automatically",
    )
    fit_parser.add_argument("--aer-cache-dir", type=Path, help="AER catalogue and line-window cache directory")
    fit_parser.add_argument(
        "--aer-source",
        help="AER 3.9 archive URL/path used on cache miss; defaults to the official Zenodo record",
    )
    fit_parser.add_argument(
        "--aer-offline",
        action="store_true",
        help="use only an installed AER catalogue and cached line windows",
    )
    fit_parser.add_argument(
        "--no-auto-aer",
        action="store_true",
        help="disable automatic AER line data; an explicit opacity source is then required",
    )
    fit_parser.add_argument(
        "--no-reuse-local-aer",
        "--no-reuse-molecfit-aer",
        dest="no_reuse_molecfit_aer",
        action="store_true",
        help="do not reuse a verified exact local AER catalogue",
    )
    fit_parser.add_argument("--aer-timeout-s", type=float, default=120.0)
    fit_parser.add_argument(
        "--demo-lines",
        action="store_true",
        help="use the small synthetic demo line list; never use this for scientific data",
    )
    fit_parser.add_argument(
        "--mtckd-h2o",
        type=Path,
        help="AER MT_CKD H2O continuum netCDF coefficient file",
    )
    fit_parser.add_argument(
        "--lblrtm-h2o-continuum",
        action="store_true",
        help="use the packaged LBLRTM 12.11 / MT_CKD 3.5 H2O continuum",
    )
    fit_parser.add_argument(
        "--mtckd-h2o-foreign-closure",
        action="store_true",
        help="use the MT_CKD foreign-closure H2O continuum coefficients",
    )
    fit_parser.add_argument(
        "--co2-continuum",
        type=Path,
        help="tabulated CO2 continuum coefficient table",
    )
    fit_parser.add_argument(
        "--lblrtm-co2-continuum",
        action="store_true",
        help="use the packaged LBLRTM 12.11 CO2 continuum",
    )
    fit_parser.add_argument(
        "--o2-cia",
        type=Path,
        help="HITRAN CIA file for an O2 collision-induced absorption band",
    )
    fit_parser.add_argument(
        "--n2-cia",
        type=Path,
        help="HITRAN CIA file for an N2 collision-induced absorption band",
    )
    fit_parser.add_argument(
        "--cia-table",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="generic HITRAN CIA file, e.g. CO2-H2O_CIA=CO2-H2O_2024.cia; may be repeated",
    )
    fit_parser.add_argument("--format", choices=["ascii", "fits"], help="input spectrum format")
    fit_parser.add_argument("--wavelength-col", default=None, help="wavelength column name or index")
    fit_parser.add_argument("--flux-col", default=None, help="flux column name or index")
    fit_parser.add_argument("--uncertainty-col", default=None, help="uncertainty column name or index")
    fit_parser.add_argument("--wavelength-unit", default="micron", help="input wavelength unit: micron, nm, angstrom")
    fit_parser.add_argument(
        "--wavelength-medium",
        choices=["vacuum", "air"],
        default="vacuum",
        help="whether input wavelengths are vacuum or standard-air wavelengths",
    )
    fit_parser.add_argument("--airmass", type=float, default=1.0)
    fit_parser.add_argument("--continuum-order", type=int, default=1)
    fit_parser.add_argument(
        "--solve-continuum-linear",
        action="store_true",
        help="solve polynomial continuum coefficients exactly at each nonlinear step",
    )
    fit_parser.add_argument("--lsf-sigma-pixels", type=float, default=0.0)
    fit_parser.add_argument("--lsf-box-width-pixels", type=float, default=0.0)
    fit_parser.add_argument("--lsf-lorentz-fwhm-pixels", type=float, default=0.0)
    fit_parser.add_argument(
        "--lsf-variable-width",
        action="store_true",
        help="scale LSF widths by wavelength/reference wavelength",
    )
    fit_parser.add_argument(
        "--lsf-reference-wavelength-micron",
        type=float,
        help="reference wavelength for --lsf-variable-width; defaults to segment median",
    )
    fit_parser.add_argument(
        "--lsf-kernel-width-fwhm",
        type=float,
        default=3.0,
        help="kernel support in units of FWHM for Gaussian/Lorentz LSF components",
    )
    fit_parser.add_argument(
        "--lsf-molecfit-voigt",
        action="store_true",
        help="use Molecfit's synthetic Voigt approximation for the instrumental Gaussian+Lorentzian kernel",
    )
    fit_parser.add_argument(
        "--high-resolution-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute telluric transmission on an oversampled internal wavenumber grid before convolution/rebinning",
    )
    fit_parser.add_argument(
        "--high-resolution-oversampling",
        type=float,
        default=5.0,
        help="number of internal wavenumber samples per observed pixel for --high-resolution-grid",
    )
    fit_parser.add_argument(
        "--high-resolution-margin-pixels",
        type=float,
        default=2.0,
        help="extra internal-grid margin around the observed segment in observed-pixel units",
    )
    fit_parser.add_argument(
        "--high-resolution-rebin-mode",
        choices=["integrate", "center", "sample_average", "molecfit_overlap", "molecfit_average"],
        default="molecfit_overlap",
        help="rebin convention for high-resolution models: integrate pixel bins or sample pixel centers",
    )
    fit_parser.add_argument(
        "--radiative-transfer-grid",
        choices=["auto", "model"],
        default="auto",
        help="evaluate opacity on an LBLRTM layer-resolved native grid or directly on the model grid",
    )
    fit_parser.add_argument(
        "--radiative-transfer-step-cm",
        type=float,
        help="explicit native radiative-transfer wavenumber step in cm^-1",
    )
    fit_parser.add_argument(
        "--radiative-transfer-max-points",
        type=int,
        default=2_000_000,
        help="maximum native radiative-transfer grid size before requiring spectral segmentation",
    )
    fit_parser.add_argument(
        "--auto-segment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically split broad 1D spectra and fit shared molecular columns",
    )
    fit_parser.add_argument(
        "--segment-size",
        type=float,
        default=DEFAULT_SEGMENT_SIZE_MICRON,
        metavar="MICRON",
        help="maximum automatic segment width in microns (default: 0.01 = 100 Angstrom)",
    )
    fit_parser.add_argument(
        "--line-cutoff-cm",
        type=float,
        help="finite Voigt wing cutoff in cm^-1, e.g. 25 for an LBLRTM-like calculation window",
    )
    fit_parser.add_argument(
        "--line-wing-mode",
        choices=[
            "full",
            "hard_cutoff",
            "subtracted_cutoff",
            "tapered_cutoff",
            "lblrtm_subtracted",
            "lblrtm_dynamic",
            "lblrtm_table",
            "lblrtm_panel",
        ],
        default="lblrtm_panel",
        help="line-wing treatment; lblrtm_panel is the source-parity default",
    )
    fit_parser.add_argument(
        "--subtract-cutoff-profile",
        action="store_true",
        help="subtract the Voigt value at --line-cutoff-cm before truncating each line wing",
    )
    fit_parser.add_argument("--line-taper-cm", type=float, default=0.0, help="cosine taper width at the line cutoff")
    fit_parser.add_argument("--lblrtm-sample", type=float, default=4.0, help="LBLRTM SAMPLE control for LBLRTM line-wing modes")
    fit_parser.add_argument(
        "--lblrtm-alfal0",
        type=float,
        default=0.04,
        help="LBLRTM ALFAL0 control for LBLRTM line-wing modes; 0 disables the finite ALFMAX cap",
    )
    fit_parser.add_argument(
        "--lblrtm-avmass-amu",
        type=float,
        default=LBLRTM_DEFAULT_AVMASS_AMU,
        help="LBLRTM representative molecular mass used to construct layer sampling grids",
    )
    fit_parser.add_argument("--lblrtm-hwf3", type=float, default=64.0, help="LBLRTM outer Voigt domain in half-widths")
    fit_parser.add_argument("--rayleigh", action="store_true", help="include the LBLRTM contnm.f90 Rayleigh scattering branch")
    fit_parser.add_argument("--rayleigh-xrayl", type=float, default=1.0, help="LBLRTM Rayleigh scale factor")
    fit_parser.add_argument(
        "--n2-continuum",
        action="store_true",
        help="include the LBLRTM N2 pure-rotation, fundamental, and first-overtone branches",
    )
    fit_parser.add_argument("--n2-continuum-xn2cn", type=float, default=1.0, help="LBLRTM N2 continuum scale factor")
    fit_parser.add_argument(
        "--o2-continuum",
        action="store_true",
        help="include the source-backed LBLRTM ground-based O2 continuum branches",
    )
    fit_parser.add_argument("--o2-continuum-xo2cn", type=float, default=1.0, help="LBLRTM O2 continuum scale factor")
    fit_parser.add_argument("--line-margin-micron", type=float, default=0.01, help="minimum extra wavelength margin for selected lines")
    fit_parser.add_argument(
        "--min-transmission",
        type=float,
        default=0.03,
        help="mask corrected pixels whose fitted atmospheric transmission is below this fraction",
    )
    fit_parser.add_argument("--fit-wavelength-shift", action="store_true", help="fit a constant wavelength shift")
    fit_parser.add_argument(
        "--fit-wavelength-polynomial",
        action="store_true",
        help="fit one wavelength-correction polynomial over the full spectrum",
    )
    fit_parser.add_argument(
        "--wavelength-polynomial-order",
        type=int,
        default=1,
        help="order of --fit-wavelength-polynomial in a normalized global coordinate",
    )
    fit_parser.add_argument("--initial-wavelength-shift", type=float, default=0.0)
    fit_parser.add_argument("--wavelength-shift-bounds", nargs=2, type=float, default=(-5.0e-4, 5.0e-4))
    fit_parser.add_argument("--fit-lsf-sigma", action="store_true", help="fit Gaussian LSF sigma in pixels")
    fit_parser.add_argument("--lsf-sigma-bounds", nargs=2, type=float, default=(0.0, 5.0))
    fit_parser.add_argument("--fit-lsf-box-width", action="store_true", help="fit boxcar LSF width in pixels")
    fit_parser.add_argument("--lsf-box-width-bounds", nargs=2, type=float, default=(0.0, 10.0))
    fit_parser.add_argument("--fit-lsf-lorentz-fwhm", action="store_true", help="fit Lorentzian LSF FWHM in pixels")
    fit_parser.add_argument("--lsf-lorentz-fwhm-bounds", nargs=2, type=float, default=(0.0, 10.0))
    fit_parser.add_argument("--fit-range", action="append", default=[], metavar="START:STOP")
    fit_parser.add_argument("--exclude-range", action="append", default=[], metavar="START:STOP")
    fit_parser.add_argument("--loss", default="linear", help="scipy least_squares loss, e.g. linear or soft_l1")
    fit_parser.add_argument("--f-scale", type=float, default=1.0)
    fit_parser.add_argument("--ftol", type=float, default=1.0e-10, help="relative cost convergence tolerance")
    fit_parser.add_argument("--xtol", type=float, default=1.0e-10, help="relative parameter convergence tolerance")
    fit_parser.add_argument("--gtol", type=float, default=1.0e-10, help="gradient convergence tolerance")
    fit_parser.add_argument(
        "--estimate-uncertainties",
        action="store_true",
        help="estimate local parameter/transmission errors and propagate them to corrected flux",
    )
    fit_parser.add_argument("--physical", action="store_true", help="use self-contained HITRAN atmosphere physics")
    fit_parser.add_argument(
        "--atmosphere",
        choices=["mipas_gdas", "single", "standard"],
        default="mipas_gdas",
        help="atmosphere builder for physical HITRAN models",
    )
    fit_parser.add_argument(
        "--mipas-profile",
        choices=["equ", "std", "tro", "auto"],
        default="equ",
        help="MIPAS climatology profile for --atmosphere=mipas_gdas",
    )
    fit_parser.add_argument(
        "--gdas-profile",
        type=Path,
        help="optional GDAS FITS profile with press/height/temp/relhum columns; otherwise use bundled monthly averages",
    )
    fit_parser.add_argument(
        "--gdas-mode",
        choices=["auto", "online", "cache", "average"],
        default="auto",
        help="GDAS source for MIPAS+GDAS: auto downloads/caches exact profiles then falls back to averages",
    )
    fit_parser.add_argument(
        "--gdas-cache-dir",
        type=Path,
        help="cache directory for downloaded ESO GDAS tarballs and interpolated profiles",
    )
    fit_parser.add_argument(
        "--gdas-timeout-s",
        type=float,
        default=15.0,
        help="per-URL timeout in seconds when downloading ESO GDAS tarballs",
    )
    fit_parser.add_argument(
        "--observatory-latitude-deg",
        type=float,
        help="observatory geodetic latitude in degrees; overrides FITS metadata",
    )
    fit_parser.add_argument(
        "--observatory-longitude-deg",
        type=float,
        help="observatory east-positive longitude in degrees; overrides FITS metadata",
    )
    fit_parser.add_argument(
        "--observatory-altitude-m",
        type=float,
        help="observatory altitude in meters; overrides FITS metadata",
    )
    fit_parser.add_argument(
        "--allow-default-observatory",
        action="store_true",
        help="explicitly allow Paranal geometry when the input has no resolvable site coordinates",
    )
    fit_parser.add_argument("--atmosphere-table", type=Path, help="Astropy-readable atmosphere profile table")
    fit_parser.add_argument("--pressure-atm", type=float, default=0.75)
    fit_parser.add_argument("--temperature-k", type=float, default=280.0)
    fit_parser.add_argument("--path-length-m", type=float, default=8000.0)
    fit_parser.add_argument("--pwv-mm", type=float, help="scale H2O to precipitable water vapor in mm")
    fit_parser.add_argument("--relative-humidity", type=float, help="local relative humidity in percent")
    fit_parser.add_argument("--partition-table", type=Path, help="optional partition-function table")
    fit_parser.add_argument(
        "--mixing-ratio",
        action="append",
        default=[],
        metavar="SPECIES=VALUE",
        help="override a volume mixing ratio, e.g. H2O=0.001",
    )

    convert_parser = subparsers.add_parser("convert-hitran", help="convert HITRAN .par to a line-list table")
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--wavenumber-min", type=float)
    convert_parser.add_argument("--wavenumber-max", type=float)
    convert_parser.add_argument("--species", action="append", default=[])
    convert_parser.add_argument("--min-strength", type=float)
    convert_parser.add_argument("--max-lines", type=int)
    convert_parser.add_argument("--format", default="ascii.ecsv", help="Astropy output table format")

    fetch_parser = subparsers.add_parser(
        "fetch-hitran", help="download and cache an authenticated HITRAN line window"
    )
    fetch_parser.add_argument("--species", action="append", required=True)
    fetch_parser.add_argument("--wavelength-min", type=float, help="vacuum wavelength lower bound in micron")
    fetch_parser.add_argument("--wavelength-max", type=float, help="vacuum wavelength upper bound in micron")
    fetch_parser.add_argument("--wavenumber-min", type=float, help="wavenumber lower bound in cm^-1")
    fetch_parser.add_argument("--wavenumber-max", type=float, help="wavenumber upper bound in cm^-1")
    fetch_parser.add_argument("--api-key-env", default="HITRAN_API_KEY")
    fetch_parser.add_argument("--cache-dir", type=Path)
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.add_argument("--timeout", type=float, default=60.0)

    cache_parser = subparsers.add_parser(
        "cache-hitran", help="validate and cache a locally supplied HITRAN .par window"
    )
    cache_parser.add_argument("input", type=Path)
    cache_parser.add_argument("--species", action="append", required=True)
    cache_parser.add_argument("--wavelength-min", type=float)
    cache_parser.add_argument("--wavelength-max", type=float)
    cache_parser.add_argument("--wavenumber-min", type=float)
    cache_parser.add_argument("--wavenumber-max", type=float)
    cache_parser.add_argument("--cache-dir", type=Path)
    cache_parser.add_argument("--force", action="store_true")

    install_aer_parser = subparsers.add_parser(
        "install-aer",
        help="install and verify the official AER line catalogue",
    )
    install_aer_parser.add_argument(
        "--source",
        help="AER 3.9 archive URL/path; defaults to the official pinned Zenodo record",
    )
    install_aer_parser.add_argument("--source-sha256", help="optional source archive SHA-256")
    install_aer_parser.add_argument("--catalog-path", type=Path, help="existing exact AER catalogue to reuse")
    install_aer_parser.add_argument("--cache-dir", type=Path)
    install_aer_parser.add_argument("--force", action="store_true")
    install_aer_parser.add_argument("--offline", action="store_true")
    install_aer_parser.add_argument(
        "--no-reuse-local-aer",
        "--no-reuse-molecfit",
        dest="no_reuse_molecfit",
        action="store_true",
    )
    install_aer_parser.add_argument("--timeout", type=float, default=120.0)

    aer_status_parser = subparsers.add_parser(
        "aer-status",
        help="report the verified AER catalogue available to PyMolFit",
    )
    aer_status_parser.add_argument("--catalog-path", type=Path)
    aer_status_parser.add_argument("--cache-dir", type=Path)
    aer_status_parser.add_argument(
        "--no-reuse-local-aer",
        "--no-reuse-molecfit",
        dest="no_reuse_molecfit",
        action="store_true",
    )

    compare_parser = subparsers.add_parser("compare", help="compare two spectra on their overlap")
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("--candidate-format", choices=["ascii", "fits"])
    compare_parser.add_argument("--reference-format", choices=["ascii", "fits"])
    compare_parser.add_argument("--candidate-unit", default="micron")
    compare_parser.add_argument("--reference-unit", default="micron")
    compare_parser.add_argument("--normalize", action="store_true")

    return parser


def _column_arg(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _parse_mixing_ratios(values: list[str]) -> dict[str, float]:
    ratios = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"mixing ratio must look like SPECIES=VALUE, got {value!r}")
        species, ratio = value.split("=", 1)
        ratios[species.strip()] = float(ratio)
    return ratios


def _parse_cia_tables(values: list[str]) -> dict[str, Path] | None:
    if not values:
        return None
    tables = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"CIA table must look like NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"CIA table name is empty in {value!r}")
        tables[name] = Path(path)
    return tables


def _parse_ranges(values: list[str]) -> tuple[tuple[float, float], ...] | None:
    if not values:
        return None
    ranges = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"range must look like START:STOP, got {value!r}")
        start, stop = value.split(":", 1)
        ranges.append((float(start), float(stop)))
    return tuple(ranges)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "install-aer":
        try:
            artifact = install_aer_catalog(
                source=args.source,
                source_sha256=args.source_sha256,
                catalog_path=args.catalog_path,
                cache_dir=args.cache_dir,
                force=args.force,
                offline=args.offline,
                reuse_molecfit=not args.no_reuse_molecfit,
                timeout_s=args.timeout,
                progress=print,
            )
        except (AERDataError, FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(f"catalogue: {artifact.catalog_path}")
        print(f"version: {artifact.manifest['catalog_version']}")
        print(f"source: {artifact.source}")
        print(f"managed cache: {artifact.managed}")
        return 0

    if args.command == "aer-status":
        artifact = aer_catalog_status(
            catalog_path=args.catalog_path,
            cache_dir=args.cache_dir,
            reuse_molecfit=not args.no_reuse_molecfit,
        )
        if artifact is None:
            print("AER catalogue: not installed")
            return 1
        print("AER catalogue: ready")
        print(f"catalogue: {artifact.catalog_path}")
        print(f"version: {artifact.manifest['catalog_version']}")
        print(f"source: {artifact.source}")
        print(f"managed cache: {artifact.managed}")
        return 0

    if args.command == "fit":
        if args.fit_wavelength_shift and args.fit_wavelength_polynomial:
            parser.error(
                "use either --fit-wavelength-shift or --fit-wavelength-polynomial, not both"
            )
        if args.hitran_par is not None and args.line_list is not None:
            parser.error("use either --hitran-par or --line-list, not both")
        if args.lblrtm_h2o_continuum and args.mtckd_h2o is not None:
            parser.error("use either --lblrtm-h2o-continuum or --mtckd-h2o, not both")
        if args.lblrtm_co2_continuum and args.co2_continuum is not None:
            parser.error("use either --lblrtm-co2-continuum or --co2-continuum, not both")
        has_opacity_input = any(
            (
                args.line_list is not None,
                args.hitran_par is not None,
                args.demo_lines,
                args.mtckd_h2o is not None,
                args.lblrtm_h2o_continuum,
                args.co2_continuum is not None,
                args.lblrtm_co2_continuum,
                args.o2_cia is not None,
                args.n2_cia is not None,
                bool(args.cia_table),
                args.rayleigh,
                args.n2_continuum,
                args.o2_continuum,
            )
        )
        if not has_opacity_input and args.no_auto_aer:
            parser.error(
                "scientific fitting requires --hitran-par or --line-list; "
                "omit --no-auto-aer to use the managed AER catalogue"
            )
        try:
            mixing_ratios = _parse_mixing_ratios(args.mixing_ratio)
            cia_tables = _parse_cia_tables(args.cia_table)
            fit_ranges = _parse_ranges(args.fit_range)
            exclude_ranges = _parse_ranges(args.exclude_range)
        except ValueError as exc:
            parser.error(str(exc))

        result = correct_file(
            args.input,
            args.output,
            input_format=args.format,
            wavelength_col=_column_arg(args.wavelength_col),
            flux_col=_column_arg(args.flux_col),
            uncertainty_col=_column_arg(args.uncertainty_col),
            wavelength_unit=args.wavelength_unit,
            wavelength_medium=args.wavelength_medium,
            line_list_path=args.line_list,
            hitran_par=args.hitran_par,
            hitran_species=tuple(args.hitran_species) if args.hitran_species else None,
            hitran_min_strength=args.hitran_min_strength,
            hitran_max_lines=args.hitran_max_lines,
            demo_line_list=args.demo_lines,
            aer_catalog=None if args.no_auto_aer else (args.aer_catalog or "auto"),
            aer_cache_dir=args.aer_cache_dir,
            aer_source=args.aer_source,
            aer_offline=args.aer_offline,
            aer_reuse_molecfit=not args.no_reuse_molecfit_aer,
            aer_timeout_s=args.aer_timeout_s,
            partition_table=args.partition_table,
            h2o_continuum="lblrtm" if args.lblrtm_h2o_continuum else args.mtckd_h2o,
            h2o_continuum_foreign_closure=args.mtckd_h2o_foreign_closure,
            co2_continuum="lblrtm" if args.lblrtm_co2_continuum else args.co2_continuum,
            o2_cia=args.o2_cia,
            n2_cia=args.n2_cia,
            cia_tables=cia_tables,
            physical=(
                True
                if args.n2_continuum
                or args.o2_continuum
                or cia_tables
                or args.lblrtm_h2o_continuum
                or args.lblrtm_co2_continuum
                or any(
                    value is not None
                    for value in (args.hitran_par, args.mtckd_h2o, args.co2_continuum, args.o2_cia, args.n2_cia)
                )
                else (True if args.physical else None)
            ),
            atmosphere_table=args.atmosphere_table,
            atmosphere_mode=args.atmosphere,
            mipas_profile=args.mipas_profile,
            gdas_profile=args.gdas_profile,
            gdas_mode=args.gdas_mode,
            gdas_cache_dir=args.gdas_cache_dir,
            gdas_download_timeout_s=args.gdas_timeout_s,
            observatory_latitude_deg=args.observatory_latitude_deg,
            observatory_longitude_deg=args.observatory_longitude_deg,
            observatory_altitude_m=args.observatory_altitude_m,
            allow_default_observatory=args.allow_default_observatory,
            airmass=args.airmass,
            pressure_atm=args.pressure_atm,
            temperature_k=args.temperature_k,
            path_length_m=args.path_length_m,
            pwv_mm=args.pwv_mm,
            relative_humidity_percent=args.relative_humidity,
            mixing_ratios=mixing_ratios if mixing_ratios else None,
            continuum_order=args.continuum_order,
            solve_continuum_linear=args.solve_continuum_linear,
            lsf_sigma_pixels=args.lsf_sigma_pixels,
            lsf_box_width_pixels=args.lsf_box_width_pixels,
            lsf_lorentz_fwhm_pixels=args.lsf_lorentz_fwhm_pixels,
            lsf_variable_width=args.lsf_variable_width,
            lsf_reference_wavelength_micron=args.lsf_reference_wavelength_micron,
            lsf_kernel_width_fwhm=args.lsf_kernel_width_fwhm,
            lsf_molecfit_voigt=args.lsf_molecfit_voigt,
            high_resolution_grid=args.high_resolution_grid,
            high_resolution_oversampling=args.high_resolution_oversampling,
            high_resolution_margin_pixels=args.high_resolution_margin_pixels,
            high_resolution_rebin_mode=args.high_resolution_rebin_mode,
            radiative_transfer_grid=args.radiative_transfer_grid,
            radiative_transfer_step_cm=args.radiative_transfer_step_cm,
            radiative_transfer_max_points=args.radiative_transfer_max_points,
            auto_segment=args.auto_segment,
            segment_size=args.segment_size,
            line_cutoff_cm=args.line_cutoff_cm,
            subtract_cutoff_profile=args.subtract_cutoff_profile,
            line_taper_cm=args.line_taper_cm,
            line_wing_mode=args.line_wing_mode,
            lblrtm_sample=args.lblrtm_sample,
            lblrtm_alfal0=args.lblrtm_alfal0,
            lblrtm_avmass_amu=args.lblrtm_avmass_amu,
            lblrtm_hwf3=args.lblrtm_hwf3,
            rayleigh=args.rayleigh,
            rayleigh_xrayl=args.rayleigh_xrayl,
            n2_continuum=args.n2_continuum,
            n2_continuum_xn2cn=args.n2_continuum_xn2cn,
            o2_continuum=args.o2_continuum,
            o2_continuum_xo2cn=args.o2_continuum_xo2cn,
            line_margin_micron=args.line_margin_micron,
            min_transmission=args.min_transmission,
            fit_wavelength_shift=args.fit_wavelength_shift,
            fit_wavelength_polynomial=args.fit_wavelength_polynomial,
            wavelength_polynomial_order=args.wavelength_polynomial_order,
            initial_wavelength_shift=args.initial_wavelength_shift,
            wavelength_shift_bounds=tuple(args.wavelength_shift_bounds),
            fit_lsf_sigma=args.fit_lsf_sigma,
            lsf_sigma_bounds=tuple(args.lsf_sigma_bounds),
            fit_lsf_box_width=args.fit_lsf_box_width,
            lsf_box_width_bounds=tuple(args.lsf_box_width_bounds),
            fit_lsf_lorentz_fwhm=args.fit_lsf_lorentz_fwhm,
            lsf_lorentz_fwhm_bounds=tuple(args.lsf_lorentz_fwhm_bounds),
            fit_ranges=fit_ranges,
            exclude_ranges=exclude_ranges,
            loss=args.loss,
            f_scale=args.f_scale,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
            estimate_uncertainties=args.estimate_uncertainties,
            product_path=args.product,
            product_format=args.product_format,
            plot_path=args.plot,
        )
        print(f"success: {result.success}")
        print(f"cost: {result.cost:.6g}")
        print(f"wavelength shift: {result.wavelength_shift:.6g} micron")
        print(f"lsf sigma: {result.lsf_sigma_pixels:.6g} pixels")
        print(f"lsf box width: {result.lsf_box_width_pixels:.6g} pixels")
        print(f"lsf lorentz fwhm: {result.lsf_lorentz_fwhm_pixels:.6g} pixels")
        print("species scales:")
        for name, scale in result.species_scales.items():
            error = result.species_scale_uncertainties.get(name)
            suffix = "" if error is None else f" +/- {error:.3g}"
            print(f"  {name}: {scale:.6g}{suffix}")
        print(f"median transmission: {np.nanmedian(result.transmission):.6g}")
        print("metrics:")
        for name, value in result.metrics.items():
            print(f"  {name}: {value:.6g}")
        return 0

    if args.command == "convert-hitran":
        species = tuple(args.species) if args.species else None
        line_list = LineList.from_hitran_par(
            args.input,
            wavenumber_min=args.wavenumber_min,
            wavenumber_max=args.wavenumber_max,
            species=species,
            min_strength=args.min_strength,
            max_lines=args.max_lines,
        )
        line_list.write(args.output, format=args.format)
        print(f"lines: {line_list.wavelength.size}")
        print(f"species: {', '.join(line_list.species_names)}")
        return 0

    if args.command in {"fetch-hitran", "cache-hitran"}:
        options = {
            "species": tuple(args.species),
            "wavelength_min_micron": args.wavelength_min,
            "wavelength_max_micron": args.wavelength_max,
            "wavenumber_min_cm": args.wavenumber_min,
            "wavenumber_max_cm": args.wavenumber_max,
            "cache_dir": args.cache_dir,
            "force": args.force,
        }
        try:
            if args.command == "fetch-hitran":
                artifact = fetch_hitran_lines(
                    **options,
                    api_key_env=args.api_key_env,
                    timeout_s=args.timeout,
                )
            else:
                artifact = cache_hitran_par(args.input, **options)
        except (ValueError, HitranAcquisitionError) as exc:
            parser.error(str(exc))
        print(f"cache hit: {artifact.cache_hit}")
        print(f"lines: {artifact.line_list.wavelength.size}")
        print(f"line table: {artifact.table_path}")
        print(f"HITRAN par: {artifact.par_path}")
        print(f"manifest: {artifact.manifest_path}")
        return 0

    if args.command == "compare":
        candidate = load_spectrum(
            args.candidate,
            format=args.candidate_format,
            wavelength_unit=args.candidate_unit,
        )
        reference = load_spectrum(
            args.reference,
            format=args.reference_format,
            wavelength_unit=args.reference_unit,
        )
        comparison = compare_spectra(candidate, reference, normalize=args.normalize)
        for name, value in comparison.as_dict().items():
            print(f"{name}: {value}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
