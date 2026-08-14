from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, Mapping
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks, peak_widths

from .aer_data import AERCatalogArtifact, load_aer_line_window
from .atmosphere import (
    AtmosphereProfile,
    DEFAULT_OBSERVATORY_ALTITUDE_M,
    DEFAULT_OBSERVATORY_LATITUDE_DEG,
    DEFAULT_OBSERVATORY_LONGITUDE_DEG,
    DEFAULT_TELLURIC_MIXING_RATIOS,
    _header_representative_observation_time,
)
from .components import (
    AbsorptionComponent,
    CO2ContinuumAbsorption,
    H2OContinuumAbsorption,
    HitranLineAbsorption,
    N2CIAAbsorption,
    N2ContinuumAbsorption,
    O2CIAAbsorption,
    O2ContinuumAbsorption,
    PairCIAAbsorption,
    RayleighScatteringAbsorption,
    line_wing_effective_cutoff_cm,
)
from .continuum import HitranCIATable, LBLRTMCO2Continuum, LBLRTMH2OContinuum, MTCKDH2OContinuum, TabulatedContinuum
from .diagnostics import fit_quality_diagnostics, print_fit_summary
from .errors import ConfigurationError, WavelengthMetadataError
from .fit import (
    DEFAULT_MINIMUM_SPECIES_PEAK_OPTICAL_DEPTH,
    FitConfig,
    MultiTelluricFitResult,
    StellarForwardModel,
    TelluricFitResult,
    _apply_multi_fit_to_segment,
    _fit_metrics,
    _radiative_transfer_point_count,
    fit_telluric_segments,
    fit_tellurics,
)
from .io import (
    infer_wavelength_medium_from_header as _infer_wavelength_medium_from_header,
    infer_spectrum_format,
    load_spectrum,
    save_corrected_txt,
    save_fit_product_ecsv,
)
from .linelist import LineList
from .observation import Observation
from .partition import PartitionTable
from .physics import (
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_AVMASS_AMU,
    LBLRTM_DEFAULT_SAMPLE,
    LBLRTM_VOIGT_DOMAIN_HWF3,
    SPEED_OF_LIGHT_M_PER_S,
    lblrtm_dynamic_max_line_cutoff_cm,
    wavelength_micron_to_wavenumber_cm,
)
from .plotting import plot_fit
from .regions import RegionSelection, load_region_file
from .spectrum import Spectrum, normalize_wavelength_medium
from .theoretical import StellarMaskResult, TheoreticalSpectrum


DEFAULT_SEGMENT_SIZE_MICRON = 0.005
AUTO_LINEAR_CONTINUUM_MAX_NFEV = 100
DEFAULT_LSF_SIGMA_BOUNDS = (0.0, 5.0)
AUTO_LSF_SIGMA_FALLBACK_PIXELS = 2.0
AUTO_LSF_SIGMA_FEATURE_MAX_PIXELS = 6.0
AUTO_LSF_SIGMA_MAX_PIXELS = 50.0
AUTO_LSF_RESOLUTION_LOWER_FACTOR = 0.5
AUTO_LSF_RESOLUTION_UPPER_FACTOR = 2.0
AUTO_LSF_LORENTZ_PILOT_WIDTH_MICRON = 0.0005
AUTO_LSF_LORENTZ_MAX_PILOT_REGIONS = 3
AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS = 2
AUTO_LSF_LORENTZ_MIN_PILOT_PIXELS = 30
AUTO_LSF_LORENTZ_MIN_BIC_IMPROVEMENT = 10.0
AUTO_LSF_LORENTZ_MIN_REGION_FRACTION = 0.6
AUTO_LSF_LORENTZ_MIN_REGION_IMPROVEMENT = 1.0e-3
AUTO_LSF_LORENTZ_MAX_PIXELS = 50.0
AUTO_LSF_VARIABLE_EXPONENT_BOUNDS = (-2.0, 2.0)
AUTO_LSF_VARIABLE_MIN_BIC_IMPROVEMENT = 6.0
AUTO_LSF_VARIABLE_MIN_REGION_FRACTION = 0.6
AUTO_LSF_VARIABLE_MIN_LOG_WAVELENGTH_SPAN = 0.02
AUTO_WAVELENGTH_PILOT_WIDTH_MICRON = 0.002
AUTO_WAVELENGTH_SHIFT_BOUNDS_PIXELS = (-3.0, 3.0)
AUTO_WAVELENGTH_SHIFT_EXPANDED_BOUNDS_PIXELS = (-12.0, 12.0)
AUTO_WAVELENGTH_MIN_BIC_IMPROVEMENT = 6.0
AUTO_WAVELENGTH_MIN_REGION_FRACTION = 0.5

InputSpectrumFormat = Literal["txt", "dat", "csv", "ascii", "ecsv", "fits", "fit", "fz"]
WavelengthMedium = Literal["vacuum", "vac", "air"]
OptionalWavelengthMedium = Literal["vacuum", "vac", "air"] | None
ContinuumSolveMode = bool | Literal["auto"]
LSFSigmaInput = float | Literal["auto"]
LSFLorentzInput = float | Literal["auto"]
LSFFitMode = bool | Literal["auto"]
LSFVariableWidthMode = bool | Literal["auto"]
WavelengthFitMode = bool | Literal["auto"]
AtmosphereMode = Literal["mipas_gdas", "mipas", "gdas", "single", "standard"]
MIPASProfileName = Literal["equ", "std", "tro", "auto"]
GDASMode = Literal["auto", "online", "cache", "average"]
HighResolutionRebinMode = Literal[
    "integrate",
    "center",
    "sample_average",
    "molecfit_overlap",
    "molecfit_average",
]
RadiativeTransferGrid = Literal["auto", "model"]
LineWingMode = Literal[
    "full",
    "hard_cutoff",
    "subtracted_cutoff",
    "tapered_cutoff",
    "lblrtm_subtracted",
    "lblrtm_dynamic",
    "lblrtm_table",
    "lblrtm_panel",
]
LeastSquaresLoss = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]


def correct_arrays(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    uncertainty: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: WavelengthMedium = "vacuum",
    observation: Observation | None = None,
    line_list: LineList | None = None,
    line_list_path: str | Path | None = None,
    hitran_par: str | Path | None = None,
    hitran_species: tuple[str, ...] | None = None,
    hitran_min_strength: float | None = None,
    hitran_max_lines: int | None = None,
    demo_line_list: bool = False,
    aer_catalog: AERCatalogArtifact | str | Path | None = "auto",
    aer_cache_dir: str | Path | None = None,
    aer_source: str | Path | None = None,
    aer_offline: bool = False,
    aer_reuse_molecfit: bool = True,
    aer_timeout_s: float = 120.0,
    partition_table: PartitionTable | str | Path | None = None,
    h2o_continuum: MTCKDH2OContinuum | LBLRTMH2OContinuum | str | Path | None = None,
    h2o_continuum_foreign_closure: bool = False,
    co2_continuum: TabulatedContinuum | LBLRTMCO2Continuum | str | Path | None = None,
    o2_cia: HitranCIATable | str | Path | None = None,
    n2_cia: HitranCIATable | str | Path | None = None,
    cia_tables: Mapping[str, HitranCIATable | str | Path] | None = None,
    components: tuple[AbsorptionComponent, ...] | None = None,
    physical: bool | None = None,
    atmosphere: AtmosphereProfile | None = None,
    atmosphere_table: str | Path | None = None,
    atmosphere_mode: AtmosphereMode = "mipas_gdas",
    mipas_profile: MIPASProfileName = "equ",
    gdas_profile: str | Path | None = None,
    gdas_mode: GDASMode = "auto",
    gdas_cache_dir: str | Path | None = None,
    gdas_download_timeout_s: float = 15.0,
    observatory_latitude_deg: float | None = None,
    observatory_longitude_deg: float | None = None,
    observatory_altitude_m: float | None = None,
    allow_default_observatory: bool = False,
    airmass: float = 1.0,
    pressure_atm: float = 0.75,
    temperature_k: float = 280.0,
    path_length_m: float = 8_000.0,
    pwv_mm: float | None = None,
    relative_humidity_percent: float | None = None,
    mixing_ratios: Mapping[str, float] | None = None,
    continuum_order: int = 1,
    solve_continuum_linear: ContinuumSolveMode = "auto",
    lsf_sigma_pixels: LSFSigmaInput = "auto",
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: LSFLorentzInput = "auto",
    lsf_variable_width: LSFVariableWidthMode = "auto",
    lsf_reference_wavelength_micron: float | None = None,
    lsf_kernel_width_fwhm: float = 3.0,
    lsf_molecfit_voigt: bool = False,
    high_resolution_grid: bool = True,
    high_resolution_oversampling: float = 5.0,
    high_resolution_margin_pixels: float = 2.0,
    high_resolution_rebin_mode: HighResolutionRebinMode = "molecfit_overlap",
    radiative_transfer_grid: RadiativeTransferGrid = "auto",
    radiative_transfer_step_cm: float | None = None,
    radiative_transfer_max_points: int = 2_000_000,
    auto_segment: bool = True,
    segment_size: float = DEFAULT_SEGMENT_SIZE_MICRON,
    line_cutoff_cm: float | None = None,
    subtract_cutoff_profile: bool = False,
    line_taper_cm: float = 0.0,
    line_wing_mode: LineWingMode = "lblrtm_panel",
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE,
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3,
    rayleigh: bool = False,
    rayleigh_xrayl: float = 1.0,
    n2_continuum: bool = False,
    n2_continuum_xn2cn: float = 1.0,
    o2_continuum: bool = False,
    o2_continuum_xo2cn: float = 1.0,
    line_margin_micron: float = 0.01,
    min_transmission: float = 0.01,
    minimum_species_peak_optical_depth: float = (
        DEFAULT_MINIMUM_SPECIES_PEAK_OPTICAL_DEPTH
    ),
    fit_wavelength_shift: WavelengthFitMode = "auto",
    fit_wavelength_polynomial: bool = False,
    wavelength_polynomial_order: int = 1,
    fit_segment_wavelength_shifts: bool = False,
    fit_segment_wavelength_polynomial: bool = False,
    segment_wavelength_polynomial_order: int = 1,
    initial_wavelength_shift: float | None = None,
    wavelength_shift_bounds: tuple[float, float] | None = None,
    fit_lsf_sigma: LSFFitMode = "auto",
    lsf_sigma_bounds: tuple[float, float] | None = None,
    fit_lsf_box_width: bool = False,
    lsf_box_width_bounds: tuple[float, float] = (0.0, 10.0),
    fit_lsf_lorentz_fwhm: LSFFitMode = "auto",
    lsf_lorentz_fwhm_bounds: tuple[float, float] | None = None,
    fit_ranges: tuple[tuple[float, float], ...] | None = None,
    exclude_ranges: tuple[tuple[float, float], ...] | None = None,
    region_file: str | Path | None = None,
    theoretical_spectrum: TheoreticalSpectrum | None = None,
    stellar_mask_path: str | Path | None = None,
    loss: LeastSquaresLoss = "linear",
    f_scale: float = 1.0,
    ftol: float = 1.0e-10,
    xtol: float = 1.0e-10,
    gtol: float = 1.0e-10,
    estimate_uncertainties: bool = False,
) -> TelluricFitResult:
    """High-level telluric correction for wavelength and flux arrays.

    ``observation`` supplies metadata that arrays lack, including observation
    time/site for GDAS, resolving power for the automatic LSF, local weather,
    and the wavelength velocity frame. ``group_id`` identifies independent
    detector orders or chips when their wavelength ranges overlap; equal IDs
    are kept together during segmentation, fitting, and output stitching.
    Calls without an observation retain the legacy behavior and treat the
    arrays as already being in the observatory frame.
    """

    atmosphere_header = (
        None
        if observation is None
        else observation.to_header()
    )
    spectrum = Spectrum(
        wavelength=wavelength,
        flux=flux,
        uncertainty=uncertainty,
        mask=mask,
        group_id=group_id,
        wavelength_unit=wavelength_unit,
        wavelength_medium=wavelength_medium,
        meta=(
            {}
            if observation is None
            else {"observation": observation.to_header()}
        ),
    )
    fit_ranges, exclude_ranges = _resolve_region_file_ranges(
        region_file=region_file,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        spectrum=spectrum,
    )
    return _correct_spectrum_workflow(
        spectrum,
        line_list=line_list,
        line_list_path=line_list_path,
        hitran_par=hitran_par,
        hitran_species=hitran_species,
        hitran_min_strength=hitran_min_strength,
        hitran_max_lines=hitran_max_lines,
        demo_line_list=demo_line_list,
        aer_catalog=aer_catalog,
        aer_cache_dir=aer_cache_dir,
        aer_source=aer_source,
        aer_offline=aer_offline,
        aer_reuse_molecfit=aer_reuse_molecfit,
        aer_timeout_s=aer_timeout_s,
        partition_table=partition_table,
        h2o_continuum=h2o_continuum,
        h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
        co2_continuum=co2_continuum,
        o2_cia=o2_cia,
        n2_cia=n2_cia,
        cia_tables=cia_tables,
        components=components,
        physical=physical,
        atmosphere=atmosphere,
        atmosphere_table=atmosphere_table,
        atmosphere_mode=atmosphere_mode,
        atmosphere_header=atmosphere_header,
        mipas_profile=mipas_profile,
        gdas_profile=gdas_profile,
        gdas_mode=gdas_mode,
        gdas_cache_dir=gdas_cache_dir,
        gdas_download_timeout_s=gdas_download_timeout_s,
        observatory_latitude_deg=observatory_latitude_deg,
        observatory_longitude_deg=observatory_longitude_deg,
        observatory_altitude_m=observatory_altitude_m,
        allow_default_observatory=allow_default_observatory,
        airmass=airmass,
        pressure_atm=pressure_atm,
        temperature_k=temperature_k,
        path_length_m=path_length_m,
        pwv_mm=(
            observation.pwv_mm
            if pwv_mm is None and observation is not None
            else pwv_mm
        ),
        relative_humidity_percent=relative_humidity_percent,
        mixing_ratios=mixing_ratios,
        continuum_order=continuum_order,
        solve_continuum_linear=solve_continuum_linear,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        lsf_variable_width=lsf_variable_width,
        lsf_reference_wavelength_micron=lsf_reference_wavelength_micron,
        lsf_kernel_width_fwhm=lsf_kernel_width_fwhm,
        lsf_molecfit_voigt=lsf_molecfit_voigt,
        high_resolution_grid=high_resolution_grid,
        high_resolution_oversampling=high_resolution_oversampling,
        high_resolution_margin_pixels=high_resolution_margin_pixels,
        high_resolution_rebin_mode=high_resolution_rebin_mode,
        radiative_transfer_grid=radiative_transfer_grid,
        radiative_transfer_step_cm=radiative_transfer_step_cm,
        radiative_transfer_max_points=radiative_transfer_max_points,
        auto_segment=auto_segment,
        segment_size=segment_size,
        line_cutoff_cm=line_cutoff_cm,
        subtract_cutoff_profile=subtract_cutoff_profile,
        line_taper_cm=line_taper_cm,
        line_wing_mode=line_wing_mode,
        lblrtm_sample=lblrtm_sample,
        lblrtm_alfal0=lblrtm_alfal0,
        lblrtm_avmass_amu=lblrtm_avmass_amu,
        lblrtm_hwf3=lblrtm_hwf3,
        rayleigh=rayleigh,
        rayleigh_xrayl=rayleigh_xrayl,
        n2_continuum=n2_continuum,
        n2_continuum_xn2cn=n2_continuum_xn2cn,
        o2_continuum=o2_continuum,
        o2_continuum_xo2cn=o2_continuum_xo2cn,
        line_margin_micron=line_margin_micron,
        min_transmission=min_transmission,
        minimum_species_peak_optical_depth=minimum_species_peak_optical_depth,
        fit_wavelength_shift=fit_wavelength_shift,
        fit_wavelength_polynomial=fit_wavelength_polynomial,
        wavelength_polynomial_order=wavelength_polynomial_order,
        fit_segment_wavelength_shifts=fit_segment_wavelength_shifts,
        fit_segment_wavelength_polynomial=fit_segment_wavelength_polynomial,
        segment_wavelength_polynomial_order=segment_wavelength_polynomial_order,
        initial_wavelength_shift=initial_wavelength_shift,
        wavelength_shift_bounds=wavelength_shift_bounds,
        fit_lsf_sigma=fit_lsf_sigma,
        lsf_sigma_bounds=lsf_sigma_bounds,
        fit_lsf_box_width=fit_lsf_box_width,
        lsf_box_width_bounds=lsf_box_width_bounds,
        fit_lsf_lorentz_fwhm=fit_lsf_lorentz_fwhm,
        lsf_lorentz_fwhm_bounds=lsf_lorentz_fwhm_bounds,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        theoretical_spectrum=theoretical_spectrum,
        stellar_mask_path=stellar_mask_path,
        joint_stellar_model=False,
        loss=loss,
        f_scale=f_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        estimate_uncertainties=estimate_uncertainties,
    )


def correct_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    input_format: InputSpectrumFormat | None = None,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    hdu: int = 1,
    image_index: int | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: OptionalWavelengthMedium = None,
    observation: Observation | None = None,
    line_list: LineList | None = None,
    line_list_path: str | Path | None = None,
    hitran_par: str | Path | None = None,
    hitran_species: tuple[str, ...] | None = None,
    hitran_min_strength: float | None = None,
    hitran_max_lines: int | None = None,
    demo_line_list: bool = False,
    aer_catalog: AERCatalogArtifact | str | Path | None = "auto",
    aer_cache_dir: str | Path | None = None,
    aer_source: str | Path | None = None,
    aer_offline: bool = False,
    aer_reuse_molecfit: bool = True,
    aer_timeout_s: float = 120.0,
    partition_table: PartitionTable | str | Path | None = None,
    h2o_continuum: MTCKDH2OContinuum | LBLRTMH2OContinuum | str | Path | None = None,
    h2o_continuum_foreign_closure: bool = False,
    co2_continuum: TabulatedContinuum | LBLRTMCO2Continuum | str | Path | None = None,
    o2_cia: HitranCIATable | str | Path | None = None,
    n2_cia: HitranCIATable | str | Path | None = None,
    cia_tables: Mapping[str, HitranCIATable | str | Path] | None = None,
    components: tuple[AbsorptionComponent, ...] | None = None,
    physical: bool | None = None,
    atmosphere: AtmosphereProfile | None = None,
    atmosphere_table: str | Path | None = None,
    atmosphere_mode: AtmosphereMode = "mipas_gdas",
    mipas_profile: MIPASProfileName = "equ",
    gdas_profile: str | Path | None = None,
    gdas_mode: GDASMode = "auto",
    gdas_cache_dir: str | Path | None = None,
    gdas_download_timeout_s: float = 15.0,
    observatory_latitude_deg: float | None = None,
    observatory_longitude_deg: float | None = None,
    observatory_altitude_m: float | None = None,
    allow_default_observatory: bool = False,
    airmass: float = 1.0,
    pressure_atm: float = 0.75,
    temperature_k: float = 280.0,
    path_length_m: float = 8_000.0,
    pwv_mm: float | None = None,
    relative_humidity_percent: float | None = None,
    mixing_ratios: Mapping[str, float] | None = None,
    continuum_order: int = 1,
    solve_continuum_linear: ContinuumSolveMode = "auto",
    lsf_sigma_pixels: LSFSigmaInput = "auto",
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: LSFLorentzInput = "auto",
    lsf_variable_width: LSFVariableWidthMode = "auto",
    lsf_reference_wavelength_micron: float | None = None,
    lsf_kernel_width_fwhm: float = 3.0,
    lsf_molecfit_voigt: bool = False,
    high_resolution_grid: bool = True,
    high_resolution_oversampling: float = 5.0,
    high_resolution_margin_pixels: float = 2.0,
    high_resolution_rebin_mode: HighResolutionRebinMode = "molecfit_overlap",
    radiative_transfer_grid: RadiativeTransferGrid = "auto",
    radiative_transfer_step_cm: float | None = None,
    radiative_transfer_max_points: int = 2_000_000,
    auto_segment: bool = True,
    segment_size: float = DEFAULT_SEGMENT_SIZE_MICRON,
    line_cutoff_cm: float | None = None,
    subtract_cutoff_profile: bool = False,
    line_taper_cm: float = 0.0,
    line_wing_mode: LineWingMode = "lblrtm_panel",
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE,
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3,
    rayleigh: bool = False,
    rayleigh_xrayl: float = 1.0,
    n2_continuum: bool = False,
    n2_continuum_xn2cn: float = 1.0,
    o2_continuum: bool = False,
    o2_continuum_xo2cn: float = 1.0,
    line_margin_micron: float = 0.01,
    min_transmission: float = 0.01,
    minimum_species_peak_optical_depth: float = (
        DEFAULT_MINIMUM_SPECIES_PEAK_OPTICAL_DEPTH
    ),
    fit_wavelength_shift: WavelengthFitMode = "auto",
    fit_wavelength_polynomial: bool = False,
    wavelength_polynomial_order: int = 1,
    fit_segment_wavelength_shifts: bool = False,
    fit_segment_wavelength_polynomial: bool = False,
    segment_wavelength_polynomial_order: int = 1,
    initial_wavelength_shift: float | None = None,
    wavelength_shift_bounds: tuple[float, float] | None = None,
    fit_lsf_sigma: LSFFitMode = "auto",
    lsf_sigma_bounds: tuple[float, float] | None = None,
    fit_lsf_box_width: bool = False,
    lsf_box_width_bounds: tuple[float, float] = (0.0, 10.0),
    fit_lsf_lorentz_fwhm: LSFFitMode = "auto",
    lsf_lorentz_fwhm_bounds: tuple[float, float] | None = None,
    fit_ranges: tuple[tuple[float, float], ...] | None = None,
    exclude_ranges: tuple[tuple[float, float], ...] | None = None,
    region_file: str | Path | None = None,
    theoretical_spectrum: TheoreticalSpectrum | None = None,
    stellar_mask_path: str | Path | None = None,
    loss: LeastSquaresLoss = "linear",
    f_scale: float = 1.0,
    ftol: float = 1.0e-10,
    xtol: float = 1.0e-10,
    gtol: float = 1.0e-10,
    estimate_uncertainties: bool = False,
    product_path: str | Path | None = None,
    product_format: str = "ascii.ecsv",
    plot_path: str | Path | None = None,
    show_plot: bool = False,
    report: bool = True,
) -> TelluricFitResult:
    """Load a one-dimensional spectrum, fit telluric absorption, and correct it.

    String-valued options use the canonical choices listed below. ``None``
    generally means automatic discovery or that the optional feature is
    disabled, as described for the individual parameter.

    :param input_path: Input reduced 1D spectrum in FITS, text, CSV, or ECSV format.
    :param output_path: Optional path for the corrected wavelength/flux ASCII spectrum; ``None`` keeps it in memory only.
    :param input_format: ``None`` infers from the filename; explicit choices are ``txt``, ``dat``, ``csv``, ``ascii``, ``ecsv``, ``fits``, ``fit``, or ``fz``.
    :param wavelength_col: Wavelength column name or zero-based index; ``None`` uses recognized names or the first numeric column.
    :param flux_col: Flux column name or zero-based index; ``None`` uses recognized names or the second numeric column.
    :param uncertainty_col: Optional uncertainty column name or zero-based index; ``None`` performs an unweighted fit.
    :param hdu: FITS HDU containing the spectrum; ignored for text input.
    :param image_index: Row to extract from a two-dimensional FITS image; ``None`` requires a one-dimensional image.
    :param wavelength_unit: Unit of input wavelengths, such as ``micron``/``um``, ``nm``, ``angstrom``/``aa``, or ``m``.
    :param wavelength_medium: ``air`` declares standard-air wavelengths; ``vacuum`` or ``vac`` declares vacuum wavelengths. ``None`` accepts only an unambiguous air/vacuum declaration in FITS metadata; otherwise PyMolFit stops before fitting and asks for an explicit value. ``SPECSYS`` alone is not sufficient because it describes the velocity frame, not the wavelength medium.
    :param observation: Optional structured observing metadata. Explicit values override matching FITS-header values and can supply time/site, weather, resolving power, or spectral-frame information for text inputs.
    :param line_list: In-memory PyMolFit ``LineList``; use this instead of ``line_list_path`` or ``hitran_par``.
    :param line_list_path: Astropy-readable PyMolFit line-list table path; ``None`` leaves line-data resolution to another source.
    :param hitran_par: Path to a HITRAN (High-resolution Transmission Molecular Absorption Database) ``.par`` file containing molecular line positions, strengths, pressure broadening, and lower-state energies; ``None`` normally uses the managed AER catalogue.
    :param hitran_species: Molecule names retained from HITRAN/AER, such as ``H2O``, ``O2``, or ``CO2``; ``None`` keeps every atmospheric species with transitions in the wavelength window.
    :param hitran_min_strength: Minimum HITRAN reference line intensity to retain, controlling whether weak molecular transitions enter the opacity calculation; ``None`` applies no intensity threshold.
    :param hitran_max_lines: Maximum number of strongest retained lines; ``None`` keeps all lines passing the other filters.
    :param demo_line_list: ``True`` uses a tiny synthetic test list; ``False`` uses scientific line data. Never enable this for scientific correction.
    :param aer_catalog: AER is Atmospheric and Environmental Research's LBLRTM-ready molecular line catalogue derived from HITRAN; ``auto`` discovers/downloads it, a path or artifact selects it explicitly, and ``None`` disables automatic AER data.
    :param aer_cache_dir: Directory for managed AER catalogues and wavelength-window caches; ``None`` uses PyMolFit's user cache.
    :param aer_source: Override archive URL or local archive path used on an AER cache miss; ``None`` uses the official configured source.
    :param aer_offline: ``True`` forbids AER network access and requires cached data; ``False`` permits download on a cache miss.
    :param aer_reuse_molecfit: ``True`` may reuse a verified compatible local AER/Molecfit catalogue; ``False`` uses only the managed or explicit source.
    :param aer_timeout_s: Per-request AER download timeout in seconds.
    :param partition_table: Molecular partition sums convert HITRAN reference line strengths to each atmospheric layer's temperature; provide an object/table path, or use ``None`` for packaged LBLRTM/TIPS-compatible data.
    :param h2o_continuum: H2O continuum represents broad water absorption not captured by isolated discrete lines; provide an object/coefficient path, use ``lblrtm`` for packaged coefficients, or ``None`` for automatic packaged data when physical H2O lines are present.
    :param h2o_continuum_foreign_closure: ``True`` enables the optional MT_CKD foreign-continuum closure coefficients; ``False`` uses the normal selected continuum.
    :param co2_continuum: CO2 continuum represents broad carbon-dioxide absorption between/under discrete lines; provide an object/coefficient path, use ``lblrtm`` for packaged coefficients, or ``None`` for automatic packaged data when physical CO2 lines are present.
    :param o2_cia: Collision-induced absorption (CIA) is broadband absorption created during molecular collisions; provide a HITRAN O2 CIA object/file path, or ``None`` to omit this explicit table.
    :param n2_cia: Collision-induced absorption (CIA) is broadband absorption created during molecular collisions; provide a HITRAN N2 CIA object/file path, or ``None`` to omit this explicit table.
    :param cia_tables: Mapping from additional collision-pair names to HITRAN CIA objects or paths, adding broadband collision opacity beyond O2/N2; ``None`` adds no generic CIA tables.
    :param components: Additional or replacement absorption-component objects; ``None`` builds components from the selected physical data.
    :param physical: ``None`` auto-detects physical HITRAN modelling, ``True`` requires it, and ``False`` disables the layered physical-atmosphere path.
    :param atmosphere: Explicit ``AtmosphereProfile``; when provided it overrides the atmosphere builder and cannot be combined with ``atmosphere_table``.
    :param atmosphere_table: Astropy-readable atmosphere profile table; ``None`` builds an atmosphere from FITS metadata and the selected mode.
    :param atmosphere_mode: Selects how the vertical pressure, temperature, humidity, and gas profile is built: ``mipas_gdas``/``mipas``/``gdas`` merges a MIPAS satellite-derived climatological reference for the full/upper atmosphere with time-and-location-specific GDAS weather in the lower atmosphere, ``standard`` builds a generic layered midlatitude atmosphere, and ``single`` uses one homogeneous layer.
    :param mipas_profile: MIPAS (Michelson Interferometer for Passive Atmospheric Sounding) supplies climatological vertical pressure, temperature, and trace-gas profiles, especially above GDAS coverage; ``equ`` selects equatorial, ``std`` midlatitude standard, ``tro`` tropical, and ``auto`` currently follows Molecfit's equatorial default.
    :param gdas_profile: GDAS (NOAA Global Data Assimilation System) supplies observation-time/location meteorology such as pressure, height, temperature, and humidity for the lower atmosphere; provide an explicit FITS profile or use ``None`` to resolve one according to ``gdas_mode``.
    :param gdas_mode: Controls the NOAA GDAS weather profile used to replace the lower part of the MIPAS climatology: ``auto`` tries exact cached/downloaded time-local data then falls back to a monthly average, ``online`` requires exact cache/download success, ``cache`` requires exact cached data without network access, and ``average`` always uses a generic monthly profile.
    :param gdas_cache_dir: Directory for GDAS archives and interpolated profiles; ``None`` uses ``PYMOLFIT_GDAS_CACHE`` or the default user cache.
    :param gdas_download_timeout_s: Per-URL timeout in seconds for ESO GDAS downloads.
    :param observatory_latitude_deg: Geodetic observatory latitude in degrees; ``None`` reads FITS metadata when available.
    :param observatory_longitude_deg: Geodetic observatory longitude in degrees, positive east; ``None`` reads FITS metadata when available.
    :param observatory_altitude_m: Observatory altitude above sea level in metres; ``None`` reads FITS metadata when available.
    :param allow_default_observatory: ``True`` permits the Paranal fallback when geometry is missing; ``False`` raises instead of silently assuming a site.
    :param airmass: Line-of-sight airmass; the default allows usable FITS airmass metadata to supply the observation value in MIPAS/GDAS mode.
    :param pressure_atm: Surface pressure in atmospheres for ``single``/``standard`` or metadata fallback atmosphere construction.
    :param temperature_k: Surface or single-layer temperature in kelvin for fallback atmosphere construction.
    :param path_length_m: Vertical single-layer path length in metres; used only by ``atmosphere_mode="single"``.
    :param pwv_mm: Optional precipitable-water-vapour override in millimetres; ``None`` derives water from GDAS/MIPAS or other atmosphere inputs.
    :param relative_humidity_percent: Optional surface relative-humidity override in percent; ``None`` uses profile or FITS information.
    :param mixing_ratios: Optional mapping of molecule name to volume mixing ratio; supplied entries override builder defaults.
    :param continuum_order: Polynomial continuum degree per fitted segment: ``0`` constant, ``1`` linear, ``2`` quadratic, and so on.
    :param solve_continuum_linear: ``"auto"`` profiles continuum coefficients outside the nonlinear optimizer. Ordinary least squares uses the exact weighted linear solution; robust losses use an iteratively reweighted version of the same solve. ``True`` always selects this stable profiled approach, while ``False`` includes continuum coefficients in the nonlinear parameter vector.
    :param lsf_sigma_pixels: The line-spread function (LSF) describes instrumental broadening of intrinsically narrow telluric lines. ``"auto"`` first derives Gaussian sigma in detector pixels from FITS resolving-power metadata and wavelength sampling, or estimates narrow observed-feature widths when metadata is unavailable; a numeric value supplies the initial/fixed sigma directly, and ``0`` disables the Gaussian component unless it is fitted.
    :param lsf_box_width_pixels: A boxcar LSF component approximates finite pixel/slit integration; this is its initial/fixed width in detector pixels, and ``0`` disables it.
    :param lsf_lorentz_fwhm_pixels: A Lorentzian LSF component represents extended instrumental wings. ``"auto"`` compares Gaussian-only and Gaussian-plus-Lorentzian models on several telluric-rich pilot regions distributed over the spectrum; a numeric value supplies a fixed/initial full width at half maximum in detector pixels, and ``0`` disables the component unless explicitly fitted.
    :param lsf_variable_width: Controls wavelength dependence of instrumental broadening. ``"auto"`` (default) compares a constant-width LSF with ``width(lambda) = width_ref * (lambda / lambda_ref)**alpha`` on distributed telluric-rich pilot regions and keeps the power law only when BIC and cross-region improvement support it. ``True`` preserves the fixed legacy rule ``alpha=1``; ``False`` keeps all LSF widths constant in detector pixels.
    :param lsf_reference_wavelength_micron: Global reference wavelength ``lambda_ref`` in microns for the LSF width law. ``None`` uses the median wavelength of the complete input spectrum once, so separately processed orders share the same width definition.
    :param lsf_kernel_width_fwhm: Half-support control for numerical LSF kernels in multiples of component FWHM; larger values retain farther wings but cost more.
    :param lsf_molecfit_voigt: ``True`` uses Molecfit's synthetic Gaussian-plus-Lorentzian Voigt approximation; ``False`` convolves the configured components normally.
    :param high_resolution_grid: ``True`` computes, convolves, and rebins an oversampled internal model; ``False`` evaluates directly at observed samples.
    :param high_resolution_oversampling: Approximate internal samples per observed pixel when ``high_resolution_grid=True``.
    :param high_resolution_margin_pixels: Extra internal-grid margin on each segment edge in observed-pixel units, reducing convolution edge effects.
    :param high_resolution_rebin_mode: ``integrate`` averages pixel bins, ``center`` samples centres, ``sample_average`` averages enclosed samples, ``molecfit_overlap`` uses Molecfit-style overlap weights, and ``molecfit_average`` aliases sample averaging.
    :param radiative_transfer_grid: ``auto`` uses a layer-resolved native wavenumber grid; ``model`` evaluates opacity directly on the model grid for lower cost and lower fidelity.
    :param radiative_transfer_step_cm: Explicit native radiative-transfer spacing in inverse centimetres; ``None`` derives it from atmospheric layers and line widths.
    :param radiative_transfer_max_points: Maximum native-grid samples before PyMolFit asks for segmentation or a larger explicit safety limit.
    :param auto_segment: ``True`` separates FITS orders/detectors and wavelength discontinuities into physical groups, then divides wide groups into bounded radiative-transfer chunks. Molecular columns remain global, continuum coefficients remain local to numerical chunks, and automatic wavelength polynomials are shared across every chunk from the same physical group. ``False`` fits the input as one segment and may exceed grid limits.
    :param segment_size: Maximum numerical radiative-transfer chunk width in microns; ``0.005`` equals 50 Angstrom. Reducing it limits memory use but does not create independent physical orders or independent wavelength solutions.
    :param line_cutoff_cm: Optional maximum Voigt-wing distance in inverse centimetres; ``None`` follows ``line_wing_mode`` defaults.
    :param subtract_cutoff_profile: ``True`` subtracts the profile value at the cutoff before truncation; ``False`` leaves the chosen wing mode unchanged.
    :param line_taper_cm: Cosine-taper width in inverse centimetres at a finite line cutoff; ``0`` disables an added taper.
    :param line_wing_mode: ``full`` keeps all wings; ``hard_cutoff`` truncates; ``subtracted_cutoff`` edge-subtracts; ``tapered_cutoff`` tapers; ``lblrtm_subtracted`` uses fixed LBLRTM-style subtraction; ``lblrtm_dynamic`` uses dynamic per-line domains; ``lblrtm_table`` adds table-style accumulation; ``lblrtm_panel`` uses the source-parity panel/F4 treatment.
    :param lblrtm_sample: LBLRTM ``SAMPLE`` control used to derive dynamic line domains; it must be positive.
    :param lblrtm_alfal0: LBLRTM ``ALFAL0`` finite-domain control; ``0`` disables the corresponding finite ALFMAX cap.
    :param lblrtm_avmass_amu: Representative atmospheric molecular mass in atomic mass units used for LBLRTM layer sampling.
    :param lblrtm_hwf3: Outer LBLRTM Voigt F3 domain in line half-widths.
    :param rayleigh: ``True`` includes the LBLRTM Rayleigh-scattering branch; ``False`` omits it.
    :param rayleigh_xrayl: Multiplicative LBLRTM Rayleigh coefficient scale, normally ``1``.
    :param n2_continuum: ``True`` includes LBLRTM N2 pure-rotation, fundamental, and first-overtone continuum branches; ``False`` omits them.
    :param n2_continuum_xn2cn: Multiplicative LBLRTM N2-continuum coefficient scale, normally ``1``.
    :param o2_continuum: ``True`` includes source-backed ground-based LBLRTM O2 continuum branches; ``False`` omits them.
    :param o2_continuum_xo2cn: Multiplicative LBLRTM O2-continuum coefficient scale, normally ``1``.
    :param line_margin_micron: Extra line-selection margin in microns around each modelled spectral interval.
    :param min_transmission: Pixels with fitted atmospheric transmission below this fraction are masked in the corrected spectrum because division cannot recover reliable flux from nearly opaque regions; it must be strictly between ``0`` and ``1``.
    :param minimum_species_peak_optical_depth: A molecule's abundance scale is fitted only when its strongest expected absorption in the selected fit pixels reaches this optical-depth threshold. Weaker species remain in the atmospheric model at their profile abundance instead of acquiring an unconstrained extreme scale; lower the default only when deliberately measuring weak trace-gas lines.
    :param fit_wavelength_shift: ``"auto"`` uses one constant pixel offset per detected order/detector group for multi-order spectra. For a single physical group it compares no residual correction, a constant detector-pixel offset, and a smooth linear pixel-offset trend on distributed telluric-rich pilot regions, selecting by penalized fit quality. ``True`` preserves the explicit legacy constant-micron fit and ``False`` disables automatic residual alignment. Explicit polynomial or per-segment wavelength options take precedence over ``"auto"``.
    :param fit_wavelength_polynomial: ``True`` fits a global wavelength-correction polynomial; ``False`` disables it. Do not combine it with ``fit_wavelength_shift``.
    :param wavelength_polynomial_order: Degree of the global wavelength-correction polynomial in normalized wavelength coordinates.
    :param fit_segment_wavelength_shifts: ``True`` fits one constant wavelength offset per detected physical order/detector group and shares it across that group's numerical radiative-transfer chunks; it cannot be combined with either global wavelength-fit option.
    :param fit_segment_wavelength_polynomial: ``True`` fits one wavelength-correction polynomial per detected physical order/detector group and shares it across that group's numerical chunks; use this when line residuals show within-order wavelength distortion, and do not combine it with global or constant per-group wavelength fitting.
    :param segment_wavelength_polynomial_order: Degree of each physical-group wavelength polynomial when ``fit_segment_wavelength_polynomial=True``; ``0`` is a constant offset and ``1`` also permits a smooth linear distortion across the full order/detector group.
    :param initial_wavelength_shift: Initial constant wavelength offset in microns; ``None`` derives a suitable initial value from FITS spectral-frame metadata.
    :param wavelength_shift_bounds: Lower and upper coefficient bounds. ``None`` uses ``(-3, 3)`` detector pixels for automatic model selection or the legacy micron bounds for explicit wavelength models; supplied values are pixels in automatic mode and microns in explicit mode.
    :param fit_lsf_sigma: ``"auto"`` refines an automatically estimated ``lsf_sigma_pixels`` when telluric lines are available but keeps an explicitly numeric sigma fixed; ``True`` always fits Gaussian sigma and ``False`` always keeps the resolved value fixed.
    :param lsf_sigma_bounds: Optional lower and upper Gaussian sigma bounds in pixels. ``None`` generates broad non-negative bounds from the automatic estimate; explicit values must be increasing and non-negative.
    :param fit_lsf_box_width: ``True`` fits boxcar LSF width within ``lsf_box_width_bounds``; ``False`` keeps it fixed.
    :param lsf_box_width_bounds: Lower and upper boxcar-width bounds in pixels; values must be increasing and non-negative.
    :param fit_lsf_lorentz_fwhm: ``"auto"`` performs penalized pilot-model selection when ``lsf_lorentz_fwhm_pixels="auto"`` and otherwise keeps an explicit numeric width fixed; ``True`` forces Lorentzian fitting and ``False`` disables automatic fitting.
    :param lsf_lorentz_fwhm_bounds: Optional lower and upper Lorentzian FWHM bounds in pixels. ``None`` generates broad non-negative bounds from the Gaussian width; explicit values must be increasing and non-negative.
    :param fit_ranges: Wavelength intervals whose observed telluric features constrain molecular columns, wavelength alignment, LSF, and continuum; supply ``((start, stop), ...)`` in microns and the declared input medium, or ``None`` to let every valid pixel influence those fitted parameters.
    :param exclude_ranges: Wavelength intervals ignored only while estimating fit parameters, normally to protect stellar/circumstellar lines, detector defects, or saturated pixels; values are in microns/input medium and the final atmospheric correction is still evaluated there.
    :param region_file: Optional PyMolFit ECSV file created by ``select_telluric_regions``. Its native wavelength unit and air/vacuum medium are converted to the input spectrum coordinates. Do not combine it with ``fit_ranges`` or ``exclude_ranges``.
    :param theoretical_spectrum: Optional ``TheoreticalSpectrum`` used to identify stellar features represented by the supplied model that must not constrain atmospheric parameters. The broadened template creates additional exclusion regions only; it never replaces science flux and correction is still evaluated at those wavelengths.
    :param stellar_mask_path: Optional ECSV path for the automatically generated stellar exclusion regions in the input spectrum's native wavelength coordinates.
    :param loss: Controls how residuals influence the fit. ``linear`` is ordinary squared-residual least squares and is the statistically preferred default for a clean, well-masked spectrum with reliable uncertainties. ``soft_l1`` is usually appropriate for a mostly clean spectrum containing a limited number of cosmic rays, bad pixels, or unmasked spectral features because it reduces their influence without ignoring normal residuals. ``huber`` is another moderate robust loss, while ``cauchy`` and ``arctan`` suppress large outliers more strongly. Robust loss cannot repair generally poor calibration, wavelength misalignment, incorrect uncertainties, or an unsuitable atmospheric model.
    :param f_scale: Residual scale separating normal points from downweighted outliers for robust losses; larger values treat more residuals as normal and smaller values reject deviations more aggressively. It has no effect for ``loss="linear"``.
    :param ftol: Positive relative cost-change tolerance for optimizer termination.
    :param xtol: Positive relative parameter-step tolerance for optimizer termination.
    :param gtol: Positive gradient-norm tolerance for optimizer termination.
    :param estimate_uncertainties: ``True`` estimates local covariance and propagates transmission uncertainty; ``False`` skips this additional work.
    :param product_path: Optional path for the full fit-product table containing model, transmission, masks, corrected flux, metadata, and provenance.
    :param product_format: Astropy table writer format used for ``product_path``, for example ``ascii.ecsv`` or ``fits``.
    :param plot_path: Optional diagnostic-plot output path; ``None`` does not save a plot.
    :param show_plot: ``True`` opens/displays the diagnostic plot; ``False`` only saves it when ``plot_path`` is provided.
    :param report: Print the resolved fit configuration and result summary when ``True``.
    :return: ``TelluricFitResult`` containing the input, model, transmission, corrected spectrum, fitted parameters, diagnostics, and provenance.
    """

    atmosphere_header = _load_fits_header_if_available(
        input_path,
        input_format,
        hdu=hdu,
    )
    if observation is not None:
        atmosphere_header = observation.to_header(atmosphere_header)
    resolved_wavelength_medium = _resolve_wavelength_medium(
        wavelength_medium,
        atmosphere_header,
        wavelength_col=wavelength_col,
    )
    spectrum = load_spectrum(
        input_path,
        format=input_format,
        wavelength_col=wavelength_col,
        flux_col=flux_col,
        uncertainty_col=uncertainty_col,
        hdu=hdu,
        wavelength_unit=wavelength_unit,
        wavelength_medium=resolved_wavelength_medium,
        image_index=image_index,
        save_header=False,
    )
    fit_ranges, exclude_ranges = _resolve_region_file_ranges(
        region_file=region_file,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        spectrum=spectrum,
    )
    result = _correct_spectrum_workflow(
        spectrum,
        line_list=line_list,
        line_list_path=line_list_path,
        hitran_par=hitran_par,
        hitran_species=hitran_species,
        hitran_min_strength=hitran_min_strength,
        hitran_max_lines=hitran_max_lines,
        demo_line_list=demo_line_list,
        aer_catalog=aer_catalog,
        aer_cache_dir=aer_cache_dir,
        aer_source=aer_source,
        aer_offline=aer_offline,
        aer_reuse_molecfit=aer_reuse_molecfit,
        aer_timeout_s=aer_timeout_s,
        partition_table=partition_table,
        h2o_continuum=h2o_continuum,
        h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
        co2_continuum=co2_continuum,
        o2_cia=o2_cia,
        n2_cia=n2_cia,
        cia_tables=cia_tables,
        components=components,
        physical=physical,
        atmosphere=atmosphere,
        atmosphere_table=atmosphere_table,
        atmosphere_mode=atmosphere_mode,
        atmosphere_header=atmosphere_header,
        mipas_profile=mipas_profile,
        gdas_profile=gdas_profile,
        gdas_mode=gdas_mode,
        gdas_cache_dir=gdas_cache_dir,
        gdas_download_timeout_s=gdas_download_timeout_s,
        observatory_latitude_deg=observatory_latitude_deg,
        observatory_longitude_deg=observatory_longitude_deg,
        observatory_altitude_m=observatory_altitude_m,
        allow_default_observatory=allow_default_observatory,
        airmass=airmass,
        pressure_atm=pressure_atm,
        temperature_k=temperature_k,
        path_length_m=path_length_m,
        pwv_mm=(
            observation.pwv_mm
            if pwv_mm is None and observation is not None
            else pwv_mm
        ),
        relative_humidity_percent=relative_humidity_percent,
        mixing_ratios=mixing_ratios,
        continuum_order=continuum_order,
        solve_continuum_linear=solve_continuum_linear,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        lsf_variable_width=lsf_variable_width,
        lsf_reference_wavelength_micron=lsf_reference_wavelength_micron,
        lsf_kernel_width_fwhm=lsf_kernel_width_fwhm,
        lsf_molecfit_voigt=lsf_molecfit_voigt,
        high_resolution_grid=high_resolution_grid,
        high_resolution_oversampling=high_resolution_oversampling,
        high_resolution_margin_pixels=high_resolution_margin_pixels,
        high_resolution_rebin_mode=high_resolution_rebin_mode,
        radiative_transfer_grid=radiative_transfer_grid,
        radiative_transfer_step_cm=radiative_transfer_step_cm,
        radiative_transfer_max_points=radiative_transfer_max_points,
        auto_segment=auto_segment,
        segment_size=segment_size,
        line_cutoff_cm=line_cutoff_cm,
        subtract_cutoff_profile=subtract_cutoff_profile,
        line_taper_cm=line_taper_cm,
        line_wing_mode=line_wing_mode,
        lblrtm_sample=lblrtm_sample,
        lblrtm_alfal0=lblrtm_alfal0,
        lblrtm_avmass_amu=lblrtm_avmass_amu,
        lblrtm_hwf3=lblrtm_hwf3,
        rayleigh=rayleigh,
        rayleigh_xrayl=rayleigh_xrayl,
        n2_continuum=n2_continuum,
        n2_continuum_xn2cn=n2_continuum_xn2cn,
        o2_continuum=o2_continuum,
        o2_continuum_xo2cn=o2_continuum_xo2cn,
        line_margin_micron=line_margin_micron,
        min_transmission=min_transmission,
        minimum_species_peak_optical_depth=minimum_species_peak_optical_depth,
        fit_wavelength_shift=fit_wavelength_shift,
        fit_wavelength_polynomial=fit_wavelength_polynomial,
        wavelength_polynomial_order=wavelength_polynomial_order,
        fit_segment_wavelength_shifts=fit_segment_wavelength_shifts,
        fit_segment_wavelength_polynomial=fit_segment_wavelength_polynomial,
        segment_wavelength_polynomial_order=segment_wavelength_polynomial_order,
        initial_wavelength_shift=initial_wavelength_shift,
        wavelength_shift_bounds=wavelength_shift_bounds,
        fit_lsf_sigma=fit_lsf_sigma,
        lsf_sigma_bounds=lsf_sigma_bounds,
        fit_lsf_box_width=fit_lsf_box_width,
        lsf_box_width_bounds=lsf_box_width_bounds,
        fit_lsf_lorentz_fwhm=fit_lsf_lorentz_fwhm,
        lsf_lorentz_fwhm_bounds=lsf_lorentz_fwhm_bounds,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        theoretical_spectrum=theoretical_spectrum,
        stellar_mask_path=stellar_mask_path,
        joint_stellar_model=False,
        loss=loss,
        f_scale=f_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        estimate_uncertainties=estimate_uncertainties,
    )

    return _finalize_correction(
        result,
        input_label=input_path,
        output_path=output_path,
        product_path=product_path,
        product_format=product_format,
        plot_path=plot_path,
        show_plot=show_plot,
        report=report,
    )


def correct(
    input_path: str | Path | None = None,
    *,
    spectrum: Spectrum | None = None,
    wavelength: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    input_format: InputSpectrumFormat | None = None,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    hdu: int = 1,
    image_index: int | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: OptionalWavelengthMedium = None,
    observation: Observation | None = None,
    line_list: LineList | None = None,
    line_list_path: str | Path | None = None,
    hitran_par: str | Path | None = None,
    hitran_species: tuple[str, ...] | None = None,
    hitran_min_strength: float | None = None,
    hitran_max_lines: int | None = None,
    demo_line_list: bool = False,
    aer_catalog: AERCatalogArtifact | str | Path | None = "auto",
    aer_cache_dir: str | Path | None = None,
    aer_source: str | Path | None = None,
    aer_offline: bool = False,
    aer_reuse_molecfit: bool = True,
    aer_timeout_s: float = 120.0,
    partition_table: PartitionTable | str | Path | None = None,
    h2o_continuum: MTCKDH2OContinuum | LBLRTMH2OContinuum | str | Path | None = None,
    h2o_continuum_foreign_closure: bool = False,
    co2_continuum: TabulatedContinuum | LBLRTMCO2Continuum | str | Path | None = None,
    o2_cia: HitranCIATable | str | Path | None = None,
    n2_cia: HitranCIATable | str | Path | None = None,
    cia_tables: Mapping[str, HitranCIATable | str | Path] | None = None,
    components: tuple[AbsorptionComponent, ...] | None = None,
    physical: bool | None = None,
    atmosphere: AtmosphereProfile | None = None,
    atmosphere_table: str | Path | None = None,
    atmosphere_mode: AtmosphereMode = "mipas_gdas",
    mipas_profile: MIPASProfileName = "equ",
    gdas_profile: str | Path | None = None,
    gdas_mode: GDASMode = "auto",
    gdas_cache_dir: str | Path | None = None,
    gdas_download_timeout_s: float = 15.0,
    observatory_latitude_deg: float | None = None,
    observatory_longitude_deg: float | None = None,
    observatory_altitude_m: float | None = None,
    allow_default_observatory: bool = False,
    airmass: float = 1.0,
    pressure_atm: float = 0.75,
    temperature_k: float = 280.0,
    path_length_m: float = 8_000.0,
    pwv_mm: float | None = None,
    relative_humidity_percent: float | None = None,
    mixing_ratios: Mapping[str, float] | None = None,
    continuum_order: int = 1,
    solve_continuum_linear: ContinuumSolveMode = "auto",
    lsf_sigma_pixels: LSFSigmaInput = "auto",
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: LSFLorentzInput = "auto",
    lsf_variable_width: LSFVariableWidthMode = "auto",
    lsf_reference_wavelength_micron: float | None = None,
    lsf_kernel_width_fwhm: float = 3.0,
    lsf_molecfit_voigt: bool = False,
    high_resolution_grid: bool = True,
    high_resolution_oversampling: float = 5.0,
    high_resolution_margin_pixels: float = 2.0,
    high_resolution_rebin_mode: HighResolutionRebinMode = "molecfit_overlap",
    radiative_transfer_grid: RadiativeTransferGrid = "auto",
    radiative_transfer_step_cm: float | None = None,
    radiative_transfer_max_points: int = 2_000_000,
    auto_segment: bool = True,
    segment_size: float = DEFAULT_SEGMENT_SIZE_MICRON,
    line_cutoff_cm: float | None = None,
    subtract_cutoff_profile: bool = False,
    line_taper_cm: float = 0.0,
    line_wing_mode: LineWingMode = "lblrtm_panel",
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE,
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3,
    rayleigh: bool = False,
    rayleigh_xrayl: float = 1.0,
    n2_continuum: bool = False,
    n2_continuum_xn2cn: float = 1.0,
    o2_continuum: bool = False,
    o2_continuum_xo2cn: float = 1.0,
    line_margin_micron: float = 0.01,
    min_transmission: float = 0.01,
    minimum_species_peak_optical_depth: float = (
        DEFAULT_MINIMUM_SPECIES_PEAK_OPTICAL_DEPTH
    ),
    fit_wavelength_shift: WavelengthFitMode = "auto",
    fit_wavelength_polynomial: bool = False,
    wavelength_polynomial_order: int = 1,
    fit_segment_wavelength_shifts: bool = False,
    fit_segment_wavelength_polynomial: bool = False,
    segment_wavelength_polynomial_order: int = 1,
    initial_wavelength_shift: float | None = None,
    wavelength_shift_bounds: tuple[float, float] | None = None,
    fit_lsf_sigma: LSFFitMode = "auto",
    lsf_sigma_bounds: tuple[float, float] | None = None,
    fit_lsf_box_width: bool = False,
    lsf_box_width_bounds: tuple[float, float] = (0.0, 10.0),
    fit_lsf_lorentz_fwhm: LSFFitMode = "auto",
    lsf_lorentz_fwhm_bounds: tuple[float, float] | None = None,
    fit_ranges: tuple[tuple[float, float], ...] | None = None,
    exclude_ranges: tuple[tuple[float, float], ...] | None = None,
    region_file: str | Path | None = None,
    theoretical_spectrum: TheoreticalSpectrum | None = None,
    stellar_mask_path: str | Path | None = None,
    joint_stellar_model: bool = False,
    loss: LeastSquaresLoss = "linear",
    f_scale: float = 1.0,
    ftol: float = 1.0e-10,
    xtol: float = 1.0e-10,
    gtol: float = 1.0e-10,
    estimate_uncertainties: bool = False,
    output_path: str | Path | None = None,
    product_path: str | Path | None = None,
    product_format: str = "ascii.ecsv",
    plot_path: str | Path | None = None,
    show_plot: bool = False,
    report: bool = True,
) -> TelluricFitResult:
    """Correct either a spectrum file or wavelength/flux arrays.

    Supply exactly one input route:

    - ``input_path`` for FITS, text, CSV, or ECSV data;
    - ``spectrum`` for a previously loaded :class:`Spectrum`; or
    - both ``wavelength`` and ``flux`` plus an ``Observation``.

    Array input must explicitly declare ``wavelength_medium`` and the
    observation's ``wavelength_frame``. This prevents air/vacuum or
    barycentric/topocentric assumptions from silently moving the telluric
    model. Scientific controls are shared with ``correct_file`` and
    ``correct_arrays`` and are declared explicitly for editor completion and
    type checking.

    Existing ``correct_file`` and ``correct_arrays`` calls remain supported.

    :param input_path: FITS, text, CSV, or ECSV spectrum path. Do not combine
        this with ``spectrum``, ``wavelength``, or ``flux``.
    :param spectrum: Previously loaded ``Spectrum``. Its units, wavelength
        medium, masks, order labels, and metadata are preserved.
    :param wavelength: One-dimensional wavelength array. It must be paired
        with ``flux``, ``observation``, and an explicit ``wavelength_medium``.
    :param flux: One-dimensional flux array sampled at ``wavelength``.
    :param uncertainty: Optional one-sigma flux uncertainty array for weighted
        fitting and propagated corrected-flux uncertainty.
    :param mask: Optional Boolean array; ``True`` values are valid fit pixels.
    :param group_id: Optional detector/order label for every array sample.
        Samples with equal labels are treated as one physical spectral group,
        preventing overlapping echelle orders from being interleaved.
    :param wavelength_unit: Unit of the input wavelengths, for example
        ``"micron"``, ``"nm"``, or ``"angstrom"``.
    :param wavelength_medium: ``"air"`` or ``"vacuum"``. A file may infer this
        from unambiguous metadata; arrays must declare it explicitly.
    :param observation: Structured observing metadata. It is required for
        arrays and may override incomplete or incorrect file-header values.
    :param region_file: Optional PyMolFit ECSV file containing fit and
        exclusion regions selected in the input spectrum's wavelength
        coordinates. Do not combine it with explicit ``fit_ranges`` or
        ``exclude_ranges``.
    :param theoretical_spectrum: Optional broadened stellar template whose
        predicted astrophysical features are excluded from telluric parameter
        estimation without changing the science flux.
    :param stellar_mask_path: Optional ECSV output for the generated stellar
        exclusions in the input spectrum's native wavelength coordinates.
    :param joint_stellar_model: When ``True``, fit the supplied theoretical
        spectrum multiplicatively with the atmospheric transmission and
        continuum. This can separate overlapping stellar and telluric lines.
        It requires ``theoretical_spectrum`` and is disabled by default. The
        corrected science flux still divides out only atmospheric transmission.
    :param output_path: Optional compact text output containing wavelength and
        corrected flux.
    :param product_path: Optional full fit-product output containing model,
        transmission, masks, metadata, and provenance.
    :param product_format: Astropy writer format for ``product_path``; the
        default is portable ``"ascii.ecsv"``.
    :param plot_path: Optional path for a saved diagnostic plot.
    :param show_plot: Display the diagnostic plot when ``True``.
    :param report: Print the resolved fit configuration and result summary.
    :return: Complete ``TelluricFitResult`` in memory.
    """

    has_path = input_path is not None
    has_spectrum = spectrum is not None
    has_wavelength = wavelength is not None
    has_flux = flux is not None
    has_array_input = has_wavelength or has_flux

    route_count = int(has_path) + int(has_spectrum) + int(has_array_input)
    if route_count == 0:
        raise ConfigurationError(
            "provide input_path, spectrum, or both "
            "wavelength and flux arrays"
        )
    if route_count > 1:
        raise ConfigurationError(
            "provide input_path, spectrum, or wavelength/flux arrays, not both "
            "or multiple input routes"
        )
    if has_array_input and not (has_wavelength and has_flux):
        raise ConfigurationError(
            "array input requires both wavelength and flux"
        )

    fit_options = {
        "line_list": line_list,
        "line_list_path": line_list_path,
        "hitran_par": hitran_par,
        "hitran_species": hitran_species,
        "hitran_min_strength": hitran_min_strength,
        "hitran_max_lines": hitran_max_lines,
        "demo_line_list": demo_line_list,
        "aer_catalog": aer_catalog,
        "aer_cache_dir": aer_cache_dir,
        "aer_source": aer_source,
        "aer_offline": aer_offline,
        "aer_reuse_molecfit": aer_reuse_molecfit,
        "aer_timeout_s": aer_timeout_s,
        "partition_table": partition_table,
        "h2o_continuum": h2o_continuum,
        "h2o_continuum_foreign_closure": h2o_continuum_foreign_closure,
        "co2_continuum": co2_continuum,
        "o2_cia": o2_cia,
        "n2_cia": n2_cia,
        "cia_tables": cia_tables,
        "components": components,
        "physical": physical,
        "atmosphere": atmosphere,
        "atmosphere_table": atmosphere_table,
        "atmosphere_mode": atmosphere_mode,
        "mipas_profile": mipas_profile,
        "gdas_profile": gdas_profile,
        "gdas_mode": gdas_mode,
        "gdas_cache_dir": gdas_cache_dir,
        "gdas_download_timeout_s": gdas_download_timeout_s,
        "observatory_latitude_deg": observatory_latitude_deg,
        "observatory_longitude_deg": observatory_longitude_deg,
        "observatory_altitude_m": observatory_altitude_m,
        "allow_default_observatory": allow_default_observatory,
        "airmass": airmass,
        "pressure_atm": pressure_atm,
        "temperature_k": temperature_k,
        "path_length_m": path_length_m,
        "pwv_mm": pwv_mm,
        "relative_humidity_percent": relative_humidity_percent,
        "mixing_ratios": mixing_ratios,
        "continuum_order": continuum_order,
        "solve_continuum_linear": solve_continuum_linear,
        "lsf_sigma_pixels": lsf_sigma_pixels,
        "lsf_box_width_pixels": lsf_box_width_pixels,
        "lsf_lorentz_fwhm_pixels": lsf_lorentz_fwhm_pixels,
        "lsf_variable_width": lsf_variable_width,
        "lsf_reference_wavelength_micron": lsf_reference_wavelength_micron,
        "lsf_kernel_width_fwhm": lsf_kernel_width_fwhm,
        "lsf_molecfit_voigt": lsf_molecfit_voigt,
        "high_resolution_grid": high_resolution_grid,
        "high_resolution_oversampling": high_resolution_oversampling,
        "high_resolution_margin_pixels": high_resolution_margin_pixels,
        "high_resolution_rebin_mode": high_resolution_rebin_mode,
        "radiative_transfer_grid": radiative_transfer_grid,
        "radiative_transfer_step_cm": radiative_transfer_step_cm,
        "radiative_transfer_max_points": radiative_transfer_max_points,
        "auto_segment": auto_segment,
        "segment_size": segment_size,
        "line_cutoff_cm": line_cutoff_cm,
        "subtract_cutoff_profile": subtract_cutoff_profile,
        "line_taper_cm": line_taper_cm,
        "line_wing_mode": line_wing_mode,
        "lblrtm_sample": lblrtm_sample,
        "lblrtm_alfal0": lblrtm_alfal0,
        "lblrtm_avmass_amu": lblrtm_avmass_amu,
        "lblrtm_hwf3": lblrtm_hwf3,
        "rayleigh": rayleigh,
        "rayleigh_xrayl": rayleigh_xrayl,
        "n2_continuum": n2_continuum,
        "n2_continuum_xn2cn": n2_continuum_xn2cn,
        "o2_continuum": o2_continuum,
        "o2_continuum_xo2cn": o2_continuum_xo2cn,
        "line_margin_micron": line_margin_micron,
        "min_transmission": min_transmission,
        "minimum_species_peak_optical_depth": minimum_species_peak_optical_depth,
        "fit_wavelength_shift": fit_wavelength_shift,
        "fit_wavelength_polynomial": fit_wavelength_polynomial,
        "wavelength_polynomial_order": wavelength_polynomial_order,
        "fit_segment_wavelength_shifts": fit_segment_wavelength_shifts,
        "fit_segment_wavelength_polynomial": fit_segment_wavelength_polynomial,
        "segment_wavelength_polynomial_order": segment_wavelength_polynomial_order,
        "initial_wavelength_shift": initial_wavelength_shift,
        "wavelength_shift_bounds": wavelength_shift_bounds,
        "fit_lsf_sigma": fit_lsf_sigma,
        "lsf_sigma_bounds": lsf_sigma_bounds,
        "fit_lsf_box_width": fit_lsf_box_width,
        "lsf_box_width_bounds": lsf_box_width_bounds,
        "fit_lsf_lorentz_fwhm": fit_lsf_lorentz_fwhm,
        "lsf_lorentz_fwhm_bounds": lsf_lorentz_fwhm_bounds,
        "fit_ranges": fit_ranges,
        "exclude_ranges": exclude_ranges,
        "theoretical_spectrum": theoretical_spectrum,
        "stellar_mask_path": stellar_mask_path,
        "loss": loss,
        "f_scale": f_scale,
        "ftol": ftol,
        "xtol": xtol,
        "gtol": gtol,
        "estimate_uncertainties": estimate_uncertainties,
    }

    if has_path and any(value is not None for value in (uncertainty, mask, group_id)):
        raise ConfigurationError(
            "uncertainty, mask, and group_id arrays cannot be combined with input_path; "
            "load the file as arrays or identify its uncertainty column"
        )

    if has_path and not joint_stellar_model:
        return correct_file(
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
            observation=observation,
            region_file=region_file,
            product_path=product_path,
            product_format=product_format,
            plot_path=plot_path,
            show_plot=show_plot,
            report=report,
            **fit_options,
        )

    if has_path:
        atmosphere_header = _load_fits_header_if_available(
            input_path,
            input_format,
            hdu=hdu,
        )
        if observation is not None:
            atmosphere_header = observation.to_header(atmosphere_header)
        resolved_medium = _resolve_wavelength_medium(
            wavelength_medium,
            atmosphere_header,
            wavelength_col=wavelength_col,
        )
        loaded = load_spectrum(
            input_path,
            format=input_format,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            hdu=hdu,
            wavelength_unit=wavelength_unit,
            wavelength_medium=resolved_medium,
            image_index=image_index,
            save_header=False,
        )
        resolved_fit_ranges, resolved_exclude_ranges = _resolve_region_file_ranges(
            region_file=region_file,
            fit_ranges=fit_ranges,
            exclude_ranges=exclude_ranges,
            spectrum=loaded,
        )
        path_options = {
            **fit_options,
            "fit_ranges": resolved_fit_ranges,
            "exclude_ranges": resolved_exclude_ranges,
            "pwv_mm": (
                observation.pwv_mm
                if pwv_mm is None and observation is not None
                else pwv_mm
            ),
        }
        result = _correct_spectrum_workflow(
            loaded,
            atmosphere_header=atmosphere_header,
            joint_stellar_model=True,
            **path_options,
        )
        return _finalize_correction(
            result,
            input_label=input_path,
            output_path=output_path,
            product_path=product_path,
            product_format=product_format,
            plot_path=plot_path,
            show_plot=show_plot,
            report=report,
        )

    if has_spectrum:
        if any(value is not None for value in (wavelength, flux, uncertainty, mask, group_id)):
            raise ConfigurationError(
                "spectrum input cannot be combined with wavelength, flux, "
                "uncertainty, mask, or group_id arrays"
            )
        if any(
            value is not None
            for value in (
                input_format,
                wavelength_col,
                flux_col,
                uncertainty_col,
                image_index,
            )
        ):
            raise ConfigurationError(
                "file format, column, HDU-image options apply only to input_path"
            )
        if (
            wavelength_medium is not None
            and spectrum.wavelength_medium
            != normalize_wavelength_medium(wavelength_medium)
        ):
            raise WavelengthMetadataError(
                "wavelength_medium conflicts with spectrum.wavelength_medium"
            )
        source = spectrum.meta.get("source")
        atmosphere_header = None
        if source:
            atmosphere_header = _load_fits_header_if_available(
                str(source),
                None,
                hdu=hdu,
            )
        if observation is not None:
            atmosphere_header = observation.to_header(atmosphere_header)
        resolved_fit_ranges, resolved_exclude_ranges = _resolve_region_file_ranges(
            region_file=region_file,
            fit_ranges=fit_ranges,
            exclude_ranges=exclude_ranges,
            spectrum=spectrum,
        )
        spectrum_options = {
            **fit_options,
            "fit_ranges": resolved_fit_ranges,
            "exclude_ranges": resolved_exclude_ranges,
        }
        result = _correct_spectrum_workflow(
            spectrum,
            atmosphere_header=atmosphere_header,
            joint_stellar_model=joint_stellar_model,
            **spectrum_options,
        )
        return _finalize_correction(
            result,
            input_label=spectrum.name or source or "<Spectrum>",
            output_path=output_path,
            product_path=product_path,
            product_format=product_format,
            plot_path=plot_path,
            show_plot=show_plot,
            report=report,
        )

    if observation is None:
        raise ConfigurationError(
            "array input requires observation=Observation(...) so observing "
            "metadata is explicit"
        )
    if observation.wavelength_frame is None:
        raise ConfigurationError(
            "array input requires observation.wavelength_frame to be "
            "'observatory', 'barycentric', or 'heliocentric'"
        )
    if wavelength_medium is None:
        raise WavelengthMetadataError(
            "array input requires wavelength_medium='air' or 'vacuum'"
        )
    if any(
        value is not None
        for value in (input_format, wavelength_col, flux_col, uncertainty_col)
    ):
        raise ConfigurationError(
            "input_format and column selectors apply only to input_path"
        )

    if joint_stellar_model:
        array_spectrum = Spectrum(
            wavelength=np.asarray(wavelength, dtype=float),
            flux=np.asarray(flux, dtype=float),
            uncertainty=uncertainty,
            mask=mask,
            group_id=group_id,
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
            meta={"observation": observation.to_header()},
        )
        resolved_fit_ranges, resolved_exclude_ranges = _resolve_region_file_ranges(
            region_file=region_file,
            fit_ranges=fit_ranges,
            exclude_ranges=exclude_ranges,
            spectrum=array_spectrum,
        )
        array_options = {
            **fit_options,
            "fit_ranges": resolved_fit_ranges,
            "exclude_ranges": resolved_exclude_ranges,
            "pwv_mm": observation.pwv_mm if pwv_mm is None else pwv_mm,
        }
        result = _correct_spectrum_workflow(
            array_spectrum,
            atmosphere_header=observation.to_header(),
            joint_stellar_model=True,
            **array_options,
        )
    else:
        result = correct_arrays(
            wavelength=np.asarray(wavelength, dtype=float),
            flux=np.asarray(flux, dtype=float),
            uncertainty=uncertainty,
            mask=mask,
            group_id=group_id,
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
            observation=observation,
            region_file=region_file,
            **fit_options,
        )
    return _finalize_correction(
        result,
        input_label="<wavelength/flux arrays>",
        output_path=output_path,
        product_path=product_path,
        product_format=product_format,
        plot_path=plot_path,
        show_plot=show_plot,
        report=report,
    )


def _inherit_correct_file_parameter_docs() -> None:
    """Add shared detailed parameter descriptions to the unified API docs."""

    target_doc = correct.__doc__ or ""
    source_doc = correct_file.__doc__ or ""
    inherited = []
    for line in source_doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith(":param "):
            continue
        parameter = stripped.removeprefix(":param ").split(":", 1)[0]
        if f":param {parameter}:" not in target_doc:
            inherited.append(stripped)
    if inherited:
        correct.__doc__ = target_doc.rstrip() + "\n\n" + "\n".join(inherited)


_inherit_correct_file_parameter_docs()


def _finalize_correction(
    result: TelluricFitResult,
    *,
    input_label: str | Path,
    output_path: str | Path | None,
    product_path: str | Path | None,
    product_format: str,
    plot_path: str | Path | None,
    show_plot: bool,
    report: bool,
) -> TelluricFitResult:
    if report:
        print_fit_summary(result, input_path=input_label)
    if output_path is not None:
        save_corrected_txt(result, output_path)
    if product_path is not None:
        if product_format == "ascii.ecsv":
            save_fit_product_ecsv(result, product_path)
        else:
            result.write(product_path, format=product_format)
    if plot_path is not None:
        plot_fit(result, path=plot_path, show=show_plot)
    elif show_plot:
        plot_fit(result, show=True)
    return result


def _correct_spectrum_workflow(
    spectrum: Spectrum,
    *,
    line_list: LineList | None,
    line_list_path: str | Path | None,
    hitran_par: str | Path | None,
    hitran_species: tuple[str, ...] | None,
    hitran_min_strength: float | None,
    hitran_max_lines: int | None,
    demo_line_list: bool,
    aer_catalog: AERCatalogArtifact | str | Path | None,
    aer_cache_dir: str | Path | None,
    aer_source: str | Path | None,
    aer_offline: bool,
    aer_reuse_molecfit: bool,
    aer_timeout_s: float,
    partition_table: PartitionTable | str | Path | None,
    h2o_continuum: MTCKDH2OContinuum | LBLRTMH2OContinuum | str | Path | None,
    h2o_continuum_foreign_closure: bool,
    co2_continuum: TabulatedContinuum | LBLRTMCO2Continuum | str | Path | None,
    o2_cia: HitranCIATable | str | Path | None,
    n2_cia: HitranCIATable | str | Path | None,
    cia_tables: Mapping[str, HitranCIATable | str | Path] | None,
    components: tuple[AbsorptionComponent, ...] | None,
    physical: bool | None,
    atmosphere: AtmosphereProfile | None,
    atmosphere_table: str | Path | None,
    atmosphere_mode: str,
    atmosphere_header: Mapping[str, object] | None,
    mipas_profile: str,
    gdas_profile: str | Path | None,
    gdas_mode: str,
    gdas_cache_dir: str | Path | None,
    gdas_download_timeout_s: float,
    observatory_latitude_deg: float | None,
    observatory_longitude_deg: float | None,
    observatory_altitude_m: float | None,
    allow_default_observatory: bool,
    airmass: float,
    pressure_atm: float,
    temperature_k: float,
    path_length_m: float,
    pwv_mm: float | None,
    relative_humidity_percent: float | None,
    mixing_ratios: Mapping[str, float] | None,
    continuum_order: int,
    solve_continuum_linear: ContinuumSolveMode,
    lsf_sigma_pixels: LSFSigmaInput,
    lsf_box_width_pixels: float,
    lsf_lorentz_fwhm_pixels: LSFLorentzInput,
    lsf_variable_width: LSFVariableWidthMode,
    lsf_reference_wavelength_micron: float | None,
    lsf_kernel_width_fwhm: float,
    lsf_molecfit_voigt: bool,
    high_resolution_grid: bool,
    high_resolution_oversampling: float,
    high_resolution_margin_pixels: float,
    high_resolution_rebin_mode: str,
    radiative_transfer_grid: str,
    radiative_transfer_step_cm: float | None,
    radiative_transfer_max_points: int,
    auto_segment: bool,
    segment_size: float,
    line_cutoff_cm: float | None,
    subtract_cutoff_profile: bool,
    line_taper_cm: float,
    line_wing_mode: str,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_avmass_amu: float,
    lblrtm_hwf3: float,
    rayleigh: bool,
    rayleigh_xrayl: float,
    n2_continuum: bool,
    n2_continuum_xn2cn: float,
    o2_continuum: bool,
    o2_continuum_xo2cn: float,
    line_margin_micron: float,
    min_transmission: float,
    minimum_species_peak_optical_depth: float,
    fit_wavelength_shift: WavelengthFitMode,
    fit_wavelength_polynomial: bool,
    wavelength_polynomial_order: int,
    fit_segment_wavelength_shifts: bool,
    fit_segment_wavelength_polynomial: bool,
    segment_wavelength_polynomial_order: int,
    initial_wavelength_shift: float | None,
    wavelength_shift_bounds: tuple[float, float] | None,
    fit_lsf_sigma: LSFFitMode,
    lsf_sigma_bounds: tuple[float, float] | None,
    fit_lsf_box_width: bool,
    lsf_box_width_bounds: tuple[float, float],
    fit_lsf_lorentz_fwhm: LSFFitMode,
    lsf_lorentz_fwhm_bounds: tuple[float, float] | None,
    fit_ranges: tuple[tuple[float, float], ...] | None,
    exclude_ranges: tuple[tuple[float, float], ...] | None,
    theoretical_spectrum: TheoreticalSpectrum | None,
    stellar_mask_path: str | Path | None,
    joint_stellar_model: bool,
    loss: str,
    f_scale: float,
    ftol: float,
    xtol: float,
    gtol: float,
    estimate_uncertainties: bool,
) -> TelluricFitResult:
    if stellar_mask_path is not None and theoretical_spectrum is None:
        raise ConfigurationError(
            "stellar_mask_path requires theoretical_spectrum=TheoreticalSpectrum(...)"
        )
    if joint_stellar_model and theoretical_spectrum is None:
        raise ConfigurationError(
            "joint_stellar_model=True requires "
            "theoretical_spectrum=TheoreticalSpectrum(...)"
        )
    _validate_correction_options(
        spectrum,
        continuum_order=continuum_order,
        auto_segment=auto_segment,
        segment_size=segment_size,
        radiative_transfer_max_points=radiative_transfer_max_points,
        min_transmission=min_transmission,
        fit_wavelength_shift=fit_wavelength_shift,
        fit_wavelength_polynomial=fit_wavelength_polynomial,
        fit_segment_wavelength_shifts=fit_segment_wavelength_shifts,
        fit_segment_wavelength_polynomial=fit_segment_wavelength_polynomial,
        wavelength_polynomial_order=wavelength_polynomial_order,
        segment_wavelength_polynomial_order=segment_wavelength_polynomial_order,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        loss=loss,
        f_scale=f_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        aer_timeout_s=aer_timeout_s,
        gdas_download_timeout_s=gdas_download_timeout_s,
    )
    input_spectrum = spectrum
    input_medium = spectrum.wavelength_medium
    fit_ranges = _ranges_to_observatory_vacuum(fit_ranges, input_medium, atmosphere_header)
    exclude_ranges = _ranges_to_observatory_vacuum(exclude_ranges, input_medium, atmosphere_header)
    spectrum = _spectrum_to_observatory_vacuum(spectrum, atmosphere_header)
    stellar_mask_result: StellarMaskResult | None = None
    stellar_fit_weights: np.ndarray | None = None
    stellar_forward_model: StellarForwardModel | None = None
    if theoretical_spectrum is not None:
        resolution = _estimate_lsf_sigma_from_resolving_power(
            spectrum,
            atmosphere_header,
        )
        resolving_power = (
            None
            if resolution is None
            else float(resolution["resolving_power"])
        )
        stellar_mask_result = theoretical_spectrum.build_mask(
            spectrum,
            frame_correction_factor=_stellar_template_frame_correction_factor(
                spectrum,
                atmosphere_header,
            ),
            resolving_power=resolving_power,
        )
        if joint_stellar_model:
            stellar_forward_model = StellarForwardModel(
                wavelength_micron=np.asarray(
                    stellar_mask_result.intrinsic_wavelength_micron,
                    dtype=float,
                ),
                normalized_flux=np.asarray(
                    stellar_mask_result.intrinsic_normalized_flux,
                    dtype=float,
                ),
            )
        if theoretical_spectrum.confidence_weighted_masking:
            stellar_fit_weights = stellar_mask_result.fit_weights
            if stellar_fit_weights is None:
                raise RuntimeError(
                    "confidence-weighted stellar masking did not produce weights"
                )
        else:
            exclude_ranges = _merge_exclusion_ranges(
                exclude_ranges,
                stellar_mask_result.selection.exclude_ranges,
            )
        masked_fraction = float(
            stellar_mask_result.diagnostics.get("masked_fraction_of_covered", 0.0)
        )
        if masked_fraction > 0.75:
            warnings.warn(
                "The theoretical stellar template excludes more than 75 percent "
                "of covered pixels; consider increasing mask_depth.",
                RuntimeWarning,
                stacklevel=2,
            )
        if stellar_mask_path is not None:
            stellar_mask_result.selection_for_spectrum(input_spectrum).write(
                stellar_mask_path
            )
    spectrum_wavenumber = wavelength_micron_to_wavenumber_cm(
        spectrum.to_unit("micron").wavelength
    )
    finite_wavenumber = spectrum_wavenumber[np.isfinite(spectrum_wavenumber)]
    reference_wavenumber_cm = (
        float(np.nanmedian(finite_wavenumber))
        if finite_wavenumber.size
        else 10_000.0
    )
    resolved_initial_wavelength_shift = _resolve_initial_wavelength_shift(
        spectrum,
        initial_wavelength_shift,
        atmosphere_header,
    )
    resolved_h2o_continuum = _resolve_h2o_continuum(h2o_continuum)
    resolved_co2_continuum = _resolve_tabulated_continuum(co2_continuum)
    resolved_o2_cia = _resolve_cia_table(o2_cia)
    resolved_n2_cia = _resolve_cia_table(n2_cia)
    resolved_pair_cia_tables = _resolve_pair_cia_tables(cia_tables)
    has_component_options = any(
        value is not None
        for value in (components, resolved_co2_continuum, resolved_o2_cia, resolved_n2_cia)
    ) or n2_continuum or o2_continuum
    has_component_options = has_component_options or bool(resolved_pair_cia_tables)
    resolved_line_list = _resolve_line_list(
        spectrum,
        line_list=line_list,
        line_list_path=line_list_path,
        hitran_par=hitran_par,
        hitran_species=hitran_species,
        hitran_min_strength=hitran_min_strength,
        hitran_max_lines=hitran_max_lines,
        demo_line_list=demo_line_list,
        aer_catalog=aer_catalog,
        aer_cache_dir=aer_cache_dir,
        aer_source=aer_source,
        aer_offline=aer_offline,
        aer_reuse_molecfit=aer_reuse_molecfit,
        aer_timeout_s=aer_timeout_s,
        line_cutoff_cm=line_cutoff_cm,
        line_wing_mode=line_wing_mode,
        lblrtm_sample=lblrtm_sample,
        lblrtm_alfal0=lblrtm_alfal0,
        lblrtm_hwf3=lblrtm_hwf3,
        allow_empty_hitran=resolved_h2o_continuum is not None or has_component_options or rayleigh,
    )
    has_physical_lines = bool(
        resolved_line_list.has_hitran_parameters
        and resolved_line_list.wavenumber is not None
        and resolved_line_list.wavenumber.size > 0
    )
    (
        resolved_lsf_sigma_pixels,
        resolved_fit_lsf_sigma,
        resolved_lsf_sigma_bounds,
        initialize_lsf_sigma_grid,
        lsf_sigma_resolution,
    ) = _resolve_lsf_sigma(
        spectrum,
        lsf_sigma_pixels=lsf_sigma_pixels,
        fit_lsf_sigma=fit_lsf_sigma,
        lsf_sigma_bounds=lsf_sigma_bounds,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        atmosphere_header=atmosphere_header,
        has_telluric_lines=resolved_line_list.wavelength.size > 0,
    )
    (
        resolved_lsf_lorentz_fwhm_pixels,
        resolved_fit_lsf_lorentz_fwhm,
        resolved_lsf_lorentz_fwhm_bounds,
        auto_select_lsf_lorentz,
        lsf_lorentz_resolution,
    ) = _resolve_lsf_lorentz(
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        fit_lsf_lorentz_fwhm=fit_lsf_lorentz_fwhm,
        lsf_lorentz_fwhm_bounds=lsf_lorentz_fwhm_bounds,
        gaussian_sigma_pixels=resolved_lsf_sigma_pixels,
        has_telluric_lines=resolved_line_list.wavelength.size > 0,
    )
    if lsf_variable_width != "auto" and not isinstance(lsf_variable_width, bool):
        raise ValueError("lsf_variable_width must be 'auto', True, or False")
    auto_select_lsf_variable_width = lsf_variable_width == "auto"
    resolved_lsf_variable_width = bool(lsf_variable_width is True)
    finite_wavelength = spectrum.to_unit("micron").wavelength
    finite_wavelength = finite_wavelength[np.isfinite(finite_wavelength)]
    if finite_wavelength.size == 0:
        raise ValueError("LSF configuration requires at least one finite wavelength")
    resolved_lsf_reference_wavelength_micron = (
        float(np.nanmedian(finite_wavelength))
        if lsf_reference_wavelength_micron is None
        else float(lsf_reference_wavelength_micron)
    )
    if (
        not np.isfinite(resolved_lsf_reference_wavelength_micron)
        or resolved_lsf_reference_wavelength_micron <= 0
    ):
        raise ValueError("lsf_reference_wavelength_micron must be positive")
    lsf_variable_width_resolution: dict[str, object] = {
        "requested": (
            "auto"
            if auto_select_lsf_variable_width
            else bool(lsf_variable_width)
        ),
        "reference_wavelength_micron": (
            resolved_lsf_reference_wavelength_micron
        ),
        "exponent_bounds": list(AUTO_LSF_VARIABLE_EXPONENT_BOUNDS),
        "selected_model": (
            "pending_pilot_selection"
            if auto_select_lsf_variable_width
            else (
                "fixed_power_law"
                if resolved_lsf_variable_width
                else "constant"
            )
        ),
    }
    if components is None and has_physical_lines:
        line_species = set(resolved_line_list.species_names)
        if h2o_continuum is None and "H2O" in line_species:
            resolved_h2o_continuum = LBLRTMH2OContinuum.from_package_data()
        if co2_continuum is None and "CO2" in line_species:
            resolved_co2_continuum = LBLRTMCO2Continuum.from_package_data()
    resolved_high_resolution_grid = bool(
        high_resolution_grid
        and has_physical_lines
    )
    resolved_components = _build_components(
        extra_components=components,
        line_list=resolved_line_list,
        chunk_size=0,
        partition_table=None,
        line_cutoff_cm=line_cutoff_cm,
        subtract_cutoff_profile=subtract_cutoff_profile,
        line_taper_cm=line_taper_cm,
        line_wing_mode=line_wing_mode,
        lblrtm_sample=lblrtm_sample,
        lblrtm_alfal0=lblrtm_alfal0,
        lblrtm_avmass_amu=lblrtm_avmass_amu,
        lblrtm_hwf3=lblrtm_hwf3,
        rayleigh=rayleigh,
        rayleigh_xrayl=rayleigh_xrayl,
        n2_continuum=n2_continuum,
        n2_continuum_xn2cn=n2_continuum_xn2cn,
        o2_continuum=o2_continuum,
        o2_continuum_xo2cn=o2_continuum_xo2cn,
        h2o_continuum=resolved_h2o_continuum,
        h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
        co2_continuum=resolved_co2_continuum,
        o2_cia=resolved_o2_cia,
        n2_cia=resolved_n2_cia,
        cia_tables=resolved_pair_cia_tables,
    )
    resolved_physical = _resolve_physical(
        physical=physical,
        atmosphere=atmosphere,
        atmosphere_table=atmosphere_table,
        hitran_par=hitran_par,
        line_list=resolved_line_list,
        h2o_continuum=resolved_h2o_continuum,
        components=resolved_components,
    )
    resolved_atmosphere = None
    fit_airmass = airmass
    if resolved_physical:
        if atmosphere is not None and atmosphere_table is not None:
            raise ValueError("provide either atmosphere or atmosphere_table, not both")
        if atmosphere is not None:
            resolved_atmosphere = atmosphere
        else:
            resolved_atmosphere = _make_atmosphere(
                atmosphere_table=atmosphere_table,
                atmosphere_mode=atmosphere_mode,
                atmosphere_header=atmosphere_header,
                mipas_profile=mipas_profile,
                gdas_profile=gdas_profile,
                gdas_mode=gdas_mode,
                gdas_cache_dir=gdas_cache_dir,
                gdas_download_timeout_s=gdas_download_timeout_s,
                observatory_latitude_deg=observatory_latitude_deg,
                observatory_longitude_deg=observatory_longitude_deg,
                observatory_altitude_m=observatory_altitude_m,
                allow_default_observatory=allow_default_observatory,
                airmass=airmass,
                pressure_atm=pressure_atm,
                temperature_k=temperature_k,
                path_length_m=path_length_m,
                pwv_mm=pwv_mm,
                relative_humidity_percent=relative_humidity_percent,
                mixing_ratios=mixing_ratios,
                reference_wavenumber_cm=reference_wavenumber_cm,
            )
            fit_airmass = 1.0

    resolved_partition = _resolve_partition_table(partition_table)
    if resolved_components is not None:
        resolved_components = _build_components(
            extra_components=components,
            line_list=resolved_line_list,
            chunk_size=0,
            partition_table=resolved_partition,
            line_cutoff_cm=line_cutoff_cm,
            subtract_cutoff_profile=subtract_cutoff_profile,
            line_taper_cm=line_taper_cm,
            line_wing_mode=line_wing_mode,
            lblrtm_sample=lblrtm_sample,
            lblrtm_alfal0=lblrtm_alfal0,
            lblrtm_avmass_amu=lblrtm_avmass_amu,
            lblrtm_hwf3=lblrtm_hwf3,
            rayleigh=rayleigh,
            rayleigh_xrayl=rayleigh_xrayl,
            n2_continuum=n2_continuum,
            n2_continuum_xn2cn=n2_continuum_xn2cn,
            o2_continuum=o2_continuum,
            o2_continuum_xo2cn=o2_continuum_xo2cn,
            h2o_continuum=resolved_h2o_continuum,
            h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
            co2_continuum=resolved_co2_continuum,
            o2_cia=resolved_o2_cia,
            n2_cia=resolved_n2_cia,
            cia_tables=resolved_pair_cia_tables,
        )
    fixed_component_scales: dict[str, float] = {}
    if n2_continuum:
        fixed_component_scales["N2_continuum"] = 1.0
    if o2_continuum:
        fixed_component_scales["O2_continuum"] = 1.0
    if rayleigh:
        fixed_component_scales["Rayleigh"] = 1.0

    if solve_continuum_linear != "auto" and not isinstance(
        solve_continuum_linear,
        bool,
    ):
        raise ValueError("solve_continuum_linear must be 'auto', True, or False")
    automatic_continuum_solver = solve_continuum_linear == "auto"
    use_linear_continuum = (
        True if automatic_continuum_solver else bool(solve_continuum_linear)
    )
    if fit_wavelength_shift != "auto" and not isinstance(
        fit_wavelength_shift,
        bool,
    ):
        raise ValueError("fit_wavelength_shift must be 'auto', True, or False")
    explicit_wavelength_model = bool(
        fit_wavelength_polynomial
        or fit_segment_wavelength_shifts
        or fit_segment_wavelength_polynomial
    )
    if fit_wavelength_shift is True and explicit_wavelength_model:
        raise ValueError(
            "fit_wavelength_shift=True cannot be combined with another "
            "wavelength-correction model"
        )
    auto_select_wavelength_model = bool(
        fit_wavelength_shift == "auto" and not explicit_wavelength_model
    )
    resolved_fit_wavelength_shift = bool(fit_wavelength_shift is True)
    automatic_pixel_wavelength_bounds = bool(
        wavelength_shift_bounds is None
        and (
            auto_select_wavelength_model
            or fit_segment_wavelength_shifts
            or fit_segment_wavelength_polynomial
        )
    )
    if wavelength_shift_bounds is None:
        resolved_wavelength_shift_bounds = (
            AUTO_WAVELENGTH_SHIFT_BOUNDS_PIXELS
            if automatic_pixel_wavelength_bounds
            else (-5.0e-4, 5.0e-4)
        )
    else:
        resolved_wavelength_shift_bounds = tuple(
            float(value) for value in wavelength_shift_bounds
        )
    if (
        len(resolved_wavelength_shift_bounds) != 2
        or not np.all(np.isfinite(resolved_wavelength_shift_bounds))
        or resolved_wavelength_shift_bounds[1]
        <= resolved_wavelength_shift_bounds[0]
    ):
        raise ValueError("wavelength_shift_bounds must be finite and increasing")
    wavelength_shift_unit = "pixel" if automatic_pixel_wavelength_bounds else "micron"

    fit_initial_wavelength_shift = (
        _micron_shift_to_pixel(
            spectrum,
            resolved_initial_wavelength_shift,
        )
        if automatic_pixel_wavelength_bounds
        else resolved_initial_wavelength_shift
    )
    fit_config = FitConfig(
        airmass=fit_airmass,
        continuum_order=continuum_order,
        fixed_species_scales=fixed_component_scales or None,
        solve_continuum_linear=use_linear_continuum,
        lsf_sigma_pixels=resolved_lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=resolved_lsf_lorentz_fwhm_pixels,
        lsf_variable_width=resolved_lsf_variable_width,
        lsf_reference_wavelength_micron=(
            resolved_lsf_reference_wavelength_micron
        ),
        lsf_wavelength_exponent=(
            1.0 if resolved_lsf_variable_width else 0.0
        ),
        fit_lsf_wavelength_exponent=False,
        lsf_wavelength_exponent_bounds=AUTO_LSF_VARIABLE_EXPONENT_BOUNDS,
        lsf_kernel_width_fwhm=lsf_kernel_width_fwhm,
        lsf_molecfit_voigt=lsf_molecfit_voigt,
        high_resolution_grid=resolved_high_resolution_grid,
        high_resolution_oversampling=high_resolution_oversampling,
        high_resolution_margin_pixels=high_resolution_margin_pixels,
        high_resolution_rebin_mode=high_resolution_rebin_mode,
        radiative_transfer_grid=radiative_transfer_grid,
        radiative_transfer_step_cm=radiative_transfer_step_cm,
        radiative_transfer_max_points=radiative_transfer_max_points,
        line_cutoff_cm=line_cutoff_cm,
        subtract_cutoff_profile=subtract_cutoff_profile,
        line_taper_cm=line_taper_cm,
        line_wing_mode=line_wing_mode,
        lblrtm_sample=lblrtm_sample,
        lblrtm_alfal0=lblrtm_alfal0,
        lblrtm_avmass_amu=lblrtm_avmass_amu,
        lblrtm_hwf3=lblrtm_hwf3,
        rayleigh=rayleigh,
        rayleigh_xrayl=rayleigh_xrayl,
        n2_continuum=n2_continuum,
        n2_continuum_xn2cn=n2_continuum_xn2cn,
        o2_continuum=o2_continuum,
        o2_continuum_xo2cn=o2_continuum_xo2cn,
        line_margin_micron=line_margin_micron,
        min_transmission=min_transmission,
        minimum_species_peak_optical_depth=minimum_species_peak_optical_depth,
        atmosphere=resolved_atmosphere,
        partition_table=resolved_partition,
        h2o_continuum=resolved_h2o_continuum,
        h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
        components=resolved_components,
        fit_wavelength_shift=resolved_fit_wavelength_shift,
        fit_wavelength_polynomial=fit_wavelength_polynomial,
        wavelength_polynomial_order=wavelength_polynomial_order,
        fit_segment_wavelength_shifts=fit_segment_wavelength_shifts,
        fit_segment_wavelength_polynomial=fit_segment_wavelength_polynomial,
        segment_wavelength_polynomial_order=segment_wavelength_polynomial_order,
        initial_wavelength_shift=fit_initial_wavelength_shift,
        wavelength_shift_bounds=resolved_wavelength_shift_bounds,
        wavelength_shift_unit=wavelength_shift_unit,
        fit_lsf_sigma=resolved_fit_lsf_sigma,
        lsf_sigma_bounds=resolved_lsf_sigma_bounds,
        initialize_lsf_sigma_grid=initialize_lsf_sigma_grid,
        fit_lsf_box_width=fit_lsf_box_width,
        lsf_box_width_bounds=lsf_box_width_bounds,
        fit_lsf_lorentz_fwhm=resolved_fit_lsf_lorentz_fwhm,
        lsf_lorentz_fwhm_bounds=resolved_lsf_lorentz_fwhm_bounds,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        loss=loss,
        f_scale=f_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=(
            AUTO_LINEAR_CONTINUUM_MAX_NFEV
            if (
                automatic_continuum_solver
                and use_linear_continuum
                and loss == "linear"
            )
            else None
        ),
        estimate_uncertainties=estimate_uncertainties,
    )
    if auto_segment and (not np.isfinite(segment_size) or segment_size <= 0):
        raise ValueError("segment_size must be a positive finite value in microns")
    physical_group_preview: tuple[Spectrum, ...] = ()
    physical_group_ids: set[int] = set()
    automatic_physical_group_wavelength = False
    if auto_segment:
        physical_group_preview = _split_spectrum(
            spectrum,
            segment_size=segment_size,
            minimum_points=continuum_order + 2,
        )
        physical_group_ids = {
            int(group.meta.get("physical_group_index", index))
            for index, group in enumerate(physical_group_preview)
        }
        automatic_physical_group_wavelength = bool(
            auto_select_wavelength_model
            and len(physical_group_ids) > 1
            and all(
                group.meta.get("segmentation_source")
                in {"fits_order_detector", "wavelength_gaps"}
                for group in physical_group_preview
            )
        )
    if automatic_physical_group_wavelength:
        fit_config = replace(
            fit_config,
            fit_wavelength_shift=False,
            fit_wavelength_polynomial=False,
            fit_segment_wavelength_shifts=True,
            fit_segment_wavelength_polynomial=False,
            wavelength_shift_unit="pixel",
            wavelength_shift_bounds=AUTO_WAVELENGTH_SHIFT_BOUNDS_PIXELS,
            initial_wavelength_shift=0.0,
        )
        auto_select_wavelength_model = False

    def run_fit(config: FitConfig) -> TelluricFitResult:
        single_fit_options = {
            **(
                {}
                if stellar_fit_weights is None
                else {"fit_weights": stellar_fit_weights}
            ),
            **(
                {}
                if stellar_forward_model is None
                else {"stellar_model": stellar_forward_model}
            ),
        }
        per_segment_wavelength_fit = bool(
            config.fit_segment_wavelength_shifts
            or config.fit_segment_wavelength_polynomial
        )
        if not auto_segment:
            if per_segment_wavelength_fit:
                raise ValueError(
                    "per-segment wavelength fitting requires auto_segment=True"
                )
            return fit_tellurics(
                spectrum,
                line_list=resolved_line_list,
                config=config,
                **single_fit_options,
            )
        if not resolved_high_resolution_grid and not per_segment_wavelength_fit:
            return fit_tellurics(
                spectrum,
                line_list=resolved_line_list,
                config=config,
                **single_fit_options,
            )
        segments = physical_group_preview or _split_spectrum(
            spectrum,
            segment_size=segment_size,
            minimum_points=continuum_order + 2,
        )
        if resolved_high_resolution_grid:
            segments = _subdivide_segments_for_grid_limit(
                segments,
                config=config,
                minimum_points=continuum_order + 2,
            )
        if len(segments) == 1 and not per_segment_wavelength_fit:
            return fit_tellurics(
                spectrum,
                line_list=resolved_line_list,
                config=config,
                **single_fit_options,
            )
        active = tuple(
            _segment_has_fit_pixels(segment, config)
            for segment in segments
        )
        active_segments = tuple(
            segment
            for segment, is_active in zip(segments, active, strict=True)
            if is_active
        )
        active_group_ids = tuple(
            int(segment.meta.get("physical_group_index", index))
            for index, (segment, is_active) in enumerate(
                zip(segments, active, strict=True)
            )
            if is_active
        )
        active_fit_weights = (
            None
            if stellar_fit_weights is None
            else tuple(
                _weights_for_segment(
                    spectrum,
                    stellar_fit_weights,
                    segment,
                )
                for segment, is_active in zip(segments, active, strict=True)
                if is_active
            )
        )
        active_stellar_models = (
            None
            if stellar_forward_model is None
            else tuple(
                stellar_forward_model
                for segment, is_active in zip(segments, active, strict=True)
                if is_active
            )
        )
        physical_group_bounds = {
            int(segment.meta.get("physical_group_index", index)): tuple(
                float(value)
                for value in segment.meta.get(
                    "physical_group_bounds_micron",
                    (
                        float(np.nanmin(segment.wavelength)),
                        float(np.nanmax(segment.wavelength)),
                    ),
                )
            )
            for index, segment in enumerate(segments)
        }
        if not active_segments:
            raise ValueError(
                "fit_ranges and exclude_ranges leave no segment with enough fit pixels"
            )
        full_wavelength_micron = spectrum.to_unit("micron").wavelength
        full_bounds = (
            float(np.nanmin(full_wavelength_micron)),
            float(np.nanmax(full_wavelength_micron)),
        )
        multi_result = fit_telluric_segments(
            active_segments,
            line_list=resolved_line_list,
            config=config,
            fit_weights=active_fit_weights,
            stellar_models=active_stellar_models,
            global_wavelength_bounds=full_bounds,
            wavelength_group_ids=active_group_ids,
            wavelength_group_bounds=physical_group_bounds,
        )
        fitted_results = iter(multi_result.segment_results)
        segment_results = tuple(
            next(fitted_results)
            if is_active
            else _apply_multi_fit_to_segment(
                segment,
                line_list=resolved_line_list,
                config=config,
                fit_result=multi_result,
                stellar_model=(
                    None
                    if stellar_forward_model is None
                    else stellar_forward_model
                ),
                global_wavelength_bounds=full_bounds,
                wavelength_group_id=int(
                    segment.meta.get("physical_group_index", segment_index)
                ),
            )
            for segment_index, (segment, is_active) in enumerate(
                zip(segments, active, strict=True)
            )
        )
        return _stitch_segment_results(
            multi_result,
            segment_size=segment_size,
            segment_results=segment_results,
        )

    wavelength_model_resolution: dict[str, object] = {
        "requested": (
            "auto"
            if fit_wavelength_shift == "auto"
            else bool(fit_wavelength_shift)
        ),
        "coefficient_unit": fit_config.wavelength_shift_unit,
        "bounds": list(fit_config.wavelength_shift_bounds),
        "selected_model": (
            "pending_pilot_selection"
            if auto_select_wavelength_model
            else _configured_wavelength_model_name(fit_config)
        ),
    }
    if automatic_physical_group_wavelength:
        wavelength_model_resolution = {
            **wavelength_model_resolution,
            "selected_model": "physical_group_constant",
            "selection_reason": "detected_independent_order_or_detector_groups",
            "physical_group_count": len(physical_group_ids),
        }
    pilot_fit_config = fit_config
    if stellar_fit_weights is not None and stellar_mask_result is not None:
        pilot_fit_config = replace(
            fit_config,
            exclude_ranges=_merge_exclusion_ranges(
                fit_config.exclude_ranges,
                stellar_mask_result.selection.exclude_ranges,
            ),
        )
    if auto_select_wavelength_model:
        try:
            selected_config, wavelength_model_resolution = (
                _select_wavelength_model_from_pilots(
                    spectrum,
                    line_list=resolved_line_list,
                    config=pilot_fit_config,
                    segment_size=segment_size,
                    resolution=wavelength_model_resolution,
                )
            )
            fit_config = replace(
                selected_config,
                exclude_ranges=fit_config.exclude_ranges,
            )
            pilot_fit_config = replace(
                selected_config,
                exclude_ranges=pilot_fit_config.exclude_ranges,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            fit_config = replace(
                fit_config,
                fit_wavelength_shift=False,
                fit_wavelength_polynomial=False,
                wavelength_shift_unit="pixel",
            )
            wavelength_model_resolution = {
                **wavelength_model_resolution,
                "selected_model": "none",
                "selection_reason": "pilot_fit_error",
                "pilot_error": str(exc),
            }
            warnings.warn(
                "Automatic wavelength-alignment pilot fitting was not usable; "
                "continuing without a fitted residual wavelength correction.",
                RuntimeWarning,
                stacklevel=2,
            )
        wavelength_pilot_coarse_search = (
            wavelength_model_resolution.get("none_model", {})
            .get("coarse_search")
        )
        if wavelength_pilot_coarse_search is not None:
            lsf_sigma_resolution = {
                **lsf_sigma_resolution,
                "coarse_search": wavelength_pilot_coarse_search,
            }

    if auto_select_lsf_variable_width:
        try:
            selected_config, lsf_variable_width_resolution = (
                _select_lsf_variable_width_from_pilots(
                    spectrum,
                    line_list=resolved_line_list,
                    config=pilot_fit_config,
                    segment_size=segment_size,
                    resolution=lsf_variable_width_resolution,
                )
            )
            fit_config = replace(
                selected_config,
                exclude_ranges=fit_config.exclude_ranges,
            )
            pilot_fit_config = replace(
                selected_config,
                exclude_ranges=pilot_fit_config.exclude_ranges,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            fit_config = replace(
                fit_config,
                lsf_variable_width=False,
                fit_lsf_wavelength_exponent=False,
                lsf_wavelength_exponent=0.0,
            )
            lsf_variable_width_resolution = {
                **lsf_variable_width_resolution,
                "selected_model": "constant",
                "selection_reason": "pilot_fit_error",
                "pilot_error": str(exc),
            }
            warnings.warn(
                "Automatic wavelength-dependent LSF pilot fitting was not "
                "usable; continuing with a constant-width LSF.",
                RuntimeWarning,
                stacklevel=2,
            )

    if auto_select_lsf_lorentz:
        try:
            selected_config, lsf_lorentz_resolution = _select_lsf_lorentz_from_pilots(
                spectrum,
                line_list=resolved_line_list,
                config=pilot_fit_config,
                segment_size=segment_size,
                resolution=lsf_lorentz_resolution,
            )
            fit_config = replace(
                selected_config,
                exclude_ranges=fit_config.exclude_ranges,
            )
            pilot_fit_config = replace(
                selected_config,
                exclude_ranges=pilot_fit_config.exclude_ranges,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            lsf_lorentz_resolution = {
                **lsf_lorentz_resolution,
                "selected_model": "gaussian",
                "selection_reason": "pilot_fit_error",
                "pilot_error": str(exc),
            }
            warnings.warn(
                "Automatic Lorentzian LSF pilot fitting was not usable; "
                "continuing with a Gaussian-only LSF.",
                RuntimeWarning,
                stacklevel=2,
            )
        pilot_coarse_search = (
            lsf_lorentz_resolution.get("gaussian_model", {})
            .get("coarse_search")
        )
        if pilot_coarse_search is not None:
            lsf_sigma_resolution = {
                **lsf_sigma_resolution,
                "coarse_search": pilot_coarse_search,
            }

    initial_solver = "linear" if use_linear_continuum else "nonlinear"
    initial_result = run_fit(fit_config)
    attempts = [_continuum_solver_attempt(initial_solver, initial_result)]
    selected_result = initial_result
    selected_solver = initial_solver
    fallback_reason = None

    if automatic_continuum_solver and use_linear_continuum and loss == "linear":
        fallback_reason = _continuum_fit_problem(initial_result)
        if fallback_reason is not None:
            warnings.warn(
                "The automatic linear continuum fit was not usable "
                f"({fallback_reason}); retrying with nonlinear continuum fitting.",
                RuntimeWarning,
                stacklevel=2,
            )
            fallback_config = replace(
                fit_config,
                solve_continuum_linear=False,
                max_nfev=None,
            )
            selected_result = run_fit(fallback_config)
            selected_solver = "nonlinear"
            attempts.append(
                _continuum_solver_attempt(selected_solver, selected_result)
            )

    selection_reason = None
    if (
        automatic_continuum_solver
        and selected_solver == "linear"
        and loss != "linear"
    ):
        selection_reason = (
            f"loss={loss!r} uses robust iteratively reweighted continuum profiling"
        )
    selected_result = _with_continuum_solver_provenance(
        selected_result,
        requested=(
            "auto"
            if automatic_continuum_solver
            else ("linear" if use_linear_continuum else "nonlinear")
        ),
        selected=selected_solver,
        attempts=attempts,
        fallback_reason=fallback_reason,
        selection_reason=selection_reason,
    )
    lsf_bound_status = selected_result.parameter_bound_status.get(
        "lsf_sigma_pixels"
    )
    if lsf_sigma_resolution["requested"] == "auto" and lsf_bound_status is not None:
        warnings.warn(
            "The automatically fitted Gaussian LSF sigma reached its "
            f"{lsf_bound_status} bound; inspect the fit or provide explicit bounds.",
            RuntimeWarning,
            stacklevel=2,
        )
    selected_result = _with_lsf_sigma_provenance(
        selected_result,
        resolution=lsf_sigma_resolution,
        bounds=resolved_lsf_sigma_bounds,
        fit_enabled=resolved_fit_lsf_sigma,
        bound_status=lsf_bound_status,
    )
    lsf_lorentz_bound_status = selected_result.parameter_bound_status.get(
        "lsf_lorentz_fwhm_pixels"
    )
    if (
        lsf_lorentz_resolution.get("selected_model") == "gaussian_lorentz"
        and lsf_lorentz_bound_status is not None
    ):
        warnings.warn(
            "The automatically selected Lorentzian LSF FWHM reached its "
            f"{lsf_lorentz_bound_status} bound in the full fit; inspect the fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    selected_result = _with_lsf_lorentz_provenance(
        selected_result,
        resolution=lsf_lorentz_resolution,
        bounds=resolved_lsf_lorentz_fwhm_bounds,
        fit_enabled=fit_config.fit_lsf_lorentz_fwhm,
        bound_status=lsf_lorentz_bound_status,
    )
    lsf_exponent_bound_status = selected_result.parameter_bound_status.get(
        "lsf_wavelength_exponent"
    )
    if (
        lsf_variable_width_resolution.get("selected_model") == "power_law"
        and lsf_exponent_bound_status is not None
    ):
        warnings.warn(
            "The automatically selected LSF wavelength exponent reached its "
            f"{lsf_exponent_bound_status} bound in the full fit; inspect the fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    selected_result = _with_lsf_variable_width_provenance(
        selected_result,
        resolution=lsf_variable_width_resolution,
        config=fit_config,
        bound_status=lsf_exponent_bound_status,
    )
    wavelength_bound_status = {
        parameter: status
        for parameter, status in selected_result.parameter_bound_status.items()
        if "wavelength_" in parameter
    }
    if auto_select_wavelength_model and wavelength_bound_status:
        warnings.warn(
            "The automatically selected wavelength model reached a coefficient "
            "bound in the full fit; inspect the alignment or provide wider "
            "pixel bounds.",
            RuntimeWarning,
            stacklevel=2,
        )
    selected_result = _with_wavelength_model_provenance(
        selected_result,
        resolution=wavelength_model_resolution,
        config=fit_config,
        bound_status=wavelength_bound_status,
    )
    quality = fit_quality_diagnostics(selected_result)
    median_shift = quality.get("median_absolute_residual_shift_pixels")
    if median_shift is not None and np.isfinite(median_shift) and median_shift > 0.25:
        warnings.warn(
            "Residual telluric alignment remains larger than 0.25 pixel in "
            "the median physical group; inspect fit_quality diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
    return replace(
        selected_result,
        provenance={
            **dict(selected_result.provenance),
            "fit_quality": quality,
            **(
                {}
                if stellar_mask_result is None
                else {
                    "stellar_template": dict(
                        stellar_mask_result.diagnostics
                    )
                    | {"joint_forward_model": bool(joint_stellar_model)}
                }
            ),
        },
    )


def _validate_correction_options(
    spectrum: Spectrum,
    *,
    continuum_order: int,
    auto_segment: bool,
    segment_size: float,
    radiative_transfer_max_points: int,
    min_transmission: float,
    fit_wavelength_shift: WavelengthFitMode,
    fit_wavelength_polynomial: bool,
    fit_segment_wavelength_shifts: bool,
    fit_segment_wavelength_polynomial: bool,
    wavelength_polynomial_order: int,
    segment_wavelength_polynomial_order: int,
    fit_ranges: tuple[tuple[float, float], ...] | None,
    exclude_ranges: tuple[tuple[float, float], ...] | None,
    loss: str,
    f_scale: float,
    ftol: float,
    xtol: float,
    gtol: float,
    aer_timeout_s: float,
    gdas_download_timeout_s: float,
) -> None:
    """Reject invalid public options before line-data or network work starts."""

    if np.count_nonzero(spectrum.valid) < 3:
        raise ConfigurationError(
            "telluric correction requires at least three valid spectrum pixels"
        )
    if not isinstance(continuum_order, (int, np.integer)) or continuum_order < 0:
        raise ConfigurationError("continuum_order must be a non-negative integer")
    if auto_segment and (not np.isfinite(segment_size) or segment_size <= 0):
        raise ConfigurationError("segment_size must be positive and finite")
    if int(radiative_transfer_max_points) < 2:
        raise ConfigurationError("radiative_transfer_max_points must be at least two")
    if not np.isfinite(min_transmission) or not 0 < min_transmission < 1:
        raise ConfigurationError("min_transmission must be between zero and one")
    if fit_wavelength_shift != "auto" and not isinstance(
        fit_wavelength_shift,
        (bool, np.bool_),
    ):
        raise ConfigurationError(
            "fit_wavelength_shift must be 'auto', True, or False"
        )
    explicit_models = sum(
        bool(value)
        for value in (
            fit_wavelength_polynomial,
            fit_segment_wavelength_shifts,
            fit_segment_wavelength_polynomial,
        )
    )
    if explicit_models > 1:
        raise ConfigurationError(
            "choose only one explicit wavelength model: global polynomial, "
            "per-group shifts, or per-group polynomial"
        )
    if fit_wavelength_shift is True and explicit_models:
        raise ConfigurationError(
            "fit_wavelength_shift=True cannot be combined with another "
            "wavelength-correction model"
        )
    if wavelength_polynomial_order < 0 or segment_wavelength_polynomial_order < 0:
        raise ConfigurationError("wavelength polynomial orders must be non-negative")
    for label, ranges in (("fit_ranges", fit_ranges), ("exclude_ranges", exclude_ranges)):
        if ranges is None:
            continue
        for bounds in ranges:
            if len(bounds) != 2:
                raise ConfigurationError(
                    f"each {label} entry must contain two endpoints"
                )
            lower, upper = map(float, bounds)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
                raise ConfigurationError(
                    f"{label} endpoints must be finite and distinct"
                )
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ConfigurationError(f"unsupported least-squares loss: {loss!r}")
    for label, value in (
        ("f_scale", f_scale),
        ("ftol", ftol),
        ("xtol", xtol),
        ("gtol", gtol),
        ("aer_timeout_s", aer_timeout_s),
        ("gdas_download_timeout_s", gdas_download_timeout_s),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ConfigurationError(f"{label} must be positive and finite")


def _continuum_fit_problem(result: TelluricFitResult) -> str | None:
    if not result.success:
        return f"optimizer did not converge: {result.message}"
    scalar_values = {
        "cost": result.cost,
        "wavelength shift": result.wavelength_shift,
        "Gaussian LSF width": result.lsf_sigma_pixels,
        "box LSF width": result.lsf_box_width_pixels,
        "Lorentzian LSF width": result.lsf_lorentz_fwhm_pixels,
        "LSF wavelength exponent": result.lsf_wavelength_exponent,
    }
    scalar_values.update(
        {
            f"{species} scale": scale
            for species, scale in result.species_scales.items()
        }
    )
    invalid = [
        name
        for name, value in scalar_values.items()
        if not np.isfinite(value)
    ]
    if invalid:
        return "non-finite fitted " + ", ".join(invalid)
    nonpositive_scales = [
        species
        for species, scale in result.species_scales.items()
        if scale <= 0
    ]
    if nonpositive_scales:
        return "non-positive molecular scale for " + ", ".join(nonpositive_scales)
    return None


def _continuum_solver_attempt(
    solver: str,
    result: TelluricFitResult,
) -> dict[str, object]:
    return {
        "solver": solver,
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }


def _with_continuum_solver_provenance(
    result: TelluricFitResult,
    *,
    requested: str,
    selected: str,
    attempts: list[dict[str, object]],
    fallback_reason: str | None,
    selection_reason: str | None,
) -> TelluricFitResult:
    details: dict[str, object] = {
        "requested": requested,
        "selected": selected,
        "fallback_used": len(attempts) > 1,
        "attempts": attempts,
    }
    if fallback_reason is not None:
        details["fallback_reason"] = fallback_reason
    if selection_reason is not None:
        details["selection_reason"] = selection_reason
    return replace(
        result,
        provenance={
            **dict(result.provenance),
            "continuum_solver": details,
        },
    )


def _resolve_lsf_sigma(
    spectrum: Spectrum,
    *,
    lsf_sigma_pixels: LSFSigmaInput,
    fit_lsf_sigma: LSFFitMode,
    lsf_sigma_bounds: tuple[float, float] | None,
    fit_ranges: tuple[tuple[float, float], ...] | None,
    exclude_ranges: tuple[tuple[float, float], ...] | None,
    atmosphere_header: Mapping[str, object] | None,
    has_telluric_lines: bool,
) -> tuple[float, bool, tuple[float, float], bool, dict[str, object]]:
    if fit_lsf_sigma != "auto" and not isinstance(fit_lsf_sigma, bool):
        raise ValueError("fit_lsf_sigma must be 'auto', True, or False")

    automatic_sigma = lsf_sigma_pixels == "auto"
    if automatic_sigma:
        resolution = _estimate_lsf_sigma_from_resolving_power(
            spectrum,
            atmosphere_header,
        )
        if resolution is None:
            resolution = _estimate_lsf_sigma_from_spectral_features(
                spectrum,
                fit_ranges=fit_ranges,
                exclude_ranges=exclude_ranges,
            )
        if resolution is None:
            resolution = {
                "requested": "auto",
                "source": "generic_fallback",
                "initial_sigma_pixels": AUTO_LSF_SIGMA_FALLBACK_PIXELS,
            }
        else:
            resolution = {"requested": "auto", **resolution}
        initial_sigma = float(resolution["initial_sigma_pixels"])
    else:
        try:
            initial_sigma = float(lsf_sigma_pixels)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lsf_sigma_pixels must be 'auto' or a non-negative number"
            ) from exc
        if not np.isfinite(initial_sigma) or initial_sigma < 0:
            raise ValueError(
                "lsf_sigma_pixels must be 'auto' or a non-negative number"
            )
        resolution = {
            "requested": initial_sigma,
            "source": "user",
            "initial_sigma_pixels": initial_sigma,
        }

    if fit_lsf_sigma == "auto":
        resolved_fit = bool(
            automatic_sigma
            and has_telluric_lines
            and resolution["source"] != "generic_fallback"
        )
    else:
        resolved_fit = bool(fit_lsf_sigma)

    if automatic_sigma and not has_telluric_lines:
        initial_sigma = 0.0
        resolution = {
            **resolution,
            "source": "disabled_no_telluric_lines",
            "initial_sigma_pixels": 0.0,
        }
    elif (
        automatic_sigma
        and resolution["source"] == "generic_fallback"
        and fit_lsf_sigma is not True
    ):
        initial_sigma = 0.0
        resolution = {
            **resolution,
            "source": "disabled_no_lsf_information",
            "initial_sigma_pixels": 0.0,
        }

    if lsf_sigma_bounds is None:
        if automatic_sigma:
            if resolution["source"] == "fits_resolving_power":
                resolved_bounds = (
                    AUTO_LSF_RESOLUTION_LOWER_FACTOR * initial_sigma,
                    min(
                        AUTO_LSF_SIGMA_MAX_PIXELS,
                        AUTO_LSF_RESOLUTION_UPPER_FACTOR * initial_sigma,
                    ),
                )
            else:
                upper = min(
                    AUTO_LSF_SIGMA_MAX_PIXELS,
                    max(6.0, 4.0 * max(initial_sigma, 1.0)),
                )
                resolved_bounds = (0.0, upper)
        else:
            resolved_bounds = DEFAULT_LSF_SIGMA_BOUNDS
    else:
        if len(lsf_sigma_bounds) != 2:
            raise ValueError("lsf_sigma_bounds must contain two values")
        resolved_bounds = (
            float(lsf_sigma_bounds[0]),
            float(lsf_sigma_bounds[1]),
        )
    if (
        not np.all(np.isfinite(resolved_bounds))
        or resolved_bounds[0] < 0
        or resolved_bounds[1] <= resolved_bounds[0]
    ):
        raise ValueError(
            "lsf_sigma_bounds must be finite, non-negative, and increasing"
        )

    initialize_grid = bool(
        automatic_sigma
        and resolved_fit
        and resolution["source"] != "fits_resolving_power"
    )
    resolution = {
        **resolution,
        "fit_requested": (
            "auto" if fit_lsf_sigma == "auto" else bool(fit_lsf_sigma)
        ),
    }
    return (
        initial_sigma,
        resolved_fit,
        resolved_bounds,
        initialize_grid,
        resolution,
    )


def _resolve_lsf_lorentz(
    *,
    lsf_lorentz_fwhm_pixels: LSFLorentzInput,
    fit_lsf_lorentz_fwhm: LSFFitMode,
    lsf_lorentz_fwhm_bounds: tuple[float, float] | None,
    gaussian_sigma_pixels: float,
    has_telluric_lines: bool,
) -> tuple[float, bool, tuple[float, float], bool, dict[str, object]]:
    if (
        fit_lsf_lorentz_fwhm != "auto"
        and not isinstance(fit_lsf_lorentz_fwhm, bool)
    ):
        raise ValueError(
            "fit_lsf_lorentz_fwhm must be 'auto', True, or False"
        )

    automatic_width = lsf_lorentz_fwhm_pixels == "auto"
    if automatic_width:
        initial_fwhm = 0.0
        source = "automatic_pilot_selection"
    else:
        try:
            initial_fwhm = float(lsf_lorentz_fwhm_pixels)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lsf_lorentz_fwhm_pixels must be 'auto' or a non-negative number"
            ) from exc
        if not np.isfinite(initial_fwhm) or initial_fwhm < 0:
            raise ValueError(
                "lsf_lorentz_fwhm_pixels must be 'auto' or a non-negative number"
            )
        source = "user"

    if lsf_lorentz_fwhm_bounds is None:
        upper = min(
            AUTO_LSF_LORENTZ_MAX_PIXELS,
            max(6.0, 4.0 * max(float(gaussian_sigma_pixels), 1.0)),
        )
        resolved_bounds = (0.0, upper)
    else:
        if len(lsf_lorentz_fwhm_bounds) != 2:
            raise ValueError("lsf_lorentz_fwhm_bounds must contain two values")
        resolved_bounds = (
            float(lsf_lorentz_fwhm_bounds[0]),
            float(lsf_lorentz_fwhm_bounds[1]),
        )
    if (
        not np.all(np.isfinite(resolved_bounds))
        or resolved_bounds[0] < 0
        or resolved_bounds[1] <= resolved_bounds[0]
    ):
        raise ValueError(
            "lsf_lorentz_fwhm_bounds must be finite, non-negative, and increasing"
        )

    auto_select = bool(
        automatic_width
        and fit_lsf_lorentz_fwhm == "auto"
        and has_telluric_lines
    )
    if fit_lsf_lorentz_fwhm == "auto":
        resolved_fit = False
    else:
        resolved_fit = bool(fit_lsf_lorentz_fwhm)
    if automatic_width and resolved_fit:
        gaussian_fwhm = 2.354820045 * max(float(gaussian_sigma_pixels), 0.2)
        initial_fwhm = float(
            np.clip(
                0.5 * gaussian_fwhm,
                resolved_bounds[0] + 0.01 * np.diff(resolved_bounds)[0],
                resolved_bounds[1] - 0.01 * np.diff(resolved_bounds)[0],
            )
        )
        source = "automatic_initial_value_for_forced_fit"
    elif automatic_width and not has_telluric_lines:
        source = "disabled_no_telluric_lines"

    resolution = {
        "requested": (
            "auto" if automatic_width else float(initial_fwhm)
        ),
        "fit_requested": (
            "auto"
            if fit_lsf_lorentz_fwhm == "auto"
            else bool(fit_lsf_lorentz_fwhm)
        ),
        "source": source,
        "selected_model": (
            "pending_pilot_selection"
            if auto_select
            else (
                "gaussian_lorentz"
                if resolved_fit or initial_fwhm > 0
                else "gaussian"
            )
        ),
        "initial_fwhm_pixels": float(initial_fwhm),
    }
    return (
        initial_fwhm,
        resolved_fit,
        resolved_bounds,
        auto_select,
        resolution,
    )


def _configured_wavelength_model_name(config: FitConfig) -> str:
    if config.fit_segment_wavelength_polynomial:
        return "per_segment_polynomial"
    if config.fit_segment_wavelength_shifts:
        return "per_segment_constant"
    if config.fit_wavelength_polynomial:
        return "linear_pixel_trend" if (
            config.wavelength_shift_unit == "pixel"
            and config.wavelength_polynomial_order == 1
        ) else "global_polynomial"
    if config.fit_wavelength_shift:
        return (
            "constant_pixel_shift"
            if config.wavelength_shift_unit == "pixel"
            else "constant_micron_shift"
        )
    return "none"


def _wavelength_candidate_improvement(
    simpler: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[float, float]:
    simpler_rss = np.asarray(simpler["region_weighted_rss"], dtype=float)
    candidate_rss = np.asarray(candidate["region_weighted_rss"], dtype=float)
    informative = np.asarray(
        simpler["region_max_absorption"],
        dtype=float,
    ) >= 0.01
    relative = (
        simpler_rss - candidate_rss
    ) / np.maximum(simpler_rss, np.finfo(float).tiny)
    improved_fraction = (
        float(np.count_nonzero(informative & (relative > 0.0)))
        / float(np.count_nonzero(informative))
        if np.any(informative)
        else 0.0
    )
    return (
        float(simpler["bic"]) - float(candidate["bic"]),
        improved_fraction,
    )


def _select_wavelength_model_from_pilots(
    spectrum: Spectrum,
    *,
    line_list: LineList,
    config: FitConfig,
    segment_size: float,
    resolution: Mapping[str, object],
) -> tuple[FitConfig, dict[str, object]]:
    pilot_width = AUTO_WAVELENGTH_PILOT_WIDTH_MICRON
    if np.isfinite(segment_size) and segment_size > 0:
        pilot_width = min(pilot_width, float(segment_size))
    candidates = _split_spectrum(
        spectrum,
        segment_size=pilot_width,
        minimum_points=config.continuum_order + 2,
    )
    if config.high_resolution_grid:
        candidates = _subdivide_segments_for_grid_limit(
            candidates,
            config=config,
            minimum_points=config.continuum_order + 2,
        )
    pilot_segments, pilot_records = _select_distributed_lsf_pilot_segments(
        candidates,
        source_spectrum=spectrum,
        line_list=line_list,
        config=config,
    )
    details: dict[str, object] = {
        **dict(resolution),
        "pilot_width_micron": float(pilot_width),
        "pilot_region_count": len(pilot_segments),
        "pilot_regions": pilot_records,
        "minimum_bic_improvement": AUTO_WAVELENGTH_MIN_BIC_IMPROVEMENT,
        "minimum_improved_region_fraction": AUTO_WAVELENGTH_MIN_REGION_FRACTION,
    }
    if not pilot_segments:
        return replace(
            config,
            fit_wavelength_shift=False,
            fit_wavelength_polynomial=False,
            wavelength_shift_unit="pixel",
        ), {
            **details,
            "selected_model": "none",
            "selection_reason": "no_telluric_rich_pilot_regions",
        }

    full_wavelength = spectrum.to_unit("micron").wavelength
    global_bounds = (
        float(np.nanmin(full_wavelength)),
        float(np.nanmax(full_wavelength)),
    )
    base_config = replace(
        config,
        fit_wavelength_shift=False,
        fit_wavelength_polynomial=False,
        fit_segment_wavelength_shifts=False,
        fit_segment_wavelength_polynomial=False,
        wavelength_shift_unit="pixel",
        estimate_uncertainties=False,
        use_jacobian_sparsity=False,
    )
    no_shift_result = fit_telluric_segments(
        pilot_segments,
        line_list=line_list,
        config=base_config,
        global_wavelength_bounds=global_bounds,
    )
    no_shift_metrics = _lsf_pilot_model_metrics(no_shift_result)
    details["none_model"] = no_shift_metrics
    if not no_shift_result.success:
        return base_config, {
            **details,
            "selected_model": "none",
            "selection_reason": "baseline_pilot_fit_did_not_converge",
        }

    initial_pixel_shift = float(config.initial_wavelength_shift)
    if abs(initial_pixel_shift) < 0.02:
        initial_pixel_shift = 0.02 * (
            config.wavelength_shift_bounds[1]
            - config.wavelength_shift_bounds[0]
        )
    initial_pixel_shift = float(
        np.clip(
            initial_pixel_shift,
            config.wavelength_shift_bounds[0],
            config.wavelength_shift_bounds[1],
        )
    )
    constant_config = replace(
        base_config,
        initial_species_scales=dict(no_shift_result.species_scales),
        lsf_sigma_pixels=float(no_shift_result.lsf_sigma_pixels),
        initialize_lsf_sigma_grid=False,
        fit_wavelength_shift=True,
        initial_wavelength_shift=initial_pixel_shift,
    )
    constant_result = fit_telluric_segments(
        pilot_segments,
        line_list=line_list,
        config=constant_config,
        global_wavelength_bounds=global_bounds,
    )
    constant_metrics = _lsf_pilot_model_metrics(constant_result)
    details["constant_pixel_model"] = {
        **constant_metrics,
        "coefficients_pixels": (
            constant_result.segment_results[0]
            .wavelength_coefficients.tolist()
        ),
    }

    constant_bound_status = {
        parameter: status
        for parameter, status in constant_result.parameter_bound_status.items()
        if "wavelength_" in parameter
    }
    constant_coefficient = float(
        constant_result.segment_results[0].wavelength_coefficients[0]
    )
    bound_lower, bound_upper = map(
        float,
        constant_config.wavelength_shift_bounds,
    )
    bound_margin = 0.01 * (bound_upper - bound_lower)
    if not constant_bound_status:
        if constant_coefficient <= bound_lower + bound_margin:
            constant_bound_status["wavelength_shift_pixels"] = "lower"
        elif constant_coefficient >= bound_upper - bound_margin:
            constant_bound_status["wavelength_shift_pixels"] = "upper"
    constant_bic_improvement, constant_improved_fraction = (
        _wavelength_candidate_improvement(
            no_shift_metrics,
            constant_metrics,
        )
    )
    should_expand_bounds = bool(
        constant_result.success
        and constant_bound_status
        and constant_bic_improvement >= AUTO_WAVELENGTH_MIN_BIC_IMPROVEMENT
        and constant_improved_fraction >= AUTO_WAVELENGTH_MIN_REGION_FRACTION
        and tuple(config.wavelength_shift_bounds)
        == AUTO_WAVELENGTH_SHIFT_BOUNDS_PIXELS
    )
    if should_expand_bounds:
        bounded_model = dict(details["constant_pixel_model"])
        expanded_initial = float(
            np.clip(
                constant_result.segment_results[0].wavelength_coefficients[0],
                AUTO_WAVELENGTH_SHIFT_EXPANDED_BOUNDS_PIXELS[0],
                AUTO_WAVELENGTH_SHIFT_EXPANDED_BOUNDS_PIXELS[1],
            )
        )
        expanded_constant_config = replace(
            constant_config,
            initial_wavelength_shift=expanded_initial,
            wavelength_shift_bounds=(
                AUTO_WAVELENGTH_SHIFT_EXPANDED_BOUNDS_PIXELS
            ),
        )
        expanded_constant_result = fit_telluric_segments(
            pilot_segments,
            line_list=line_list,
            config=expanded_constant_config,
            global_wavelength_bounds=global_bounds,
        )
        expanded_constant_metrics = _lsf_pilot_model_metrics(
            expanded_constant_result
        )
        details["wavelength_shift_bound_expansion"] = {
            "triggered": True,
            "initial_bounds_pixels": list(
                AUTO_WAVELENGTH_SHIFT_BOUNDS_PIXELS
            ),
            "expanded_bounds_pixels": list(
                AUTO_WAVELENGTH_SHIFT_EXPANDED_BOUNDS_PIXELS
            ),
            "initial_bound_status": constant_bound_status,
            "initial_bic_improvement": constant_bic_improvement,
            "initial_improved_region_fraction": constant_improved_fraction,
            "bounded_constant_model": bounded_model,
        }
        if expanded_constant_result.success:
            constant_config = expanded_constant_config
            constant_result = expanded_constant_result
            constant_metrics = expanded_constant_metrics
            details["constant_pixel_model"] = {
                **constant_metrics,
                "coefficients_pixels": (
                    constant_result.segment_results[0]
                    .wavelength_coefficients.tolist()
                ),
            }
    else:
        details["wavelength_shift_bound_expansion"] = {
            "triggered": False,
        }

    model_records: list[
        tuple[str, FitConfig, MultiTelluricFitResult, dict[str, object]]
    ] = [
        ("none", base_config, no_shift_result, no_shift_metrics),
    ]
    if constant_result.success:
        model_records.append(
            (
                "constant_pixel_shift",
                constant_config,
                constant_result,
                constant_metrics,
            )
        )

    if len(pilot_segments) >= 2:
        linear_config = replace(
            constant_config,
            fit_wavelength_shift=False,
            fit_wavelength_polynomial=True,
            wavelength_polynomial_order=1,
            initial_wavelength_shift=float(
                constant_result.segment_results[0].wavelength_coefficients[0]
                if constant_result.success
                else config.initial_wavelength_shift
            ),
        )
        linear_result = fit_telluric_segments(
            pilot_segments,
            line_list=line_list,
            config=linear_config,
            global_wavelength_bounds=global_bounds,
        )
        linear_metrics = _lsf_pilot_model_metrics(linear_result)
        details["linear_pixel_trend_model"] = {
            **linear_metrics,
            "coefficients_pixels": (
                linear_result.segment_results[0]
                .wavelength_coefficients.tolist()
            ),
        }
        if linear_result.success:
            model_records.append(
                (
                    "linear_pixel_trend",
                    linear_config,
                    linear_result,
                    linear_metrics,
                )
            )

    selected_name, selected_config, selected_result, selected_metrics = (
        model_records[0]
    )
    comparisons: list[dict[str, object]] = []
    for name, candidate_config, candidate_result, candidate_metrics in model_records[1:]:
        bic_improvement, improved_fraction = _wavelength_candidate_improvement(
            selected_metrics,
            candidate_metrics,
        )
        bound_status = {
            parameter: status
            for parameter, status in candidate_result.parameter_bound_status.items()
            if "wavelength_" in parameter
        }
        accepted = bool(
            bic_improvement >= AUTO_WAVELENGTH_MIN_BIC_IMPROVEMENT
            and improved_fraction >= AUTO_WAVELENGTH_MIN_REGION_FRACTION
            and not bound_status
        )
        comparisons.append(
            {
                "from_model": selected_name,
                "to_model": name,
                "bic_improvement": bic_improvement,
                "improved_region_fraction": improved_fraction,
                "bound_status": bound_status,
                "accepted": accepted,
            }
        )
        if accepted:
            selected_name = name
            selected_config = candidate_config
            selected_result = candidate_result
            selected_metrics = candidate_metrics

    selected_coefficients = (
        selected_result.segment_results[0].wavelength_coefficients.tolist()
        if selected_name != "none"
        else [0.0]
    )
    selected_config = replace(
        config,
        fit_wavelength_shift=selected_name == "constant_pixel_shift",
        fit_wavelength_polynomial=selected_name == "linear_pixel_trend",
        wavelength_polynomial_order=1,
        wavelength_shift_unit="pixel",
        wavelength_shift_bounds=tuple(selected_config.wavelength_shift_bounds),
        initial_wavelength_shift=float(selected_coefficients[0]),
        initialize_lsf_sigma_grid=False,
        lsf_sigma_pixels=float(selected_result.lsf_sigma_pixels),
    )
    return selected_config, {
        **details,
        "bounds": list(selected_config.wavelength_shift_bounds),
        "selected_model": selected_name,
        "selection_reason": (
            "penalized_distributed_pilot_evidence"
            if selected_name != "none"
            else "no_supported_wavelength_correction"
        ),
        "selected_coefficients_pixels": selected_coefficients,
        "comparisons": comparisons,
    }


def _select_lsf_variable_width_from_pilots(
    spectrum: Spectrum,
    *,
    line_list: LineList,
    config: FitConfig,
    segment_size: float,
    resolution: Mapping[str, object],
) -> tuple[FitConfig, dict[str, object]]:
    pilot_width = AUTO_LSF_LORENTZ_PILOT_WIDTH_MICRON
    if np.isfinite(segment_size) and segment_size > 0:
        pilot_width = min(pilot_width, float(segment_size))
    candidates = _split_spectrum(
        spectrum,
        segment_size=pilot_width,
        minimum_points=config.continuum_order + 2,
    )
    if config.high_resolution_grid:
        candidates = _subdivide_segments_for_grid_limit(
            candidates,
            config=config,
            minimum_points=config.continuum_order + 2,
        )
    pilot_segments, pilot_records = _select_distributed_lsf_pilot_segments(
        candidates,
        source_spectrum=spectrum,
        line_list=line_list,
        config=config,
    )
    pilot_centers = np.asarray(
        [
            0.5 * (
                float(record["lower_micron"])
                + float(record["upper_micron"])
            )
            for record in pilot_records
        ],
        dtype=float,
    )
    log_wavelength_span = (
        float(np.ptp(np.log(pilot_centers)))
        if pilot_centers.size >= 2
        else 0.0
    )
    details = {
        **dict(resolution),
        "pilot_width_micron": float(pilot_width),
        "pilot_region_count": len(pilot_segments),
        "pilot_regions": pilot_records,
        "pilot_log_wavelength_span": log_wavelength_span,
        "minimum_log_wavelength_span": (
            AUTO_LSF_VARIABLE_MIN_LOG_WAVELENGTH_SPAN
        ),
        "minimum_bic_improvement": AUTO_LSF_VARIABLE_MIN_BIC_IMPROVEMENT,
        "minimum_improved_region_fraction": (
            AUTO_LSF_VARIABLE_MIN_REGION_FRACTION
        ),
    }
    if len(pilot_segments) < AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS:
        return replace(
            config,
            lsf_variable_width=False,
            fit_lsf_wavelength_exponent=False,
            lsf_wavelength_exponent=0.0,
        ), {
            **details,
            "selected_model": "constant",
            "selection_reason": "fewer_than_two_telluric_rich_pilot_regions",
        }
    if log_wavelength_span < AUTO_LSF_VARIABLE_MIN_LOG_WAVELENGTH_SPAN:
        return replace(
            config,
            lsf_variable_width=False,
            fit_lsf_wavelength_exponent=False,
            lsf_wavelength_exponent=0.0,
        ), {
            **details,
            "selected_model": "constant",
            "selection_reason": "pilot_wavelength_span_too_narrow",
        }

    full_wavelength = spectrum.to_unit("micron").wavelength
    global_bounds = (
        float(np.nanmin(full_wavelength)),
        float(np.nanmax(full_wavelength)),
    )
    constant_config = replace(
        config,
        lsf_variable_width=False,
        fit_lsf_wavelength_exponent=False,
        lsf_wavelength_exponent=0.0,
        estimate_uncertainties=False,
    )
    constant_result = fit_telluric_segments(
        pilot_segments,
        line_list=line_list,
        config=constant_config,
        global_wavelength_bounds=global_bounds,
    )
    constant_metrics = _lsf_pilot_model_metrics(constant_result)
    details["constant_width_model"] = constant_metrics
    if not constant_result.success:
        return constant_config, {
            **details,
            "selected_model": "constant",
            "selection_reason": "constant_width_pilot_fit_did_not_converge",
        }

    variable_config = replace(
        constant_config,
        initial_species_scales=dict(constant_result.species_scales),
        lsf_sigma_pixels=float(constant_result.lsf_sigma_pixels),
        lsf_box_width_pixels=float(constant_result.lsf_box_width_pixels),
        lsf_lorentz_fwhm_pixels=float(
            constant_result.lsf_lorentz_fwhm_pixels
        ),
        initialize_lsf_sigma_grid=False,
        lsf_variable_width=True,
        lsf_wavelength_exponent=1.0,
        fit_lsf_wavelength_exponent=True,
        lsf_wavelength_exponent_bounds=(
            AUTO_LSF_VARIABLE_EXPONENT_BOUNDS
        ),
    )
    variable_result = fit_telluric_segments(
        pilot_segments,
        line_list=line_list,
        config=variable_config,
        global_wavelength_bounds=global_bounds,
    )
    variable_metrics = _lsf_pilot_model_metrics(variable_result)
    details["power_law_width_model"] = variable_metrics
    if not variable_result.success:
        return replace(
            config,
            lsf_sigma_pixels=float(constant_result.lsf_sigma_pixels),
            initialize_lsf_sigma_grid=False,
            lsf_variable_width=False,
            fit_lsf_wavelength_exponent=False,
            lsf_wavelength_exponent=0.0,
        ), {
            **details,
            "selected_model": "constant",
            "selection_reason": "power_law_pilot_fit_did_not_converge",
        }

    constant_region_rss = np.asarray(
        constant_metrics["region_weighted_rss"],
        dtype=float,
    )
    variable_region_rss = np.asarray(
        variable_metrics["region_weighted_rss"],
        dtype=float,
    )
    informative = np.asarray(
        constant_metrics["region_max_absorption"],
        dtype=float,
    ) >= 0.01
    relative_improvement = (
        constant_region_rss - variable_region_rss
    ) / np.maximum(constant_region_rss, np.finfo(float).tiny)
    improved = informative & (
        relative_improvement >= AUTO_LSF_LORENTZ_MIN_REGION_IMPROVEMENT
    )
    informative_count = int(np.count_nonzero(informative))
    improved_count = int(np.count_nonzero(improved))
    improved_fraction = (
        improved_count / informative_count if informative_count else 0.0
    )
    bic_improvement = float(
        constant_metrics["bic"] - variable_metrics["bic"]
    )
    bound_status = variable_result.parameter_bound_status.get(
        "lsf_wavelength_exponent"
    )
    details.update(
        {
            "bic_improvement": bic_improvement,
            "informative_region_count": informative_count,
            "improved_region_count": improved_count,
            "improved_region_fraction": float(improved_fraction),
            "region_relative_rss_improvement": (
                relative_improvement.tolist()
            ),
            "pilot_wavelength_exponent": float(
                variable_result.lsf_wavelength_exponent
            ),
        }
    )
    if bound_status is not None:
        details["pilot_bound_status"] = bound_status

    select_variable = bool(
        informative_count >= AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS
        and bic_improvement >= AUTO_LSF_VARIABLE_MIN_BIC_IMPROVEMENT
        and improved_fraction >= AUTO_LSF_VARIABLE_MIN_REGION_FRACTION
        and bound_status is None
    )
    if select_variable:
        return replace(
            config,
            lsf_sigma_pixels=float(variable_result.lsf_sigma_pixels),
            lsf_box_width_pixels=float(
                variable_result.lsf_box_width_pixels
            ),
            lsf_lorentz_fwhm_pixels=float(
                variable_result.lsf_lorentz_fwhm_pixels
            ),
            initialize_lsf_sigma_grid=False,
            lsf_variable_width=True,
            lsf_wavelength_exponent=float(
                variable_result.lsf_wavelength_exponent
            ),
            fit_lsf_wavelength_exponent=True,
            lsf_wavelength_exponent_bounds=(
                AUTO_LSF_VARIABLE_EXPONENT_BOUNDS
            ),
        ), {
            **details,
            "selected_model": "power_law",
            "selection_reason": "strong_distributed_pilot_evidence",
        }

    reasons = []
    if informative_count < AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS:
        reasons.append("too_few_informative_regions")
    if bic_improvement < AUTO_LSF_VARIABLE_MIN_BIC_IMPROVEMENT:
        reasons.append("insufficient_bic_improvement")
    if improved_fraction < AUTO_LSF_VARIABLE_MIN_REGION_FRACTION:
        reasons.append("improvement_not_consistent_across_regions")
    if bound_status is not None:
        reasons.append("wavelength_exponent_reached_bound")
    return replace(
        config,
        lsf_sigma_pixels=float(constant_result.lsf_sigma_pixels),
        lsf_box_width_pixels=float(constant_result.lsf_box_width_pixels),
        lsf_lorentz_fwhm_pixels=float(
            constant_result.lsf_lorentz_fwhm_pixels
        ),
        initialize_lsf_sigma_grid=False,
        lsf_variable_width=False,
        fit_lsf_wavelength_exponent=False,
        lsf_wavelength_exponent=0.0,
    ), {
        **details,
        "selected_model": "constant",
        "selection_reason": ",".join(reasons),
    }


def _select_lsf_lorentz_from_pilots(
    spectrum: Spectrum,
    *,
    line_list: LineList,
    config: FitConfig,
    segment_size: float,
    resolution: Mapping[str, object],
) -> tuple[FitConfig, dict[str, object]]:
    pilot_width = AUTO_LSF_LORENTZ_PILOT_WIDTH_MICRON
    if np.isfinite(segment_size) and segment_size > 0:
        pilot_width = min(pilot_width, float(segment_size))
    candidates = _split_spectrum(
        spectrum,
        segment_size=pilot_width,
        minimum_points=config.continuum_order + 2,
    )
    if config.high_resolution_grid:
        candidates = _subdivide_segments_for_grid_limit(
            candidates,
            config=config,
            minimum_points=config.continuum_order + 2,
        )
    pilot_segments, pilot_records = _select_distributed_lsf_pilot_segments(
        candidates,
        source_spectrum=spectrum,
        line_list=line_list,
        config=config,
    )
    details = {
        **dict(resolution),
        "pilot_width_micron": float(pilot_width),
        "pilot_region_count": len(pilot_segments),
        "pilot_regions": pilot_records,
        "minimum_bic_improvement": AUTO_LSF_LORENTZ_MIN_BIC_IMPROVEMENT,
        "minimum_improved_region_fraction": AUTO_LSF_LORENTZ_MIN_REGION_FRACTION,
    }
    if len(pilot_segments) < AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS:
        return config, {
            **details,
            "selected_model": "gaussian",
            "selection_reason": "fewer_than_two_telluric_rich_pilot_regions",
        }

    full_wavelength = spectrum.to_unit("micron").wavelength
    global_bounds = (
        float(np.nanmin(full_wavelength)),
        float(np.nanmax(full_wavelength)),
    )
    gaussian_config = replace(
        config,
        lsf_lorentz_fwhm_pixels=0.0,
        fit_lsf_lorentz_fwhm=False,
        estimate_uncertainties=False,
    )
    gaussian_result = fit_telluric_segments(
        pilot_segments,
        line_list=line_list,
        config=gaussian_config,
        global_wavelength_bounds=global_bounds,
    )
    gaussian_metrics = _lsf_pilot_model_metrics(gaussian_result)
    details["gaussian_model"] = gaussian_metrics
    if not gaussian_result.success:
        return config, {
            **details,
            "selected_model": "gaussian",
            "selection_reason": "gaussian_pilot_fit_did_not_converge",
        }

    lower, upper = config.lsf_lorentz_fwhm_bounds
    gaussian_fwhm = 2.354820045 * max(
        float(gaussian_result.lsf_sigma_pixels),
        0.2,
    )
    interval = upper - lower
    lorentz_initial = float(
        np.clip(
            0.5 * gaussian_fwhm,
            lower + 0.01 * interval,
            upper - 0.01 * interval,
        )
    )
    lorentz_config = replace(
        gaussian_config,
        initial_species_scales=dict(gaussian_result.species_scales),
        lsf_sigma_pixels=float(gaussian_result.lsf_sigma_pixels),
        initialize_lsf_sigma_grid=False,
        lsf_lorentz_fwhm_pixels=lorentz_initial,
        fit_lsf_lorentz_fwhm=True,
    )
    lorentz_result = fit_telluric_segments(
        pilot_segments,
        line_list=line_list,
        config=lorentz_config,
        global_wavelength_bounds=global_bounds,
    )
    lorentz_metrics = _lsf_pilot_model_metrics(lorentz_result)
    details["gaussian_lorentz_model"] = lorentz_metrics
    if not lorentz_result.success:
        return config, {
            **details,
            "selected_model": "gaussian",
            "selection_reason": "gaussian_lorentz_pilot_fit_did_not_converge",
        }

    gaussian_region_rss = np.asarray(
        gaussian_metrics["region_weighted_rss"],
        dtype=float,
    )
    lorentz_region_rss = np.asarray(
        lorentz_metrics["region_weighted_rss"],
        dtype=float,
    )
    informative = np.asarray(
        gaussian_metrics["region_max_absorption"],
        dtype=float,
    ) >= 0.01
    relative_improvement = (
        gaussian_region_rss - lorentz_region_rss
    ) / np.maximum(gaussian_region_rss, np.finfo(float).tiny)
    improved = informative & (
        relative_improvement >= AUTO_LSF_LORENTZ_MIN_REGION_IMPROVEMENT
    )
    informative_count = int(np.count_nonzero(informative))
    improved_count = int(np.count_nonzero(improved))
    improved_fraction = (
        improved_count / informative_count if informative_count else 0.0
    )
    bic_improvement = float(
        gaussian_metrics["bic"] - lorentz_metrics["bic"]
    )
    bound_status = lorentz_result.parameter_bound_status.get(
        "lsf_lorentz_fwhm_pixels"
    )
    details.update(
        {
            "bic_improvement": bic_improvement,
            "informative_region_count": informative_count,
            "improved_region_count": improved_count,
            "improved_region_fraction": float(improved_fraction),
            "region_relative_rss_improvement": relative_improvement.tolist(),
            "pilot_lorentz_fwhm_pixels": float(
                lorentz_result.lsf_lorentz_fwhm_pixels
            ),
        }
    )
    if bound_status is not None:
        details["pilot_bound_status"] = bound_status

    select_lorentz = bool(
        informative_count >= AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS
        and bic_improvement >= AUTO_LSF_LORENTZ_MIN_BIC_IMPROVEMENT
        and improved_fraction >= AUTO_LSF_LORENTZ_MIN_REGION_FRACTION
        and bound_status is None
    )
    if select_lorentz:
        selected_config = replace(
            config,
            lsf_sigma_pixels=float(lorentz_result.lsf_sigma_pixels),
            initialize_lsf_sigma_grid=False,
            lsf_lorentz_fwhm_pixels=float(
                lorentz_result.lsf_lorentz_fwhm_pixels
            ),
            fit_lsf_lorentz_fwhm=True,
        )
        return selected_config, {
            **details,
            "selected_model": "gaussian_lorentz",
            "selection_reason": "strong_distributed_pilot_evidence",
        }

    reasons = []
    if informative_count < AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS:
        reasons.append("too_few_informative_regions")
    if bic_improvement < AUTO_LSF_LORENTZ_MIN_BIC_IMPROVEMENT:
        reasons.append("insufficient_bic_improvement")
    if improved_fraction < AUTO_LSF_LORENTZ_MIN_REGION_FRACTION:
        reasons.append("improvement_not_consistent_across_regions")
    if bound_status is not None:
        reasons.append("lorentz_width_reached_bound")
    return config, {
        **details,
        "selected_model": "gaussian",
        "selection_reason": ",".join(reasons),
    }


def _select_distributed_lsf_pilot_segments(
    candidates: tuple[Spectrum, ...],
    *,
    source_spectrum: Spectrum,
    line_list: LineList,
    config: FitConfig,
) -> tuple[tuple[Spectrum, ...], list[dict[str, object]]]:
    finite_strengths = line_list.strength[
        np.isfinite(line_list.strength) & (line_list.strength > 0)
    ]
    if finite_strengths.size == 0:
        return (), []
    strength_scale = float(np.nanmax(finite_strengths))
    scored = []
    for segment in candidates:
        if not _segment_has_fit_pixels(segment, config):
            continue
        if _segment_fit_pixel_count(segment, config) < max(
            AUTO_LSF_LORENTZ_MIN_PILOT_PIXELS,
            config.continuum_order + 2,
        ):
            continue
        wavelength = segment.to_unit("micron").wavelength
        lower = float(np.nanmin(wavelength))
        upper = float(np.nanmax(wavelength))
        positive_steps = np.diff(np.sort(wavelength))
        positive_steps = positive_steps[
            np.isfinite(positive_steps) & (positive_steps > 0)
        ]
        half_pixel = (
            0.5 * float(np.nanmedian(positive_steps))
            if positive_steps.size
            else 0.0
        )
        line_mask = (
            np.isfinite(line_list.wavelength)
            & (line_list.wavelength >= lower - half_pixel)
            & (line_list.wavelength <= upper + half_pixel)
            & np.isfinite(line_list.strength)
            & (line_list.strength > 0)
        )
        if config.fit_ranges is not None:
            line_mask &= np.any(
                [
                    (line_list.wavelength >= fit_lower)
                    & (line_list.wavelength <= fit_upper)
                    for fit_lower, fit_upper in config.fit_ranges
                ],
                axis=0,
            )
        if config.exclude_ranges is not None:
            for exclude_lower, exclude_upper in config.exclude_ranges:
                line_mask &= ~(
                    (line_list.wavelength >= exclude_lower)
                    & (line_list.wavelength <= exclude_upper)
                )
        line_count = int(np.count_nonzero(line_mask))
        if line_count == 0:
            continue
        score = float(
            np.sum(np.sqrt(line_list.strength[line_mask] / strength_scale))
        )
        line_weights = np.sqrt(
            line_list.strength[line_mask] / strength_scale
        )
        line_center = float(
            np.average(line_list.wavelength[line_mask], weights=line_weights)
        )
        scored.append(
            {
                "segment": segment,
                "center": 0.5 * (lower + upper),
                "line_center": line_center,
                "lower": lower,
                "upper": upper,
                "line_count": line_count,
                "line_score": score,
            }
        )
    if not scored:
        return (), []

    scored.sort(key=lambda item: float(item["center"]))
    centers = np.asarray([item["center"] for item in scored], dtype=float)
    if np.allclose(centers[0], centers[-1]):
        selected_indices = [int(np.argmax([item["line_score"] for item in scored]))]
    else:
        edges = np.linspace(
            centers[0],
            centers[-1],
            AUTO_LSF_LORENTZ_MAX_PILOT_REGIONS + 1,
        )
        selected_indices = []
        for bin_index in range(AUTO_LSF_LORENTZ_MAX_PILOT_REGIONS):
            in_bin = np.flatnonzero(
                (centers >= edges[bin_index])
                & (
                    (centers <= edges[bin_index + 1])
                    if bin_index == AUTO_LSF_LORENTZ_MAX_PILOT_REGIONS - 1
                    else (centers < edges[bin_index + 1])
                )
            )
            if in_bin.size:
                local_scores = [
                    float(scored[index]["line_score"])
                    for index in in_bin
                ]
                selected_indices.append(
                    int(in_bin[int(np.argmax(local_scores))])
                )

    if len(selected_indices) < AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS:
        ranked = sorted(
            range(len(scored)),
            key=lambda index: float(scored[index]["line_score"]),
            reverse=True,
        )
        for index in ranked:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= AUTO_LSF_LORENTZ_MIN_PILOT_REGIONS:
                break
    selected_indices = sorted(set(selected_indices))
    selected = [scored[index] for index in selected_indices]
    segments = tuple(
        _centered_spectrum_window(
            source_spectrum,
            center=float(item["line_center"]),
            width=float(item["upper"]) - float(item["lower"]),
        )
        for item in selected
    )
    records = [
        {
            "lower_micron": float(np.nanmin(segment.wavelength)),
            "upper_micron": float(np.nanmax(segment.wavelength)),
            "line_count": int(item["line_count"]),
            "line_score": float(item["line_score"]),
        }
        for item, segment in zip(selected, segments, strict=True)
    ]
    return segments, records


def _centered_spectrum_window(
    spectrum: Spectrum,
    *,
    center: float,
    width: float,
) -> Spectrum:
    ordered = spectrum.to_unit("micron").sorted()
    wavelength = ordered.wavelength
    if wavelength.size < 2:
        return ordered
    steps = np.diff(wavelength)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    representative_step = (
        float(np.nanmedian(positive)) if positive.size else np.inf
    )
    gaps = np.flatnonzero(steps > 10.0 * representative_step) + 1
    nearest = int(np.argmin(np.abs(wavelength - center)))
    block_start_candidates = gaps[gaps <= nearest]
    block_stop_candidates = gaps[gaps > nearest]
    block_start = (
        int(block_start_candidates[-1])
        if block_start_candidates.size
        else 0
    )
    block_stop = (
        int(block_stop_candidates[0])
        if block_stop_candidates.size
        else wavelength.size
    )
    lower = center - 0.5 * width
    upper = center + 0.5 * width
    start = block_start + int(
        np.searchsorted(
            wavelength[block_start:block_stop],
            lower,
            side="left",
        )
    )
    stop = block_start + int(
        np.searchsorted(
            wavelength[block_start:block_stop],
            upper,
            side="right",
        )
    )
    start = min(max(start, block_start), block_stop - 1)
    stop = min(max(stop, start + 1), block_stop)
    return _slice_spectrum(ordered, start, stop)


def _lsf_pilot_model_metrics(
    result: MultiTelluricFitResult,
) -> dict[str, object]:
    region_rss = []
    region_points = []
    region_depth = []
    for segment_result in result.segment_results:
        fit_mask = (
            segment_result.spectrum.valid
            if segment_result.fit_mask is None
            else np.asarray(segment_result.fit_mask, dtype=bool)
        )
        flux = segment_result.spectrum.flux[fit_mask]
        model = segment_result.model_flux[fit_mask]
        if segment_result.spectrum.uncertainty is None:
            mean_flux = float(np.nanmean(np.abs(flux)))
            sigma = np.full_like(
                flux,
                0.01 * mean_flux if mean_flux > 0 else 1.0,
            )
        else:
            sigma = segment_result.spectrum.uncertainty[fit_mask]
        residual = (flux - model) / sigma
        finite = np.isfinite(residual)
        rss = float(np.sum(np.square(residual[finite])))
        region_rss.append(rss)
        region_points.append(int(np.count_nonzero(finite)))
        absorption = 1.0 - segment_result.transmission[fit_mask]
        finite_absorption = absorption[np.isfinite(absorption)]
        region_depth.append(
            float(np.nanmax(finite_absorption))
            if finite_absorption.size
            else 0.0
        )

    weighted_rss = float(np.sum(region_rss))
    point_count = int(np.sum(region_points))
    parameter_count = len(result.parameter_names)
    mean_square = weighted_rss / max(point_count, 1)
    bic = (
        point_count * np.log(max(mean_square, np.finfo(float).tiny))
        + parameter_count * np.log(max(point_count, 2))
    )
    metrics = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "weighted_rss": weighted_rss,
        "fit_pixel_count": point_count,
        "parameter_count": parameter_count,
        "bic": float(bic),
        "region_weighted_rss": region_rss,
        "region_fit_pixel_count": region_points,
        "region_max_absorption": region_depth,
        "lsf_sigma_pixels": float(result.lsf_sigma_pixels),
        "lsf_lorentz_fwhm_pixels": float(result.lsf_lorentz_fwhm_pixels),
        "lsf_wavelength_exponent": float(result.lsf_wavelength_exponent),
    }
    coarse_search = result.provenance.get("lsf_sigma_coarse_search")
    if coarse_search is not None:
        metrics["coarse_search"] = coarse_search
    return metrics


_RESOLVING_POWER_KEYS = (
    "SPEC_RES",
    "SPEC_RP",
    "RESPOWER",
    "RES_POWER",
    "RPOWER",
    "RESOLVING",
    "ESO INS SPEC RES",
    "ESO INS WLEN RESOL",
    "ESO INS GRAT1 RESOL",
    "ESO INS GRAT2 RESOL",
)


def _estimate_lsf_sigma_from_resolving_power(
    spectrum: Spectrum,
    header: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if header is None:
        return None
    normalized_keys = {
        str(key).strip().upper().removeprefix("HIERARCH "): key
        for key in header
    }
    resolving_power = None
    source_key = None
    for candidate in _RESOLVING_POWER_KEYS:
        actual_key = normalized_keys.get(candidate)
        if actual_key is None:
            continue
        try:
            value = float(header[actual_key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and 100.0 <= value <= 10_000_000.0:
            resolving_power = value
            source_key = str(actual_key)
            break
    if resolving_power is None:
        return None

    wavelength = spectrum.to_unit("micron").wavelength
    wavelength = np.sort(wavelength[np.isfinite(wavelength)])
    if wavelength.size < 2:
        return None
    spacing = np.diff(wavelength)
    positive = spacing[np.isfinite(spacing) & (spacing > 0)]
    if positive.size == 0:
        return None
    representative_spacing = float(np.nanmedian(positive))
    midpoint = 0.5 * (wavelength[:-1] + wavelength[1:])
    usable = (
        np.isfinite(spacing)
        & (spacing > 0)
        & (spacing <= 10.0 * representative_spacing)
    )
    fwhm_pixels = midpoint[usable] / (resolving_power * spacing[usable])
    fwhm_pixels = fwhm_pixels[
        np.isfinite(fwhm_pixels)
        & (fwhm_pixels >= 0.2)
        & (fwhm_pixels <= 100.0)
    ]
    if fwhm_pixels.size == 0:
        return None
    sigma_pixels = float(np.nanmedian(fwhm_pixels) / 2.354820045)
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        return None
    return {
        "source": "fits_resolving_power",
        "header_keyword": source_key,
        "resolving_power": float(resolving_power),
        "initial_sigma_pixels": sigma_pixels,
        "sampling_count": int(fwhm_pixels.size),
    }


def _estimate_lsf_sigma_from_spectral_features(
    spectrum: Spectrum,
    *,
    fit_ranges: tuple[tuple[float, float], ...] | None,
    exclude_ranges: tuple[tuple[float, float], ...] | None,
) -> dict[str, object] | None:
    ordered = spectrum.to_unit("micron").sorted()
    wavelength = ordered.wavelength
    flux = ordered.flux
    usable = ordered.valid.copy()
    if fit_ranges:
        usable &= np.any(
            [
                (wavelength >= lower) & (wavelength <= upper)
                for lower, upper in fit_ranges
            ],
            axis=0,
        )
    if exclude_ranges:
        for lower, upper in exclude_ranges:
            usable &= ~((wavelength >= lower) & (wavelength <= upper))

    positive_steps = np.diff(wavelength)
    finite_steps = positive_steps[
        np.isfinite(positive_steps) & (positive_steps > 0)
    ]
    if finite_steps.size == 0:
        return None
    representative_step = float(np.nanmedian(finite_steps))
    boundaries = np.concatenate(
        (
            [0],
            np.flatnonzero(positive_steps > 10.0 * representative_step) + 1,
            [wavelength.size],
        )
    )
    widths = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        local_usable = usable[start:stop]
        if np.count_nonzero(local_usable) < 15:
            continue
        local_flux = np.asarray(flux[start:stop], dtype=float)
        finite_flux = local_flux[local_usable]
        scale = float(np.nanmedian(np.abs(finite_flux)))
        if not np.isfinite(scale) or scale <= 0:
            continue
        filled = local_flux.copy()
        filled[~local_usable] = float(np.nanmedian(finite_flux))
        window = min(101, max(15, 2 * (filled.size // 20) + 1))
        if window % 2 == 0:
            window += 1
        continuum = percentile_filter(
            filled,
            percentile=80.0,
            size=window,
            mode="nearest",
        )
        valid_continuum = np.isfinite(continuum) & (np.abs(continuum) > 1.0e-12 * scale)
        normalized = np.ones_like(filled)
        normalized[valid_continuum] = (
            filled[valid_continuum] / continuum[valid_continuum]
        )
        absorption = 1.0 - normalized
        differences = np.diff(normalized[local_usable])
        noise = (
            1.4826
            * float(np.nanmedian(np.abs(differences - np.nanmedian(differences))))
            / np.sqrt(2.0)
            if differences.size
            else 0.0
        )
        prominence = max(0.01, 4.0 * noise)
        peaks, properties = find_peaks(
            absorption,
            height=max(0.015, 3.0 * noise),
            prominence=prominence,
            distance=2,
        )
        if peaks.size == 0:
            continue
        local_widths = peak_widths(
            absorption,
            peaks,
            rel_height=0.5,
        )[0]
        valid_widths = (
            np.isfinite(local_widths)
            & (local_widths >= 0.8)
            & (local_widths <= 30.0)
            & (properties["prominences"] >= prominence)
        )
        widths.extend(local_widths[valid_widths].tolist())
    if not widths:
        return None
    fwhm_pixels = float(np.nanpercentile(np.asarray(widths), 40.0))
    sigma_pixels = fwhm_pixels / 2.354820045
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        return None
    return {
        "source": "spectrum_features",
        "initial_sigma_pixels": float(
            np.clip(sigma_pixels, 0.2, AUTO_LSF_SIGMA_FEATURE_MAX_PIXELS)
        ),
        "feature_count": len(widths),
        "feature_fwhm_pixels": fwhm_pixels,
    }


def _with_lsf_sigma_provenance(
    result: TelluricFitResult,
    *,
    resolution: Mapping[str, object],
    bounds: tuple[float, float],
    fit_enabled: bool,
    bound_status: str | None,
) -> TelluricFitResult:
    details = {
        **dict(resolution),
        "bounds_pixels": [float(bounds[0]), float(bounds[1])],
        "fit_enabled": bool(fit_enabled),
        "final_sigma_pixels": float(result.lsf_sigma_pixels),
    }
    if bound_status is not None:
        details["bound_status"] = bound_status
    coarse_search = result.provenance.get("lsf_sigma_coarse_search")
    if coarse_search is not None:
        details["coarse_search"] = coarse_search
    return replace(
        result,
        provenance={
            **dict(result.provenance),
            "lsf_sigma": details,
        },
    )


def _with_lsf_lorentz_provenance(
    result: TelluricFitResult,
    *,
    resolution: Mapping[str, object],
    bounds: tuple[float, float],
    fit_enabled: bool,
    bound_status: str | None,
) -> TelluricFitResult:
    details = {
        **dict(resolution),
        "bounds_pixels": [float(bounds[0]), float(bounds[1])],
        "fit_enabled_in_full_fit": bool(fit_enabled),
        "final_fwhm_pixels": float(result.lsf_lorentz_fwhm_pixels),
    }
    if bound_status is not None:
        details["full_fit_bound_status"] = bound_status
    return replace(
        result,
        provenance={
            **dict(result.provenance),
            "lsf_lorentz": details,
        },
    )


def _with_lsf_variable_width_provenance(
    result: TelluricFitResult,
    *,
    resolution: Mapping[str, object],
    config: FitConfig,
    bound_status: str | None,
) -> TelluricFitResult:
    details = {
        **dict(resolution),
        "selected_model": resolution.get(
            "selected_model",
            "power_law" if config.lsf_variable_width else "constant",
        ),
        "fit_enabled_in_full_fit": bool(
            config.fit_lsf_wavelength_exponent
        ),
        "reference_wavelength_micron": float(
            config.lsf_reference_wavelength_micron
        ),
        "final_wavelength_exponent": float(
            result.lsf_wavelength_exponent
        ),
    }
    if bound_status is not None:
        details["full_fit_bound_status"] = bound_status
    return replace(
        result,
        provenance={
            **dict(result.provenance),
            "lsf_variable_width": details,
        },
    )


def _with_wavelength_model_provenance(
    result: TelluricFitResult,
    *,
    resolution: Mapping[str, object],
    config: FitConfig,
    bound_status: Mapping[str, str],
) -> TelluricFitResult:
    details = {
        **dict(resolution),
        "selected_model": _configured_wavelength_model_name(config),
        "coefficient_unit": config.wavelength_shift_unit,
        "final_coefficients": np.asarray(
            result.wavelength_coefficients,
            dtype=float,
        ).tolist(),
        "final_median_shift_micron": float(result.wavelength_shift),
        "full_fit_bound_status": dict(bound_status),
    }
    return replace(
        result,
        provenance={
            **dict(result.provenance),
            "wavelength_alignment": details,
        },
    )


def _split_spectrum(
    spectrum: Spectrum,
    *,
    segment_size: float,
    minimum_points: int = 3,
) -> tuple[Spectrum, ...]:
    """Split into numerical chunks while retaining physical-group identity."""

    if not np.isfinite(segment_size) or segment_size <= 0:
        raise ValueError("segment_size must be a positive finite value in microns")
    if minimum_points < 2:
        raise ValueError("minimum_points must be at least two")

    unit_spectrum = spectrum.to_unit("micron")
    if not np.all(np.isfinite(unit_spectrum.wavelength)):
        raise ValueError(
            "automatic segmentation requires finite wavelengths; remove or mask "
            "rows with invalid wavelength coordinates"
        )
    physical_groups: list[Spectrum] = []
    source = "segment_size"
    if unit_spectrum.group_id is not None:
        source = "fits_order_detector"
        for group in np.unique(unit_spectrum.group_id):
            indices = np.flatnonzero(unit_spectrum.group_id == group)
            if indices.size >= minimum_points:
                physical_groups.append(_take_spectrum(unit_spectrum, indices).sorted())
    else:
        ordered = unit_spectrum.sorted()
        if ordered.wavelength.size < minimum_points:
            return (ordered,)
        steps = np.diff(ordered.wavelength)
        positive = steps[np.isfinite(steps) & (steps > 0)]
        representative = float(np.nanmedian(positive)) if positive.size else np.inf
        stops = np.flatnonzero(steps > 10.0 * representative) + 1
        source = "wavelength_gaps" if stops.size else "segment_size"
        gap_ranges = [
            [int(start), int(stop)]
            for start, stop in zip(
                np.concatenate(([0], stops)),
                np.concatenate((stops, [ordered.wavelength.size])),
                strict=True,
            )
        ]
        # A gap can isolate one or two valid samples. They cannot support an
        # independent continuum, but silently discarding them changes the
        # output grid. Attach each short island to its nearest fitted group;
        # numerical chunking below still keeps the discontinuity harmless.
        gap_ranges = _merge_short_index_ranges(
            gap_ranges,
            minimum_points=minimum_points,
        )
        physical_groups.extend(
            _slice_spectrum(ordered, start, stop)
            for start, stop in gap_ranges
        )

    physical_groups.sort(key=lambda item: float(np.nanmin(item.wavelength)))
    chunks: list[Spectrum] = []
    for physical_index, group in enumerate(physical_groups):
        ranges = _width_chunk_ranges(
            group.wavelength,
            segment_size=segment_size,
            minimum_points=minimum_points,
        )
        physical_bounds = [
            float(np.nanmin(group.wavelength)),
            float(np.nanmax(group.wavelength)),
        ]
        for start, stop in ranges:
            chunk = _slice_spectrum(group, start, stop)
            chunks.append(
                replace(
                    chunk,
                    meta={
                        **dict(chunk.meta),
                        "physical_group_index": physical_index,
                        "physical_group_count": len(physical_groups),
                        "physical_group_bounds_micron": physical_bounds,
                        "segmentation_source": source,
                    },
                )
            )
    return tuple(
        replace(
            chunk,
            meta={
                **dict(chunk.meta),
                "segment_index": index,
                "segment_count": len(chunks),
                "segment_size_micron": float(segment_size),
            },
        )
        for index, chunk in enumerate(chunks)
    )


def _weights_for_segment(
    parent: Spectrum,
    parent_weights: np.ndarray,
    segment: Spectrum,
) -> np.ndarray:
    """Map parent-grid fit weights onto a sorted numerical segment."""

    weights = np.asarray(parent_weights, dtype=float)
    if weights.shape != parent.wavelength.shape:
        raise ValueError("parent fit weights must match the parent spectrum")
    parent_unit = parent.to_unit("micron")
    segment_unit = segment.to_unit("micron")
    if parent_unit.group_id is None or segment_unit.group_id is None:
        candidate_indices = np.arange(parent_unit.wavelength.size)
    else:
        segment_groups = np.unique(segment_unit.group_id)
        candidate_indices = np.flatnonzero(
            np.isin(parent_unit.group_id, segment_groups)
        )
        if candidate_indices.size == 0:
            raise RuntimeError("segment group was not found on the parent spectrum")
    order = candidate_indices[
        np.argsort(parent_unit.wavelength[candidate_indices], kind="stable")
    ]
    parent_wavelength = parent_unit.wavelength[order]
    sorted_weights = weights[order]
    indices = np.searchsorted(parent_wavelength, segment_unit.wavelength)
    indices = np.clip(indices, 0, parent_wavelength.size - 1)
    left = np.maximum(indices - 1, 0)
    choose_left = (
        np.abs(parent_wavelength[left] - segment_unit.wavelength)
        < np.abs(parent_wavelength[indices] - segment_unit.wavelength)
    )
    indices = np.where(choose_left, left, indices)
    tolerance = 32.0 * np.finfo(float).eps * np.maximum(
        1.0,
        np.abs(segment_unit.wavelength),
    )
    if np.any(
        np.abs(parent_wavelength[indices] - segment_unit.wavelength) > tolerance
    ):
        raise RuntimeError("could not map stellar fit weights onto a segment")
    return sorted_weights[indices]


def _width_chunk_ranges(
    wavelength: np.ndarray,
    *,
    segment_size: float,
    minimum_points: int,
) -> tuple[tuple[int, int], ...]:
    if wavelength.size < minimum_points:
        return ((0, int(wavelength.size)),)
    span = float(wavelength[-1] - wavelength[0])
    if span <= segment_size:
        return ((0, int(wavelength.size)),)
    ratio = span / segment_size
    ratio -= 1.0e-12 * max(1.0, abs(ratio))
    count = max(1, int(np.ceil(ratio)))
    edges = np.linspace(wavelength[0], wavelength[-1], count + 1)
    boundaries = [0]
    boundaries.extend(
        np.searchsorted(wavelength, edges[1:-1], side="left").tolist()
    )
    boundaries.append(int(wavelength.size))
    ranges = [
        [int(start), int(stop)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if stop > start
    ]
    ranges = _merge_short_index_ranges(
        ranges,
        minimum_points=minimum_points,
    )
    return tuple((start, stop) for start, stop in ranges)


def _merge_short_index_ranges(
    ranges: list[list[int]],
    *,
    minimum_points: int,
) -> list[list[int]]:
    """Merge undersized adjacent index ranges without losing samples."""

    index = 0
    while index < len(ranges):
        start, stop = ranges[index]
        if stop - start >= minimum_points or len(ranges) == 1:
            index += 1
        elif index > 0:
            ranges[index - 1][1] = stop
            ranges.pop(index)
        else:
            ranges[1][0] = start
            ranges.pop(0)
    return ranges


def _subdivide_segments_for_grid_limit(
    segments: tuple[Spectrum, ...],
    *,
    config: FitConfig,
    minimum_points: int,
) -> tuple[Spectrum, ...]:
    pending = list(segments)
    accepted: list[Spectrum] = []
    while pending:
        segment = pending.pop(0)
        required_points = _radiative_transfer_point_count(segment.wavelength, config)
        if required_points <= config.radiative_transfer_max_points:
            accepted.append(segment)
            continue
        if segment.wavelength.size < 2 * minimum_points:
            raise ValueError(
                "automatic segmentation cannot satisfy radiative_transfer_max_points "
                f"without producing fewer than {minimum_points} pixels per segment; "
                "raise radiative_transfer_max_points or reduce the continuum order"
            )
        midpoint = 0.5 * (segment.wavelength[0] + segment.wavelength[-1])
        split = int(np.searchsorted(segment.wavelength, midpoint, side="left"))
        split = min(
            max(split, minimum_points),
            segment.wavelength.size - minimum_points,
        )
        left = _slice_spectrum(segment, 0, split)
        right = _slice_spectrum(segment, split, segment.wavelength.size)
        pending[0:0] = [left, right]

    segment_count = len(accepted)
    return tuple(
        Spectrum(
            wavelength=segment.wavelength,
            flux=segment.flux,
            uncertainty=segment.uncertainty,
            mask=segment.mask,
            group_id=segment.group_id,
            wavelength_unit=segment.wavelength_unit,
            wavelength_medium=segment.wavelength_medium,
            meta={
                **dict(segment.meta),
                "segment_index": index,
                "segment_count": segment_count,
            },
        )
        for index, segment in enumerate(accepted)
    )


def _segment_has_fit_pixels(segment: Spectrum, config: FitConfig) -> bool:
    return bool(
        _segment_fit_pixel_count(segment, config)
        >= config.continuum_order + 2
    )


def _segment_fit_pixel_count(segment: Spectrum, config: FitConfig) -> int:
    wavelength = segment.to_unit("micron").wavelength
    selected = segment.valid.copy()
    if config.fit_ranges is not None:
        include = np.zeros(wavelength.shape, dtype=bool)
        for lower, upper in config.fit_ranges:
            include |= (wavelength >= lower) & (wavelength <= upper)
        selected &= include
    if config.exclude_ranges is not None:
        for lower, upper in config.exclude_ranges:
            selected &= ~((wavelength >= lower) & (wavelength <= upper))
    return int(np.count_nonzero(selected))


def _slice_spectrum(spectrum: Spectrum, start: int, stop: int) -> Spectrum:
    return Spectrum(
        wavelength=spectrum.wavelength[start:stop].copy(),
        flux=spectrum.flux[start:stop].copy(),
        uncertainty=(
            None
            if spectrum.uncertainty is None
            else spectrum.uncertainty[start:stop].copy()
        ),
        mask=None if spectrum.mask is None else spectrum.mask[start:stop].copy(),
        group_id=(
            None
            if spectrum.group_id is None
            else spectrum.group_id[start:stop].copy()
        ),
        wavelength_unit=spectrum.wavelength_unit,
        wavelength_medium=spectrum.wavelength_medium,
        meta=dict(spectrum.meta),
    )


def _take_spectrum(spectrum: Spectrum, indices: np.ndarray) -> Spectrum:
    indices = np.asarray(indices, dtype=int)
    return Spectrum(
        wavelength=spectrum.wavelength[indices].copy(),
        flux=spectrum.flux[indices].copy(),
        uncertainty=(
            None
            if spectrum.uncertainty is None
            else spectrum.uncertainty[indices].copy()
        ),
        mask=None if spectrum.mask is None else spectrum.mask[indices].copy(),
        group_id=(
            None
            if spectrum.group_id is None
            else spectrum.group_id[indices].copy()
        ),
        wavelength_unit=spectrum.wavelength_unit,
        wavelength_medium=spectrum.wavelength_medium,
        meta=dict(spectrum.meta),
    )


def _concatenate_spectra(
    spectra: tuple[Spectrum, ...],
    *,
    corrected: bool,
    segment_size: float,
) -> Spectrum:
    first = spectra[0]
    uncertainty = None
    if all(spectrum.uncertainty is not None for spectrum in spectra):
        uncertainty = np.concatenate(
            [np.asarray(spectrum.uncertainty, dtype=float) for spectrum in spectra]
        )
    mask = None
    if any(spectrum.mask is not None for spectrum in spectra):
        mask = np.concatenate(
            [
                np.ones(spectrum.wavelength.size, dtype=bool)
                if spectrum.mask is None
                else np.asarray(spectrum.mask, dtype=bool)
                for spectrum in spectra
            ]
        )
    group_id = None
    if any(spectrum.group_id is not None for spectrum in spectra):
        group_id = np.concatenate(
            [
                np.full(spectrum.wavelength.size, index, dtype=int)
                if spectrum.group_id is None
                else np.asarray(spectrum.group_id)
                for index, spectrum in enumerate(spectra)
            ]
        )
    return Spectrum(
        wavelength=np.concatenate([spectrum.wavelength for spectrum in spectra]),
        flux=np.concatenate([spectrum.flux for spectrum in spectra]),
        uncertainty=uncertainty,
        mask=mask,
        group_id=group_id,
        wavelength_unit=first.wavelength_unit,
        wavelength_medium=first.wavelength_medium,
        meta={
            **dict(first.meta),
            "telluric_corrected": corrected,
            "automatic_segmentation": True,
            "segment_count": len(spectra),
            "segment_size_micron": float(segment_size),
        },
    )


def _stitch_segment_results(
    result: MultiTelluricFitResult,
    *,
    segment_size: float,
    segment_results: tuple[TelluricFitResult, ...] | None = None,
) -> TelluricFitResult:
    """Return the normal single-result interface for an automatic segmented fit."""

    source_results = result.segment_results if segment_results is None else segment_results
    segment_results = tuple(
        sorted(
            source_results,
            key=lambda item: (
                int(item.spectrum.meta.get("segment_index", 0)),
                float(np.nanmin(item.spectrum.wavelength)),
            ),
        )
    )
    spectra = tuple(item.spectrum for item in segment_results)
    corrected_spectra = tuple(item.corrected for item in segment_results)
    spectrum = _concatenate_spectra(
        spectra,
        corrected=False,
        segment_size=segment_size,
    )
    corrected = _concatenate_spectra(
        corrected_spectra,
        corrected=True,
        segment_size=segment_size,
    )
    transmission = np.concatenate([item.transmission for item in segment_results])
    continuum = np.concatenate([item.continuum for item in segment_results])
    model_flux = np.concatenate([item.model_flux for item in segment_results])
    stellar_model = None
    if any(item.stellar_model is not None for item in segment_results):
        stellar_model = np.concatenate(
            [
                np.ones(item.spectrum.wavelength.size, dtype=float)
                if item.stellar_model is None
                else np.asarray(item.stellar_model, dtype=float)
                for item in segment_results
            ]
        )
    fit_mask = np.concatenate(
        [
            np.zeros(item.spectrum.wavelength.size, dtype=bool)
            if item.fit_mask is None
            else np.asarray(item.fit_mask, dtype=bool)
            for item in segment_results
        ]
    )
    fit_weights = None
    if any(item.fit_weights is not None for item in segment_results):
        fit_weights = np.concatenate(
            [
                np.ones(item.spectrum.wavelength.size, dtype=float)
                if item.fit_weights is None
                else np.asarray(item.fit_weights, dtype=float)
                for item in segment_results
            ]
        )
    transmission_uncertainty = None
    if all(item.transmission_uncertainty is not None for item in segment_results):
        transmission_uncertainty = np.concatenate(
            [
                np.asarray(item.transmission_uncertainty, dtype=float)
                for item in segment_results
            ]
        )
    continuum_coefficients = np.concatenate(
        [np.asarray(item.continuum_coefficients, dtype=float) for item in segment_results]
    )
    wavelength_coefficients = np.asarray(
        segment_results[0].wavelength_coefficients,
        dtype=float,
    )
    boundaries = [
        [
            float(np.nanmin(item.spectrum.wavelength)),
            float(np.nanmax(item.spectrum.wavelength)),
        ]
        for item in segment_results
    ]
    provenance = {
        **dict(result.provenance),
        "segmentation": {
            "automatic": True,
            "segment_size_micron": float(segment_size),
            "segment_count": len(segment_results),
            "physical_group_count": len(
                {
                    int(item.spectrum.meta.get("physical_group_index", index))
                    for index, item in enumerate(segment_results)
                }
            ),
            "boundaries_micron": boundaries,
            "wavelength_shifts_micron": [
                float(item.wavelength_shift) for item in segment_results
            ],
            "wavelength_coefficients": [
                np.asarray(item.wavelength_coefficients, dtype=float).tolist()
                for item in segment_results
            ],
        },
    }
    return TelluricFitResult(
        spectrum=spectrum,
        corrected=corrected,
        transmission=transmission,
        continuum=continuum,
        model_flux=model_flux,
        species_scales=dict(result.species_scales),
        wavelength_shift=float(result.wavelength_shift),
        wavelength_coefficients=wavelength_coefficients,
        lsf_sigma_pixels=float(result.lsf_sigma_pixels),
        lsf_box_width_pixels=float(result.lsf_box_width_pixels),
        lsf_lorentz_fwhm_pixels=float(result.lsf_lorentz_fwhm_pixels),
        lsf_wavelength_exponent=float(result.lsf_wavelength_exponent),
        continuum_coefficients=continuum_coefficients,
        metrics=_fit_metrics(spectrum.flux, model_flux, continuum),
        success=bool(result.success),
        message=f"{result.message} (automatic segmentation: {len(segment_results)} segments)",
        cost=float(result.cost),
        nfev=int(result.nfev),
        parameter_names=tuple(result.parameter_names),
        parameter_covariance=result.parameter_covariance,
        parameter_standard_errors=dict(result.parameter_standard_errors),
        species_scale_uncertainties=dict(result.species_scale_uncertainties),
        transmission_uncertainty=transmission_uncertainty,
        reduced_chi_square=float(result.reduced_chi_square),
        covariance_rank=int(result.covariance_rank),
        fit_mask=fit_mask,
        fit_weights=fit_weights,
        stellar_model=stellar_model,
        parameter_bound_status=dict(result.parameter_bound_status),
        wavelength_group_coefficients={
            key: np.asarray(value, dtype=float)
            for key, value in result.wavelength_group_coefficients.items()
        },
        wavelength_group_bounds=dict(result.wavelength_group_bounds),
        provenance=provenance,
    )


def _resolve_line_list(
    spectrum: Spectrum,
    *,
    line_list: LineList | None,
    line_list_path: str | Path | None,
    hitran_par: str | Path | None,
    hitran_species: tuple[str, ...] | None,
    hitran_min_strength: float | None,
    hitran_max_lines: int | None,
    demo_line_list: bool = False,
    line_cutoff_cm: float | None,
    line_wing_mode: str,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_hwf3: float,
    aer_catalog: AERCatalogArtifact | str | Path | None = "auto",
    aer_cache_dir: str | Path | None = None,
    aer_source: str | Path | None = None,
    aer_offline: bool = False,
    aer_reuse_molecfit: bool = True,
    aer_timeout_s: float = 120.0,
    allow_empty_hitran: bool = False,
) -> LineList:
    provided = sum(value is not None for value in (line_list, line_list_path, hitran_par))
    if provided > 1:
        raise ValueError("provide only one of line_list, line_list_path, or hitran_par")
    if line_list is not None:
        return line_list
    if line_list_path is not None:
        return LineList.from_table(line_list_path)
    if hitran_par is not None:
        spectrum_micron = spectrum.to_unit("micron")
        wavenumber = wavelength_micron_to_wavenumber_cm(spectrum_micron.wavelength)
        margin_cm = _line_list_selection_margin_cm(
            spectrum_micron.wavelength,
            line_wing_mode=line_wing_mode,
            line_cutoff_cm=line_cutoff_cm,
            lblrtm_sample=lblrtm_sample,
            lblrtm_alfal0=lblrtm_alfal0,
            lblrtm_hwf3=lblrtm_hwf3,
        )
        return LineList.from_hitran_par(
            hitran_par,
            wavenumber_min=float(np.nanmin(wavenumber) - margin_cm),
            wavenumber_max=float(np.nanmax(wavenumber) + margin_cm),
            species=hitran_species,
            min_strength=hitran_min_strength,
            max_lines=hitran_max_lines,
        )
    if demo_line_list:
        return LineList.demo_near_ir()
    if allow_empty_hitran and (
        aer_catalog is None
        or (aer_catalog == "auto" and hitran_species is None)
    ):
        return LineList.empty_hitran()
    if aer_catalog is not None:
        spectrum_micron = spectrum.to_unit("micron")
        wavenumber = wavelength_micron_to_wavenumber_cm(spectrum_micron.wavelength)
        margin_cm = _line_list_selection_margin_cm(
            spectrum_micron.wavelength,
            line_wing_mode=line_wing_mode,
            line_cutoff_cm=line_cutoff_cm,
            lblrtm_sample=lblrtm_sample,
            lblrtm_alfal0=lblrtm_alfal0,
            lblrtm_hwf3=lblrtm_hwf3,
        )
        resolved_catalog = None if aer_catalog == "auto" else aer_catalog
        return load_aer_line_window(
            wavenumber_min_cm=max(1.0e-9, float(np.nanmin(wavenumber) - margin_cm)),
            wavenumber_max_cm=float(np.nanmax(wavenumber) + margin_cm),
            species=hitran_species,
            min_strength=hitran_min_strength,
            max_lines=hitran_max_lines,
            catalog=resolved_catalog,
            cache_dir=aer_cache_dir,
            source=aer_source,
            offline=aer_offline,
            reuse_molecfit=aer_reuse_molecfit,
            timeout_s=aer_timeout_s,
        ).line_list
    if allow_empty_hitran:
        return LineList.empty_hitran()
    raise ValueError(
        "no molecular line data supplied; provide line_list, line_list_path, "
        "hitran_par, or enable the automatic AER catalogue. Use "
        "demo_line_list=True only for the synthetic demo."
    )


def _line_list_selection_margin_cm(
    wavelength_micron: np.ndarray,
    *,
    line_wing_mode: str,
    line_cutoff_cm: float | None,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_hwf3: float,
) -> float:
    cutoff = line_wing_effective_cutoff_cm(line_wing_mode, line_cutoff_cm)
    if str(line_wing_mode).strip().lower() in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}:
        dynamic_cutoff = lblrtm_dynamic_max_line_cutoff_cm(
            _wavenumber_grid_spacing_cm(wavelength_micron),
            sample=lblrtm_sample,
            alfal0=lblrtm_alfal0,
            hwf3=lblrtm_hwf3,
        )
        if line_cutoff_cm is not None:
            return min(dynamic_cutoff, float(line_cutoff_cm))
        cutoff = dynamic_cutoff if cutoff is None else max(float(cutoff), dynamic_cutoff)
    return max(25.0, 0.0 if cutoff is None else float(cutoff))


def _wavenumber_grid_spacing_cm(wavelength_micron: np.ndarray) -> float:
    wavenumber = wavelength_micron_to_wavenumber_cm(np.asarray(wavelength_micron, dtype=float))
    finite = np.sort(wavenumber[np.isfinite(wavenumber)])
    if finite.size < 2:
        raise ValueError("line selection for LBLRTM line-wing modes requires at least two wavelength pixels")
    spacing = np.diff(finite)
    spacing = spacing[spacing > 0]
    if spacing.size == 0:
        raise ValueError("wavelength grid must span a non-zero range")
    return float(np.nanmedian(spacing))


def _resolve_physical(
    *,
    physical: bool | None,
    atmosphere: AtmosphereProfile | None,
    atmosphere_table: str | Path | None,
    hitran_par: str | Path | None,
    line_list: LineList,
    h2o_continuum: MTCKDH2OContinuum | None,
    components: tuple[AbsorptionComponent, ...] | None,
) -> bool:
    if physical is not None:
        if physical and not line_list.has_hitran_parameters:
            raise ValueError("physical=True requires a HITRAN-style line list")
        return physical
    return (
        atmosphere is not None
        or atmosphere_table is not None
        or hitran_par is not None
        or h2o_continuum is not None
        or components is not None
        or line_list.has_hitran_parameters
    )


def _make_atmosphere(
    *,
    atmosphere_table: str | Path | None,
    atmosphere_mode: str,
    atmosphere_header: Mapping[str, object] | None,
    mipas_profile: str,
    gdas_profile: str | Path | None,
    gdas_mode: str,
    gdas_cache_dir: str | Path | None,
    gdas_download_timeout_s: float,
    observatory_latitude_deg: float | None,
    observatory_longitude_deg: float | None,
    observatory_altitude_m: float | None,
    allow_default_observatory: bool,
    airmass: float,
    pressure_atm: float,
    temperature_k: float,
    path_length_m: float,
    pwv_mm: float | None,
    relative_humidity_percent: float | None,
    mixing_ratios: Mapping[str, float] | None,
    reference_wavenumber_cm: float = 10_000.0,
) -> AtmosphereProfile:
    if atmosphere_table is not None:
        atmosphere = AtmosphereProfile.from_table(atmosphere_table, airmass=airmass)
        return atmosphere.with_pwv_mm(pwv_mm) if pwv_mm is not None else atmosphere

    mode = str(atmosphere_mode).strip().lower().replace("-", "_")
    if mode in {"mipas_gdas", "mipas", "gdas"}:
        if atmosphere_header is not None:
            return AtmosphereProfile.from_fits_header_mipas_gdas(
                atmosphere_header,
                airmass=None if np.isclose(airmass, 1.0) else airmass,
                mipas_profile=mipas_profile,
                gdas_profile=gdas_profile,
                gdas_mode=gdas_mode,
                gdas_cache_dir=gdas_cache_dir,
                gdas_download_timeout_s=gdas_download_timeout_s,
                latitude_deg=observatory_latitude_deg,
                longitude_deg=observatory_longitude_deg,
                observatory_altitude_m=observatory_altitude_m,
                allow_default_observatory=allow_default_observatory,
                relative_humidity_percent=relative_humidity_percent,
                pwv_mm=pwv_mm,
                mixing_ratios=mixing_ratios,
                reference_wavenumber_cm=reference_wavenumber_cm,
            )
        missing_geometry = [
            name
            for name, value in (
                ("observatory_latitude_deg", observatory_latitude_deg),
                ("observatory_longitude_deg", observatory_longitude_deg),
                ("observatory_altitude_m", observatory_altitude_m),
            )
            if value is None
        ]
        if missing_geometry and not allow_default_observatory:
            raise ValueError(
                "MIPAS/GDAS array or text input requires explicit observatory geometry; "
                f"missing {', '.join(missing_geometry)}. Provide the values, pass an "
                "AtmosphereProfile, or set allow_default_observatory=True to explicitly "
                "use the Paranal default."
            )
        return AtmosphereProfile.from_mipas_gdas(
            latitude_deg=(
                DEFAULT_OBSERVATORY_LATITUDE_DEG
                if observatory_latitude_deg is None
                else observatory_latitude_deg
            ),
            longitude_deg=(
                DEFAULT_OBSERVATORY_LONGITUDE_DEG
                if observatory_longitude_deg is None
                else observatory_longitude_deg
            ),
            observatory_altitude_m=(
                DEFAULT_OBSERVATORY_ALTITUDE_M
                if observatory_altitude_m is None
                else observatory_altitude_m
            ),
            airmass=airmass,
            mipas_profile=mipas_profile,
            gdas_profile=gdas_profile,
            gdas_mode=gdas_mode,
            gdas_cache_dir=gdas_cache_dir,
            gdas_download_timeout_s=gdas_download_timeout_s,
            pressure_at_observatory_atm=pressure_atm,
            temperature_at_observatory_k=temperature_k,
            relative_humidity_percent=relative_humidity_percent,
            pwv_mm=pwv_mm,
            mixing_ratios=mixing_ratios,
            reference_wavenumber_cm=reference_wavenumber_cm,
        )

    ratios = dict(DEFAULT_TELLURIC_MIXING_RATIOS)
    if mixing_ratios is not None:
        ratios.update(dict(mixing_ratios))
    if mode == "standard":
        atmosphere = AtmosphereProfile.standard_midlatitude(
            airmass=airmass,
            pressure_at_observatory_atm=pressure_atm,
            temperature_at_observatory_k=temperature_k,
            mixing_ratios=ratios,
        )
    elif mode == "single":
        atmosphere = AtmosphereProfile.single_layer(
            pressure_atm=pressure_atm,
            temperature_k=temperature_k,
            path_length_m=path_length_m,
            airmass=airmass,
            mixing_ratios=ratios,
        )
    else:
        raise ValueError("atmosphere_mode must be 'mipas_gdas', 'single', or 'standard'")
    return atmosphere.with_pwv_mm(pwv_mm) if pwv_mm is not None else atmosphere


def _resolve_initial_wavelength_shift(
    spectrum: Spectrum,
    initial_wavelength_shift: float | None,
    header: Mapping[str, object] | None,
) -> float:
    if initial_wavelength_shift is not None:
        return float(initial_wavelength_shift)
    if header is None:
        return 0.0
    if bool(spectrum.meta.get("observatory_frame_correction", False)):
        return 0.0

    frame_velocity = _spectral_frame_velocity_km_s(header)
    if frame_velocity is None:
        return 0.0
    _, velocity_km_s = frame_velocity

    wavelength = spectrum.to_unit("micron").wavelength
    finite = wavelength[np.isfinite(wavelength)]
    if finite.size == 0:
        return 0.0

    speed_of_light_km_s = SPEED_OF_LIGHT_M_PER_S / 1000.0
    return float(np.nanmedian(finite) * velocity_km_s / speed_of_light_km_s)


def _micron_shift_to_pixel(
    spectrum: Spectrum,
    shift_micron: float,
) -> float:
    wavelength = np.sort(spectrum.to_unit("micron").wavelength)
    steps = np.diff(wavelength)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    if positive.size == 0:
        return 0.0
    return float(shift_micron / np.nanmedian(positive))


def _spectrum_to_observatory_vacuum(
    spectrum: Spectrum,
    header: Mapping[str, object] | None,
) -> Spectrum:
    """Apply Molecfit's AIR_RV/VACUUM_RV preprocessing, then use vacuum.

    Barycentric wavelength products must be moved back to the observatory
    frame before telluric lines are modelled. Molecfit divides by its ERF
    factor first and performs the air-to-vacuum conversion afterwards.
    """

    if header is None:
        return spectrum.to_vacuum()
    frame_velocity = _spectral_frame_velocity_km_s(header)
    if frame_velocity is None:
        return spectrum.to_vacuum()
    frame_name, velocity_km_s = frame_velocity

    speed_of_light_km_s = SPEED_OF_LIGHT_M_PER_S / 1000.0
    erf_factor = (1.0 + 1.55e-8) * (1.0 + velocity_km_s / speed_of_light_km_s)
    observatory = Spectrum(
        wavelength=spectrum.wavelength / erf_factor,
        flux=spectrum.flux.copy(),
        uncertainty=None if spectrum.uncertainty is None else spectrum.uncertainty.copy(),
        mask=None if spectrum.mask is None else spectrum.mask.copy(),
        group_id=None if spectrum.group_id is None else spectrum.group_id.copy(),
        wavelength_unit=spectrum.wavelength_unit,
        wavelength_medium=spectrum.wavelength_medium,
        meta={
            **dict(spectrum.meta),
            "observatory_frame_correction": True,
            "observatory_erf_factor": erf_factor,
            "original_spectral_frame": frame_name,
            "observatory_frame_velocity_km_s": velocity_km_s,
        },
    )
    return observatory.to_vacuum()


def _ranges_to_observatory_vacuum(
    ranges: tuple[tuple[float, float], ...] | None,
    wavelength_medium: str,
    header: Mapping[str, object] | None,
) -> tuple[tuple[float, float], ...] | None:
    """Transform micron-valued fit windows through the spectrum frame path."""

    if ranges is None:
        return None
    flattened = np.asarray(ranges, dtype=float).reshape(-1)
    marker = Spectrum(
        wavelength=flattened,
        flux=np.ones(flattened.shape, dtype=float),
        wavelength_unit="micron",
        wavelength_medium=wavelength_medium,
    )
    converted = _spectrum_to_observatory_vacuum(marker, header)
    values = converted.wavelength.reshape(-1, 2)
    return tuple((float(lower), float(upper)) for lower, upper in values)


def _resolve_region_file_ranges(
    *,
    region_file: str | Path | None,
    fit_ranges: tuple[tuple[float, float], ...] | None,
    exclude_ranges: tuple[tuple[float, float], ...] | None,
    spectrum: Spectrum,
) -> tuple[
    tuple[tuple[float, float], ...] | None,
    tuple[tuple[float, float], ...] | None,
]:
    if region_file is None:
        return fit_ranges, exclude_ranges
    if fit_ranges is not None or exclude_ranges is not None:
        raise ValueError(
            "region_file cannot be combined with fit_ranges or exclude_ranges"
        )

    selection = load_region_file(region_file)
    if selection.is_empty:
        raise ValueError(f"region file {region_file} does not contain any regions")
    converted = selection.converted(
        wavelength_unit="micron",
        wavelength_medium=spectrum.wavelength_medium,
    )
    return (
        converted.fit_ranges or None,
        converted.exclude_ranges or None,
    )


def _merge_exclusion_ranges(
    first: tuple[tuple[float, float], ...] | None,
    second: tuple[tuple[float, float], ...] | None,
) -> tuple[tuple[float, float], ...] | None:
    combined = tuple(first or ()) + tuple(second or ())
    if not combined:
        return None
    return RegionSelection(
        exclude_ranges=combined,
        wavelength_unit="micron",
        wavelength_medium="vacuum",
    ).exclude_ranges


def _stellar_template_frame_correction_factor(
    observatory_spectrum: Spectrum,
    header: Mapping[str, object] | None,
) -> float:
    """Return the barycentric-to-observatory factor for a stellar template."""

    applied = observatory_spectrum.meta.get("observatory_erf_factor")
    try:
        applied_value = float(applied)
    except (TypeError, ValueError):
        applied_value = np.nan
    if np.isfinite(applied_value) and applied_value > 0:
        return applied_value
    if header is None:
        return 1.0

    velocity_km_s = _first_header_float(
        header,
        (
            "ESO DRS BERV",
            "HIERARCH ESO DRS BERV",
            "BERV",
            "BARYCORR",
        ),
    )
    if not np.isfinite(velocity_km_s):
        velocity_km_s = _barycentric_velocity_from_header_km_s(header)
    if not np.isfinite(velocity_km_s):
        return 1.0
    speed_of_light_km_s = SPEED_OF_LIGHT_M_PER_S / 1000.0
    return float((1.0 + 1.55e-8) * (1.0 + velocity_km_s / speed_of_light_km_s))


def _first_header_float(header: Mapping[str, object], keys: tuple[str, ...]) -> float:
    for key in keys:
        try:
            value = header[key]
        except Exception:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return np.nan


def _spectral_frame_velocity_km_s(
    header: Mapping[str, object],
) -> tuple[str, float] | None:
    specs = str(header.get("SPECSYS", "")).strip().upper()
    if specs in {"BARYCENT", "BARYCENTRIC"}:
        velocity = _first_header_float(
            header,
            ("ESO DRS BERV", "HIERARCH ESO DRS BERV", "BERV", "BARYCORR"),
        )
        if not np.isfinite(velocity):
            velocity = _barycentric_velocity_from_header_km_s(header)
        return None if not np.isfinite(velocity) else ("BARYCENTRIC", float(velocity))

    heliocentric = specs in {"HELIOCEN", "HELIOCENT", "HELIOCENTRIC"}
    if not heliocentric:
        note = str(header.get("HELIOCNT", "")).strip().upper()
        heliocentric = bool(note) and any(token in note for token in ("APPLIED", "HELIO"))
    if heliocentric:
        velocity = _first_header_float(
            header,
            ("HELIOVEL", "VHELIO", "HELIO_RV", "HELCORR"),
        )
        return None if not np.isfinite(velocity) else ("HELIOCENTRIC", float(velocity))
    return None


def _barycentric_velocity_from_header_km_s(
    header: Mapping[str, object],
) -> float:
    """Reconstruct a missing BERV from standard observation metadata."""

    ra_deg = _first_header_float(header, ("RA", "OBJRA"))
    dec_deg = _first_header_float(header, ("DEC", "OBJDEC"))
    longitude_deg = _first_header_float(
        header,
        (
            "ESO TEL GEOLON",
            "HIERARCH ESO TEL GEOLON",
            "ESO TEL1 GEOLON",
            "HIERARCH ESO TEL1 GEOLON",
            "LONGITUD",
            "OBSGEO-L",
        ),
    )
    latitude_deg = _first_header_float(
        header,
        (
            "ESO TEL GEOLAT",
            "HIERARCH ESO TEL GEOLAT",
            "ESO TEL1 GEOLAT",
            "HIERARCH ESO TEL1 GEOLAT",
            "LATITUDE",
            "OBSGEO-B",
        ),
    )
    altitude_m = _first_header_float(
        header,
        (
            "ESO TEL GEOELEV",
            "HIERARCH ESO TEL GEOELEV",
            "ESO TEL1 GEOELEV",
            "HIERARCH ESO TEL1 GEOELEV",
            "ALTITUDE",
            "OBSGEO-H",
        ),
    )
    observation_time = _header_representative_observation_time(header)
    date_obs = str(header.get("DATE-OBS", "")).strip()
    required = np.asarray(
        (ra_deg, dec_deg, longitude_deg, latitude_deg, altitude_m),
        dtype=float,
    )
    if observation_time is None and not date_obs:
        return np.nan
    if not np.all(np.isfinite(required)):
        return np.nan

    try:
        location = EarthLocation.from_geodetic(
            longitude_deg * u.deg,
            latitude_deg * u.deg,
            altitude_m * u.m,
        )
        target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        if observation_time is None:
            observation_time = Time(date_obs, format="isot", scale="utc")
        correction = target.radial_velocity_correction(
            obstime=observation_time,
            location=location,
        )
    except (TypeError, ValueError):
        return np.nan
    return float(correction.to_value(u.km / u.s))


def _load_fits_header_if_available(
    input_path: str | Path,
    input_format: str | None,
    *,
    hdu: int = 1,
) -> Mapping[str, object] | None:
    path = Path(input_path)
    chosen_format = infer_spectrum_format(path, input_format)
    if chosen_format not in {"fits", "fit", "fz"}:
        return None
    try:
        with fits.open(path) as hdul:
            header = dict(hdul[0].header)
            if 0 <= int(hdu) < len(hdul) and int(hdu) != 0:
                for key, value in hdul[int(hdu)].header.items():
                    header.setdefault(key, value)
            return header
    except Exception:
        return None


def _resolve_wavelength_medium(
    wavelength_medium: str | None,
    header: Mapping[str, object] | None,
    *,
    wavelength_col: int | str | None = None,
) -> str:
    if wavelength_medium is not None:
        return wavelength_medium

    inferred = _infer_wavelength_medium_from_header(
        header,
        wavelength_col=wavelength_col,
    )
    if inferred is not None:
        return inferred

    raise WavelengthMetadataError(
        "PyMolFit stopped the correction because wavelength_medium was not "
        "provided and no reliable air/vacuum wavelength convention was found "
        "in the FITS metadata. Pass wavelength_medium='air' or "
        "wavelength_medium='vacuum'. SPECSYS is not sufficient because it "
        "describes the velocity reference frame, not whether wavelengths are "
        "in air or vacuum."
    )

def _resolve_partition_table(partition_table: PartitionTable | str | Path | None) -> PartitionTable | None:
    if partition_table is None:
        return PartitionTable.from_lblrtm_package_data()
    if isinstance(partition_table, PartitionTable):
        return partition_table
    return PartitionTable.from_table(partition_table)


def _resolve_h2o_continuum(
    h2o_continuum: MTCKDH2OContinuum | LBLRTMH2OContinuum | str | Path | None,
) -> MTCKDH2OContinuum | LBLRTMH2OContinuum | None:
    if h2o_continuum is None:
        return None
    if isinstance(h2o_continuum, (MTCKDH2OContinuum, LBLRTMH2OContinuum)):
        return h2o_continuum
    continuum_name = str(h2o_continuum).strip().lower()
    if continuum_name in {"none", "off", "false"}:
        return None
    if continuum_name == "lblrtm":
        return LBLRTMH2OContinuum.from_package_data()
    return MTCKDH2OContinuum.from_netcdf(h2o_continuum)


def _resolve_tabulated_continuum(
    continuum: TabulatedContinuum | LBLRTMCO2Continuum | str | Path | None,
) -> TabulatedContinuum | LBLRTMCO2Continuum | None:
    if continuum is None:
        return None
    if isinstance(continuum, (TabulatedContinuum, LBLRTMCO2Continuum)):
        return continuum
    continuum_name = str(continuum).strip().lower()
    if continuum_name in {"none", "off", "false"}:
        return None
    if continuum_name == "lblrtm":
        return LBLRTMCO2Continuum.from_package_data()
    return TabulatedContinuum.from_table(continuum)


def _resolve_cia_table(cia: HitranCIATable | str | Path | None) -> HitranCIATable | None:
    if cia is None or isinstance(cia, HitranCIATable):
        return cia
    return HitranCIATable.from_hitran_cia(cia)


def _resolve_pair_cia_tables(
    cia_tables: Mapping[str, HitranCIATable | str | Path] | None,
) -> dict[str, HitranCIATable]:
    if cia_tables is None:
        return {}
    resolved: dict[str, HitranCIATable] = {}
    for name, table in cia_tables.items():
        resolved[str(name)] = table if isinstance(table, HitranCIATable) else HitranCIATable.from_hitran_cia(table)
    return resolved


def _build_components(
    *,
    extra_components: tuple[AbsorptionComponent, ...] | None,
    line_list: LineList,
    chunk_size: int,
    partition_table: PartitionTable | None,
    line_cutoff_cm: float | None,
    subtract_cutoff_profile: bool,
    line_taper_cm: float,
    line_wing_mode: str,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_avmass_amu: float,
    lblrtm_hwf3: float,
    rayleigh: bool,
    rayleigh_xrayl: float,
    n2_continuum: bool,
    n2_continuum_xn2cn: float,
    o2_continuum: bool,
    o2_continuum_xo2cn: float,
    h2o_continuum: MTCKDH2OContinuum | None,
    h2o_continuum_foreign_closure: bool,
    co2_continuum: TabulatedContinuum | None,
    o2_cia: HitranCIATable | None,
    n2_cia: HitranCIATable | None,
    cia_tables: Mapping[str, HitranCIATable] | None,
) -> tuple[AbsorptionComponent, ...] | None:
    if (
        all(value is None for value in (extra_components, h2o_continuum, co2_continuum, o2_cia, n2_cia))
        and not rayleigh
        and not n2_continuum
        and not o2_continuum
        and not cia_tables
    ):
        return None

    if n2_continuum:
        overlapping = []
        for label, table in (
            ("n2_cia", n2_cia),
            *((str(name), table) for name, table in (cia_tables or {}).items()),
        ):
            if table is not None and _overlaps_lblrtm_n2_continuum(table):
                overlapping.append(label)
        if overlapping:
            labels = ", ".join(overlapping)
            raise ValueError(
                "n2_continuum=True overlaps N2 collision-induced absorption "
                f"provided by {labels}; use the source-backed LBLRTM N2 continuum "
                "or those CIA tables, not both"
            )
    if o2_continuum:
        overlapping = []
        for label, table in (
            ("o2_cia", o2_cia),
            *((str(name), table) for name, table in (cia_tables or {}).items()),
        ):
            if table is not None and _overlaps_lblrtm_o2_continuum(table):
                overlapping.append(label)
        if overlapping:
            labels = ", ".join(overlapping)
            raise ValueError(
                "o2_continuum=True overlaps O2 collision-induced absorption "
                f"provided by {labels}; use the source-backed LBLRTM O2 continuum "
                "or those CIA tables, not both"
            )

    built: list[AbsorptionComponent] = []
    if line_list.has_hitran_parameters:
        built.append(
            HitranLineAbsorption(
                line_list=line_list,
                chunk_size=chunk_size,
                partition_table=partition_table,
                line_cutoff_cm=line_cutoff_cm,
                subtract_cutoff_profile=subtract_cutoff_profile,
                line_taper_cm=line_taper_cm,
                line_wing_mode=line_wing_mode,
                lblrtm_sample=lblrtm_sample,
                lblrtm_alfal0=lblrtm_alfal0,
                lblrtm_avmass_amu=lblrtm_avmass_amu,
                lblrtm_hwf3=lblrtm_hwf3,
            )
        )
    if h2o_continuum is not None:
        built.append(
            H2OContinuumAbsorption(
                continuum=h2o_continuum,
                use_foreign_closure=h2o_continuum_foreign_closure,
            )
        )
    if co2_continuum is not None:
        built.append(CO2ContinuumAbsorption(co2_continuum))
    if o2_cia is not None:
        built.append(O2CIAAbsorption(o2_cia))
    if n2_cia is not None:
        built.append(N2CIAAbsorption(n2_cia))
    for name, table in (cia_tables or {}).items():
        built.append(PairCIAAbsorption(table, basis_name=str(name)))
    if rayleigh:
        built.append(RayleighScatteringAbsorption(xrayl=rayleigh_xrayl))
    if n2_continuum:
        built.append(N2ContinuumAbsorption(xn2cn=n2_continuum_xn2cn))
    if o2_continuum:
        built.append(O2ContinuumAbsorption(xo2cn=o2_continuum_xo2cn))
    if extra_components is not None:
        built.extend(extra_components)
    return tuple(built)


def _overlaps_lblrtm_n2_continuum(table: HitranCIATable) -> bool:
    pair = tuple(str(species).strip().upper() for species in (table.pair or ()))
    return "N2" in pair and any(
        partner in {"N2", "O2", "H2O", "AIR"} for partner in pair
    )


def _overlaps_lblrtm_o2_continuum(table: HitranCIATable) -> bool:
    pair = tuple(str(species).strip().upper() for species in (table.pair or ()))
    return "O2" in pair and any(
        partner in {"N2", "O2", "H2O", "AIR"} for partner in pair
    )
