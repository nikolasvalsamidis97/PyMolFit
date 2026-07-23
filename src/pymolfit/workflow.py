from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from .aer_data import AERCatalogArtifact, load_aer_line_window
from .atmosphere import (
    AtmosphereProfile,
    DEFAULT_OBSERVATORY_ALTITUDE_M,
    DEFAULT_OBSERVATORY_LATITUDE_DEG,
    DEFAULT_OBSERVATORY_LONGITUDE_DEG,
    DEFAULT_TELLURIC_MIXING_RATIOS,
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
from .fit import (
    FitConfig,
    MultiTelluricFitResult,
    TelluricFitResult,
    _apply_multi_fit_to_segment,
    _fit_metrics,
    _radiative_transfer_point_count,
    fit_telluric_segments,
    fit_tellurics,
)
from .io import infer_spectrum_format, load_spectrum, save_spectrum
from .linelist import LineList
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
from .spectrum import Spectrum


DEFAULT_SEGMENT_SIZE_MICRON = 0.01

InputSpectrumFormat = Literal["txt", "dat", "csv", "ascii", "ecsv", "fits", "fit", "fz"]
WavelengthMedium = Literal["vacuum", "vac", "air"]
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
    wavelength_unit: str = "micron",
    wavelength_medium: WavelengthMedium = "vacuum",
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
    solve_continuum_linear: bool = False,
    lsf_sigma_pixels: float = 0.0,
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: float = 0.0,
    lsf_variable_width: bool = False,
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
    min_transmission: float = 0.03,
    fit_wavelength_shift: bool = False,
    fit_wavelength_polynomial: bool = False,
    wavelength_polynomial_order: int = 1,
    fit_segment_wavelength_shifts: bool = False,
    fit_segment_wavelength_polynomial: bool = False,
    segment_wavelength_polynomial_order: int = 1,
    initial_wavelength_shift: float | None = None,
    wavelength_shift_bounds: tuple[float, float] = (-5.0e-4, 5.0e-4),
    fit_lsf_sigma: bool = False,
    lsf_sigma_bounds: tuple[float, float] = (0.0, 5.0),
    fit_lsf_box_width: bool = False,
    lsf_box_width_bounds: tuple[float, float] = (0.0, 10.0),
    fit_lsf_lorentz_fwhm: bool = False,
    lsf_lorentz_fwhm_bounds: tuple[float, float] = (0.0, 10.0),
    fit_ranges: tuple[tuple[float, float], ...] | None = None,
    exclude_ranges: tuple[tuple[float, float], ...] | None = None,
    loss: LeastSquaresLoss = "linear",
    f_scale: float = 1.0,
    ftol: float = 1.0e-10,
    xtol: float = 1.0e-10,
    gtol: float = 1.0e-10,
    estimate_uncertainties: bool = False,
) -> TelluricFitResult:
    """High-level telluric correction for arrays.

    This is the notebook-friendly workflow: pass wavelength/flux arrays and,
    optionally, a HITRAN `.par` or PyMolFit line list. The lower-level
    `fit_tellurics` function remains available for full manual control.
    """

    spectrum = Spectrum(
        wavelength=wavelength,
        flux=flux,
        uncertainty=uncertainty,
        wavelength_unit=wavelength_unit,
        wavelength_medium=wavelength_medium,
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
        atmosphere_header=None,
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
    wavelength_unit: str = "micron",
    wavelength_medium: WavelengthMedium = "vacuum",
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
    solve_continuum_linear: bool = False,
    lsf_sigma_pixels: float = 0.0,
    lsf_box_width_pixels: float = 0.0,
    lsf_lorentz_fwhm_pixels: float = 0.0,
    lsf_variable_width: bool = False,
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
    min_transmission: float = 0.03,
    fit_wavelength_shift: bool = False,
    fit_wavelength_polynomial: bool = False,
    wavelength_polynomial_order: int = 1,
    fit_segment_wavelength_shifts: bool = False,
    fit_segment_wavelength_polynomial: bool = False,
    segment_wavelength_polynomial_order: int = 1,
    initial_wavelength_shift: float | None = None,
    wavelength_shift_bounds: tuple[float, float] = (-5.0e-4, 5.0e-4),
    fit_lsf_sigma: bool = False,
    lsf_sigma_bounds: tuple[float, float] = (0.0, 5.0),
    fit_lsf_box_width: bool = False,
    lsf_box_width_bounds: tuple[float, float] = (0.0, 10.0),
    fit_lsf_lorentz_fwhm: bool = False,
    lsf_lorentz_fwhm_bounds: tuple[float, float] = (0.0, 10.0),
    fit_ranges: tuple[tuple[float, float], ...] | None = None,
    exclude_ranges: tuple[tuple[float, float], ...] | None = None,
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
    :param wavelength_unit: Unit of input wavelengths, such as ``micron``/``um``, ``nm``, ``angstrom``/``aa``, or ``m``.
    :param wavelength_medium: ``air`` declares standard-air wavelengths; ``vacuum`` or ``vac`` declares vacuum wavelengths. PyMolFit converts internally, rather than changing what the file contains.
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
    :param solve_continuum_linear: ``True`` solves continuum coefficients exactly at each nonlinear step and requires ``loss="linear"``; ``False`` includes them in the nonlinear parameter vector.
    :param lsf_sigma_pixels: The line-spread function (LSF) describes instrumental broadening of intrinsically narrow telluric lines; this is its initial/fixed Gaussian sigma in detector pixels, and ``0`` disables that component unless it is fitted.
    :param lsf_box_width_pixels: A boxcar LSF component approximates finite pixel/slit integration; this is its initial/fixed width in detector pixels, and ``0`` disables it.
    :param lsf_lorentz_fwhm_pixels: A Lorentzian LSF component represents extended instrumental wings; this is its initial/fixed full width at half maximum in detector pixels, and ``0`` disables it unless fitted.
    :param lsf_variable_width: Instruments can have wavelength-dependent resolution; ``True`` scales all LSF widths with wavelength relative to ``lsf_reference_wavelength_micron``, while ``False`` keeps their pixel widths constant.
    :param lsf_reference_wavelength_micron: Reference wavelength in microns for variable-width LSF scaling; ``None`` uses each segment's median wavelength.
    :param lsf_kernel_width_fwhm: Half-support control for numerical LSF kernels in multiples of component FWHM; larger values retain farther wings but cost more.
    :param lsf_molecfit_voigt: ``True`` uses Molecfit's synthetic Gaussian-plus-Lorentzian Voigt approximation; ``False`` convolves the configured components normally.
    :param high_resolution_grid: ``True`` computes, convolves, and rebins an oversampled internal model; ``False`` evaluates directly at observed samples.
    :param high_resolution_oversampling: Approximate internal samples per observed pixel when ``high_resolution_grid=True``.
    :param high_resolution_margin_pixels: Extra internal-grid margin on each segment edge in observed-pixel units, reducing convolution edge effects.
    :param high_resolution_rebin_mode: ``integrate`` averages pixel bins, ``center`` samples centres, ``sample_average`` averages enclosed samples, ``molecfit_overlap`` uses Molecfit-style overlap weights, and ``molecfit_average`` aliases sample averaging.
    :param radiative_transfer_grid: ``auto`` uses a layer-resolved native wavenumber grid; ``model`` evaluates opacity directly on the model grid for lower cost and lower fidelity.
    :param radiative_transfer_step_cm: Explicit native radiative-transfer spacing in inverse centimetres; ``None`` derives it from atmospheric layers and line widths.
    :param radiative_transfer_max_points: Maximum native-grid samples before PyMolFit asks for segmentation or a larger explicit safety limit.
    :param auto_segment: ``True`` splits broad spectra into shared-parameter segments automatically; ``False`` fits the input as one segment and may exceed grid limits.
    :param segment_size: Maximum automatic segment width in microns; ``0.01`` equals 100 Angstrom.
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
    :param fit_wavelength_shift: ``True`` fits one constant wavelength offset; ``False`` fixes the initial/header-derived offset.
    :param fit_wavelength_polynomial: ``True`` fits a global wavelength-correction polynomial; ``False`` disables it. Do not combine it with ``fit_wavelength_shift``.
    :param wavelength_polynomial_order: Degree of the global wavelength-correction polynomial in normalized wavelength coordinates.
    :param fit_segment_wavelength_shifts: ``True`` fits an independent constant wavelength offset for every automatic segment, which is useful for echelle orders or detectors with separate wavelength solutions; it cannot be combined with either global wavelength-fit option.
    :param fit_segment_wavelength_polynomial: ``True`` fits an independent wavelength-correction polynomial for every automatic segment; use this only when line residuals show within-segment wavelength distortion, and do not combine it with global or constant per-segment wavelength fitting.
    :param segment_wavelength_polynomial_order: Degree of each independent segment wavelength polynomial when ``fit_segment_wavelength_polynomial=True``; ``0`` is a constant offset and ``1`` also permits a linear distortion.
    :param initial_wavelength_shift: Initial constant wavelength offset in microns; ``None`` derives a suitable initial value from FITS spectral-frame metadata.
    :param wavelength_shift_bounds: Inclusive lower and upper allowed constant shift in microns; the upper value must exceed the lower.
    :param fit_lsf_sigma: ``True`` fits Gaussian LSF sigma from ``lsf_sigma_pixels`` within ``lsf_sigma_bounds``; ``False`` keeps it fixed.
    :param lsf_sigma_bounds: Lower and upper Gaussian sigma bounds in pixels; values must be increasing and non-negative.
    :param fit_lsf_box_width: ``True`` fits boxcar LSF width within ``lsf_box_width_bounds``; ``False`` keeps it fixed.
    :param lsf_box_width_bounds: Lower and upper boxcar-width bounds in pixels; values must be increasing and non-negative.
    :param fit_lsf_lorentz_fwhm: ``True`` fits Lorentzian FWHM within ``lsf_lorentz_fwhm_bounds``; ``False`` keeps it fixed.
    :param lsf_lorentz_fwhm_bounds: Lower and upper Lorentzian FWHM bounds in pixels; values must be increasing and non-negative.
    :param fit_ranges: Wavelength intervals whose observed telluric features constrain molecular columns, wavelength alignment, LSF, and continuum; supply ``((start, stop), ...)`` in microns and the declared input medium, or ``None`` to let every valid pixel influence those fitted parameters.
    :param exclude_ranges: Wavelength intervals ignored only while estimating fit parameters, normally to protect stellar/circumstellar lines, detector defects, or saturated pixels; values are in microns/input medium and the final atmospheric correction is still evaluated there.
    :param loss: SciPy least-squares loss: ``linear`` is ordinary squared residuals; ``soft_l1`` and ``huber`` are moderate robust losses; ``cauchy`` and ``arctan`` suppress outliers more strongly.
    :param f_scale: Residual scale separating inliers from outliers for robust losses; it has no effect for ``loss="linear"``.
    :param ftol: Positive relative cost-change tolerance for optimizer termination.
    :param xtol: Positive relative parameter-step tolerance for optimizer termination.
    :param gtol: Positive gradient-norm tolerance for optimizer termination.
    :param estimate_uncertainties: ``True`` estimates local covariance and propagates transmission uncertainty; ``False`` skips this additional work.
    :param product_path: Optional path for the full fit-product table containing model, transmission, masks, corrected flux, metadata, and provenance.
    :param product_format: Astropy table writer format used for ``product_path``, for example ``ascii.ecsv`` or ``fits``.
    :param plot_path: Optional diagnostic-plot output path; ``None`` does not save a plot.
    :param show_plot: ``True`` opens/displays the diagnostic plot; ``False`` only saves it when ``plot_path`` is provided.
    :return: ``TelluricFitResult`` containing the input, model, transmission, corrected spectrum, fitted parameters, diagnostics, and provenance.
    """

    spectrum = load_spectrum(
        input_path,
        format=input_format,
        wavelength_col=wavelength_col,
        flux_col=flux_col,
        uncertainty_col=uncertainty_col,
        wavelength_unit=wavelength_unit,
        wavelength_medium=wavelength_medium,
    )
    atmosphere_header = _load_fits_header_if_available(input_path, input_format)
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
        pwv_mm=pwv_mm,
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
        loss=loss,
        f_scale=f_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        estimate_uncertainties=estimate_uncertainties,
    )

    if output_path is not None:
        save_spectrum(output_path, result.corrected)
    if product_path is not None:
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
    solve_continuum_linear: bool,
    lsf_sigma_pixels: float,
    lsf_box_width_pixels: float,
    lsf_lorentz_fwhm_pixels: float,
    lsf_variable_width: bool,
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
    fit_wavelength_shift: bool,
    fit_wavelength_polynomial: bool,
    wavelength_polynomial_order: int,
    fit_segment_wavelength_shifts: bool,
    fit_segment_wavelength_polynomial: bool,
    segment_wavelength_polynomial_order: int,
    initial_wavelength_shift: float | None,
    wavelength_shift_bounds: tuple[float, float],
    fit_lsf_sigma: bool,
    lsf_sigma_bounds: tuple[float, float],
    fit_lsf_box_width: bool,
    lsf_box_width_bounds: tuple[float, float],
    fit_lsf_lorentz_fwhm: bool,
    lsf_lorentz_fwhm_bounds: tuple[float, float],
    fit_ranges: tuple[tuple[float, float], ...] | None,
    exclude_ranges: tuple[tuple[float, float], ...] | None,
    loss: str,
    f_scale: float,
    ftol: float,
    xtol: float,
    gtol: float,
    estimate_uncertainties: bool,
) -> TelluricFitResult:
    input_medium = spectrum.wavelength_medium
    fit_ranges = _ranges_to_observatory_vacuum(fit_ranges, input_medium, atmosphere_header)
    exclude_ranges = _ranges_to_observatory_vacuum(exclude_ranges, input_medium, atmosphere_header)
    spectrum = _spectrum_to_observatory_vacuum(spectrum, atmosphere_header)
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
        fixed_component_scales["AIR"] = 1.0

    fit_config = FitConfig(
        airmass=fit_airmass,
        continuum_order=continuum_order,
        fixed_species_scales=fixed_component_scales or None,
        solve_continuum_linear=solve_continuum_linear,
        lsf_sigma_pixels=lsf_sigma_pixels,
        lsf_box_width_pixels=lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=lsf_lorentz_fwhm_pixels,
        lsf_variable_width=lsf_variable_width,
        lsf_reference_wavelength_micron=lsf_reference_wavelength_micron,
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
        atmosphere=resolved_atmosphere,
        partition_table=resolved_partition,
        h2o_continuum=resolved_h2o_continuum,
        h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
        components=resolved_components,
        fit_wavelength_shift=fit_wavelength_shift,
        fit_wavelength_polynomial=fit_wavelength_polynomial,
        wavelength_polynomial_order=wavelength_polynomial_order,
        fit_segment_wavelength_shifts=fit_segment_wavelength_shifts,
        fit_segment_wavelength_polynomial=fit_segment_wavelength_polynomial,
        segment_wavelength_polynomial_order=segment_wavelength_polynomial_order,
        initial_wavelength_shift=resolved_initial_wavelength_shift,
        wavelength_shift_bounds=wavelength_shift_bounds,
        fit_lsf_sigma=fit_lsf_sigma,
        lsf_sigma_bounds=lsf_sigma_bounds,
        fit_lsf_box_width=fit_lsf_box_width,
        lsf_box_width_bounds=lsf_box_width_bounds,
        fit_lsf_lorentz_fwhm=fit_lsf_lorentz_fwhm,
        lsf_lorentz_fwhm_bounds=lsf_lorentz_fwhm_bounds,
        fit_ranges=fit_ranges,
        exclude_ranges=exclude_ranges,
        loss=loss,
        f_scale=f_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        estimate_uncertainties=estimate_uncertainties,
    )
    if auto_segment and (not np.isfinite(segment_size) or segment_size <= 0):
        raise ValueError("segment_size must be a positive finite value in microns")
    per_segment_wavelength_fit = (
        fit_segment_wavelength_shifts or fit_segment_wavelength_polynomial
    )
    if not auto_segment:
        if per_segment_wavelength_fit:
            raise ValueError(
                "per-segment wavelength fitting requires auto_segment=True"
            )
        return fit_tellurics(spectrum, line_list=resolved_line_list, config=fit_config)
    if not resolved_high_resolution_grid and not per_segment_wavelength_fit:
        return fit_tellurics(spectrum, line_list=resolved_line_list, config=fit_config)
    segments = _split_spectrum(
        spectrum,
        segment_size=segment_size,
        minimum_points=continuum_order + 2,
    )
    if resolved_high_resolution_grid:
        segments = _subdivide_segments_for_grid_limit(
            segments,
            config=fit_config,
            minimum_points=continuum_order + 2,
        )
    if len(segments) == 1 and not per_segment_wavelength_fit:
        return fit_tellurics(spectrum, line_list=resolved_line_list, config=fit_config)
    active = tuple(
        _segment_has_fit_pixels(segment, fit_config)
        for segment in segments
    )
    active_segments = tuple(
        segment for segment, is_active in zip(segments, active, strict=True) if is_active
    )
    if not active_segments:
        raise ValueError("fit_ranges and exclude_ranges leave no segment with enough fit pixels")
    full_wavelength_micron = spectrum.to_unit("micron").wavelength
    full_bounds = (
        float(np.nanmin(full_wavelength_micron)),
        float(np.nanmax(full_wavelength_micron)),
    )
    multi_result = fit_telluric_segments(
        active_segments,
        line_list=resolved_line_list,
        config=fit_config,
        global_wavelength_bounds=full_bounds,
    )
    fitted_results = iter(multi_result.segment_results)
    segment_results = tuple(
        next(fitted_results)
        if is_active
        else _apply_multi_fit_to_segment(
            segment,
            line_list=resolved_line_list,
            config=fit_config,
            fit_result=multi_result,
            global_wavelength_bounds=full_bounds,
        )
        for segment, is_active in zip(segments, active, strict=True)
    )
    return _stitch_segment_results(
        multi_result,
        segment_size=segment_size,
        segment_results=segment_results,
    )


def _split_spectrum(
    spectrum: Spectrum,
    *,
    segment_size: float,
    minimum_points: int = 3,
) -> tuple[Spectrum, ...]:
    """Split a spectrum into contiguous wavelength intervals in microns."""

    if not np.isfinite(segment_size) or segment_size <= 0:
        raise ValueError("segment_size must be a positive finite value in microns")
    if minimum_points < 2:
        raise ValueError("minimum_points must be at least two")

    ordered = spectrum.to_unit("micron").sorted()
    wavelength = ordered.wavelength
    if wavelength.size < minimum_points:
        return (ordered,)
    if not np.all(np.isfinite(wavelength)):
        raise ValueError(
            "automatic segmentation requires finite wavelengths; remove or mask "
            "rows with invalid wavelength coordinates"
        )

    positive_steps = np.diff(wavelength)
    finite_positive_steps = positive_steps[
        np.isfinite(positive_steps) & (positive_steps > 0)
    ]
    representative_step = (
        float(np.nanmedian(finite_positive_steps))
        if finite_positive_steps.size
        else np.inf
    )
    gap_stops = np.flatnonzero(positive_steps > 10.0 * representative_step) + 1

    boundaries = [0]
    for chunk_start, chunk_stop in zip(
        np.concatenate(([0], gap_stops)),
        np.concatenate((gap_stops, [wavelength.size])),
        strict=True,
    ):
        chunk_wavelength = wavelength[chunk_start:chunk_stop]
        span = float(chunk_wavelength[-1] - chunk_wavelength[0])
        if span > segment_size:
            ratio = span / segment_size
            ratio -= 1.0e-12 * max(1.0, abs(ratio))
            segment_count = max(1, int(np.ceil(ratio)))
            edges = np.linspace(chunk_wavelength[0], chunk_wavelength[-1], segment_count + 1)
            boundaries.extend(
                np.searchsorted(wavelength, edges[1:-1], side="left").tolist()
            )
        boundaries.append(int(chunk_stop))
    boundaries = np.asarray(boundaries, dtype=int)
    ranges = [
        [int(start), int(stop)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if stop > start
    ]

    index = 0
    while index < len(ranges):
        start, stop = ranges[index]
        if stop - start >= minimum_points or len(ranges) == 1:
            index += 1
            continue
        if index > 0:
            ranges[index - 1][1] = stop
            ranges.pop(index)
        else:
            ranges[1][0] = start
            ranges.pop(0)

    segments = []
    for segment_index, (start, stop) in enumerate(ranges):
        uncertainty = (
            None
            if ordered.uncertainty is None
            else ordered.uncertainty[start:stop].copy()
        )
        mask = None if ordered.mask is None else ordered.mask[start:stop].copy()
        segments.append(
            Spectrum(
                wavelength=ordered.wavelength[start:stop].copy(),
                flux=ordered.flux[start:stop].copy(),
                uncertainty=uncertainty,
                mask=mask,
                wavelength_unit="micron",
                wavelength_medium=ordered.wavelength_medium,
                meta={
                    **dict(ordered.meta),
                    "segment_index": segment_index,
                    "segment_count": len(ranges),
                    "segment_size_micron": float(segment_size),
                },
            )
        )
    return tuple(segments)


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
    return bool(np.count_nonzero(selected) >= config.continuum_order + 2)


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
    return Spectrum(
        wavelength=np.concatenate([spectrum.wavelength for spectrum in spectra]),
        flux=np.concatenate([spectrum.flux for spectrum in spectra]),
        uncertainty=uncertainty,
        mask=mask,
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
            key=lambda item: float(np.nanmin(item.spectrum.wavelength)),
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
    fit_mask = np.concatenate(
        [
            np.zeros(item.spectrum.wavelength.size, dtype=bool)
            if item.fit_mask is None
            else np.asarray(item.fit_mask, dtype=bool)
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
        parameter_bound_status=dict(result.parameter_bound_status),
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
    date_obs = str(header.get("DATE-OBS", "")).strip()
    required = np.asarray(
        (ra_deg, dec_deg, longitude_deg, latitude_deg, altitude_m),
        dtype=float,
    )
    if not date_obs or not np.all(np.isfinite(required)):
        return np.nan

    try:
        location = EarthLocation.from_geodetic(
            longitude_deg * u.deg,
            latitude_deg * u.deg,
            altitude_m * u.m,
        )
        target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
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
) -> Mapping[str, object] | None:
    path = Path(input_path)
    chosen_format = infer_spectrum_format(path, input_format)
    if chosen_format not in {"fits", "fit", "fz"}:
        return None
    try:
        with fits.open(path) as hdul:
            header = dict(hdul[0].header)
            if len(hdul) > 1:
                for key, value in hdul[1].header.items():
                    header.setdefault(key, value)
            return header
    except Exception:
        return None


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
