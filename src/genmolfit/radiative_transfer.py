from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .atmosphere import AtmosphereProfile
from .components import (
    AbsorptionComponent,
    H2OContinuumAbsorption,
    HitranLineAbsorption,
    N2ContinuumAbsorption,
    O2ContinuumAbsorption,
    RayleighScatteringAbsorption,
    combine_optical_depth_components,
)
from .continuum import MTCKDH2OContinuum
from .linelist import LineList
from .model import transmission_from_basis
from .partition import PartitionTable
from .physics import (
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_AVMASS_AMU,
    LBLRTM_DEFAULT_SAMPLE,
    LBLRTM_VOIGT_DOMAIN_HWF3,
)


@dataclass(frozen=True)
class PhysicalModelConfig:
    species_scales: Mapping[str, float] = field(default_factory=dict)
    lsf_sigma_pixels: float = 0.0
    lsf_box_width_pixels: float = 0.0
    lsf_lorentz_fwhm_pixels: float = 0.0
    lsf_variable_width: bool = False
    lsf_reference_wavelength_micron: float | None = None
    lsf_kernel_width_fwhm: float = 3.0
    lsf_molecfit_voigt: bool = False
    chunk_size: int = 0
    partition_exponent: float = 1.5
    partition_table: PartitionTable | None = None
    h2o_continuum: MTCKDH2OContinuum | None = None
    h2o_continuum_foreign_closure: bool = False
    line_cutoff_cm: float | None = None
    subtract_cutoff_profile: bool = False
    line_taper_cm: float = 0.0
    line_wing_mode: str = "full"
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3
    rayleigh: bool = False
    rayleigh_xrayl: float = 1.0
    n2_continuum: bool = False
    n2_continuum_xn2cn: float = 1.0
    o2_continuum: bool = False
    o2_continuum_xo2cn: float = 1.0
    components: tuple[AbsorptionComponent, ...] | None = None


def hitran_optical_depth_basis(
    wavelength_micron: np.ndarray,
    line_list: LineList,
    atmosphere: AtmosphereProfile,
    *,
    species: tuple[str, ...] | None = None,
    chunk_size: int = 0,
    partition_exponent: float = 1.5,
    partition_table: PartitionTable | None = None,
    h2o_continuum: MTCKDH2OContinuum | None = None,
    h2o_continuum_foreign_closure: bool = False,
    line_cutoff_cm: float | None = None,
    subtract_cutoff_profile: bool = False,
    line_taper_cm: float = 0.0,
    line_wing_mode: str = "full",
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
) -> tuple[tuple[str, ...], np.ndarray]:
    """Calculate optical-depth basis vectors from current physical components.

    This compatibility function now builds a component set internally:
    HITRAN line absorption plus, optionally, MT_CKD H2O continuum.
    """

    components = physical_components_from_options(
        line_list=line_list,
        chunk_size=chunk_size,
        partition_exponent=partition_exponent,
        partition_table=partition_table,
        h2o_continuum=h2o_continuum,
        h2o_continuum_foreign_closure=h2o_continuum_foreign_closure,
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
    )
    return combine_optical_depth_components(
        wavelength_micron,
        atmosphere,
        components,
        species=species,
    )


def physical_optical_depth_basis(
    wavelength_micron: np.ndarray,
    line_list: LineList,
    atmosphere: AtmosphereProfile,
    config: PhysicalModelConfig | None = None,
    *,
    species: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Calculate optical-depth basis using configured absorption components."""

    config = PhysicalModelConfig() if config is None else config
    components = config.components
    if components is None:
        components = physical_components_from_options(
            line_list=line_list,
            chunk_size=config.chunk_size,
            partition_exponent=config.partition_exponent,
            partition_table=config.partition_table,
            h2o_continuum=config.h2o_continuum,
            h2o_continuum_foreign_closure=config.h2o_continuum_foreign_closure,
            line_cutoff_cm=config.line_cutoff_cm,
            subtract_cutoff_profile=config.subtract_cutoff_profile,
            line_taper_cm=config.line_taper_cm,
            line_wing_mode=config.line_wing_mode,
            lblrtm_sample=config.lblrtm_sample,
            lblrtm_alfal0=config.lblrtm_alfal0,
            lblrtm_avmass_amu=config.lblrtm_avmass_amu,
            lblrtm_hwf3=config.lblrtm_hwf3,
            rayleigh=config.rayleigh,
            rayleigh_xrayl=config.rayleigh_xrayl,
            n2_continuum=config.n2_continuum,
            n2_continuum_xn2cn=config.n2_continuum_xn2cn,
            o2_continuum=config.o2_continuum,
            o2_continuum_xo2cn=config.o2_continuum_xo2cn,
        )
    return combine_optical_depth_components(
        wavelength_micron,
        atmosphere,
        components,
        species=species,
    )


def physical_components_from_options(
    *,
    line_list: LineList,
    chunk_size: int = 0,
    partition_exponent: float = 1.5,
    partition_table: PartitionTable | None = None,
    h2o_continuum: MTCKDH2OContinuum | None = None,
    h2o_continuum_foreign_closure: bool = False,
    line_cutoff_cm: float | None = None,
    subtract_cutoff_profile: bool = False,
    line_taper_cm: float = 0.0,
    line_wing_mode: str = "full",
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
) -> tuple[AbsorptionComponent, ...]:
    """Build the default pure-Python component set."""

    components: list[AbsorptionComponent] = [
        HitranLineAbsorption(
            line_list=line_list,
            chunk_size=chunk_size,
            partition_exponent=partition_exponent,
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
    ]
    if h2o_continuum is not None:
        components.append(
            H2OContinuumAbsorption(
                continuum=h2o_continuum,
                use_foreign_closure=h2o_continuum_foreign_closure,
            )
        )
    if rayleigh:
        components.append(RayleighScatteringAbsorption(xrayl=rayleigh_xrayl))
    if n2_continuum:
        components.append(N2ContinuumAbsorption(xn2cn=n2_continuum_xn2cn))
    if o2_continuum:
        components.append(O2ContinuumAbsorption(xo2cn=o2_continuum_xo2cn))
    return tuple(components)


def physical_transmission_model(
    wavelength_micron: np.ndarray,
    line_list: LineList,
    atmosphere: AtmosphereProfile,
    config: PhysicalModelConfig | None = None,
) -> np.ndarray:
    config = PhysicalModelConfig() if config is None else config
    species_names, basis = physical_optical_depth_basis(
        wavelength_micron,
        line_list,
        atmosphere,
        config,
    )
    return transmission_from_basis(
        species_names,
        basis,
        species_scales=config.species_scales,
        airmass=1.0,
        lsf_sigma_pixels=config.lsf_sigma_pixels,
        lsf_box_width_pixels=config.lsf_box_width_pixels,
        lsf_lorentz_fwhm_pixels=config.lsf_lorentz_fwhm_pixels,
        wavelength_micron=wavelength_micron,
        lsf_variable_width=config.lsf_variable_width,
        lsf_reference_wavelength_micron=config.lsf_reference_wavelength_micron,
        lsf_kernel_width_fwhm=config.lsf_kernel_width_fwhm,
        lsf_molecfit_voigt=config.lsf_molecfit_voigt,
    )
