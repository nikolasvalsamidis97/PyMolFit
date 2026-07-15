from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .atmosphere import AtmosphereProfile, CM_PER_M, PA_PER_ATM
from .continuum import (
    HitranCIATable,
    LBLRTMCO2Continuum,
    LBLRTMN2FundamentalContinuum,
    LBLRTMN2OvertoneContinuum,
    LBLRTMO2Continuum,
    MTCKDH2OContinuum,
    TabulatedContinuum,
    _lblrtm_n2_rot_grid,
    _prepare_cubic_regular_interpolation,
    _regular_grid_has_target_support,
    lblrtm_n2_rototranslational_optical_depth,
    lblrtm_rayleigh_optical_depth,
)
from .linelist import LBLRTM_BROADENER_SPECIES, LineList
from .partition import PartitionTable
from .physics import (
    BOLTZMANN_J_PER_K,
    LBLRTM_DEFAULT_ALFAL0,
    LBLRTM_DEFAULT_AVMASS_AMU,
    LBLRTM_DEFAULT_DPTFAC,
    LBLRTM_DEFAULT_DPTMIN,
    LBLRTM_DEFAULT_SAMPLE,
    LBLRTM_F4_BOUND_CM,
    LBLRTM_F4_GRID_RATIO,
    LBLRTM_VOIGT_DOMAIN_HWF3,
    LOSCHMIDT_CM3,
    doppler_sigma_wavenumber,
    lblrtm_dynamic_line_cutoff_cm,
    lblrtm_layer_wavenumber_spacings_cm,
    lblrtm_f4_coefficients,
    lblrtm_panel_accumulate_wavenumber,
    lblrtm_panel_interpolate_f4_wavenumber,
    lblrtm_panel_voigt_profile_wavenumber,
    lblrtm_tabulated_voigt_profile_offset,
    lblrtm_temperature_scaling_lower_energy,
    lblrtm_voigt_hwhm,
    lblrtm_radiation_term,
    line_strength_temperature,
    lorentz_hwhm_wavenumber,
    pressure_shift_wavenumber,
    voigt_profile_offset,
    voigt_profile_wavenumber,
    wavelength_micron_to_wavenumber_cm,
)

DEFAULT_LBLRTM_LINE_CUTOFF_CM = 25.0
DEFAULT_LINE_CHUNK_SIZE = 512
DEFAULT_PANEL_LINE_CHUNK_SIZE = 16_384
_USE_SPARSE_FINITE_VOIGT = True
_CACHE_LBLRTM_SCREENED_LINE_STATE = True
LINE_WING_MODES = frozenset(
    (
        "full",
        "hard_cutoff",
        "subtracted_cutoff",
        "tapered_cutoff",
        "lblrtm_subtracted",
        "lblrtm_dynamic",
        "lblrtm_table",
        "lblrtm_panel",
    )
)


@dataclass(frozen=True)
class _LBLRTMPreparedLayerChunk:
    layer_index: int
    centers: np.ndarray
    sigma: np.ndarray
    gamma: np.ndarray
    line_scale: np.ndarray
    group_index: np.ndarray
    profile_coupling: np.ndarray
    effective_hwhm_cm: np.ndarray
    maximum_cutoff_cm: float | None
    dptmin: float


@dataclass
class _LBLRTMF4State:
    grid_cm: np.ndarray
    total_by_layer: tuple[np.ndarray, ...]
    grouped_by_layer: tuple[np.ndarray, ...]
    prepared_chunks: tuple[_LBLRTMPreparedLayerChunk, ...] = ()


class AbsorptionComponent(Protocol):
    """A radiative-transfer component that contributes optical depth."""

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Return species names and optical-depth basis rows."""


@dataclass(frozen=True)
class HitranLineAbsorption:
    """Line-by-line HITRAN absorption component."""

    line_list: LineList
    species: tuple[str, ...] | None = None
    chunk_size: int = 0
    partition_exponent: float = 1.5
    partition_table: PartitionTable | None = None
    line_cutoff_cm: float | None = None
    subtract_cutoff_profile: bool = False
    line_taper_cm: float = 0.0
    line_wing_mode: str = "full"
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        selected_species = _combine_species_filters(self.species, species)
        return hitran_line_optical_depth_basis(
            wavelength_micron,
            self.line_list,
            atmosphere,
            species=selected_species,
            chunk_size=self.chunk_size,
            partition_exponent=self.partition_exponent,
            partition_table=self.partition_table,
            line_cutoff_cm=self.line_cutoff_cm,
            subtract_cutoff_profile=self.subtract_cutoff_profile,
            line_taper_cm=self.line_taper_cm,
            line_wing_mode=self.line_wing_mode,
            lblrtm_sample=self.lblrtm_sample,
            lblrtm_alfal0=self.lblrtm_alfal0,
            lblrtm_avmass_amu=self.lblrtm_avmass_amu,
            lblrtm_hwf3=self.lblrtm_hwf3,
        )


@dataclass(frozen=True)
class H2OContinuumAbsorption:
    """MT_CKD H2O continuum absorption component."""

    continuum: MTCKDH2OContinuum
    use_foreign_closure: bool = False
    include_radiation_term: bool = True

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if species is not None and "H2O" not in species:
            return _empty_basis(wavelength_micron)
        if not _atmosphere_has_species(atmosphere, "H2O"):
            return _empty_basis(wavelength_micron)

        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        tau = self.continuum.optical_depth(
            wavenumber_grid,
            atmosphere,
            include_radiation_term=self.include_radiation_term,
            use_foreign_closure=self.use_foreign_closure,
        )
        return ("H2O",), tau[None, :]


@dataclass(frozen=True)
class TabulatedContinuumAbsorption:
    """Continuum absorption from an external coefficient table.

    ``coefficient_kind`` defines how table coefficients are converted into
    optical depth:

    - ``cross_section_cm2``: tau = coefficient * absorber column
    - ``density_scaled_cross_section_cm2_per_amagat``:
      tau = coefficient * absorber column * air density in amagats
    """

    species_name: str
    continuum: TabulatedContinuum
    coefficient_kind: str = "density_scaled_cross_section_cm2_per_amagat"
    basis_name: str | None = None
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        name = self.basis_name or self.species_name
        if species is not None and name not in species and self.species_name not in species:
            return _empty_basis(wavelength_micron)

        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        co2_interpolation_plan = (
            _prepare_cubic_regular_interpolation(
                self.continuum.wavenumber_cm,
                wavenumber_grid,
            )
            if isinstance(self.continuum, LBLRTMCO2Continuum)
            else None
        )
        tau = np.zeros(wavenumber_grid.shape, dtype=float)
        for layer in atmosphere.layers:
            column = layer.column_density_cm2(self.species_name)
            if column <= 0:
                continue
            if co2_interpolation_plan is None:
                coefficient = self.continuum.coefficient_at(
                    wavenumber_grid,
                    layer.temperature_k,
                )
            else:
                coefficient = self.continuum.coefficient_at(
                    wavenumber_grid,
                    layer.temperature_k,
                    _interpolation_plan=co2_interpolation_plan,
                )
            if self.coefficient_kind == "cross_section_cm2":
                tau += coefficient * column
            elif self.coefficient_kind == "density_scaled_cross_section_cm2_per_amagat":
                tau += coefficient * column * _air_amagat(layer)
            else:
                raise ValueError(f"unknown continuum coefficient_kind: {self.coefficient_kind!r}")
        return (name,), (self.scale * tau)[None, :]


@dataclass(frozen=True)
class RayleighScatteringAbsorption:
    """LBLRTM ``contnm.f90`` Rayleigh scattering component."""

    basis_name: str = "Rayleigh"
    xrayl: float = 1.0
    jrad: int = 1
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if species is not None and self.basis_name not in species and "AIR" not in species:
            return _empty_basis(wavelength_micron)

        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        tau = np.zeros(wavenumber_grid.shape, dtype=float)
        for layer in atmosphere.layers:
            tau += lblrtm_rayleigh_optical_depth(
                wavenumber_grid,
                _air_column_density_cm2(layer),
                xrayl=self.xrayl,
                jrad=self.jrad,
                temperature_k=layer.temperature_k,
            )
        return (self.basis_name,), (self.scale * tau)[None, :]


@dataclass(frozen=True)
class N2RototranslationalContinuumAbsorption:
    """LBLRTM `contnm.f90` N2 pure-rotation continuum component."""

    basis_name: str = "N2_rototranslational"
    xn2cn: float = 1.0
    jrad: int = 1
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if species is not None and self.basis_name not in species and "N2" not in species:
            return _empty_basis(wavelength_micron)

        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        tau = np.zeros(wavenumber_grid.shape, dtype=float)
        for layer in atmosphere.layers:
            n2_vmr = _species_vmr(layer, "N2")
            n2_column = _air_column_density_cm2(layer) * n2_vmr
            if n2_column <= 0:
                continue
            tau += lblrtm_n2_rototranslational_optical_depth(
                wavenumber_grid,
                n2_column_cm2=n2_column,
                air_amagat=_air_amagat(layer),
                temperature_k=layer.temperature_k,
                n2_vmr=n2_vmr,
                o2_vmr=_species_vmr(layer, "O2"),
                h2o_vmr=_species_vmr(layer, "H2O"),
                xn2cn=self.xn2cn,
                jrad=self.jrad,
            )
        return (self.basis_name,), (self.scale * tau)[None, :]


@dataclass(frozen=True)
class N2ContinuumAbsorption:
    """LBLRTM N2 pure-rotation, fundamental, and first-overtone branches."""

    fundamental: LBLRTMN2FundamentalContinuum | None = None
    overtone: LBLRTMN2OvertoneContinuum | None = None
    basis_name: str = "N2_continuum"
    xn2cn: float = 1.0
    jrad: int = 1
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.fundamental is None:
            object.__setattr__(self, "fundamental", LBLRTMN2FundamentalContinuum.from_package_data())
        if self.overtone is None:
            object.__setattr__(self, "overtone", LBLRTMN2OvertoneContinuum.from_package_data())

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if species is not None and self.basis_name not in species and "N2" not in species:
            return _empty_basis(wavelength_micron)
        if self.xn2cn < 0:
            raise ValueError("xn2cn must be non-negative")
        if self.jrad not in (0, 1):
            raise ValueError("jrad must be 0 or 1")

        fundamental = self.fundamental
        overtone = self.overtone
        if fundamental is None or overtone is None:
            raise RuntimeError("N2 continuum tables were not initialized")
        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        rot_grid = _lblrtm_n2_rot_grid()
        rot_plan = (
            _prepare_cubic_regular_interpolation(rot_grid, wavenumber_grid)
            if _regular_grid_has_target_support(rot_grid, wavenumber_grid)
            else None
        )
        fundamental_plan = (
            _prepare_cubic_regular_interpolation(fundamental.wavenumber_cm, wavenumber_grid)
            if _regular_grid_has_target_support(fundamental.wavenumber_cm, wavenumber_grid)
            else None
        )
        overtone_plan = (
            _prepare_cubic_regular_interpolation(overtone.wavenumber_cm, wavenumber_grid)
            if _regular_grid_has_target_support(overtone.wavenumber_cm, wavenumber_grid)
            else None
        )
        tau = np.zeros(wavenumber_grid.shape, dtype=float)
        for layer in atmosphere.layers:
            h2o_vmr = _species_vmr(layer, "H2O")
            o2_vmr = _species_vmr(layer, "O2")
            # LBLRTM contnm defines N2 as the broadening-air remainder after
            # H2O and O2, irrespective of explicitly modelled trace gases.
            n2_vmr = max(0.0, 1.0 - h2o_vmr - o2_vmr)
            n2_column = _air_column_density_cm2(layer) * n2_vmr
            if n2_column <= 0:
                continue
            common = {
                "n2_column_cm2": n2_column,
                "air_amagat": _air_amagat(layer),
                "temperature_k": layer.temperature_k,
                "n2_vmr": n2_vmr,
                "o2_vmr": o2_vmr,
                "h2o_vmr": h2o_vmr,
                "xn2cn": self.xn2cn,
                "jrad": self.jrad,
            }
            if rot_plan is not None:
                tau += lblrtm_n2_rototranslational_optical_depth(
                    wavenumber_grid,
                    **common,
                    _interpolation_plan=rot_plan,
                )
            if fundamental_plan is not None:
                tau += fundamental.optical_depth_layer(
                    wavenumber_grid,
                    **common,
                    _interpolation_plan=fundamental_plan,
                )
            if overtone_plan is not None:
                tau += overtone.optical_depth_layer(
                    wavenumber_grid,
                    **common,
                    _interpolation_plan=overtone_plan,
                )
        return (self.basis_name,), (self.scale * tau)[None, :]


@dataclass(frozen=True)
class O2ContinuumAbsorption:
    """LBLRTM ground-based O2 continuum component."""

    continuum: LBLRTMO2Continuum | None = None
    basis_name: str = "O2_continuum"
    xo2cn: float = 1.0
    jrad: int = 1
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if species is not None and self.basis_name not in species and "O2" not in species:
            return _empty_basis(wavelength_micron)
        if self.xo2cn < 0:
            raise ValueError("xo2cn must be non-negative")
        if self.jrad not in (0, 1):
            raise ValueError("jrad must be 0 or 1")

        continuum = self.continuum or LBLRTMO2Continuum.from_package_data()
        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        tau = np.zeros(wavenumber_grid.shape, dtype=float)
        for layer in atmosphere.layers:
            air_column = _air_column_density_cm2(layer)
            o2_vmr = _species_vmr(layer, "O2")
            o2_column = air_column * o2_vmr
            if o2_column <= 0:
                continue
            h2o_vmr = _species_vmr(layer, "H2O")
            n2_vmr = max(0.0, 1.0 - h2o_vmr - o2_vmr)
            tau += continuum.optical_depth_layer(
                wavenumber_grid,
                o2_column_cm2=o2_column,
                air_column_cm2=air_column,
                air_amagat=_air_amagat(layer),
                pressure_mbar=layer.pressure_atm * 1013.25,
                temperature_k=layer.temperature_k,
                n2_vmr=n2_vmr,
                o2_vmr=o2_vmr,
                h2o_vmr=h2o_vmr,
                xo2cn=self.xo2cn,
                jrad=self.jrad,
            )
        return (self.basis_name,), (self.scale * tau)[None, :]


@dataclass(frozen=True)
class CO2ContinuumAbsorption:
    """CO2 continuum component loaded from an external coefficient table."""

    continuum: TabulatedContinuum
    coefficient_kind: str = "density_scaled_cross_section_cm2_per_amagat"
    basis_name: str = "CO2"
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        return TabulatedContinuumAbsorption(
            species_name="CO2",
            continuum=self.continuum,
            coefficient_kind=self.coefficient_kind,
            basis_name=self.basis_name,
            scale=self.scale,
        ).optical_depth_basis(wavelength_micron, atmosphere, species=species)


@dataclass(frozen=True)
class PairCIAAbsorption:
    """Collision-induced absorption from HITRAN CIA coefficients."""

    cia_table: HitranCIATable
    pair_species: tuple[str, str] | None = None
    basis_name: str | None = None
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        pair = self.pair_species or self.cia_table.pair
        if pair is None:
            return _empty_basis(wavelength_micron)
        name = self.basis_name or f"{pair[0]}-{pair[1]}_CIA"
        if species is not None and name not in species and not any(part in species for part in pair):
            return _empty_basis(wavelength_micron)

        wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
        active_layers = []
        layer_scales = []
        for layer in atmosphere.layers:
            density_a = _species_number_density_cm3(layer, pair[0])
            density_b = _species_number_density_cm3(layer, pair[1])
            if density_a <= 0 or density_b <= 0:
                continue
            active_layers.append(layer)
            layer_scales.append(density_a * density_b * layer.path_length_m * CM_PER_M)
        if not active_layers:
            return (name,), np.zeros((1, wavenumber_grid.size), dtype=float)
        coefficients = self.cia_table.coefficients_at(
            wavenumber_grid,
            np.array([layer.temperature_k for layer in active_layers], dtype=float),
        )
        tau = np.asarray(layer_scales, dtype=float) @ coefficients
        return (name,), (self.scale * tau)[None, :]


@dataclass(frozen=True)
class O2CIAAbsorption:
    """O2 collision-induced absorption component loaded from HITRAN CIA."""

    cia_table: HitranCIATable
    pair_species: tuple[str, str] | None = None
    basis_name: str = "O2_CIA"
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        return PairCIAAbsorption(
            cia_table=self.cia_table,
            pair_species=self.pair_species,
            basis_name=self.basis_name,
            scale=self.scale,
        ).optical_depth_basis(wavelength_micron, atmosphere, species=species)


@dataclass(frozen=True)
class N2CIAAbsorption:
    """N2 collision-induced absorption component loaded from HITRAN CIA."""

    cia_table: HitranCIATable
    pair_species: tuple[str, str] | None = None
    basis_name: str = "N2_CIA"
    scale: float = 1.0

    def optical_depth_basis(
        self,
        wavelength_micron: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        species: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        return PairCIAAbsorption(
            cia_table=self.cia_table,
            pair_species=self.pair_species,
            basis_name=self.basis_name,
            scale=self.scale,
        ).optical_depth_basis(wavelength_micron, atmosphere, species=species)


def combine_optical_depth_components(
    wavelength_micron: np.ndarray,
    atmosphere: AtmosphereProfile,
    components: tuple[AbsorptionComponent, ...],
    *,
    species: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Sum optical-depth basis vectors from multiple components.

    Components that report the same species are added into the same basis row,
    so H2O line absorption and H2O continuum are fitted by one H2O scale factor.
    """

    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    species_index: dict[str, int] = {}
    basis_rows: list[np.ndarray] = []
    requested = None if species is None else set(species)

    for component in components:
        names, local_basis = component.optical_depth_basis(
            wavelength_micron,
            atmosphere,
            species=species,
        )
        local_basis = np.asarray(local_basis, dtype=float)
        if local_basis.shape != (len(names), wavelength_micron.size):
            raise ValueError("component optical-depth basis has an invalid shape")

        for name, row in zip(names, local_basis, strict=True):
            if requested is not None and name not in requested:
                continue
            if name not in species_index:
                species_index[name] = len(basis_rows)
                basis_rows.append(np.zeros(wavelength_micron.size, dtype=float))
            basis_rows[species_index[name]] += row

    species_names = tuple(species_index)
    if not basis_rows:
        return species_names, np.zeros((0, wavelength_micron.size), dtype=float)
    return species_names, np.vstack(basis_rows)


def hitran_line_optical_depth_basis(
    wavelength_micron: np.ndarray,
    line_list: LineList,
    atmosphere: AtmosphereProfile,
    *,
    species: tuple[str, ...] | None = None,
    chunk_size: int = 0,
    partition_exponent: float = 1.5,
    partition_table: PartitionTable | None = None,
    line_cutoff_cm: float | None = None,
    subtract_cutoff_profile: bool = False,
    line_taper_cm: float = 0.0,
    line_wing_mode: str = "full",
    lblrtm_sample: float = LBLRTM_DEFAULT_SAMPLE,
    lblrtm_alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    lblrtm_avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
    lblrtm_hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Calculate physical optical-depth basis vectors from HITRAN fields."""

    if chunk_size < 0:
        raise ValueError("chunk_size must be non-negative")
    if chunk_size == 0:
        chunk_size = (
            DEFAULT_PANEL_LINE_CHUNK_SIZE
            if str(line_wing_mode).strip().lower().replace("-", "_") == "lblrtm_panel"
            else DEFAULT_LINE_CHUNK_SIZE
        )

    completed_r4 = None
    if str(line_wing_mode).strip().lower() in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}:
        _, _, completed_r4 = _hitran_line_optical_depth_basis_impl(
            wavelength_micron,
            line_list,
            atmosphere,
            species=species,
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
            f4_screening_pass=True,
            completed_r4=None,
        )

    names, basis, _ = _hitran_line_optical_depth_basis_impl(
        wavelength_micron,
        line_list,
        atmosphere,
        species=species,
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
        f4_screening_pass=False,
        completed_r4=completed_r4,
    )
    return names, basis


def _hitran_line_optical_depth_basis_impl(
    wavelength_micron: np.ndarray,
    line_list: LineList,
    atmosphere: AtmosphereProfile,
    *,
    species: tuple[str, ...] | None,
    chunk_size: int,
    partition_exponent: float,
    partition_table: PartitionTable | None,
    line_cutoff_cm: float | None,
    subtract_cutoff_profile: bool,
    line_taper_cm: float,
    line_wing_mode: str,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_avmass_amu: float,
    lblrtm_hwf3: float,
    f4_screening_pass: bool,
    completed_r4: _LBLRTMF4State | None,
) -> tuple[tuple[str, ...], np.ndarray, _LBLRTMF4State | None]:
    """Internal two-pass implementation of the HITRAN/LBLRTM line path."""

    if not line_list.has_hitran_parameters:
        raise ValueError("line_list does not contain the HITRAN fields required for physical modelling")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    line_wing_mode, line_cutoff_cm, subtract_cutoff_profile, line_taper_cm = _line_wing_settings(
        line_wing_mode=line_wing_mode,
        line_cutoff_cm=line_cutoff_cm,
        subtract_cutoff_profile=subtract_cutoff_profile,
        line_taper_cm=line_taper_cm,
    )
    _validate_lblrtm_dynamic_controls(
        sample=lblrtm_sample,
        alfal0=lblrtm_alfal0,
        avmass_amu=lblrtm_avmass_amu,
        hwf3=lblrtm_hwf3,
    )

    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    wavenumber_grid = wavelength_micron_to_wavenumber_cm(wavelength_micron)
    dynamic_wavenumber_spacing_cm = (
        _wavenumber_spacing_cm(wavenumber_grid)
        if line_wing_mode in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}
        else None
    )
    representative_wavenumber_cm = (
        0.5 * (float(np.nanmin(wavenumber_grid)) + float(np.nanmax(wavenumber_grid)))
        if wavenumber_grid.size
        else np.nan
    )
    species_names = line_list.species_names if species is None else tuple(species)
    species_index = {name: idx for idx, name in enumerate(species_names)}
    line_species = np.asarray(line_list.species)
    all_line_species_names = tuple(dict.fromkeys(line_species.tolist()))
    line_species_code_map = {name: index for index, name in enumerate(all_line_species_names)}
    line_species_codes = np.array([line_species_code_map[name] for name in line_species], dtype=int)
    line_basis_index = np.fromiter(
        (species_index.get(str(name), -1) for name in line_species),
        dtype=int,
        count=line_species.size,
    )
    basis = np.zeros((len(species_names), wavelength_micron.size), dtype=float)
    layer_cache = tuple(
        (
            layer,
            layer.pressure_atm,
            layer.temperature_k,
            layer.pressure_atm * PA_PER_ATM * layer.path_length_m / (BOLTZMANN_J_PER_K * layer.temperature_k * CM_PER_M**2),
            np.array([layer.mixing_ratios.get(name, 0.0) for name in all_line_species_names], dtype=float),
        )
        for layer in atmosphere.layers
    )
    layer_wavenumber_spacings_cm = None
    if dynamic_wavenumber_spacing_cm is not None and layer_cache:
        layer_wavenumber_spacings_cm = lblrtm_layer_wavenumber_spacings_cm(
            representative_wavenumber_cm,
            np.asarray([item[1] for item in layer_cache], dtype=float),
            np.asarray([item[2] for item in layer_cache], dtype=float),
            h2o_fraction=np.asarray(
                [item[0].mixing_ratios.get("H2O", 0.0) for item in layer_cache],
                dtype=float,
            ),
            sample=lblrtm_sample,
            alfal0=lblrtm_alfal0,
            avmass_amu=lblrtm_avmass_amu,
        )

    wavenumber = np.asarray(line_list.wavenumber, dtype=float)
    screening_r4_grid = None
    screening_r4_by_layer = None
    grouped_r4_by_layer = None
    prepared_chunks: list[_LBLRTMPreparedLayerChunk] = []
    if dynamic_wavenumber_spacing_cm is not None and wavenumber_grid.size:
        r4_spacing = LBLRTM_F4_GRID_RATIO * dynamic_wavenumber_spacing_cm
        r4_start = float(np.nanmin(wavenumber_grid)) - LBLRTM_F4_BOUND_CM - 2.0 * r4_spacing
        r4_stop = float(np.nanmax(wavenumber_grid)) + LBLRTM_F4_BOUND_CM + 2.5 * r4_spacing
        screening_r4_grid = np.arange(r4_start, r4_stop + 0.5 * r4_spacing, r4_spacing)
        if completed_r4 is None:
            screening_r4_by_layer = [
                np.zeros(screening_r4_grid.size, dtype=float) for _ in atmosphere.layers
            ]
            grouped_r4_by_layer = [
                np.zeros((len(species_names), screening_r4_grid.size), dtype=float)
                for _ in atmosphere.layers
            ]
        else:
            if not np.array_equal(screening_r4_grid, completed_r4.grid_cm):
                raise ValueError("completed F4 grid does not match the current spectral grid")
            screening_r4_by_layer = list(completed_r4.total_by_layer)
            grouped_r4_by_layer = list(completed_r4.grouped_by_layer)
    if (
        not f4_screening_pass
        and completed_r4 is not None
        and completed_r4.prepared_chunks
        and line_wing_mode == "lblrtm_panel"
        and line_cutoff_cm is None
    ):
        _accumulate_prepared_lblrtm_panel_basis(
            basis,
            wavenumber_grid=wavenumber_grid,
            completed_r4=completed_r4,
        )
        grouped_r4 = np.sum(np.stack(completed_r4.grouped_by_layer), axis=0)
        basis += lblrtm_panel_interpolate_f4_wavenumber(
            wavenumber_grid,
            completed_r4.grid_cm,
            grouped_r4,
        )
        return species_names, basis, completed_r4
    strength = np.asarray(line_list.strength, dtype=float)
    air_width = np.asarray(line_list.air_width, dtype=float)
    self_width = np.asarray(line_list.self_width, dtype=float)
    lower_state_energy = np.asarray(line_list.lower_state_energy, dtype=float)
    temperature_exponent = np.asarray(line_list.temperature_exponent, dtype=float)
    pressure_shift = np.asarray(line_list.pressure_shift, dtype=float)
    molecular_mass_amu = np.asarray(line_list.molecular_mass_amu, dtype=float)
    mol_id = None if line_list.mol_id is None else np.asarray(line_list.mol_id, dtype=int)
    iso_id = None if line_list.iso_id is None else np.asarray(line_list.iso_id, dtype=int)
    isotopologue_abundance_scale = (
        None
        if line_list.isotopologue_abundance_scale is None
        else np.asarray(line_list.isotopologue_abundance_scale, dtype=float)
    )
    broadener_flags = None if line_list.broadener_flags is None else np.asarray(line_list.broadener_flags, dtype=int)
    broadener_widths = (
        None if line_list.broadener_widths is None else np.asarray(line_list.broadener_widths, dtype=float)
    )
    broadener_temperature_exponents = (
        None
        if line_list.broadener_temperature_exponents is None
        else np.asarray(line_list.broadener_temperature_exponents, dtype=float)
    )
    broadener_pressure_shifts = (
        None
        if line_list.broadener_pressure_shifts is None
        else np.asarray(line_list.broadener_pressure_shifts, dtype=float)
    )
    line_flags = None if line_list.line_flags is None else np.asarray(line_list.line_flags, dtype=int)
    line_coupling_a = (
        None if line_list.line_coupling_a is None else np.asarray(line_list.line_coupling_a, dtype=float)
    )
    line_coupling_b = (
        None if line_list.line_coupling_b is None else np.asarray(line_list.line_coupling_b, dtype=float)
    )
    if wavenumber.size > 1 and not np.all(wavenumber[:-1] <= wavenumber[1:]):
        processing_order: slice | np.ndarray = np.argsort(wavenumber)
    else:
        processing_order = slice(None)
    sparse_grid_order = np.argsort(wavenumber_grid)
    sparse_sorted_grid = wavenumber_grid[sparse_grid_order]
    sparse_grid_available = bool(
        sparse_sorted_grid.size == wavenumber_grid.size
        and np.all(np.isfinite(sparse_sorted_grid))
        and (sparse_sorted_grid.size < 2 or np.all(np.diff(sparse_sorted_grid) > 0))
    )

    for start in range(0, wavenumber.size, chunk_size):
        stop = min(start + chunk_size, wavenumber.size)
        line_selector = (
            slice(start, stop)
            if isinstance(processing_order, slice)
            else processing_order[start:stop]
        )
        chunk_wavenumber = wavenumber[line_selector]
        chunk_strength = strength[line_selector]
        chunk_mass = molecular_mass_amu[line_selector]
        local_basis_index = line_basis_index[line_selector]
        active = local_basis_index >= 0
        active &= np.isfinite(chunk_mass)
        active &= np.isfinite(chunk_wavenumber)
        active &= chunk_strength > 0
        if not np.any(active):
            continue

        local_wavenumber = chunk_wavenumber[active]
        local_strength = chunk_strength[active]
        local_air_width = air_width[line_selector][active]
        local_self_width = self_width[line_selector][active]
        local_lower_energy = lower_state_energy[line_selector][active]
        local_temp_exp = temperature_exponent[line_selector][active]
        local_pressure_shift = pressure_shift[line_selector][active]
        local_mass = chunk_mass[active]
        local_species_codes = line_species_codes[line_selector][active]
        local_basis_index = local_basis_index[active]
        local_mol_id = None if mol_id is None else mol_id[line_selector][active]
        local_iso_id = None if iso_id is None else iso_id[line_selector][active]
        local_abundance_scale = (
            1.0
            if isotopologue_abundance_scale is None
            else isotopologue_abundance_scale[line_selector][active]
        )
        local_broadener_flags = None if broadener_flags is None else broadener_flags[line_selector][active]
        local_broadener_widths = None if broadener_widths is None else broadener_widths[line_selector][active]
        local_broadener_temperature_exponents = (
            None
            if broadener_temperature_exponents is None
            else broadener_temperature_exponents[line_selector][active]
        )
        local_broadener_pressure_shifts = (
            None if broadener_pressure_shifts is None else broadener_pressure_shifts[line_selector][active]
        )
        local_line_flags = None if line_flags is None else line_flags[line_selector][active]
        local_line_coupling_a = None if line_coupling_a is None else line_coupling_a[line_selector][active]
        local_line_coupling_b = None if line_coupling_b is None else line_coupling_b[line_selector][active]
        local_air_width = _lblrtm_self_mixture_corrected_air_width(
            local_air_width,
            local_self_width,
            local_mol_id,
        )
        local_pressure_shift = _lblrtm_self_mixture_corrected_pressure_shift(
            local_pressure_shift,
            local_broadener_flags,
            local_broadener_pressure_shifts,
            local_mol_id,
        )
        local_broadener_shift_delta = None
        if local_broadener_flags is not None and local_broadener_pressure_shifts is not None:
            local_broadener_shift_delta = np.where(
                local_broadener_flags > 0,
                local_broadener_pressure_shifts - local_pressure_shift[:, None],
                0.0,
            )
        species_row_groups = tuple(
            (int(row_index), local_basis_index == row_index)
            for row_index in np.unique(local_basis_index)
        )
        effective_lower_energy, unknown_lower_energy = lblrtm_temperature_scaling_lower_energy(
            local_lower_energy
        )
        q_ref = None
        if partition_table is not None and local_mol_id is not None and local_iso_id is not None:
            q_ref = partition_table.value(local_mol_id, local_iso_id, line_list.reference_temperature)

        for layer_index, (
            layer,
            pressure_atm,
            temperature_k,
            column_factor_cm2,
            layer_vmr_by_code,
        ) in enumerate(layer_cache):
            absorber_fraction = layer_vmr_by_code[local_species_codes]
            columns = column_factor_cm2 * absorber_fraction
            if not np.any(columns > 0):
                continue

            shifted_centers = local_wavenumber + pressure_shift_wavenumber(
                local_pressure_shift,
                pressure_atm,
                temperature_k,
                reference_temperature_k=line_list.reference_temperature,
                convention=line_list.pressure_shift_convention,
            )
            shifted_centers += _lblrtm_broadener_shift_correction(
                local_pressure_shift,
                local_broadener_flags,
                local_broadener_pressure_shifts,
                layer,
                reference_temperature_k=line_list.reference_temperature,
                prepared_shift_delta=local_broadener_shift_delta,
            )
            partition_ratio = None
            if q_ref is not None:
                q_t = partition_table.value(local_mol_id, local_iso_id, temperature_k)
                partition_ratio = q_ref / q_t
                approximate = ~np.isfinite(partition_ratio)
                if np.any(approximate):
                    partition_ratio[approximate] = (
                        line_list.reference_temperature / temperature_k
                    ) ** partition_exponent
            strength_t = line_strength_temperature(
                local_strength,
                local_wavenumber,
                effective_lower_energy,
                temperature_k,
                reference_temperature_k=line_list.reference_temperature,
                partition_exponent=partition_exponent,
                partition_ratio=partition_ratio,
            )
            if np.any(unknown_lower_energy):
                strength_t = np.array(strength_t, dtype=float, copy=True)
                strength_t[unknown_lower_energy] = local_strength[unknown_lower_energy]
            sigma = doppler_sigma_wavenumber(shifted_centers, temperature_k, local_mass)
            gamma = lorentz_hwhm_wavenumber(
                local_air_width,
                local_self_width,
                local_temp_exp,
                pressure_atm,
                temperature_k,
                absorber_fraction=absorber_fraction,
                reference_temperature_k=line_list.reference_temperature,
            )
            gamma = _lblrtm_broadener_lorentz_hwhm(
                base_gamma=gamma,
                air_width_cm=local_air_width,
                self_width_cm=local_self_width,
                temperature_exponent=local_temp_exp,
                mol_id=local_mol_id,
                broadener_flags=local_broadener_flags,
                broadener_widths=local_broadener_widths,
                broadener_temperature_exponents=local_broadener_temperature_exponents,
                layer=layer,
                reference_temperature_k=line_list.reference_temperature,
            )
            gamma, line_strength_multiplier, profile_coupling = _lblrtm_line_coupling_corrections(
                gamma=gamma,
                sigma=sigma,
                line_flags=local_line_flags,
                line_coupling_a=local_line_coupling_a,
                line_coupling_b=local_line_coupling_b,
                pressure_atm=pressure_atm,
                temperature_k=temperature_k,
            )
            source_alfv = None
            raw_alfv = None
            layer_wavenumber_spacing_cm = dynamic_wavenumber_spacing_cm
            if line_wing_mode in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}:
                raw_alfv = lblrtm_voigt_hwhm(gamma, sigma)
                if layer_wavenumber_spacings_cm is None:
                    raise RuntimeError("LBLRTM layer spacings were not initialized")
                layer_wavenumber_spacing_cm = float(
                    layer_wavenumber_spacings_cm[layer_index]
                )
                minimum_width = layer_wavenumber_spacing_cm
                if lblrtm_alfal0 > 0:
                    maximum_width = 4.0 * lblrtm_sample * minimum_width * 0.04 / lblrtm_alfal0
                    bounded_alfv = np.clip(raw_alfv, minimum_width, maximum_width)
                else:
                    bounded_alfv = np.maximum(raw_alfv, minimum_width)
                source_alfv = bounded_alfv
            dynamic_cutoff_cm = _line_cutoff_by_line(
                line_wing_mode=line_wing_mode,
                line_cutoff_cm=line_cutoff_cm,
                wavenumber_grid=wavenumber_grid,
                sigma=sigma,
                gamma=gamma,
                lblrtm_sample=lblrtm_sample,
                lblrtm_alfal0=lblrtm_alfal0,
                lblrtm_hwf3=lblrtm_hwf3,
                wavenumber_spacing_cm=layer_wavenumber_spacing_cm,
            )
            max_cutoff_cm = None if dynamic_cutoff_cm is None else float(np.nanmax(dynamic_cutoff_cm))
            line_scale = strength_t * columns * local_abundance_scale * line_strength_multiplier
            if source_alfv is not None:
                radiation = float(
                    lblrtm_radiation_term(np.nanmax(wavenumber_grid), temperature_k)
                )
                dptmin = LBLRTM_DEFAULT_DPTMIN / max(radiation, np.finfo(float).tiny)
                if f4_screening_pass:
                    if (
                        screening_r4_grid is None
                        or screening_r4_by_layer is None
                        or grouped_r4_by_layer is None
                        or raw_alfv is None
                    ):
                        raise RuntimeError("F4 screening state was not initialized")
                    if (
                        _CACHE_LBLRTM_SCREENED_LINE_STATE
                        and line_wing_mode == "lblrtm_panel"
                        and line_cutoff_cm is None
                    ):
                        prepared_chunks.append(
                            _LBLRTMPreparedLayerChunk(
                                layer_index=layer_index,
                                centers=shifted_centers,
                                sigma=sigma,
                                gamma=gamma,
                                line_scale=line_scale,
                                group_index=local_basis_index,
                                profile_coupling=profile_coupling,
                                effective_hwhm_cm=source_alfv,
                                maximum_cutoff_cm=max_cutoff_cm,
                                dptmin=dptmin,
                            )
                        )
                    line_scale = _screen_and_accumulate_f4_chunk(
                        r4_grid=screening_r4_grid,
                        total_r4=screening_r4_by_layer[layer_index],
                        grouped_r4=grouped_r4_by_layer[layer_index],
                        centers=shifted_centers,
                        sigma=sigma,
                        gamma=gamma,
                        raw_alfv=raw_alfv,
                        line_scale=line_scale,
                        group_index=local_basis_index,
                        profile_coupling=profile_coupling,
                        dptmin=dptmin,
                    )
                    continue

                threshold = np.full(line_scale.shape, dptmin, dtype=float)
                if screening_r4_grid is not None and screening_r4_by_layer is not None:
                    r4_spacing = float(screening_r4_grid[1] - screening_r4_grid[0])
                    r4_index = np.floor(
                        (shifted_centers - screening_r4_grid[0]) / r4_spacing
                    ).astype(int)
                    r4_index = np.clip(r4_index, 0, screening_r4_grid.size - 1)
                    threshold += LBLRTM_DEFAULT_DPTFAC * screening_r4_by_layer[layer_index][r4_index]
                reject = (line_scale / source_alfv <= threshold) & (profile_coupling == 0)
                line_scale = np.where(reject, 0.0, line_scale)
            voigt_hwhm = None
            if np.any(profile_coupling != 0):
                voigt_hwhm = lblrtm_voigt_hwhm(gamma, sigma)

            if (
                max_cutoff_cm is not None
                and _USE_SPARSE_FINITE_VOIGT
                and sparse_grid_available
                and line_wing_mode not in {"lblrtm_table", "lblrtm_panel"}
            ):
                _accumulate_sparse_voigt_basis(
                    basis,
                    wavenumber_grid=wavenumber_grid,
                    sorted_grid=sparse_sorted_grid,
                    grid_order=sparse_grid_order,
                    centers=shifted_centers,
                    sigma=sigma,
                    gamma=gamma,
                    line_scale=line_scale,
                    row_index=local_basis_index,
                    line_cutoff_cm=dynamic_cutoff_cm,
                    subtract_cutoff_profile=subtract_cutoff_profile,
                    line_taper_cm=line_taper_cm,
                    profile_coupling=profile_coupling,
                    profile_width=voigt_hwhm,
                )
                continue

            if max_cutoff_cm is None:
                grid_keep = slice(None)
                local_grid = wavenumber_grid
            else:
                local_grid_mask = (
                    (wavenumber_grid >= np.nanmin(shifted_centers) - max_cutoff_cm)
                    & (wavenumber_grid <= np.nanmax(shifted_centers) + max_cutoff_cm)
                )
                if not np.any(local_grid_mask):
                    continue
                grid_keep = local_grid_mask
                local_grid = wavenumber_grid[local_grid_mask]

            if line_wing_mode == "lblrtm_panel":
                if line_cutoff_cm is None:
                    basis[:, grid_keep] += lblrtm_panel_accumulate_wavenumber(
                        local_grid,
                        shifted_centers,
                        sigma,
                        gamma,
                        line_scale,
                        local_basis_index,
                        len(species_names),
                        profile_coupling=profile_coupling,
                        effective_hwhm_cm=source_alfv,
                        include_f4=completed_r4 is None,
                    )
                    continue
                profile = lblrtm_panel_voigt_profile_wavenumber(
                    local_grid,
                    shifted_centers,
                    sigma,
                    gamma,
                    effective_hwhm_cm=source_alfv,
                )
                if line_cutoff_cm is not None:
                    profile = _apply_line_wing_treatment(
                        profile,
                        local_grid,
                        shifted_centers,
                        sigma,
                        gamma,
                        line_cutoff_cm=dynamic_cutoff_cm,
                        subtract_cutoff_profile=False,
                        line_taper_cm=0.0,
                        line_wing_mode=line_wing_mode,
                    )
            elif line_wing_mode == "lblrtm_table":
                profile = lblrtm_tabulated_voigt_profile_offset(
                    local_grid[None, :] - shifted_centers[:, None],
                    gamma,
                    sigma,
                    effective_hwhm_cm=source_alfv,
                )
                if line_cutoff_cm is not None:
                    profile = _apply_line_wing_treatment(
                        profile,
                        local_grid,
                        shifted_centers,
                        sigma,
                        gamma,
                        line_cutoff_cm=dynamic_cutoff_cm,
                        subtract_cutoff_profile=False,
                        line_taper_cm=0.0,
                        line_wing_mode=line_wing_mode,
                    )
            else:
                profile = voigt_profile_wavenumber(local_grid, shifted_centers, sigma, gamma)
                profile = _apply_line_wing_treatment(
                    profile,
                    local_grid,
                    shifted_centers,
                    sigma,
                    gamma,
                    line_cutoff_cm=dynamic_cutoff_cm,
                    subtract_cutoff_profile=subtract_cutoff_profile,
                    line_taper_cm=line_taper_cm,
                    line_wing_mode=line_wing_mode,
                )
            if voigt_hwhm is not None:
                profile = _apply_lblrtm_line_coupling_profile(
                    profile,
                    local_grid,
                    shifted_centers,
                    profile_coupling,
                    voigt_hwhm,
                )
            tau_lines = profile * line_scale[:, None]

            for row_index, line_keep in species_row_groups:
                basis[row_index, grid_keep] += np.sum(tau_lines[line_keep], axis=0)

    f4_state = None
    if (
        screening_r4_grid is not None
        and screening_r4_by_layer is not None
        and grouped_r4_by_layer is not None
    ):
        f4_state = _LBLRTMF4State(
            grid_cm=screening_r4_grid,
            total_by_layer=tuple(screening_r4_by_layer),
            grouped_by_layer=tuple(grouped_r4_by_layer),
            prepared_chunks=tuple(prepared_chunks),
        )
    if (
        not f4_screening_pass
        and completed_r4 is not None
        and line_wing_mode == "lblrtm_panel"
        and line_cutoff_cm is None
    ):
        grouped_r4 = np.sum(np.stack(completed_r4.grouped_by_layer), axis=0)
        basis += lblrtm_panel_interpolate_f4_wavenumber(
            wavenumber_grid,
            completed_r4.grid_cm,
            grouped_r4,
        )

    return species_names, basis, f4_state


def _accumulate_prepared_lblrtm_panel_basis(
    basis: np.ndarray,
    *,
    wavenumber_grid: np.ndarray,
    completed_r4: _LBLRTMF4State,
) -> None:
    """Accumulate the second LBLRTM pass from first-pass physical state."""

    r4_spacing = float(completed_r4.grid_cm[1] - completed_r4.grid_cm[0])
    for prepared in completed_r4.prepared_chunks:
        r4_index = np.floor(
            (prepared.centers - completed_r4.grid_cm[0]) / r4_spacing
        ).astype(int)
        r4_index = np.clip(r4_index, 0, completed_r4.grid_cm.size - 1)
        threshold = prepared.dptmin + LBLRTM_DEFAULT_DPTFAC * completed_r4.total_by_layer[
            prepared.layer_index
        ][r4_index]
        reject = (
            prepared.line_scale / prepared.effective_hwhm_cm <= threshold
        ) & (prepared.profile_coupling == 0)
        line_scale = np.where(reject, 0.0, prepared.line_scale)

        if prepared.maximum_cutoff_cm is None:
            grid_keep: slice | np.ndarray = slice(None)
            local_grid = wavenumber_grid
        else:
            local_grid_mask = (
                wavenumber_grid
                >= np.nanmin(prepared.centers) - prepared.maximum_cutoff_cm
            ) & (
                wavenumber_grid
                <= np.nanmax(prepared.centers) + prepared.maximum_cutoff_cm
            )
            if not np.any(local_grid_mask):
                continue
            grid_keep = local_grid_mask
            local_grid = wavenumber_grid[local_grid_mask]

        basis[:, grid_keep] += lblrtm_panel_accumulate_wavenumber(
            local_grid,
            prepared.centers,
            prepared.sigma,
            prepared.gamma,
            line_scale,
            prepared.group_index,
            basis.shape[0],
            profile_coupling=prepared.profile_coupling,
            effective_hwhm_cm=prepared.effective_hwhm_cm,
            include_f4=False,
        )


def _screen_and_accumulate_f4_chunk(
    *,
    r4_grid: np.ndarray,
    total_r4: np.ndarray,
    grouped_r4: np.ndarray,
    centers: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    raw_alfv: np.ndarray,
    line_scale: np.ndarray,
    group_index: np.ndarray,
    profile_coupling: np.ndarray,
    dptmin: float,
) -> np.ndarray:
    """Apply the ordered ``CONVF4`` rejection test and update R4 in place."""

    a3, b3 = lblrtm_f4_coefficients(gamma, sigma)
    peak_factor = a3
    spacing = float(r4_grid[1] - r4_grid[0])
    center_index = np.floor((centers - r4_grid[0]) / spacing).astype(int)
    center_index = np.clip(center_index, 0, r4_grid.size - 1)
    support_start = np.searchsorted(
        r4_grid,
        centers - LBLRTM_F4_BOUND_CM,
        side="left",
    )
    support_stop = np.searchsorted(
        r4_grid,
        centers + LBLRTM_F4_BOUND_CM,
        side="right",
    )
    accepted_scale = np.zeros_like(line_scale)

    # Acceptance is causally dependent on R4 from earlier source-ordered lines.
    for line_index in range(line_scale.size):
        speak = peak_factor[line_index] * abs(line_scale[line_index] / raw_alfv[line_index])
        threshold = dptmin + LBLRTM_DEFAULT_DPTFAC * total_r4[center_index[line_index]]
        if profile_coupling[line_index] == 0 and speak <= threshold:
            continue
        accepted_scale[line_index] = line_scale[line_index]
        center = centers[line_index]
        start = int(support_start[line_index])
        stop = int(support_stop[line_index])
        if start >= stop:
            continue
        offset = r4_grid[start:stop] - center
        offset_sq = offset * offset
        alfv = raw_alfv[line_index]
        lorentz_sq = gamma[line_index] ** 2
        z_sq = offset_sq / alfv**2
        z_bound_sq = LBLRTM_VOIGT_DOMAIN_HWF3**2
        f4_at_64 = a3[line_index] + b3[line_index] * z_bound_sq
        lorentz_numerator = f4_at_64 / alfv * (lorentz_sq + alfv**2 * z_bound_sq)
        boundary_value = lorentz_numerator / (lorentz_sq + LBLRTM_F4_BOUND_CM**2)
        near = (a3[line_index] + b3[line_index] * z_sq) / alfv
        far = lorentz_numerator / (lorentz_sq + offset_sq)
        profile = np.where(z_sq <= z_bound_sq, near, far) - boundary_value
        if profile_coupling[line_index] != 0:
            profile *= 1.0 + profile_coupling[line_index] * offset / alfv
        contribution = profile * line_scale[line_index]
        total_r4[start:stop] += contribution
        group = int(group_index[line_index])
        if 0 <= group < grouped_r4.shape[0]:
            grouped_r4[group, start:stop] += contribution
    return accepted_scale


def _lblrtm_broadener_shift_correction(
    pressure_shift: np.ndarray,
    broadener_flags: np.ndarray | None,
    broadener_pressure_shifts: np.ndarray | None,
    layer,
    *,
    reference_temperature_k: float,
    prepared_shift_delta: np.ndarray | None = None,
) -> np.ndarray:
    pressure_shift = np.asarray(pressure_shift, dtype=float)
    if broadener_flags is None or broadener_pressure_shifts is None:
        return np.zeros(pressure_shift.shape, dtype=float)
    active = np.asarray(broadener_flags, dtype=int) > 0
    if not np.any(active):
        return np.zeros(pressure_shift.shape, dtype=float)
    density_ratio = layer.pressure_atm * (reference_temperature_k / layer.temperature_k)
    broadener_vmr = np.array([_species_vmr(layer, name) for name in LBLRTM_BROADENER_SPECIES], dtype=float)
    rhoslf = density_ratio * broadener_vmr
    shift_delta = (
        np.where(active, broadener_pressure_shifts - pressure_shift[:, None], 0.0)
        if prepared_shift_delta is None
        else np.asarray(prepared_shift_delta, dtype=float)
    )
    if shift_delta.shape != active.shape:
        raise ValueError("prepared broadener shift deltas do not match the line arrays")
    return np.einsum("ij,j->i", shift_delta, rhoslf, optimize=False)


def _lblrtm_self_mixture_corrected_air_width(
    air_width_cm: np.ndarray,
    self_width_cm: np.ndarray,
    mol_id: np.ndarray | None,
) -> np.ndarray:
    """Convert HITRAN air widths to LBLRTM foreign-width convention.

    LBLRTM adjusts O2 and N2 line widths while reading line data because HITRAN
    air broadening includes a contribution from the molecule's terrestrial
    self abundance. LBLRTM then treats ``ALFA0`` as the true foreign width in
    the layer correction.
    """

    air = np.asarray(air_width_cm, dtype=float).copy()
    if mol_id is None:
        return air

    mol = np.asarray(mol_id, dtype=int)
    for molecule_id, terrestrial_vmr in ((7, 0.21), (22, 0.79)):
        keep = mol == molecule_id
        if np.any(keep):
            air[keep] = (air[keep] - terrestrial_vmr * np.asarray(self_width_cm, dtype=float)[keep]) / (
                1.0 - terrestrial_vmr
            )
    return np.maximum(air, 0.0)


def _lblrtm_self_mixture_corrected_pressure_shift(
    pressure_shift_cm_per_atm: np.ndarray,
    broadener_flags: np.ndarray | None,
    broadener_pressure_shifts: np.ndarray | None,
    mol_id: np.ndarray | None,
) -> np.ndarray:
    """Apply LBLRTM's O2 air-shift correction when self-shift data exist."""

    pressure_shift = np.asarray(pressure_shift_cm_per_atm, dtype=float).copy()
    if mol_id is None or broadener_flags is None or broadener_pressure_shifts is None:
        return pressure_shift

    mol = np.asarray(mol_id, dtype=int)
    flags = np.asarray(broadener_flags, dtype=int)
    shifts = np.asarray(broadener_pressure_shifts, dtype=float)
    o2_index = LBLRTM_BROADENER_SPECIES.index("O2")
    keep = (mol == 7) & (flags[:, o2_index] > 0)
    if np.any(keep):
        terrestrial_o2_vmr = 0.21
        pressure_shift[keep] = (
            pressure_shift[keep] - terrestrial_o2_vmr * shifts[keep, o2_index]
        ) / (1.0 - terrestrial_o2_vmr)
    return pressure_shift


def _lblrtm_broadener_lorentz_hwhm(
    *,
    base_gamma: np.ndarray,
    air_width_cm: np.ndarray,
    self_width_cm: np.ndarray,
    temperature_exponent: np.ndarray,
    mol_id: np.ndarray | None,
    broadener_flags: np.ndarray | None,
    broadener_widths: np.ndarray | None,
    broadener_temperature_exponents: np.ndarray | None,
    layer,
    reference_temperature_k: float,
) -> np.ndarray:
    if broadener_flags is None or broadener_widths is None or broadener_temperature_exponents is None:
        return base_gamma
    active = np.asarray(broadener_flags, dtype=int) > 0
    if not np.any(active):
        return base_gamma

    density_ratio = layer.pressure_atm * (reference_temperature_k / layer.temperature_k)
    temperature_ratio = layer.temperature_k / reference_temperature_k
    broadener_vmr = np.array([_species_vmr(layer, name) for name in LBLRTM_BROADENER_SPECIES], dtype=float)
    rhoslf = density_ratio * broadener_vmr

    tmpalf_air = 1.0 - np.asarray(temperature_exponent, dtype=float)
    alfa0i = np.asarray(air_width_cm, dtype=float) * temperature_ratio**tmpalf_air
    hwhmsi = np.asarray(self_width_cm, dtype=float) * temperature_ratio**tmpalf_air
    flag_density = np.sum(rhoslf[None, :] * active, axis=1)
    gamma = (density_ratio - flag_density) * alfa0i
    gamma += np.sum(
        rhoslf[None, :]
        * active
        * np.asarray(broadener_widths, dtype=float)
        * temperature_ratio ** np.asarray(broadener_temperature_exponents, dtype=float),
        axis=1,
    )

    if mol_id is not None:
        mol_index = np.asarray(mol_id, dtype=int) - 1
        valid = (mol_index >= 0) & (mol_index < len(LBLRTM_BROADENER_SPECIES))
        if np.any(valid):
            rows = np.nonzero(valid)[0]
            self_flagged = active[rows, mol_index[valid]]
            self_rhoslf = rhoslf[mol_index[valid]]
            gamma[rows] += np.where(self_flagged, 0.0, self_rhoslf * (hwhmsi[rows] - alfa0i[rows]))

    return np.maximum(gamma, 0.0)


def _lblrtm_line_coupling_corrections(
    *,
    gamma: np.ndarray,
    sigma: np.ndarray,
    line_flags: np.ndarray | None,
    line_coupling_a: np.ndarray | None,
    line_coupling_b: np.ndarray | None,
    pressure_atm: float,
    temperature_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma = np.asarray(gamma, dtype=float)
    strength_multiplier = np.ones(gamma.shape, dtype=float)
    profile_coupling = np.zeros(gamma.shape, dtype=float)
    if line_flags is None or line_coupling_a is None or line_coupling_b is None:
        return gamma, strength_multiplier, profile_coupling

    flags = np.asarray(line_flags, dtype=int)
    if not np.any((flags == 1) | (flags == 3)):
        return gamma, strength_multiplier, profile_coupling

    interp_a, interp_b = _interpolate_lblrtm_line_coupling(
        np.asarray(line_coupling_a, dtype=float),
        np.asarray(line_coupling_b, dtype=float),
        temperature_k,
    )
    pressure_ratio = float(pressure_atm)
    pressure_ratio2 = pressure_ratio * pressure_ratio

    reduced_width = flags == 3
    if np.any(reduced_width):
        correction = 1.0 - interp_a[reduced_width] * pressure_ratio - interp_b[reduced_width] * pressure_ratio2
        gamma = np.array(gamma, dtype=float, copy=True)
        gamma[reduced_width] = np.maximum(gamma[reduced_width] * correction, 0.0)

    line_coupled = flags == 1
    if np.any(line_coupled):
        multiplier = 1.0 + interp_b[line_coupled] * pressure_ratio2
        tiny = np.abs(multiplier) < 1.0e-300
        if np.any(tiny):
            multiplier = np.array(multiplier, dtype=float, copy=True)
            multiplier[tiny] = np.copysign(1.0e-300, multiplier[tiny])
        strength_multiplier[line_coupled] = multiplier
        profile_coupling[line_coupled] = interp_a[line_coupled] * pressure_ratio / multiplier

    del sigma
    return gamma, strength_multiplier, profile_coupling


def _interpolate_lblrtm_line_coupling(
    coupling_a: np.ndarray,
    coupling_b: np.ndarray,
    temperature_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    temperatures = np.array([200.0, 250.0, 296.0, 340.0], dtype=float)
    # LBLRTM uses the first interval whose upper temperature is strictly above
    # TAVE; equality falls through to the next interval. Values outside the
    # table are extrapolated from the nearest end interval.
    interval = int(np.searchsorted(temperatures[1:], float(temperature_k), side="right"))
    interval = min(max(interval, 0), temperatures.size - 2)
    t0 = temperatures[interval]
    t1 = temperatures[interval + 1]
    fraction = (float(temperature_k) - t0) / (t1 - t0)
    a = coupling_a[:, interval] + fraction * (coupling_a[:, interval + 1] - coupling_a[:, interval])
    b = coupling_b[:, interval] + fraction * (coupling_b[:, interval + 1] - coupling_b[:, interval])
    return a, b


def _apply_lblrtm_line_coupling_profile(
    profile: np.ndarray,
    wavenumber_grid: np.ndarray,
    centers: np.ndarray,
    profile_coupling: np.ndarray,
    profile_width: np.ndarray,
) -> np.ndarray:
    coupling = np.asarray(profile_coupling, dtype=float)
    if not np.any(coupling != 0):
        return profile
    width = np.asarray(profile_width, dtype=float)
    valid_width = np.isfinite(width) & (width > 0)
    if not np.any(valid_width):
        return profile
    normalized_offset = np.zeros(profile.shape, dtype=float)
    normalized_offset[valid_width] = (
        np.asarray(wavenumber_grid, dtype=float)[None, :] - np.asarray(centers, dtype=float)[:, None]
    )[valid_width] / width[valid_width, None]
    return profile * (1.0 + coupling[:, None] * normalized_offset)


def _apply_line_wing_treatment(
    profile: np.ndarray,
    wavenumber_grid: np.ndarray,
    centers: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    *,
    line_cutoff_cm: float | np.ndarray | None,
    subtract_cutoff_profile: bool,
    line_taper_cm: float,
    line_wing_mode: str,
) -> np.ndarray:
    if line_cutoff_cm is None:
        return profile

    distance = np.abs(wavenumber_grid[None, :] - centers[:, None])
    cutoff = np.asarray(line_cutoff_cm, dtype=float)
    if cutoff.ndim == 0:
        cutoff_by_line = np.full(centers.shape, float(cutoff), dtype=float)
    else:
        cutoff_by_line = np.broadcast_to(cutoff, centers.shape).astype(float, copy=False)
    cutoff_grid = cutoff_by_line[:, None]
    treated = profile
    if subtract_cutoff_profile:
        edge = voigt_profile_offset(cutoff_by_line, sigma, gamma)
        treated = np.maximum(treated - edge[:, None], 0.0)

    if line_taper_cm > 0:
        taper_width = np.minimum(float(line_taper_cm), cutoff_by_line)
        inner = cutoff_by_line - taper_width
        weight = np.ones_like(treated)
        in_taper = (distance > inner[:, None]) & (distance < cutoff_grid) & (taper_width[:, None] > 0)
        weight[distance >= cutoff_grid] = 0.0
        weight[in_taper] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (distance[in_taper] - np.broadcast_to(inner[:, None], distance.shape)[in_taper])
                / np.broadcast_to(taper_width[:, None], distance.shape)[in_taper]
            )
        )
        return treated * weight

    return np.where(distance <= cutoff_grid, treated, 0.0)


def _accumulate_sparse_voigt_basis(
    basis: np.ndarray,
    *,
    wavenumber_grid: np.ndarray,
    sorted_grid: np.ndarray,
    grid_order: np.ndarray,
    centers: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    line_scale: np.ndarray,
    row_index: np.ndarray,
    line_cutoff_cm: float | np.ndarray,
    subtract_cutoff_profile: bool,
    line_taper_cm: float,
    profile_coupling: np.ndarray | None = None,
    profile_width: np.ndarray | None = None,
) -> None:
    """Accumulate finite-wing Voigt profiles without building a dense matrix."""

    centers = np.asarray(centers, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    line_scale = np.asarray(line_scale, dtype=float)
    row_index = np.asarray(row_index, dtype=int)
    cutoff = np.asarray(line_cutoff_cm, dtype=float)
    if cutoff.ndim == 0:
        cutoff_by_line = np.full(centers.shape, float(cutoff), dtype=float)
    else:
        cutoff_by_line = np.broadcast_to(cutoff, centers.shape).astype(float, copy=False)

    valid_line = (
        np.isfinite(centers)
        & np.isfinite(sigma)
        & np.isfinite(gamma)
        & np.isfinite(line_scale)
        & np.isfinite(cutoff_by_line)
        & (sigma > 0)
        & (gamma >= 0)
        & (line_scale != 0)
        & (cutoff_by_line > 0)
        & (row_index >= 0)
    )
    if not np.any(valid_line):
        return

    left = np.searchsorted(sorted_grid, centers - cutoff_by_line, side="left")
    right = np.searchsorted(sorted_grid, centers + cutoff_by_line, side="right")
    counts = np.maximum(right - left, 0)
    valid_line &= counts > 0
    if not np.any(valid_line):
        return

    valid_rows = np.nonzero(valid_line)[0]
    counts = counts[valid_rows]
    total = int(np.sum(counts))
    if total == 0:
        return

    repeated_rows = np.repeat(valid_rows, counts)
    first = np.repeat(left[valid_rows], counts)
    offsets = np.arange(total, dtype=int) - np.repeat(np.cumsum(counts) - counts, counts)
    grid_indices = grid_order[first + offsets]
    distance = np.asarray(wavenumber_grid, dtype=float)[grid_indices] - centers[repeated_rows]
    distance_abs = np.abs(distance)
    line_cutoff = cutoff_by_line[repeated_rows]
    inside = distance_abs <= line_cutoff
    if not np.all(inside):
        repeated_rows = repeated_rows[inside]
        grid_indices = grid_indices[inside]
        distance = distance[inside]
        distance_abs = distance_abs[inside]
        line_cutoff = line_cutoff[inside]
    if repeated_rows.size == 0:
        return

    profile = voigt_profile_offset(distance, sigma[repeated_rows], gamma[repeated_rows])
    if subtract_cutoff_profile:
        edge = voigt_profile_offset(cutoff_by_line, sigma, gamma)
        profile = np.maximum(profile - edge[repeated_rows], 0.0)

    if line_taper_cm > 0:
        taper_width_by_line = np.minimum(float(line_taper_cm), cutoff_by_line)
        taper_width = taper_width_by_line[repeated_rows]
        inner = line_cutoff - taper_width
        in_taper = (distance_abs > inner) & (distance_abs < line_cutoff) & (taper_width > 0)
        if np.any(in_taper):
            weight = np.ones(profile.shape, dtype=float)
            weight[in_taper] = 0.5 * (
                1.0 + np.cos(np.pi * (distance_abs[in_taper] - inner[in_taper]) / taper_width[in_taper])
            )
            profile = profile * weight

    if profile_coupling is not None and profile_width is not None:
        coupling = np.asarray(profile_coupling, dtype=float)
        width = np.asarray(profile_width, dtype=float)
        line_coupling = coupling[repeated_rows]
        line_width = width[repeated_rows]
        coupled = (line_coupling != 0) & np.isfinite(line_width) & (line_width > 0)
        if np.any(coupled):
            profile = np.array(profile, dtype=float, copy=True)
            profile[coupled] *= 1.0 + line_coupling[coupled] * distance[coupled] / line_width[coupled]

    values = profile * line_scale[repeated_rows]
    finite = np.isfinite(values) & (values != 0)
    if not np.any(finite):
        return
    np.add.at(basis, (row_index[repeated_rows[finite]], grid_indices[finite]), values[finite])


def line_wing_effective_cutoff_cm(line_wing_mode: str = "full", line_cutoff_cm: float | None = None) -> float | None:
    """Return the finite wing cutoff implied by a line-wing mode."""

    mode = _normalize_line_wing_mode(line_wing_mode)
    if line_cutoff_cm is not None:
        if line_cutoff_cm <= 0:
            raise ValueError("line_cutoff_cm must be positive when provided")
        return float(line_cutoff_cm)
    if mode in {
        "hard_cutoff",
        "subtracted_cutoff",
        "tapered_cutoff",
        "lblrtm_subtracted",
        "lblrtm_dynamic",
        "lblrtm_table",
        "lblrtm_panel",
    }:
        return DEFAULT_LBLRTM_LINE_CUTOFF_CM
    return None


def _line_wing_settings(
    *,
    line_wing_mode: str,
    line_cutoff_cm: float | None,
    subtract_cutoff_profile: bool,
    line_taper_cm: float,
) -> tuple[str, float | None, bool, float]:
    mode = _normalize_line_wing_mode(line_wing_mode)
    if line_taper_cm < 0:
        raise ValueError("line_taper_cm must be non-negative")
    if mode in {"lblrtm_dynamic", "lblrtm_panel"}:
        if line_cutoff_cm is not None and line_cutoff_cm <= 0:
            raise ValueError("line_cutoff_cm must be positive when provided")
        cutoff = None if line_cutoff_cm is None else float(line_cutoff_cm)
    else:
        cutoff = line_wing_effective_cutoff_cm(mode, line_cutoff_cm)
    if mode in {"subtracted_cutoff", "lblrtm_subtracted"}:
        subtract_cutoff_profile = True
    elif mode in {"lblrtm_table", "lblrtm_panel"}:
        subtract_cutoff_profile = False
        line_taper_cm = 0.0
    elif mode == "hard_cutoff":
        subtract_cutoff_profile = False
        line_taper_cm = 0.0
    elif mode == "tapered_cutoff" and line_taper_cm == 0:
        line_taper_cm = min(1.0, cutoff if cutoff is not None else DEFAULT_LBLRTM_LINE_CUTOFF_CM)
    if mode == "lblrtm_dynamic" and line_taper_cm == 0:
        line_taper_cm = 0.0
    return mode, cutoff, subtract_cutoff_profile, float(line_taper_cm)


def _line_cutoff_by_line(
    *,
    line_wing_mode: str,
    line_cutoff_cm: float | None,
    wavenumber_grid: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    lblrtm_sample: float,
    lblrtm_alfal0: float,
    lblrtm_hwf3: float,
    wavenumber_spacing_cm: float | None = None,
) -> np.ndarray | None:
    if line_wing_mode not in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}:
        return None if line_cutoff_cm is None else np.full(sigma.shape, float(line_cutoff_cm), dtype=float)

    cutoff = lblrtm_dynamic_line_cutoff_cm(
        gamma,
        sigma,
        _wavenumber_spacing_cm(wavenumber_grid) if wavenumber_spacing_cm is None else wavenumber_spacing_cm,
        sample=lblrtm_sample,
        alfal0=lblrtm_alfal0,
        hwf3=lblrtm_hwf3,
    )
    if line_wing_mode == "lblrtm_panel":
        # LBLRTM's separately accumulated F4 closure extends to BOUND4=25
        # cm-1 even when the F1/F2/F3 dynamic core is narrower.
        cutoff = np.maximum(cutoff, DEFAULT_LBLRTM_LINE_CUTOFF_CM)
    if line_cutoff_cm is not None:
        cutoff = np.minimum(cutoff, float(line_cutoff_cm))
    return cutoff


def _wavenumber_spacing_cm(wavenumber_grid: np.ndarray) -> float:
    grid = np.asarray(wavenumber_grid, dtype=float)
    finite = np.sort(grid[np.isfinite(grid)])
    if finite.size < 2:
        raise ValueError("lblrtm_dynamic line-wing mode requires at least two wavenumber pixels")
    spacing = np.diff(finite)
    spacing = spacing[spacing > 0]
    if spacing.size == 0:
        raise ValueError("wavenumber grid must span a non-zero range")
    return float(np.nanmedian(spacing))


def _validate_lblrtm_dynamic_controls(
    *, sample: float, alfal0: float, avmass_amu: float, hwf3: float
) -> None:
    if sample <= 0:
        raise ValueError("lblrtm_sample must be positive")
    if alfal0 < 0:
        raise ValueError("lblrtm_alfal0 must be non-negative")
    if avmass_amu <= 0 or not np.isfinite(avmass_amu):
        raise ValueError("lblrtm_avmass_amu must be positive and finite")
    if hwf3 <= 0:
        raise ValueError("lblrtm_hwf3 must be positive")


def _normalize_line_wing_mode(line_wing_mode: str) -> str:
    mode = str(line_wing_mode).strip().lower()
    if mode not in LINE_WING_MODES:
        choices = ", ".join(sorted(LINE_WING_MODES))
        raise ValueError(f"unknown line_wing_mode {line_wing_mode!r}; expected one of: {choices}")
    return mode


def _combine_species_filters(
    component_species: tuple[str, ...] | None,
    requested_species: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if component_species is None:
        return requested_species
    if requested_species is None:
        return component_species
    requested = set(requested_species)
    return tuple(name for name in component_species if name in requested)


def _empty_basis(wavelength_micron: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    return (), np.zeros((0, wavelength_micron.size), dtype=float)


def _atmosphere_has_species(atmosphere: AtmosphereProfile, species: str) -> bool:
    return any(layer.mixing_ratios.get(species, 0.0) > 0 for layer in atmosphere.layers)


def _air_number_density_cm3(layer) -> float:
    pressure_pa = layer.pressure_atm * PA_PER_ATM
    return pressure_pa / (BOLTZMANN_J_PER_K * layer.temperature_k) / (CM_PER_M**3)


def _air_amagat(layer) -> float:
    return _air_number_density_cm3(layer) / LOSCHMIDT_CM3


def _air_column_density_cm2(layer) -> float:
    return _air_number_density_cm3(layer) * layer.path_length_m * CM_PER_M


def _species_number_density_cm3(layer, species: str) -> float:
    return _air_number_density_cm3(layer) * _species_vmr(layer, species)


def _species_vmr(layer, species: str) -> float:
    if species in layer.mixing_ratios:
        return float(layer.mixing_ratios[species])
    if species == "N2":
        return max(0.0, 1.0 - sum(float(value) for value in layer.mixing_ratios.values()))
    if species.upper() == "AIR":
        return 1.0
    return 0.0
