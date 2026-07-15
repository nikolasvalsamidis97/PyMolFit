from __future__ import annotations

from functools import lru_cache

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.special import wofz

BOLTZMANN_J_PER_K = const.k_B.si.value
SPEED_OF_LIGHT_M_PER_S = const.c.si.value
AMU_KG = const.u.si.value
SECOND_RADIATION_CONSTANT_CM_K = (const.h * const.c / const.k_B).to_value(u.cm * u.K)
LOSCHMIDT_CM3 = ((101_325.0 * u.Pa) / (const.k_B * (273.15 * u.K))).to_value(1 / u.cm**3)

# Legacy constants used by the LBLRTM source bundled with Molecfit 4.4.4
# (lblrtm/src/phys_consts.f90). PyMolFit defaults to modern Astropy values,
# but these constants make source-parity tests explicit and reproducible.
LBLRTM_BOLTZMANN_J_PER_K = 1.3806503e-23
LBLRTM_SPEED_OF_LIGHT_M_PER_S = 2.99792458e8
LBLRTM_AVOGADRO_PER_MOL = 6.02214199e23
LBLRTM_LOSCHMIDT_CM3 = 2.6867775e19
LBLRTM_SECOND_RADIATION_CONSTANT_CM_K = 1.4387752
LBLRTM_VOIGT_DOMAIN_HWF3 = 64.0
LBLRTM_DEFAULT_SAMPLE = 4.0
LBLRTM_DEFAULT_ALFAL0 = 0.04
LBLRTM_DEFAULT_AVMASS_AMU = 36.0
LBLRTM_REFERENCE_PRESSURE_ATM = 1.0
LBLRTM_REFERENCE_TEMPERATURE_K = 296.0
# ``MANE`` in LBLRTM computes the representative Doppler HWHM as
# 3.58115e-7 * wavenumber * sqrt(temperature / AVMASS).
LBLRTM_DOPPLER_HWHM_FACTOR = 3.58115e-7
LBLRTM_DEFAULT_DPTMIN = 2.0e-4
LBLRTM_DEFAULT_DPTFAC = 1.0e-3
LBLRTM_VOIGT_TABLE_POINTS = 2001
LBLRTM_VOIGT_TABLE_DOMAINS = (4.0, 16.0, 64.0)
LBLRTM_F4_BOUND_CM = 25.0
LBLRTM_F4_GRID_RATIO = 64

LBLRTM_AVRAT = np.array(
    [
        1.00000,
        0.99535,
        0.99073,
        0.98613,
        0.98155,
        0.97700,
        0.97247,
        0.96797,
        0.96350,
        0.95905,
        0.95464,
        0.95025,
        0.94589,
        0.94156,
        0.93727,
        0.93301,
        0.92879,
        0.92460,
        0.92045,
        0.91634,
        0.91227,
        0.90824,
        0.90425,
        0.90031,
        0.89641,
        0.89256,
        0.88876,
        0.88501,
        0.88132,
        0.87768,
        0.87410,
        0.87058,
        0.86712,
        0.86372,
        0.86039,
        0.85713,
        0.85395,
        0.85083,
        0.84780,
        0.84484,
        0.84197,
        0.83919,
        0.83650,
        0.83390,
        0.83141,
        0.82901,
        0.82672,
        0.82454,
        0.82248,
        0.82053,
        0.81871,
        0.81702,
        0.81547,
        0.81405,
        0.81278,
        0.81166,
        0.81069,
        0.80989,
        0.80925,
        0.80879,
        0.80851,
        0.80842,
        0.80852,
        0.80882,
        0.80932,
        0.81004,
        0.81098,
        0.81214,
        0.81353,
        0.81516,
        0.81704,
        0.81916,
        0.82154,
        0.82418,
        0.82708,
        0.83025,
        0.83370,
        0.83742,
        0.84143,
        0.84572,
        0.85029,
        0.85515,
        0.86030,
        0.86573,
        0.87146,
        0.87747,
        0.88376,
        0.89035,
        0.89721,
        0.90435,
        0.91176,
        0.91945,
        0.92741,
        0.93562,
        0.94409,
        0.95282,
        0.96179,
        0.97100,
        0.98044,
        0.99011,
        1.00000,
        1.00000,
    ],
    dtype=float,
)
LBLRTM_AVRAT_ZETA_GRID = np.linspace(0.0, 1.01, LBLRTM_AVRAT.size)


def wavelength_micron_to_wavenumber_cm(wavelength_micron: np.ndarray) -> np.ndarray:
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    return 1.0e4 / wavelength_micron


def wavenumber_cm_to_wavelength_micron(wavenumber_cm: np.ndarray) -> np.ndarray:
    wavenumber_cm = np.asarray(wavenumber_cm, dtype=float)
    return 1.0e4 / wavenumber_cm


def line_strength_temperature(
    strength_ref: np.ndarray,
    wavenumber_cm: np.ndarray,
    lower_state_energy_cm: np.ndarray,
    temperature_k: float,
    *,
    reference_temperature_k: float = 296.0,
    partition_exponent: float = 1.5,
    partition_ratio: np.ndarray | None = None,
    second_radiation_constant_cm_k: float = SECOND_RADIATION_CONSTANT_CM_K,
) -> np.ndarray:
    """Scale HITRAN line intensities from reference temperature to layer T.

    If no tabulated partition ratio is supplied, the partition-function ratio
    is approximated as ``(T_ref / T)**exponent``.
    """

    strength_ref = np.asarray(strength_ref, dtype=float)
    wavenumber_cm = np.asarray(wavenumber_cm, dtype=float)
    lower_state_energy_cm = np.asarray(lower_state_energy_cm, dtype=float)
    t = float(temperature_k)
    tref = float(reference_temperature_k)
    c2 = float(second_radiation_constant_cm_k)

    if partition_ratio is None:
        partition_ratio = (tref / t) ** partition_exponent
    else:
        partition_ratio = np.asarray(partition_ratio, dtype=float)
    lower_state = np.exp(-c2 * lower_state_energy_cm * (1.0 / t - 1.0 / tref))
    stimulated = (1.0 - np.exp(-c2 * wavenumber_cm / t)) / (
        1.0 - np.exp(-c2 * wavenumber_cm / tref)
    )
    return strength_ref * partition_ratio * lower_state * stimulated


def lblrtm_temperature_scaling_lower_energy(
    lower_state_energy_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return LBLRTM-compatible lower energies and unknown-EPP mask.

    LBLRTM treats lower-state energy values around ``-1`` as an unknown-EPP
    sentinel and skips the line-strength temperature correction for those
    transitions. Values below ``-1.001`` are treated as user-supplied negative
    energies and converted to their absolute value before scaling.
    """

    lower_state_energy = np.asarray(lower_state_energy_cm, dtype=float)
    effective = np.array(lower_state_energy, dtype=float, copy=True)
    effective[effective <= -1.001] = np.abs(effective[effective <= -1.001])
    unknown = effective <= -0.999
    return effective, unknown


def doppler_sigma_wavenumber(
    wavenumber_cm: np.ndarray,
    temperature_k: float,
    molecular_mass_amu: np.ndarray,
) -> np.ndarray:
    mass_kg = np.asarray(molecular_mass_amu, dtype=float) * AMU_KG
    wavenumber_cm = np.asarray(wavenumber_cm, dtype=float)
    return wavenumber_cm * np.sqrt(
        BOLTZMANN_J_PER_K * temperature_k / (mass_kg * SPEED_OF_LIGHT_M_PER_S**2)
    )


def lorentz_hwhm_wavenumber(
    air_width_cm: np.ndarray,
    self_width_cm: np.ndarray,
    temperature_exponent: np.ndarray,
    pressure_atm: float,
    temperature_k: float,
    *,
    absorber_fraction: float,
    reference_temperature_k: float = 296.0,
) -> np.ndarray:
    air_width_cm = np.asarray(air_width_cm, dtype=float)
    self_width_cm = np.asarray(self_width_cm, dtype=float)
    temperature_exponent = np.asarray(temperature_exponent, dtype=float)
    broadening = (1.0 - absorber_fraction) * air_width_cm + absorber_fraction * self_width_cm
    return broadening * pressure_atm * (reference_temperature_k / temperature_k) ** temperature_exponent


def lblrtm_voigt_hwhm(
    lorentz_hwhm_cm: np.ndarray,
    doppler_sigma_cm: np.ndarray,
) -> np.ndarray:
    """LBLRTM effective Voigt half-width from Lorentz HWHM and Gaussian sigma.

    This follows `oprop_voigt.f90`: zeta is `ALFL / (ALFL + ALFAD)`, where
    `ALFAD` is the Doppler half-width at half height, and the `AVRAT` table
    maps `(ALFL + ALFAD)` to the Voigt half-width used by the line-window
    machinery.
    """

    lorentz = np.asarray(lorentz_hwhm_cm, dtype=float)
    doppler_sigma = np.asarray(doppler_sigma_cm, dtype=float)
    doppler_hwhm = doppler_sigma * np.sqrt(2.0 * np.log(2.0))
    total = lorentz + doppler_hwhm
    zeta = np.zeros(np.broadcast_shapes(lorentz.shape, doppler_hwhm.shape), dtype=float)
    total_b = np.broadcast_to(total, zeta.shape)
    lorentz_b = np.broadcast_to(lorentz, zeta.shape)
    valid = total_b > 0
    zeta[valid] = np.clip(lorentz_b[valid] / total_b[valid], 0.0, 1.0)
    ratio = np.interp(zeta, LBLRTM_AVRAT_ZETA_GRID, LBLRTM_AVRAT)
    return ratio * total_b


def lblrtm_layer_wavenumber_spacing_cm(
    representative_wavenumber_cm: float | np.ndarray,
    pressure_atm: float | np.ndarray,
    temperature_k: float | np.ndarray,
    *,
    h2o_fraction: float | np.ndarray = 0.0,
    sample: float = LBLRTM_DEFAULT_SAMPLE,
    alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
) -> np.ndarray:
    """Return LBLRTM's unquantized, layer-specific ``CALC DV``.

    This follows ``MANE`` in ``lblrtm.f90`` for the normal Voigt path
    (``IHIRAC=0``): the representative Lorentz and Doppler HWHM values are
    combined into ``AVBAR``, then divided by ``SAMPLE``.  The result belongs
    to one atmospheric layer.  It must not be replaced by the finest spacing
    of a subsequently merged multi-layer grid when LBLRTM bounds individual
    line widths.
    """

    wavenumber = np.asarray(representative_wavenumber_cm, dtype=float)
    pressure = np.asarray(pressure_atm, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    water = np.asarray(h2o_fraction, dtype=float)
    if sample <= 0 or not np.isfinite(sample):
        raise ValueError("sample must be positive and finite")
    if alfal0 < 0 or not np.isfinite(alfal0):
        raise ValueError("alfal0 must be non-negative and finite")
    if avmass_amu <= 0 or not np.isfinite(avmass_amu):
        raise ValueError("avmass_amu must be positive and finite")
    if np.any(~np.isfinite(wavenumber)) or np.any(wavenumber <= 0):
        raise ValueError("representative_wavenumber_cm must be positive and finite")
    if np.any(~np.isfinite(pressure)) or np.any(pressure < 0):
        raise ValueError("pressure_atm must be non-negative and finite")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("temperature_k must be positive and finite")
    if np.any(~np.isfinite(water)) or np.any((water < 0) | (water > 1)):
        raise ValueError("h2o_fraction must be finite and between zero and one")

    h2o_self_factor = 1.0 + 4.0 * water
    lorentz_hwhm = (
        float(alfal0)
        * pressure
        * np.sqrt(LBLRTM_REFERENCE_TEMPERATURE_K / temperature)
        * h2o_self_factor
    )
    doppler_hwhm = (
        LBLRTM_DOPPLER_HWHM_FACTOR
        * wavenumber
        * np.sqrt(temperature / float(avmass_amu))
    )
    voigt_hwhm = 0.5 * (
        lorentz_hwhm
        + np.sqrt(lorentz_hwhm * lorentz_hwhm + 4.0 * doppler_hwhm * doppler_hwhm)
    )
    return voigt_hwhm / float(sample)


def lblrtm_merge_layer_wavenumber_spacings_cm(
    calculated_spacing_cm: np.ndarray,
    *,
    emission: bool = True,
) -> np.ndarray:
    """Apply LBLRTM's sequential ``DVL`` quantization and merge rules.

    ``MANE`` first rounds the lowest-layer spacing to three significant
    figures with an even integer mantissa.  Later layers either reuse that
    spacing or select an exactly mergeable rational subdivision.  This is
    the default ``IOD=0, IMRG=0`` path used by Molecfit's LBLRTM calls.
    """

    calculated = np.asarray(calculated_spacing_cm, dtype=float)
    if calculated.ndim != 1 or calculated.size == 0:
        raise ValueError("calculated_spacing_cm must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(calculated)) or np.any(calculated <= 0):
        raise ValueError("calculated_spacing_cm must contain positive finite values")

    merged = np.empty_like(calculated)
    previous = 0.0
    for index, calculated_dv in enumerate(calculated):
        dv = float(calculated_dv)
        if index == 0:
            # Fortran integer assignment truncates toward zero.
            exponent = int(np.trunc(np.log10(dv) - 3.0))
            scale = 10.0**exponent
            integer_dv = int(np.trunc(dv / scale + 0.5))
            if integer_dv % 2:
                integer_dv += 1
            dv = scale * float(integer_dv)
        else:
            ratio = previous / dv
            if ratio > 2.5:
                # LBLRTM flags this profile for diagnostics but leaves the
                # current calculated spacing in DVL.
                pass
            elif ratio >= 1.2:
                merge_index = int(np.trunc(1.0 / (ratio - 1.0) + 0.5))
                if merge_index == 3:
                    merge_index = 2
                dv = previous * float(merge_index) / float(merge_index + 1)
            elif ratio >= 0.8 or emission:
                dv = previous
            else:
                merge_index = int(np.trunc(ratio / (1.0 - ratio) + 0.5))
                if merge_index < 1:
                    raise ValueError("LBLRTM layer merge produced an invalid spacing ratio")
                dv = previous * float(merge_index + 1) / float(merge_index)
        merged[index] = dv
        previous = dv
    return merged


def lblrtm_layer_wavenumber_spacings_cm(
    representative_wavenumber_cm: float,
    pressure_atm: np.ndarray,
    temperature_k: np.ndarray,
    *,
    h2o_fraction: np.ndarray | float = 0.0,
    sample: float = LBLRTM_DEFAULT_SAMPLE,
    alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    avmass_amu: float = LBLRTM_DEFAULT_AVMASS_AMU,
    merge: bool = True,
    emission: bool = True,
) -> np.ndarray:
    """Return the ordered LBLRTM spectral spacing for every layer."""

    calculated = np.asarray(
        lblrtm_layer_wavenumber_spacing_cm(
            representative_wavenumber_cm,
            pressure_atm,
            temperature_k,
            h2o_fraction=h2o_fraction,
            sample=sample,
            alfal0=alfal0,
            avmass_amu=avmass_amu,
        ),
        dtype=float,
    )
    if calculated.ndim != 1:
        raise ValueError("layer pressure, temperature, and H2O fraction must be one-dimensional")
    if not merge:
        return calculated
    return lblrtm_merge_layer_wavenumber_spacings_cm(calculated, emission=emission)


def lblrtm_dynamic_line_cutoff_cm(
    lorentz_hwhm_cm: np.ndarray,
    doppler_sigma_cm: np.ndarray,
    grid_spacing_cm: float,
    *,
    sample: float = LBLRTM_DEFAULT_SAMPLE,
    alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3,
) -> np.ndarray:
    """LBLRTM-style dynamic line window in cm-1.

    Source correspondence:

    - `ALFV = AVRAT(zeta) * (ALFL + ALFAD)`
    - `ALFV = max(ALFV, DV)`
    - `ALFMAX = 4 * SAMPLE * DV * 0.04 / ALFAL0`
    - `ALFV = min(ALFV, ALFMAX)` for positive `ALFAL0`
    - active line extent is `HWF3 * ALFV`

    Molecfit writes `ALFAL0=0` by default for some runs. The bundled LBLRTM
    source divides by that value, which effectively removes the finite
    `ALFMAX` cap on IEEE platforms. PyMolFit treats zero the same way while
    still rejecting negative values.
    """

    dv = float(grid_spacing_cm)
    if dv <= 0 or not np.isfinite(dv):
        raise ValueError("grid_spacing_cm must be positive")
    if sample <= 0:
        raise ValueError("sample must be positive")
    if alfal0 < 0:
        raise ValueError("alfal0 must be non-negative")
    if hwf3 <= 0:
        raise ValueError("hwf3 must be positive")

    alfv = lblrtm_voigt_hwhm(lorentz_hwhm_cm, doppler_sigma_cm)
    alfv = np.maximum(alfv, dv)
    if alfal0 > 0:
        alfmax = 4.0 * float(sample) * dv * 0.04 / float(alfal0)
        alfv = np.minimum(alfv, alfmax)
    return float(hwf3) * alfv


def lblrtm_dynamic_max_line_cutoff_cm(
    grid_spacing_cm: float,
    *,
    sample: float = LBLRTM_DEFAULT_SAMPLE,
    alfal0: float = LBLRTM_DEFAULT_ALFAL0,
    hwf3: float = LBLRTM_VOIGT_DOMAIN_HWF3,
) -> float:
    """Conservative upper bound for the LBLRTM dynamic line window.

    This is used when selecting lines before their layer-dependent Lorentz and
    Doppler widths are known. It mirrors the source controls used by
    `lblrtm_dynamic_line_cutoff_cm` and intentionally over-selects if users set
    unusual `SAMPLE`/`ALFAL0` combinations.
    """

    dv = float(grid_spacing_cm)
    if dv <= 0 or not np.isfinite(dv):
        raise ValueError("grid_spacing_cm must be positive")
    if sample <= 0:
        raise ValueError("sample must be positive")
    if alfal0 < 0:
        raise ValueError("alfal0 must be non-negative")
    if hwf3 <= 0:
        raise ValueError("hwf3 must be positive")
    if alfal0 == 0:
        return np.inf

    alfmax = 4.0 * float(sample) * dv * 0.04 / float(alfal0)
    return float(hwf3) * max(dv, alfmax)


def lblrtm_radiation_term(wavenumber_cm: float | np.ndarray, temperature_k: float) -> np.ndarray:
    """LBLRTM ``RADFN`` radiation term used by optical-depth screening."""

    wavenumber = np.asarray(wavenumber_cm, dtype=float)
    xkt = float(temperature_k) / SECOND_RADIATION_CONSTANT_CM_K
    ratio = wavenumber / xkt
    return np.where(
        ratio <= 0.01,
        0.5 * ratio * wavenumber,
        np.where(ratio <= 10.0, wavenumber * -np.expm1(-ratio) / (1.0 + np.exp(-ratio)), wavenumber),
    )


def pressure_shift_wavenumber(
    pressure_shift_cm_per_atm: np.ndarray,
    pressure_atm: float,
    temperature_k: float,
    *,
    reference_temperature_k: float = 296.0,
    convention: str = "hitran",
) -> np.ndarray:
    """Return pressure-induced line-center shift in cm-1.

    ``hitran`` uses the HITRAN text convention, ``delta * pressure``.
    ``lblrtm_density`` follows LBLRTM's internal density-ratio scaling,
    ``delta * pressure * T_ref / T`` for pressures expressed relative to 1 atm.
    """

    pressure_shift = np.asarray(pressure_shift_cm_per_atm, dtype=float)
    if convention == "hitran":
        return pressure_shift * pressure_atm
    if convention == "lblrtm_density":
        return pressure_shift * pressure_atm * (reference_temperature_k / temperature_k)
    raise ValueError(f"unknown pressure shift convention: {convention!r}")


def voigt_profile_wavenumber(
    wavenumber_grid_cm: np.ndarray,
    centers_cm: np.ndarray,
    sigma_cm: np.ndarray,
    gamma_cm: np.ndarray,
) -> np.ndarray:
    x = wavenumber_grid_cm[None, :] - centers_cm[:, None]
    z = (x + 1j * gamma_cm[:, None]) / (sigma_cm[:, None] * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_cm[:, None] * np.sqrt(2.0 * np.pi))


def voigt_profile_offset(
    offset_cm: np.ndarray | float,
    sigma_cm: np.ndarray,
    gamma_cm: np.ndarray,
) -> np.ndarray:
    """Voigt profile value at distance ``offset_cm`` from each line center."""

    offset_cm = np.asarray(offset_cm, dtype=float)
    sigma_cm = np.asarray(sigma_cm, dtype=float)
    gamma_cm = np.asarray(gamma_cm, dtype=float)
    z = (offset_cm + 1j * gamma_cm) / (sigma_cm * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_cm * np.sqrt(2.0 * np.pi))


def lblrtm_tabulated_voigt_profile_offset(
    offset_cm: np.ndarray,
    lorentz_hwhm_cm: np.ndarray,
    doppler_sigma_cm: np.ndarray,
    *,
    effective_hwhm_cm: np.ndarray | None = None,
) -> np.ndarray:
    """LBLRTM-style finite Voigt profile from tabulated subfunctions.

    This ports the shape decomposition in ``oprop_voigt.f90::voigt_init`` at
    the profile-evaluation level. It uses SciPy's Faddeeva implementation to
    build the reference Voigt table, then applies LBLRTM's 0-4, 0-16, and
    0-64 Voigt-halfwidth subfunction decomposition. The result is normalized
    like a line profile per cm-1 but finite at 64 effective Voigt half-widths.
    """

    offset = np.asarray(offset_cm, dtype=float)
    lorentz = np.asarray(lorentz_hwhm_cm, dtype=float)
    doppler_sigma = np.asarray(doppler_sigma_cm, dtype=float)
    if offset.ndim != 2:
        raise ValueError("offset_cm must have shape (n_lines, n_grid)")
    if lorentz.ndim != 1 or doppler_sigma.ndim != 1:
        raise ValueError("lorentz_hwhm_cm and doppler_sigma_cm must be one-dimensional")
    if offset.shape[0] != lorentz.size or lorentz.size != doppler_sigma.size:
        raise ValueError("line-shape inputs must have the same number of lines")

    doppler_hwhm = doppler_sigma * np.sqrt(2.0 * np.log(2.0))
    total = lorentz + doppler_hwhm
    valid = (total > 0) & np.isfinite(total)
    zeta = np.zeros(lorentz.shape, dtype=float)
    zeta[valid] = np.clip(lorentz[valid] / total[valid], 0.0, 1.0)
    alfv = (
        lblrtm_voigt_hwhm(lorentz, doppler_sigma)
        if effective_hwhm_cm is None
        else np.asarray(effective_hwhm_cm, dtype=float)
    )
    if alfv.shape != lorentz.shape:
        raise ValueError("effective_hwhm_cm must match the line arrays")

    profile = np.zeros(offset.shape, dtype=float)
    usable = valid & (alfv > 0) & np.isfinite(alfv)
    if not np.any(usable):
        return profile

    rows = np.nonzero(usable)[0]
    z = np.abs(offset[rows]) / alfv[rows, None]
    scaled = sum(_lblrtm_voigt_subfunction_values(z, zeta[rows]))

    profile[rows] = np.maximum(scaled / alfv[rows, None], 0.0)
    return profile


def lblrtm_panel_voigt_profile_wavenumber(
    wavenumber_grid_cm: np.ndarray,
    centers_cm: np.ndarray,
    sigma_cm: np.ndarray,
    gamma_cm: np.ndarray,
    *,
    effective_hwhm_cm: np.ndarray | None = None,
    pad_grid_points: int = 32,
) -> np.ndarray:
    """Evaluate Voigt profiles through LBLRTM-style R1/R2/R3 panel grids.

    The source routine `CNVFNV` deposits the 0-4, 0-16, and 0-64 effective
    half-width subfunctions into R1/R2/R3 arrays. `PANEL` then interpolates
    R3 into R2 and R2 into R1 using fixed four-point coefficients. This helper
    reproduces that numerical structure for a local spectrum grid. If the
    target grid is not uniform in wavenumber, it computes on a uniform internal
    grid and interpolates back to the requested points.
    """

    grid = np.asarray(wavenumber_grid_cm, dtype=float)
    centers = np.asarray(centers_cm, dtype=float)
    sigma = np.asarray(sigma_cm, dtype=float)
    gamma = np.asarray(gamma_cm, dtype=float)
    effective_hwhm = (
        None if effective_hwhm_cm is None else np.asarray(effective_hwhm_cm, dtype=float)
    )
    if grid.ndim != 1:
        raise ValueError("wavenumber_grid_cm must be one-dimensional")
    if centers.ndim != 1 or sigma.ndim != 1 or gamma.ndim != 1:
        raise ValueError("centers_cm, sigma_cm, and gamma_cm must be one-dimensional")
    if not (centers.size == sigma.size == gamma.size):
        raise ValueError("line-shape inputs must have the same number of lines")
    if effective_hwhm is not None and effective_hwhm.shape != centers.shape:
        raise ValueError("effective_hwhm_cm must match the line arrays")
    if grid.size == 0:
        return np.zeros((centers.size, 0), dtype=float)
    if grid.size == 1:
        return lblrtm_tabulated_voigt_profile_offset(
            grid[None, :] - centers[:, None],
            gamma,
            sigma,
            effective_hwhm_cm=effective_hwhm,
        )

    order = np.argsort(grid)
    sorted_grid = grid[order]
    spacing = np.diff(sorted_grid)
    if not np.all(spacing > 0):
        raise ValueError("wavenumber grid points must be unique")
    median_spacing = float(np.nanmedian(spacing))
    uniform = np.allclose(spacing, median_spacing, rtol=1.0e-5, atol=max(1.0e-10, 1.0e-8 * median_spacing))

    if not uniform:
        uniform_grid = np.arange(sorted_grid[0], sorted_grid[-1] + 0.5 * median_spacing, median_spacing)
        uniform_profile = _lblrtm_panel_voigt_profile_uniform(
            uniform_grid,
            centers,
            sigma,
            gamma,
            effective_hwhm_cm=effective_hwhm,
            pad_grid_points=pad_grid_points,
        )
        sorted_profile = _interp_uniform_rows(uniform_grid, uniform_profile, sorted_grid)
    else:
        sorted_profile = _lblrtm_panel_voigt_profile_uniform(
            sorted_grid,
            centers,
            sigma,
            gamma,
            effective_hwhm_cm=effective_hwhm,
            pad_grid_points=pad_grid_points,
        )

    profile = np.empty_like(sorted_profile)
    profile[:, order] = sorted_profile
    return profile


def lblrtm_panel_accumulate_wavenumber(
    wavenumber_grid_cm: np.ndarray,
    centers_cm: np.ndarray,
    sigma_cm: np.ndarray,
    gamma_cm: np.ndarray,
    line_scale: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    *,
    profile_coupling: np.ndarray | None = None,
    effective_hwhm_cm: np.ndarray | None = None,
    include_f4: bool = True,
    pad_grid_points: int = 32,
) -> np.ndarray:
    """Accumulate many LBLRTM panel profiles before grid interpolation.

    ``CNVFNV`` adds every line to shared R1/R2/R3 arrays and ``LBLF4`` does
    the same on R4.  Only then do ``XINT`` and ``PANEL`` interpolate the
    accumulated arrays.  This routine preserves that source ordering and
    adaptively uses vectorized or sparse deposition while keeping memory
    bounded for unusually large direct calls.
    """

    grid = np.asarray(wavenumber_grid_cm, dtype=float)
    centers = np.asarray(centers_cm, dtype=float)
    sigma = np.asarray(sigma_cm, dtype=float)
    gamma = np.asarray(gamma_cm, dtype=float)
    scale = np.asarray(line_scale, dtype=float)
    groups = np.asarray(group_index, dtype=int)
    coupling = (
        np.zeros(centers.shape, dtype=float)
        if profile_coupling is None
        else np.asarray(profile_coupling, dtype=float)
    )
    effective_hwhm = (
        None if effective_hwhm_cm is None else np.asarray(effective_hwhm_cm, dtype=float)
    )
    if grid.ndim != 1:
        raise ValueError("wavenumber_grid_cm must be one-dimensional")
    if any(value.ndim != 1 for value in (centers, sigma, gamma, scale, groups, coupling)):
        raise ValueError("line inputs must be one-dimensional")
    if not (centers.size == sigma.size == gamma.size == scale.size == groups.size == coupling.size):
        raise ValueError("line inputs must have the same length")
    if effective_hwhm is not None and effective_hwhm.shape != centers.shape:
        raise ValueError("effective_hwhm_cm must match the line arrays")
    if n_groups < 0:
        raise ValueError("n_groups must be non-negative")
    if grid.size == 0:
        return np.zeros((n_groups, 0), dtype=float)
    if grid.size == 1:
        profile = lblrtm_panel_voigt_profile_wavenumber(
            grid,
            centers,
            sigma,
            gamma,
            effective_hwhm_cm=effective_hwhm,
        )
        if np.any(coupling != 0):
            alfv = lblrtm_voigt_hwhm(gamma, sigma)
            profile *= 1.0 + coupling[:, None] * (grid[None, :] - centers[:, None]) / alfv[:, None]
        result = np.zeros((n_groups, 1), dtype=float)
        valid = (groups >= 0) & (groups < n_groups)
        np.add.at(result[:, 0], groups[valid], profile[valid, 0] * scale[valid])
        return result

    order = np.argsort(grid)
    sorted_grid = grid[order]
    spacing = np.diff(sorted_grid)
    if not np.all(spacing > 0):
        raise ValueError("wavenumber grid points must be unique")
    median_spacing = float(np.nanmedian(spacing))
    uniform = np.allclose(
        spacing,
        median_spacing,
        rtol=1.0e-5,
        atol=max(1.0e-10, 1.0e-8 * median_spacing),
    )
    if uniform:
        sorted_result = _lblrtm_panel_accumulate_uniform(
            sorted_grid,
            centers,
            sigma,
            gamma,
            scale,
            groups,
            n_groups,
            coupling,
            effective_hwhm_cm=effective_hwhm,
            include_f4=include_f4,
            pad_grid_points=pad_grid_points,
        )
    else:
        uniform_grid = np.arange(sorted_grid[0], sorted_grid[-1] + 0.5 * median_spacing, median_spacing)
        uniform_result = _lblrtm_panel_accumulate_uniform(
            uniform_grid,
            centers,
            sigma,
            gamma,
            scale,
            groups,
            n_groups,
            coupling,
            effective_hwhm_cm=effective_hwhm,
            include_f4=include_f4,
            pad_grid_points=pad_grid_points,
        )
        sorted_result = _interp_uniform_rows(uniform_grid, uniform_result, sorted_grid)

    result = np.empty_like(sorted_result)
    result[:, order] = sorted_result
    return result


def lblrtm_f4_profile_offset(
    offset_cm: np.ndarray,
    lorentz_hwhm_cm: np.ndarray,
    doppler_sigma_cm: np.ndarray,
    *,
    bound_cm: float = LBLRTM_F4_BOUND_CM,
    effective_hwhm_cm: np.ndarray | None = None,
) -> np.ndarray:
    """Return LBLRTM's normalized F4 line-wing/closure subfunction.

    This is a direct vectorized transcription of ``CONVF4`` for the current
    LBLRTM source path.  It supplies the fourth part of the Voigt decomposition
    that is accumulated separately from F1/F2/F3 on a grid spaced by
    ``64 * DV``.  The bundled source currently forces the CO2 chi factor to
    one, so the same expression applies to every molecule.
    """

    offset = np.asarray(offset_cm, dtype=float)
    lorentz = np.asarray(lorentz_hwhm_cm, dtype=float)
    doppler_sigma = np.asarray(doppler_sigma_cm, dtype=float)
    if offset.ndim != 2:
        raise ValueError("offset_cm must have shape (n_lines, n_grid)")
    if lorentz.ndim != 1 or doppler_sigma.ndim != 1:
        raise ValueError("lorentz_hwhm_cm and doppler_sigma_cm must be one-dimensional")
    if offset.shape[0] != lorentz.size or lorentz.size != doppler_sigma.size:
        raise ValueError("line-shape inputs must have the same number of lines")
    if bound_cm <= 0 or not np.isfinite(bound_cm):
        raise ValueError("bound_cm must be positive and finite")

    doppler_hwhm = doppler_sigma * np.sqrt(2.0 * np.log(2.0))
    total = lorentz + doppler_hwhm
    valid = (total > 0) & np.isfinite(total) & (lorentz >= 0)
    zeta = np.zeros(lorentz.shape, dtype=float)
    zeta[valid] = np.clip(lorentz[valid] / total[valid], 0.0, 1.0)
    alfv = (
        lblrtm_voigt_hwhm(lorentz, doppler_sigma)
        if effective_hwhm_cm is None
        else np.asarray(effective_hwhm_cm, dtype=float)
    )
    if alfv.shape != lorentz.shape:
        raise ValueError("effective_hwhm_cm must match the line arrays")

    profile = np.zeros(offset.shape, dtype=float)
    rows = np.nonzero(valid & (alfv > 0) & np.isfinite(alfv))[0]
    if rows.size == 0:
        return profile

    zeta_index, zeta_fraction = _lblrtm_zeta_interpolation(zeta[rows])
    a3_table, b3_table = _lblrtm_f4_coefficient_tables()
    a3 = a3_table[zeta_index] + zeta_fraction * (
        a3_table[zeta_index + 1] - a3_table[zeta_index]
    )
    b3 = b3_table[zeta_index] + zeta_fraction * (
        b3_table[zeta_index + 1] - b3_table[zeta_index]
    )

    local_offset = offset[rows]
    offset_sq = local_offset**2
    alfv_local = alfv[rows]
    lorentz_sq = lorentz[rows] ** 2
    z_sq = offset_sq / alfv_local[:, None] ** 2
    z_bound_sq = LBLRTM_VOIGT_DOMAIN_HWF3**2
    f4_at_64 = a3 + b3 * z_bound_sq
    lorentz_numerator = (
        f4_at_64
        / alfv_local
        * (lorentz_sq + alfv_local**2 * z_bound_sq)
    )
    boundary_value = lorentz_numerator / (lorentz_sq + float(bound_cm) ** 2)

    near = (a3[:, None] + b3[:, None] * z_sq) / alfv_local[:, None]
    far = lorentz_numerator[:, None] / (lorentz_sq[:, None] + offset_sq)
    values = np.where(z_sq <= z_bound_sq, near, far) - boundary_value[:, None]
    values = np.where(np.abs(local_offset) <= float(bound_cm), values, 0.0)
    profile[rows] = values
    return profile


def lblrtm_f4_peak_factor(
    lorentz_hwhm_cm: np.ndarray,
    doppler_sigma_cm: np.ndarray,
) -> np.ndarray:
    """Return the source ``A3x`` coefficient used by ``CONVF4`` screening."""

    a3, _ = lblrtm_f4_coefficients(lorentz_hwhm_cm, doppler_sigma_cm)
    return a3


def lblrtm_f4_coefficients(
    lorentz_hwhm_cm: np.ndarray,
    doppler_sigma_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interpolated source A3/B3 coefficients for F4 profiles."""

    lorentz = np.asarray(lorentz_hwhm_cm, dtype=float)
    doppler_sigma = np.asarray(doppler_sigma_cm, dtype=float)
    doppler_hwhm = doppler_sigma * np.sqrt(2.0 * np.log(2.0))
    total = lorentz + doppler_hwhm
    zeta = np.divide(lorentz, total, out=np.zeros_like(total), where=total > 0)
    zeta_index, zeta_fraction = _lblrtm_zeta_interpolation(np.clip(zeta, 0.0, 1.0))
    a3_table, b3_table = _lblrtm_f4_coefficient_tables()
    a3 = a3_table[zeta_index] + zeta_fraction * (a3_table[zeta_index + 1] - a3_table[zeta_index])
    b3 = b3_table[zeta_index] + zeta_fraction * (b3_table[zeta_index + 1] - b3_table[zeta_index])
    return a3, b3


def lblrtm_panel_interpolate_f4_wavenumber(
    wavenumber_grid_cm: np.ndarray,
    r4_grid_cm: np.ndarray,
    r4_by_group: np.ndarray,
    *,
    pad_grid_points: int = 32,
) -> np.ndarray:
    """Interpolate a completed source-order R4 field through R3/R2/R1."""

    grid = np.asarray(wavenumber_grid_cm, dtype=float)
    r4_grid = np.asarray(r4_grid_cm, dtype=float)
    r4 = np.asarray(r4_by_group, dtype=float)
    if grid.ndim != 1 or r4_grid.ndim != 1 or r4.ndim != 2:
        raise ValueError("F4 interpolation inputs must be one- or two-dimensional arrays")
    if r4.shape[1] != r4_grid.size:
        raise ValueError("r4_by_group columns must match r4_grid_cm")
    if grid.size < 2:
        return np.zeros((r4.shape[0], grid.size), dtype=float)

    order = np.argsort(grid)
    sorted_grid = grid[order]
    grid_steps = np.diff(sorted_grid)
    if not np.all(grid_steps > 0):
        raise ValueError("wavenumber grid points must be unique")
    spacing = float(np.nanmedian(grid_steps))
    uniform = np.allclose(
        grid_steps,
        spacing,
        rtol=1.0e-5,
        atol=max(1.0e-10, 1.0e-8 * spacing),
    )
    if uniform:
        # Preserve the caller's exact endpoints. np.arange with a floating
        # stop can silently drop the final panel sample.
        uniform_grid = sorted_grid
    else:
        count = int(np.ceil((sorted_grid[-1] - sorted_grid[0]) / spacing)) + 1
        uniform_grid = sorted_grid[0] + spacing * np.arange(count, dtype=float)
    pad = max(32, int(pad_grid_points))
    pad += (-pad) % 16
    fine = uniform_grid[0] + spacing * np.arange(-pad, uniform_grid.size + pad, dtype=float)
    coarse2 = fine[0] + 4.0 * spacing * np.arange(int(np.ceil(fine.size / 4.0)) + 4)
    coarse3 = fine[0] + 16.0 * spacing * np.arange(int(np.ceil(fine.size / 16.0)) + 4)
    r3 = _lblrtm_xint_rows(r4_grid, r4, coarse3)
    combined = lblrtm_panel_interpolate_r1_r2_r3(
        np.zeros((r4.shape[0], fine.size), dtype=float),
        np.zeros((r4.shape[0], coarse2.size), dtype=float),
        r3,
    )[:, pad : pad + uniform_grid.size]
    sorted_result = combined if uniform else _interp_uniform_rows(uniform_grid, combined, sorted_grid)
    result = np.empty_like(sorted_result)
    result[:, order] = sorted_result
    return result


def lblrtm_panel_interpolate_r1_r2_r3(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
) -> np.ndarray:
    """Combine R1/R2/R3 panel arrays using LBLRTM `PANEL` coefficients."""

    r1_arr = np.asarray(r1, dtype=float)
    r2_arr = np.asarray(r2, dtype=float)
    r3_arr = np.asarray(r3, dtype=float)
    return r1_arr + _lblrtm_interpolate_four_to_one(
        r2_arr + _lblrtm_interpolate_four_to_one(r3_arr, r2_arr.shape[-1]),
        r1_arr.shape[-1],
    )


@lru_cache(maxsize=1)
def _lblrtm_voigt_subfunction_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LBLRTM Voigt subfunction tables with the source decomposition."""

    nx = LBLRTM_VOIGT_TABLE_POINTS
    domains = LBLRTM_VOIGT_TABLE_DOMAINS
    zeta_grid = np.linspace(0.0, 1.0, 101)
    f_tables = [np.zeros((102, nx), dtype=float) for _ in domains]

    for zeta_index, zeta in enumerate(zeta_grid):
        if zeta < 1.0:
            doppler_hwhm = 1.0
            lorentz_hwhm = zeta / (1.0 - zeta)
            sigma = doppler_hwhm / np.sqrt(2.0 * np.log(2.0))
        else:
            doppler_hwhm = 0.0
            lorentz_hwhm = 1.0
            sigma = 1.0

        alfv = LBLRTM_AVRAT[zeta_index] * (doppler_hwhm + lorentz_hwhm)
        xfv_by_domain = []
        derivatives = []
        for domain in domains:
            z = np.linspace(0.0, domain, nx)
            if zeta < 1.0:
                offset = z * alfv
                xfv = voigt_profile_offset(offset, sigma, np.array(lorentz_hwhm)) * alfv
            else:
                xfv = (1.0 / np.pi) / (1.0 + z**2)
            dz = domain / (nx - 1)
            derivatives.append(0.5 * (xfv[-1] - xfv[-3]) / dz)
            xfv_by_domain.append(xfv)

        a_coeff = []
        b_coeff = []
        for xfv, derivative, domain in zip(xfv_by_domain, derivatives, domains, strict=True):
            b = derivative / (2.0 * domain)
            a = xfv[-2] - b * domain**2
            a_coeff.append(a)
            b_coeff.append(b)

        for domain_index, domain in enumerate(domains):
            z = np.linspace(0.0, domain, nx)
            q1 = a_coeff[0] + b_coeff[0] * z**2
            q2 = a_coeff[1] + b_coeff[1] * z**2
            q3 = a_coeff[2] + b_coeff[2] * z**2
            if domain_index == 0:
                values = xfv_by_domain[0] - q1
            elif domain_index == 1:
                values = np.where(z <= domains[0], q1 - q2, xfv_by_domain[1] - q2)
            else:
                values = np.where(z <= domains[1], q2 - q3, xfv_by_domain[2] - q3)
            values[-1] = 0.0
            f_tables[domain_index][zeta_index] = values

    for table in f_tables:
        table[101] = table[100]

    # F1/F2/F3 are default REAL arrays in oprop_voigt.f90 even though
    # voigt_init performs the coefficient construction in double precision.
    # Retaining that final single-precision quantisation is part of reproducing
    # the numerical path used by LBLRTM rather than only its analytic profile.
    return tuple(np.asarray(table, dtype=np.float32) for table in f_tables)


@lru_cache(maxsize=1)
def _lblrtm_f4_coefficient_tables() -> tuple[np.ndarray, np.ndarray]:
    """Build the double-precision a3/b3 tables retained by ``CONVF4``."""

    zeta = np.linspace(0.0, 1.0, 101)
    doppler_hwhm = np.ones(zeta.shape, dtype=float)
    lorentz_hwhm = np.divide(
        zeta,
        1.0 - zeta,
        out=np.ones(zeta.shape, dtype=float),
        where=zeta < 1.0,
    )
    sigma = doppler_hwhm / np.sqrt(2.0 * np.log(2.0))
    alfv = LBLRTM_AVRAT[:101] * (doppler_hwhm + lorentz_hwhm)

    domain = LBLRTM_VOIGT_DOMAIN_HWF3
    dz = domain / (LBLRTM_VOIGT_TABLE_POINTS - 1)
    sample_indices = np.array(
        [LBLRTM_VOIGT_TABLE_POINTS - 3, LBLRTM_VOIGT_TABLE_POINTS - 2, LBLRTM_VOIGT_TABLE_POINTS - 1],
        dtype=float,
    )
    normalized_offset = sample_indices * dz
    offset = alfv[:, None] * normalized_offset[None, :]
    scaled_profile = voigt_profile_offset(
        offset,
        sigma[:, None],
        lorentz_hwhm[:, None],
    ) * alfv[:, None]

    pure_lorentz = (1.0 / np.pi) / (1.0 + normalized_offset**2)
    scaled_profile[-1] = pure_lorentz
    derivative = 0.5 * (scaled_profile[:, 2] - scaled_profile[:, 0]) / dz
    b3 = derivative / (2.0 * domain)
    a3 = scaled_profile[:, 1] - b3 * domain**2
    return np.concatenate((a3, a3[-1:])), np.concatenate((b3, b3[-1:]))


def _lblrtm_voigt_subfunction_values(z: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeta_index, zeta_fraction = _lblrtm_zeta_interpolation(zeta)
    return tuple(
        _lblrtm_voigt_subfunction_value(
            z,
            domain_index,
            zeta_index=zeta_index,
            zeta_fraction=zeta_fraction,
        )
        for domain_index in range(3)
    )


def _lblrtm_voigt_subfunction_value(
    z: np.ndarray,
    domain_index: int,
    *,
    zeta_index: np.ndarray,
    zeta_fraction: np.ndarray,
) -> np.ndarray:
    table = _lblrtm_voigt_subfunction_tables()[domain_index]
    domain = LBLRTM_VOIGT_TABLE_DOMAINS[domain_index]
    u = np.asarray(z, dtype=float) * ((LBLRTM_VOIGT_TABLE_POINTS - 1) / domain)
    left_raw = np.floor(u).astype(int)
    keep = (left_raw >= 0) & (left_raw < LBLRTM_VOIGT_TABLE_POINTS - 1)
    left = np.clip(left_raw, 0, LBLRTM_VOIGT_TABLE_POINTS - 2)
    right = left + 1
    fraction = u - left
    row0 = zeta_index[:, None]
    row1 = row0 + 1
    zeta_weight = zeta_fraction[:, None]
    left0 = table[row0, left]
    left1 = table[row1, left]
    right0 = table[row0, right]
    right1 = table[row1, right]
    left_value = left0 + zeta_weight * (left1 - left0)
    right_value = right0 + zeta_weight * (right1 - right0)
    values = (1.0 - fraction) * left_value + fraction * right_value
    values[~keep] = 0.0
    return values


def _lblrtm_voigt_subfunction_nearest_value(
    z: np.ndarray,
    domain_index: int,
    *,
    zeta_index: np.ndarray,
    zeta_fraction: np.ndarray,
) -> np.ndarray:
    """Sample one Voigt subfunction as ``CNVFNV`` samples F1/F2/F3.

    LBLRTM linearly interpolates between adjacent zeta tables, but it does not
    interpolate in normalized distance.  ``CNVFNV`` uses
    ``IZ = ABS(ZF) + 1.5`` with one-based Fortran indexing, which is nearest
    neighbour sampling on each subfunction's 2001-point distance grid.
    """

    table = _lblrtm_voigt_subfunction_tables()[domain_index]
    domain = LBLRTM_VOIGT_TABLE_DOMAINS[domain_index]
    scaled_distance = np.asarray(z, dtype=float) * (
        (LBLRTM_VOIGT_TABLE_POINTS - 1) / domain
    )
    distance_index = np.floor(scaled_distance + 0.5).astype(int)
    keep = (distance_index >= 0) & (distance_index < LBLRTM_VOIGT_TABLE_POINTS)
    distance_index = np.clip(distance_index, 0, LBLRTM_VOIGT_TABLE_POINTS - 1)

    row0 = zeta_index[:, None]
    row1 = row0 + 1
    zeta_weight = zeta_fraction[:, None]
    value0 = table[row0, distance_index]
    value1 = table[row1, distance_index]
    values = value0 + zeta_weight * (value1 - value0)
    return np.where(keep, values, 0.0)


def _lblrtm_zeta_interpolation(zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fzeta = 100.0 * np.asarray(zeta, dtype=float)
    zeta_index = np.floor(fzeta).astype(int)
    zeta_index = np.clip(zeta_index, 0, 100)
    return zeta_index, fzeta - zeta_index


def _lblrtm_panel_voigt_profile_uniform(
    wavenumber_grid_cm: np.ndarray,
    centers_cm: np.ndarray,
    sigma_cm: np.ndarray,
    gamma_cm: np.ndarray,
    *,
    effective_hwhm_cm: np.ndarray | None,
    pad_grid_points: int,
) -> np.ndarray:
    grid = np.asarray(wavenumber_grid_cm, dtype=float)
    spacing = float(grid[1] - grid[0])
    pad = max(32, int(pad_grid_points))
    pad += (-pad) % 16
    padded_grid = grid[0] + spacing * np.arange(-pad, grid.size + pad, dtype=float)
    coarse2_grid = padded_grid[0] + 4.0 * spacing * np.arange(int(np.ceil(padded_grid.size / 4.0)) + 4)
    coarse3_grid = padded_grid[0] + 16.0 * spacing * np.arange(int(np.ceil(padded_grid.size / 16.0)) + 4)

    gamma = np.asarray(gamma_cm, dtype=float)
    sigma = np.asarray(sigma_cm, dtype=float)
    centers = np.asarray(centers_cm, dtype=float)
    doppler_hwhm = sigma * np.sqrt(2.0 * np.log(2.0))
    total = gamma + doppler_hwhm
    valid = (total > 0) & np.isfinite(total)
    zeta = np.zeros(gamma.shape, dtype=float)
    zeta[valid] = np.clip(gamma[valid] / total[valid], 0.0, 1.0)
    alfv = (
        lblrtm_voigt_hwhm(gamma, sigma)
        if effective_hwhm_cm is None
        else np.asarray(effective_hwhm_cm, dtype=float)
    )

    profile = np.zeros((centers.size, grid.size), dtype=float)
    rows = np.nonzero(valid & (alfv > 0) & np.isfinite(alfv))[0]
    if rows.size == 0:
        return profile

    zeta_index, zeta_fraction = _lblrtm_zeta_interpolation(zeta[rows])

    def subfunction_on(target_grid: np.ndarray, domain_index: int) -> np.ndarray:
        z = np.abs(target_grid[None, :] - centers[rows, None]) / alfv[rows, None]
        return (
            _lblrtm_voigt_subfunction_nearest_value(
                z,
                domain_index,
                zeta_index=zeta_index,
                zeta_fraction=zeta_fraction,
            )
            / alfv[rows, None]
        )

    r1 = subfunction_on(padded_grid, 0)
    r2 = subfunction_on(coarse2_grid, 1)
    r3 = subfunction_on(coarse3_grid, 2)

    r4_spacing = LBLRTM_F4_GRID_RATIO * spacing
    r4_start = grid[0] - 2.0 * r4_spacing
    r4_stop = grid[-1] + 2.0 * r4_spacing
    r4_grid = np.arange(r4_start, r4_stop + 0.5 * r4_spacing, r4_spacing)
    r4 = lblrtm_f4_profile_offset(
        r4_grid[None, :] - centers[rows, None],
        gamma[rows],
        sigma[rows],
        effective_hwhm_cm=alfv[rows],
    )
    r3 += _lblrtm_xint_rows(r4_grid, r4, coarse3_grid)

    combined = lblrtm_panel_interpolate_r1_r2_r3(r1, r2, r3)
    profile[rows] = combined[:, pad:pad + grid.size]
    return profile


def _lblrtm_panel_accumulate_uniform(
    wavenumber_grid_cm: np.ndarray,
    centers_cm: np.ndarray,
    sigma_cm: np.ndarray,
    gamma_cm: np.ndarray,
    line_scale: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    profile_coupling: np.ndarray,
    *,
    effective_hwhm_cm: np.ndarray | None,
    include_f4: bool,
    pad_grid_points: int,
) -> np.ndarray:
    grid = np.asarray(wavenumber_grid_cm, dtype=float)
    spacing = float(grid[1] - grid[0])
    pad = max(32, int(pad_grid_points))
    pad += (-pad) % 16
    fine = grid[0] + spacing * np.arange(-pad, grid.size + pad, dtype=float)
    coarse2 = fine[0] + 4.0 * spacing * np.arange(int(np.ceil(fine.size / 4.0)) + 4)
    coarse3 = fine[0] + 16.0 * spacing * np.arange(int(np.ceil(fine.size / 16.0)) + 4)

    centers = np.asarray(centers_cm, dtype=float)
    sigma = np.asarray(sigma_cm, dtype=float)
    gamma = np.asarray(gamma_cm, dtype=float)
    scale = np.asarray(line_scale, dtype=float)
    groups = np.asarray(group_index, dtype=int)
    coupling = np.asarray(profile_coupling, dtype=float)
    doppler_hwhm = sigma * np.sqrt(2.0 * np.log(2.0))
    total = gamma + doppler_hwhm
    valid = (
        (total > 0)
        & np.isfinite(total)
        & np.isfinite(centers)
        & np.isfinite(scale)
        & (scale != 0)
        & (groups >= 0)
        & (groups < n_groups)
    )
    zeta = np.zeros(gamma.shape, dtype=float)
    zeta[valid] = np.clip(gamma[valid] / total[valid], 0.0, 1.0)
    alfv = (
        lblrtm_voigt_hwhm(gamma, sigma)
        if effective_hwhm_cm is None
        else np.asarray(effective_hwhm_cm, dtype=float)
    )
    valid &= (alfv > 0) & np.isfinite(alfv)
    rows = np.nonzero(valid)[0]
    if rows.size == 0:
        return np.zeros((n_groups, grid.size), dtype=float)

    r1 = np.zeros((n_groups, fine.size), dtype=float)
    r2 = np.zeros((n_groups, coarse2.size), dtype=float)
    r3 = np.zeros((n_groups, coarse3.size), dtype=float)
    _lblrtm_subfunction_accumulate(fine, 0, rows, centers, alfv, zeta, scale, groups, coupling, r1)
    _lblrtm_subfunction_accumulate(coarse2, 1, rows, centers, alfv, zeta, scale, groups, coupling, r2)
    _lblrtm_subfunction_accumulate(coarse3, 2, rows, centers, alfv, zeta, scale, groups, coupling, r3)

    if include_f4:
        r4_spacing = LBLRTM_F4_GRID_RATIO * spacing
        r4_grid = np.arange(
            grid[0] - 2.0 * r4_spacing,
            grid[-1] + 2.5 * r4_spacing,
            r4_spacing,
        )
        r4_lines = lblrtm_f4_profile_offset(
            r4_grid[None, :] - centers[rows, None],
            gamma[rows],
            sigma[rows],
            effective_hwhm_cm=alfv[rows],
        )
        if np.any(coupling[rows] != 0):
            r4_lines *= 1.0 + coupling[rows, None] * (
                r4_grid[None, :] - centers[rows, None]
            ) / alfv[rows, None]
        r4 = np.zeros((n_groups, r4_grid.size), dtype=float)
        for group in np.unique(groups[rows]):
            keep = groups[rows] == group
            r4[group] = np.sum(r4_lines[keep] * scale[rows][keep, None], axis=0)
        r3 += _lblrtm_xint_rows(r4_grid, r4, coarse3)

    combined = lblrtm_panel_interpolate_r1_r2_r3(r1, r2, r3)
    return combined[:, pad : pad + grid.size]


def _lblrtm_subfunction_accumulate(
    target_grid: np.ndarray,
    domain_index: int,
    rows: np.ndarray,
    centers: np.ndarray,
    alfv: np.ndarray,
    zeta: np.ndarray,
    scale: np.ndarray,
    groups: np.ndarray,
    coupling: np.ndarray,
    output: np.ndarray,
) -> None:
    # Chunked production calls normally contain at most 128 lines, where a
    # dense vectorized table lookup is substantially faster than np.add.at.
    # Keep sparse deposition for unusually large direct calls so memory use is
    # bounded without altering the numerical result.
    if rows.size * target_grid.size <= 2_000_000:
        domain = LBLRTM_VOIGT_TABLE_DOMAINS[domain_index]
        signed_offset = target_grid[None, :] - centers[rows, None]
        normalized = np.abs(signed_offset) / alfv[rows, None]
        zeta_index, zeta_fraction = _lblrtm_zeta_interpolation(zeta[rows])
        values = _lblrtm_voigt_subfunction_nearest_value(
            normalized,
            domain_index,
            zeta_index=zeta_index,
            zeta_fraction=zeta_fraction,
        ) / alfv[rows, None]
        values = np.where(normalized <= domain, values, 0.0)
        if np.any(coupling[rows] != 0):
            values *= 1.0 + coupling[rows, None] * signed_offset / alfv[rows, None]
        values *= scale[rows, None]
        for group in np.unique(groups[rows]):
            output[group] += np.sum(values[groups[rows] == group], axis=0)
        return

    _lblrtm_sparse_subfunction_deposit(
        target_grid,
        domain_index,
        rows,
        centers,
        alfv,
        zeta,
        scale,
        groups,
        coupling,
        output,
    )


def _lblrtm_sparse_subfunction_deposit(
    target_grid: np.ndarray,
    domain_index: int,
    rows: np.ndarray,
    centers: np.ndarray,
    alfv: np.ndarray,
    zeta: np.ndarray,
    scale: np.ndarray,
    groups: np.ndarray,
    coupling: np.ndarray,
    output: np.ndarray,
) -> None:
    domain = LBLRTM_VOIGT_TABLE_DOMAINS[domain_index]
    cutoff = domain * alfv[rows]
    left = np.searchsorted(target_grid, centers[rows] - cutoff, side="left")
    right = np.searchsorted(target_grid, centers[rows] + cutoff, side="right")
    counts = np.maximum(right - left, 0)
    keep_rows = counts > 0
    if not np.any(keep_rows):
        return

    selected = rows[keep_rows]
    counts = counts[keep_rows]
    total_points = int(np.sum(counts))
    repeated = np.repeat(selected, counts)
    starts = np.repeat(left[keep_rows], counts)
    offsets = np.arange(total_points, dtype=int) - np.repeat(np.cumsum(counts) - counts, counts)
    grid_index = starts + offsets
    signed_offset = target_grid[grid_index] - centers[repeated]
    normalized = np.abs(signed_offset) / alfv[repeated]
    distance_index = np.floor(
        normalized * (LBLRTM_VOIGT_TABLE_POINTS - 1) / domain + 0.5
    ).astype(int)
    inside = (distance_index >= 0) & (distance_index < LBLRTM_VOIGT_TABLE_POINTS)
    if not np.all(inside):
        repeated = repeated[inside]
        grid_index = grid_index[inside]
        signed_offset = signed_offset[inside]
        distance_index = distance_index[inside]
    if repeated.size == 0:
        return

    zeta_position = 100.0 * zeta[repeated]
    zeta_index = np.clip(np.floor(zeta_position).astype(int), 0, 100)
    zeta_fraction = zeta_position - zeta_index
    table = _lblrtm_voigt_subfunction_tables()[domain_index]
    value0 = table[zeta_index, distance_index]
    value1 = table[zeta_index + 1, distance_index]
    values = (value0 + zeta_fraction * (value1 - value0)) / alfv[repeated]
    coupled = coupling[repeated] != 0
    if np.any(coupled):
        values[coupled] *= 1.0 + coupling[repeated[coupled]] * (
            signed_offset[coupled] / alfv[repeated[coupled]]
        )
    values *= scale[repeated]
    np.add.at(output, (groups[repeated], grid_index), values)


def _lblrtm_xint_rows(x_source: np.ndarray, y_rows: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    """Vectorized four-point interpolation from LBLRTM ``XINT``."""

    source = np.asarray(x_source, dtype=float)
    values = np.asarray(y_rows, dtype=float)
    target = np.asarray(x_target, dtype=float)
    if source.ndim != 1 or target.ndim != 1 or values.ndim != 2:
        raise ValueError("XINT inputs must be one-dimensional grids and a two-dimensional value array")
    if values.shape[1] != source.size or source.size < 4:
        raise ValueError("XINT requires at least four source samples per row")

    spacing = float(source[1] - source[0])
    position = (target - source[0]) / spacing
    base = np.floor(position).astype(int)
    fraction = position - base
    valid = (base >= 1) & (base + 2 < source.size)
    safe = np.clip(base, 1, source.size - 3)

    p = fraction
    c = (3.0 - 2.0 * p) * p**2
    b = 0.5 * p * (1.0 - p)
    b1 = b * (1.0 - p)
    b2 = b * p
    interpolated = (
        -values[:, safe - 1] * b1[None, :]
        + values[:, safe] * (1.0 - c + b2)[None, :]
        + values[:, safe + 1] * (c + b1)[None, :]
        - values[:, safe + 2] * b2[None, :]
    )
    return np.where(valid[None, :], interpolated, 0.0)


def _interp_uniform_rows(x_grid: np.ndarray, y_rows: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    """Linear interpolation for many rows sharing one regular x grid."""

    x_grid = np.asarray(x_grid, dtype=float)
    y_rows = np.asarray(y_rows, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    if x_grid.size < 2:
        return np.zeros(y_rows.shape[:-1] + x_target.shape, dtype=float)

    spacing = x_grid[1] - x_grid[0]
    position = (x_target - x_grid[0]) / spacing
    left = np.floor(position).astype(int)
    fraction = position - left
    valid = (left >= 0) & (left + 1 < x_grid.size)
    left = np.clip(left, 0, x_grid.size - 2)
    interpolated = (1.0 - fraction)[None, :] * y_rows[:, left] + fraction[None, :] * y_rows[:, left + 1]
    return np.where(valid[None, :], interpolated, 0.0)


def _lblrtm_interpolate_four_to_one(coarse: np.ndarray, fine_size: int) -> np.ndarray:
    coarse_arr = np.asarray(coarse, dtype=float)
    if fine_size < 0:
        raise ValueError("fine_size must be non-negative")
    result = np.zeros(coarse_arr.shape[:-1] + (fine_size,), dtype=float)
    if fine_size == 0 or coarse_arr.shape[-1] == 0:
        return result

    padded = np.pad(coarse_arr, [(0, 0)] * (coarse_arr.ndim - 1) + [(1, 2)], mode="constant")
    base = np.arange((fine_size + 3) // 4)
    coarse_index = np.minimum(base, coarse_arr.shape[-1] - 1)
    x00 = -7.0 / 128.0
    x01 = 105.0 / 128.0
    x02 = 35.0 / 128.0
    x03 = -5.0 / 128.0
    x10 = -1.0 / 16.0
    x11 = 9.0 / 16.0

    center = coarse_index + 1
    pm1 = padded[..., center - 1]
    p0 = padded[..., center]
    p1 = padded[..., center + 1]
    p2 = padded[..., center + 2]

    def assign(offset: int, values: np.ndarray) -> None:
        if offset >= fine_size:
            return
        target = result[..., offset::4]
        target[...] = values[..., : target.shape[-1]]

    assign(0, p0)
    assign(1, x00 * pm1 + x01 * p0 + x02 * p1 + x03 * p2)
    assign(2, x10 * (pm1 + p2) + x11 * (p0 + p1))
    assign(3, x03 * pm1 + x02 * p0 + x01 * p1 + x00 * p2)
    return result
