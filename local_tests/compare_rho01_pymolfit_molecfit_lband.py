from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from pymolfit import (
    AtmosphereLayer,
    AtmosphereProfile,
    CO2ContinuumAbsorption,
    FitConfig,
    H2OContinuumAbsorption,
    HitranCIATable,
    HitranLineAbsorption,
    IsotopologueMetadata,
    LBLRTMCO2Continuum,
    LBLRTMH2OContinuum,
    LineList,
    MTCKDH2OContinuum,
    N2ContinuumAbsorption,
    PairCIAAbsorption,
    PartitionTable,
    RayleighScatteringAbsorption,
    Spectrum,
    fit_telluric_segments,
    fit_tellurics,
    lblrtm_dynamic_max_line_cutoff_cm,
    line_wing_effective_cutoff_cm,
)


PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = PROJECT / "local_tests" / "external_absorption"
RHO01 = Path(
    os.environ.get(
        "PYMOLFIT_RHO01_PRODUCTS",
        PROJECT / "local_tests" / "data" / "rho01",
    )
)
MOLECFIT = RHO01 / "molecfit"
OUTPUT = Path(
    os.environ.get(
        "PYMOLFIT_COMPARISON_OUTPUT",
        PROJECT / "local_tests" / "rho01_molecfit_vs_pymolfit_lband",
    )
)

LINE_LIST = Path(
    os.environ.get(
        "PYMOLFIT_LINE_LIST",
        EXTERNAL / "aer_lband_h2o_co2_co_ch4_o2_strength1e-32.ecsv",
    )
)
ISO_METADATA = EXTERNAL / "hitran_iso_metadata_lband.ecsv"
HITRAN_Q_DIR = EXTERNAL / "hitran_q"
H2O_CONTINUUM = EXTERNAL / "absco-ref_wv-mt-ckd.nc"
CIA_TABLES = {
    "CO2-CO2_CIA": EXTERNAL / "CO2-CO2_2024.cia",
    "O2-O2_CIA": EXTERNAL / "O2-O2_2024.cia",
    "O2-N2_CIA": EXTERNAL / "O2-N2_2024.cia",
    "N2-N2_CIA": EXTERNAL / "N2-N2_2021.cia",
}
EXTRA_CIA_TABLES = {
    "CO2-H2O_CIA": EXTERNAL / "CO2-H2O_2024.cia",
    "O2-air_CIA": EXTERNAL / "O2-air_2024.cia",
    "N2-air_CIA": EXTERNAL / "N2-air_2018.cia",
}

CHIPS = tuple(range(1, 19))


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in {"", "none", "null", "off"}:
        return None
    return float(value)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _bounds_env(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if value is None:
        return default
    parts = [part for part in re.split(r"[:,\s]+", value.strip()) if part]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain two bounds")
    return float(parts[0]), float(parts[1])


def _molecfit_header_float(keyword: str, default: float) -> float:
    path = MOLECFIT / "molecfit_model" / "ATM_PROFILE_COMBINED.fits"
    if not path.exists():
        return default
    try:
        with fits.open(path) as hdul:
            return float(hdul[0].header.get(keyword, default))
    except Exception:
        return default


def _molecfit_header_bool(keyword: str, default: bool) -> bool:
    path = MOLECFIT / "molecfit_model" / "ATM_PROFILE_COMBINED.fits"
    if not path.exists():
        return default
    try:
        with fits.open(path) as hdul:
            value = hdul[0].header.get(keyword, default)
    except Exception:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "none", ""}


LINE_CUTOFF_CM = _optional_float_env("PYMOLFIT_LINE_CUTOFF_CM")
LINE_WING_MODE = os.environ.get("PYMOLFIT_LINE_WING_MODE", "lblrtm_panel")
LBLRTM_SAMPLE = float(
    os.environ.get(
        "PYMOLFIT_LBLRTM_SAMPLE",
        _molecfit_header_float("ESO DRS MF PARAM LBLRTM_SAMPLE", 4.0),
    )
)
LBLRTM_ALFAL0 = float(
    os.environ.get(
        "PYMOLFIT_LBLRTM_ALFAL0",
        _molecfit_header_float("ESO DRS MF PARAM LBLRTM_ALFAL0", 0.04),
    )
)
LBLRTM_HWF3 = float(os.environ.get("PYMOLFIT_LBLRTM_HWF3", "64.0"))
LINE_TAPER_CM = float(os.environ.get("PYMOLFIT_LINE_TAPER_CM", "0.0" if LINE_CUTOFF_CM is None else "2.0"))
SUBTRACT_CUTOFF_PROFILE = os.environ.get("PYMOLFIT_SUBTRACT_CUTOFF_PROFILE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
USE_SOURCE_BOX_WIDTH = _bool_env("PYMOLFIT_SOURCE_BOX_WIDTH", False)
USE_VARIABLE_LSF = _bool_env(
    "PYMOLFIT_VARIABLE_LSF",
    _molecfit_header_bool("ESO DRS MF PARAM VARKERN", False),
)
USE_MOLECFIT_VOIGT_LSF = _bool_env(
    "PYMOLFIT_MOLECFIT_VOIGT_LSF",
    _molecfit_header_bool("ESO DRS MF PARAM KERNMODE", False),
)
LSF_KERNEL_WIDTH_FWHM = float(
    os.environ.get(
        "PYMOLFIT_LSF_KERNEL_WIDTH_FWHM",
        _molecfit_header_float("ESO DRS MF PARAM KERNFAC", 3.0),
    )
)
FIT_LSF_BOX_WIDTH = _bool_env("PYMOLFIT_FIT_LSF_BOX_WIDTH", False)
LSF_BOX_WIDTH_BOUNDS = _bounds_env("PYMOLFIT_LSF_BOX_WIDTH_BOUNDS", (0.0, 10.0))
FIT_LSF_LORENTZ_FWHM = _bool_env("PYMOLFIT_FIT_LSF_LORENTZ_FWHM", False)
LSF_LORENTZ_FWHM_BOUNDS = _bounds_env("PYMOLFIT_LSF_LORENTZ_FWHM_BOUNDS", (0.0, 10.0))
HIGH_RESOLUTION_GRID = _bool_env("PYMOLFIT_HIGH_RESOLUTION_GRID", True)
HIGH_RESOLUTION_OVERSAMPLING = float(os.environ.get("PYMOLFIT_HIGH_RESOLUTION_OVERSAMPLING", "5.0"))
HIGH_RESOLUTION_MARGIN_PIXELS = float(os.environ.get("PYMOLFIT_HIGH_RESOLUTION_MARGIN_PIXELS", "2.0"))
HIGH_RESOLUTION_REBIN_MODE = os.environ.get("PYMOLFIT_HIGH_RESOLUTION_REBIN_MODE", "molecfit_overlap")
CONTINUUM_ORDER_ENV = os.environ.get("PYMOLFIT_CONTINUUM_ORDER")
LINE_MARGIN_MICRON = float(
    os.environ.get(
        "PYMOLFIT_LINE_MARGIN_MICRON",
        "0.03" if LBLRTM_ALFAL0 == 0 and LINE_CUTOFF_CM is None else "0.01",
    )
)
USE_EXTRA_CIA = _bool_env("PYMOLFIT_EXTRA_CIA", False)
USE_RAYLEIGH = _bool_env("PYMOLFIT_RAYLEIGH", False)
USE_N2_CONTINUUM = _bool_env("PYMOLFIT_N2_CONTINUUM", True)
USE_LBLRTM_H2O_CONTINUUM = _bool_env("PYMOLFIT_LBLRTM_H2O_CONTINUUM", True)
USE_LBLRTM_CO2_CONTINUUM = _bool_env("PYMOLFIT_LBLRTM_CO2_CONTINUUM", True)
ATMOSPHERE_SOURCE = os.environ.get("PYMOLFIT_ATMOSPHERE_SOURCE", "mipas_gdas").strip().lower()
WRITE_PLOTS = _bool_env("PYMOLFIT_WRITE_PLOTS", True)
USE_MOLECFIT_WEIGHTS = _bool_env("PYMOLFIT_USE_MOLECFIT_WEIGHTS", True)
FIT_LOSS = os.environ.get("PYMOLFIT_FIT_LOSS", "linear")
FIT_F_SCALE = float(os.environ.get("PYMOLFIT_F_SCALE", "1.0"))
SOLVE_CONTINUUM_LINEAR = _bool_env("PYMOLFIT_SOLVE_CONTINUUM_LINEAR", True)
MAX_NFEV_ENV = os.environ.get("PYMOLFIT_MAX_NFEV", "80")
MAX_NFEV = None if MAX_NFEV_ENV.strip().lower() in {"", "none", "null", "off"} else int(MAX_NFEV_ENV)
FIT_SEGMENT_WAVELENGTH_SHIFTS = _bool_env("PYMOLFIT_FIT_SEGMENT_WAVELENGTH_SHIFTS", True)
FIT_SEGMENT_WAVELENGTH_POLYNOMIAL_ENV = os.environ.get("PYMOLFIT_FIT_SEGMENT_WAVELENGTH_POLYNOMIAL")
SEGMENT_WAVELENGTH_POLYNOMIAL_ORDER_ENV = os.environ.get("PYMOLFIT_SEGMENT_WAVELENGTH_POLYNOMIAL_ORDER")
WAVELENGTH_SHIFT_BOUND = float(os.environ.get("PYMOLFIT_WAVELENGTH_SHIFT_BOUND", "5.0e-4"))
USE_MOLECFIT_INITIAL_SCALES = _bool_env("PYMOLFIT_USE_MOLECFIT_INITIAL_SCALES", False)


def uncertainty_from_molecfit_weight(weight: np.ndarray) -> np.ndarray | None:
    """Represent Molecfit residuals `(flux - model) * weight` as uncertainties."""

    if not USE_MOLECFIT_WEIGHTS:
        return None
    weight = np.asarray(weight, dtype=float)
    uncertainty = np.full(weight.shape, np.inf, dtype=float)
    positive = np.isfinite(weight) & (weight > 0)
    uncertainty[positive] = 1.0 / weight[positive]
    return uncertainty


def line_margin_micron(wavelength: np.ndarray) -> float:
    effective_cutoff = line_wing_effective_cutoff_cm(LINE_WING_MODE, LINE_CUTOFF_CM)
    if LINE_WING_MODE.strip().lower() in {"lblrtm_dynamic", "lblrtm_table", "lblrtm_panel"}:
        wavenumber = np.sort(1.0e4 / np.asarray(wavelength, dtype=float))
        spacing = np.diff(wavenumber[np.isfinite(wavenumber)])
        spacing = spacing[spacing > 0]
        if spacing.size:
            dynamic_cutoff = lblrtm_dynamic_max_line_cutoff_cm(
                float(np.nanmedian(spacing)),
                sample=LBLRTM_SAMPLE,
                alfal0=LBLRTM_ALFAL0,
                hwf3=LBLRTM_HWF3,
            )
            if LINE_CUTOFF_CM is not None:
                dynamic_cutoff = min(dynamic_cutoff, float(LINE_CUTOFF_CM))
            effective_cutoff = dynamic_cutoff if effective_cutoff is None else max(effective_cutoff, dynamic_cutoff)
    if effective_cutoff is not None and not np.isfinite(effective_cutoff):
        return LINE_MARGIN_MICRON
    if effective_cutoff is None:
        return LINE_MARGIN_MICRON
    return max(LINE_MARGIN_MICRON, 1.1 * float(np.nanmax(wavelength)) ** 2 * effective_cutoff / 1.0e4)


def atmosphere_from_molecfit_profile(path: Path, *, airmass: float) -> AtmosphereProfile:
    with fits.open(path) as hdul:
        data = hdul[1].data
        height_km = np.asarray(data["HGT"], dtype=float)
        pressure_atm = np.asarray(data["PRE"], dtype=float) / 1013.25
        temperature_k = np.asarray(data["TEM"], dtype=float)

        if height_km.size == 1:
            thickness_m = np.array([10_000.0], dtype=float)
        else:
            edges = np.empty(height_km.size + 1, dtype=float)
            edges[1:-1] = 0.5 * (height_km[:-1] + height_km[1:])
            edges[0] = height_km[0] - 0.5 * (height_km[1] - height_km[0])
            edges[-1] = height_km[-1] + 0.5 * (height_km[-1] - height_km[-2])
            thickness_m = np.diff(edges) * 1000.0 * airmass

        layers = []
        for index in range(height_km.size):
            mixing = {
                "H2O": float(data["H2O"][index]) * 1.0e-6,
                "CO2": float(data["CO2"][index]) * 1.0e-6,
                "CO": float(data["CO"][index]) * 1.0e-6,
                "CH4": float(data["CH4"][index]) * 1.0e-6,
                "O2": float(data["O2"][index]) * 1.0e-6,
            }
            mixing["N2"] = max(0.0, 1.0 - sum(mixing.values()))
            layers.append(
                AtmosphereLayer(
                    pressure_atm=float(pressure_atm[index]),
                    temperature_k=float(temperature_k[index]),
                    path_length_m=float(thickness_m[index]),
                    mixing_ratios=mixing,
                )
            )
    return AtmosphereProfile(tuple(layers))


def comparison_atmosphere(
    *,
    source: str,
    airmass: float,
    h2o_col_mm: float,
    reference_wavenumber_cm: float,
) -> tuple[AtmosphereProfile, float]:
    normalized = source.strip().lower().replace("-", "_")
    if normalized in {"molecfit", "molecfit_profile", "profile"}:
        atmosphere = atmosphere_from_molecfit_profile(
            MOLECFIT / "molecfit_model" / "ATM_PROFILE_COMBINED.fits",
            airmass=1.0,
        )
        return atmosphere.with_pwv_mm(h2o_col_mm), airmass

    with fits.open(MOLECFIT / "molecfit_input" / "SCIENCE_A.fits") as hdul:
        header = hdul[0].header.copy()
        observatory_altitude_m = float(
            header.get("ESO TEL GEOELEV", header.get("HIERARCH ESO TEL GEOELEV", 2648.0))
        )
    if normalized in {"mipas_gdas", "gdas", "self_contained"}:
        atmosphere = AtmosphereProfile.from_fits_header_mipas_gdas(
            header,
            airmass=airmass,
            observatory_altitude_m=observatory_altitude_m,
            gdas_mode="auto",
            reference_wavenumber_cm=reference_wavenumber_cm,
        )
        return atmosphere, 1.0
    if normalized in {"header", "fits_header", "header_slant"}:
        atmosphere = AtmosphereProfile.from_fits_header(
            header,
            airmass=airmass,
            observatory_altitude_m=observatory_altitude_m,
            pwv_mm=h2o_col_mm,
            n_layers=48,
        )
        return atmosphere, 1.0
    if normalized in {"header_vertical", "fits_header_vertical"}:
        atmosphere = AtmosphereProfile.from_fits_header(
            header,
            airmass=1.0,
            observatory_altitude_m=observatory_altitude_m,
            pwv_mm=h2o_col_mm,
            n_layers=48,
        )
        return atmosphere, airmass
    raise ValueError(
        "PYMOLFIT_ATMOSPHERE_SOURCE must be mipas_gdas, molecfit, header_slant, or header_vertical"
    )


def representative_airmass(path: Path) -> float:
    with fits.open(path) as hdul:
        header = hdul[0].header
        start = header.get("ESO TEL AIRM START", header.get("HIERARCH ESO TEL AIRM START", 1.0))
        end = header.get("ESO TEL AIRM END", header.get("HIERARCH ESO TEL AIRM END", start))
    return 0.5 * (float(start) + float(end))


def best_fit_parameter(path: Path, name: str) -> float:
    with fits.open(path) as hdul:
        data = hdul[1].data
        for row in data:
            parameter = row["parameter"]
            if not isinstance(parameter, str):
                parameter = parameter.decode()
            if parameter.strip() == name:
                return float(row["value"])
    raise KeyError(f"best-fit parameter {name!r} not found")


def molecfit_species_initial_scales(path: Path) -> dict[str, float]:
    if not USE_MOLECFIT_INITIAL_SCALES:
        return {}
    scales = {}
    with fits.open(path) as hdul:
        data = hdul[1].data
        for row in data:
            parameter = row["parameter"]
            if not isinstance(parameter, str):
                parameter = parameter.decode()
            parameter = parameter.strip()
            prefix = "rel_mol_col_"
            if parameter.startswith(prefix):
                scales[parameter.removeprefix(prefix)] = float(row["value"])
    return scales


def molecfit_continuum_order(path: Path) -> int:
    if CONTINUUM_ORDER_ENV is not None:
        return int(CONTINUUM_ORDER_ENV)
    with fits.open(path) as hdul:
        table = hdul["WAVE_INCLUDE"].data if "WAVE_INCLUDE" in hdul else hdul[2].data
        orders = np.asarray(table["CONT_POLY_ORDER"], dtype=int)
    orders = orders[np.isfinite(orders)]
    if orders.size == 0:
        return 1
    return int(np.nanmax(orders))


def molecfit_wavelength_polynomial_order(path: Path) -> int:
    if SEGMENT_WAVELENGTH_POLYNOMIAL_ORDER_ENV is not None:
        return int(SEGMENT_WAVELENGTH_POLYNOMIAL_ORDER_ENV)
    coefficient_numbers = []
    pattern = re.compile(r"^chip\s+\d+,\s*coef\s+(\d+)$", re.IGNORECASE)
    with fits.open(path) as hdul:
        for row in hdul[1].data:
            parameter = row["parameter"]
            if not isinstance(parameter, str):
                parameter = parameter.decode()
            match = pattern.match(parameter.strip())
            if match:
                coefficient_numbers.append(int(match.group(1)))
    if not coefficient_numbers:
        return 0
    return int(max(coefficient_numbers))


def segment_wavelength_fit_options(path: Path) -> tuple[bool, bool, int]:
    if not FIT_SEGMENT_WAVELENGTH_SHIFTS:
        return False, False, 0
    inferred_order = molecfit_wavelength_polynomial_order(path)
    if FIT_SEGMENT_WAVELENGTH_POLYNOMIAL_ENV is None:
        use_polynomial = False
    else:
        use_polynomial = FIT_SEGMENT_WAVELENGTH_POLYNOMIAL_ENV.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if use_polynomial:
        return False, True, max(1, inferred_order)
    return True, False, 0


def molecfit_reference_wavelength(path: Path) -> float:
    with fits.open(path) as hdul:
        wave_include = hdul["WAVE_INCLUDE"].data
        return 0.5 * (float(wave_include["LOWER_LIMIT"][0]) + float(wave_include["UPPER_LIMIT"][-1]))


def molecfit_box_width_pixels(path: Path) -> float:
    box_factor = best_fit_parameter(path, "boxfwhm_pix")
    if not USE_SOURCE_BOX_WIDTH:
        return box_factor
    with fits.open(path) as hdul:
        header = hdul[0].header
        slit_width = float(header.get("ESO DRS MF PARAM SLIT_WIDTH_VALUE", 0.0))
        pixel_scale = float(header.get("ESO DRS MF PARAM PIX_SCALE_VALUE", 0.0))
    if slit_width > 0 and pixel_scale > 0:
        return box_factor * slit_width / pixel_scale
    return box_factor


def build_components(
    line_list: LineList,
    wavelength: np.ndarray,
    partition_table: PartitionTable,
    h2o_continuum: MTCKDH2OContinuum,
    co2_continuum: LBLRTMCO2Continuum | None,
    cia_tables: dict[str, HitranCIATable],
):
    selected_lines = line_list.select_range(
        float(np.nanmin(wavelength)),
        float(np.nanmax(wavelength)),
        margin=line_margin_micron(wavelength),
    )
    components = [
        HitranLineAbsorption(
            selected_lines,
            chunk_size=0,
            partition_table=partition_table,
            line_wing_mode=LINE_WING_MODE,
            line_cutoff_cm=LINE_CUTOFF_CM,
            subtract_cutoff_profile=SUBTRACT_CUTOFF_PROFILE,
            line_taper_cm=LINE_TAPER_CM,
            lblrtm_sample=LBLRTM_SAMPLE,
            lblrtm_alfal0=LBLRTM_ALFAL0,
            lblrtm_hwf3=LBLRTM_HWF3,
        ),
        H2OContinuumAbsorption(h2o_continuum),
    ]
    if co2_continuum is not None:
        components.append(CO2ContinuumAbsorption(co2_continuum))
    for name, cia in cia_tables.items():
        components.append(PairCIAAbsorption(cia, basis_name=name))
    return tuple(components), selected_lines


def fit_chip(chip: int, data, line_list, atmosphere, partition_table, h2o_continuum, co2_continuum, cia_tables):
    keep = data["chip"] == chip
    wavelength = np.asarray(data["mlambda"][keep], dtype=float)
    flux = np.asarray(data["flux"][keep], dtype=float)
    molecfit_transmission = np.asarray(data["mtrans"][keep], dtype=float)
    weight = np.asarray(data["weight"][keep], dtype=float)
    valid = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0) & (weight > 0)

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    molecfit_transmission = molecfit_transmission[order]
    weight = weight[order]
    valid = valid[order]
    uncertainty = uncertainty_from_molecfit_weight(weight)

    components, selected_lines = build_components(
        line_list,
        wavelength,
        partition_table,
        h2o_continuum,
        co2_continuum,
        cia_tables,
    )
    result = fit_tellurics(
        Spectrum(
            wavelength=wavelength,
            flux=flux,
            uncertainty=uncertainty,
            wavelength_unit="micron",
        ),
        line_list=selected_lines,
        config=FitConfig(
            atmosphere=atmosphere,
            continuum_order=molecfit_continuum_order(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits"),
            components=components,
            partition_table=partition_table,
            line_wing_mode=LINE_WING_MODE,
            line_cutoff_cm=LINE_CUTOFF_CM,
            subtract_cutoff_profile=SUBTRACT_CUTOFF_PROFILE,
            line_taper_cm=LINE_TAPER_CM,
            lblrtm_sample=LBLRTM_SAMPLE,
            lblrtm_alfal0=LBLRTM_ALFAL0,
            lblrtm_hwf3=LBLRTM_HWF3,
            line_margin_micron=LINE_MARGIN_MICRON,
            initial_species_scales=molecfit_species_initial_scales(
                MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits"
            ),
            scale_bounds=(1.0e-5, 1.0e5),
            solve_continuum_linear=SOLVE_CONTINUUM_LINEAR,
            lsf_sigma_pixels=1.0,
            fit_wavelength_shift=False,
            wavelength_shift_bounds=(-WAVELENGTH_SHIFT_BOUND, WAVELENGTH_SHIFT_BOUND),
            fit_lsf_sigma=False,
            fit_lsf_box_width=FIT_LSF_BOX_WIDTH,
            lsf_box_width_bounds=LSF_BOX_WIDTH_BOUNDS,
            fit_lsf_lorentz_fwhm=FIT_LSF_LORENTZ_FWHM,
            lsf_lorentz_fwhm_bounds=LSF_LORENTZ_FWHM_BOUNDS,
            loss=FIT_LOSS,
            f_scale=FIT_F_SCALE,
            max_nfev=MAX_NFEV,
            min_transmission=0.03,
        ),
        fit_mask=valid,
    )

    finite = valid & np.isfinite(result.transmission) & np.isfinite(molecfit_transmission)
    diff = result.transmission[finite] - molecfit_transmission[finite]
    summary = {
        "chip": chip,
        "n_pixels": int(wavelength.size),
        "n_fit_pixels": int(np.count_nonzero(valid)),
            "n_selected_lines": int(selected_lines.wavelength.size),
            "line_wing_mode": LINE_WING_MODE,
            "line_cutoff_cm": np.nan if LINE_CUTOFF_CM is None else float(LINE_CUTOFF_CM),
            "molecfit_weighted_residuals": bool(USE_MOLECFIT_WEIGHTS),
            "fit_loss": FIT_LOSS,
            "fit_f_scale": float(FIT_F_SCALE),
            "solve_continuum_linear": bool(SOLVE_CONTINUUM_LINEAR),
            "max_nfev": -1 if MAX_NFEV is None else int(MAX_NFEV),
            "source_box_width": bool(USE_SOURCE_BOX_WIDTH),
            "variable_lsf": bool(USE_VARIABLE_LSF),
            "lsf_kernel_width_fwhm": float(LSF_KERNEL_WIDTH_FWHM),
            "fit_lsf_box_width": bool(FIT_LSF_BOX_WIDTH),
            "fit_lsf_lorentz_fwhm": bool(FIT_LSF_LORENTZ_FWHM),
            "fit_segment_wavelength_shifts": False,
            "fit_segment_wavelength_polynomial": False,
            "segment_wavelength_polynomial_order": 0,
            "wavelength_shift_bound": float(WAVELENGTH_SHIFT_BOUND),
            "molecfit_initial_scales": bool(USE_MOLECFIT_INITIAL_SCALES),
            "success": bool(result.success),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "pymolfit_min_transmission": float(np.nanmin(result.transmission)),
        "pymolfit_median_transmission": float(np.nanmedian(result.transmission)),
        "molecfit_min_transmission": float(np.nanmin(molecfit_transmission)),
        "molecfit_median_transmission": float(np.nanmedian(molecfit_transmission)),
        "transmission_rms_difference": float(np.sqrt(np.nanmean(diff**2))),
        "transmission_median_abs_difference": float(np.nanmedian(np.abs(diff))),
        "wavelength_shift_micron": float(result.wavelength_shift),
        "wavelength_shift_kms_at_chip_center": float(
            299_792.458 * result.wavelength_shift / np.nanmedian(wavelength)
        ),
        "lsf_sigma_pixels": float(result.lsf_sigma_pixels),
        "lsf_box_width_pixels": float(result.lsf_box_width_pixels),
        "lsf_lorentz_fwhm_pixels": float(result.lsf_lorentz_fwhm_pixels),
        **{f"scale_{key}": value for key, value in result.species_scales.items()},
    }

    table = result.to_table()
    table["molecfit_transmission"] = molecfit_transmission
    table["transmission_difference"] = result.transmission - molecfit_transmission
    table["optical_depth_difference"] = (
        -np.log(np.clip(result.transmission, 1.0e-12, np.inf))
        + np.log(np.clip(molecfit_transmission, 1.0e-12, np.inf))
    )
    table["fit_mask"] = valid
    table.meta.update(summary)
    return result, table, summary


def plot_chip(chip: int, table: Table, output: Path) -> None:
    wavelength = np.asarray(table["wavelength"], dtype=float)
    flux = np.asarray(table["flux"], dtype=float)
    continuum = np.asarray(table["continuum"], dtype=float)
    gen_trans = np.asarray(table["transmission"], dtype=float)
    mol_trans = np.asarray(table["molecfit_transmission"], dtype=float)
    corrected = np.asarray(table["corrected_flux"], dtype=float)

    norm_flux = flux / continuum
    molecfit_corrected_norm = norm_flux / np.clip(mol_trans, 0.03, np.inf)
    pymolfit_corrected_norm = corrected / continuum

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(wavelength, norm_flux, color="0.35", lw=0.8, label="Observed / PyMolFit continuum")
    axes[0].plot(wavelength, gen_trans, color="tab:orange", lw=1.0, label="PyMolFit transmission")
    axes[0].plot(wavelength, mol_trans, color="tab:blue", lw=1.0, alpha=0.8, label="Molecfit transmission")
    axes[0].set_ylabel("Normalized flux")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(wavelength, mol_trans, color="tab:blue", lw=1.0, label="Molecfit")
    axes[1].plot(wavelength, gen_trans, color="tab:orange", lw=1.0, label="PyMolFit")
    axes[1].set_ylabel("Transmission")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(wavelength, molecfit_corrected_norm, color="tab:blue", lw=0.8, label="Molecfit corrected")
    axes[2].plot(wavelength, pymolfit_corrected_norm, color="tab:orange", lw=0.8, label="PyMolFit corrected")
    axes[2].set_xlabel("Wavelength [micron]")
    axes[2].set_ylabel("Corrected / continuum")
    axes[2].legend(loc="best", fontsize=8)

    fig.suptitle(f"rho01 CNC L-band chip {chip}: Molecfit vs PyMolFit")
    fig.savefig(output / f"chip_{chip:02d}_molecfit_vs_pymolfit.png", dpi=180)
    plt.close(fig)


def plot_summary(summary_table: Table, output: Path) -> None:
    chip = np.asarray(summary_table["chip"], dtype=int)
    rms = np.asarray(summary_table["transmission_rms_difference"], dtype=float)
    median_abs = np.asarray(summary_table["transmission_median_abs_difference"], dtype=float)
    n_lines = np.asarray(summary_table["n_selected_lines"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(chip, rms, marker="o", lw=1.5, label="RMS difference")
    axes[0].plot(chip, median_abs, marker="s", lw=1.5, label="Median absolute difference")
    axes[0].set_ylabel("Transmission difference")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].bar(chip, n_lines, color="0.35")
    axes[1].set_xlabel("Chip")
    axes[1].set_ylabel("Selected lines")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(chip)

    mode = str(summary_table["line_wing_mode"][0]) if "line_wing_mode" in summary_table.colnames else "unknown"
    fig.suptitle(f"rho01 CNC L-band Molecfit vs PyMolFit summary ({mode})")
    fig.savefig(output / "summary_transmission_differences.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with fits.open(MOLECFIT / "molecfit_model" / "MOLECFIT_DATA.fits") as hdul:
        data = hdul[1].data.copy()

    airmass = representative_airmass(MOLECFIT / "molecfit_input" / "SCIENCE_A.fits")
    initial_species_scales = molecfit_species_initial_scales(
        MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits"
    )
    h2o_col_mm = best_fit_parameter(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits", "h2o_col_mm")
    boxfwhm_pix = molecfit_box_width_pixels(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits")
    lorentzfwhm_pix = best_fit_parameter(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits", "lorentzfwhm")
    lsf_reference = molecfit_reference_wavelength(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits")
    continuum_order = molecfit_continuum_order(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits")
    (
        fit_segment_wavelength_shifts,
        fit_segment_wavelength_polynomial,
        segment_wavelength_polynomial_order,
    ) = segment_wavelength_fit_options(MOLECFIT / "molecfit_model" / "BEST_FIT_PARAMETERS.fits")
    atmosphere, fit_airmass = comparison_atmosphere(
        source=ATMOSPHERE_SOURCE,
        airmass=airmass,
        h2o_col_mm=h2o_col_mm,
        reference_wavenumber_cm=float(
            np.nanmedian(1.0e4 / np.asarray(data["lambda"], dtype=float))
        ),
    )
    isotopologues = IsotopologueMetadata.from_table(ISO_METADATA)
    partition_table = PartitionTable.from_lblrtm_package_data()
    line_list = LineList.from_table(LINE_LIST).with_isotopologue_metadata(isotopologues)
    h2o_continuum = (
        LBLRTMH2OContinuum.from_package_data()
        if USE_LBLRTM_H2O_CONTINUUM
        else MTCKDH2OContinuum.from_netcdf(H2O_CONTINUUM)
    )
    co2_continuum = LBLRTMCO2Continuum.from_package_data() if USE_LBLRTM_CO2_CONTINUUM else None
    cia_paths = dict(CIA_TABLES)
    if USE_EXTRA_CIA:
        cia_paths.update(EXTRA_CIA_TABLES)
    if USE_N2_CONTINUUM:
        # LBLRTM's N2 continuum already contains the N2-N2 and N2-O2
        # collision partners. Loading the corresponding HITRAN CIA tables at
        # the same time would count those continua twice.
        cia_paths = {
            name: path
            for name, path in cia_paths.items()
            if name not in {"N2-N2_CIA", "O2-N2_CIA", "N2-air_CIA"}
        }
    cia_tables = {name: HitranCIATable.from_hitran_cia(path) for name, path in cia_paths.items()}

    components = [
        HitranLineAbsorption(
            line_list,
            chunk_size=0,
            partition_table=partition_table,
            line_wing_mode=LINE_WING_MODE,
            line_cutoff_cm=LINE_CUTOFF_CM,
            subtract_cutoff_profile=SUBTRACT_CUTOFF_PROFILE,
            line_taper_cm=LINE_TAPER_CM,
            lblrtm_sample=LBLRTM_SAMPLE,
            lblrtm_alfal0=LBLRTM_ALFAL0,
            lblrtm_hwf3=LBLRTM_HWF3,
        ),
        H2OContinuumAbsorption(h2o_continuum),
    ]
    if co2_continuum is not None:
        components.append(CO2ContinuumAbsorption(co2_continuum))
    for name, cia in cia_tables.items():
        components.append(PairCIAAbsorption(cia, basis_name=name))
    if USE_RAYLEIGH:
        components.append(RayleighScatteringAbsorption())
    if USE_N2_CONTINUUM:
        components.append(N2ContinuumAbsorption())

    fixed_component_scales = {name: 1.0 for name in cia_paths}
    if USE_RAYLEIGH:
        fixed_component_scales["Rayleigh"] = 1.0
    if USE_N2_CONTINUUM:
        fixed_component_scales["N2_continuum"] = 1.0

    spectra = []
    fit_masks = []
    molecfit_transmissions = []
    molecfit_model_fluxes = []
    molecfit_continua = []
    continuum_priors = []
    selected_line_counts = []
    for chip in CHIPS:
        keep = data["chip"] == chip
        wavelength = np.asarray(data["mlambda"][keep], dtype=float)
        flux = np.asarray(data["flux"][keep], dtype=float)
        molecfit_transmission = np.asarray(data["mtrans"][keep], dtype=float)
        molecfit_continuum = np.asarray(data["mscal"][keep], dtype=float)
        molecfit_model_flux = np.asarray(data["mflux"][keep], dtype=float)
        weight = np.asarray(data["weight"][keep], dtype=float)
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        flux = flux[order]
        molecfit_transmission = molecfit_transmission[order]
        molecfit_continuum = molecfit_continuum[order]
        molecfit_model_flux = molecfit_model_flux[order]
        weight = weight[order]
        valid = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0) & (weight > 0)
        spectra.append(
            Spectrum(
                wavelength=wavelength,
                flux=flux,
                uncertainty=uncertainty_from_molecfit_weight(weight),
                wavelength_unit="micron",
            )
        )
        fit_masks.append(valid)
        molecfit_transmissions.append(molecfit_transmission)
        molecfit_model_fluxes.append(molecfit_model_flux)
        molecfit_continua.append(molecfit_continuum)
        continuum_priors.append(molecfit_continuum)
        selected_line_counts.append(
            line_list.select_range(
                float(np.nanmin(wavelength)),
                float(np.nanmax(wavelength)),
                margin=line_margin_micron(wavelength),
            ).wavelength.size
        )

    print("Fitting all chips with shared species scales...")
    global_result = fit_telluric_segments(
        spectra,
        line_list=line_list,
        config=FitConfig(
            atmosphere=atmosphere,
            airmass=fit_airmass,
            continuum_order=continuum_order,
            continuum_prior_weight=1.0,
            continuum_prior_fractional_sigma=0.03,
            components=tuple(components),
            partition_table=partition_table,
            line_wing_mode=LINE_WING_MODE,
            line_cutoff_cm=LINE_CUTOFF_CM,
            subtract_cutoff_profile=SUBTRACT_CUTOFF_PROFILE,
            line_taper_cm=LINE_TAPER_CM,
            lblrtm_sample=LBLRTM_SAMPLE,
            lblrtm_alfal0=LBLRTM_ALFAL0,
            lblrtm_hwf3=LBLRTM_HWF3,
            line_margin_micron=LINE_MARGIN_MICRON,
            fixed_species_scales=fixed_component_scales,
            initial_species_scales=initial_species_scales,
            scale_bounds=(1.0e-5, 1.0e5),
            solve_continuum_linear=SOLVE_CONTINUUM_LINEAR,
            lsf_sigma_pixels=0.0,
            lsf_box_width_pixels=boxfwhm_pix,
            lsf_lorentz_fwhm_pixels=lorentzfwhm_pix,
            lsf_variable_width=USE_VARIABLE_LSF,
            lsf_reference_wavelength_micron=lsf_reference,
            lsf_kernel_width_fwhm=LSF_KERNEL_WIDTH_FWHM,
            lsf_molecfit_voigt=USE_MOLECFIT_VOIGT_LSF,
            high_resolution_grid=HIGH_RESOLUTION_GRID,
            high_resolution_oversampling=HIGH_RESOLUTION_OVERSAMPLING,
            high_resolution_margin_pixels=HIGH_RESOLUTION_MARGIN_PIXELS,
            high_resolution_rebin_mode=HIGH_RESOLUTION_REBIN_MODE,
            fit_wavelength_shift=False,
            fit_segment_wavelength_shifts=fit_segment_wavelength_shifts,
            fit_segment_wavelength_polynomial=fit_segment_wavelength_polynomial,
            segment_wavelength_polynomial_order=segment_wavelength_polynomial_order,
            wavelength_shift_bounds=(-WAVELENGTH_SHIFT_BOUND, WAVELENGTH_SHIFT_BOUND),
            fit_lsf_sigma=False,
            fit_lsf_box_width=FIT_LSF_BOX_WIDTH,
            lsf_box_width_bounds=LSF_BOX_WIDTH_BOUNDS,
            fit_lsf_lorentz_fwhm=FIT_LSF_LORENTZ_FWHM,
            lsf_lorentz_fwhm_bounds=LSF_LORENTZ_FWHM_BOUNDS,
            loss=FIT_LOSS,
            f_scale=FIT_F_SCALE,
            max_nfev=MAX_NFEV,
            min_transmission=0.03,
        ),
        fit_masks=fit_masks,
        continuum_priors=continuum_priors,
    )
    print(
        f"  success={global_result.success}, nfev={global_result.nfev}, "
        f"scales={global_result.species_scales}"
    )

    summaries = []
    for chip, result, molecfit_transmission, molecfit_model_flux, molecfit_continuum, n_lines in zip(
        CHIPS,
        global_result.segment_results,
        molecfit_transmissions,
        molecfit_model_fluxes,
        molecfit_continua,
        selected_line_counts,
        strict=True,
    ):
        table = result.to_table()
        transmission = np.asarray(result.transmission, dtype=float)
        finite = np.isfinite(transmission) & np.isfinite(molecfit_transmission)
        diff = transmission[finite] - molecfit_transmission[finite]
        flux = np.asarray(result.spectrum.flux, dtype=float)
        uncertainty = np.asarray(result.spectrum.uncertainty, dtype=float)
        fit_mask = np.asarray(fit_masks[chip - 1], dtype=bool)
        objective_mask = (
            fit_mask
            & np.isfinite(flux)
            & np.isfinite(uncertainty)
            & (uncertainty > 0)
            & np.isfinite(result.model_flux)
            & np.isfinite(molecfit_model_flux)
        )
        gen_objective = float(
            np.sum(((flux[objective_mask] - result.model_flux[objective_mask]) / uncertainty[objective_mask]) ** 2)
        )
        molecfit_objective = float(
            np.sum(((flux[objective_mask] - molecfit_model_flux[objective_mask]) / uncertainty[objective_mask]) ** 2)
        )
        reliable = (
            objective_mask
            & (transmission > 0.2)
            & (np.asarray(molecfit_transmission, dtype=float) > 0.2)
            & np.isfinite(result.continuum)
            & (result.continuum != 0)
            & np.isfinite(molecfit_continuum)
            & (molecfit_continuum != 0)
        )
        gen_corrected_relative = flux / np.clip(transmission, 1.0e-12, np.inf) / result.continuum
        molecfit_corrected_relative = (
            flux / np.clip(molecfit_transmission, 1.0e-12, np.inf) / molecfit_continuum
        )
        wavelength = np.asarray(result.spectrum.wavelength, dtype=float)
        shape_mask = reliable & np.isfinite(wavelength)
        if np.count_nonzero(shape_mask) > continuum_order + 2:
            shape_wavelength = wavelength[shape_mask]
            x_shape = (shape_wavelength - np.mean(shape_wavelength)) / np.ptp(shape_wavelength)
            log_ratio = np.log(np.clip(transmission[shape_mask], 1.0e-12, np.inf)) - np.log(
                np.clip(molecfit_transmission[shape_mask], 1.0e-12, np.inf)
            )
            smooth_log_ratio = np.polyval(
                np.polyfit(x_shape, log_ratio, continuum_order), x_shape
            )
            adjusted_transmission = transmission[shape_mask] / np.exp(smooth_log_ratio)
            shape_rms = float(
                np.sqrt(np.mean((adjusted_transmission - molecfit_transmission[shape_mask]) ** 2))
            )
        else:
            shape_rms = np.nan
        summary = {
            "chip": chip,
            "n_pixels": int(result.spectrum.wavelength.size),
            "n_fit_pixels": int(np.count_nonzero(fit_masks[chip - 1])),
            "n_selected_lines": int(n_lines),
            "line_wing_mode": LINE_WING_MODE,
            "line_cutoff_cm": np.nan if LINE_CUTOFF_CM is None else float(LINE_CUTOFF_CM),
            "line_margin_micron": float(LINE_MARGIN_MICRON),
            "lblrtm_sample": float(LBLRTM_SAMPLE),
            "lblrtm_alfal0": float(LBLRTM_ALFAL0),
            "lblrtm_hwf3": float(LBLRTM_HWF3),
            "atmosphere_source": ATMOSPHERE_SOURCE,
            "fit_airmass": float(fit_airmass),
            "lsf_molecfit_voigt": bool(USE_MOLECFIT_VOIGT_LSF),
            "extra_cia": bool(USE_EXTRA_CIA),
            "rayleigh": bool(USE_RAYLEIGH),
            "n2_continuum": bool(USE_N2_CONTINUUM),
            "lblrtm_h2o_continuum": bool(USE_LBLRTM_H2O_CONTINUUM),
            "lblrtm_co2_continuum": bool(USE_LBLRTM_CO2_CONTINUUM),
            "molecfit_weighted_residuals": bool(USE_MOLECFIT_WEIGHTS),
            "fit_loss": FIT_LOSS,
            "fit_f_scale": float(FIT_F_SCALE),
            "solve_continuum_linear": bool(SOLVE_CONTINUUM_LINEAR),
            "max_nfev": -1 if MAX_NFEV is None else int(MAX_NFEV),
            "high_resolution_grid": bool(HIGH_RESOLUTION_GRID),
            "high_resolution_oversampling": float(HIGH_RESOLUTION_OVERSAMPLING),
            "high_resolution_rebin_mode": HIGH_RESOLUTION_REBIN_MODE,
            "continuum_order": int(continuum_order),
            "source_box_width": bool(USE_SOURCE_BOX_WIDTH),
            "variable_lsf": bool(USE_VARIABLE_LSF),
            "lsf_kernel_width_fwhm": float(LSF_KERNEL_WIDTH_FWHM),
            "lsf_box_width_pixels": float(boxfwhm_pix),
            "lsf_lorentz_fwhm_pixels": float(lorentzfwhm_pix),
            "fit_lsf_box_width": bool(FIT_LSF_BOX_WIDTH),
            "fit_lsf_lorentz_fwhm": bool(FIT_LSF_LORENTZ_FWHM),
            "fit_segment_wavelength_shifts": bool(fit_segment_wavelength_shifts),
            "fit_segment_wavelength_polynomial": bool(fit_segment_wavelength_polynomial),
            "segment_wavelength_polynomial_order": int(segment_wavelength_polynomial_order),
            "wavelength_shift_bound": float(WAVELENGTH_SHIFT_BOUND),
            "molecfit_initial_scales": bool(USE_MOLECFIT_INITIAL_SCALES),
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "pymolfit_min_transmission": float(np.nanmin(transmission)),
            "pymolfit_median_transmission": float(np.nanmedian(transmission)),
            "molecfit_min_transmission": float(np.nanmin(molecfit_transmission)),
            "molecfit_median_transmission": float(np.nanmedian(molecfit_transmission)),
            "transmission_rms_difference": float(np.sqrt(np.nanmean(diff**2))),
            "transmission_median_abs_difference": float(np.nanmedian(np.abs(diff))),
            "continuum_invariant_shape_rms": shape_rms,
            "pymolfit_weighted_objective": gen_objective,
            "molecfit_weighted_objective": molecfit_objective,
            "weighted_objective_ratio": (
                gen_objective / molecfit_objective if molecfit_objective > 0 else np.nan
            ),
            "pymolfit_corrected_scatter": float(
                np.nanstd(gen_corrected_relative[reliable] - 1.0)
            ),
            "molecfit_corrected_scatter": float(
                np.nanstd(molecfit_corrected_relative[reliable] - 1.0)
            ),
            "wavelength_shift_micron": float(result.wavelength_shift),
            "wavelength_shift_kms_at_chip_center": float(
                299_792.458 * result.wavelength_shift / np.nanmedian(result.spectrum.wavelength)
            ),
            "lsf_sigma_pixels": float(result.lsf_sigma_pixels),
            "fitted_lsf_box_width_pixels": float(result.lsf_box_width_pixels),
            "fitted_lsf_lorentz_fwhm_pixels": float(result.lsf_lorentz_fwhm_pixels),
            **{
                f"wavelength_coef_{index}": float(value)
                for index, value in enumerate(result.wavelength_coefficients)
            },
            **{f"scale_{key}": value for key, value in result.species_scales.items()},
        }
        finite_flux = np.isfinite(result.model_flux) & np.isfinite(molecfit_model_flux)
        finite_continuum = np.isfinite(result.continuum) & np.isfinite(molecfit_continuum)
        if np.any(finite_flux):
            summary["model_flux_rms_difference"] = float(
                np.sqrt(np.nanmean((result.model_flux[finite_flux] - molecfit_model_flux[finite_flux]) ** 2))
            )
        else:
            summary["model_flux_rms_difference"] = np.nan
        if np.any(finite_continuum):
            summary["continuum_rms_difference"] = float(
                np.sqrt(np.nanmean((result.continuum[finite_continuum] - molecfit_continuum[finite_continuum]) ** 2))
            )
        else:
            summary["continuum_rms_difference"] = np.nan
        table["molecfit_transmission"] = molecfit_transmission
        table["molecfit_model_flux"] = molecfit_model_flux
        table["molecfit_continuum"] = molecfit_continuum
        table["model_flux_difference"] = result.model_flux - molecfit_model_flux
        table["continuum_difference"] = result.continuum - molecfit_continuum
        table["transmission_difference"] = transmission - molecfit_transmission
        table["optical_depth_difference"] = (
            -np.log(np.clip(transmission, 1.0e-12, np.inf))
            + np.log(np.clip(molecfit_transmission, 1.0e-12, np.inf))
        )
        table["fit_mask"] = fit_masks[chip - 1]
        table.meta.update(summary)
        table.write(OUTPUT / f"chip_{chip:02d}_comparison.ecsv", format="ascii.ecsv", overwrite=True)
        if WRITE_PLOTS:
            plot_chip(chip, table, OUTPUT)
        summaries.append(summary)
        print(
            f"chip {chip:02d}: rms={summary['transmission_rms_difference']:.4g}, "
            f"median_abs={summary['transmission_median_abs_difference']:.4g}, "
            f"lines={summary['n_selected_lines']}"
        )

    summary_table = Table(rows=summaries)
    summary_table.write(OUTPUT / "summary.ecsv", format="ascii.ecsv", overwrite=True)
    summary_table.write(OUTPUT / "summary.csv", format="ascii.csv", overwrite=True)
    if WRITE_PLOTS:
        plot_summary(summary_table, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
