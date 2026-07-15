from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.io import netcdf_file

from .atmosphere import AtmosphereProfile
from .physics import SECOND_RADIATION_CONSTANT_CM_K

LBLRTM_RAYLEIGH_MIN_WAVENUMBER_CM = 820.0
LBLRTM_RAYLEIGH_LOSCHMIDT_SCALED = 2.68675e-1
LBLRTM_CONTNM_LOSCHMIDT_CM3 = 2.68675e19
LBLRTM_N2_ROT_MIN_WAVENUMBER_CM = -10.0
LBLRTM_N2_ROT_MAX_WAVENUMBER_CM = 350.0
LBLRTM_N2_ROT_SPACING_CM = 5.0
LBLRTM_H2O_CONTINUUM_DATA = "lblrtm_v12_11_h2o_continuum.npz"
LBLRTM_CO2_CONTINUUM_DATA = "lblrtm_v12_11_co2_continuum.npz"
LBLRTM_N2_FUNDAMENTAL_DATA = "lblrtm_v12_11_n2_fundamental.npz"
LBLRTM_O2_CONTINUUM_DATA = "lblrtm_v12_11_o2_continuum.npz"
LBLRTM_CO2_BANDHEAD_T_EFF_K = 246.0
LBLRTM_CO2_BANDHEAD_TDEP = np.array(
    [
        1.44e-01, 3.61e-01, 5.71e-01, 7.63e-01, 8.95e-01,
        9.33e-01, 8.75e-01, 7.30e-01, 5.47e-01, 3.79e-01,
        2.55e-01, 1.78e-01, 1.34e-01, 1.07e-01, 9.06e-02,
        7.83e-02, 6.83e-02, 6.00e-02, 5.30e-02, 4.72e-02,
        4.24e-02, 3.83e-02, 3.50e-02, 3.23e-02, 3.01e-02,
    ],
    dtype=float,
)
LBLRTM_N2_ROT_T296 = np.array(
    [
        0.4303e-06, 0.4850e-06, 0.4979e-06, 0.4850e-06, 0.4303e-06,
        0.3715e-06, 0.3292e-06, 0.3086e-06, 0.2920e-06, 0.2813e-06,
        0.2804e-06, 0.2738e-06, 0.2726e-06, 0.2724e-06, 0.2635e-06,
        0.2621e-06, 0.2547e-06, 0.2428e-06, 0.2371e-06, 0.2228e-06,
        0.2100e-06, 0.1991e-06, 0.1822e-06, 0.1697e-06, 0.1555e-06,
        0.1398e-06, 0.1281e-06, 0.1138e-06, 0.1012e-06, 0.9078e-07,
        0.7879e-07, 0.6944e-07, 0.6084e-07, 0.5207e-07, 0.4540e-07,
        0.3897e-07, 0.3313e-07, 0.2852e-07, 0.2413e-07, 0.2045e-07,
        0.1737e-07, 0.1458e-07, 0.1231e-07, 0.1031e-07, 0.8586e-08,
        0.7162e-08, 0.5963e-08, 0.4999e-08, 0.4226e-08, 0.3607e-08,
        0.3090e-08, 0.2669e-08, 0.2325e-08, 0.2024e-08, 0.1783e-08,
        0.1574e-08, 0.1387e-08, 0.1236e-08, 0.1098e-08, 0.9777e-09,
        0.8765e-09, 0.7833e-09, 0.7022e-09, 0.6317e-09, 0.5650e-09,
        0.5100e-09, 0.4572e-09, 0.4115e-09, 0.3721e-09, 0.3339e-09,
        0.3005e-09, 0.2715e-09, 0.2428e-09,
    ],
    dtype=float,
)
LBLRTM_N2_ROT_T220 = np.array(
    [
        0.4946e-06, 0.5756e-06, 0.5964e-06, 0.5756e-06, 0.4946e-06,
        0.4145e-06, 0.3641e-06, 0.3482e-06, 0.3340e-06, 0.3252e-06,
        0.3299e-06, 0.3206e-06, 0.3184e-06, 0.3167e-06, 0.2994e-06,
        0.2943e-06, 0.2794e-06, 0.2582e-06, 0.2468e-06, 0.2237e-06,
        0.2038e-06, 0.1873e-06, 0.1641e-06, 0.1474e-06, 0.1297e-06,
        0.1114e-06, 0.9813e-07, 0.8309e-07, 0.7059e-07, 0.6068e-07,
        0.5008e-07, 0.4221e-07, 0.3537e-07, 0.2885e-07, 0.2407e-07,
        0.1977e-07, 0.1605e-07, 0.1313e-07, 0.1057e-07, 0.8482e-08,
        0.6844e-08, 0.5595e-08, 0.4616e-08, 0.3854e-08, 0.3257e-08,
        0.2757e-08, 0.2372e-08, 0.2039e-08, 0.1767e-08, 0.1548e-08,
        0.1346e-08, 0.1181e-08, 0.1043e-08, 0.9110e-09, 0.8103e-09,
        0.7189e-09, 0.6314e-09, 0.5635e-09, 0.4976e-09, 0.4401e-09,
        0.3926e-09, 0.3477e-09, 0.3085e-09, 0.2745e-09, 0.2416e-09,
        0.2155e-09, 0.1895e-09, 0.1678e-09, 0.1493e-09, 0.1310e-09,
        0.1154e-09, 0.1019e-09, 0.8855e-10,
    ],
    dtype=float,
)
LBLRTM_N2_ROT_SF296 = np.array(
    [
        1.3534, 1.3517, 1.3508, 1.3517, 1.3534, 1.3558, 1.3584, 1.3607,
        1.3623, 1.3632, 1.3634, 1.3632, 1.3627, 1.3620, 1.3612, 1.3605,
        1.3597, 1.3590, 1.3585, 1.3582, 1.3579, 1.3577, 1.3577, 1.3580,
        1.3586, 1.3594, 1.3604, 1.3617, 1.3633, 1.3653, 1.3677, 1.3706,
        1.3742, 1.3780, 1.3822, 1.3868, 1.3923, 1.3989, 1.4062, 1.4138,
        1.4216, 1.4298, 1.4388, 1.4491, 1.4604, 1.4718, 1.4829, 1.4930,
        1.5028, 1.5138, 1.5265, 1.5392, 1.5499, 1.5577, 1.5639, 1.5714,
        1.5816, 1.5920, 1.6003, 1.6051, 1.6072, 1.6097, 1.6157, 1.6157,
        1.6157, 1.6157, 1.6157, 1.6157, 1.6157, 1.6157, 1.6157, 1.6157,
        1.6157,
    ],
    dtype=float,
)
LBLRTM_N2_ROT_SF220 = np.array(
    [
        1.3536, 1.3515, 1.3502, 1.3515, 1.3536, 1.3565, 1.3592, 1.3612,
        1.3623, 1.3626, 1.3623, 1.3616, 1.3609, 1.3600, 1.3591, 1.3583,
        1.3576, 1.3571, 1.3571, 1.3572, 1.3574, 1.3578, 1.3585, 1.3597,
        1.3616, 1.3640, 1.3666, 1.3698, 1.3734, 1.3776, 1.3828, 1.3894,
        1.3969, 1.4049, 1.4127, 1.4204, 1.4302, 1.4427, 1.4562, 1.4687,
        1.4798, 1.4894, 1.5000, 1.5142, 1.5299, 1.5441, 1.5555, 1.5615,
        1.5645, 1.5730, 1.5880, 1.6028, 1.6121, 1.6133, 1.6094, 1.6117,
        1.6244, 1.6389, 1.6485, 1.6513, 1.6468, 1.6438, 1.6523, 1.6523,
        1.6523, 1.6523, 1.6523, 1.6523, 1.6523, 1.6523, 1.6523, 1.6523,
        1.6523,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class MTCKDH2OContinuum:
    """MT_CKD water-vapor continuum coefficients.

    The coefficient file used by LBLRTM/MT_CKD stores reference self and
    foreign H2O continuum coefficients on a regular wavenumber grid. This
    class applies the same density scaling, self-continuum temperature
    exponent, radiation term, and cubic interpolation used by the reference
    MT_CKD H2O routine.
    """

    wavenumber_cm: np.ndarray
    self_absco_ref: np.ndarray
    foreign_absco_ref: np.ndarray
    foreign_closure_absco_ref: np.ndarray
    self_temperature_exponent: np.ndarray
    reference_pressure_mbar: float = 1013.0
    reference_temperature_k: float = 296.0
    title: str = ""

    def __post_init__(self) -> None:
        arrays = {
            "wavenumber_cm": np.asarray(self.wavenumber_cm, dtype=float),
            "self_absco_ref": np.asarray(self.self_absco_ref, dtype=float),
            "foreign_absco_ref": np.asarray(self.foreign_absco_ref, dtype=float),
            "foreign_closure_absco_ref": np.asarray(self.foreign_closure_absco_ref, dtype=float),
            "self_temperature_exponent": np.asarray(self.self_temperature_exponent, dtype=float),
        }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("all continuum coefficient arrays must have the same shape")
        if arrays["wavenumber_cm"].ndim != 1:
            raise ValueError("continuum coefficient arrays must be one-dimensional")
        if arrays["wavenumber_cm"].size < 4:
            raise ValueError("at least four continuum grid points are required for cubic interpolation")
        spacing = np.diff(arrays["wavenumber_cm"])
        if not np.all(spacing > 0):
            raise ValueError("continuum wavenumber grid must be strictly increasing")
        if not np.allclose(spacing, spacing[0], rtol=1.0e-8, atol=1.0e-10):
            raise ValueError("continuum wavenumber grid must be regular")
        if self.reference_pressure_mbar <= 0:
            raise ValueError("reference_pressure_mbar must be positive")
        if self.reference_temperature_k <= 0:
            raise ValueError("reference_temperature_k must be positive")

        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> "MTCKDH2OContinuum":
        """Read an AER MT_CKD H2O coefficient netCDF file."""

        path = Path(path)
        with netcdf_file(path, mode="r", mmap=False) as dataset:
            variables = dataset.variables
            required = (
                "wavenumbers",
                "self_absco_ref",
                "for_absco_ref",
                "for_closure_absco_ref",
                "self_texp",
                "ref_press",
                "ref_temp",
            )
            missing = [name for name in required if name not in variables]
            if missing:
                raise ValueError(f"MT_CKD file is missing variables: {', '.join(missing)}")

            title = _decode_attr(dataset._attributes.get("Title", b""))
            return cls(
                wavenumber_cm=np.asarray(variables["wavenumbers"].data, dtype=float),
                self_absco_ref=np.asarray(variables["self_absco_ref"].data, dtype=float),
                foreign_absco_ref=np.asarray(variables["for_absco_ref"].data, dtype=float),
                foreign_closure_absco_ref=np.asarray(variables["for_closure_absco_ref"].data, dtype=float),
                self_temperature_exponent=np.asarray(variables["self_texp"].data, dtype=float),
                reference_pressure_mbar=float(np.asarray(variables["ref_press"].data).item()),
                reference_temperature_k=float(np.asarray(variables["ref_temp"].data).item()),
                title=title,
            )

    @property
    def spacing_cm(self) -> float:
        return float(self.wavenumber_cm[1] - self.wavenumber_cm[0])

    def absorption_coefficients(
        self,
        wavenumber_cm: np.ndarray,
        *,
        pressure_mbar: float,
        temperature_k: float,
        h2o_vmr: float,
        include_radiation_term: bool = True,
        use_foreign_closure: bool = False,
        _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return self and foreign H2O continuum absorption coefficients.

        Returned coefficients are in cm2/molecule after density and collision
        partner scaling. If ``include_radiation_term`` is true, they can be
        multiplied by the H2O column amount in molecules/cm2 to get optical
        depth.
        """

        if pressure_mbar <= 0:
            raise ValueError("pressure_mbar must be positive")
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if h2o_vmr < 0:
            raise ValueError("h2o_vmr must be non-negative")

        target = np.asarray(wavenumber_cm, dtype=float)
        density_ratio = (pressure_mbar / self.reference_pressure_mbar) * (
            self.reference_temperature_k / temperature_k
        )

        temperature_factor = (self.reference_temperature_k / temperature_k) ** self.self_temperature_exponent
        self_coeff_grid = self.self_absco_ref * temperature_factor * h2o_vmr * density_ratio
        foreign_ref = self.foreign_closure_absco_ref if use_foreign_closure else self.foreign_absco_ref
        foreign_coeff_grid = foreign_ref * max(0.0, 1.0 - h2o_vmr) * density_ratio

        if include_radiation_term:
            radiation = radiation_term_cm(self.wavenumber_cm, temperature_k)
            self_coeff_grid = self_coeff_grid * radiation
            foreign_coeff_grid = foreign_coeff_grid * radiation

        interpolation_plan = _interpolation_plan or _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        return (
            interpolation_plan.apply(self_coeff_grid),
            interpolation_plan.apply(foreign_coeff_grid),
        )

    def optical_depth(
        self,
        wavenumber_cm: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        include_radiation_term: bool = True,
        use_foreign_closure: bool = False,
    ) -> np.ndarray:
        """Return total H2O continuum optical depth for an atmosphere."""

        target = np.asarray(wavenumber_cm, dtype=float)
        tau = np.zeros(target.shape, dtype=float)
        interpolation_plan = _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        for layer in atmosphere.layers:
            h2o_vmr = float(layer.mixing_ratios.get("H2O", 0.0))
            h2o_column = layer.column_density_cm2("H2O")
            if h2o_vmr <= 0 or h2o_column <= 0:
                continue
            self_coeff, foreign_coeff = self.absorption_coefficients(
                target,
                pressure_mbar=layer.pressure_atm * 1013.25,
                temperature_k=layer.temperature_k,
                h2o_vmr=h2o_vmr,
                include_radiation_term=include_radiation_term,
                use_foreign_closure=use_foreign_closure,
                _interpolation_plan=interpolation_plan,
            )
            tau += (self_coeff + foreign_coeff) * h2o_column
        return tau


@dataclass(frozen=True)
class LBLRTMH2OContinuum:
    """LBLRTM 12.11 / MT_CKD 3.5 H2O continuum from `contnm.f90`.

    This is the source-backed counterpart to `MTCKDH2OContinuum.from_netcdf`.
    The packaged coefficients are the self 296 K, self 260 K, and foreign
    296 K block-data tables used by the Molecfit-bundled LBLRTM.
    """

    wavenumber_cm: np.ndarray
    self_296: np.ndarray
    self_260: np.ndarray
    foreign_296: np.ndarray
    xfac_rhu_index: np.ndarray
    xfac_rhu: np.ndarray
    reference_pressure_mbar: float = 1013.0
    reference_temperature_k: float = 296.0

    def __post_init__(self) -> None:
        arrays = {
            "wavenumber_cm": np.asarray(self.wavenumber_cm, dtype=float),
            "self_296": np.asarray(self.self_296, dtype=float),
            "self_260": np.asarray(self.self_260, dtype=float),
            "foreign_296": np.asarray(self.foreign_296, dtype=float),
        }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("H2O continuum coefficient arrays must have the same shape")
        if arrays["wavenumber_cm"].ndim != 1:
            raise ValueError("H2O continuum coefficient arrays must be one-dimensional")
        if not np.all(np.diff(arrays["wavenumber_cm"]) > 0):
            raise ValueError("H2O continuum wavenumber grid must increase")
        xfac_index = np.asarray(self.xfac_rhu_index, dtype=int)
        xfac = np.asarray(self.xfac_rhu, dtype=float)
        if xfac_index.shape != xfac.shape:
            raise ValueError("XFAC_RHU index and factor arrays must have the same shape")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "xfac_rhu_index", xfac_index)
        object.__setattr__(self, "xfac_rhu", xfac)

    @classmethod
    def from_package_data(cls) -> "LBLRTMH2OContinuum":
        with resources.files("genmolfit").joinpath("data", LBLRTM_H2O_CONTINUUM_DATA).open("rb") as handle:
            data = np.load(handle)
            return cls(
                wavenumber_cm=np.asarray(data["wavenumber_cm"], dtype=float),
                self_296=np.asarray(data["self_296"], dtype=float),
                self_260=np.asarray(data["self_260"], dtype=float),
                foreign_296=np.asarray(data["foreign_296"], dtype=float),
                xfac_rhu_index=np.asarray(data["xfac_rhu_index"], dtype=int),
                xfac_rhu=np.asarray(data["xfac_rhu"], dtype=float),
            )

    def absorption_coefficients(
        self,
        wavenumber_cm: np.ndarray,
        *,
        pressure_mbar: float,
        temperature_k: float,
        h2o_vmr: float,
        include_radiation_term: bool = True,
        use_foreign_closure: bool = False,
        _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del use_foreign_closure
        if pressure_mbar <= 0:
            raise ValueError("pressure_mbar must be positive")
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if h2o_vmr < 0:
            raise ValueError("h2o_vmr must be non-negative")

        target = np.asarray(wavenumber_cm, dtype=float)
        density_ratio = (pressure_mbar / self.reference_pressure_mbar) * (
            self.reference_temperature_k / temperature_k
        )
        tfac = (float(temperature_k) - self.reference_temperature_k) / (260.0 - self.reference_temperature_k)
        positive_self = self.self_296 > 0
        self_grid = np.zeros(self.self_296.shape, dtype=float)
        self_grid[positive_self] = self.self_296[positive_self] * (
            self.self_260[positive_self] / self.self_296[positive_self]
        ) ** tfac
        foreign_grid = self.foreign_296 * _lblrtm_h2o_foreign_scale(
            self.wavenumber_cm,
            self.xfac_rhu_index,
            self.xfac_rhu,
        )

        self_coeff_grid = self_grid * h2o_vmr * density_ratio * 1.0e-20
        foreign_coeff_grid = foreign_grid * max(0.0, 1.0 - h2o_vmr) * density_ratio * 1.0e-20
        if include_radiation_term:
            radiation = radiation_term_cm(self.wavenumber_cm, temperature_k)
            self_coeff_grid *= radiation
            foreign_coeff_grid *= radiation

        interpolation_plan = _interpolation_plan or _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        return (
            interpolation_plan.apply(self_coeff_grid),
            interpolation_plan.apply(foreign_coeff_grid),
        )

    def optical_depth(
        self,
        wavenumber_cm: np.ndarray,
        atmosphere: AtmosphereProfile,
        *,
        include_radiation_term: bool = True,
        use_foreign_closure: bool = False,
    ) -> np.ndarray:
        target = np.asarray(wavenumber_cm, dtype=float)
        tau = np.zeros(target.shape, dtype=float)
        interpolation_plan = _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        for layer in atmosphere.layers:
            h2o_vmr = float(layer.mixing_ratios.get("H2O", 0.0))
            h2o_column = layer.column_density_cm2("H2O")
            if h2o_vmr <= 0 or h2o_column <= 0:
                continue
            self_coeff, foreign_coeff = self.absorption_coefficients(
                target,
                pressure_mbar=layer.pressure_atm * 1013.25,
                temperature_k=layer.temperature_k,
                h2o_vmr=h2o_vmr,
                include_radiation_term=include_radiation_term,
                use_foreign_closure=use_foreign_closure,
                _interpolation_plan=interpolation_plan,
            )
            tau += (self_coeff + foreign_coeff) * h2o_column
        return tau


@dataclass(frozen=True)
class LBLRTMCO2Continuum:
    """LBLRTM `contnm.f90` CO2-air continuum coefficients.

    The bundled table is extracted from the Molecfit 4.4.4 LBLRTM 12.11
    `BFCO2` block data. Coefficients are in the source convention and are
    converted to density-scaled cm2/molecule/amagat coefficients at runtime.
    """

    wavenumber_cm: np.ndarray
    coefficient_source: np.ndarray
    correction_wavenumber_cm: np.ndarray
    correction_factor: np.ndarray

    def __post_init__(self) -> None:
        wavenumber = np.asarray(self.wavenumber_cm, dtype=float)
        coefficient = np.asarray(self.coefficient_source, dtype=float)
        correction_wavenumber = np.asarray(self.correction_wavenumber_cm, dtype=float)
        correction_factor = np.asarray(self.correction_factor, dtype=float)
        if wavenumber.ndim != 1 or coefficient.ndim != 1:
            raise ValueError("CO2 continuum arrays must be one-dimensional")
        if wavenumber.shape != coefficient.shape:
            raise ValueError("CO2 continuum wavenumber and coefficient arrays must have the same shape")
        if correction_wavenumber.shape != correction_factor.shape:
            raise ValueError("CO2 correction arrays must have the same shape")
        if not np.all(np.diff(wavenumber) > 0):
            raise ValueError("CO2 continuum wavenumber grid must increase")
        if not np.all(np.diff(correction_wavenumber) > 0):
            raise ValueError("CO2 correction wavenumber grid must increase")
        object.__setattr__(self, "wavenumber_cm", wavenumber)
        object.__setattr__(self, "coefficient_source", coefficient)
        object.__setattr__(self, "correction_wavenumber_cm", correction_wavenumber)
        object.__setattr__(self, "correction_factor", correction_factor)

    @classmethod
    def from_package_data(cls) -> "LBLRTMCO2Continuum":
        with resources.files("genmolfit").joinpath("data", LBLRTM_CO2_CONTINUUM_DATA).open("rb") as handle:
            data = np.load(handle)
            return cls(
                wavenumber_cm=np.asarray(data["wavenumber_cm"], dtype=float),
                coefficient_source=np.asarray(data["coefficient"], dtype=float),
                correction_wavenumber_cm=np.asarray(data["xfac_wavenumber_cm"], dtype=float),
                correction_factor=np.asarray(data["xfac"], dtype=float),
            )

    def coefficient_at(
        self,
        wavenumber_cm: np.ndarray,
        temperature_k: float,
        *,
        _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
    ) -> np.ndarray:
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        source_coeff = np.array(self.coefficient_source, dtype=float, copy=True)
        bandhead = (self.wavenumber_cm >= 2386.0) & (self.wavenumber_cm <= 2434.0)
        if np.any(bandhead):
            source_coeff[bandhead] *= (
                float(temperature_k) / LBLRTM_CO2_BANDHEAD_T_EFF_K
            ) ** LBLRTM_CO2_BANDHEAD_TDEP

        correction = np.ones(self.wavenumber_cm.shape, dtype=float)
        in_correction = (self.wavenumber_cm >= 2000.0) & (self.wavenumber_cm <= 2998.0)
        if np.any(in_correction):
            # LBLRTM indexes this correction with JFAC=(VJ-1998)/2 using
            # one-based Fortran indexing. The table itself is on the same
            # 2 cm-1 spacing as the continuum grid.
            correction[in_correction] = _interp_1d_sorted(
                self.correction_wavenumber_cm,
                self.correction_factor,
                self.wavenumber_cm[in_correction],
            )
        source_coeff *= correction
        target = np.asarray(wavenumber_cm, dtype=float)
        interpolation_plan = _interpolation_plan or _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        coefficient = interpolation_plan.apply(source_coeff)
        coefficient *= 1.0e-20
        coefficient *= radiation_term_cm(target, temperature_k)
        return coefficient


@dataclass(frozen=True)
class LBLRTMN2FundamentalContinuum:
    """LBLRTM N2 collision-induced fundamental-band continuum.

    The source table is the Lafferty et al. N2-N2 model used by LBLRTM
    12.11, with the MT_CKD 2.8 wavelength-dependent N2-H2O collision-partner
    efficiency. Coefficients are evaluated with the reciprocal-temperature
    interpolation and four-point spectral interpolation in ``contnm.f90``.
    """

    wavenumber_cm: np.ndarray
    n2_n2_272: np.ndarray
    n2_n2_228: np.ndarray
    h2o_relative_efficiency: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "wavenumber_cm": np.asarray(self.wavenumber_cm, dtype=float),
            "n2_n2_272": np.asarray(self.n2_n2_272, dtype=float),
            "n2_n2_228": np.asarray(self.n2_n2_228, dtype=float),
            "h2o_relative_efficiency": np.asarray(self.h2o_relative_efficiency, dtype=float),
        }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1 or arrays["wavenumber_cm"].ndim != 1:
            raise ValueError("N2 fundamental coefficient arrays must be one-dimensional and equal-sized")
        if arrays["wavenumber_cm"].size < 4:
            raise ValueError("N2 fundamental table must contain at least four points")
        if not np.all(np.diff(arrays["wavenumber_cm"]) > 0):
            raise ValueError("N2 fundamental wavenumber grid must increase")
        if np.any(arrays["n2_n2_272"] < 0) or np.any(arrays["n2_n2_228"] < 0):
            raise ValueError("N2 fundamental coefficients must be non-negative")
        if np.any(arrays["h2o_relative_efficiency"] < 0):
            raise ValueError("N2-H2O relative efficiencies must be non-negative")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @classmethod
    def from_package_data(cls) -> "LBLRTMN2FundamentalContinuum":
        with resources.files("genmolfit").joinpath("data", LBLRTM_N2_FUNDAMENTAL_DATA).open("rb") as handle:
            data = np.load(handle)
            return cls(
                wavenumber_cm=np.asarray(data["wavenumber_cm"], dtype=float),
                n2_n2_272=np.asarray(data["n2_n2_272"], dtype=float),
                n2_n2_228=np.asarray(data["n2_n2_228"], dtype=float),
                h2o_relative_efficiency=np.asarray(data["h2o_relative_efficiency"], dtype=float),
            )

    def source_coefficients(
        self,
        temperature_k: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source-grid N2-N2, N2-O2, and N2-H2O coefficients."""

        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        temperature = float(temperature_k)
        reciprocal_fraction = ((1.0 / temperature) - (1.0 / 272.0)) / (
            (1.0 / 228.0) - (1.0 / 272.0)
        )
        linear_fraction = (temperature - 272.0) / (228.0 - 272.0)
        positive = (self.n2_n2_272 > 0) & (self.n2_n2_228 > 0)
        n2_n2 = self.n2_n2_272 + (self.n2_n2_228 - self.n2_n2_272) * linear_fraction
        n2_n2[positive] = self.n2_n2_272[positive] * (
            self.n2_n2_228[positive] / self.n2_n2_272[positive]
        ) ** reciprocal_fraction

        # n2_ver_1 removes the radiation field by dividing by wavenumber.
        n2_n2 = n2_n2 / self.wavenumber_cm
        oxygen_efficiency = 1.294 - 0.4545 * temperature / 296.0
        n2_o2 = oxygen_efficiency * n2_n2
        n2_h2o = (9.0 / 7.0) * self.h2o_relative_efficiency * n2_n2
        return n2_n2, n2_o2, n2_h2o

    def mixed_coefficient_at(
        self,
        wavenumber_cm: np.ndarray,
        *,
        temperature_k: float,
        n2_vmr: float,
        o2_vmr: float,
        h2o_vmr: float,
        include_radiation_term: bool = True,
        _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
    ) -> np.ndarray:
        """Return the collision-partner-weighted source coefficient."""

        target = np.asarray(wavenumber_cm, dtype=float)
        if _interpolation_plan is None and not _regular_grid_has_target_support(
            self.wavenumber_cm,
            target,
        ):
            return np.zeros(target.shape, dtype=float)
        n2_n2, n2_o2, n2_h2o = self.source_coefficients(temperature_k)
        mixed = (
            max(0.0, float(n2_vmr)) * n2_n2
            + max(0.0, float(o2_vmr)) * n2_o2
            + max(0.0, float(h2o_vmr)) * n2_h2o
        )
        if include_radiation_term:
            mixed *= radiation_term_cm(self.wavenumber_cm, temperature_k)
        interpolation_plan = _interpolation_plan or _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        return interpolation_plan.apply(mixed)

    def optical_depth_layer(
        self,
        wavenumber_cm: np.ndarray,
        *,
        n2_column_cm2: float,
        air_amagat: float,
        temperature_k: float,
        n2_vmr: float,
        o2_vmr: float,
        h2o_vmr: float,
        xn2cn: float = 1.0,
        jrad: int = 1,
        _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
    ) -> np.ndarray:
        """Return one layer's N2 fundamental optical depth."""

        if n2_column_cm2 < 0:
            raise ValueError("n2_column_cm2 must be non-negative")
        if air_amagat < 0:
            raise ValueError("air_amagat must be non-negative")
        if xn2cn < 0:
            raise ValueError("xn2cn must be non-negative")
        if jrad not in (0, 1):
            raise ValueError("jrad must be 0 or 1")
        coefficient = self.mixed_coefficient_at(
            wavenumber_cm,
            temperature_k=temperature_k,
            n2_vmr=n2_vmr,
            o2_vmr=o2_vmr,
            h2o_vmr=h2o_vmr,
            include_radiation_term=jrad == 1,
            _interpolation_plan=_interpolation_plan,
        )
        return (
            float(xn2cn)
            * (float(n2_column_cm2) / LBLRTM_CONTNM_LOSCHMIDT_CM3)
            * float(air_amagat)
            * coefficient
        )


@dataclass(frozen=True)
class LBLRTMN2OvertoneContinuum:
    """LBLRTM N2 collision-induced first-overtone continuum.

    LBLRTM uses the Shapiro-Gush spectrum modified by Mlawer and Gombos
    over 4340--4910 cm-1. The source assumes equal N2, O2, and H2O
    collision-partner efficiencies and no explicit temperature dependence.
    """

    wavenumber_cm: np.ndarray
    n2_n2: np.ndarray

    def __post_init__(self) -> None:
        wavenumber = np.asarray(self.wavenumber_cm, dtype=float)
        coefficient = np.asarray(self.n2_n2, dtype=float)
        if wavenumber.ndim != 1 or coefficient.shape != wavenumber.shape:
            raise ValueError("N2 overtone arrays must be one-dimensional and equal-sized")
        if wavenumber.size < 4 or not np.all(np.diff(wavenumber) > 0):
            raise ValueError("N2 overtone grid must contain at least four increasing points")
        if np.any(coefficient < 0):
            raise ValueError("N2 overtone coefficients must be non-negative")
        object.__setattr__(self, "wavenumber_cm", wavenumber)
        object.__setattr__(self, "n2_n2", coefficient)

    @classmethod
    def from_package_data(cls) -> "LBLRTMN2OvertoneContinuum":
        with resources.files("genmolfit").joinpath("data", LBLRTM_N2_FUNDAMENTAL_DATA).open("rb") as handle:
            data = np.load(handle)
            return cls(
                wavenumber_cm=np.asarray(data["overtone_wavenumber_cm"], dtype=float),
                n2_n2=np.asarray(data["overtone_n2_n2"], dtype=float),
            )

    def optical_depth_layer(
        self,
        wavenumber_cm: np.ndarray,
        *,
        n2_column_cm2: float,
        air_amagat: float,
        temperature_k: float,
        n2_vmr: float,
        o2_vmr: float,
        h2o_vmr: float,
        xn2cn: float = 1.0,
        jrad: int = 1,
        _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
    ) -> np.ndarray:
        """Return one layer's N2 first-overtone optical depth."""

        if n2_column_cm2 < 0 or air_amagat < 0 or xn2cn < 0:
            raise ValueError("columns, density, and xn2cn must be non-negative")
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if jrad not in (0, 1):
            raise ValueError("jrad must be 0 or 1")

        target = np.asarray(wavenumber_cm, dtype=float)
        if _interpolation_plan is None and not _regular_grid_has_target_support(
            self.wavenumber_cm,
            target,
        ):
            return np.zeros(target.shape, dtype=float)
        source = self.n2_n2 / self.wavenumber_cm
        if jrad == 1:
            source = source * radiation_term_cm(self.wavenumber_cm, temperature_k)
        interpolation_plan = _interpolation_plan or _prepare_cubic_regular_interpolation(
            self.wavenumber_cm,
            target,
        )
        coefficient = interpolation_plan.apply(source)
        partner_efficiency = sum(
            max(0.0, float(value)) for value in (n2_vmr, o2_vmr, h2o_vmr)
        )
        return (
            float(xn2cn)
            * (float(n2_column_cm2) / LBLRTM_CONTNM_LOSCHMIDT_CM3)
            * float(air_amagat)
            * partner_efficiency
            * coefficient
        )


@dataclass(frozen=True)
class LBLRTMO2Continuum:
    """Ground-based O2 continuum branches from LBLRTM 12.11.

    The implementation follows ``contnm.f90`` for the O2 fundamental,
    1.27 micron CIA, 9100--11000 cm-1 analytic band, A-band continuum,
    and Greenblatt visible continuum. Each branch retains its original
    density and collision-partner convention.
    """

    fundamental_wavenumber_cm: np.ndarray
    fundamental_coefficient: np.ndarray
    fundamental_temperature_coefficient: np.ndarray
    inf1_wavenumber_cm: np.ndarray
    inf1_coefficient: np.ndarray
    aband_wavenumber_cm: np.ndarray
    aband_coefficient: np.ndarray
    visible_wavenumber_cm: np.ndarray
    visible_coefficient: np.ndarray

    def __post_init__(self) -> None:
        pairs = (
            ("fundamental", self.fundamental_wavenumber_cm, self.fundamental_coefficient),
            (
                "fundamental temperature",
                self.fundamental_wavenumber_cm,
                self.fundamental_temperature_coefficient,
            ),
            ("1.27 micron", self.inf1_wavenumber_cm, self.inf1_coefficient),
            ("A band", self.aband_wavenumber_cm, self.aband_coefficient),
            ("visible", self.visible_wavenumber_cm, self.visible_coefficient),
        )
        converted: dict[str, np.ndarray] = {}
        for name, grid_value, coefficient_value in pairs:
            grid = np.asarray(grid_value, dtype=float)
            coefficient = np.asarray(coefficient_value, dtype=float)
            if grid.ndim != 1 or coefficient.shape != grid.shape:
                raise ValueError(f"O2 {name} arrays must be one-dimensional and equal-sized")
            if grid.size < 4 or not np.all(np.diff(grid) > 0):
                raise ValueError(f"O2 {name} grid must contain at least four increasing points")
            converted[name + "_grid"] = grid
            converted[name + "_coefficient"] = coefficient
        if any(
            np.any(values < 0)
            for key, values in converted.items()
            if key.endswith("_coefficient") and key != "fundamental temperature_coefficient"
        ):
            raise ValueError("O2 absorption coefficients must be non-negative")
        object.__setattr__(self, "fundamental_wavenumber_cm", converted["fundamental_grid"])
        object.__setattr__(self, "fundamental_coefficient", converted["fundamental_coefficient"])
        object.__setattr__(
            self,
            "fundamental_temperature_coefficient",
            converted["fundamental temperature_coefficient"],
        )
        object.__setattr__(self, "inf1_wavenumber_cm", converted["1.27 micron_grid"])
        object.__setattr__(self, "inf1_coefficient", converted["1.27 micron_coefficient"])
        object.__setattr__(self, "aband_wavenumber_cm", converted["A band_grid"])
        object.__setattr__(self, "aband_coefficient", converted["A band_coefficient"])
        object.__setattr__(self, "visible_wavenumber_cm", converted["visible_grid"])
        object.__setattr__(self, "visible_coefficient", converted["visible_coefficient"])

    @classmethod
    def from_package_data(cls) -> "LBLRTMO2Continuum":
        with resources.files("genmolfit").joinpath("data", LBLRTM_O2_CONTINUUM_DATA).open("rb") as handle:
            data = np.load(handle)
            return cls(
                fundamental_wavenumber_cm=np.asarray(data["fundamental_wavenumber_cm"], dtype=float),
                fundamental_coefficient=np.asarray(data["fundamental_coefficient"], dtype=float),
                fundamental_temperature_coefficient=np.asarray(
                    data["fundamental_temperature_coefficient"], dtype=float
                ),
                inf1_wavenumber_cm=np.asarray(data["inf1_wavenumber_cm"], dtype=float),
                inf1_coefficient=np.asarray(data["inf1_coefficient"], dtype=float),
                aband_wavenumber_cm=np.asarray(data["aband_wavenumber_cm"], dtype=float),
                aband_coefficient=np.asarray(data["aband_coefficient"], dtype=float),
                visible_wavenumber_cm=np.asarray(data["visible_wavenumber_cm"], dtype=float),
                visible_coefficient=np.asarray(data["visible_coefficient"], dtype=float),
            )

    @staticmethod
    def _interpolate_source(
        grid: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        *,
        temperature_k: float,
        jrad: int,
    ) -> np.ndarray:
        if jrad == 1:
            source = source * radiation_term_cm(grid, temperature_k)
        return cubic_interpolate_regular(grid, source, target)

    def optical_depth_layer(
        self,
        wavenumber_cm: np.ndarray,
        *,
        o2_column_cm2: float,
        air_column_cm2: float,
        air_amagat: float,
        pressure_mbar: float,
        temperature_k: float,
        n2_vmr: float,
        o2_vmr: float,
        h2o_vmr: float,
        xo2cn: float = 1.0,
        jrad: int = 1,
    ) -> np.ndarray:
        """Return one layer's complete supported O2 continuum optical depth."""

        if min(o2_column_cm2, air_column_cm2, air_amagat, pressure_mbar, xo2cn) < 0:
            raise ValueError("O2 columns, density, pressure, and xo2cn must be non-negative")
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if jrad not in (0, 1):
            raise ValueError("jrad must be 0 or 1")
        target = np.asarray(wavenumber_cm, dtype=float)
        tau = np.zeros(target.shape, dtype=float)
        if o2_column_cm2 == 0 or xo2cn == 0:
            return tau

        # Thibault et al. O2 fundamental, 1340--1850 cm-1.
        fundamental_source = (
            (1.0e20 / LBLRTM_CONTNM_LOSCHMIDT_CM3)
            * self.fundamental_coefficient
            * np.exp(
                self.fundamental_temperature_coefficient
                * ((1.0 / 296.0) - (1.0 / float(temperature_k)))
            )
            / self.fundamental_wavenumber_cm
        )
        tau += (
            float(xo2cn)
            * float(o2_column_cm2)
            * 1.0e-20
            * float(air_amagat)
            * self._interpolate_source(
                self.fundamental_wavenumber_cm,
                fundamental_source,
                target,
                temperature_k=temperature_k,
                jrad=jrad,
            )
        )

        # Mate et al. 1.27 micron collision-induced band.
        partner = (
            max(0.0, float(o2_vmr)) / 0.446
            + 0.3 * max(0.0, float(n2_vmr)) / 0.446
            + max(0.0, float(h2o_vmr))
        )
        inf1_source = self.inf1_coefficient / self.inf1_wavenumber_cm
        tau += (
            float(xo2cn)
            * (float(o2_column_cm2) / LBLRTM_CONTNM_LOSCHMIDT_CM3)
            * float(air_amagat)
            * partner
            * self._interpolate_source(
                self.inf1_wavenumber_cm,
                inf1_source,
                target,
                temperature_k=temperature_k,
                jrad=jrad,
            )
        )

        # Analytic 9100--11000 cm-1 O2 continuum from Mlawer et al.
        inf2_grid = np.arange(9100.0, 11000.0 + 2.0, 2.0)
        delta1 = inf2_grid - 9375.0
        delta2 = inf2_grid - 9439.0
        damp1 = np.where(delta1 < 0.0, np.exp(delta1 / 176.1), 1.0)
        damp2 = np.where(delta2 < 0.0, np.exp(delta2 / 176.1), 1.0)
        inf2 = 0.31831 * (
            (1.166e-4 * damp1 / 58.96) / (1.0 + (delta1 / 58.96) ** 2)
            + (3.086e-5 * damp2 / 45.04) / (1.0 + (delta2 / 45.04) ** 2)
        ) * 1.054
        inf2_source = inf2 / inf2_grid
        inf2_source[[0, -1]] = 0.0
        density_296 = (float(pressure_mbar) / 1013.0) * (296.0 / float(temperature_k))
        o2_fraction = float(o2_column_cm2) / float(air_column_cm2) if air_column_cm2 > 0 else 0.0
        tau += (
            float(xo2cn)
            * float(o2_column_cm2)
            * 1.0e-20
            * density_296
            * (o2_fraction / 0.209)
            * self._interpolate_source(
                inf2_grid,
                inf2_source,
                target,
                temperature_k=temperature_k,
                jrad=jrad,
            )
        )

        # Mlawer O2 A-band continuum.
        aband_source = self.aband_coefficient / self.aband_wavenumber_cm
        tau += (
            float(xo2cn)
            * (float(o2_column_cm2) / LBLRTM_CONTNM_LOSCHMIDT_CM3)
            * float(air_amagat)
            * self._interpolate_source(
                self.aband_wavenumber_cm,
                aband_source,
                target,
                temperature_k=temperature_k,
                jrad=jrad,
            )
        )

        # Greenblatt visible O2-O2 continuum, activated by LBLRTM above
        # 15000 cm-1 even though the source table starts at 15140 cm-1.
        visible_factor = 1.0 / (
            LBLRTM_CONTNM_LOSCHMIDT_CM3
            * 1.0e-20
            * (55.0 * 273.0 / 296.0) ** 2
            * 89.5
        )
        visible_source = visible_factor * self.visible_coefficient / self.visible_wavenumber_cm
        tau += (
            float(xo2cn)
            * float(o2_column_cm2)
            * 1.0e-20
            * float(air_amagat)
            * max(0.0, float(o2_vmr))
            * self._interpolate_source(
                self.visible_wavenumber_cm,
                visible_source,
                target,
                temperature_k=temperature_k,
                jrad=jrad,
            )
        )
        return tau


def _lblrtm_h2o_foreign_scale(
    wavenumber_cm: np.ndarray,
    xfac_rhu_index: np.ndarray,
    xfac_rhu: np.ndarray,
) -> np.ndarray:
    wavenumber = np.asarray(wavenumber_cm, dtype=float)
    scale = np.ones(wavenumber.shape, dtype=float)
    low = wavenumber <= 600.0
    if np.any(low):
        jfac = np.trunc((wavenumber[low] + 10.0) / 10.0 + 0.00001).astype(int)
        scale[low] = np.interp(jfac, xfac_rhu_index, xfac_rhu, left=xfac_rhu[0], right=xfac_rhu[-1])

    high = ~low
    if np.any(high):
        v = wavenumber[high]
        f0 = 0.06
        v0f1 = 255.67
        hwsq1 = 240.0**2
        beta1 = 57.83
        c1 = -0.42
        n1 = 8
        c2 = 0.3
        beta2 = 630.0
        n2 = 8
        vdelsq1 = (v - v0f1) ** 2
        vdelmsq1 = (v + v0f1) ** 2
        vf1 = ((v - v0f1) / beta1) ** n1
        vmf1 = ((v + v0f1) / beta1) ** n1
        vf2 = (v / beta2) ** n2
        scale[high] = 1.0 + (
            f0
            + c1
            * (
                hwsq1 / (vdelsq1 + hwsq1 + vf1)
                + hwsq1 / (vdelmsq1 + hwsq1 + vmf1)
            )
        ) / (1.0 + c2 * vf2)
    return scale


def radiation_term_cm(
    wavenumber_cm: np.ndarray,
    temperature_k: float,
    *,
    second_radiation_constant_cm_k: float = SECOND_RADIATION_CONSTANT_CM_K,
) -> np.ndarray:
    """LBLRTM/MT_CKD radiation term in cm-1."""

    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    wavenumber = np.asarray(wavenumber_cm, dtype=float)
    x = wavenumber * float(second_radiation_constant_cm_k) / temperature_k
    radiation = np.array(wavenumber, dtype=float, copy=True)

    small = x <= 0.01
    middle = (x > 0.01) & (x <= 10.0)
    radiation[small] = 0.5 * x[small] * wavenumber[small]
    expvkt = np.exp(-x[middle])
    radiation[middle] = wavenumber[middle] * (1.0 - expvkt) / (1.0 + expvkt)
    return radiation


def radiation_term_interval_cm(
    vi_cm: float,
    dvi_cm: float,
    temperature_k: float,
    *,
    vinew_cm: float = 0.0,
    rdlast_cm: float = -1.0,
    second_radiation_constant_cm_k: float = SECOND_RADIATION_CONSTANT_CM_K,
) -> tuple[float, float, float, float, int]:
    """LBLRTM ``RADFNI`` interval radiation-term helper.

    Returns ``(radfni, vinew, rdel, rdlast, intervals)``. ``radfni`` is the
    radiation term at ``vi_cm``; ``vinew`` is moved to an integer number of
    ``dvi_cm`` intervals from ``vi_cm``; ``rdel`` is the per-grid-point linear
    increment from ``radfni`` to the radiation term at ``vinew``; ``rdlast`` is
    the radiation term at ``vinew`` for the next call.
    """

    vi = float(vi_cm)
    dvi = float(dvi_cm)
    temperature = float(temperature_k)
    c2 = float(second_radiation_constant_cm_k)
    if dvi <= 0 or not np.isfinite(dvi):
        raise ValueError("dvi_cm must be positive")
    if temperature <= 0:
        raise ValueError("temperature_k must be positive")
    xkt = temperature / c2

    radfni = (
        float(radiation_term_cm(np.array([vi]), temperature, second_radiation_constant_cm_k=c2)[0])
        if rdlast_cm < 0
        else float(rdlast_cm)
    )

    xviokt = vi / xkt
    if vinew_cm >= 0:
        if xviokt <= 0.01:
            target_vinew = vi + 3.0e-3 * 0.5 * vi
        elif xviokt <= 10.0:
            expvkt = np.exp(-xviokt)
            xminus = 1.0 - expvkt
            xplus = 1.0 + expvkt
            cvikt = xviokt * expvkt
            target_vinew = vi + 3.0e-3 * vi / (1.0 + (cvikt / xminus + cvikt / xplus))
        else:
            target_vinew = vi + 3.0e-3 * vi
    else:
        target_vinew = abs(float(vinew_cm))

    intervals = max(int((target_vinew - vi) / dvi), 1)
    vinew = vi + dvi * float(intervals) if vinew_cm >= 0 else target_vinew
    if vinew_cm < 0:
        intervals = max(int((vinew - vi) / dvi), 1)

    rdnext = float(radiation_term_cm(np.array([vinew]), temperature, second_radiation_constant_cm_k=c2)[0])
    rdel = (rdnext - radfni) / float(intervals)
    return radfni, vinew, rdel, rdnext, intervals


def lblrtm_rayleigh_optical_depth(
    wavenumber_cm: np.ndarray,
    total_column_cm2: float,
    *,
    xrayl: float = 1.0,
    jrad: int = 1,
    temperature_k: float = 296.0,
) -> np.ndarray:
    """Rayleigh optical depth using the LBLRTM ``contnm.f90`` formula.

    ``total_column_cm2`` is the total air column in molecules/cm2. The
    implementation follows the vectorized form of the source block that is
    active for wavenumbers >= 820 cm-1. ``jrad=1`` is the direct-transmission
    branch; ``jrad=0`` applies the source-code radiation-term division used for
    radiance calculations.
    """

    if total_column_cm2 < 0:
        raise ValueError("total_column_cm2 must be non-negative")
    if xrayl < 0:
        raise ValueError("xrayl must be non-negative")
    if jrad not in (0, 1):
        raise ValueError("jrad must be 0 or 1")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")

    wavenumber = np.asarray(wavenumber_cm, dtype=float)
    x = wavenumber / 1.0e4
    denominator = 9.38076e2 - 10.8426 * x**2
    coefficient = np.zeros(wavenumber.shape, dtype=float)
    valid = (wavenumber >= LBLRTM_RAYLEIGH_MIN_WAVENUMBER_CM) & (denominator > 0)
    if not np.any(valid):
        return coefficient

    conv_cm2mol = float(xrayl) * 1.0e-20 / (LBLRTM_RAYLEIGH_LOSCHMIDT_SCALED * 1.0e5)
    coefficient[valid] = x[valid] ** 3 / denominator[valid] * conv_cm2mol
    if jrad == 1:
        coefficient[valid] *= x[valid]
    else:
        radiation = radiation_term_cm(wavenumber[valid], temperature_k)
        coefficient[valid] *= x[valid] / radiation

    return coefficient * float(total_column_cm2)


def lblrtm_n2_rototranslational_coefficients(
    wavenumber_cm: np.ndarray,
    temperature_k: float,
    *,
    _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """LBLRTM `contnm.f90` N2 pure-rotation CIA coefficients.

    Returns the N2-N2 coefficient in cm-1 amagat-2 and the O2 relative
    broadening efficiency used by the `xn2_r` branch. The source tabulates
    coefficients at 296 K and 220 K on a 5 cm-1 grid from -10 to 350 cm-1.
    """

    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    target = np.asarray(wavenumber_cm, dtype=float)
    source_grid = _lblrtm_n2_rot_grid()
    if _interpolation_plan is None and not _regular_grid_has_target_support(source_grid, target):
        empty = np.zeros(target.shape, dtype=float)
        return empty, empty.copy()
    tfac = (float(temperature_k) - 296.0) / (220.0 - 296.0)
    coeff_grid = LBLRTM_N2_ROT_T296 * (LBLRTM_N2_ROT_T220 / LBLRTM_N2_ROT_T296) ** tfac
    sf_grid = LBLRTM_N2_ROT_SF296 * (LBLRTM_N2_ROT_SF220 / LBLRTM_N2_ROT_SF296) ** tfac
    oxygen_efficiency_grid = (sf_grid - 1.0) * (0.79 / 0.21)
    interpolation_plan = _interpolation_plan or _prepare_cubic_regular_interpolation(
        source_grid,
        target,
    )
    return interpolation_plan.apply(coeff_grid), interpolation_plan.apply(oxygen_efficiency_grid)


def lblrtm_n2_rototranslational_optical_depth(
    wavenumber_cm: np.ndarray,
    *,
    n2_column_cm2: float,
    air_amagat: float,
    temperature_k: float,
    n2_vmr: float,
    o2_vmr: float,
    h2o_vmr: float = 0.0,
    xn2cn: float = 1.0,
    jrad: int = 1,
    _interpolation_plan: _CubicRegularInterpolationPlan | None = None,
) -> np.ndarray:
    """Optical depth for the LBLRTM N2 rototranslational continuum branch."""

    if n2_column_cm2 < 0:
        raise ValueError("n2_column_cm2 must be non-negative")
    if air_amagat < 0:
        raise ValueError("air_amagat must be non-negative")
    if xn2cn < 0:
        raise ValueError("xn2cn must be non-negative")
    if jrad not in (0, 1):
        raise ValueError("jrad must be 0 or 1")
    coefficient, oxygen_efficiency = lblrtm_n2_rototranslational_coefficients(
        wavenumber_cm,
        temperature_k,
        _interpolation_plan=_interpolation_plan,
    )
    partner_efficiency = max(0.0, n2_vmr) + oxygen_efficiency * max(0.0, o2_vmr) + max(0.0, h2o_vmr)
    tau = (
        float(xn2cn)
        * (float(n2_column_cm2) / LBLRTM_CONTNM_LOSCHMIDT_CM3)
        * float(air_amagat)
        * coefficient
        * partner_efficiency
    )
    if jrad == 1:
        tau = tau * radiation_term_cm(np.asarray(wavenumber_cm, dtype=float), temperature_k)
    return tau


@dataclass(frozen=True)
class _CubicRegularInterpolationPlan:
    source_size: int
    output_shape: tuple[int, ...]
    output_indices: np.ndarray
    source_indices: np.ndarray
    weight_minus_one: np.ndarray
    weight_zero: np.ndarray
    weight_plus_one: np.ndarray
    weight_plus_two: np.ndarray

    def apply(self, y_grid: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
        values = np.asarray(y_grid, dtype=float)
        if values.ndim != 1 or values.size != self.source_size:
            raise ValueError("y_grid does not match the cubic interpolation plan")
        output = np.full(int(np.prod(self.output_shape)), fill_value, dtype=float)
        j = self.source_indices
        output[self.output_indices] = (
            values[j - 1] * self.weight_minus_one
            + values[j] * self.weight_zero
            + values[j + 1] * self.weight_plus_one
            + values[j + 2] * self.weight_plus_two
        )
        return output.reshape(self.output_shape)


def _regular_grid_has_target_support(
    x_grid: np.ndarray,
    x_target: np.ndarray,
) -> bool:
    target = np.asarray(x_target, dtype=float)
    finite = target[np.isfinite(target)]
    return bool(
        finite.size
        and np.nanmax(finite) >= float(x_grid[0])
        and np.nanmin(finite) <= float(x_grid[-1])
    )


def _prepare_cubic_regular_interpolation(
    x_grid: np.ndarray,
    x_target: np.ndarray,
) -> _CubicRegularInterpolationPlan:
    x_grid = np.asarray(x_grid, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    if x_grid.ndim != 1:
        raise ValueError("x_grid must be one-dimensional")
    if x_grid.size < 4:
        raise ValueError("at least four grid points are required")

    spacing = x_grid[1] - x_grid[0]
    if spacing <= 0:
        raise ValueError("x_grid must be increasing")

    position = (x_target - x_grid[0]) / spacing
    finite = np.isfinite(position)
    j = np.zeros(x_target.shape, dtype=int)
    j[finite] = np.floor(position[finite] + 0.001).astype(int)
    valid = finite & (j >= 1) & (j + 2 < x_grid.size)
    output_indices = np.flatnonzero(valid)
    jv = j.ravel()[output_indices]
    vj = x_grid[0] + spacing * jv
    p = (x_target.ravel()[output_indices] - vj) / spacing
    c = (3.0 - 2.0 * p) * p * p
    b = 0.5 * p * (1.0 - p)
    b1 = b * (1.0 - p)
    b2 = b * p
    return _CubicRegularInterpolationPlan(
        source_size=x_grid.size,
        output_shape=x_target.shape,
        output_indices=output_indices,
        source_indices=jv,
        weight_minus_one=-b1,
        weight_zero=1.0 - c + b2,
        weight_plus_one=c + b1,
        weight_plus_two=-b2,
    )


def cubic_interpolate_regular(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_target: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Cubic interpolation used by the MT_CKD reference routine.

    This matches the four-point formula in AER's ``myxint`` routine for a
    regular, increasing source grid. Values too close to the source boundaries
    are filled because the four-point stencil is not available there.
    """

    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional")
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid and y_grid must have the same shape")
    if x_grid.size < 4:
        raise ValueError("at least four grid points are required")

    plan = _prepare_cubic_regular_interpolation(x_grid, x_target)
    return plan.apply(y_grid, fill_value=fill_value)


def _lblrtm_n2_rot_grid() -> np.ndarray:
    return LBLRTM_N2_ROT_MIN_WAVENUMBER_CM + LBLRTM_N2_ROT_SPACING_CM * np.arange(
        LBLRTM_N2_ROT_T296.size,
        dtype=float,
    )


def _decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return value.tobytes().decode("utf-8", errors="replace").strip()
    return str(value).strip()


@dataclass(frozen=True)
class TabulatedContinuum:
    """Generic tabulated continuum coefficients.

    Tables can be one-dimensional in wavenumber or two-dimensional in
    wavenumber and temperature. Coefficients are intentionally unit-agnostic;
    the component that consumes this table defines how they scale into optical
    depth.
    """

    wavenumber_cm: np.ndarray
    coefficient: np.ndarray
    temperature_k: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        wavenumber = np.asarray(self.wavenumber_cm, dtype=float)
        coefficient = np.asarray(self.coefficient, dtype=float)
        temperature = None if self.temperature_k is None else np.asarray(self.temperature_k, dtype=float)

        if temperature is None:
            if wavenumber.ndim != 1 or coefficient.ndim != 1:
                raise ValueError("wavenumber_cm and coefficient must be one-dimensional without temperature_k")
            if wavenumber.shape != coefficient.shape:
                raise ValueError("wavenumber_cm and coefficient must have the same shape")
        else:
            if wavenumber.shape != coefficient.shape or temperature.shape != coefficient.shape:
                raise ValueError("wavenumber_cm, temperature_k, and coefficient must have the same shape")
        if np.any(~np.isfinite(wavenumber)):
            raise ValueError("wavenumber_cm must be finite")
        if np.any(~np.isfinite(coefficient)):
            raise ValueError("coefficient must be finite")
        if temperature is not None and np.any(~np.isfinite(temperature)):
            raise ValueError("temperature_k must be finite")

        object.__setattr__(self, "wavenumber_cm", wavenumber)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "temperature_k", temperature)

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        wavenumber_col: str = "wavenumber_cm",
        coefficient_col: str = "coefficient",
        temperature_col: str | None = "temperature_k",
        format: str | None = None,
    ) -> "TabulatedContinuum":
        table = Table.read(path, format=format)
        temperature = None
        if temperature_col is not None and temperature_col in table.colnames:
            temperature = np.asarray(table[temperature_col], dtype=float)
        return cls(
            wavenumber_cm=np.asarray(table[wavenumber_col], dtype=float),
            coefficient=np.asarray(table[coefficient_col], dtype=float),
            temperature_k=temperature,
            name=str(table.meta.get("name", Path(path).stem)),
        )

    def coefficient_at(self, wavenumber_cm: np.ndarray, temperature_k: float) -> np.ndarray:
        target = np.asarray(wavenumber_cm, dtype=float)
        if self.temperature_k is None:
            return _interp_1d_sorted(self.wavenumber_cm, self.coefficient, target)
        return _interpolate_temperature_wavenumber(
            self.wavenumber_cm,
            self.temperature_k,
            self.coefficient,
            target,
            temperature_k,
        )


@dataclass(frozen=True)
class CIABlock:
    pair: tuple[str, str]
    wavenumber_cm: np.ndarray
    temperature_k: float
    coefficient_cm5_molecule2: np.ndarray
    comment: str = ""

    def __post_init__(self) -> None:
        wavenumber = np.asarray(self.wavenumber_cm, dtype=float)
        coefficient = np.asarray(self.coefficient_cm5_molecule2, dtype=float)
        if wavenumber.ndim != 1 or coefficient.ndim != 1:
            raise ValueError("CIA block arrays must be one-dimensional")
        if wavenumber.shape != coefficient.shape:
            raise ValueError("CIA wavenumber and coefficient arrays must have the same shape")
        if self.temperature_k <= 0:
            raise ValueError("CIA temperature_k must be positive")
        order = np.argsort(wavenumber)
        object.__setattr__(self, "wavenumber_cm", wavenumber[order])
        object.__setattr__(self, "coefficient_cm5_molecule2", np.maximum(coefficient[order], 0.0))


@dataclass(frozen=True)
class HitranCIATable:
    """HITRAN CIA cross-section table.

    HITRAN CIA coefficients have units cm5 molecule-2. Optical depth is
    obtained by multiplying by both collision partner number densities and
    path length.
    """

    blocks: tuple[CIABlock, ...]
    pair: tuple[str, str] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("HitranCIATable requires at least one block")
        pairs = {block.pair for block in self.blocks}
        pair = self.pair
        if pair is None:
            if len(pairs) != 1:
                raise ValueError("pair must be supplied when CIA blocks contain multiple pairs")
            pair = next(iter(pairs))
        object.__setattr__(self, "pair", pair)

    @classmethod
    def from_hitran_cia(cls, path: str | Path) -> "HitranCIATable":
        path = Path(path)
        blocks: list[CIABlock] = []
        pending_comments: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if line.startswith("!"):
                    pending_comments.append(line[1:].strip())
                    continue

                pair, n_points, temperature, comment = _parse_hitran_cia_header(line)
                wavenumber = np.empty(n_points, dtype=float)
                coefficient = np.empty(n_points, dtype=float)
                for index in range(n_points):
                    data_line = handle.readline()
                    if not data_line:
                        raise ValueError("unexpected end of CIA file while reading data")
                    parts = data_line.split()
                    if len(parts) < 2:
                        raise ValueError(f"invalid CIA data row: {data_line!r}")
                    wavenumber[index] = float(parts[0])
                    coefficient[index] = float(parts[1])
                full_comment = " ".join([*pending_comments, comment]).strip()
                pending_comments.clear()
                blocks.append(
                    CIABlock(
                        pair=pair,
                        wavenumber_cm=wavenumber,
                        temperature_k=temperature,
                        coefficient_cm5_molecule2=coefficient,
                        comment=full_comment,
                    )
                )
        return cls(blocks=tuple(blocks), name=path.stem)

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        pair: tuple[str, str],
        wavenumber_col: str = "wavenumber_cm",
        coefficient_col: str = "coefficient_cm5_molecule2",
        temperature_col: str = "temperature_k",
        format: str | None = None,
    ) -> "HitranCIATable":
        table = Table.read(path, format=format)
        blocks = []
        for temperature in np.unique(np.asarray(table[temperature_col], dtype=float)):
            keep = np.asarray(table[temperature_col], dtype=float) == temperature
            blocks.append(
                CIABlock(
                    pair=pair,
                    wavenumber_cm=np.asarray(table[wavenumber_col], dtype=float)[keep],
                    temperature_k=float(temperature),
                    coefficient_cm5_molecule2=np.asarray(table[coefficient_col], dtype=float)[keep],
                )
            )
        return cls(blocks=tuple(blocks), pair=pair, name=str(table.meta.get("name", Path(path).stem)))

    def coefficient_at(self, wavenumber_cm: np.ndarray, temperature_k: float) -> np.ndarray:
        return self.coefficients_at(wavenumber_cm, np.array([temperature_k], dtype=float))[0]

    def coefficients_at(
        self,
        wavenumber_cm: np.ndarray,
        temperatures_k: np.ndarray,
    ) -> np.ndarray:
        """Interpolate coefficients for several atmospheric temperatures.

        Wavelength interpolation depends only on the target grid, so it is
        performed once per CIA table rather than once per atmospheric layer.
        Temperature interpolation is then vectorized over both layers and
        wavelength columns while preserving gaps between HITRAN blocks.
        """

        target = np.asarray(wavenumber_cm, dtype=float)
        target_temperatures = np.asarray(temperatures_k, dtype=float)
        if target.ndim != 1 or target_temperatures.ndim != 1:
            raise ValueError("wavenumber_cm and temperatures_k must be one-dimensional")
        temperatures = np.array([block.temperature_k for block in self.blocks], dtype=float)
        unique_temperatures = np.unique(temperatures)
        by_temperature = np.full((unique_temperatures.size, target.size), np.nan, dtype=float)

        for temp_index, temperature in enumerate(unique_temperatures):
            for block in self.blocks:
                if block.temperature_k != temperature:
                    continue
                values = np.interp(
                    target,
                    block.wavenumber_cm,
                    block.coefficient_cm5_molecule2,
                    left=np.nan,
                    right=np.nan,
                )
                fill = np.isfinite(values)
                by_temperature[temp_index, fill] = values[fill]

        return _interp_temperature_rows(unique_temperatures, by_temperature, target_temperatures)


def _parse_hitran_cia_header(line: str) -> tuple[tuple[str, str], int, float, str]:
    if len(line) < 54:
        raise ValueError(f"invalid HITRAN CIA header: {line!r}")
    molecule = line[:20].strip()
    if "-" not in molecule:
        raise ValueError(f"CIA molecule field must look like A-B, got {molecule!r}")
    species_a, species_b = (part.strip() for part in molecule.split("-", 1))
    n_points = int(line[40:47])
    temperature = float(line[47:54])
    comment = line[70:97].strip() if len(line) >= 97 else ""
    return (species_a, species_b), n_points, temperature, comment


def _interpolate_temperature_wavenumber(
    wavenumber_cm: np.ndarray,
    temperature_k: np.ndarray,
    coefficient: np.ndarray,
    target_wavenumber_cm: np.ndarray,
    target_temperature_k: float,
) -> np.ndarray:
    unique_temperatures = np.unique(temperature_k)
    by_temperature = np.full((unique_temperatures.size, target_wavenumber_cm.size), np.nan, dtype=float)
    for index, temperature in enumerate(unique_temperatures):
        keep = temperature_k == temperature
        by_temperature[index] = _interp_1d_sorted(
            wavenumber_cm[keep],
            coefficient[keep],
            target_wavenumber_cm,
            fill_value=np.nan,
        )
    return _interp_temperature_columns(unique_temperatures, by_temperature, target_temperature_k)


def _interp_temperature_columns(
    temperature_grid: np.ndarray,
    values_by_temperature: np.ndarray,
    target_temperature_k: float,
) -> np.ndarray:
    return _interp_temperature_rows(
        temperature_grid,
        values_by_temperature,
        np.array([target_temperature_k], dtype=float),
    )[0]


def _interp_temperature_rows(
    temperature_grid: np.ndarray,
    values_by_temperature: np.ndarray,
    target_temperatures_k: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of independent ``np.interp`` calls per column."""

    temperature_grid = np.asarray(temperature_grid, dtype=float)
    values = np.asarray(values_by_temperature, dtype=float)
    targets = np.asarray(target_temperatures_k, dtype=float)
    if temperature_grid.ndim != 1 or values.ndim != 2 or targets.ndim != 1:
        raise ValueError("temperature interpolation inputs have invalid dimensions")
    if values.shape[0] != temperature_grid.size:
        raise ValueError("temperature grid length must match coefficient rows")

    result = np.zeros((targets.size, values.shape[1]), dtype=float)
    if values.shape[1] == 0 or targets.size == 0:
        return result

    finite = np.isfinite(values)
    # HITRAN files typically have only a few distinct wavelength-coverage
    # patterns. Group columns with the same valid temperature rows so each
    # group can be interpolated as one dense matrix.
    packed = np.packbits(finite.T, axis=1)
    _, first_columns, group_index = np.unique(
        packed,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    for group, first_column in enumerate(first_columns):
        columns = np.flatnonzero(group_index == group)
        valid_rows = finite[:, first_column]
        if not np.any(valid_rows):
            continue
        source_temperature = temperature_grid[valid_rows]
        source_values = values[np.ix_(valid_rows, columns)]
        if source_temperature.size == 1:
            result[:, columns] = source_values[0]
            continue

        upper = np.searchsorted(source_temperature, targets, side="left")
        below = upper == 0
        above = upper == source_temperature.size
        lower_index = np.clip(upper - 1, 0, source_temperature.size - 1)
        upper_index = np.clip(upper, 0, source_temperature.size - 1)
        denominator = source_temperature[upper_index] - source_temperature[lower_index]
        fraction = np.divide(
            targets - source_temperature[lower_index],
            denominator,
            out=np.zeros(targets.shape, dtype=float),
            where=denominator != 0,
        )
        interpolated = source_values[lower_index] + fraction[:, None] * (
            source_values[upper_index] - source_values[lower_index]
        )
        interpolated[below] = source_values[0]
        interpolated[above] = source_values[-1]
        result[:, columns] = interpolated
    return result


def _interp_1d_sorted(
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    return np.interp(target, x_sorted, y_sorted, left=fill_value, right=fill_value)
