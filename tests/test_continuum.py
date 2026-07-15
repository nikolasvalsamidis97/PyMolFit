import numpy as np
from scipy.io import netcdf_file

from genmolfit import (
    AtmosphereProfile,
    LBLRTMN2FundamentalContinuum,
    LBLRTMN2OvertoneContinuum,
    LBLRTMO2Continuum,
    MTCKDH2OContinuum,
    N2RototranslationalContinuumAbsorption,
    O2ContinuumAbsorption,
    PhysicalModelConfig,
    correct_arrays,
    physical_transmission_model,
)
from genmolfit.continuum import (
    LBLRTM_CONTNM_LOSCHMIDT_CM3,
    LBLRTM_N2_ROT_SF296,
    LBLRTM_N2_ROT_T296,
    cubic_interpolate_regular,
    lblrtm_n2_rototranslational_coefficients,
    lblrtm_n2_rototranslational_optical_depth,
    lblrtm_rayleigh_optical_depth,
    radiation_term_cm,
)
from genmolfit.linelist import LineList


def _write_continuum_file(path):
    wavenumber = np.arange(100.0, 180.0, 10.0, dtype=float)
    with netcdf_file(path, mode="w") as dataset:
        dataset.createDimension("wavenumbers", wavenumber.size)
        wv = dataset.createVariable("wavenumbers", "d", ("wavenumbers",))
        wv[:] = wavenumber
        self_ref = dataset.createVariable("self_absco_ref", "d", ("wavenumbers",))
        self_ref[:] = np.linspace(1.0e-24, 8.0e-24, wavenumber.size)
        foreign_ref = dataset.createVariable("for_absco_ref", "d", ("wavenumbers",))
        foreign_ref[:] = np.linspace(2.0e-25, 9.0e-25, wavenumber.size)
        closure_ref = dataset.createVariable("for_closure_absco_ref", "d", ("wavenumbers",))
        closure_ref[:] = np.linspace(3.0e-25, 1.0e-24, wavenumber.size)
        texp = dataset.createVariable("self_texp", "d", ("wavenumbers",))
        texp[:] = np.linspace(2.0, 3.0, wavenumber.size)
        pressure = dataset.createVariable("ref_press", "d", ())
        pressure.data[...] = 1013.0
        temperature = dataset.createVariable("ref_temp", "d", ())
        temperature.data[...] = 296.0
        dataset.Title = b"Synthetic MT_CKD test file"


def test_radiation_term_matches_lblrtm_branches():
    wavenumber = np.array([0.1, 1000.0, 10000.0])
    term = radiation_term_cm(wavenumber, 296.0)

    assert term[0] < wavenumber[0]
    assert 0 < term[1] < wavenumber[1]
    np.testing.assert_allclose(term[2], wavenumber[2], rtol=1.0e-8)


def test_lblrtm_rayleigh_matches_contnm_formula_and_threshold():
    wavenumber = np.array([800.0, 1000.0])
    column = 2.5e25

    tau = lblrtm_rayleigh_optical_depth(wavenumber, column)

    x = 1000.0 / 1.0e4
    conv_cm2mol = 1.0e-20 / (2.68675e-1 * 1.0e5)
    expected = x**4 / (9.38076e2 - 10.8426 * x**2) * conv_cm2mol * column
    assert tau[0] == 0.0
    np.testing.assert_allclose(tau[1], expected)


def test_rayleigh_component_adds_optical_absorption():
    wavenumber = np.linspace(9_000.0, 12_000.0, 80)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=10_000.0,
        mixing_ratios={"H2O": 0.0},
    )
    line_list = LineList.empty_hitran()

    without_rayleigh = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(),
    )
    with_rayleigh = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(rayleigh=True),
    )

    assert np.all(with_rayleigh <= without_rayleigh)
    assert np.nanmin(with_rayleigh) < 1.0


def test_lblrtm_n2_rototranslational_coefficients_match_source_grid():
    coefficient, oxygen_efficiency = lblrtm_n2_rototranslational_coefficients(np.array([0.0]), 296.0)

    np.testing.assert_allclose(coefficient[0], LBLRTM_N2_ROT_T296[2])
    np.testing.assert_allclose(oxygen_efficiency[0], (LBLRTM_N2_ROT_SF296[2] - 1.0) * (0.79 / 0.21))


def test_lblrtm_n2_rototranslational_tau_matches_contnm_formula_without_radiation():
    wavenumber = np.array([0.0])
    n2_column = 2.0e25
    air_amagat = 0.72
    n2_vmr = 0.79
    o2_vmr = 0.21

    tau = lblrtm_n2_rototranslational_optical_depth(
        wavenumber,
        n2_column_cm2=n2_column,
        air_amagat=air_amagat,
        temperature_k=296.0,
        n2_vmr=n2_vmr,
        o2_vmr=o2_vmr,
        h2o_vmr=0.0,
        jrad=0,
    )

    oxygen_efficiency = (LBLRTM_N2_ROT_SF296[2] - 1.0) * (0.79 / 0.21)
    expected = (
        (n2_column / LBLRTM_CONTNM_LOSCHMIDT_CM3)
        * air_amagat
        * LBLRTM_N2_ROT_T296[2]
        * (n2_vmr + oxygen_efficiency * o2_vmr)
    )
    np.testing.assert_allclose(tau[0], expected)


def test_n2_rototranslational_component_adds_far_ir_absorption():
    wavenumber = np.linspace(25.0, 250.0, 80)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=10_000.0,
        mixing_ratios={"H2O": 0.0, "O2": 0.21},
    )
    line_list = LineList.empty_hitran()

    without_n2 = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(),
    )
    with_n2 = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(n2_continuum=True),
    )

    assert np.all(with_n2 <= without_n2)
    assert np.nanmin(with_n2) < 1.0


def test_cubic_interpolate_regular_matches_grid_points():
    x_grid = np.arange(0.0, 80.0, 10.0)
    y_grid = x_grid**2
    x_target = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    interpolated = cubic_interpolate_regular(x_grid, y_grid, x_target)

    np.testing.assert_allclose(interpolated, x_target**2)


def test_mtckd_h2o_reads_netcdf_and_scales_reference_state(tmp_path):
    path = tmp_path / "absco-ref_wv-mt-ckd.nc"
    _write_continuum_file(path)
    continuum = MTCKDH2OContinuum.from_netcdf(path)
    target = np.array([120.0, 130.0, 140.0])

    self_coeff, foreign_coeff = continuum.absorption_coefficients(
        target,
        pressure_mbar=1013.0,
        temperature_k=296.0,
        h2o_vmr=0.01,
        include_radiation_term=False,
    )

    np.testing.assert_allclose(
        self_coeff,
        continuum.self_absco_ref[2:5] * 0.01,
    )
    np.testing.assert_allclose(
        foreign_coeff,
        continuum.foreign_absco_ref[2:5] * 0.99,
    )


def test_mtckd_h2o_temperature_density_and_closure_options(tmp_path):
    path = tmp_path / "absco-ref_wv-mt-ckd.nc"
    _write_continuum_file(path)
    continuum = MTCKDH2OContinuum.from_netcdf(path)
    target = np.array([120.0, 130.0, 140.0])

    cold_self, standard_foreign = continuum.absorption_coefficients(
        target,
        pressure_mbar=506.5,
        temperature_k=250.0,
        h2o_vmr=0.01,
        include_radiation_term=False,
    )
    _, closure_foreign = continuum.absorption_coefficients(
        target,
        pressure_mbar=506.5,
        temperature_k=250.0,
        h2o_vmr=0.01,
        include_radiation_term=False,
        use_foreign_closure=True,
    )

    reference_self, _ = continuum.absorption_coefficients(
        target,
        pressure_mbar=1013.0,
        temperature_k=296.0,
        h2o_vmr=0.01,
        include_radiation_term=False,
    )
    assert np.all(cold_self > 0.5 * reference_self)
    assert np.all(closure_foreign > standard_foreign)


def test_h2o_continuum_adds_to_physical_transmission(tmp_path):
    path = tmp_path / "absco-ref_wv-mt-ckd.nc"
    _write_continuum_file(path)
    continuum = MTCKDH2OContinuum.from_netcdf(path)
    wavenumber = np.linspace(120.0, 150.0, 80)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=5000.0,
        mixing_ratios={"H2O": 0.01},
    )
    line_list = LineList.empty_hitran()

    without_continuum = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(),
    )
    with_continuum = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(h2o_continuum=continuum),
    )

    assert np.all(with_continuum <= without_continuum)
    assert np.nanmin(with_continuum) < 1.0


def test_workflow_can_fit_continuum_only_from_file_path(tmp_path):
    path = tmp_path / "absco-ref_wv-mt-ckd.nc"
    _write_continuum_file(path)
    continuum = MTCKDH2OContinuum.from_netcdf(path)
    wavenumber = np.linspace(120.0, 150.0, 120)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=5000.0,
        mixing_ratios={"H2O": 0.01},
    )
    flux = 1.2 * physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        atmosphere,
        PhysicalModelConfig(h2o_continuum=continuum),
    )

    result = correct_arrays(
        wavelength,
        flux,
        h2o_continuum=path,
        atmosphere=atmosphere,
        continuum_order=0,
    )

    assert result.success
    assert "H2O" in result.species_scales
    assert np.nanstd(result.corrected.flux / result.continuum - 1.0) < 1.0e-6


def test_workflow_can_fit_n2_rototranslational_continuum_only():
    wavenumber = np.linspace(50.0, 240.0, 120)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=5_000.0,
        mixing_ratios={"H2O": 0.0, "O2": 0.21},
    )
    component = N2RototranslationalContinuumAbsorption()
    _, basis = component.optical_depth_basis(wavelength, atmosphere)
    flux = np.exp(-basis[0])

    result = correct_arrays(
        wavelength,
        flux,
        n2_continuum=True,
        atmosphere=atmosphere,
        continuum_order=0,
    )

    assert result.success
    assert "N2_continuum" in result.species_scales
    assert result.species_scales["N2_continuum"] == 1.0
    assert np.nanstd(result.corrected.flux - 1.0) < 1.0e-5


def test_lblrtm_n2_fundamental_source_temperature_and_partner_scaling():
    continuum = LBLRTMN2FundamentalContinuum.from_package_data()
    index = 100
    wavenumber = np.array([continuum.wavenumber_cm[index]])
    n2_n2, n2_o2, n2_h2o = continuum.source_coefficients(272.0)

    coefficient = continuum.mixed_coefficient_at(
        wavenumber,
        temperature_k=272.0,
        n2_vmr=0.78,
        o2_vmr=0.21,
        h2o_vmr=0.01,
        include_radiation_term=False,
    )

    expected = 0.78 * n2_n2[index] + 0.21 * n2_o2[index] + 0.01 * n2_h2o[index]
    np.testing.assert_allclose(coefficient[0], expected, rtol=1.0e-13)


def test_lblrtm_n2_fundamental_h2o_partner_strengthens_l_band_absorption():
    continuum = LBLRTMN2FundamentalContinuum.from_package_data()
    wavenumber = np.array([2500.0])
    dry = continuum.mixed_coefficient_at(
        wavenumber,
        temperature_k=280.0,
        n2_vmr=0.79,
        o2_vmr=0.21,
        h2o_vmr=0.0,
    )
    humid = continuum.mixed_coefficient_at(
        wavenumber,
        temperature_k=280.0,
        n2_vmr=0.78,
        o2_vmr=0.21,
        h2o_vmr=0.01,
    )

    assert humid[0] > dry[0]


def test_lblrtm_n2_overtone_matches_source_grid_and_partner_scaling():
    continuum = LBLRTMN2OvertoneContinuum.from_package_data()
    index = 95
    wavenumber = np.array([continuum.wavenumber_cm[index]])
    common = {
        "n2_column_cm2": LBLRTM_CONTNM_LOSCHMIDT_CM3,
        "air_amagat": 1.0,
        "temperature_k": 296.0,
        "n2_vmr": 0.78,
        "o2_vmr": 0.21,
        "h2o_vmr": 0.01,
        "jrad": 0,
    }

    tau = continuum.optical_depth_layer(wavenumber, **common)

    expected = continuum.n2_n2[index] / continuum.wavenumber_cm[index]
    np.testing.assert_allclose(tau[0], expected, rtol=1.0e-13)


def test_n2_continuum_component_adds_k_band_overtone_absorption():
    wavenumber = np.linspace(4400.0, 4850.0, 100)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=10_000.0,
        mixing_ratios={"H2O": 0.01, "O2": 0.21},
    )
    line_list = LineList.empty_hitran()

    without_n2 = physical_transmission_model(
        wavelength, line_list, atmosphere, PhysicalModelConfig()
    )
    with_n2 = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(n2_continuum=True),
    )

    assert np.all(with_n2 <= without_n2)
    assert np.nanmin(with_n2) < 1.0


def test_lblrtm_o2_continuum_has_absorption_in_all_ground_based_branches():
    continuum = LBLRTMO2Continuum.from_package_data()
    wavenumber = np.array([1550.0, 7900.0, 9400.0, 13080.0, 20000.0])

    tau = continuum.optical_depth_layer(
        wavenumber,
        o2_column_cm2=4.0e24,
        air_column_cm2=2.0e25,
        air_amagat=0.8,
        pressure_mbar=800.0,
        temperature_k=280.0,
        n2_vmr=0.78,
        o2_vmr=0.21,
        h2o_vmr=0.01,
    )

    assert np.all(tau > 0.0)


def test_o2_continuum_component_adds_optical_and_near_ir_absorption():
    wavenumber = np.linspace(12_980.0, 13_200.0, 100)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.8,
        temperature_k=280.0,
        path_length_m=10_000.0,
        mixing_ratios={"H2O": 0.01, "O2": 0.21},
    )
    names, basis = O2ContinuumAbsorption().optical_depth_basis(wavelength, atmosphere)

    assert names == ("O2_continuum",)
    assert np.nanmax(basis) > 0.0
