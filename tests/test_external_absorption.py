import numpy as np
import pytest
from astropy.table import Table

from genmolfit import (
    AtmosphereProfile,
    CO2ContinuumAbsorption,
    H2OContinuumAbsorption,
    HitranCIATable,
    LBLRTMCO2Continuum,
    LBLRTMH2OContinuum,
    LineList,
    N2CIAAbsorption,
    O2CIAAbsorption,
    PhysicalModelConfig,
    TabulatedContinuum,
    correct_arrays,
    physical_transmission_model,
)
from genmolfit.components import PairCIAAbsorption


def _atmosphere():
    return AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=100.0,
        mixing_ratios={"CO2": 4.0e-4, "O2": 0.2, "H2O": 0.01},
    )


def _write_continuum_table(path):
    table = Table()
    table["wavenumber_cm"] = [1000.0, 1010.0, 1020.0, 1000.0, 1010.0, 1020.0]
    table["temperature_k"] = [250.0, 250.0, 250.0, 300.0, 300.0, 300.0]
    table["coefficient"] = [1.0e-28, 2.0e-28, 3.0e-28, 2.0e-28, 4.0e-28, 6.0e-28]
    table.write(path, format="ascii.ecsv")


def _cia_header(pair, wmin, wmax, npt, temp, amax=1.0e-46, res=10.0, comment="synthetic", ref=1):
    return f"{pair:>20}{wmin:10.3f}{wmax:10.3f}{npt:7d}{temp:7.1f}{amax:10.3E}{res:6.3f}{comment:<27}{ref:3d}"


def _write_cia_file(path, pair="O2-O2"):
    rows = [
        _cia_header(pair, 1000.0, 1020.0, 3, 250.0),
        " 1000.0000 1.000E-46",
        " 1010.0000 2.000E-46",
        " 1020.0000 3.000E-46",
        _cia_header(pair, 1000.0, 1020.0, 3, 300.0),
        " 1000.0000 2.000E-46",
        " 1010.0000 4.000E-46",
        " 1020.0000 6.000E-46",
    ]
    path.write_text("\n".join(rows) + "\n")


def test_tabulated_continuum_interpolates_temperature_and_wavenumber(tmp_path):
    path = tmp_path / "co2_continuum.ecsv"
    _write_continuum_table(path)
    continuum = TabulatedContinuum.from_table(path)

    coefficient = continuum.coefficient_at(np.array([1005.0, 1015.0]), 275.0)

    np.testing.assert_allclose(coefficient, [2.25e-28, 3.75e-28])


def test_co2_continuum_component_adds_absorption(tmp_path):
    path = tmp_path / "co2_continuum.ecsv"
    _write_continuum_table(path)
    component = CO2ContinuumAbsorption(TabulatedContinuum.from_table(path))
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 30)

    transmission = physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        _atmosphere(),
        PhysicalModelConfig(components=(component,)),
    )

    assert np.nanmin(transmission) < 1.0
    assert np.nanmax(transmission) <= 1.0


def test_lblrtm_co2_continuum_package_data_adds_lband_absorption():
    continuum = LBLRTMCO2Continuum.from_package_data()
    wavenumber = np.array([2388.0, 2500.0, 2900.0])
    coefficient_296 = continuum.coefficient_at(wavenumber, 296.0)
    coefficient_250 = continuum.coefficient_at(wavenumber, 250.0)

    assert coefficient_296.shape == wavenumber.shape
    assert np.all(coefficient_296 > 0)
    assert coefficient_250[0] != coefficient_296[0]

    wavelength = 1.0e4 / np.linspace(2400.0, 3000.0, 100)
    transmission = physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        _atmosphere(),
        PhysicalModelConfig(components=(CO2ContinuumAbsorption(continuum),)),
    )

    assert np.nanmin(transmission) < 1.0
    assert np.nanmax(transmission) <= 1.0


def test_lblrtm_h2o_continuum_package_data_adds_lband_absorption():
    continuum = LBLRTMH2OContinuum.from_package_data()
    wavenumber = np.array([2400.0, 2800.0, 3200.0])
    self_296, foreign_296 = continuum.absorption_coefficients(
        wavenumber,
        pressure_mbar=1013.25,
        temperature_k=296.0,
        h2o_vmr=0.01,
    )
    self_260, _ = continuum.absorption_coefficients(
        wavenumber,
        pressure_mbar=1013.25,
        temperature_k=260.0,
        h2o_vmr=0.01,
    )

    assert np.all(self_296 > 0)
    assert np.all(foreign_296 > 0)
    assert np.nanmax(np.abs(self_296 - self_260) / self_296) > 0.1

    wavelength = 1.0e4 / np.linspace(2400.0, 3200.0, 100)
    transmission = physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        _atmosphere(),
        PhysicalModelConfig(components=(H2OContinuumAbsorption(continuum),)),
    )

    assert np.nanmin(transmission) < 1.0
    assert np.nanmax(transmission) <= 1.0


def test_hitran_cia_parser_interpolates_blocks(tmp_path):
    path = tmp_path / "O2-O2.cia"
    _write_cia_file(path)

    cia = HitranCIATable.from_hitran_cia(path)
    coefficient = cia.coefficient_at(np.array([1005.0, 1015.0]), 275.0)

    assert cia.pair == ("O2", "O2")
    np.testing.assert_allclose(coefficient, [2.25e-46, 3.75e-46])


def test_hitran_cia_vectorized_temperatures_match_scalar_interpolation(tmp_path):
    path = tmp_path / "O2-O2.cia"
    _write_cia_file(path)
    cia = HitranCIATable.from_hitran_cia(path)
    wavenumber = np.array([999.0, 1005.0, 1015.0, 1021.0])
    temperatures = np.array([220.0, 250.0, 275.0, 300.0, 320.0])

    vectorized = cia.coefficients_at(wavenumber, temperatures)
    scalar = np.vstack([cia.coefficient_at(wavenumber, temperature) for temperature in temperatures])

    np.testing.assert_allclose(vectorized, scalar, rtol=2.0e-15, atol=0.0)


def test_o2_and_n2_cia_components_add_absorption(tmp_path):
    o2_path = tmp_path / "O2-O2.cia"
    n2_path = tmp_path / "N2-N2.cia"
    _write_cia_file(o2_path, "O2-O2")
    _write_cia_file(n2_path, "N2-N2")
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 30)
    atmosphere = _atmosphere()

    o2_transmission = physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        atmosphere,
        PhysicalModelConfig(components=(O2CIAAbsorption(HitranCIATable.from_hitran_cia(o2_path)),)),
    )
    n2_transmission = physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        atmosphere,
        PhysicalModelConfig(components=(N2CIAAbsorption(HitranCIATable.from_hitran_cia(n2_path)),)),
    )

    assert np.nanmin(o2_transmission) < 1.0
    assert np.nanmin(n2_transmission) < 1.0


def test_pair_cia_uses_requested_pair_species(tmp_path):
    path = tmp_path / "O2-N2.cia"
    _write_cia_file(path, "O2-N2")
    cia = HitranCIATable.from_hitran_cia(path)
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 30)

    names, basis = PairCIAAbsorption(cia).optical_depth_basis(wavelength, _atmosphere())

    assert names == ("O2-N2_CIA",)
    assert np.nanmax(basis) > 0


def test_workflow_accepts_external_continuum_and_cia_paths(tmp_path):
    continuum_path = tmp_path / "co2_continuum.ecsv"
    cia_path = tmp_path / "O2-O2.cia"
    _write_continuum_table(continuum_path)
    _write_cia_file(cia_path)
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 100)
    flux = np.ones_like(wavelength)

    result = correct_arrays(
        wavelength,
        flux,
        atmosphere=_atmosphere(),
        co2_continuum=continuum_path,
        o2_cia=cia_path,
        continuum_order=0,
    )

    assert result.success
    assert "CO2" in result.species_scales
    assert "O2_CIA" in result.species_scales


def test_workflow_accepts_packaged_lblrtm_continua():
    wavelength = 1.0e4 / np.linspace(2400.0, 3200.0, 100)
    flux = np.ones_like(wavelength)

    result = correct_arrays(
        wavelength,
        flux,
        atmosphere=_atmosphere(),
        h2o_continuum="lblrtm",
        co2_continuum="lblrtm",
        continuum_order=0,
    )

    assert result.success
    assert "H2O" in result.species_scales
    assert "CO2" in result.species_scales


def test_workflow_accepts_generic_pair_cia_mapping(tmp_path):
    cia_path = tmp_path / "CO2-H2O.cia"
    _write_cia_file(cia_path, "CO2-H2O")
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 100)
    flux = np.ones_like(wavelength)

    result = correct_arrays(
        wavelength,
        flux,
        atmosphere=_atmosphere(),
        cia_tables={"CO2-H2O_CIA": cia_path},
        continuum_order=0,
    )

    assert result.success
    assert "CO2-H2O_CIA" in result.species_scales


def test_workflow_rejects_double_counted_n2_continuum_and_cia(tmp_path):
    cia_path = tmp_path / "N2-N2.cia"
    _write_cia_file(cia_path, "N2-N2")
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 100)

    with pytest.raises(ValueError, match="overlaps N2 collision-induced absorption"):
        correct_arrays(
            wavelength,
            np.ones_like(wavelength),
            atmosphere=_atmosphere(),
            n2_continuum=True,
            n2_cia=cia_path,
            continuum_order=0,
        )


def test_workflow_accepts_source_backed_o2_continuum():
    wavelength = 1.0e4 / np.linspace(12_980.0, 13_200.0, 120)

    result = correct_arrays(
        wavelength,
        np.ones_like(wavelength),
        atmosphere=_atmosphere(),
        o2_continuum=True,
        continuum_order=0,
    )

    assert result.success
    assert "O2_continuum" in result.species_scales
    assert result.species_scales["O2_continuum"] == 1.0


def test_workflow_rejects_double_counted_o2_continuum_and_cia(tmp_path):
    cia_path = tmp_path / "O2-O2.cia"
    _write_cia_file(cia_path, "O2-O2")
    wavelength = 1.0e4 / np.linspace(1000.0, 1020.0, 100)

    with pytest.raises(ValueError, match="overlaps O2 collision-induced absorption"):
        correct_arrays(
            wavelength,
            np.ones_like(wavelength),
            atmosphere=_atmosphere(),
            o2_continuum=True,
            o2_cia=cia_path,
            continuum_order=0,
        )
