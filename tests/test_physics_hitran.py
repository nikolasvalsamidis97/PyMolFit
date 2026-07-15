import numpy as np
import pytest
from astropy.table import Table

import pymolfit.components as components_module
from pymolfit import (
    AtmosphereProfile,
    FitConfig,
    IsotopologueMetadata,
    LineList,
    PartitionTable,
    PhysicalModelConfig,
    Spectrum,
    fit_tellurics,
    physical_transmission_model,
    read_aer_line_file,
    read_hitran_par,
)
from pymolfit.physics import (
    lblrtm_dynamic_line_cutoff_cm,
    lblrtm_dynamic_max_line_cutoff_cm,
    lblrtm_layer_wavenumber_spacing_cm,
    lblrtm_voigt_hwhm,
    line_strength_temperature,
)
from pymolfit.components import hitran_line_optical_depth_basis, line_wing_effective_cutoff_cm


def _fixed_decimal(value, width, decimals):
    text = f"{value:.{decimals}f}"
    if text.startswith("0"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text[2:]
    return f"{text:>{width}}"[-width:]


def test_lblrtm_layer_spacing_matches_mane_calc_dv_reference():
    # Layer 1 of the captured Molecfit/LBLRTM H2O J-band TAPE6 audit.
    spacing = lblrtm_layer_wavenumber_spacing_cm(
        8816.5,
        709.82703 / 1013.25,
        284.20,
        h2o_fraction=9.873542e-3,
    )

    np.testing.assert_allclose(spacing, 0.0080432756141004, rtol=2.0e-14)


def test_lblrtm_layer_spacing_validates_physical_inputs():
    with pytest.raises(ValueError, match="temperature_k"):
        lblrtm_layer_wavenumber_spacing_cm(8800.0, 0.7, 0.0)
    with pytest.raises(ValueError, match="h2o_fraction"):
        lblrtm_layer_wavenumber_spacing_cm(8800.0, 0.7, 280.0, h2o_fraction=1.1)


def _hitran_row(
    *,
    mol_id=1,
    iso_id=1,
    wavenumber=4320.0,
    intensity=1.0e-24,
    air_width=0.07,
    self_width=0.30,
    lower_energy=100.0,
    n_air=0.75,
    pressure_shift=-0.001,
):
    row = (
        f"{mol_id:2d}"
        f"{iso_id:1d}"
        f"{wavenumber:12.6f}"
        f"{intensity:10.3E}"
        f"{1.0:10.3E}"
        f"{_fixed_decimal(air_width, 5, 4)}"
        f"{_fixed_decimal(self_width, 5, 4)}"
        f"{lower_energy:10.4f}"
        f"{n_air:4.2f}"
        f"{_fixed_decimal(pressure_shift, 8, 6)}"
    )
    return row + " " * (160 - len(row))


def _aer_row_with_broadener_flags(row, flags):
    flag_text = f"{0:2d}" + "".join(f"{flag:2d}" for flag in flags) + f"{0.0:9.5f}"
    return row[:98] + flag_text + row[98 + len(flag_text):]


def _aer_row_with_f100_flag(row, flag):
    return row[:98] + f"{flag:2d}" + row[100:]


def _aer_line_coupling_aux_row(values, flag=-1):
    if len(values) != 8:
        raise ValueError("values must contain 8 line-coupling fields")
    return "  " + "".join(f"{value:13.6e}" for value in values) + f"{flag:2d}"


def _aer_broadener_aux_row(mol_id, values):
    if len(values) != 21:
        raise ValueError("values must contain 21 broadener fields")
    return f"{mol_id:2d}" + "".join(f"{value:8.4f}" for value in values)


def test_read_hitran_par(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row() + "\n")

    line_list = read_hitran_par(path)

    assert line_list.has_hitran_parameters
    assert line_list.line_source == "hitran_par"
    assert line_list.pressure_shift_convention == "hitran"
    assert line_list.species.tolist() == ["H2O"]
    np.testing.assert_allclose(line_list.wavenumber, [4320.0])
    np.testing.assert_allclose(line_list.air_width, [0.07])


def test_read_aer_line_file_skips_headers_and_auxiliary_rows(tmp_path):
    path = tmp_path / "aer_lines.dat"
    auxiliary_row = " 7 0.500000E+00 0.0000E+00 5.000000E-01 0.0000E+00 5.000000E-01"
    path.write_text(
        "> AER header\n"
        "%%%%%%%%\n"
        f"{auxiliary_row}\n"
        f"{_hitran_row(mol_id=7, iso_id=1, wavenumber=4320.0, intensity=1.0e-24)}\n"
    )

    line_list = read_aer_line_file(path, species={"O2"})

    assert line_list.has_hitran_parameters
    assert line_list.line_source == "aer_lnfl_tape8"
    assert line_list.pressure_shift_convention == "lblrtm_density"
    assert line_list.species.tolist() == ["O2"]
    np.testing.assert_allclose(line_list.wavenumber, [4320.0])
    np.testing.assert_allclose(line_list.iso_id, [1])


def test_read_aer_line_file_uses_lblrtm_isotopologue_masses(tmp_path):
    path = tmp_path / "aer_h2o_isotopologues.dat"
    path.write_text(
        _hitran_row(mol_id=1, iso_id=1, wavenumber=4319.0) + "\n"
        + _hitran_row(mol_id=1, iso_id=2, wavenumber=4320.0) + "\n"
        + _hitran_row(mol_id=1, iso_id=3, wavenumber=4321.0) + "\n"
    )

    line_list = read_aer_line_file(path)

    np.testing.assert_allclose(line_list.molecular_mass_amu, [18.01, 20.01, 19.01])
    np.testing.assert_allclose(line_list.iso_id, [1, 2, 3])


def test_read_aer_line_file_selects_disjoint_wavenumber_ranges(tmp_path):
    path = tmp_path / "aer_windows.dat"
    path.write_text(
        "\n".join(
            [
                _hitran_row(wavenumber=2000.0),
                _hitran_row(wavenumber=3000.0),
                _hitran_row(wavenumber=4000.0),
            ]
        )
        + "\n"
    )

    line_list = read_aer_line_file(
        path,
        wavenumber_ranges=((1990.0, 2010.0), (3990.0, 4010.0)),
    )

    np.testing.assert_allclose(line_list.wavenumber, [2000.0, 4000.0])


def test_read_aer_line_file_reads_optional_broadener_rows(tmp_path):
    path = tmp_path / "aer_broadener_lines.dat"
    flags = [1, 0, 0, 0, 0, 0, 0]
    values = [0.5, 0.2, 0.03] + [0.0] * 18
    path.write_text(
        _aer_row_with_broadener_flags(
            _hitran_row(mol_id=2, iso_id=1, wavenumber=4320.0, intensity=1.0e-24),
            flags,
        )
        + "\n"
        + _aer_broadener_aux_row(2, values)
        + "\n"
    )

    line_list = read_aer_line_file(path)

    assert line_list.has_broadener_parameters
    np.testing.assert_array_equal(line_list.broadener_flags[0], flags)
    np.testing.assert_allclose(line_list.broadener_widths[0, 0], 0.5)
    np.testing.assert_allclose(line_list.broadener_temperature_exponents[0, 0], 0.2)
    np.testing.assert_allclose(line_list.broadener_pressure_shifts[0, 0], 0.03)


def test_read_aer_line_file_reads_f100_negative_line_coupling_rows(tmp_path):
    path = tmp_path / "aer_line_coupling.dat"
    main_row = _aer_row_with_f100_flag(
        _hitran_row(mol_id=7, iso_id=1, wavenumber=4320.0, intensity=1.0e-24),
        -1,
    )
    aux_values = [1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4]
    path.write_text(main_row + "\n" + _aer_line_coupling_aux_row(aux_values, flag=-1) + "\n")

    line_list = read_aer_line_file(path)

    np.testing.assert_array_equal(line_list.line_flags, [1])
    np.testing.assert_allclose(line_list.line_coupling_a[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(line_list.line_coupling_b[0], [0.1, 0.2, 0.3, 0.4])


def test_read_aer_line_file_imports_extra_broadener_files(tmp_path):
    line_path = tmp_path / "aer_lines.dat"
    line_path.write_text(
        _hitran_row(mol_id=2, iso_id=1, wavenumber=4320.0, intensity=1.0e-24)
        + "\n"
    )
    (tmp_path / "co2_h2o_brd_param").write_text("0 4320.000000 0.5100 0.2300 0.0040\n")

    line_list = read_aer_line_file(line_path)

    h2o_index = 0
    assert line_list.has_broadener_parameters
    assert line_list.broadener_flags[0, h2o_index] == 1
    np.testing.assert_allclose(line_list.broadener_widths[0, h2o_index], 0.51)
    np.testing.assert_allclose(line_list.broadener_temperature_exponents[0, h2o_index], 0.23)
    np.testing.assert_allclose(line_list.broadener_pressure_shifts[0, h2o_index], 0.004)


def test_read_hitran_par_filters_strength_and_max_lines(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(
        "\n".join(
            [
                _hitran_row(wavenumber=4320.0, intensity=1.0e-24),
                _hitran_row(wavenumber=4321.0, intensity=5.0e-25),
                _hitran_row(wavenumber=4322.0, intensity=2.0e-24),
            ]
        )
        + "\n"
    )

    filtered = read_hitran_par(path, min_strength=9.0e-25, max_lines=1)

    np.testing.assert_allclose(filtered.wavenumber, [4322.0])
    np.testing.assert_allclose(filtered.strength, [2.0e-24])


def test_hitran_line_list_table_roundtrip(tmp_path):
    par_path = tmp_path / "h2o.par"
    table_path = tmp_path / "h2o.ecsv"
    par_path.write_text(_hitran_row() + "\n")
    line_list = read_hitran_par(par_path)

    line_list.write(table_path)
    loaded = LineList.from_table(table_path)

    assert loaded.has_hitran_parameters
    assert loaded.line_source == "hitran_par"
    np.testing.assert_allclose(loaded.wavenumber, line_list.wavenumber)
    np.testing.assert_allclose(loaded.molecular_mass_amu, line_list.molecular_mass_amu)


def test_aer_line_coupling_table_roundtrip(tmp_path):
    path = tmp_path / "aer_line_coupling.dat"
    table_path = tmp_path / "aer_line_coupling.ecsv"
    main_row = _aer_row_with_f100_flag(
        _hitran_row(mol_id=7, iso_id=1, wavenumber=4320.0, intensity=1.0e-24),
        -3,
    )
    aux_values = [0.01, 0.001, 0.02, 0.002, 0.03, 0.003, 0.04, 0.004]
    path.write_text(main_row + "\n" + _aer_line_coupling_aux_row(aux_values, flag=-3) + "\n")

    read_aer_line_file(path).write(table_path)
    loaded = LineList.from_table(table_path)

    np.testing.assert_array_equal(loaded.line_flags, [3])
    np.testing.assert_allclose(loaded.line_coupling_a, [[0.01, 0.02, 0.03, 0.04]])
    np.testing.assert_allclose(loaded.line_coupling_b, [[0.001, 0.002, 0.003, 0.004]])



def test_aer_line_source_uses_lblrtm_density_pressure_shift(tmp_path):
    row = _hitran_row(
        wavenumber=4320.0,
        intensity=1.0e-21,
        air_width=0.001,
        self_width=0.001,
        pressure_shift=0.1,
    )
    hitran_path = tmp_path / "h2o.par"
    aer_path = tmp_path / "aer_lines.dat"
    hitran_path.write_text(row + "\n")
    aer_path.write_text(row + "\n")
    hitran_lines = read_hitran_par(hitran_path)
    aer_lines = read_aer_line_file(aer_path)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.5,
        temperature_k=148.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-4},
    )
    wavenumber_grid = np.array([4320.05, 4320.10])
    wavelength = 1.0e4 / wavenumber_grid

    _, hitran_basis = hitran_line_optical_depth_basis(wavelength, hitran_lines, atmosphere, chunk_size=1)
    _, aer_basis = hitran_line_optical_depth_basis(wavelength, aer_lines, atmosphere, chunk_size=1)

    assert hitran_basis[0, 0] > hitran_basis[0, 1]
    assert aer_basis[0, 1] > aer_basis[0, 0]


@pytest.mark.parametrize(
    ("line_wing_mode", "line_cutoff_cm", "line_taper_cm"),
    [
        ("hard_cutoff", 0.35, 0.0),
        ("subtracted_cutoff", 0.35, 0.0),
        ("tapered_cutoff", 0.35, 0.05),
        ("lblrtm_dynamic", None, 0.0),
    ],
)
def test_sparse_finite_voigt_matches_dense_path(
    monkeypatch,
    line_wing_mode,
    line_cutoff_cm,
    line_taper_cm,
):
    wavenumber = np.array([4320.15, 4319.95, 4320.28, 4319.82])
    lines = LineList(
        wavelength=1.0e4 / wavenumber,
        strength=np.array([1.0e-21, 7.0e-22, 3.0e-22, 5.0e-22]),
        sigma=np.full(4, 1.0e-5),
        gamma=np.full(4, 1.0e-5),
        species=np.array(["H2O", "CO2", "H2O", "CO2"]),
        wavenumber=wavenumber,
        mol_id=np.array([1, 2, 1, 2]),
        iso_id=np.ones(4, dtype=int),
        air_width=np.array([0.07, 0.06, 0.08, 0.05]),
        self_width=np.array([0.30, 0.10, 0.35, 0.11]),
        lower_state_energy=np.array([100.0, 80.0, 140.0, 50.0]),
        temperature_exponent=np.array([0.75, 0.70, 0.72, 0.68]),
        pressure_shift=np.array([-0.001, 0.0, 0.001, -0.0005]),
        molecular_mass_amu=np.array([18.0, 44.0, 18.0, 44.0]),
    )
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.7,
        temperature_k=285.0,
        path_length_m=1500.0,
        mixing_ratios={"H2O": 2.0e-3, "CO2": 4.2e-4},
    )
    grid_wavenumber = np.linspace(4319.65, 4320.45, 151)
    wavelength = 1.0e4 / grid_wavenumber

    monkeypatch.setattr(components_module, "_USE_SPARSE_FINITE_VOIGT", False)
    dense_names, dense_basis = hitran_line_optical_depth_basis(
        wavelength,
        lines,
        atmosphere,
        chunk_size=2,
        line_wing_mode=line_wing_mode,
        line_cutoff_cm=line_cutoff_cm,
        line_taper_cm=line_taper_cm,
    )
    monkeypatch.setattr(components_module, "_USE_SPARSE_FINITE_VOIGT", True)
    sparse_names, sparse_basis = hitran_line_optical_depth_basis(
        wavelength,
        lines,
        atmosphere,
        chunk_size=2,
        line_wing_mode=line_wing_mode,
        line_cutoff_cm=line_cutoff_cm,
        line_taper_cm=line_taper_cm,
    )

    assert sparse_names == dense_names
    np.testing.assert_allclose(sparse_basis, dense_basis, rtol=5.0e-13, atol=1.0e-20)


def test_aer_broadener_parameters_affect_line_wings(tmp_path):
    base_path = tmp_path / "co2_base.dat"
    broadener_path = tmp_path / "co2_h2o_broadener.dat"
    base_row = _hitran_row(
        mol_id=2,
        iso_id=1,
        wavenumber=4320.0,
        intensity=1.0e-20,
        air_width=0.001,
        self_width=0.001,
        pressure_shift=0.0,
    )
    flags = [1, 0, 0, 0, 0, 0, 0]
    values = [0.5, 0.0, 0.0] + [0.0] * 18
    base_path.write_text(base_row + "\n")
    broadener_path.write_text(
        _aer_row_with_broadener_flags(base_row, flags)
        + "\n"
        + _aer_broadener_aux_row(2, values)
        + "\n"
    )
    base_lines = read_aer_line_file(base_path)
    broadener_lines = read_aer_line_file(broadener_path)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=1000.0,
        mixing_ratios={"CO2": 1.0e-4, "H2O": 0.5},
    )
    wavelength = 1.0e4 / np.array([4319.5])

    _, base_basis = hitran_line_optical_depth_basis(wavelength, base_lines, atmosphere, chunk_size=1)
    _, broadener_basis = hitran_line_optical_depth_basis(wavelength, broadener_lines, atmosphere, chunk_size=1)

    assert broadener_basis[0, 0] > base_basis[0, 0]


def test_lblrtm_line_coupling_makes_profile_asymmetric():
    lines = LineList(
        wavelength=np.array([1.0e4 / 4320.0]),
        strength=np.array([1.0e-20]),
        sigma=np.array([1.0e-5]),
        gamma=np.array([1.0e-5]),
        species=np.array(["O2"]),
        wavenumber=np.array([4320.0]),
        mol_id=np.array([7]),
        iso_id=np.array([1]),
        air_width=np.array([0.05]),
        self_width=np.array([0.05]),
        lower_state_energy=np.array([100.0]),
        temperature_exponent=np.array([0.75]),
        pressure_shift=np.array([0.0]),
        molecular_mass_amu=np.array([32.0]),
        line_flags=np.array([1]),
        line_coupling_a=np.array([[0.0, 0.0, 0.02, 0.02]]),
        line_coupling_b=np.zeros((1, 4)),
        line_source="aer_lnfl_tape8",
    )
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=1000.0,
        mixing_ratios={"O2": 0.21},
    )
    wavelength = 1.0e4 / np.array([4319.9, 4320.1])

    _, basis = hitran_line_optical_depth_basis(wavelength, lines, atmosphere, chunk_size=1)

    assert basis[0, 1] > basis[0, 0]


def test_lblrtm_reduced_width_flag_changes_line_wing():
    base_kwargs = dict(
        wavelength=np.array([1.0e4 / 4320.0]),
        strength=np.array([1.0e-20]),
        sigma=np.array([1.0e-5]),
        gamma=np.array([1.0e-5]),
        species=np.array(["O2"]),
        wavenumber=np.array([4320.0]),
        mol_id=np.array([7]),
        iso_id=np.array([1]),
        air_width=np.array([0.05]),
        self_width=np.array([0.05]),
        lower_state_energy=np.array([100.0]),
        temperature_exponent=np.array([0.75]),
        pressure_shift=np.array([0.0]),
        molecular_mass_amu=np.array([32.0]),
        line_source="aer_lnfl_tape8",
    )
    base = LineList(**base_kwargs)
    reduced = LineList(
        **base_kwargs,
        line_flags=np.array([3]),
        line_coupling_a=np.array([[0.0, 0.0, 0.2, 0.2]]),
        line_coupling_b=np.zeros((1, 4)),
    )
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=296.0,
        path_length_m=1000.0,
        mixing_ratios={"O2": 0.21},
    )
    wavelength = 1.0e4 / np.array([4319.5])

    _, base_basis = hitran_line_optical_depth_basis(wavelength, base, atmosphere, chunk_size=1)
    _, reduced_basis = hitran_line_optical_depth_basis(wavelength, reduced, atmosphere, chunk_size=1)

    assert reduced_basis[0, 0] < base_basis[0, 0]


def test_temperature_scaled_line_strength_changes():
    warm = line_strength_temperature(
        np.array([1.0e-24]),
        np.array([4320.0]),
        np.array([100.0]),
        300.0,
    )
    cold = line_strength_temperature(
        np.array([1.0e-24]),
        np.array([4320.0]),
        np.array([100.0]),
        250.0,
    )

    assert np.all(np.isfinite(warm))
    assert np.all(warm > 0)
    assert np.abs(warm[0] - cold[0]) / warm[0] > 0.01


def test_partition_table_interpolates_q_values(tmp_path):
    path = tmp_path / "partition.ecsv"
    table = Table()
    table["mol_id"] = [1, 1, 1]
    table["iso_id"] = [1, 1, 1]
    table["temperature_k"] = [200.0, 300.0, 400.0]
    table["q"] = [100.0, 200.0, 400.0]
    table.write(path, format="ascii.ecsv")

    partition = PartitionTable.from_table(path)
    q = partition.value(np.array([1]), np.array([1]), 300.0)
    missing = partition.value(np.array([2]), np.array([1]), 300.0)
    roundtrip_path = tmp_path / "partition_roundtrip.ecsv"
    partition.write(roundtrip_path)
    loaded = PartitionTable.from_table(roundtrip_path)

    np.testing.assert_allclose(q, [200.0])
    assert np.isnan(missing[0])
    np.testing.assert_allclose(loaded.q, partition.q)


def test_lblrtm_package_partition_uses_source_tips_table():
    partition = PartitionTable.from_lblrtm_package_data()

    # O2 isotopologue 1 values in LBLRTM 12.11 QT_O2 at table nodes.
    np.testing.assert_allclose(partition.value(np.array([7]), np.array([1]), 285.0), [207.75])
    np.testing.assert_allclose(partition.value(np.array([7]), np.array([1]), 310.0), [226.00])
    assert partition.interpolation == "lblrtm_lagrange"


def test_partition_table_reads_hitran_q_directory(tmp_path):
    q_dir = tmp_path / "q"
    q_dir.mkdir()
    (q_dir / "q1.txt").write_text("200 100\n296 200\n400 500\n")
    metadata = IsotopologueMetadata(
        global_iso_id=np.array([1]),
        mol_id=np.array([1]),
        iso_id=np.array([1]),
        abundance=np.array([0.997317]),
        molar_mass=np.array([18.010565]),
        q296=np.array([200.0]),
        q_file=np.array(["q1.txt"]),
        formula=np.array(["H2O"]),
    )

    partition = PartitionTable.from_hitran_q_directory(q_dir, metadata)

    np.testing.assert_allclose(partition.value(np.array([1]), np.array([1]), 296.0), [200.0])


def test_isotopologue_metadata_reads_hitran_iso_meta_html(tmp_path):
    path = tmp_path / "iso.html"
    path.write_text(
        """
        <h4>1: H<sub>2</sub>O</h4>
        <table><thead><tr><th>global ID</th></tr></thead><tbody>
        <tr>
        <td>1</td><td>1</td><td>H<sub>2</sub><sup>16</sup>O</td><td>161</td>
        <td>9.97317&nbsp;×&nbsp;10<sup>-1</sup></td>
        <td>18.010565</td>
        <td>1.7458&nbsp;×&nbsp;10<sup>2</sup></td>
        <td><a href="/data/Q/q1.txt">q1.txt</a></td><td>1</td>
        </tr>
        </tbody></table>
        """,
        encoding="utf-8",
    )

    metadata = IsotopologueMetadata.from_hitran_iso_meta_html(path)

    np.testing.assert_array_equal(metadata.global_iso_id, [1])
    np.testing.assert_array_equal(metadata.mol_id, [1])
    np.testing.assert_array_equal(metadata.iso_id, [1])
    np.testing.assert_allclose(metadata.abundance, [0.997317])
    np.testing.assert_allclose(metadata.q296, [174.58])
    assert metadata.q_file.tolist() == ["q1.txt"]


def test_line_list_attaches_isotopologue_metadata_and_abundance_scale(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row() + "\n")
    metadata = IsotopologueMetadata(
        global_iso_id=np.array([1]),
        mol_id=np.array([1]),
        iso_id=np.array([1]),
        abundance=np.array([0.997317]),
        molar_mass=np.array([18.010565]),
        q296=np.array([174.58]),
        q_file=np.array(["q1.txt"]),
    )

    line_list = LineList.from_hitran_par(
        path,
        isotopologue_metadata=metadata,
        abundance_overrides={1: 0.5 * 0.997317},
    )

    assert line_list.has_isotopologue_metadata
    np.testing.assert_array_equal(line_list.global_iso_id, [1])
    np.testing.assert_allclose(line_list.natural_abundance, [0.997317])
    np.testing.assert_allclose(line_list.molecular_mass_amu, [18.010565])
    np.testing.assert_allclose(line_list.isotopologue_abundance_scale, [0.5])


def test_physical_transmission_model_uses_hitran_fields(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=1.0e-23) + "\n")
    line_list = LineList.from_hitran_par(path)
    center_micron = 1.0e4 / 4320.0
    wavelength = np.linspace(center_micron - 0.002, center_micron + 0.002, 250)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    transmission = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(chunk_size=4),
    )

    assert transmission.shape == wavelength.shape
    assert np.nanmin(transmission) < 0.999
    assert np.nanmax(transmission) <= 1.0


def test_physical_transmission_uses_isotopologue_abundance_scale(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=1.0e-23) + "\n")
    metadata = IsotopologueMetadata(
        global_iso_id=np.array([1]),
        mol_id=np.array([1]),
        iso_id=np.array([1]),
        abundance=np.array([0.997317]),
        molar_mass=np.array([18.010565]),
        q296=np.array([174.58]),
    )
    base = LineList.from_hitran_par(path, isotopologue_metadata=metadata)
    half = LineList.from_hitran_par(
        path,
        isotopologue_metadata=metadata,
        abundance_overrides={1: 0.5 * 0.997317},
    )
    center_micron = 1.0e4 / 4320.0
    wavelength = np.linspace(center_micron - 0.002, center_micron + 0.002, 250)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    base_transmission = physical_transmission_model(wavelength, base, atmosphere, PhysicalModelConfig(chunk_size=4))
    half_transmission = physical_transmission_model(wavelength, half, atmosphere, PhysicalModelConfig(chunk_size=4))

    assert np.nanmin(half_transmission) > np.nanmin(base_transmission)


def test_physical_transmission_uses_partition_table(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=1.0e-23) + "\n")
    line_list = LineList.from_hitran_par(path)
    center_micron = 1.0e4 / 4320.0
    wavelength = np.linspace(center_micron - 0.002, center_micron + 0.002, 250)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )
    partition = PartitionTable(
        mol_id=np.array([1, 1, 1]),
        iso_id=np.array([1, 1, 1]),
        temperature_k=np.array([250.0, 296.0, 320.0]),
        q=np.array([10.0, 100.0, 1000.0]),
    )

    approximate = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(chunk_size=4),
    )
    tabulated = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(chunk_size=4, partition_table=partition),
    )

    assert np.nanmin(tabulated) != np.nanmin(approximate)


def test_physical_transmission_applies_line_wing_cutoff(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=5.0e-22, air_width=0.2, self_width=0.4) + "\n")
    line_list = LineList.from_hitran_par(path)
    wavenumber = np.array([4310.0, 4320.0, 4330.0])
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=280.0,
        path_length_m=2000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    full_wing = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(chunk_size=4),
    )
    truncated = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(
            chunk_size=4,
            line_cutoff_cm=2.0,
            subtract_cutoff_profile=True,
            line_taper_cm=0.5,
        ),
    )

    center = np.argmin(np.abs(wavenumber - 4320.0))
    wings = np.array([0, 2])
    assert np.nanmin(truncated[wings]) > np.nanmin(full_wing[wings])
    assert truncated[center] < 1.0


def test_lblrtm_voigt_hwhm_uses_avrat_source_table():
    doppler_sigma_for_hwhm_one = 1.0 / np.sqrt(2.0 * np.log(2.0))

    hwhm = lblrtm_voigt_hwhm(
        np.array([0.0, 1.0, 1.0]),
        np.array([doppler_sigma_for_hwhm_one, 0.0, doppler_sigma_for_hwhm_one]),
    )

    np.testing.assert_allclose(hwhm[0], 1.0)
    np.testing.assert_allclose(hwhm[1], 1.0)
    np.testing.assert_allclose(hwhm[2], 0.81871 * 2.0, rtol=1.0e-7)


def test_lblrtm_dynamic_line_cutoff_matches_alfv_clamps():
    tiny = lblrtm_dynamic_line_cutoff_cm(
        np.array([1.0e-5]),
        np.array([1.0e-5]),
        grid_spacing_cm=0.01,
    )
    broad = lblrtm_dynamic_line_cutoff_cm(
        np.array([1.0]),
        np.array([1.0]),
        grid_spacing_cm=0.01,
    )

    np.testing.assert_allclose(tiny, [64.0 * 0.01])
    np.testing.assert_allclose(broad, [64.0 * 4.0 * 4.0 * 0.01])


def test_lblrtm_dynamic_line_cutoff_alfal0_zero_disables_upper_clamp():
    capped = lblrtm_dynamic_line_cutoff_cm(
        np.array([1.0]),
        np.array([1.0]),
        grid_spacing_cm=0.01,
        alfal0=0.04,
    )
    uncapped = lblrtm_dynamic_line_cutoff_cm(
        np.array([1.0]),
        np.array([1.0]),
        grid_spacing_cm=0.01,
        alfal0=0.0,
    )

    assert uncapped[0] > capped[0]
    assert np.isinf(lblrtm_dynamic_max_line_cutoff_cm(0.01, alfal0=0.0))


def test_lblrtm_dynamic_max_line_cutoff_is_safe_for_line_selection():
    np.testing.assert_allclose(
        lblrtm_dynamic_max_line_cutoff_cm(0.01),
        64.0 * 4.0 * 4.0 * 0.01,
    )
    assert lblrtm_dynamic_max_line_cutoff_cm(0.1) > 25.0


def test_lblrtm_subtracted_line_wing_mode_uses_default_cutoff(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=5.0e-20, pressure_shift=0.0) + "\n")
    line_list = LineList.from_hitran_par(path)
    wavenumber = np.array([4295.0, 4320.0, 4345.0])
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=280.0,
        path_length_m=2000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    _, hard_basis = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        chunk_size=4,
        line_wing_mode="hard_cutoff",
    )
    _, subtracted_basis = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        chunk_size=4,
        line_wing_mode="lblrtm_subtracted",
    )

    assert line_wing_effective_cutoff_cm("lblrtm_subtracted") == 25.0
    assert hard_basis[0, 0] > 0
    assert hard_basis[0, 2] > 0
    np.testing.assert_allclose(subtracted_basis[0, [0, 2]], 0.0, atol=1.0e-30)
    assert subtracted_basis[0, 1] > 0


def test_lblrtm_dynamic_line_wing_mode_uses_per_line_widths(tmp_path):
    narrow_path = tmp_path / "narrow.par"
    broad_path = tmp_path / "broad.par"
    narrow_path.write_text(
        _hitran_row(intensity=5.0e-20, air_width=0.001, self_width=0.001, pressure_shift=0.0) + "\n"
    )
    broad_path.write_text(
        _hitran_row(intensity=5.0e-20, air_width=0.2, self_width=0.2, pressure_shift=0.0) + "\n"
    )
    narrow_lines = LineList.from_hitran_par(narrow_path)
    broad_lines = LineList.from_hitran_par(broad_path)
    wavenumber = np.linspace(4319.0, 4321.0, 201)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=280.0,
        path_length_m=2000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    _, full_basis = hitran_line_optical_depth_basis(
        wavelength,
        narrow_lines,
        atmosphere,
        chunk_size=4,
    )
    _, narrow_dynamic = hitran_line_optical_depth_basis(
        wavelength,
        narrow_lines,
        atmosphere,
        chunk_size=4,
        line_wing_mode="lblrtm_dynamic",
    )
    _, broad_dynamic = hitran_line_optical_depth_basis(
        wavelength,
        broad_lines,
        atmosphere,
        chunk_size=4,
        line_wing_mode="lblrtm_dynamic",
    )

    far = np.argmin(np.abs(wavenumber - 4319.0))
    center = np.argmin(np.abs(wavenumber - 4320.0))
    assert full_basis[0, far] > 0
    np.testing.assert_allclose(narrow_dynamic[0, far], 0.0, atol=1.0e-30)
    assert broad_dynamic[0, far] > narrow_dynamic[0, far]
    assert narrow_dynamic[0, center] > 0


def test_lblrtm_dynamic_does_not_subtract_cutoff_profile_by_default(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=5.0e-20, air_width=0.2, self_width=0.2, pressure_shift=0.0) + "\n")
    line_list = LineList.from_hitran_par(path)
    wavenumber = np.linspace(4319.0, 4321.0, 201)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=1.0,
        temperature_k=280.0,
        path_length_m=2000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    _, dynamic = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        chunk_size=4,
        line_wing_mode="lblrtm_dynamic",
        subtract_cutoff_profile=False,
    )
    _, dynamic_subtracted = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        chunk_size=4,
        line_wing_mode="lblrtm_dynamic",
        subtract_cutoff_profile=True,
    )

    assert np.nanmax(dynamic) > np.nanmax(dynamic_subtracted)
    assert np.nanmedian(dynamic) > np.nanmedian(dynamic_subtracted)


def test_unknown_line_wing_mode_raises(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row() + "\n")
    line_list = LineList.from_hitran_par(path)
    atmosphere = AtmosphereProfile.single_layer(mixing_ratios={"H2O": 1.0e-5})
    wavelength = 1.0e4 / np.array([4320.0])

    with pytest.raises(ValueError, match="line_wing_mode"):
        hitran_line_optical_depth_basis(
            wavelength,
            line_list,
            atmosphere,
            line_wing_mode="not_a_mode",
        )


def test_lblrtm_table_line_wing_mode_runs(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=2.0e-24) + "\n")
    line_list = LineList.from_hitran_par(path)
    wavenumber = np.linspace(4319.0, 4321.0, 120)
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.7,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    _, basis = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        line_wing_mode="lblrtm_table",
    )

    assert basis.shape == (1, wavelength.size)
    assert np.nanmax(basis) > 0


def test_lblrtm_panel_line_wing_mode_runs_on_nonuniform_grid(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=2.0e-24) + "\n")
    line_list = LineList.from_hitran_par(path)
    wavenumber = np.linspace(4319.0, 4321.0, 120)
    wavenumber[30:] += 0.0002
    wavelength = 1.0e4 / wavenumber
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.7,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    _, basis = hitran_line_optical_depth_basis(
        wavelength,
        line_list,
        atmosphere,
        line_wing_mode="lblrtm_panel",
    )

    assert basis.shape == (1, wavelength.size)
    assert np.nanmax(basis) > 0


def test_fit_tellurics_with_physical_hitran_basis(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=6.0e-26) + "\n")
    line_list = LineList.from_hitran_par(path)
    center_micron = 1.0e4 / 4320.0
    wavelength = np.linspace(center_micron - 0.002, center_micron + 0.002, 300)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1200.0,
        mixing_ratios={"H2O": 1.0e-5},
    )
    transmission = physical_transmission_model(
        wavelength,
        line_list,
        atmosphere,
        PhysicalModelConfig(species_scales={"H2O": 1.5}, chunk_size=4),
    )
    flux = 1.1 * transmission
    spectrum = Spectrum(wavelength=wavelength, flux=flux)

    result = fit_tellurics(
        spectrum,
        line_list=line_list,
        config=FitConfig(atmosphere=atmosphere, continuum_order=0, chunk_size=4),
    )

    assert result.success
    assert result.species_scales["H2O"] > 0
    assert np.nanstd(result.corrected.flux / result.continuum - 1.0) < 1.0e-3


def test_fit_tellurics_applies_airmass_to_physical_atmosphere(tmp_path):
    path = tmp_path / "h2o.par"
    path.write_text(_hitran_row(intensity=1.0e-23) + "\n")
    line_list = LineList.from_hitran_par(path)
    center_micron = 1.0e4 / 4320.0
    wavelength = np.linspace(center_micron - 0.002, center_micron + 0.002, 250)
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )
    flux = np.ones_like(wavelength)

    low_airmass = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(
            atmosphere=atmosphere,
            continuum_order=0,
            fixed_species_scales={"H2O": 1.0},
            airmass=1.0,
        ),
    )
    high_airmass = fit_tellurics(
        Spectrum(wavelength=wavelength, flux=flux),
        line_list=line_list,
        config=FitConfig(
            atmosphere=atmosphere,
            continuum_order=0,
            fixed_species_scales={"H2O": 1.0},
            airmass=2.0,
        ),
    )

    assert np.nanmin(high_airmass.transmission) < np.nanmin(low_airmass.transmission)
