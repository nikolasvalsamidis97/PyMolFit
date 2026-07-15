from dataclasses import dataclass

import numpy as np

from pymolfit import (
    AtmosphereProfile,
    CIABlock,
    FitConfig,
    HitranCIATable,
    LineList,
    PairCIAAbsorption,
    PhysicalModelConfig,
    Spectrum,
    combine_optical_depth_components,
    correct_arrays,
    fit_tellurics,
    physical_transmission_model,
)


@dataclass(frozen=True)
class FixedComponent:
    names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    def optical_depth_basis(self, wavelength_micron, atmosphere, *, species=None):
        requested = None if species is None else set(species)
        rows = np.asarray(self.rows, dtype=float)
        keep = np.ones(len(self.names), dtype=bool)
        if requested is not None:
            keep = np.array([name in requested for name in self.names], dtype=bool)
        if rows.shape[1] != len(wavelength_micron):
            rows = np.tile(rows[:, :1], (1, len(wavelength_micron)))
        return tuple(name for name, use in zip(self.names, keep, strict=True) if use), rows[keep]


def _atmosphere():
    return AtmosphereProfile.single_layer(
        pressure_atm=0.8,
        temperature_k=285.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-3, "CO2": 4.2e-4},
    )


def test_combine_optical_depth_components_sums_matching_species():
    wavelength = np.linspace(2.0, 2.1, 4)
    component_a = FixedComponent(("H2O",), ((0.1, 0.2, 0.3, 0.4),))
    component_b = FixedComponent(("H2O", "CO2"), ((0.5, 0.4, 0.3, 0.2), (1.0, 1.0, 1.0, 1.0)))

    names, basis = combine_optical_depth_components(
        wavelength,
        _atmosphere(),
        (component_a, component_b),
    )

    assert names == ("H2O", "CO2")
    np.testing.assert_allclose(basis[0], [0.6, 0.6, 0.6, 0.6])
    np.testing.assert_allclose(basis[1], [1.0, 1.0, 1.0, 1.0])


def test_combine_optical_depth_components_filters_species():
    wavelength = np.linspace(2.0, 2.1, 4)
    component = FixedComponent(("H2O", "CO2"), ((0.1, 0.2, 0.3, 0.4), (1.0, 1.0, 1.0, 1.0)))

    names, basis = combine_optical_depth_components(
        wavelength,
        _atmosphere(),
        (component,),
        species=("CO2",),
    )

    assert names == ("CO2",)
    np.testing.assert_allclose(basis, [[1.0, 1.0, 1.0, 1.0]])


def test_physical_transmission_model_accepts_custom_components():
    wavelength = np.linspace(2.0, 2.1, 5)
    component = FixedComponent(("H2O",), ((0.1,),))

    transmission = physical_transmission_model(
        wavelength,
        LineList.empty_hitran(),
        _atmosphere(),
        PhysicalModelConfig(
            species_scales={"H2O": 2.0},
            components=(component,),
        ),
    )

    np.testing.assert_allclose(transmission, np.exp(-0.2))


def test_pair_cia_treats_capitalized_air_as_total_air():
    wavelength = np.linspace(3.0, 3.1, 5)
    cia = HitranCIATable(
        blocks=(
            CIABlock(
                pair=("O2", "Air"),
                wavenumber_cm=np.array([3200.0, 3400.0]),
                temperature_k=285.0,
                coefficient_cm5_molecule2=np.array([1.0e-46, 1.0e-46]),
            ),
        )
    )
    atmosphere = AtmosphereProfile.single_layer(
        pressure_atm=0.8,
        temperature_k=285.0,
        path_length_m=1000.0,
        mixing_ratios={"O2": 0.21},
    )

    names, basis = PairCIAAbsorption(cia).optical_depth_basis(wavelength, atmosphere)

    assert names == ("O2-Air_CIA",)
    assert np.all(basis > 0)


def test_fit_tellurics_accepts_custom_components():
    wavelength = np.linspace(2.0, 2.1, 100)
    tau = 0.03 + 0.2 * np.exp(-0.5 * ((wavelength - 2.05) / 0.01) ** 2)
    component = FixedComponent(("H2O",), (tuple(tau),))
    true_scale = 1.7
    continuum = 1.2
    flux = continuum * np.exp(-true_scale * tau)
    spectrum = Spectrum(wavelength=wavelength, flux=flux)

    result = fit_tellurics(
        spectrum,
        line_list=LineList.empty_hitran(),
        config=FitConfig(
            atmosphere=_atmosphere(),
            components=(component,),
            continuum_order=0,
        ),
    )

    assert result.success
    np.testing.assert_allclose(result.species_scales["H2O"], true_scale, rtol=1.0e-3)
    np.testing.assert_allclose(result.continuum[0], continuum, rtol=1.0e-3)


def test_correct_arrays_accepts_custom_components_without_line_list():
    wavelength = np.linspace(2.0, 2.1, 100)
    tau = 0.03 + 0.2 * np.exp(-0.5 * ((wavelength - 2.05) / 0.01) ** 2)
    component = FixedComponent(("H2O",), (tuple(tau),))
    flux = 1.2 * np.exp(-tau)

    result = correct_arrays(
        wavelength,
        flux,
        atmosphere=_atmosphere(),
        components=(component,),
        continuum_order=0,
    )

    assert result.success
    np.testing.assert_allclose(result.corrected.flux / result.continuum, 1.0, rtol=1.0e-4)
