import numpy as np
import pytest
from astropy.table import Table

from pymolfit import AtmosphereProfile
from pymolfit.atmosphere import (
    BOLTZMANN_J_PER_K,
    DEFAULT_OBSERVATORY_ALTITUDE_M,
    DEFAULT_OBSERVATORY_LATITUDE_DEG,
    DEFAULT_OBSERVATORY_LONGITUDE_DEG,
    _header_gdas_observation_time,
    _layers_from_atmosphere_levels,
    _merge_mipas_gdas_fixed_levels,
    _molecfit_fixed_height_levels_m,
    lblrtm_lowtran6_refractivity,
)


def test_level_layers_preserve_exponentially_interpolated_lblrtm_amount():
    layers = _layers_from_atmosphere_levels(
        np.array([0.0, 1000.0]),
        np.array([1000.0, 800.0]),
        np.array([280.0, 280.0]),
        {"O2": np.array([0.2, 0.2]), "H2O": np.array([0.0, 0.0])},
        observatory_altitude_m=0.0,
        airmass=1.0,
        spherical_slant=False,
        earth_radius_m=6_371_000.0,
    )

    n0 = 1000.0 * 100.0 / (BOLTZMANN_J_PER_K * 280.0)
    n1 = 800.0 * 100.0 / (BOLTZMANN_J_PER_K * 280.0)
    log_mean_density = (n0 - n1) / np.log(n0 / n1)
    expected_o2_column_cm2 = 0.2 * log_mean_density * 1000.0 / 1.0e4

    np.testing.assert_allclose(layers[0].column_density_cm2("O2"), expected_o2_column_cm2, rtol=1e-12)


def test_atmosphere_profile_table_roundtrip(tmp_path):
    profile = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5, "CO2": 4.0e-4},
    )
    path = tmp_path / "profile.ecsv"

    profile.write(path)
    loaded = AtmosphereProfile.from_table(path)

    assert loaded.species_names == ("CO2", "H2O")
    np.testing.assert_allclose(loaded.total_column_cm2("H2O"), profile.total_column_cm2("H2O"))


def test_atmosphere_table_accepts_hpa_pressure_and_vmr_prefix(tmp_path):
    path = tmp_path / "profile.ecsv"
    table = Table()
    table["pressure_hpa"] = [750.0, 500.0]
    table["temperature_k"] = [280.0, 250.0]
    table["thickness_m"] = [1000.0, 2000.0]
    table["vmr_H2O"] = [1.0e-5, 5.0e-6]
    table.write(path, format="ascii.ecsv")

    profile = AtmosphereProfile.from_table(path, airmass=2.0)

    assert len(profile.layers) == 2
    np.testing.assert_allclose(profile.layers[0].pressure_atm, 750.0 / 1013.25)
    np.testing.assert_allclose(profile.layers[0].path_length_m, 2000.0)
    assert profile.total_column_cm2("H2O") > 0


def test_atmosphere_profile_path_scale_changes_columns():
    profile = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-5},
    )

    scaled = profile.with_path_scale(2.5)

    np.testing.assert_allclose(
        scaled.total_column_cm2("H2O"),
        2.5 * profile.total_column_cm2("H2O"),
    )


def test_standard_midlatitude_uses_spherical_slant_geometry():
    vertical = AtmosphereProfile.standard_midlatitude(airmass=1.0, n_layers=8)
    spherical = AtmosphereProfile.standard_midlatitude(airmass=2.0, n_layers=8)
    plane_parallel = AtmosphereProfile.standard_midlatitude(
        airmass=2.0,
        n_layers=8,
        spherical_slant=False,
    )

    vertical_paths = np.array([layer.path_length_m for layer in vertical.layers])
    spherical_paths = np.array([layer.path_length_m for layer in spherical.layers])
    plane_paths = np.array([layer.path_length_m for layer in plane_parallel.layers])

    assert np.sum(vertical_paths) < np.sum(spherical_paths) < np.sum(plane_paths)
    assert spherical_paths[0] / vertical_paths[0] > spherical_paths[-1] / vertical_paths[-1]


def test_lowtran6_refractivity_matches_lblrtm_expression():
    pressure_hpa = np.array([743.0, 300.0])
    temperature_k = np.array([281.0, 240.0])
    h2o = np.array([0.002, 2.0e-5])
    wavenumber_cm = 16_980.0
    expected = (
        (
            83.42
            + 185.08 / (1.0 - (wavenumber_cm / 1.14e5) ** 2)
            + 4.11 / (1.0 - (wavenumber_cm / 6.24e4) ** 2)
        )
        * (pressure_hpa * 288.15)
        / (1013.25 * temperature_k)
        - (43.49 - (wavenumber_cm / 1.7e4) ** 2)
        * (pressure_hpa * h2o / 1013.25)
    ) * 1.0e-6

    np.testing.assert_allclose(
        lblrtm_lowtran6_refractivity(
            pressure_hpa,
            temperature_k,
            h2o,
            wavenumber_cm,
        ),
        expected,
        rtol=2.0e-15,
    )


def test_lblrtm_refraction_lengthens_spherical_slant_path():
    altitude = np.array([0.0, 1_000.0, 5_000.0, 20_000.0])
    pressure = np.array([1013.25, 900.0, 540.0, 55.0])
    temperature = np.array([288.15, 281.0, 255.0, 217.0])
    ratios = {
        "H2O": np.array([0.01, 0.006, 0.001, 1.0e-6]),
        "O2": np.full(4, 0.2095),
    }
    straight = _layers_from_atmosphere_levels(
        altitude,
        pressure,
        temperature,
        ratios,
        observatory_altitude_m=0.0,
        airmass=2.0,
        spherical_slant=True,
        refracted_slant=False,
        reference_wavenumber_cm=16_980.0,
    )
    refracted = _layers_from_atmosphere_levels(
        altitude,
        pressure,
        temperature,
        ratios,
        observatory_altitude_m=0.0,
        airmass=2.0,
        spherical_slant=True,
        refracted_slant=True,
        reference_wavenumber_cm=16_980.0,
    )

    assert sum(layer.path_length_m for layer in refracted) > sum(
        layer.path_length_m for layer in straight
    )
    np.testing.assert_allclose(
        AtmosphereProfile(refracted).total_vertical_column_cm2("O2"),
        AtmosphereProfile(straight).total_vertical_column_cm2("O2"),
        rtol=1.0e-7,
    )


def test_pwv_scaling_uses_vertical_column_for_slant_profile():
    profile = AtmosphereProfile.standard_midlatitude(airmass=2.0, n_layers=8).with_pwv_mm(2.0)

    np.testing.assert_allclose(profile.total_vertical_column_cm2("H2O"), 2.0 * 3.34556e21)
    assert profile.total_column_cm2("H2O") > profile.total_vertical_column_cm2("H2O")


def test_atmosphere_from_fits_header_uses_observing_metadata():
    header = {
        "ESO TEL AIRM START": 1.4,
        "ESO TEL AIRM END": 1.6,
        "ESO TEL GEOELEV": 2600.0,
        "ESO TEL AMBI PRES START": 743.0,
        "ESO TEL AMBI TEMP": 8.0,
    }

    profile = AtmosphereProfile.from_fits_header(
        header,
        n_layers=10,
        top_altitude_m=3600.0,
        pwv_mm=1.5,
    )

    assert len(profile.layers) == 10
    np.testing.assert_allclose(profile.total_vertical_column_cm2("H2O"), 1.5 * 3.34556e21)
    assert profile.total_column_cm2("H2O") > profile.total_vertical_column_cm2("H2O")
    np.testing.assert_allclose(profile.layers[0].temperature_k, 281.15, rtol=0.01)


def test_mipas_gdas_profile_uses_packaged_profiles():
    profile = AtmosphereProfile.from_mipas_gdas(
        observation_time="2022-01-02T05:17:35",
        observatory_altitude_m=2635.0,
        airmass=1.2,
        gdas_mode="average",
    )

    assert len(profile.layers) > 40
    assert "H2O" in profile.species_names
    assert "CO2" in profile.species_names
    assert "O2" in profile.species_names
    assert profile.layers[0].pressure_atm < 1.0
    assert profile.total_column_cm2("H2O") > profile.total_vertical_column_cm2("H2O")


def test_mipas_gdas_from_fits_header_reads_metadata():
    header = {
        "MJD-OBS": 59581.2,
        "ESO TEL AIRM START": 1.1,
        "ESO TEL AIRM END": 1.3,
        "ESO TEL GEOELEV": 2635.0,
        "ESO TEL GEOLAT": -24.6,
        "ESO TEL GEOLON": -70.4,
        "ESO TEL AMBI PRES START": 743.0,
        "ESO TEL AMBI TEMP": 8.0,
        "ESO TEL AMBI RHUM": 25.0,
    }

    profile = AtmosphereProfile.from_fits_header_mipas_gdas(header, gdas_mode="average")

    assert len(profile.layers) > 40
    assert profile.layers[0].pressure_atm < 743.0 / 1013.25
    np.testing.assert_allclose(profile.layers[0].pressure_atm, 743.0 / 1013.25, rtol=0.03)
    np.testing.assert_allclose(profile.layers[0].temperature_k, 281.15, rtol=0.01)


def test_gdas_time_uses_eso_utc_seconds_like_molecfit():
    header = {
        "MJD-OBS": 57849.00350496,
        "DATE-OBS": "2017-04-06T00:05:02.828",
        "UTC": 298.0,
    }

    resolved = _header_gdas_observation_time(header)

    assert resolved is not None
    assert resolved.isot == "2017-04-06T00:04:58.000"


def test_mipas_gdas_reads_legacy_keck_time_site_and_weather_metadata():
    header = {
        "TELESCOP": "Keck I",
        "DATE-OBS": "2004-08-24",
        "UTC": "05:15:25.65",
        "AIRMASS": 1.01,
        "LATITUDE": 19.82658656,
        "LONGITUD": 155.4722,
        "WXPRESS": 623.3,
        "WXOUTTMP": 5.23,
        "RELHUM": 11.16,
    }

    resolved = _header_gdas_observation_time(header)
    profile = AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        gdas_mode="average",
    )

    assert resolved is not None
    assert resolved.isot == "2004-08-24T05:15:25.650"
    assert profile.metadata["latitude_deg"] == 19.82658656
    assert profile.metadata["longitude_deg"] == -155.4722
    assert profile.metadata["observatory_altitude_m"] == 4145.0
    np.testing.assert_allclose(
        profile.metadata["pressure_at_observatory_atm"],
        623.3 / 1013.25,
    )
    np.testing.assert_allclose(profile.metadata["temperature_at_observatory_k"], 278.38)


def test_mipas_gdas_resolves_modern_kpf_site_and_weather_metadata():
    header = {
        "TELESCOP": "Keck I",
        "OBSERVAT": "KECK",
        "DATE-OBS": "2025-05-19T15:17:09.400",
        "AIRMASS": 1.2,
        "PRES": 621.002,
        "RELH": 10.633,
    }

    profile = AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        gdas_mode="average",
    )

    assert profile.metadata["observation_time_utc"] == "2025-05-19T15:17:09.400"
    assert profile.metadata["observatory_site"] == "W. M. Keck Observatory"
    assert profile.metadata["observatory_site_source"] == "fits:OBSERVAT"
    assert profile.metadata["observatory_coordinate_source"] == "observatory_registry"
    assert profile.metadata["latitude_deg"] == 19.82658656
    assert profile.metadata["longitude_deg"] == -155.4722
    assert profile.metadata["observatory_altitude_m"] == 4145.0
    np.testing.assert_allclose(
        profile.metadata["pressure_at_observatory_atm"],
        621.002 / 1013.25,
    )
    assert profile.metadata["relative_humidity_percent"] == 10.633


def test_explicit_coordinates_override_named_observatory_registry():
    header = {
        "OBSERVAT": "KECK",
        "DATE-OBS": "2025-05-19T15:17:09.400",
        "LAT-OBS": 20.0,
        "LONG-OBS": -156.0,
        "ALT-OBS": 4000.0,
    }

    profile = AtmosphereProfile.from_fits_header_mipas_gdas(
        header,
        gdas_mode="average",
    )

    assert profile.metadata["latitude_deg"] == 20.0
    assert profile.metadata["longitude_deg"] == -156.0
    assert profile.metadata["observatory_altitude_m"] == 4000.0
    assert profile.metadata["observatory_coordinate_source"] == "fits_header"


def test_standard_fits_atmosphere_uses_named_site_and_kpf_weather():
    profile = AtmosphereProfile.from_fits_header(
        {
            "OBSERVAT": "KECK",
            "AIRMASS": 1.2,
            "PRES": 621.002,
            "WXOUTTMP": -0.5,
        }
    )

    assert profile.metadata["observatory_site"] == "W. M. Keck Observatory"
    assert profile.metadata["observatory_altitude_m"] == 4145.0
    np.testing.assert_allclose(
        profile.metadata["pressure_at_observatory_atm"],
        621.002 / 1013.25,
    )
    assert profile.metadata["temperature_at_observatory_k"] == 272.65


def test_mipas_gdas_rejects_unknown_observatory_without_coordinates():
    with pytest.raises(ValueError, match="cannot resolve latitude_deg"):
        AtmosphereProfile.from_fits_header_mipas_gdas(
            {
                "OBSERVAT": "UNREGISTERED TEST SITE",
                "DATE-OBS": "2025-05-19T15:17:09.400",
                "AIRMASS": 1.2,
            },
            gdas_mode="average",
        )


def test_mipas_gdas_accepts_explicit_geometry_for_unknown_observatory():
    profile = AtmosphereProfile.from_fits_header_mipas_gdas(
        {
            "OBSERVAT": "UNREGISTERED TEST SITE",
            "DATE-OBS": "2025-05-19T15:17:09.400",
            "AIRMASS": 1.2,
        },
        latitude_deg=28.3,
        longitude_deg=-16.5,
        observatory_altitude_m=2390.0,
        gdas_mode="average",
    )

    assert profile.metadata["latitude_deg"] == 28.3
    assert profile.metadata["longitude_deg"] == -16.5
    assert profile.metadata["observatory_altitude_m"] == 2390.0
    assert profile.metadata["observatory_site_source"] == "unrecognized:OBSERVAT"
    assert profile.metadata["observatory_coordinate_source"] == "explicit"
    assert profile.metadata["default_observatory_allowed"] is False


def test_mipas_gdas_default_observatory_requires_explicit_opt_in():
    profile = AtmosphereProfile.from_fits_header_mipas_gdas(
        {"DATE-OBS": "2025-05-19T15:17:09.400", "AIRMASS": 1.2},
        allow_default_observatory=True,
        gdas_mode="average",
    )

    assert profile.metadata["observatory_coordinate_source"] == "default_paranal"
    assert profile.metadata["default_observatory_allowed"] is True
    assert profile.metadata["latitude_deg"] == DEFAULT_OBSERVATORY_LATITUDE_DEG
    assert profile.metadata["longitude_deg"] == DEFAULT_OBSERVATORY_LONGITUDE_DEG
    assert profile.metadata["observatory_altitude_m"] == DEFAULT_OBSERVATORY_ALTITUDE_M


def test_standard_fits_atmosphere_rejects_missing_site_altitude():
    with pytest.raises(ValueError, match="cannot resolve observatory_altitude_m"):
        AtmosphereProfile.from_fits_header({"OBSERVAT": "UNREGISTERED TEST SITE"})


def test_atmosphere_perturbation_is_physical_and_traceable():
    profile = AtmosphereProfile.single_layer(
        pressure_atm=0.75,
        temperature_k=280.0,
        path_length_m=1000.0,
        mixing_ratios={"H2O": 1.0e-3, "O2": 0.2},
    )

    perturbed = profile.perturbed(
        pressure_scale=1.02,
        temperature_offset_k=-2.0,
        path_scale=1.01,
        species_scales={"H2O": 1.1},
        label="upper water column",
    )

    layer = perturbed.layers[0]
    assert layer.pressure_atm == 0.75 * 1.02
    assert layer.temperature_k == 278.0
    assert layer.path_length_m == 1010.0
    assert layer.mixing_ratios["H2O"] == pytest.approx(1.1e-3)
    assert layer.mixing_ratios["O2"] == 0.2
    assert perturbed.metadata["systematic_perturbation"]["label"] == "upper water column"


def test_molecfit_fixed_height_grid_matches_source_layout():
    levels = _molecfit_fixed_height_levels_m(
        observatory_altitude_m=2400.0,
        top_altitude_m=120_000.0,
    )

    assert levels.size == 51
    np.testing.assert_allclose(levels[:6], [1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0])
    np.testing.assert_allclose(levels[26:30], [20_000.0, 22_000.0, 24_000.0, 26_000.0])
    np.testing.assert_allclose(levels[-1], 120_000.0)


def test_molecfit_fixed_merge_uses_source_overlap_fractions():
    altitude = np.array([18_000.0, 20_000.0, 22_000.0, 24_000.0, 26_000.0, 30_000.0])
    reference_height = np.array([0.0, 120_000.0])
    mipas = {
        "height_m": reference_height,
        "pressure_hpa": np.full(2, 100.0),
        "temperature_k": np.full(2, 200.0),
        "mixing_ratios": {"H2O": np.full(2, 1.0e-6), "O2": np.full(2, 0.21)},
    }
    gdas = {
        "height_m": np.array([0.0, 30_000.0]),
        "pressure_hpa": np.full(2, 200.0),
        "temperature_k": np.full(2, 300.0),
        "h2o_mixing_ratio": np.full(2, 11.0e-6),
    }

    pressure, temperature, ratios = _merge_mipas_gdas_fixed_levels(
        altitude,
        mipas=mipas,
        gdas=gdas,
    )

    np.testing.assert_allclose(pressure, [200.0, 180.0, 160.0, 140.0, 120.0, 100.0])
    np.testing.assert_allclose(temperature, [300.0, 280.0, 260.0, 240.0, 220.0, 200.0])
    np.testing.assert_allclose(ratios["H2O"] * 1.0e6, [11.0, 9.0, 7.0, 5.0, 3.0, 1.0])
    np.testing.assert_allclose(ratios["O2"], 0.21)
