from datetime import datetime, timezone

import pytest

from pymolfit import Observation


def test_observation_exports_fits_like_metadata() -> None:
    observation = Observation(
        time=datetime(2025, 9, 28, 7, 44, 51, tzinfo=timezone.utc),
        latitude_deg=-24.627,
        longitude_deg=-70.404,
        altitude_m=2635.0,
        airmass=1.18,
        resolving_power=140_000.0,
        wavelength_frame="topocentric",
        pressure_hpa=743.0,
        temperature_c=11.2,
        relative_humidity_percent=18.0,
        pwv_mm=2.4,
        instrument="ESPRESSO",
    )

    header = observation.to_header({"SPEC_RES": 80_000.0, "ORIGIN": "base"})

    assert header["DATE-OBS"].startswith("2025-09-28T07:44:51")
    assert header["ESO TEL GEOLAT"] == -24.627
    assert header["ESO TEL GEOLON"] == -70.404
    assert header["ESO TEL GEOELEV"] == 2635.0
    assert header["AIRMASS"] == 1.18
    assert header["SPEC_RES"] == 140_000.0
    assert header["SPECSYS"] == "TOPOCENT"
    assert header["ESO TEL AMBI PRES START"] == 743.0
    assert header["ESO TEL AMBI TEMP"] == 11.2
    assert header["ESO TEL AMBI RHUM"] == 18.0
    assert header["PWV"] == 2.4
    assert header["INSTRUME"] == "ESPRESSO"
    assert header["ORIGIN"] == "base"


def test_observation_barycentric_frame_accepts_explicit_velocity() -> None:
    observation = Observation(
        wavelength_frame="barycentric",
        frame_velocity_km_s=-7.5,
    )

    header = observation.to_header()

    assert header["SPECSYS"] == "BARYCENT"
    assert header["BERV"] == -7.5


def test_observation_barycentric_frame_requires_velocity_or_geometry() -> None:
    with pytest.raises(ValueError, match="barycentric wavelengths"):
        Observation(wavelength_frame="barycentric")


def test_observation_barycentric_frame_can_reconstruct_velocity() -> None:
    observation = Observation(
        time="2021-09-13T02:18:06.238",
        latitude_deg=-29.2584,
        longitude_deg=-70.7345,
        altitude_m=2400.0,
        target_ra_deg=86.8212,
        target_dec_deg=-51.0665,
        wavelength_frame="barycentric",
    )

    header = observation.to_header()

    assert header["SPECSYS"] == "BARYCENT"
    assert "BERV" not in header
    assert header["RA"] == 86.8212
    assert header["DEC"] == -51.0665


def test_observation_rejects_ambiguous_heliocentric_frame() -> None:
    with pytest.raises(ValueError, match="heliocentric wavelengths require"):
        Observation(wavelength_frame="heliocentric")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latitude_deg": 91.0}, "latitude_deg"),
        ({"airmass": 0.0}, "airmass"),
        ({"resolving_power": -1.0}, "resolving_power"),
        ({"relative_humidity_percent": 101.0}, "relative_humidity_percent"),
        ({"temperature_c": -273.15}, "temperature_c"),
        ({"pwv_mm": -0.1}, "pwv_mm"),
    ],
)
def test_observation_validates_physical_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Observation(**kwargs)
