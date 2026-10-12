# `Observation`

Describe where, when and in which wavelength frame a spectrum was observed. Attach it to `Spectrum(observation=...)`; it is not a second input to `correct()`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `time` | `None` | Observation time: UTC ISO string, `datetime` or Astropy `Time`. |
| `latitude_deg` | `None` | Observatory latitude, degrees north; southern values negative. |
| `longitude_deg` | `None` | Observatory longitude, degrees east; western values negative. |
| `altitude_m` | `None` | Observatory altitude in metres. |
| `airmass` | `None` | Line-of-sight airmass used in atmospheric path construction. |
| `resolving_power` | `None` | Instrumental resolving power, wavelength divided by instrumental FWHM. |
| `wavelength_frame` | `None` | Existing frame: `"observatory"` (`"topocentric"`), `"barycentric"` or `"heliocentric"`. This does not mean stellar rest frame. |
| `frame_velocity_km_s` | `None` | Velocity correction already applied to the wavelengths, in km/s; not the stellar radial velocity. Use the instrument's documented convention. |
| `target_ra_deg` | `None` | Target right ascension in degrees, not hours. |
| `target_dec_deg` | `None` | Target declination in degrees. |
| `pressure_hpa` | `None` | Local surface pressure in hPa, not atmospheric pressure in atm. |
| `temperature_c` | `None` | Local air temperature in degrees Celsius. |
| `relative_humidity_percent` | `None` | Local relative humidity, 0--100 percent. |
| `pwv_mm` | `None` | Precipitable water vapour column in millimetres. |
| `instrument` | `None` | Instrument name. |
| `metadata` | empty mapping | Original FITS-like keywords or extra instrument metadata. Explicit fields override corresponding metadata when converted to a header. |

`None` means unspecified, not a generic measurement. Supply the real site, time and airmass for array data. Surface weather is optional; it refines the lower atmosphere when supplied.

Barycentric wavelengths require either `frame_velocity_km_s` or enough time, site and target coordinates to calculate the correction. Heliocentric wavelengths require the explicit velocity. Observatory-frame wavelengths do not use `frame_velocity_km_s`.

For the bundled HARPS crop:

```python
from pymolfit import Observation

observation = Observation(
    time="2017-04-06T00:05:02.828",
    latitude_deg=-29.2584,
    longitude_deg=-70.7345,
    altitude_m=2400.0,
    airmass=1.2335,
    resolving_power=115000.0,
    wavelength_frame="barycentric",
    frame_velocity_km_s=-7.52937637262916,
)
```

These are example-specific values, not defaults for other observations. FITS users normally let `load_spectrum()` retain the header in `spectrum.observation.metadata` instead of typing these fields.
