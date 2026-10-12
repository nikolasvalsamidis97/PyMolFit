# `Spectrum`

Store one-dimensional observed arrays and their metadata. Use this route when your input is not a self-describing spectrum file.

```python
from pymolfit import Spectrum, correct

spectrum = Spectrum(
    wavelength=wavelength,
    flux=flux,
    wavelength_unit="nm",
    wavelength_medium="air",
    observation=observation,
)
result = correct(spectrum)
```

`wavelength`, `flux` and `observation` in this example are your arrays and observing information. See the [runnable array notebook](../user_guide/03_wavelength_flux_input.ipynb).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `wavelength` | required | One-dimensional wavelength array, in `wavelength_unit`. |
| `flux` | required | Corresponding observed flux; normalisation is not required. |
| `uncertainty` | `None` | Per-pixel standard deviation in the same units as flux. Otherwise the correction estimates local noise. |
| `mask` | `None` | Boolean array: **True means usable**, False excluded. |
| `group_id` | `None` | One physical order/detector identifier per pixel, preserving boundaries after combining arrays. |
| `wavelength_unit` | `"micron"` | Units of the supplied values; common alternatives are `"nm"` and `"angstrom"`. |
| `wavelength_medium` | `"vacuum"` | Existing `"air"` or `"vacuum"` convention. Setting this does not convert values. |
| `meta` | empty mapping | Additional descriptive metadata. Prefer `Observation` for observing conditions. |
| `observation` | `None` | Attached [`Observation`](Observation.md), needed when the file/arrays lack adequate observing metadata. |

All per-pixel arrays must have the same shape. Non-finite samples, non-positive supplied uncertainties and False mask entries are excluded by `spectrum.valid`. Do not remove or replace samples just to hide these flags.

The object copies input arrays. `spectrum.to_unit("nm")` returns a converted object. Use `spectrum.to_vacuum()` or `spectrum.to_air()` to convert air/vacuum values, rather than relabelling them. A prepared object's units and observation are retained by `correct()`.

To update metadata explicitly, use `dataclasses.replace(spectrum, observation=updated_observation)`. To load a FITS file into an object, use `load_spectrum(path)`.
