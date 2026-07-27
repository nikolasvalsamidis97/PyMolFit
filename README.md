# PyMolFit

PyMolFit is an independent pure-Python package for modelling and correcting
telluric absorption in astronomical spectra. It is based on the concepts,
workflow, and radiative-transfer approach of the original
[ESO Molecfit](https://www.eso.org/sci/software/pipelines/skytools/molecfit),
with a Python API for both spectrum files and wavelength-flux arrays.

PyMolFit does not import or run Molecfit and is not affiliated with or endorsed
by ESO.

## Installation

```bash
python -m pip install pymolfit
```

Install plotting support for the tutorial notebooks:

```bash
python -m pip install "pymolfit[plot]" ipympl
```

PyMolFit supports Python 3.10 and newer.

## Correct A Spectrum File

The main API is `correct()`:

```python
from pymolfit import correct

result = correct(
    input_path="spectrum.fits",
    wavelength_medium="air",
)

wavelength = result.corrected.wavelength
corrected_flux = result.corrected.flux
```

Use `wavelength_medium="vacuum"` for vacuum wavelengths. The argument may be
omitted only when the FITS metadata declares the wavelength medium
unambiguously.

Results remain in memory unless an output is requested:

```python
from pymolfit import save_corrected_txt, save_fit_product_ecsv

save_corrected_txt(result, "corrected_spectrum.txt")
save_fit_product_ecsv(result, "fit_product.ecsv")
```

The text file contains a compact corrected spectrum. The ECSV product also
contains the fitted transmission, model, masks, metadata, and provenance.

## Correct Wavelength And Flux Arrays

Arrays do not have a FITS header, so observing metadata is supplied with an
`Observation`:

```python
from pymolfit import Observation, correct

observation = Observation(
    time="2025-09-28T07:44:51",
    latitude_deg=-24.627,
    longitude_deg=-70.404,
    altitude_m=2635.0,
    airmass=1.18,
    resolving_power=140_000,
    wavelength_frame="observatory",
    instrument="ESPRESSO",
)

result = correct(
    wavelength=wavelength,
    flux=flux,
    wavelength_unit="angstrom",
    wavelength_medium="air",
    observation=observation,
)
```

`wavelength_medium` states whether the numbers are air or vacuum wavelengths.
`wavelength_frame` separately states their velocity reference frame. Supported
frames are `observatory`, `barycentric`, and `heliocentric`.

## Tutorials

The runnable notebooks in
[`tutorials/`](https://github.com/nikolasvalsamidis97/PyMolFit/tree/main/tutorials)
cover:

1. [Correcting part of a spectrum](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/01_partial_spectrum.ipynb)
2. [Correcting a full spectrum](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/02_full_spectrum.ipynb)
3. [Expert parameters](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/03_expert_mode.ipynb)
4. [Correcting wavelength-flux arrays](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/04_array_input.ipynb)

Start with Tutorial 1 for a short example or Tutorial 2 for a complete
one-dimensional echelle spectrum.

## Molecular And Atmospheric Data

The standard workflow automatically obtains the versioned AER molecular line
catalogue, verifies its checksum, and caches it under
`~/.cache/pymolfit/aer`. It is not bundled in the Python wheel, and the normal
workflow does not require a HITRAN API key.

Users may instead provide their own HITRAN `.par` files or other supported
line and continuum data.

## Scientific Basis And References

PyMolFit builds on the methods and scientific data developed by several
projects. Publications using PyMolFit should cite the relevant upstream
resources:

- ESO Molecfit:
  [Smette et al. 2015, A&A 576, A77](https://doi.org/10.1051/0004-6361/201423932)
  and
  [Kausch et al. 2015, A&A 576, A78](https://doi.org/10.1051/0004-6361/201423909)
- [AER LBLRTM](https://github.com/AER-RC/LBLRTM) line-by-line
  radiative-transfer methods and data
- [AER MT_CKD](https://github.com/AER-RC/MT_CKD) atmospheric continuum model
- [HITRAN](https://hitran.org/) molecular spectroscopic data
- [MIPAS](https://earth.esa.int/eogateway/instruments/mipas) atmospheric
  reference profiles
- [NOAA GDAS](https://www.ncei.noaa.gov/products/weather-climate-models/global-data-assimilation)
  meteorological profiles

PyMolFit uses [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and
[Astropy](https://www.astropy.org/).

## Development

```bash
git clone https://github.com/nikolasvalsamidis97/PyMolFit.git
cd PyMolFit
python -m pip install -e ".[dev,plot]"
python -m pytest
```

Report problems through the
[GitHub issue tracker](https://github.com/nikolasvalsamidis97/PyMolFit/issues).

## Licensing And Third-Party Data

PyMolFit's original Python code is available under the
[BSD 3-Clause License](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/LICENSE).
Scientific datasets and derived coefficient tables retain their upstream
terms and attribution requirements. See
[`THIRD_PARTY_NOTICES.md`](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/THIRD_PARTY_NOTICES.md)
for the included and automatically downloaded data.
