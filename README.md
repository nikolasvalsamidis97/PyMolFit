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
python -m pip install "pymolfit[interactive]"
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

The same function accepts a previously loaded spectrum:

```python
from pymolfit import correct, load_spectrum

spectrum = load_spectrum("spectrum.fits", wavelength_medium="air")
result = correct(spectrum=spectrum)
```

Use `wavelength_medium="vacuum"` for vacuum wavelengths. The argument may be
omitted only when the FITS metadata declares the wavelength medium
unambiguously.

For table-based echelle products, PyMolFit uses common order/detector columns
to preserve real physical groups and common quality columns such as `QUAL` or
`DQ` to exclude flagged pixels. Wide orders may be divided into smaller
radiative-transfer chunks for memory control, but those chunks continue to
share one smooth wavelength-alignment model for their physical order.

Results remain in memory unless an output is requested:

```python
from pymolfit import load_fit_product, save_corrected_txt, save_fit_product

save_corrected_txt(result, "corrected_spectrum.txt")
save_fit_product(result, "fit_product.ecsv")
reloaded = load_fit_product("fit_product.ecsv")
```

The text file contains a compact corrected spectrum. The ECSV product also
contains the fitted transmission, model, masks, metadata, and provenance.
The printed fit summary reports the resolved atmosphere and line data,
numerical chunks versus physical groups, fitted parameters, masked-pixel
counts, and residual line-alignment diagnostics.

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

## Select Fitting Regions Interactively

PyMolFit can save fit and exclusion windows without manually copying
wavelength values from a plot:

```python
from pymolfit import TheoreticalSpectrum, load_spectrum, select_telluric_regions

stellar_template = TheoreticalSpectrum(
    path="stellar_model.dat",
    radial_velocity_kms=21.3,
    vsini_kms=124.0,
)

spectrum = load_spectrum(
    "spectrum.fits",
    wavelength_medium="air",
)

selector = select_telluric_regions(
    spectrum,
    theoretical_spectrum=stellar_template,
    output_path="telluric_regions.ecsv",
)
```

Candidate telluric transitions from the AER catalogue are marked automatically
and colored by molecule. To create an initial selection automatically, enter a
line count such as `100` and press **Automatic**. PyMolFit proposes fit windows
around the transitions with the strongest expected atmospheric absorption,
using both AER line intensity and the representative atmospheric column of
each molecule. Automatic windows extend 12 sampled detector pixels to either
side of each line by default, retaining the line core, wings, and nearby
continuum needed to constrain alignment and instrumental broadening. It skips
lines in detector/order gaps and merges overlapping windows. These proposals
can be edited like any manual selection.

For manual editing, first zoom or pan to the desired area. Choose **Fit** or
**Exclude**, enable the **Draw regions** checkbox, and drag rectangles around
the lines; only their horizontal wavelength limits are stored. Drawing stays
active until you clear the checkbox. Every stored region is numbered on the
plot and listed in the side panel. Edit the filename field if needed, then press
**Save All** once to write the complete collection. In a Jupyter notebook, run
`%matplotlib widget` before opening the selector.

When a theoretical spectrum is supplied, automatically identified stellar
features appear immediately as red exclusion regions. To edit the physical
template parameters in the selector, pass
`enable_theoretical_controls=True`. The optional panel then exposes radial
velocity, `v sin(i)`, resolving power, mask depth, mask padding, limb
darkening, continuum window, and residual velocity search.

On later runs, the same `output_path` is detected and loaded automatically, so
the selector window is skipped. Pass `reuse_existing=False` to reopen the
saved regions for editing.

The ECSV file records both region types and their wavelength unit and
air/vacuum medium. Apply it directly without transcribing any endpoints:

```python
from pymolfit import correct

result = correct(
    input_path="spectrum.fits",
    wavelength_medium="air",
    region_file="telluric_regions.ecsv",
)
```

`region_file` cannot be combined with explicit `fit_ranges` or
`exclude_ranges`.

## Protect Stellar Features With A Theoretical Spectrum

An optional theoretical stellar spectrum can identify astrophysical lines that
must not bias the telluric fit. The recommended workflow is to pass it to the
region selector as shown above, producing one file containing telluric fit
intervals and stellar exclusions. It can also be supplied directly to
`correct()` when an interactive selector is not needed:

```python
from pymolfit import TheoreticalSpectrum, correct

stellar_template = TheoreticalSpectrum(
    path="stellar_model.dat",
    radial_velocity_kms=21.3,
    vsini_kms=124.0,
    mask_padding_kms="auto",
)

result = correct(
    input_path="spectrum.fits",
    theoretical_spectrum=stellar_template,
    stellar_mask_path="stellar_exclusions.ecsv",
)
```

The input is a two-column ASCII wavelength/flux table. Files from the
[SVO Theoretical Spectra server](https://svo2.cab.inta-csic.es/theory/newov2/)
are supported directly. PyMolFit reads their coordinate metadata, normalizes
physical flux, applies the supplied radial velocity and projected rotation,
and detects stellar features automatically. Those features become hard
exclusions from atmospheric parameter estimation by default. Set
`confidence_weighted_masking=True` on `TheoreticalSpectrum` to opt into
continuous confidence weighting instead. PyMolFit uses
observation metadata for frame and instrumental broadening, and checks a small
residual alignment. The generated stellar model affects parameter estimation
only. It does not replace the observed flux, and atmospheric transmission is
still evaluated inside downweighted or excluded stellar regions.
Automatic mask padding scales with the rotational and instrumental broadening
so the fitted mask also protects broad stellar-line wings. A numeric
`mask_padding_kms` can override that width.

`stellar_mask_path` is optional. When supplied, it saves the generated
exclusions in the input spectrum's original wavelength unit and air/vacuum
medium for inspection.

The selector displays the automatically detected exclusions when a template
is supplied. Its editable theoretical-parameter panel is disabled by default;
pass `enable_theoretical_controls=True` to the selector to show it.

For blended stellar and telluric lines, `correct()` also provides an opt-in
joint forward model:

```python
result = correct(
    input_path="spectrum.fits",
    theoretical_spectrum=stellar_template,
    joint_stellar_model=True,
)
```

This evaluates `continuum * LSF(stellar * atmosphere)` rather than using the
stellar template only as fit weights. The correction divides by the
stellar-aware atmospheric effect, preserving the convolved stellar spectrum.
The mode is disabled by default and is available only through `correct()`.

## Tutorials

The runnable notebooks in
[`tutorials/`](https://github.com/nikolasvalsamidis97/PyMolFit/tree/main/tutorials)
cover:

1. [Correcting part of a spectrum](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/01_partial_spectrum.ipynb)
2. [Correcting a full spectrum](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/02_full_spectrum.ipynb)
3. [Expert parameters](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/03_expert_mode.ipynb)
4. [Correcting wavelength-flux arrays](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/04_array_input.ipynb)
5. [Selecting telluric fit regions interactively](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/tutorials/05_telluric_region_selector.ipynb)

Start with Tutorial 1 for a short example or Tutorial 2 for a complete
one-dimensional echelle spectrum.

## Documentation

- [Software guide and public API](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/docs/software_guide.md)
- [Troubleshooting](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/docs/troubleshooting.md)
- [Physics parity audit](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/docs/physics_parity_audit.md)
- [Changelog](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/nikolasvalsamidis97/PyMolFit/blob/main/CONTRIBUTING.md)

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
