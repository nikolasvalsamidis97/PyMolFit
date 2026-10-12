# PyMolFit

A Python framework for model-based telluric absorption correction. PyMolFit builds an atmospheric transmission model, fits it to selected observed pixels, and corrects the spectrum on its original wavelength grid. It does not execute Molecfit or LBLRTM.

## Install

Requires Python 3.10 or later:

```bash
python -m pip install "pymolfit[plot]"
```

To use the unpublished development version, run this inside your checkout:

```bash
python -m pip install -e ".[plot]"
```

The first correction may download the AER molecular catalogue and atmospheric data. Later runs reuse downloaded data. A HITRAN account is not required.

## Correct a spectrum

```python
from pymolfit import correct

result = correct(
    input_path="spectrum.fits",
    output_path="corrected.ecsv",
    show_plot=True,
)
```

The returned result and saved ECSV contain the observed and corrected spectra, transmission, fitted model, masks and diagnostics. Reopen the product without fitting again:

```python
from pymolfit import load_fit_product, plot_fit

result = load_fit_product("corrected.ecsv")
plot_fit(result, show=True)
```

Input files need reliable wavelength units, air/vacuum convention and observing metadata. Ambiguous FITS extensions or wavelength conventions require an explicit choice. Array users attach their observing conditions to a `Spectrum` object. A supplied stellar template defines fitting exclusions; it is not subtracted from the observation. Additive sky emission is not corrected by this absorption model.

## Documentation

Start with the [documentation index](docs/README.md).

- [Four short notebooks](docs/user_guide/README.md): FITS, FITS with a stellar template, wavelength/flux arrays, and interactive region selection.
- [Parameter descriptions](docs/parameter_descriptions/correct.md): basic `correct()` arguments, followed by `AdvancedOptions`.
- [Spectrum](docs/parameter_descriptions/Spectrum.md), [Observation](docs/parameter_descriptions/Observation.md) and [TheoreticalSpectrum](docs/parameter_descriptions/TheoreticalSpectrum.md).
- [Interactive region selection](docs/user_guide/04_interactive_selector.ipynb).
- [Full-spectrum speed and residual comparison](docs/troubleshooting/faster_correction.ipynb).
- [Before/after and transmission figure scripts](docs/figure_examples/README.md).
- [Mathematical implementation](docs/technical_details/README.md).

## Upgrading to 0.8

This release removes deprecated calling conventions. Pass a path or a
`Spectrum` as `correct(input_path=...)`; put wavelength/flux arrays and an
`Observation` on that `Spectrum`. Put fitting overrides in
`advanced=AdvancedOptions(...)` and cache settings in `cache=CacheConfig(...)`.
Use `show_interactive_window=True` to open the region editor.
The old `spectrum=` alias, flat correction keywords and synthetic demo options
are no longer accepted. Low-level `correct_file()` and `correct_arrays()`
remain available for custom atmospheric and spectroscopic inputs.

## Development

```bash
python -m pip install -e ".[dev,plot]"
python -m pytest -q
python -m ruff check src tests tools docs/figure_examples docs/troubleshooting paper_validation
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Report problems through the [issue tracker](https://github.com/nikolasvalsamidis97/PyMolFit/issues).

## Scientific basis

PyMolFit uses established telluric-modelling methods. Publications should acknowledge the relevant upstream resources:

- Molecfit: [Smette et al. 2015](https://doi.org/10.1051/0004-6361/201423932) and [Kausch et al. 2015](https://doi.org/10.1051/0004-6361/201423909).
- [AER LBLRTM](https://github.com/AER-RC/LBLRTM), [MT_CKD](https://github.com/AER-RC/MT_CKD) and [HITRAN](https://hitran.org/).
- [MIPAS](https://earth.esa.int/eogateway/instruments/mipas) and [NOAA GDAS](https://www.ncei.noaa.gov/products/weather-climate-models/global-data-assimilation).
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Astropy](https://www.astropy.org/).

## License

Original Python code uses the [BSD 3-Clause License](LICENSE). Scientific datasets and derived coefficient tables retain their upstream terms and attribution requirements; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the [example-data provenance](docs/user_guide/data/README.md).
