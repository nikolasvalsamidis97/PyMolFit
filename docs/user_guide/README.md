# User guide

Run these notebooks from a repository checkout with the development package installed:

```sh
python -m pip install -e ".[interactive]" notebook
python -m notebook docs/user_guide
```

Select the same Python environment as the notebook kernel. The first correction may download the AER catalogue and weather data; later runs reuse cached data. A full spectrum takes longer than the bundled Na D crop.

1. [FITS input](01_fits_correction.ipynb): load, correct and plot.
2. [FITS with a stellar template](02_fits_with_stellar_template.ipynb): protect stellar features during fitting.
3. [Wavelength-flux input](03_wavelength_flux_input.ipynb): attach `Observation` to `Spectrum` and correct it.
4. [Interactive selector](04_interactive_selector.ipynb): edit, save and apply fitting/exclusion regions.

[The selector guide](interactive_selector.md) adds optional manual region editing to any of these paths. [Parameter descriptions](../parameter_descriptions/correct.md) explain overrides without adding them to the basic examples.

The examples write complete products to `outputs/`. Reopen one without fitting again:

```python
from pymolfit import load_fit_product, plot_fit

result = load_fit_product("outputs/beta_pic_fits.ecsv")
plot_fit(result)
```

Paths in this snippet are relative to this guide folder. [Data provenance](data/README.md) describes the observation and template. These are usage examples, not a new validation of correction accuracy.

[Output handling](output_handling.md) shows how to access wavelengths, uncorrected/corrected flux, uncertainties and masks, and export the corrected spectrum.
