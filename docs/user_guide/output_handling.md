# Handling output files

`correct(..., output_path="outputs/beta_pic_fits.ecsv")` saves a complete correction product, not just the corrected flux. The examples below use the output from the [FITS notebook](01_fits_correction.ipynb); paths are relative to this guide folder.

## Reopen a saved correction

```python
from pymolfit import load_fit_product

result = load_fit_product("outputs/beta_pic_fits.ecsv")
```

This does not rerun the fit. If you already have the return value from `correct()`, use it directly instead.

## Get the spectra and wavelength

```python
observed = result.spectrum.to_unit("nm")
corrected = result.corrected.to_unit("nm")

wavelength = corrected.wavelength
uncorrected_flux = observed.flux
corrected_flux = corrected.flux
uncertainty = corrected.uncertainty
```

These arrays share the same sampling. `to_unit("nm")` converts wavelength units only; flux is not normalised. The correction result uses observatory-frame vacuum wavelengths, which may differ from the input file's wavelength convention. `uncertainty` contains corrected one-sigma flux errors, or `None` if unavailable.

## Use valid pixels

```python
valid = observed.valid & corrected.valid
wavelength_valid = wavelength[valid]
corrected_flux_valid = corrected_flux[valid]
```

`True` means usable. Invalid measurements and unreliable corrections should not be used in subsequent analysis. Apply the same mask to any other arrays you compare, including uncertainties when present.

## Plot the result

```python
from pymolfit import plot_fit

plot_fit(result)
```

This displays the uncorrected and corrected spectra plus the transmission, without saving a figure. In a notebook, use `%matplotlib widget` for interactive zooming when `ipympl` is installed.

## Other useful fields

| Field | Contents |
| --- | --- |
| `result.transmission` | Fitted atmospheric transmission on the observed grid. |
| `result.continuum` | Fitted smooth flux continuum. |
| `result.model_flux` | Model flux used in fitting, not the corrected spectrum. |
| `result.spectrum.uncertainty` | Observed flux uncertainties, when available. |
| `result.transmission_uncertainty` | Optional fitted-transmission uncertainties, otherwise `None`. |
| `result.fit_mask` | Pixels used in fitting, when recorded. This is not the corrected-spectrum validity mask. |
| `result.spectrum.group_id` | Physical order/detector labels, or `None`. |
| `result.species_scales` | Fitted molecular column-density scaling factors. |
| `result.success`, `result.message` | Optimiser status and explanation; success alone does not establish correction quality. |

## Check numerical correction status

These checks work for both new results and products saved by older versions:

```python
from pymolfit.diagnostics import correction_usability_diagnostics

quality = correction_usability_diagnostics(result)
print(quality["status"])
print(quality["retained_corrected_pixels"], quality["input_valid_pixels"])
print(quality["reasons"])
```

The status is `"usable"`, `"optimizer_not_converged"`, `"invalid_model"`,
`"empty_correction"` or `"no_valid_input"`. The checks inspect all valid observed
samples, including those outside the fitting mask. Nonpositive continua,
nonfinite model values and negative transmission are model problems; negative
observed flux alone is not. An empty output can occur even when the optimiser
converged, for example when every sample is below the transmission floor.

This diagnostic does not modify `result.success`, scientific arrays or masks.
`"usable"` means the numerical checks passed and at least one corrected sample
remains; inspect the retained fraction and plots before using the result. It is
not a guarantee of accurate atmospheric removal or preserved stellar features.

New corrections also store these checks under
`result.provenance["fit_quality"]["correction_usability"]`. Older saved products
may lack that provenance entry; recomputing it above does not refit the spectrum.

## Export just wavelength and corrected flux

```python
import numpy as np

np.savetxt(
    "corrected_spectrum.txt",
    np.column_stack((wavelength_valid, corrected_flux_valid)),
    header="wavelength_nm corrected_flux",
)
```

This optional export writes only valid samples and overwrites an existing file of that name. Keep the ECSV product: the two-column text file omits uncertainties, masks, transmission and fit information, so it cannot be reopened with `load_fit_product()`.
