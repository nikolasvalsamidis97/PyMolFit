# Troubleshooting

## Wavelength Medium Cannot Be Determined

Air/vacuum wavelength convention is not the same as barycentric or
heliocentric frame. Inspect the FITS wavelength column and pass
`wavelength_medium="air"` or `"vacuum"` only when the file documentation makes
the convention clear. PyMolFit deliberately does not infer this from
`SPECSYS` alone.

## FITS Columns Or HDU Are Not Recognized

Pass `hdu`, `wavelength_col`, `flux_col`, and optionally `uncertainty_col`.
For a two-dimensional image spectrum, also pass `image_index`. Use
`save_fits_header_txt()` to create a readable header report.

## No Molecular Data Are Available

Run:

```bash
pymolfit aer-status
pymolfit install-aer
```

For an offline machine, populate the cache first and use `aer_offline=True`.
Delete only the reported corrupt cache artifact; verified catalogue and line
window files do not need to be downloaded again.

## Exact GDAS Is Unavailable

`gdas_mode="auto"` records the failure and uses the average fallback.
`gdas_mode="online"` requires an exact download, and `"cache"` requires exact
cached data. Exact GDAS also requires observation time, latitude, and
longitude.

## Fit Does Not Converge

Check `result.success`, `result.message`, `result.parameter_bound_status`, and
`result.provenance["fit_quality"]`. Common causes are an incorrect wavelength
medium/frame, insufficient telluric structure, stellar lines dominating the
fit pixels, underestimated uncertainties, or bounds that exclude the true
instrumental shift/width.

Use the region selector to provide telluric-rich fit windows and exclude
astrophysical or defective features. Robust losses help isolated outliers but
do not repair wavelength calibration or atmospheric-model errors.

## Corrected Values Are Missing

Pixels with fitted transmission below `min_transmission` are intentionally
masked because division cannot recover reliable flux from nearly opaque
absorption. Lowering the threshold displays more values but also amplifies
noise and model error.

## Full Spectrum Is Slow Or Exceeds The Grid Limit

Keep `auto_segment=True`, preserve detector/order labels, and avoid increasing
`radiative_transfer_max_points` without measuring memory. Numerical chunks
are stitched back into the original sampling and do not imply separate
molecular abundances.

## Plotting Is Unavailable

Install `pymolfit[plot]` for Matplotlib or `pymolfit[interactive]` for Jupyter
widgets. A complete saved product can be plotted with
`plot_fit("result.ecsv")`; a compact corrected text file cannot.
