# Software Guide

## Stable Public Workflow

`pymolfit.correct()` is the canonical correction entry point. It accepts
exactly one of these input routes:

| Route | Required input | Typical use |
| --- | --- | --- |
| File | `input_path=...` | FITS, ECSV, CSV, or numeric text |
| Loaded spectrum | `spectrum=load_spectrum(...)` | Inspect or modify metadata/masks before fitting |
| Arrays | `wavelength=...`, `flux=...`, `observation=...` | Pipelines without a spectrum file |

The compatibility functions `correct_file()` and `correct_arrays()` remain
supported. New code should normally use `correct()`.

`load_spectrum()` returns a `Spectrum` containing wavelength, flux, optional
uncertainty, validity mask, physical order/detector labels, wavelength unit,
wavelength medium, and source metadata. `Observation` contains metadata that
cannot be derived from arrays.

## Automatic Decisions

The default workflow:

1. validates wavelength metadata and the input layout;
2. resolves the versioned AER molecular catalogue;
3. builds the selected MIPAS/GDAS atmosphere;
4. preserves physical detector/order groups and separates discontinuities;
5. subdivides wide groups only for bounded radiative-transfer memory;
6. estimates the instrumental LSF from resolving power or observed features;
7. selects supported wavelength-alignment and LSF models with pilot fits;
8. fits molecular columns, continuum, wavelength alignment, and selected LSF
   parameters;
9. evaluates transmission across every input pixel and returns the correction.

Set `report=False` to suppress the resolved configuration report. The same
information remains available in `result.provenance`.

## Input Compatibility

Supported file layouts include:

- FITS binary tables with recognized wavelength/flux column names;
- one-dimensional FITS image spectra with linear wavelength WCS;
- a selected row of a two-dimensional FITS image through `image_index`;
- named ECSV tables;
- numeric text and CSV with wavelength, flux, and optional uncertainty;
- echelle/order tables with common order, detector, quality, and mask columns.

Use `hdu` and explicit column names when automatic selection is ambiguous.
Air/vacuum wavelength metadata and the spectral velocity frame are separate.
PyMolFit stops instead of guessing when a FITS wavelength medium is ambiguous.
Arrays must declare their medium and provide an `Observation` with a wavelength
frame.

## Result And Output Contract

`correct()` returns `TelluricFitResult`. Its stable high-level fields are:

- `spectrum`: observed input in the modelled wavelength frame;
- `corrected`: corrected `Spectrum` with propagated uncertainty when available;
- `transmission`, `continuum`, and `model_flux` arrays;
- fitted molecular, wavelength, and LSF values;
- `fit_mask`, `metrics`, `success`, and optimizer termination information;
- `provenance`, including effective configuration and scientific input hashes.

Use `save_fit_product(result, "result.ecsv")` for the canonical complete
product and `load_fit_product("result.ecsv")` to reconstruct the result later.
The format is schema-versioned and contains observed/corrected flux,
uncertainties, masks, order labels, transmission, model, parameters,
diagnostics, and provenance. `plot_fit()` accepts either the live result or
this saved product.

`save_corrected_txt()` is intentionally compact. It is useful for software
that only needs wavelength and corrected flux, but it cannot reproduce the
fit result.

## Errors

All expected package failures inherit from `PyMolFitError`:

- `ConfigurationError`: incompatible parameters;
- `SpectrumFormatError`: unsupported file or table layout;
- `WavelengthMetadataError`: ambiguous air/vacuum or frame information;
- `ProductFormatError`: incompatible saved product;
- `ExternalDataError`: unavailable or corrupt line/atmosphere data;
- `FitError`: model construction or fitting failure.

Configuration exceptions retain `ValueError` compatibility, and external-data
exceptions retain `RuntimeError` compatibility.

## External Data And Offline Runs

AER data are checksum-verified and cached under `~/.cache/pymolfit/aer` unless
`PYMOLFIT_AER_CACHE` or `aer_cache_dir` selects another location. Set
`aer_offline=True` to prohibit network access. `pymolfit aer-status` checks the
catalogue and `pymolfit install-aer` prepares it before an offline run.

Time-local GDAS files are cached under `~/.cache/pymolfit/gdas`. Downloads and
interpolated profiles are written atomically; corrupt cache archives are
rejected. `gdas_mode="cache"` is strict and offline, while `"auto"` may fall
back to the packaged monthly-average profile when exact data are unavailable.
The selected source is recorded in provenance.

## Performance

Automatic segmentation limits high-resolution radiative-transfer grids while
retaining physical order identity. `segment_size` controls numerical memory
chunks, not independent atmospheric abundances. For large spectra, keep
`auto_segment=True`; use `group_id` for overlapping array-based orders and
retain the default grid limit unless memory measurements justify changing it.

Runtime depends primarily on wavelength coverage, line count, atmosphere
layers, oversampling, LSF model selection, and uncertainty estimation. Smaller
segments reduce peak memory but can increase setup overhead. They are not
expected to improve physical accuracy by themselves.

## Compatibility Policy

PyMolFit uses semantic versioning. Before version 1.0, a minor release may
refine automatic choices or add APIs; every result-affecting change is listed
in `CHANGELOG.md`. Public call signatures and product schemas are kept
backward-compatible when practical. A future removal will first be documented
and emit a deprecation warning for at least one minor release.

Private names beginning with `_`, internal provenance details, and pilot-model
implementation details are not stable APIs.
