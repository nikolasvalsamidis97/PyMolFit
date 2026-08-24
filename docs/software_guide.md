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

## Optional Theoretical Stellar Masking

Create a `TheoreticalSpectrum` when a stellar model is available. The preferred
interactive workflow builds stellar exclusions in the same selector used for
telluric fit intervals and saves both kinds of region in one ECSV file:

```python
from pymolfit import (
    TheoreticalSpectrum,
    correct,
    load_spectrum,
    select_telluric_regions,
)

template = TheoreticalSpectrum(
    "stellar_model.dat",
    radial_velocity_kms=21.3,
    vsini_kms=124.0,
    mask_padding_kms="auto",
)
observed = load_spectrum("spectrum.fits")
selector = select_telluric_regions(
    observed,
    theoretical_spectrum=template,
    output_path="regions.ecsv",
)
result = correct(
    input_path="spectrum.fits",
    region_file="regions.ecsv",
)
```

For a barycentric or heliocentric time series, save telluric fit windows in
observer coordinates so one file can be reused:

```python
observed = load_spectrum("first_exposure.fits")
selector = select_telluric_regions(
    observed,
    wavelength_frame="observatory",
    output_path="shared_telluric_regions.ecsv",
)

result = correct(
    input_path="another_exposure.fits",
    region_file="shared_telluric_regions.ecsv",
    theoretical_spectrum=template,
)
```

The selector transforms the wavelength coordinate only, without resampling
flux, uncertainty, masks, orders, or detector gaps. The ECSV metadata records
`wavelength_frame: observatory`, and correction therefore does not apply the
target exposure's BERV or heliocentric velocity to those intervals again.
Because stellar lines move in the observer frame, reusable files should hold
telluric fit intervals and only observer-fixed exclusions. Supplying
`theoretical_spectrum` to each correction generates stellar exclusions for
that exposure. Files without frame metadata use the legacy `native` behavior.

When a theoretical template is supplied directly to `correct()`, stellar
features are detected automatically and excluded from atmospheric parameter
estimation by default:

```python
template = TheoreticalSpectrum(
    "stellar_model.dat",
    radial_velocity_kms=21.3,
    vsini_kms=124.0,
)

result = correct(
    spectrum=spectrum,
    region_file="telluric_regions.ecsv",
    theoretical_spectrum=template,
    continuum_order=2,
)
```

Set `confidence_weighted_masking=True` on `TheoreticalSpectrum` to opt into
confidence weighting. It replaces binary stellar exclusions during parameter
estimation with residual weights between `confidence_weight_floor` and one.
Pixels dominated by predicted stellar structure contribute less to the
atmospheric fit, while continuum pixels retain full weight.

When confidence weighting is enabled, pass the template to `correct()` to
apply the continuous weights. A saved ECSV region file stores fit/exclusion
intervals, but it cannot store continuous weights; therefore the template must
also be supplied during correction when confidence weighting is required.

The template predicts pixels dominated by stellar features represented by that
model. During direct correction those pixels are excluded by default; the
interactive selector represents them as editable exclusion intervals in its
saved region file. In either mode, the final atmospheric model and corrected
spectrum still cover them. The implementation
supports SVO two-column ASCII spectra, pseudo-continuum normalization,
relativistic radial velocity, Gray rotational broadening, FITS-derived
resolving power, and a guarded residual velocity alignment. The selector's
parameter fields can update radial velocity, rotation, resolution, mask depth
and width, limb darkening, continuum estimation, and residual velocity
alignment before the combined region file is saved. These controls are hidden
by default; pass `enable_theoretical_controls=True` to
`select_telluric_regions()` to expose them.

For non-interactive work, `theoretical_spectrum=template` can still be passed
directly to `correct()`; `stellar_mask_path` optionally saves that generated
mask separately.

Set `joint_stellar_model=True` on `correct()` to include the normalized
theoretical spectrum directly in the forward model. PyMolFit then evaluates
`continuum * LSF(stellar * atmosphere)`. The returned `transmission` is the
effective atmospheric divisor `LSF(stellar * atmosphere) / LSF(stellar)`, so
correction removes the atmosphere without dividing out the stellar spectrum.
This mode requires a `TheoreticalSpectrum`, is disabled by default, and is
intentionally exposed only by the unified `correct()` API while it receives
broader scientific validation.

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
