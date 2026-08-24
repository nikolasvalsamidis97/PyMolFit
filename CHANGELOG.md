# Changelog

PyMolFit follows [Semantic Versioning](https://semver.org/). Changes that can
affect scientific results are identified explicitly.

## 0.7.4 - 2026-08-24

### Added

- Added explicit `native` and `observatory` coordinate metadata to ECSV region
  files. Older files without the new field remain valid and default to
  exposure-native coordinates.
- Added `wavelength_frame="observatory"` to the interactive selector. It moves
  barycentric or heliocentric wavelength coordinates back to the observer
  frame without resampling flux, uncertainty, masks, orders, or detector gaps.

### Changed

- Observer-frame region files can now be reused unchanged across time-series
  spectra. Correction applies each exposure's velocity-frame transform to the
  spectrum while leaving the shared terrestrial fit intervals fixed.
- AER markers and automatic telluric windows now follow the selector's declared
  wavelength frame. Correction provenance records the region-file medium and
  frame used by the fit.

Regression tests verify exact fitted-model, transmission, and fit-mask
equivalence between observer-frame intervals and their exposure-specific
native-frame representation. This release changes fitting pixels only when a
user explicitly creates or supplies an observer-frame region file.

## 0.7.3 - 2026-08-24

### Changed

- Batched fit and exclusion overlays by region type and limited rendering to
  intervals intersecting the current selector viewport. Region labels are
  hidden in wide views and restored when the view is sufficiently narrow.
- Cached the normalized in-memory region selection between edits so zooming
  and panning do not repeatedly merge unchanged intervals.
- Improved the measured ESPRESSO selector pan/redraw time from 1.64 s to
  0.32 s per view (5.1 times faster) for a 443,262-sample spectrum with 735
  theoretical stellar exclusions and 136,722 available AER transitions.

The changes in this release affect interactive rendering only. Full-resolution
spectra, calculations, region coordinates, and saved ECSV files are unchanged.

## 0.7.2 - 2026-08-24

### Added

- Added adaptive region-selector rendering. Full-spectrum views now use
  extrema-preserving display reduction, close views recover every original
  sample, and zooming or panning updates the spectrum automatically without
  changing the fitting data or saved region coordinates.
- Added zoom-dependent AER marker visibility. Markers are hidden in the
  full-spectrum overview, progressively revealed as the view narrows, and all
  locally visible catalogue transitions are shown at close zoom.

### Fixed

- Retained configured physical continuum components during basis assembly.
  The O2 continuum now shares the fitted O2 column scale, as in the LBLRTM
  molecular-column path. This reduces the fixed-parameter O2 A-band
  telluric-transmission RMS against Molecfit from 0.00591 to 0.000537.

## 0.7.1 - 2026-08-17

### Changed

- Reused prepared opacity bases across automatic pilot fits and accelerated
  wide-grid LBLRTM Voigt table lookup without changing model values. The Na D
  regression transmission and corrected flux remain bit-for-bit identical.
- Tightened runtime annotations, defensive validation, cache locking, and
  segmented-uncertainty callback handling across the fitting and
  radiative-transfer paths.
- Extended linting and CI quality checks to cover source, tests, tools, and
  examples consistently.

### Removed

- Removed retired local validation campaigns, generated plots and products,
  build artifacts, and stale science-readiness documentation. The maintained
  automated test suite remains part of the project.

## 0.7.0 - 2026-08-14

### Added

- Optional `correct(..., joint_stellar_model=True)` forward modelling of a
  theoretical stellar spectrum multiplied by atmospheric transmission before
  instrumental convolution. This mode is disabled by default and is exposed
  only by the unified `correct()` API.
- Integrated theoretical stellar masking into the interactive telluric-region
  selector. Stellar exclusions are plotted immediately, their physical mask
  parameters can be edited in the window, and fit plus exclusion regions are
  saved together for `correct(region_file=...)`.
- Optional `TheoreticalSpectrum` stellar-feature masking for two-column ASCII
  and SVO model spectra, including wavelength-coordinate handling,
  pseudo-continuum normalization, relativistic radial velocity, rotational and
  instrumental broadening, guarded residual alignment, ECSV mask export, and
  provenance diagnostics.
- Added opt-in confidence-weighted stellar masking for theoretical templates.
  Automatic stellar-feature detection uses binary fit exclusions by default.
- Made the selector's editable theoretical-template controls opt-in through
  `enable_theoretical_controls=True`; automatic stellar exclusions still
  appear whenever a theoretical template is supplied.
- Velocity-aware automatic padding for theoretical stellar masks, based on the
  rotational and instrumental broadening, so broad line wings do not constrain
  the telluric fit.

## 0.6.0 - 2026-08-10

### Added

- Unified file, array, and loaded-`Spectrum` routes through `correct()`.
- Versioned, reloadable ECSV fit products through `save_fit_product()` and
  `load_fit_product()`.
- Public exception hierarchy rooted at `PyMolFitError`.
- Automatic AER-based region selection with reusable region files.
- Resolved configuration reports, fit-quality diagnostics, and source
  provenance.
- CI, clean wheel installation checks, and trusted PyPI publishing workflow.

### Changed

- Automatic LSF and wavelength-alignment selection now use distributed pilot
  fits and physical spectrum groups.
- Automatic fit-region windows use 12 sampled detector pixels on each side of
  a selected line.
- GDAS downloads and interpolated products use validated, atomic cache writes.
- FITS wavelength-medium inference and multi-HDU handling report explicit
  errors for ambiguous or unsupported data.
- Development status advanced from Alpha to Beta.

### Compatibility

- `correct_file()` and `correct_arrays()` remain supported compatibility APIs.
- Existing unversioned ECSV products remain accepted by `plot_fit()` when they
  contain the required plotting columns.

## 0.5.1

- Previous public PyPI release.
