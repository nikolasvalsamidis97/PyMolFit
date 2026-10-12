# Changelog

PyMolFit follows [Semantic Versioning](https://semver.org/). Changes that can
affect scientific results are identified explicitly.

## 0.8.0 - 2026-10-12

### Breaking API Changes

- `correct(input_path=...)` accepts a file path or a prepared `Spectrum`.
  Array data and observing metadata belong in
  `Spectrum(..., observation=Observation(...))`. Removed `spectrum=`,
  direct array/observation arguments, and deprecated flat expert keywords.
- Fit controls belong in `AdvancedOptions`, passed as `advanced=`.
  Cache/download settings belong in `CacheConfig`, passed as `cache=`.
  The expanded `correct_file()` and `correct_arrays()` APIs remain available
  for custom atmospheric, spectroscopic, and radiative-transfer inputs.
- `output_path` now saves the full correction product, with ECSV as the
  default. Removed `product_path` and `product_format`; use `output_path`
  and lower-level `output_format` instead. Compact spectrum export remains
  available through `save_corrected_txt()`.
- Consolidated wavelength controls into `wavelength_order` and
  `wavelength_scope` in `AdvancedOptions`. Exposed initial molecular scales,
  scale bounds, and solver evaluation limits.
- Renamed lower-level catalogue filters to `min_line_strength` and
  `max_lines`; consolidated external O2/N2 CIA inputs into `cia_tables`.
  Removed the old CLI flag aliases.
- Removed joint stellar-flux fitting. Theoretical-template exclusions and
  confidence weighting remain supported.
- Removed synthetic demo-line entry points from correction APIs, `LineList`,
  and the CLI. Synthetic lines now live only in the test suite.
- Removed the selector's `show=` alias. Use
  `show_interactive_window=True` to open its editor.

### Added

- Source-backed ozone Chappuis/Wulf and Hartley-Huggins absorption through
  lower-level `o3_continuum=True` / CLI `--o3-continuum`. Includes
  layer-temperature dependence, shared O3 line/continuum column scaling,
  packaged coefficients, and regression values from independently compiled
  LBLRTM 12.11 routines.
- Herzberg and far-UV oxygen absorption in the lower-level
  `o2_continuum=True` option. Enabled O2-continuum calculations above
  36000 cm-1 change; existing visible and infrared calculations are unchanged.
  New continua are bounded at zero where interpolation gives negative opacity.
- Reorganized documentation into parameter descriptions, short notebook
  guides, troubleshooting, figure examples, and detailed numerical methods.

### Changed

- High-level correction and CLI default to robust fitting and automatic
  continuum selection starting with a quadratic. These defaults can change
  scientific results relative to 0.7.5; explicit values override selection.
- Without supplied regions, correction selects distributed telluric windows;
  stellar exclusions remain opt-in. Wavelength-model selection considers
  supported quadratic corrections and different degrees per physical group.
- Missing flux errors receive guarded local estimates. The robust threshold
  and corrected-pixel uncertainty mask can be selected automatically.
  Supplied errors and numerical overrides are retained; policies record provenance.
- Pilot comparisons account for supplied noise rather than interpreting tiny
  numerical residual improvements as strong model evidence.
- FITS input defaults to `hdu=None`, selecting a unique supported spectrum.
  Ambiguous files report candidate HDUs; explicit HDU/image-row selection is
  respected. Correction, region selection, and metadata inference agree.
- `select_telluric_regions()` accepts file paths, generates automatic regions
  immediately, and opens no figure by default. New and reused selections
  support `save(path)`.
- PyPI publication now requires the complete CI matrix, package checks, and
  matching release-tag, citation, and changelog versions. It publishes the
  same distributions built by those checks.

### Fixed

- Preserve the full-segment continuum coordinate when fitting selected pixels.
- Keep overlapping physical orders separate when extracting instrumental pilots.
- Do not interpret known FITS quality-mask or order/detector columns as flux
  uncertainties when an uncertainty column is absent.
- Preserve the original exception if AER cache-lock cleanup loses filesystem
  access. Acquisition errors explain permissions, network credentials, and
  local-cache alternatives; cleanup failures remain visible as warnings.
- Report binary stellar exclusions separately from confidence-weighted masks.

## 0.7.5 - 2026-08-26

### Fixed

- Recognize the ESPRESSO `ESO QC BERV` header card when converting
  barycentric spectra to the observatory frame.
- Support VLT UT1 through UT4 geodetic header cards when a missing barycentric
  correction must be reconstructed from observation metadata.
- Apply the same corrected frame handling to theoretical stellar masks.

This release corrects a scientific-result defect affecting barycentric
ESPRESSO products whose headers provide `ESO QC BERV` and numbered telescope
coordinates. Previously, those spectra could remain on the barycentric grid
while observatory-frame telluric regions were fitted.

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
