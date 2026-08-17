# Changelog

PyMolFit follows [Semantic Versioning](https://semver.org/). Changes that can
affect scientific results are identified explicitly.

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
