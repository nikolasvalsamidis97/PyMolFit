# Changelog

PyMolFit follows [Semantic Versioning](https://semver.org/). Changes that can
affect scientific results are identified explicitly.

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
