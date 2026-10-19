# `correct()`

Fit atmospheric transmission and divide it out of an observed spectrum. Only the model is resampled during fitting.

## Basic use

```python
from pymolfit import correct

result = correct(
    input_path="spectrum.fits",
    output_path="corrected/spectrum.ecsv",
    show_plot=True,
)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `input_path` | `None` | Supply a spectrum file path or a prepared `Spectrum`. A real input is required. |
| `output_path` | `None` | Save the complete correction product, normally as ECSV. Existing output is overwritten. `None` keeps the result in memory only. |
| `theoretical_spectrum` | `None` | Optional `TheoreticalSpectrum` used to exclude stellar features from the atmospheric fit. |
| `region_file` | `None` | ECSV fitting/exclusion regions saved by the selector. Without it, fitting regions are normally generated automatically. |
| `show_plot` | `False` | Display observed/corrected flux and transmission. Requires Matplotlib. |

## Return value and saved output

`result` is a `TelluricFitResult`, not a filename. Its main attributes are:

| Attribute | Meaning |
| --- | --- |
| `spectrum`, `corrected` | Observed and corrected `Spectrum` objects. Use their wavelength units, medium and metadata when plotting. |
| `transmission` | Fitted transmission on the spectrum's pixel grid. |
| `continuum`, `model_flux` | Fitted smooth flux baseline and baseline times transmission. |
| `species_scales` | Molecular column multipliers relative to the reference atmosphere. |
| `fit_mask` | Pixels used to estimate parameters. This differs from `corrected.valid`. |
| `success`, `message` | Solver status. A successful stop does not prove the physical model is adequate. |
| `provenance` | Inputs, automatic choices and fit-quality information. |

The ECSV product retains these arrays, masks and available uncertainties. Wavelengths are expressed in the workflow's observatory-frame vacuum coordinates; the observed flux is not interpolated. Use the returned spectrum's coordinates rather than attaching the original file's wavelength array. Reopen it without repeating the correction:

```python
from pymolfit import load_fit_product, plot_fit, save_corrected_txt

result = load_fit_product("corrected/spectrum.ecsv")
plot_fit(result)
save_corrected_txt(result, "corrected/flux_only.txt")
```

The compact text export cannot reconstruct the full diagnostic plot.

## File-reading overrides

Usually omit these. They describe the input, not adjustments to improve a fit. For array data, put the corresponding information on `Spectrum` and `Observation` instead.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `wavelength_unit` | `"micron"` | Fallback input unit where file metadata do not supply one; e.g. `"nm"` or `"angstrom"`. |
| `wavelength_medium` | `None` | Declare `"air"` or `"vacuum"` if it cannot be inferred. This labels existing values; it does not convert them. Ambiguous FITS metadata require an explicit choice. |
| `input_format` | `None` | Infer from the extension; override with `"fits"`, `"fit"`, `"fz"`, `"txt"`, `"dat"`, `"csv"`, `"ascii"` or `"ecsv"` for a misleading filename. |
| `wavelength_col` | `None` | Wavelength column name or zero-based index; otherwise detect it. |
| `flux_col` | `None` | Flux column name or zero-based index; otherwise detect it. |
| `uncertainty_col` | `None` | Flux-uncertainty column name or zero-based index; otherwise detect it if present. Supply standard deviations, not variances. |
| `hdu` | `None` | Automatically select a unique spectral FITS extension. Specify its index if several are plausible. This is an extension index, not necessarily a spectral order. |
| `image_index` | `None` | Row to extract from a supported multidimensional FITS image. It does not combine orders. |

For a native CRIRES table containing several orders, select a matching pair such
as `wavelength_col="03_01_WL"` and `flux_col="03_01_SPEC"`, together with the
appropriate `hdu`. Omitting `uncertainty_col` then uses `03_01_ERR` if present;
an absent error column remains absent. A table with one unambiguous native pair
can be detected automatically. Mixed-order pairs are rejected, and selecting
only the detector HDU does not choose or combine its orders.

## Output quality and reporting

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_transmission` | `"auto"` | Mask unreliable divisions: an internal transmission floor of 0.01 plus an uncertainty-based quality mask. A number in `(0, 1)` sets an explicit transmission threshold instead. |
| `estimate_uncertainties` | `False` | Also propagate approximate fitted-parameter uncertainty into transmission and corrected flux. Measurement errors are propagated without this option. |
| `report` | `True` | Print the resolved configuration and fit summary. |
| `advanced` | `None` | Optional `AdvancedOptions`, described below. Omit to use normal defaults. |
| `cache` | `None` | Optional cache/download configuration. Omit for normal use. |

The report distinguishes optimiser `success` from numerical correction status.
Inspect `result.provenance["fit_quality"]["correction_usability"]` for the status,
retained-pixel counts and reasons. Neither convergence nor a `"usable"` status
certifies atmospheric accuracy; see [output handling](../user_guide/output_handling.md).

## Advanced options

Supply only what you want to override:

```python
from pymolfit import AdvancedOptions, correct

result = correct(
    "spectrum.fits",
    advanced=AdvancedOptions(continuum_order=2, wavelength_order=1),
)
```

### Continuum and fitting regions

| Parameter | Default | Meaning |
| --- | --- | --- |
| `continuum_order` | `"auto"` | Polynomial degree of the smooth flux baseline: 0 constant, 1 linear, 2 quadratic. Auto starts at 2 and validates degrees 1--4; promising rejected higher degrees receive guarded refit checks where supported. Coefficients are fitted separately per segment. |
| `segment_size` | `0.01` | Target processing width in microns (100 Angstrom). Physical order/detector boundaries are preserved; very large model grids are split further. |
| `solve_continuum_linear` | `"auto"` | Normally solve continuum coefficients within each nonlinear trial; robust fitting uses reweighting. `True` requests this; `False` includes coefficients in the nonlinear fit. |
| `automatic_fit_regions` | `True` | Select telluric windows when none are supplied; expand coverage within sparsely selected processing segments. If disabled without explicit windows, use all valid, non-excluded pixels. |
| `fit_ranges` | `None` | Tuple of `(lower, upper)` fitting intervals in microns, in the input medium/frame. |
| `exclude_ranges` | `None` | Same format; exclusions override fitting intervals. |

Do not combine `region_file` with `fit_ranges` or `exclude_ranges`. Exclusions affect fitting, not correction coverage. Automatic selection without usable windows warns and falls back to all valid, non-excluded pixels.

With `continuum_order="auto"`, a promising higher degree can trigger three paired fits with separate guarded wavelength blocks withheld from fitting. It must improve pooled prediction RMS by at least 5%, improve every fold, and retain or improve full-data RMS and median residuals. Original errors and score pixels are held fixed. Insufficient support, unsupported geometry or failed fits keep the previous solution. A triggered check adds up to six validation fits and one final refit; setting a numerical `continuum_order` bypasses it. Inspect `result.provenance["automatic_continuum"]["guarded_degree_validation"]` when present. See [guarded continuum selection](../technical_details/parameter_estimation.md#guarded-continuum-selection) for the scope and safeguards.

Automatic catalogue selection checks each processing segment before pilot fits. When selected pixels fall below 25% of available pixels or the minimum count of `max(32, 8 * (initial_continuum_order + 1))`, it includes all valid, non-excluded pixels in that segment; the required count is capped at the available count. Existing windows and masks are preserved. Explicit ranges/files are unchanged. Counts are recorded in `fit_region_selection.coverage`; sufficient coverage does not guarantee useful transmission or a well-constrained continuum. See [segmentation and regions](../technical_details/segmentation_and_regions.md#automatic-region-selection).

### Wavelength alignment

| Parameter | Default | Meaning |
| --- | --- | --- |
| `wavelength_order` | `"auto"` | Select whether/how to correct alignment. `None` disables fitting; 0 fits an offset; positive integers fit polynomial corrections of that degree. Auto compares supported candidate models. |
| `wavelength_scope` | `"group"` | Independent correction per physical order/detector group. `"global"` shares one correction function across the spectrum. |
| `initial_wavelength_shift` | `None` | Starting model offset in microns; `None` estimates it. Positive means redward. It is not a prescribed final offset. |
| `wavelength_shift_bounds` | `None` | Coefficient bounds: detector pixels when `wavelength_order="auto"`, microns for an explicit order. `None` starts automatic selection at ±3 pixels and permits supported expansion to ±12. Supplied bounds are never automatically widened. These bound each polynomial coefficient, not the total displacement. |

The polynomial degree is not a spectrograph order number. The observed spectrum is not resampled to achieve alignment.

With both continuum and wavelength settings left automatic, the workflow also
checks their choices together after the usual refinement stages. A simpler
continuum can rescue an invalid paired wavelength comparison; the supported pair
is deferred and refitted together, then a guarded joint comparison can test
alternatives. No additional parameters are needed. Explicit continuum or
wavelength choices are respected, and failed or unsupported comparisons retain
the preceding fit and its matching configuration.

The joint check applies to supported individual physical-order fits. A
simultaneous fit with independent wavelength models for several groups bypasses
it; extracting and correcting orders separately lets each eligible order use
the check. Unsupported kernels or wavelength geometry also keep the preceding
solution, and fixed numerical instrumental settings remain fixed.

Each replacement must pass validation and fixed original-data residual checks,
including no more than 2% RMS regression over the original fitting windows.
Correction-mask thresholds stay unchanged, although a different fitted
transmission can change retained coverage. These checks add fitting work and do
not guarantee a better correction for every order. Inspect
`result.provenance["wavelength_alignment"]["coordinated_joint_selection"]` for
stage decisions, scores and fallback reasons; see
[coordinated selection](../technical_details/parameter_estimation.md#coordinated-continuum-and-wavelength-selection)
for the criteria and scope.

### Molecular absorption depth

| Parameter | Default | Meaning |
| --- | --- | --- |
| `initial_species_scales` | `None` | Starting positive multipliers, e.g. `{"H2O": 1.5}`. Unspecified species start at unity when allowed by the bounds. Fitting can change them. |
| `fixed_species_scales` | `None` | Positive multipliers held fixed throughout fitting. They take precedence over starting values. |
| `scale_bounds` | `(0.001, 1000.0)` | Positive lower/upper limits for fitted molecular scales. |
| `minimum_species_peak_optical_depth` | `0.005` | Keep species below this expected peak optical depth in fitting pixels at their starting scale; insufficient signal should not produce an unconstrained abundance. |

A scale of 2 doubles that species' reference optical-depth contribution; it does not double the observed flux. Pressure, temperature and intrinsic line shapes are not refitted by these scales.

### Instrumental broadening

Widths below are in **detector pixels**, not km/s. Broadening applies to the model. A fixed width still broadens the model; it simply cannot be changed by the optimiser.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `fit_lsf_sigma` | `"auto"` | Fit the Gaussian width (`True`) or hold it fixed (`False`). Auto normally fits an automatically estimated width when telluric lines and an estimate are available. A small, locally unresponsive fitted width may trigger bounded retries; fixed widths are skipped. See [width recovery](../technical_details/parameter_estimation.md#recovery-of-an-unresponsive-gaussian-width). |
| `lsf_sigma_pixels` | `"auto"` | Gaussian standard deviation. Auto uses resolving power/sampling or observed narrow features. Gaussian FWHM is 2.355 times sigma. |
| `lsf_sigma_bounds` | `None` | Lower/upper fitted Gaussian sigma; `None` chooses internally. |
| `fit_lsf_lorentz_fwhm` | `"auto"` | Fit Lorentzian instrumental wings, hold their width fixed, or test whether adding them is justified. |
| `lsf_lorentz_fwhm_pixels` | `"auto"` | Starting/fixed Lorentzian full width at half maximum. Auto initialises the component. |
| `lsf_lorentz_fwhm_bounds` | `None` | Lower/upper fitted Lorentzian FWHM; `None` chooses internally. |
| `fit_lsf_box_width` | `False` | Fit an additional rectangular instrumental component. |
| `lsf_box_width_pixels` | `0.0` | Starting/fixed boxcar full width. Zero with fitting disabled removes this component. |
| `lsf_box_width_bounds` | `(0.0, 10.0)` | Allowed fitted boxcar full widths. |
| `lsf_variable_width` | `"auto"` | Compare constant widths with a fitted wavelength power law. `False` keeps widths constant in pixels; `True` uses widths proportional to wavelength. |

**A numerical Gaussian or Lorentzian width with its `fit_...="auto"` is held fixed.** To supply a starting guess that remains free, explicitly set the corresponding `fit_...=True`.

### Optimisation

| Parameter | Default | Meaning |
| --- | --- | --- |
| `loss` | `"soft_l1"` | Robust residual objective. `"linear"` is ordinary weighted least squares; `"huber"`, `"cauchy"` and `"arctan"` are alternatives. |
| `f_scale` | `"auto"` | Dimensionless robust-loss scale, called `a` in the paper. Auto starts at 1 and may adjust within 1--2. A positive number fixes it. No effect for linear loss. |
| `ftol` | `1e-10` | Relative objective-improvement stopping tolerance. |
| `xtol` | `1e-10` | Parameter-step stopping tolerance. |
| `gtol` | `1e-10` | Scaled-gradient stopping tolerance. |
| `max_nfev` | `None` | Evaluation limit per optimiser call; `None` uses internal budgets. Preliminary fits, refits, conditional width-recovery attempts and numerical derivatives add work. Reaching the limit is not convergence. |

Smaller `f_scale` reduces the influence of discrepant pixels more strongly; it does not rescale supplied uncertainties. Smaller convergence tolerances demand a stricter stopping condition, not necessarily a more physically accurate correction.

### Catalogue and extinction

| Parameter | Default | Meaning |
| --- | --- | --- |
| `aer_catalog` | `"auto"` | Obtain the managed AER line catalogue. A path or catalogue artifact selects another local copy. `None` disables automatic catalogue resolution, not individual species. |
| `rayleigh` | `False` | Include smooth Rayleigh extinction. Enable only if it has not already been removed upstream. |

Custom atmospheric profiles and alternative opacity engines use the lower-level modelling API, not this fit-tuning object. See [technical details](../technical_details/README.md).
