# The correction pipeline

PyMolFit fits an atmospheric transmission model to selected parts of an observed spectrum, then divides that transmission out over the full wavelength coverage.

**Input -> coordinate preparation -> regions and segments -> atmospheric model -> preliminary fits -> iterative main fit -> full-spectrum correction -> result**

## 1. Read the spectrum

`correct()` accepts a file path or a prepared `Spectrum`. It reads wavelengths, flux, available uncertainties, validity masks and observing metadata. Array users supply the observing conditions through an attached `Observation`.

Wavelengths are converted to observatory-frame vacuum coordinates. The flux samples are not interpolated. Missing or ambiguous metadata must be resolved before fitting.

## 2. Choose fitting pixels and processing segments

Supplied fitting regions select where the model is compared with the observation. Otherwise, PyMolFit normally proposes windows around telluric catalogue lines. Invalid pixels and exclusion regions do not constrain the fit. An optional theoretical stellar spectrum identifies stellar exclusions; it is not subtracted from the observation.

The spectrum is divided into manageable processing segments while preserving physical order/detector boundaries. Each segment has its own smooth flux continuum; segments can share molecular scales and wavelength corrections.

Details: [segmentation and regions](segmentation_and_regions.md).

## 3. Construct the reference atmosphere and absorption

MIPAS reference profiles are combined with available GDAS weather profiles and local measurements. The observation time and geometry determine the appropriate atmospheric profile and line-of-sight molecular columns.

Catalogue line strengths and profiles give each species' absorption through the atmospheric layers. Discrete lines and enabled continuum contributions produce optical depth and intrinsic transmission:

$$T_{\rm atm}(\lambda)=\exp[-\tau_{\rm atm}(\lambda)].$$

The calculation prepares reference optical-depth arrays on a fine spectral grid. These are reused during fitting; pressure and temperature profiles remain fixed.

Details: [atmosphere and absorption](atmosphere_and_absorption.md).

## 4. Initialise parameters and choose model forms

Molecular column scaling factors normally start at one. Initial alignment comes from the input calibration and frame metadata; instrumental width is estimated from resolving power and detector sampling, or narrow observed features. Explicit starting values can override these estimates.

When automatic choices are enabled, preliminary fits test wavelength-correction and instrumental-broadening model forms on selected telluric-rich regions. These fits also solve the flux continuum. They choose suitable model forms and starting values, not a separate corrected spectrum.

## 5. Construct, compare and update the model

Both preliminary and main fits use the same model-building operations. In the main fit, each trial:

1. Multiplies the prepared species optical depths by the trial molecular scales and calculates `T_atm`.
2. Applies the trial wavelength correction.
3. Averages transmission over each observed pixel's wavelength interval, then applies instrumental broadening in the default implementation.
4. Solves a local polynomial flux continuum and forms the model flux.
5. Compares model and observed flux **inside fitting regions**, after validity checks and exclusions.
6. Updates the enabled parameters and repeats.

The detector-grid transmission is denoted by $\widehat T_{kp}$. At pixel `p` in segment `k`,

$$F^{\rm mod}_{kp}=C_{kp}\widehat T_{kp},$$

where $C_{kp}$ is the fitted flux continuum. Molecular scales change atmospheric absorption depth; continuum coefficients set the smooth flux level. Wavelength and broadening parameters change line positions and widths.

Residuals are weighted using supplied or estimated flux uncertainties. The default soft-$L_1$ objective reduces the influence of large residuals. Iteration stops at a convergence criterion or evaluation limit; hitting the limit is not convergence.

Details: [spectral model](spectral_model.md) and [parameter estimation](parameter_estimation.md).

## 6. Check automatic continuum and robust-loss choices

Automatic continuum selection tests polynomial degrees on temporarily withheld wavelength blocks. The robust-loss scale can also be adjusted from the residual scatter. Accepted changes can trigger another joint fit. Explicit settings bypass the corresponding automatic choices.

## 7. Correct the full spectrum

The final fitted parameters generate transmission over all input wavelengths, including outside fitting regions and inside exclusions. The catalogue already contains lines there: fitting regions determine shared parameters, not which lines can be corrected. This is model evaluation, not another fit to the whole spectrum.

$$F^{\rm corr}_{kp}=\frac{F^{\rm obs}_{kp}}{\widehat T_{kp}}.$$

The flux continuum is **not** divided out. Stellar features and noise remain; additive sky emission is not removed.

## 8. Propagate errors and return the result

Measurement errors are propagated through the division. Optional fitted-transmission errors add an approximate parameter-uncertainty contribution where supported. By default, invalid inputs and transmission below 0.01 are masked; large propagated errors are retained alongside the corrected flux. Uncertainty-based masking is optional.

The returned `TelluricFitResult` contains observed and corrected spectra, transmission, continuum, model flux, masks, parameters and diagnostics. `output_path` saves the complete product; `show_plot=True` displays it. Inspect residuals and masks as well as solver success.

Details: [correction and uncertainties](correction_and_uncertainties.md).
