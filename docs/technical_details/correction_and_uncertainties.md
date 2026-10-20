# Correction and uncertainties

Implementation: [`fit.py`](../../src/pymolfit/fit.py), [`spectrum.py`](../../src/pymolfit/spectrum.py), [`_automatic.py`](../../src/pymolfit/_automatic.py), [`io.py`](../../src/pymolfit/io.py).

## Full-coverage evaluation

After fitting, the same atmospheric parameters generate all catalogue lines over the input coverage, including outside fitting windows and inside stellar exclusions. No third fit against the entire observed spectrum is required. Physical-group wavelength corrections and the selected instrumental model are applied consistently in these regions.

$$F^{\rm corr}_{kp}=\frac{F^{\rm obs}_{kp}}{\widehat T_{kp}}.$$

The fitted flux continuum $C_{kp}$ is not divided out. The corrected observation is not the model flux $F^{\rm mod}_{kp}=C_{kp}\widehat T_{kp}$. It still contains stellar structure and noise. Model evaluation may use new numerical segments, but it does not interpolate observations across wavelength gaps.

## Measurement errors

For fixed fitted transmission,

$$\sigma^2_{F^{\rm corr},kp}=\frac{\sigma^2_{F,kp}}{\widehat T_{kp}^2}.$$

This is the default propagation. Both signal and noise are amplified in deep absorption; dividing cannot recreate photons not detected.

## Optional fitted-transmission uncertainty

With `estimate_uncertainties=True`, the implementation estimates a local parameter covariance. Let $\boldsymbol\theta$ contain the fitted parameters, `J` be the residual Jacobian and `K` its estimated covariance. In the identifiable, locally linear case,

$$K\simeq s^2(J^\mathsf TJ)^{-1},\qquad s^2=\frac{\sum r_{kp}^2}{n_{\rm fit}-n_{\rm effective}}.$$

The implementation uses column scaling and singular-value decomposition rather than explicitly forming that inverse; it accounts for profiled continuum degrees of freedom in the residual variance estimate. Rank-deficient covariance is marked unavailable (non-finite), not interpreted as zero uncertainty. For robust fits the local Jacobian/scaling provide an approximation, not an exact Gaussian likelihood covariance.

Numerical derivatives of detector-grid transmission give

$$\sigma^2_{T,kp}\simeq\sum_{a,b}\frac{\partial\widehat T_{kp}}{\partial\theta_a}K_{ab}\frac{\partial\widehat T_{kp}}{\partial\theta_b}.$$

The indices `a,b` here label parameters and are unrelated to the scalar robust-loss scale `a`. The corrected-flux variance, neglecting covariance between observed flux and fitted transmission, is

$$\sigma^2_{F^{\rm corr},kp}\simeq\frac{\sigma^2_{F,kp}}{\widehat T_{kp}^2}+\frac{(F^{\rm obs}_{kp})^2}{\widehat T_{kp}^4}\sigma^2_{T,kp}.$$

That neglected covariance can matter because the transmission was fitted from the same observation. The current high-level propagation is limited to fitted segments; it does not supply equivalent transmission-error estimates for every extrapolated region. These statistical errors omit uncertain atmospheric structure, spectroscopy, template mismatch and instrumental-model inadequacy. Correlated detector errors are not a full covariance input to the default fit.

## Correction masks

Invalid input pixels and transmission below the configured floor are masked in the corrected output. The default `min_transmission=0.01` retains corrected values and their propagated uncertainties even when those uncertainties are large. Observed flux near zero or below zero is not itself a rejection criterion. Retaining a pixel does not establish that its correction is scientifically useful; inspect its uncertainty.

Optional `min_transmission="auto"` adds uncertainty-based rejection below transmission 0.98 when propagated uncertainty exceeds the greater of 10 percent of the fitted continuum and twice the input uncertainty. This stricter mode must be requested explicitly. Changing this output-mask policy does not change the atmospheric fit.

Input validity, fitting selection and corrected-output validity are separate masks. Quality masking does not improve a fit; it marks correction values that should not be trusted.

## Products

`correct()` returns a `TelluricFitResult` and optionally writes the full ECSV product. It stores observed and corrected arrays, fitted transmission, continuum, model, masks, parameters, available uncertainties, diagnostics and provenance. `load_fit_product()` reconstructs the result without refitting. `save_corrected_txt()` exports only the corrected spectrum for downstream tools, not a reproducible fit product.
