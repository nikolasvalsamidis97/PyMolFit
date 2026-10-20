# Parameter estimation

Implementation: [`workflow.py`](../../src/pymolfit/workflow.py), [`fit.py`](../../src/pymolfit/fit.py), [`_automatic.py`](../../src/pymolfit/_automatic.py). The model and notation are defined in [spectral model](spectral_model.md).

## Initialisation

1. Build fitting masks and the fixed reference atmosphere/opacity basis.
2. Start molecular scales at unity, except explicit initial/fixed values. Weak species with expected peak optical depth below 0.005 in fitting pixels remain fixed at their starting scales.
3. Start alignment from input calibration and frame metadata, with coarse offset searches where configured.
4. Estimate Gaussian instrumental sigma from resolving power and detector dispersion, then narrow observed features if needed. Without usable information, automatic mode does not invent a freely fitted width. Numerical widths with `fit_...="auto"` are fixed; set `True` to fit them.

For locally uniform dispersion $\Delta\lambda$, the resolving-power estimate is $\sigma_{\rm pix}\simeq\lambda/(2.355R\Delta\lambda)$, where `R` is instrumental resolving power. The workflow aggregates usable estimates rather than assuming one arbitrary sampling scale.

With physical order labels, resolving-power dispersion estimates and
narrow-feature measurements are made within each order. Overlapping orders
therefore do not create artificial subpixel spacings in the width estimate.
The resulting shared starting width still aggregates these estimates; it does
not introduce a separate instrumental width for every detector.

## Preliminary model comparisons

Distributed telluric-rich pilot windows test whether added model flexibility is supported. Wavelength candidates are unchanged calibration, constant offset, linear polynomial, and quadratic where enough separated information exists. Group-scoped selection can choose different supported degrees for physical groups. Weak groups may keep the original calibration.

LSF comparisons test wavelength dependence and Lorentzian wings. Automatic acceptance requires improvement across pilot regions as well as a complexity penalty, not merely a lower training residual in one window. A representative score is ${\rm BIC}=D+n_{\rm par}\ln n_{\rm pix}$, where `D` is the implemented residual deviance, `n_par` parameter count and `n_pix` fitted sample count. This is a model-selection heuristic, especially with robust losses and correlated pixels, not a posterior probability.

Preliminary fits solve continuum coefficients and adjust their enabled physical parameters too. Their purpose is selecting model forms and useful starts. The main fit then refines the selected forms over the complete set of fitting regions. Neither stage is the final full-spectrum division.

## Residuals and robust loss

$$r_{kp}=\frac{F^{\rm obs}_{kp}-F^{\rm mod}_{kp}}{\sigma_{F,kp}},\qquad \mathcal J=a^2\sum_k\sum_{p\in\mathcal F_k}\left[\sqrt{1+(r_{kp}/a)^2}-1\right].$$

This is SciPy's `soft_l1` objective including its one-half cost convention. `f_scale` is the dimensionless `a`. It changes residual influence gradually; it is neither a clipping threshold nor a flux multiplier. Ordinary weighted least squares (`loss="linear"`) gives $\mathcal J=\tfrac12\sum r_{kp}^2$. Optional stellar confidence weights multiply residuals by their square roots before objective evaluation.

Supplied standard deviations remain unchanged. Missing errors are estimated locally without crossing groups/gaps. For second difference $D_p=F_{p+1}-2F_p+F_{p-1}$ and independent equal-variance noise, ${\rm Var}(D)=6\sigma_F^2$, motivating $\sigma_F\simeq1.4826\,\operatorname{MAD}(D)/\sqrt6$. The implementation prefers pixels outside telluric windows, rejects insufficient estimates, applies a $10^{-4}$ local-flux floor and falls back to 1 percent of local flux level. Correlated noise and unresolved spectral structure violate this simple noise model.

## Continuum inside each trial

For fixed transmission, the continuum problem is linear in `c_km`. Define $A_{pm}=\widehat T_{kp}u_{kp}^m/\sigma_{F,kp}$ and $y_p=F^{\rm obs}_{kp}/\sigma_{F,kp}$ on fitting pixels. Weighted linear least squares solves $\min_{\boldsymbol c}\|A\boldsymbol c-\boldsymbol y\|^2$. The implementation uses a least-squares solver, not explicit inversion of normal equations.

Robust fitting uses iteratively reweighted linear least squares. For soft-L1, squared-residual influence is $w_p=[1+(r_p/a)^2]^{-1/2}$, so the rows of the linear system receive $\sqrt{w_p}$. A small inner iteration loop updates these weights; it is an approximate profiled robust solve, not a separately unconstrained nonlinear continuum at every pixel. With `solve_continuum_linear=False`, continuum coefficients instead join the nonlinear parameter vector.

## Main fit and automatic refit

Free molecular scales are parameterised as $q_s=e^{z_s}$ and bounded in log space. The bounded trust-region reflective solver updates these, wavelength coefficients and enabled instrumental parameters. Each trial rescales prepared optical depths, exponentiates, aligns, resamples, broadens, solves local continua and evaluates residuals. Pressure and temperature profiles remain fixed.

For micron-valued wavelength coefficients, the grouped numerical Jacobian uses
the relevant detector dispersion as the finite-difference scale, avoiding an
implicit one-micron parameter scale. The same
scaling is used for micron-valued wavelength derivatives in optional
transmission-uncertainty propagation. Bounds still limit every perturbation.
Pixel-valued derivative steps and SciPy's default finite-difference route are
unchanged.

After an initial joint solution, automatic continuum selection compares degrees 1--4 on temporarily withheld wavelength blocks at fixed transmission. Block selection reduces leakage from adjacent correlated samples. The quadratic is retained unless a simpler or more predictive alternative passes the score/improvement checks; insufficient data leave it unchanged. Automatic robust scale uses residual MAD, clipped to 1--2. Accepted changes can trigger an additional joint fit. Promising rejected higher degrees can then receive the guarded checks below. These checks select numerical model forms; they do not independently validate atmospheric accuracy.

## Guarded continuum selection

Optimisation uses fitting pixels. With automatic regions, continuum model
selection also checks predictions on non-excluded wavelengths outside the
initial fitting windows. User-supplied fitting regions instead bound both
optimisation and validation. Original
exclusions are removed from both training and residual-scoring domains.
Held-out validation blocks are excluded only from training, so they can still
be scored. Physical checks on finite predictions and positive continua retain
their full-coverage scope.

For `continuum_order="auto"`, the fixed-transmission screen can nominate one higher degree whose validation cost is at least 5% below the selected degree's cost. The lowest degree within 1% of the best eligible cost is tested. This nomination alone cannot change the result.

Three paired validation folds refit all enabled parameters with identical training and scoring pixels for both degrees. Each processing segment contributes a 16-pixel core near 20%, 50% and 80% of its length, adjusted deterministically to the nearest separated positions with adequate training support. A core must contain at least eight valid, originally non-excluded pixels. The surrounding excluded guard is at least 16 pixels per side and covers the maximum configured Gaussian kernel support plus twice the maximum allowed wavelength displacement. The complete guard is excluded from training; only valid core pixels outside original exclusions are scored. Segments must retain enough fitting pixels and moderately transmitting support on both sides. Unsupported geometry or insufficient support keeps the previous solution without launching these refits.

Fold starts use the caller's molecular scales, the pre-pilot wavelength start and, for an automatically estimated freely fitted Gaussian width, a bounded 1-pixel starting sigma. Full-data fitted scales, widths and wavelength coefficients are not reused. Original flux, errors, validity and grids must remain unchanged, and every refitted training mask must exactly equal the reference fitting mask minus its guards. Each fold must converge and give finite models and transmission and positive finite continua on all original-valid samples, including withheld pixels.

A candidate needs at least 5% improvement in pooled original-error-standardized RMS, lower RMS in every fold, and no increase in pooled median absolute standardized residual. It is then refitted on the original fitting pixels and must not worsen either RMS or median absolute residual on valid, originally non-excluded samples, both before and after trimming 40 samples from each outer edge. Rejected or failed comparisons retain the previous result. These residual domains do not depend on the corrected-output mask. The final transmission can nevertheless change which pixels pass the existing correction-quality thresholds.

The check applies to unweighted single-physical-group fits with supported constant-width Gaussian and global wavelength geometry. It skips explicit continuum degrees, automatic Gaussian-width estimates held fixed, stellar confidence weights, variable/non-Gaussian kernels, and unsupported group-specific wavelength geometry. It only reconsiders higher degrees rejected by the initial screen. The usual continuum solver and parameter bounds remain unchanged.

These are conditional internal checks: model families, fitting windows, support estimates and the robust-loss scale have already used the full spectrum. They are not blind validation of the complete automatic pipeline, and lower observed-model residuals do not by themselves prove a more accurate telluric transmission. Correction still divides by transmission, not by the fitted continuum. Decisions, guards, scores, starts and rejection reasons are recorded under `result.provenance["automatic_continuum"]["guarded_degree_validation"]` when the check is reached. A trigger adds up to six paired validation fits and one full refit; each fit can include ordinary preliminary stages or Gaussian-width recovery calls.

## Coordinated continuum and wavelength selection

With automatic continuum and wavelength settings, a final coordinated comparison
runs after the existing continuum and robust-scale refinements, before correction
quality masking. It requires no additional `correct()` parameters. Explicit
continuum or wavelength choices are not replaced by this comparison. Unsupported
geometry, inadequate validation support or failed comparisons retain the previous
result.

The final joint check supports individual physical-order fits with compatible
pixel-based wavelength geometry and Gaussian kernels. A simultaneous input with
independent wavelength models for several physical groups bypasses this joint
check; fitting each extracted order separately allows eligible orders to use it.
Unsupported variable-width or non-Gaussian kernels keep the preceding solution.
Fixed numerical instrumental settings remain fixed.

If a paired wavelength validation produces nonpositive continua or nonfinite
predictions, automatic mode can retry the entire comparison at a common lower
continuum degree. Degrees are tried in descending order; the first physically
valid comparison is used, rather than choosing the degree with the best withheld
residual. All wavelength families use the same training and scoring masks,
original errors, and fixed degree. A supported continuum/wavelength pair is
deferred until the normal refinement stages finish, then refitted together.
Later continuum-only selection therefore cannot undo that pair.

The joint stage compares the current continuum degree and degrees 1 and 2 with
unchanged, constant-offset, linear and quadratic wavelength models. All original
bounds remain in force. Full-data screening shortlists at most two alternatives;
three paired guarded refits then compare each with the current accepted model.
Acceptance requires at least 5% pooled prediction-RMS improvement, improvements
in at least two folds, no fold worse by more than 5%, and the median-residual and
supported transmission-band checks. Fits start from configured values and fixed
initialisation rules, not the preceding full-data fitted coefficients.

Every replacement must converge, have a positive finite continuum, preserve the
original data and fitting masks, and avoid wavelength bounds. Full-data screening
requires at least 2% RMS improvement after trimming 40 pixels from each outer
edge, at most 1% whole-spectrum RMS regression, and at most 2% regression in each
supported transmission band. Bands with fewer than 32 scoring pixels remain
unevaluated. A further guard permits at most **2% RMS regression across all
original fitting-window pixels**, without trimming. These scores use original
errors and valid, originally non-excluded pixels, never the candidate's
corrected-output mask. Non-excluded wavelengths outside automatic fitting
windows remain available for these full-coverage checks; explicit fitting
regions constrain the score domain. The original
baseline defines the final guard domains for both stages, preventing permitted
regressions from accumulating across replacements.

The accepted result and its matching configuration replace the preceding pair
together. Rejection restores that pair exactly, so a failed joint comparison
cannot discard an accepted rescue. Existing correction-quality thresholds are
applied after selection; different transmission can still change pixel retention.
Per-stage candidates, scores, guards and rejection reasons are recorded under
`result.provenance["wavelength_alignment"]["coordinated_joint_selection"]`, with
rescue and joint details also exposed as `coordinated_continuum_rescue` and
`joint_model_reconsideration` in `wavelength_alignment`. These are conditional
model-selection checks using structures already informed by the full spectrum;
they do not guarantee improvement for every order or establish atmospheric truth.

## Recovery of an unresponsive Gaussian width

After a solver call, a freely fitted Gaussian sigma at or below 0.25 detector pixels enters a numerical diagnostic. This threshold only selects widths to inspect; it is not a bound on the fitted answer. The diagnostic compares the change in the actual residual vector per unit width for a small forward perturbation (up to `1e-4` pixel) with a wider allowed probe. Recovery requires local sensitivity below `1e-3` of the wider sensitivity, and wider sensitivity above `1e-6`. Fixed widths are skipped, and an unresponsive wider probe does not trigger retries.

When triggered, the fitter tries nominal starting sigmas of 0.5, 1 and 2 detector pixels, clipped inside the existing bounds and deduplicated. Only starts wider than the stalled solution are considered. Each attempt refits all free parameters from the same original solver start with its Gaussian width changed. For segmented fits this start already includes any accepted preliminary optimisation stages. Profiled continua are solved again within every trial. Fitting pixels, supplied errors, weights, objective, model family and parameter bounds are unchanged; the starting widths are not fixed final widths.

The original result remains the reference. A retry can replace it only when it converges, has finite parameters and residuals, gives finite transmission and positive finite continua on original-valid samples in the fitted segments, and reduces the same objective beyond a roundoff threshold. `result.provenance["lsf_sigma_recovery"]` records the trigger or reason for skipping, sensitivities, starts, final widths, costs, validity, selected attempt and optimiser evaluations. The recovery can also occur in preliminary fits that fit the Gaussian width. Other local minima may remain.

`max_nfev` applies separately to each optimiser call, including each recovery attempt. Up to three additional calls can therefore exceed that value in total. Reported `nfev` includes completed recovery calls and, for segmented fits, preliminary optimisation stages. If an attempt raises before reporting its evaluation count, provenance marks `optimizer_nfev_complete=False`; numerical derivative and diagnostic evaluations are additional work.

## Stopping and diagnostics

The default tolerances are `ftol=xtol=gtol=1e-10`. SciPy's trust-region stopping tests use relative objective change (with adequate model agreement), scaled parameter step, or the bound-aware gradient norm. Satisfying an enabled convergence test stops that solver call. `max_nfev` limits evaluations per call; numerical derivative work and preliminary/refit calls add to total cost.

A budget stop is not convergence. Convergence is not proof of correct metadata, an identifiable molecular column or an adequate continuum/LSF. Inspect fit residuals, bound hits, selected models, masks and `result.provenance`, not only `result.success`.

`fit_quality.correction_usability` separately records numerical validity and
corrected-pixel retention; it does not override optimiser success or mask
additional pixels. Residual-shift diagnostics evaluate shifted transmission
on the full, wavelength-sorted grid of each physical group and score the
selected pixels at their original positions. Fitting exclusions no longer
compress that pixel coordinate. Missing detector rows cannot be reconstructed
from wavelength alone, so physically removed rows or unlabelled gaps can still
limit interpretation of a shift measured in pixels.

Solver conventions and termination equations: [SciPy `least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
