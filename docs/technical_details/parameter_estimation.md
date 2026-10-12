# Parameter estimation

Implementation: [`workflow.py`](../../src/pymolfit/workflow.py), [`fit.py`](../../src/pymolfit/fit.py), [`_automatic.py`](../../src/pymolfit/_automatic.py). The model and notation are defined in [spectral model](spectral_model.md).

## Initialisation

1. Build fitting masks and the fixed reference atmosphere/opacity basis.
2. Start molecular scales at unity, except explicit initial/fixed values. Weak species with expected peak optical depth below 0.005 in fitting pixels remain fixed at their starting scales.
3. Start alignment from input calibration and frame metadata, with coarse offset searches where configured.
4. Estimate Gaussian instrumental sigma from resolving power and detector dispersion, then narrow observed features if needed. Without usable information, automatic mode does not invent a freely fitted width. Numerical widths with `fit_...="auto"` are fixed; set `True` to fit them.

For locally uniform dispersion $\Delta\lambda$, the resolving-power estimate is $\sigma_{\rm pix}\simeq\lambda/(2.355R\Delta\lambda)$, where `R` is instrumental resolving power. The workflow aggregates usable estimates rather than assuming one arbitrary sampling scale.

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

After an initial joint solution, automatic continuum selection compares degrees 1--4 on temporarily withheld wavelength blocks at fixed transmission. Block selection reduces leakage from adjacent correlated samples. The quadratic is retained unless a simpler or more predictive alternative passes the score/improvement checks; insufficient data leave it unchanged. Automatic robust scale uses residual MAD, clipped to 1--2. Accepted changes can trigger one additional joint fit. This validation of numerical choices is not an independent validation of atmospheric accuracy.

## Stopping and diagnostics

The default tolerances are `ftol=xtol=gtol=1e-10`. SciPy's trust-region stopping tests use relative objective change (with adequate model agreement), scaled parameter step, or the bound-aware gradient norm. Satisfying an enabled convergence test stops that solver call. `max_nfev` limits evaluations per call; numerical derivative work and preliminary/refit calls add to total cost.

A budget stop is not convergence. Convergence is not proof of correct metadata, an identifiable molecular column or an adequate continuum/LSF. Inspect fit residuals, bound hits, selected models, masks and `result.provenance`, not only `result.success`.

Solver conventions and termination equations: [SciPy `least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
