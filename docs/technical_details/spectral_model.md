# From intrinsic transmission to model flux

Implementation: [`model.py`](../../src/pymolfit/model.py), [`fit.py`](../../src/pymolfit/fit.py), [`spectrum.py`](../../src/pymolfit/spectrum.py).

## Wavelength correction

The intrinsic $T_{\rm atm}$ is the transmission defined in [atmosphere and absorption](atmosphere_and_absorption.md). A trial polynomial correction shifts its alignment relative to observed pixel coordinates. Schematically, for physical group `g`,

$$\delta\lambda_g(u)=\sum_{m=0}^{q_g}b_{gm}u^m,\qquad T^{\rm shift}_{\rm atm}(\lambda)=T_{\rm atm}[\lambda-\delta\lambda_g(\lambda)].$$

Here `q_g` is the wavelength polynomial degree, `b_gm` its coefficients and `u` the normalised coordinate across the correction's scope. Positive constant displacement moves model lines redward. Automatic fitting can parameterise offsets in detector pixels and convert them using local dispersion; public shift overrides use microns. The global scope shares one polynomial, while group scope shares a polynomial only within a physical order/detector group.

`wavelength_order` determines model flexibility; `initial_wavelength_shift` determines the starting alignment, and bounds limit its exploration. They are not three successive corrections. Input frame/air--vacuum conversion occurs before fitting residual calibration errors.

## Grids and pixel overlap

The native radiative-transfer grid resolves narrow physical lines. A computational model grid may differ from both this grid and the observed detector grid. The usual oversampling setting is five samples per observed pixel, but the physical grid can require finer spacing from line widths. Padding extends the model beyond the fitted segment for line wings, wavelength shifts and convolution. The original detector samples are retained.

For input model bins `q` with edges $e_q^-,e_q^+$ and observed pixel edges $\lambda_{kp}^-,\lambda_{kp}^+$, define overlap

$$w_{kpq}=\max[0,\min(\lambda^+_{kp},e_q^+)-\max(\lambda^-_{kp},e_q^-)].$$

The default `molecfit_overlap` mode treats the model values as piecewise constant and forms

$$\overline T_{kp}=\frac{\sum_q w_{kpq}T^{\rm shift}_{{\rm atm},q}}{\sum_q w_{kpq}}.$$

Pixel boundaries are constructed from neighbouring wavelength centres. Edge pixels use extrapolated half spacings; padding supplies the surrounding model. The denominator normalises the actual overlap. This is not interpolation of observed flux and not an unweighted average of however many fine samples happen to fall inside a pixel.

The default then applies the instrumental kernel on the detector grid:

$$\widehat T_{kp}=\sum_{p'}K_{k,p,p'}\overline T_{kp'},\qquad \sum_{p'}K_{k,p,p'}=1.$$

For constant width this is a discrete convolution. For wavelength-dependent width each output location uses its local kernel. Lower-level modes can instead convolve on a high-resolution model grid before integration/sampling. These orders of operations are not generally interchangeable for undersampled lines. The default overlap path directly samples the native transmission before detector-grid convolution.

## Instrumental kernel

The composite kernel combines enabled boxcar, Gaussian and Lorentzian components by convolution and normalises their discrete sum. The Gaussian and Lorentzian are integrated over kernel pixels rather than evaluated only at their centres; finite support is controlled by kernel width. A separate lower-level option uses the Molecfit-compatible Voigt approximation.

$$G(v)=\frac{e^{-v^2/(2\sigma^2)}}{\sqrt{2\pi}\sigma},\quad L(v)=\frac{\gamma}{\pi(v^2+\gamma^2)},\quad B(v)=\frac1w\,\mathbf1_{|v|\le w/2}.$$

Here `v` is displacement in detector pixels, not radial velocity; `sigma` is Gaussian standard deviation, Lorentzian FWHM is $2\gamma$, and `w` is boxcar full width. Zero-width components act as identity kernels. Gaussian FWHM is $2\sqrt{2\ln2}\,\sigma$. Normalisation preserves a constant input level; broadening changes line shape, not the atmospheric column.

A variable width uses

$$w(\lambda)=w_{\rm ref}(\lambda/\lambda_{\rm ref})^\alpha.$$

The same rule applies to the enabled widths in pixels. Explicit `lsf_variable_width=True` uses $\alpha=1$; `False` is constant, and automatic selection tests a fitted exponent. This is a parameterisation in detector pixels, not an unconditional assertion of constant resolving power on every sampling grid.

## Flux continuum

$$u_{kp}=2\frac{\lambda_{kp}-\lambda_{k,\min}}{\lambda_{k,\max}-\lambda_{k,\min}}-1,\qquad C_{kp}=\sum_{m=0}^{d}c_{km}u_{kp}^m,\qquad F^{\rm mod}_{kp}=C_{kp}\widehat T_{kp}.$$

Each segment has its own coefficients `c_km`. Their smooth flux baseline can describe continuum level, slope and curvature, but not arbitrary stellar lines. This flux continuum is distinct from broad atmospheric absorption in $\tau_{\rm cont}$. The default degree starts at 2 and is checked against alternatives during fitting.

Both preliminary and main fits compare this same kind of model flux with observed flux inside $\mathcal F_k$. They never compare a dimensionless transmission directly with unnormalised observed flux.
