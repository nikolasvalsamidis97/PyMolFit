# Calculation details

This reference describes the development implementation, not a claim of numerical identity with Molecfit. The [user guide](../user_guide/README.md) needs none of these equations.

Start with the [pipeline overview](pipeline.md) for the complete sequence from input spectrum to corrected output. The pages below explain each calculation in detail.

1. [Atmosphere and absorption](atmosphere_and_absorption.md): profile merging, slant columns, spectroscopy and optical-depth preparation.
2. [Segmentation and regions](segmentation_and_regions.md): physical groups, fitting masks and stellar exclusions.
3. [Spectral model](spectral_model.md): coordinate changes, grids, instrumental convolution and continuum.
4. [Parameter estimation](parameter_estimation.md): initialisation, preliminary comparisons, profiled coefficients and nonlinear fitting.
5. [Correction and uncertainties](correction_and_uncertainties.md): full-coverage evaluation, division, covariance and masks.

## Notation

| Symbol | Meaning |
| --- | --- |
| `i`, `s`, `j` | Atmospheric layer, molecular species, transition. |
| `k`, `p`, `m` | Processing segment, observed pixel, polynomial term. |
| $h$, $\ell$ | Altitude and distance along the line of sight. |
| $P$, $T$, $x_s$, $n_s$ | Pressure, temperature, species number fraction and number density. |
| $N_{s,i}$ | Slant column through layer `i`, molecules per square centimetre. |
| $\tilde\nu=1/\lambda$ | Wavenumber, in inverse centimetres when wavelength is in centimetres. |
| $S_j$, $\phi_{j,i}$, $\sigma_{j,i}$ | Integrated strength, line profile and cross-section. |
| $\tau_{\rm atm}$, $T_{\rm atm}$ | Intrinsic optical depth and transmission before detector/instrumental processing. |
| $\widehat T_{kp}$ | Wavelength-adjusted, detector-sampled and instrumentally broadened transmission. |
| $C_{kp}$, $c_{km}$, $u_{kp}$, $d$ | Local flux continuum, its coefficients, normalised wavelength and degree. |
| $F^{\rm obs}_{kp}$, $F^{\rm mod}_{kp}$, $F^{\rm corr}_{kp}$ | Observed, modelled and corrected flux. |
| $\sigma_{F,kp}$, $\sigma_{T,kp}$ | Flux and fitted-transmission standard deviations. |
| $\mathcal F_k$, $r_{kp}$, $a$, $\mathcal J$ | Fitting pixels, uncertainty-normalised residual, robust scale and objective. |

Additional implementation symbols are defined where introduced. Atmospheric temperature `T` and transmission $T_{\rm atm}$ are different quantities.

## Scope

The default calculation models attenuation, not additive atmospheric emission. Stellar templates define exclusions/weights; there is no joint stellar-flux model in the high-level correction. Corrected spectra retain astrophysical features, continuum structure and any unmodelled additive background.

No correction recovers information in an opaque atmospheric band. Agreement with another implementation does not rule out shared errors in atmospheric inputs or line data.
