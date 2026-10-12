# Atmosphere and absorption

Implementation: [`atmosphere.py`](../../src/pymolfit/atmosphere.py), [`gdas.py`](../../src/pymolfit/gdas.py), [`hitran.py`](../../src/pymolfit/hitran.py), [`physics.py`](../../src/pymolfit/physics.py), [`components.py`](../../src/pymolfit/components.py), [`radiative_transfer.py`](../../src/pymolfit/radiative_transfer.py).

## Profiles and observation geometry

The high-level default is the MIPAS--GDAS builder. MIPAS supplies reference composition and upper-atmosphere structure; GDAS supplies time-local pressure, temperature and water profiles where available. The current reference MIPAS choice is equatorial, not an automatically fitted atmospheric type.

For GDAS times $t_0\le t\le t_1$, the temporal interpolation has the form

$$y(h,t)=(1-w)y(h,t_0)+w y(h,t_1),\qquad w=\frac{t-t_0}{t_1-t_0}.$$

Here `y` represents an interpolated profile quantity; profiles are first represented on compatible altitude levels. This is a time-weighted interpolation, not a new atmospheric observation. Automatic acquisition uses cached/downloaded time-local profiles when possible and otherwise the packaged two-month-average profiles. Acquisition mode and source are recorded. Local pressure, temperature and humidity adapt the lowest levels when supplied; the vertical grid begins at the observatory, not sea level.

## Number densities and integrated columns

For species number fraction relative to total gas,

$$n_s(h)=x_s(h)\frac{P(h)}{k_{\rm B}T(h)},\qquad N_{s,i}=10^{-4}\int_{\mathcal L_i}n_s[h(\ell)]\,d\ell.$$

The factor $10^{-4}$ converts molecules per square metre to molecules per square centimetre when pressure is Pa and distances are metres. It is absent if all lengths are already in centimetres.

Source-profile ratios and $x_s$ are not always the same convention. The MIPAS/GDAS layer builder treats the water ratio $q_w$ relative to dry air, giving $n_w=n_{\rm total}q_w/(1+q_w)$ and $n_{\rm dry}=n_{\rm total}-n_w$; other source species use $n_s=q_s n_{\rm dry}$. The resulting `AtmosphereLayer.mixing_ratios` are effective fractions relative to total gas.

Within level boundaries the implementation exponentially interpolates pressure, total number density and molecular number density. Gauss--Legendre quadrature integrates columns along the path. Effective layer pressure and temperature are density-weighted; an effective path length preserves the integrated amount rather than replacing every layer by its midpoint conditions.

Let $\eta(h)$ be refractive index and $R_\oplus$ Earth radius. Spherical ray geometry conserves

$$b=\eta(h)(R_\oplus+h)\sin\theta(h),\qquad \frac{d\ell}{dh}=\left[1-\frac{b^2}{\eta(h)^2(R_\oplus+h)^2}\right]^{-1/2}.$$

The initial zenith angle is set by $\cos\theta_0=1/X$ for input airmass `X`. The implementation uses an LBLRTM-compatible refractivity calculation. Refraction-free and plane-parallel alternatives exist at lower level. Slant columns already include the path geometry; the physical workflow does not multiply by the same airmass twice.

## Catalogue preparation

AER records supply line position, isotope, lower-state energy, reference strength, pressure shifts and broadening coefficients. Relevant wavelength windows include a wing/alignment margin. Supported auxiliary records add temperature-dependent broadening and line-coupling information. The reader skips non-line records and retains catalogue provenance. Only species present in the loaded catalogue/model contribute: including a species in the atmosphere alone does not supply missing spectroscopy.

The ideal isolated-line relations are

$$\sigma_{j,i}(\tilde\nu)=S_j(T_i)\phi_{j,i}(\tilde\nu),\qquad \int_{-\infty}^{\infty}\phi_{j,i}(\tilde\nu)\,d\tilde\nu=1,$$

$$S_j(T)=S_j(T_{\rm ref})\frac{Q_j(T_{\rm ref})}{Q_j(T)}\exp\left[-c_2E_j''\left(\frac1T-\frac1{T_{\rm ref}}\right)\right]\frac{1-e^{-c_2\tilde\nu_j/T}}{1-e^{-c_2\tilde\nu_j/T_{\rm ref}}},\qquad T_{\rm ref}=296\,{\rm K}.$$

`Q` is the isotopologue partition function, $E_j''$ lower-state energy and $c_2=hc/k_{\rm B}$ in units compatible with inverse centimetres. Partition data, supported isotope masses and legacy-compatible branches are selected by the numerical configuration. A reference strength has units cm/molecule, so the cross-section has units square centimetres per molecule.

Thermal Gaussian sigma is $\sigma_{D,j}=\tilde\nu_j\sqrt{k_{\rm B}T/m_j}/c$. A representative pressure law is $\gamma_j(T)=\gamma_{j,\rm air}(T_{\rm ref}/T)^{n_j}P_{\rm air}+\gamma_{j,\rm self}(T_{\rm ref}/T)^{n_{j,\rm self}}P_s$, with pressures in the coefficients' reference units. The local line centre includes its pressure shift. Supported alternative broadeners replace this simplified two-perturber expression where available.

The underlying Voigt profile is evaluated through SciPy's Faddeeva function. The default `lblrtm_panel` treatment additionally uses nested grids, line-dependent domains and a separate far-wing contribution extending to 25 inverse centimetres. Effective truncated/subtracted profiles are not generally unit-area profiles; the infinite isolated-Voigt normalisation above must not be imposed after these operations.

## Optical depth and continuum

$$\tau_{\rm lines}(\tilde\nu)=\sum_i\sum_sN_{s,i}\sum_{j\in s}\sigma_{j,i}(\tilde\nu),\qquad \tau_{\rm atm}=\tau_{\rm lines}+\tau_{\rm cont},\qquad T_{\rm atm}=e^{-\tau_{\rm atm}}.$$

Layer contributions retain their own pressure, temperature and columns while being evaluated; summing them produces one spectral optical-depth array. Transmission is exponentiated before detector averaging. Averaging optical depth and then exponentiating would give a different result for unresolved lines.

Available continuum/broad-band implementations include H2O, CO2, N2, O2 and O3, optional external continuum/CIA tables, and Rayleigh extinction. In physical mode, unset H2O/CO2 continuum inputs resolve to packaged coefficients when the corresponding line species is present. The N2/O2/O3 and Rayleigh switches default to off; the paper-comparison scripts explicitly enable the molecular continua. Availability is therefore not identical to activation. Inspect configuration/provenance rather than inferring continuum coverage from the species list. Overlapping built-in and external CIA contributions are checked to avoid double counting.

## What fitting updates

The physical calculation prepares reference optical-depth basis arrays $B_s(\lambda)$; enabled extra components may have their own basis labels. Trial scales $q_s$ give

$$T_{\rm atm}(\lambda;\boldsymbol q)=\exp\left[-\sum_s q_s B_s(\lambda)\right].$$

The notation above absorbs the already-resolved geometry into `B`. The engine reuses these arrays while fitting; it does not rebuild atmospheric chemistry, pressure broadening or temperature for every abundance trial. This distinction matters for continua with nonlinear density dependence and large departures from the reference atmosphere. Numerical agreement with a calculation that reconstructs the atmosphere at every trial is not guaranteed.

Spectroscopic conventions: [HITRAN definitions](https://hitran.org/docs/definitions-and-units/). Physical reference implementation: [AER LBLRTM](https://github.com/AER-RC/LBLRTM). Source-specific coefficients and attribution are listed in [third-party notices](../../THIRD_PARTY_NOTICES.md).
