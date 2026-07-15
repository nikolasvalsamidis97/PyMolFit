# PyMolFit Physics Parity Audit

This file tracks the formulas that were audited against the Molecfit 4.4.4
bundle installed locally under:

- `~/.criresflow/kits/molecfit-kit-4.4.4-9/telluriccorr-4.3.3/src`
- `~/.criresflow/kits/molecfit-kit-4.4.4-9/molecfit_third_party-1.9.5/lblrtm/src`
- `~/.criresflow/kits/molecfit-kit-4.4.4-9/molecfit_third_party-1.9.5/lnfl/src`

The goal is not to hide differences. The goal is to make each difference
explicit, tested where possible, and tied to the source routine that explains
it.

## Audited Formulas

| Area | Reference source | PyMolFit implementation | Status |
| --- | --- | --- | --- |
| Physical constants | `lblrtm/src/phys_consts.f90` | `src/pymolfit/physics.py` | Defaults to modern Astropy constants. LBLRTM legacy constants are now exposed for parity tests. |
| Radiation term | `lblrtm/src/oprop_voigt.f90`, `RADFN` and `RADFNI` | `src/pymolfit/continuum.py`, `radiation_term_cm`, `radiation_term_interval_cm` | Matched for the pointwise three-branch formula and the interval bookkeeping used by panel accumulation. Tested with LBLRTM `RADCN2 = 1.4387752`. |
| HITRAN line strength temperature scaling | HITRAN formula; LBLRTM applies related correction in `oprop_voigt.f90` after LNFL preprocessing | `src/pymolfit/physics.py`, `line_strength_temperature`, `lblrtm_temperature_scaling_lower_energy` | Correct for HITRAN `.par` intensities with partition sums. LBLRTM unknown-EPP convention is now ported: values near `-1` skip temperature scaling and values below `-1.001` are made positive. Not bit-identical to LBLRTM TAPE3 internals because LNFL preprocesses intensity terms. |
| Pressure/Lorentz width | `lblrtm/src/oprop_voigt.f90` and `nonlte_voigt.f90`, `TMPCOR = TRATIO**TMPALF`; `lnfl/src/lnfl.f`, `BRDMATCH` | `src/pymolfit/physics.py`, `lorentz_hwhm_wavenumber`; `components.py`, broadener correction; `hitran.py`, AER parser | Matched for HITRAN exported temperature exponent convention: `(T_ref/T)**n`. Raw AER F100 records are now parsed with LNFL-style negative `IFLG` side rows, and the Molecfit-installed extra broadener files (`co2_h2o_brd_param`, `wv_co2_brd_param`, etc.) are auto-imported when present beside the AER database. O2/N2 HITRAN air-width self-mixture corrections from `RDLNFL` are ported. |
| Pressure shift | `VNU(I) = VNU(I) + RHORAT*PSHIFT(I)` plus broadener shift correction in LBLRTM | `physics.py`, `pressure_shift_wavenumber`; `components.py`, broadener shift correction | HITRAN text rows use pressure-only shifts. AER/LNFL rows use LBLRTM density-ratio shifts and flagged broadener-specific shift corrections. O2 air-shift self-mixture correction from `RDLNFL` is now ported when O2 self-shift data exist. |
| Atmospheric path and columns | Molecfit atmospheric merge and `lblrtm/src/lblatm.f90` LOWTRAN6 geometry | `atmosphere.py`, `AtmosphereProfile`, `lblrtm_lowtran6_refractivity` | Source-matched for the fixed MIPAS/GDAS merge and refracted spherical path. The workflow derives the reference wavenumber from the input spectrum, evaluates the LOWTRAN6 dry-air and water refractivity terms, and integrates pressure, temperature, and species columns on the refracted path. Vertical PWV remains distinct from the slant column. |
| Doppler width | `MOLEC`, `ALFD1 = SQRT(FAD*TEMP/SMASSI)` | `physics.py`, `doppler_sigma_wavenumber` | Matched to the standard thermal Doppler sigma formula. Tested numerically. |
| Voigt profile | LBLRTM uses optimized/tabulated Voigt machinery | `physics.py`, SciPy `wofz` | Physically correct normalized Voigt evaluation, but not LBLRTM's exact approximation/table path. Tested for normalization. |
| Line wings/cutoff | `lblrtm/src/oprop_voigt.f90`, dynamic line windows, `voigt_init`, and panel logic | `physics.py`, `lblrtm_voigt_hwhm`, `lblrtm_dynamic_line_cutoff_cm`, `lblrtm_dynamic_max_line_cutoff_cm`, `lblrtm_tabulated_voigt_profile_offset`, `lblrtm_panel_voigt_profile_wavenumber`; `components.py`, `line_wing_mode` | Improved but still not bit-identical. PyMolFit has explicit `full`, `hard_cutoff`, `subtracted_cutoff`, `tapered_cutoff`, `lblrtm_subtracted`, `lblrtm_dynamic`, `lblrtm_table`, and `lblrtm_panel` modes. The panel path now performs the causal, source-order F4 prepass, applies total/species `CONVF4` screening, completes R4 before the main pass, and interpolates F4 independently through R3/R2/R1. It also preserves exact caller endpoints on uniform grids. SciPy's Faddeeva function still replaces LBLRTM's original Armstrong approximation and the Python arrays do not reproduce every Fortran panel buffer rounding operation. |
| AER/LNFL line-coupling side records | `lnfl/src/lnfl.f`, F100 `IFLG`; `lblrtm/src/oprop_voigt.f90`, `LNCOR1` and `CNVFNV` | `hitran.py`, `LineList.line_flags`, `line_coupling_a/b`; `components.py`, `_lblrtm_line_coupling_corrections` | Negative F100 flags are now consumed with their auxiliary 200/250/296/340 K rows. `IFLG=1` applies the pressure-dependent strength multiplier and odd line-coupling profile term. `IFLG=3` applies the reduced-width correction. The exact Armstrong Voigt table branch is still approximated through PyMolFit's Voigt machinery. |
| H2O MT_CKD continuum | `lblrtm/src/contnm.f90`, H2O self/foreign sections, MT_CKD 3.5 embedded block data | `continuum.py`, `LBLRTMH2OContinuum`; packaged `data/lblrtm_v12_11_h2o_continuum.npz` | Source-backed for Molecfit's bundled LBLRTM 12.11 coefficient version. The older external `MTCKDH2OContinuum.from_netcdf` path remains available, but the benchmark now defaults to the packaged LBLRTM table. |
| CO2 continuum | `lblrtm/src/contnm.f90`, `FRNCO2`, `BFCO2`, and `XFACCO2` | `continuum.py`, `LBLRTMCO2Continuum`; packaged `data/lblrtm_v12_11_co2_continuum.npz` | Source-backed for the LBLRTM 12.11 CO2-air continuum, including the 2386-2434 cm-1 temperature dependence and 2000-2998 cm-1 correction factors. Exposed through Python and CLI via the packaged LBLRTM continuum options. |
| Rayleigh scattering | `lblrtm/src/contnm.f90`, `XRAYL` branch | `continuum.py`, `lblrtm_rayleigh_optical_depth`; `components.py`, `RayleighScatteringAbsorption` | Source-formula branch ported and vectorized for wavenumbers >= 820 cm-1. Exposed as opt-in `rayleigh=True` / `--rayleigh`. |
| N2 continua | `lblrtm/src/contnm.f90`, `XN2CN` pure-rotation, fundamental, and first-overtone branches | `continuum.py`, `LBLRTMN2FundamentalContinuum`, `LBLRTMN2OvertoneContinuum`, and rototranslational helpers; `components.py`, `N2ContinuumAbsorption` | Source tables, temperature interpolation, collision-partner efficiencies, and optical-depth scaling are ported. External LBLRTM differential audits give integral ratios of 0.986 for the L-band fundamental and 0.998 for the K-band overtone. Exposed as `n2_continuum=True` / `--n2-continuum`. |
| O2 continua | `lblrtm/src/contnm.f90`, `XO2CN` ground-based branches | `continuum.py`, `LBLRTMO2Continuum`; `components.py`, `O2ContinuumAbsorption` | Source-backed O2 fundamental, 1.27 micron, 9100--11000 cm-1, A-band, and visible branches, including branch-specific density and partner scaling. The external 1.27 micron differential audit agrees in integrated optical depth to 0.18%. Exposed as `o2_continuum=True` / `--o2-continuum`. |
| Other continua/CIA | `contnm.f90` and HITRAN CIA tables | `continuum.py`, `TabulatedContinuum`, `HitranCIATable` | External-table supported. PyMolFit rejects source-continuum plus overlapping CIA combinations to prevent double counting. It does not yet bundle every UV or minor-species LBLRTM continuum model. |
| Molecfit synthetic LSF | `telluriccorr/src/mf_kernel_synthetic.c` | `model.py`, composite box/Gaussian/Lorentz kernels and Molecfit Voigt approximation | Improved. The default matches Molecfit's default independent box/Gaussian/Lorentz path but uses analytic integrated Gaussian/Lorentz kernels. Optional `lsf_molecfit_voigt=True` / `--lsf-molecfit-voigt` ports Molecfit's `kern_mode` Voigt approximation with the 200-bins-per-FWHM source sampling rule. |
| Fitting strategy | Molecfit C/CPL fitting pipeline and `cpl_mpfit` | `fit.py` | Partial. PyMolFit now uses Molecfit-compatible default `ftol`, `xtol`, and `gtol` values of `1e-10` and the same default `100 * n_parameters` evaluation budget. It fits species scales, independent segment continua, a constant or globally shared wavelength polynomial, optional per-segment wavelength polynomials, and optional LSF parameters. Local covariance rank is evaluated on a column-normalized Jacobian and transformed back to physical units. The solver remains SciPy trust-region reflective least squares, so its step selection and correlated-parameter trajectory are not MPFIT-identical. |

## Why Some Formulas Are Not Exactly the Same Yet

Molecfit delegates the physical transmission calculation to LBLRTM, and LBLRTM
expects line data after LNFL preprocessing. A raw HITRAN `.par` file and an
LBLRTM TAPE3/TAPE8-derived file do not always expose the same internal
variables, even if their column names look similar.

The clearest example is the pressure broadening exponent:

- PyMolFit's public formula uses the HITRAN convention: `gamma(T) = gamma_ref * (T_ref/T)**n`.
- LBLRTM source applies `TMPCOR = (T/T_ref)**TMPALF`.
- LNFL's optional TAPE8 text output writes `tmpalf_hit = 1.0 - TMPALF`.

That means a positive exponent in an exported AER/HITRAN-like text row should
not be copied directly into the raw LBLRTM `TMPALF` formula.

## Tests Added For Intermediate Outputs

`tests/test_physics_parity.py` now checks:

- LBLRTM `RADFN` branch values using the LBLRTM legacy radiation constant.
- LBLRTM `RADFNI` interval radiation-term bookkeeping, including reused
  `RDLAST` and explicit next-frequency branches.
- HITRAN temperature scaling of line intensity with an explicit partition ratio.
- LBLRTM lower-state-energy sentinel behavior for unknown EPP values.
- Lorentz half-width scaling with the HITRAN exponent convention.
- O2/N2 self-mixture correction of HITRAN air widths and O2 pressure shifts.
- HITRAN-vs-AER pressure-shift convention, including an end-to-end shifted-line test.
- Standard atmosphere spherical slant geometry, vertical PWV scaling, and
  workflow pre-slanting that avoids double-applying airmass.
- The exact LOWTRAN6 refractivity expression and the increase in refracted
  spherical path length relative to the unrefracted path.
- FITS-header GDAS observation time selection using ESO `UTC` seconds and
  persistence of the exact before/after GDAS provenance in cached profiles.
- AER/LNFL optional broadener row parsing and a broadener-driven line-wing test.
- LBLRTM-inspired subtracted line-wing mode defaulting to a 25 cm-1 cutoff.
- LBLRTM dynamic line-window controls from `oprop_voigt.f90`: `AVRAT`,
  `ALFV`, `ALFMAX`, `SAMPLE`, `ALFAL0`, and `HWF3`.
- Experimental LBLRTM tabulated and panel Voigt modes: finite support,
  symmetry, source interpolation coefficients, near-core agreement with exact
  Voigt, API execution, and CLI exposure.
- Source-order two-pass F4 weak-line rejection, independent F4 panel
  interpolation, chunk invariance, mixed source precision, and exact endpoint
  preservation.
- Source-derived panel windows that keep F1/F2/F3 dynamic while the separately
  accumulated F4 field supplies the 25 cm-1 closure unless a user explicitly
  requests a hard cutoff.
- HITRAN line preselection for LBLRTM-style wing modes, including a coarse-grid
  case that would be missed by the older fixed 25 cm-1 margin.
- Sparse finite-wing Voigt accumulation against the older dense profile-matrix
  path for hard cutoff, subtracted cutoff, tapered cutoff, and LBLRTM dynamic
  cutoff modes.
- Doppler sigma against the thermal formula.
- Voigt profile normalization on a dense grid.
- H2O continuum self/foreign scaling, radiation term, and density factors at grid points.
- LBLRTM `contnm.f90` Rayleigh scattering formula and opt-in transmission effect.
- LBLRTM `contnm.f90` N2 rototranslational source table interpolation,
  optical-depth formula, component behavior, and workflow/CLI exposure.
- Molecfit synthetic LSF kernel sizes and normalisation rules.
- Molecfit `kern_mode` synthetic Voigt approximation kernel size,
  symmetry, normalization, and CLI exposure.
- LNFL/TAPE8 exponent conversion convention.

These tests do not prove full Molecfit equivalence. They pin the intermediate
formula behavior so future changes cannot silently move PyMolFit away from the
audited physics.

## Remaining Work For Stronger Molecfit/LBLRTM Parity

1. Tighten `lblrtm_panel` against small golden LBLRTM cases. Source-order F4,
   interpolation coefficients, line-coupling terms, and local R1/R2/R3
   emulation are present, but exact Fortran buffer rounding and the original
   Armstrong Voigt approximation are still not proven bit-for-bit.
2. Expand continuum support beyond the current source-backed H2O, CO2, O2,
   Rayleigh, N2, and table-driven pieces, especially for O3 and remaining
   `contnm.f90` minor-species branches.
3. Add optional golden tests that run the local Molecfit/LBLRTM executable on a
   tiny synthetic case and extend the current captured-model audit to
   layer-by-layer optical depth, line-only transmission, and continuum-only
   transmission.

## Current Local Benchmark

The source-audit `lblrtm_panel` run was regenerated across all 18 rho01 CNC
L-band chips with a self-contained fixed MIPAS/GDAS atmosphere and the input
spectrum's reference wavenumber:

- Transmission RMS difference: min `0.00811`, median `0.02087`,
  mean `0.01970`, max `0.04198`
- Continuum-invariant line-shape RMS: median `0.01182`, mean `0.01283`,
  max `0.02479`
- Worst chip in this run: 13
- Previous panel benchmark: mean `0.04587`, max `0.09587`

The independent Beta Pic HARPS O2 rerun also improved for the two 2017 epochs:
RMS changed from `0.01088` to `0.00312` and from `0.00762` to `0.00372`.
The 2014 and 2019 epochs moved only slightly. This cross-instrument result is
evidence that the improvement comes from general atmosphere and calculation
changes, not detector- or observation-specific coefficients.

A fixed-parameter X-shooter audit now separates radiative-transfer parity from
optimizer trajectory. It evaluates both pipelines with the same checksum-pinned
AER 3.9 catalogue and the molecular columns, wavelength shift, and LSF reported
by Molecfit. Across O2 A, H2O J, H2O H, and H2O/CO2 K, the transmission RMS
differences are `0.00172`, `0.00117`, `0.00038`, and `0.00052`; optical-depth
correlations exceed `0.999989` in all four cases. This does not remove the
separately reported fitted-solution warnings, but it shows that the fixed
physical calculation path is closely matched without a chip-specific
correction.

An additional held-out test uses the public UVES spectrum and optimized fit
ranges published in the ESO Molecfit Reflex tutorial. With one shared
first-order wavelength polynomial and a Gaussian LSF in both solvers, the
transmission RMS is `0.00369`, the weighted-objective ratio is `0.9935`, and
the relative-residual-scatter ratio is `0.9967`. PyMolFit's five nonlinear
parameters are full rank. This case did not supply any fitted Molecfit
coefficient to PyMolFit.

For the 2017 HARPS case, the downloaded/interpolated GDAS pressure, height,
temperature, and relative humidity arrays match Molecfit's saved `GDAS.fits`
bit-for-bit. The merged fixed MIPAS/GDAS profile agrees to floating-point
roundoff (maximum height difference about `1.4e-14` km and pressure difference
about `1.1e-13` hPa).

The current self-contained, accurate-panel 18-segment benchmark takes
`81.059 s` from `SCIENCE_A.fits`, including atmosphere construction, the full
`1e-32` line list, source-backed H2O/CO2/N2 continua, non-overlapping CIA,
physical basis construction, fitting, and product writing. With both reported
optimizer tolerance values set to `1e-10`, the official Molecfit model recipe
takes `64.999 s` on the same machine and input, so the current PyMolFit run is
`1.247x` as long. MPFIT and SciPy do not have identical stopping semantics, so
this remains an operational rather than iteration-for-iteration comparison.
Molecfit takes `30.77 s` with its saved, looser `0.01` tolerance settings. On
the isolated 2017 HARPS Na D case, PyMolFit takes `2.33 s` versus Molecfit's
`29.72 s`.

The speedup from the previous `317.42 s` run comes from general algorithmic
changes only: vectorized multi-temperature CIA interpolation, reusable overlap
rebin plans, lazy post-screening F4 profiles, grouped dense finite-difference
Jacobians, bounded parallel segment basis construction, and mode-aware line
chunks (`512` for dynamic Voigt and `16384` for sparse panel accumulation).
The current path additionally reuses first-pass line/layer state, continuum
interpolation geometry, and prepared broadener deltas, skips spectrally
disjoint continuum branches, and vectorizes independent F4 support searches.
No observation-specific coefficient or initial molecular column was added.

For the 2017 HARPS O2 case, the corrected source-window panel path reduced RMS
against Molecfit from dynamic mode's `0.003716` to `0.002753`, reduced the
fit-to-data cost from `862.66` to `779.61`, and completed in `2.96 s`.
