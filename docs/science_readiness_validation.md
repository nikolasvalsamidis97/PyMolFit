# Science-readiness validation

This document defines the validation gate for PyMolFit. Passing unit tests is
necessary but is not sufficient for scientific use. The campaign is run with:

```bash
PYTHONPATH=src python local_tests/run_science_readiness_validation.py
```

It writes machine-readable results and diagnostic plots under
`local_tests/science_readiness/results/`. Existing official Molecfit products
are reused when present; remove a case's `BEST_FIT_MODEL.fits` and
`BEST_FIT_PARAMETERS.fits` to force that reference run to be regenerated.

## Validation checklist

1. **Synthetic parameter recovery:** inject known molecular columns and recover
   H2O, O2, CO2, CH4, N2O, CO, and O3. A separate joint test recovers molecular
   scale, wavelength shift, and Gaussian LSF width.
2. **Uncertainty calibration:** run deterministic Monte Carlo ensembles and
   test the nominal 68% molecular-scale interval coverage and z-score moments
   for both single- and shared multi-segment fits. Refit named alternative
   model configurations and verify that their transmission spread is
   propagated separately from the local statistical covariance.
3. **Wavelength coverage:** exercise optical, J, H, K, L, M, and N windows.
4. **Instrument coverage:** retain real HARPS, UVES, X-shooter, CRIRES+,
   Keck/HIRES, and Keck/KPF evidence, covering high-resolution optical/IR,
   medium-resolution optical/NIR, and independent non-ESO archive formats.
5. **Atmospheric conditions:** test airmass and PWV responses, exact time-local
   GDAS inputs, and the packaged seasonal GDAS fallback.
6. **Failure behavior:** test saturated-line masking, propagated uncertainty,
   low S/N, missing intervals, rank-deficient covariance, unsorted fit-mask
   alignment, operation without an online GDAS result, and fail-closed handling
   of an unknown observatory without coordinates.
7. **Numerical convergence:** compare 80 and 160 atmospheric layers and verify
   that line-chunk size does not change the transmission.
8. **External comparison:** compare fitted transmission and correction quality
   with official Molecfit products for four X-shooter bands, four HARPS epochs,
   the official ESO UVES tutorial spectrum, two Keck/HIRES O2 bands, three
   Keck/KPF O2 order segments, and 18 CRIRES+ L-band segments.
   Source-level line/continuum cases are also checked against external LBLRTM
   12.11 used only for validation.
9. **Independent review:** create a blind-review packet for an experienced
   scientist, with candidate identity permuted per case and an answer key kept
   outside the reviewer archive. The archive is built from an explicit file
   allow-list and checked against a SHA-256 manifest. A reviewer-selected
   held-out spectrum is required. This remains a manual gate and cannot be
   passed by the developer.
10. **Performance and packaging:** compare matched wall time with Molecfit, run
    the complete automated test suite, and build/test the wheel artifact.
11. **Line-data acquisition:** exercise checksum-verified AER archive
    installation, nested ESO-kit extraction, offline cache reuse, and automatic
    wavelength-window selection. The campaign and production workflow both use
    the pinned official AER 3.9 Zenodo catalogue. A forced authenticated HITRAN
    API v2 request remains an optional validation of the alternative direct-API
    feature.

No observation-specific molecular coefficient, detector correction, or fitted
Molecfit parameter is copied into PyMolFit. Molecfit is executed only by the
validation program as an external reference.

## Public validation data

The campaign downloads public ESO Phase 3 X-shooter VIS and NIR spectra of
HD 53123 from program `60.A-9022(C)`:

- `ADP.2026-03-26T15_48_39.243.fits` (VIS)
- `ADP.2026-03-26T15_48_39.204.fits` (NIR)

It also downloads the public UVES spectrum
`ADP.2020-06-08T15_07_14.471.fits` of HD 127750 used by the official ESO
Molecfit Reflex tutorial. The UVES test uses the tutorial's published
`0.586--0.600`, `0.625--0.640`, and `0.645--0.653` micron fit regions rather
than intervals selected from PyMolFit residuals.

The campaign also retrieves two public level-1 Keck/HIRES orders from KOAID
`HI.20040824.18925.fits` (BD+17 3248). The established O2 B and A bands are
fitted jointly after the documented MAKEE heliocentric-vacuum wavelength scale
is converted back to topocentric vacuum. KOA labels these automated
extractions as browse products, so this case validates portability and
telluric behavior, not publication-quality upstream extraction.

An additional independent test retrieves the public, science-ready KPF DRP
level-1 Vega product `KP.20250519.55029.51_L1.fits` from KOAID
`KP.20250519.55029.51.fits`. It fits the pre-declared O2 B band and the two KPF
orders spanning the O2 A band using the `RED_SCI_*1` slice. The KPF wavelength
arrays are vacuum wavelengths; SCI1 is analyzed without combining the three
slightly different wavelength solutions. The product was reduced by KPF DRP
`v2.8.2` and is not an ESO or Molecfit-derived reduction.

The source URLs, archive identifiers, and acknowledgement requirements are
recorded in `local_tests/science_readiness/data/README.md`. SHA-256 hashes for
both spectra and the AER line cache are embedded in every JSON report.

## Acceptance rules

- Synthetic molecular scales must be recovered within 3% and within three
  reported standard errors.
- Joint scale/shift/LSF recovery must meet 2%, 0.1 pixel, and 15%, respectively.
- Monte Carlo one-sigma coverage must lie between 0.50 and 0.82, with z-score
  mean within 0.5 and standard deviation between 0.7 and 1.3.
- A masked intrinsic absorption line must retain equivalent width within 2%
  and depth within 0.01 after telluric correction.
- Rank-deficient fits must expose non-identifiability and must not return a
  finite covariance or propagated model uncertainty.
- Real-data normalized scatter must improve by at least 10% after correction.
- Direct transmission comparison requires RMS <= 0.02 overall and <= 0.03 in
  telluric pixels. A disagreement is a warning, rather than a false failure,
  only when PyMolFit has both a lower weighted objective and lower corrected
  residual scatter than that Molecfit run.
- The held-out UVES tutorial case has a stricter direct gate: RMS <= 0.005,
  telluric-pixel RMS <= 0.01, maximum absolute difference <= 0.02, no
  regression in weighted objective/scatter, and a full-rank nonlinear
  covariance.
- The non-ESO Keck/HIRES and Keck/KPF gates require overall and telluric-pixel transmission
  RMS <= 0.01, maximum absolute difference <= 0.03, no weighted-objective or
  corrected-RMS regression beyond 5%/10%, at least fivefold raw-to-corrected
  telluric RMS improvement, and full nonlinear covariance rank.
- The CRIRES+ direct-transmission target remains mean RMS <= 0.01 and maximum
  segment RMS <= 0.03. When smooth continuum/transmission allocation differs,
  a separate science-facing gate requires continuum-invariant line-shape mean
  RMS <= 0.015, maximum <= 0.03, and no regression in weighted objective or
  corrected residual scatter. Direct disagreement remains a warning.
- The matched PyMolFit/Molecfit wall-time ratio must be <= 1.25.

These are validation criteria, not production fit coefficients. They do not
alter the radiative-transfer calculation.

## Current result

The campaign currently reports `VALIDATION_INCOMPLETE`:

- 80 checks pass, with no automated failures or skipped checks. Four of these
  are required fixed-parameter X-shooter radiative-transfer parity gates.
- 4 non-gating direct-transmission diagnostics warn: X-shooter optical, J,
  and K plus the aggregate CRIRES+ L-band comparison. Their science-facing
  weighted-fit, correction, and continuum-invariant shape checks pass. The
  X-shooter H-band direct comparison now passes.
- 1 additional non-gating O2 A-band plotting-grid diagnostic warns.
- 1 external gate remains manual: the independent blind/held-out review. A
  forced authenticated HITRAN API request is reported as an optional feature
  check and is not required by the default AER workflow.

The complete suite has 249 passing tests under both Python 3.12 and 3.13. The
review packet hashes every blind case, preserves
completed responses only while that case is unchanged, and converts a complete
signed assessment into a strict automated `PASS` or `FAIL` gate. Its
deterministic reviewer ZIP has SHA-256
`83bbc322e66a39cf8133d7f7be8af0434cb0d41c1ce974719e43c4394ff0953f`;
the sibling private answer key is not an archive member.

The molecular recovery tests, single- and multi-segment uncertainty coverage,
intrinsic-line preservation, output provenance, atmosphere, convergence,
failure, external source-physics, correction-quality, and runtime checks pass.
The campaign also verifies fail-closed behavior for unknown observatory
metadata and an explicit two-variant LSF systematics ensemble; the latter has
a 95th-percentile transmission spread of `0.00601` and propagates that term in
quadrature with local statistical uncertainty.
Fit products include hashes of the spectrum, source file, line data, selected
transitions, atmosphere, component data, and full fit configuration, together
with the exact fit mask and active parameter bounds.
PyMolFit and Molecfit consume numerically identical time-local GDAS source
profiles in the X-shooter cases (maximum differences are floating-point
roundoff).

On the official UVES tutorial spectrum, PyMolFit and Molecfit transmission
agree to RMS `0.00369` (`0.00510` on telluric pixels; maximum `0.00768`). The
PyMolFit/Molecfit weighted-objective ratio is `0.9936`, the relative-residual
scatter ratio is `0.9968`, and the five nonlinear parameters have full
covariance rank. The matched Gaussian-LSF fit took `6.00 s` in PyMolFit and
`105.62 s` in Molecfit on the validation machine.

On the non-ESO Keck/HIRES O2 case, transmission agrees to RMS `0.00524`
(`0.00563` on telluric pixels; maximum `0.02759`). The weighted-objective ratio
is `1.012`; telluric RMS improves by factors of `6.09` and `6.16` for PyMolFit
and Molecfit, respectively; and the four nonlinear parameters have full
covariance rank. The run took `1.95 s` in PyMolFit and `49.44 s` in Molecfit.
Because KOA describes this automated extraction as a browse product, it is
portability evidence rather than a substitute for the held-out review.
In a two-fold blocked validation, every pixel is predicted by a fit that did
not consume its 64-pixel block. Prediction coverage is `1.0`, reliable
correction coverage is `0.953`, unseen-pixel telluric relative RMS improves by
`3.88x`, and weighted RMS improves by `19.83x`. The two refits took `4.22 s`.

On the independently reduced Keck/KPF Vega spectrum, transmission agrees to
RMS `0.00546` (`0.00582` on telluric pixels; maximum `0.02149`). PyMolFit's
weighted objective is `0.874` times Molecfit's, and telluric-region RMS
improves by factors of `21.34` and `20.13`, respectively. All four nonlinear
parameters have full covariance rank. PyMolFit took `4.13 s`; Molecfit took
`63.59 s`. Both solvers used the same time-interpolated Keck GDAS profile.
Three pre-declared atmosphere/continuum alternatives took `12.80 s` in total.
Their transmission systematic has median RMS `0.000271`, 95th-percentile RMS
`0.00134`, maximum envelope `0.00694`, and finite coverage of `1.0`; this term
is propagated into the corrected-flux uncertainty.
Its independent blocked validation has prediction coverage `1.0`, reliable
correction coverage `0.938`, unseen-pixel telluric relative RMS improvement
`13.66x`, and weighted RMS improvement `67.87x`. The two refits took `8.20 s`.
Both blocked splits are fixed by pixel position and never inspect residuals or
Molecfit products.

The latest matched 18-segment CRIRES+ benchmark took `81.059 s` for PyMolFit
and `64.999 s` for the Molecfit model recipe, a wall-time ratio of `1.247`.
This fresh
run includes the full `1e-32` line-list threshold, source-backed LBLRTM H2O,
CO2, and N2 continua, and non-overlapping CO2-CO2/O2-O2 CIA. The two optimizers
do not have identical iteration semantics, so this is an operational rather
than per-iteration comparison.

The CRIRES+ direct transmission has mean RMS 0.01970 and maximum 0.04198 versus
Molecfit, but the continuum-invariant line-shape mean RMS is 0.01283 (maximum
0.02479), the total weighted-objective ratio is 0.760, and the median corrected
scatter ratio is 0.726. The production and Molecfit reference runs use the same
checksum-pinned AER 3.9 catalogue. The original X-shooter comparison differed in optical,
J, H, and K while PyMolFit had lower weighted objective and corrected scatter
than the retained Molecfit runs. A separate fixed-parameter audit, which uses
Molecfit-reported molecular columns, wavelength shift, and LSF rather than
refitting either model, gives transmission RMS `0.00038`--`0.00172` across O2
A, H2O J, H2O H, and H2O/CO2 K, with optical-depth correlations above
`0.999989`. After making both fitted X-shooter comparisons use only a
constant wavelength shift, H passes the direct threshold; optical, J, and K
remain warnings. In those three bands PyMolFit's weighted objectives are
`0.0590`, `0.0901`, and `0.0663` times Molecfit's and its corrected scatter is
also lower. The time-local GDAS inputs match exactly, and all 50 merged
MIPAS-GDAS height, pressure, temperature, and fitted-species levels match Molecfit to
at worst `5.82e-11` in their native output units. This localizes the remaining
disagreement to the fitted radiative-transfer/optimizer solution rather than
site metadata or profile construction. These disagreements remain visible and
must be judged in the independent review; they are not hidden by
observation-specific calibration.

## Uncertainty scope

Reported covariance is a local linearized statistical covariance conditional
on the chosen atmospheric profile, line database, continuum family, mask, and
LSF model. Corrected-flux uncertainty propagates both observed-flux and fitted
transmission uncertainty. `fit_tellurics_with_systematics` can additionally
refit user-supplied atmosphere, continuum, line-treatment, or LSF alternatives;
it stores the RMS and envelope of their transmission differences and combines
the RMS term with statistical transmission uncertainty in quadrature. The
ensemble is not automatically defined because defensible perturbation sizes
must come from independent site, atmospheric-product, line-data, or instrument
uncertainties. Shared multi-segment covariance is propagated to every segment.
Numerically non-identifiable fits report `NaN` covariance and output
uncertainty and expose the deficient rank in the fit product.

## Evidence files

- `science_readiness_report.json`: verdict, versions, hashes, and all checks.
- `science_readiness_checks.csv`: flat check table for notebooks/CI.
- `campaign_metrics.csv`: fit times, residual statistics, parameters, and RMS.
- `science_readiness_summary.png`: compact campaign overview.
- `xshooter/*/comparison.png`: raw spectrum, both transmissions, and both
  corrected residual spectra.
- `../../uves_official_demo_comparison/`: official UVES tutorial comparison,
  metrics, fitted products, and source hashes.
- `../../keck_hires_bd17_o2_comparison/`: non-ESO Keck/HIRES O2 B/A-band
  comparison, corrected products, metrics, plot, source hashes, and blocked
  out-of-fold products under `cross_validation/`.
- `../../keck_kpf_vega_o2_comparison/`: independently reduced Keck/KPF Vega
  O2 B/A-band comparison, corrected products, metrics, plot, source hashes,
  model-systematics products, and blocked out-of-fold products.
- `independent_review/`: nine anonymized blind cases, instructions, and review
  forms. Send `independent_review_packet.zip`, whose manifest is checked by the
  campaign. Keep `independent_review_answer_key.csv` from the reviewer until
  the assessment is signed.
- `authenticated_hitran_receipt.json`: optional; created only after a forced
  real API v2 request and contains hashes/request metadata but no credential or
  line data.
- `../../../dist/pymolfit-0.1.0-py3-none-any.whl`: tested wheel; SHA-256 is
  `fdfc448f447ce8c17838a8dcc5cbe0fc6baa5b0071cedb07d8ffb27cb8c45ff5` and is
  also recorded in the current campaign report and release manifest.
