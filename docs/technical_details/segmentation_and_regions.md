# Segmentation and regions

Implementation: [`workflow.py`](../../src/pymolfit/workflow.py), [`regions.py`](../../src/pymolfit/regions.py), [`theoretical.py`](../../src/pymolfit/theoretical.py), [`_automatic.py`](../../src/pymolfit/_automatic.py).

## Coordinate preparation

FITS units, air/vacuum medium and frame metadata are read before fitting. Arrays carry these on `Spectrum` and `Observation`. Internally the atmospheric model uses observatory-frame vacuum wavelengths. Standard-air conversion uses the implementation's Edlen relation; metadata-derived velocity corrections move between reference frames. These operations change corresponding pixel coordinates, not interpolate observed flux onto another grid. The high-level result uses these transformed coordinates; read its units, medium and frame metadata rather than assuming the original convention.

User regions are converted from their declared units, medium and frame to the same internal coordinates. A barycentric velocity correction is not the stellar radial velocity. Guessing one from the other can place otherwise valid telluric markers on the wrong pixels.

## Physical groups and processing segments

Physical groups follow supplied order/detector identifiers or large wavelength gaps. Each is divided into numerical segments with a default target span of 0.01 micron (100 Angstrom). Further subdivision limits a high-resolution segment to two million grid points. This reduces memory consumption without coarsening the selected atmospheric grid.

Local continua belong to processing segments; group-scoped wavelength polynomials belong to physical groups. Molecular scales are shared across the fitted segments in a joint correction. Splitting a physical order therefore does not automatically create independent column-density estimates.

An explicitly labelled physical group with too few samples for the requested
continuum raises an error identifying that group and the required count. Its
samples are not silently discarded or merged into another labelled order.
Choose a supported model for it or explicitly remove that group from the input
before correcting the remaining orders.

## Which pixels constrain the fit

Let $V_{kp}$ be validity, $I_{kp}$ membership in the union of fitting intervals and $E_{kp}$ membership in any exclusion. Then

$$\mathcal F_k=\{p: V_{kp}\land I_{kp}\land\neg E_{kp}\}.$$

Validity requires finite wavelength and flux, positive finite supplied uncertainty, and a True input mask where present. Exclusions override inclusions. No explicit fitting intervals means the high-level workflow normally attempts automatic selection; if disabled or no usable windows exist, all valid, non-excluded pixels are used, with a warning on failed automatic selection.

These masks determine parameter estimation only. The final atmospheric model is evaluated outside fitting intervals and in excluded regions too.

## Automatic region selection

Transitions are ranked using

$$A_j=S_j(T_{\rm ref})N_{s(j),\rm rep},$$

where $N_{s(j),\rm rep}$ is the species column in a fixed representative atmosphere, not the fitted column for this observation. This is a ranking score, not predicted detector line depth; saturation and instrumental broadening are not represented by that product alone.

There are two deliberately different selection policies:

| Entry point | Policy |
| --- | --- |
| `select_telluric_regions()` | Requested line count (default 100), fixed pixel half-width (default 12), strongest ranked transitions in covered wavelengths. |
| `correct()` with no supplied regions | Distributed wavelength/species coverage, coverage-dependent line count and windows based on sampling/initial instrumental width, followed by a segment-coverage safeguard. Not a universal 100-line selection. |

Lines in gaps are skipped; overlapping windows are merged. Existing region files take precedence. Selector generation does not save automatically: use `.save()` or the editor's Save All action.

After the actual processing segments are constructed, automatic catalogue windows are checked before pilot fits. For a segment with $N$ valid, non-excluded pixels and initial continuum degree $d$ (2 for `"auto"`), the required selected count is

$$N_{\rm required}=\min\left(N,\max\left(32,8(d+1),\lceil0.25N\rceil\right)\right).$$

If fewer pixels are selected, the workflow adds all valid, non-excluded samples in that segment to the fitting ranges. Existing windows remain; additions are split at invalid samples, exclusions and sampling gaps. Ranges retain their global wavelength scope, so an added interval can also include valid samples in an overlapping physical group. Segments with sufficient coverage receive no additions of their own.

`fit_region_selection.coverage` in result provenance records the threshold, affected segments, and selected counts before and after the final range union. This is a pixel-coverage safeguard: selected pixels can still be saturated, and it does not guarantee an identifiable or positive continuum. Missing-error estimation continues to use the original catalogue windows. Explicit fitting ranges, region files, and `automatic_fit_regions=False` bypass this expansion.

## Automatic wavelength selection

Automatic alignment first compares no correction, a constant pixel offset, and
supported linear/quadratic pixel trends on distributed telluric pilots. The
existing fixed-error complexity penalty remains the initial selection. It does
not estimate a smaller noise scale from unusually small residuals.

A rejected model can be reconsidered when it reduces pilot weighted residual
sum of squares by at least 5 percent and improves at least half the informative
pilots. With omitted bounds, a promising candidate at the initial ±3-pixel
coefficient limit can also be tried at ±12 pixels. This applies to polynomial
trends as well as constant offsets. Explicit bounds are never widened. These
limits apply to each coefficient, not to the total polynomial displacement.

Reconsideration uses three deterministic folds of complete absorption features
on the actual processing segments, including grid-limit subdivisions. The same
withheld pixels are scored for every candidate. Surrounding guards cover at
least one instrumental FWHM and the allowed wavelength displacement; guards
are excluded from training but unscored. Each fold refits continua, molecular
scales and free instrumental parameters from training data. It never borrows
full-pilot fitted parameters for fold initialization. Input flux and errors are
unchanged. When continuum degree is automatic, each fold also selects and refits
that degree using only its training pixels, matching the normal correction's
refinement; explicit degrees stay fixed. Saturated cores are not scored; resolved deep features require
usable moderate-transmission wings on both sides.

An override needs at least 5 percent lower pooled withheld RMS, improvements in
at least two folds, and no aggregate or majority-fold median-residual tradeoff.
Failed fits, insufficient blocks, nonfinite predictions, nonpositive continua,
or wavelength coefficients still at their bounds retain the last supported selection.
Because the full-spectrum reference model locates the features, this is
**conditional blocked model validation**, not independent external validation or
a statistical significance claim. It adds fitting work only for promising
rejected candidates and does not guarantee that every order improves.

Group-scoped selection preserves each physical order's selected coefficient
bounds, even when another order needs wider bounds. Provenance under
`wavelength_alignment` records pilot comparisons, attempted expansions, block
locations, guard widths, fold scores and acceptance/fallback reasons.

## Stellar exclusions

The template workflow estimates a pseudo-continuum, normalises the stellar spectrum, and applies stellar rotation, optional macroturbulence and an initial instrumental resolution. The latter comes from a supplied resolving power or observation metadata, not from a future atmospheric fit.

Template wavelengths are converted to the observation's medium and frame using stellar radial velocity and observation-frame correction. Optional cross-correlation refines a residual velocity offset. Pixels beyond the configured continuum-deviation threshold form exclusions, expanded by velocity padding. The automatic threshold is at least 5 percent; automatic padding has a 10 km/s floor and depends on stellar/instrumental width.

The mask stays fixed during atmospheric optimisation. It is not an iteratively fitted stellar-flux component. The optional confidence-weighted mode replaces hard template exclusions with residual weights, but explicit exclusions remain exclusions. Template mismatch, missing blends or a wrong radial velocity can still bias the atmospheric fit.
