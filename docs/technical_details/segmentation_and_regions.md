# Segmentation and regions

Implementation: [`workflow.py`](../../src/pymolfit/workflow.py), [`regions.py`](../../src/pymolfit/regions.py), [`theoretical.py`](../../src/pymolfit/theoretical.py), [`_automatic.py`](../../src/pymolfit/_automatic.py).

## Coordinate preparation

FITS units, air/vacuum medium and frame metadata are read before fitting. Arrays carry these on `Spectrum` and `Observation`. Internally the atmospheric model uses observatory-frame vacuum wavelengths. Standard-air conversion uses the implementation's Edlen relation; metadata-derived velocity corrections move between reference frames. These operations change corresponding pixel coordinates, not interpolate observed flux onto another grid. The high-level result uses these transformed coordinates; read its units, medium and frame metadata rather than assuming the original convention.

User regions are converted from their declared units, medium and frame to the same internal coordinates. A barycentric velocity correction is not the stellar radial velocity. Guessing one from the other can place otherwise valid telluric markers on the wrong pixels.

## Physical groups and processing segments

Physical groups follow supplied order/detector identifiers or large wavelength gaps. Each is divided into numerical segments with a default target span of 0.005 micron (50 Angstrom). Further subdivision limits a high-resolution segment to two million grid points. This reduces memory consumption without coarsening the selected atmospheric grid.

Local continua belong to processing segments; group-scoped wavelength polynomials belong to physical groups. Molecular scales are shared across the fitted segments in a joint correction. Splitting a physical order therefore does not automatically create independent column-density estimates.

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
| `correct()` with no supplied regions | Distributed wavelength/species coverage, coverage-dependent line count and windows based on sampling/initial instrumental width. Not a universal 100-line selection. |

Lines in gaps are skipped; overlapping windows are merged. Existing region files take precedence. Selector generation does not save automatically: use `.save()` or the editor's Save All action.

## Stellar exclusions

The template workflow estimates a pseudo-continuum, normalises the stellar spectrum, and applies stellar rotation, optional macroturbulence and an initial instrumental resolution. The latter comes from a supplied resolving power or observation metadata, not from a future atmospheric fit.

Template wavelengths are converted to the observation's medium and frame using stellar radial velocity and observation-frame correction. Optional cross-correlation refines a residual velocity offset. Pixels beyond the configured continuum-deviation threshold form exclusions, expanded by velocity padding. The automatic threshold is at least 5 percent; automatic padding has a 10 km/s floor and depends on stellar/instrumental width.

The mask stays fixed during atmospheric optimisation. It is not an iteratively fitted stellar-flux component. The optional confidence-weighted mode replaces hard template exclusions with residual weights, but explicit exclusions remain exclusions. Template mismatch, missing blends or a wrong radial velocity can still bias the atmospheric fit.
