# PyMolFit

PyMolFit is a pure-Python generic telluric correction package inspired by ESO
Molecfit. The goal is to fit and remove telluric absorption from spectra
without requiring the user to know about ESO recipe files, EsoRex-style
workflow files, shell paths, or compiled C/Fortran executables.

The current release is a science-ready candidate pending independent blind
review; it is not claimed to be a bit-identical drop-in replacement for
Molecfit/LBLRTM. Automated validation covers synthetic recovery, uncertainty
coverage, failure behavior, and real HARPS, UVES, X-shooter, and CRIRES+
spectra, plus independent Keck/HIRES and Keck/KPF spectra.
The current version implements generic spectrum input/output, a vectorized
molecular-line absorption model, a self-contained HITRAN `.par` reader,
HITRAN/TIPS partition-sum tables, HITRAN isotopologue metadata and abundance
scaling, source-backed LBLRTM H2O, CO2, N2, and O2 continua, optional external
HITRAN CIA tables, LBLRTM `contnm.f90` Rayleigh scattering, layer-by-layer
radiative transfer, Molecfit-inspired line-window and LSF controls, fitting,
correction, diagnostics, and comparison tools. Scientific workflows use the
official versioned AER 3.9 catalogue. PyMolFit downloads it from its immutable
Zenodo release, verifies it, and caches it automatically;
the small built-in synthetic list is retained only for tests and examples.

## Why This Shape

Molecfit is built from several responsibilities:

- read spectra and wavelength/fit ranges
- create an atmospheric transmission model from molecular line data and an
  atmospheric profile
- convolve the model with the instrumental profile
- fit continuum, wavelength/resolution adjustments, and molecule abundances
- divide the observed spectrum by the fitted telluric transmission

PyMolFit keeps the same conceptual split, but implements the first usable
layer in Python:

- `Spectrum`: generic 1D spectrum container
- `LineList`: molecular line centers, strengths, widths, and species labels
- `transmission_model`: vectorized optical-depth and transmission model
- `AbsorptionComponent`: plugin-style interface for physical opacity sources
- `HitranLineAbsorption`: HITRAN line-by-line component
- `IsotopologueMetadata`: HITRAN global/local isotopologue IDs, abundances,
  masses, and q-file names
- `PartitionTable`: tabulated HITRAN/TIPS partition sums
- `physical_transmission_model`: self-contained HITRAN line-by-line model
- `MTCKDH2OContinuum`: vectorized MT_CKD H2O self/foreign continuum model
- `H2OContinuumAbsorption`: MT_CKD H2O continuum component
- `CO2ContinuumAbsorption`: CO2 continuum component from external tables
- `O2CIAAbsorption` / `N2CIAAbsorption`: HITRAN CIA-file components
- `RayleighScatteringAbsorption`: LBLRTM `contnm.f90` Rayleigh component
- `N2RototranslationalContinuumAbsorption`: LBLRTM N2 pure-rotation continuum
- `fit_tellurics`: nonlinear least-squares fit of molecule scales and continuum
- `correct_spectrum`: divides flux by the fitted transmission

## Install For Development

```bash
python -m pip install -e ".[dev]"
```

For diagnostic plots:

```bash
python -m pip install -e ".[dev,plot]"
```

For a normal release install, the intended interface is simply:

```bash
python -m pip install pymolfit
pymolfit fit spectrum.fits corrected.ecsv
```

The wheel does not contain the 759 MiB uncompressed AER catalogue. On first
scientific use PyMolFit downloads the official AER 3.9 archive from
[Zenodo record 18881607](https://doi.org/10.5281/zenodo.18881607), extracts its
line catalogue, broadening tables, and line-coupling table, verifies fixed
SHA-256 hashes, and stores them under `~/.cache/pymolfit/aer`. No HITRAN
account or API key is needed.
The data can be installed ahead of time and inspected explicitly:

```bash
pymolfit install-aer
pymolfit aer-status
```

Set `PYMOLFIT_AER_URL` only to use another location containing the exact
`aer_v_3.9.tgz` artifact. The extracted scientific payload must still match
the hashes pinned from the official Zenodo release.

## Python Example

```python
from pymolfit import LineList, Spectrum, fit_tellurics

spectrum = Spectrum(wavelength=wavelength_micron, flux=flux)
result = fit_tellurics(spectrum, line_list=LineList.demo_near_ir())

corrected = result.corrected
```

For normal scripting, the higher-level workflow is shorter:

```python
from pymolfit import correct_file

result = correct_file(
    "spectrum.fits",
    "corrected.txt",
    wavelength_unit="nm",
    hitran_species=("H2O", "CO2", "CH4"),
    atmosphere_mode="mipas_gdas",
    # alternatively: atmosphere_table="profile.ecsv",
    pwv_mm=2.0,
    solve_continuum_linear=True,
    fit_wavelength_polynomial=True,
    wavelength_polynomial_order=1,
    fit_lsf_sigma=True,
    estimate_uncertainties=True,
    product_path="fit_product.ecsv",
)
```

With no `line_list`, `line_list_path`, or `hitran_par`, this workflow selects
the required wavelength window from the managed AER catalogue and caches the
result as a provenance-bearing ECSV table. Pass `aer_catalog=None` (Python) or
`--no-auto-aer` (CLI) only when intentionally running a continuum-only model
or supplying another opacity source.

`solve_continuum_linear=True` removes the linear continuum coefficients from
the nonlinear optimizer and is recommended for spectra with large absolute
flux values. With `estimate_uncertainties=True`, the result includes a
linearized parameter covariance, molecular-scale standard errors,
transmission uncertainty, and corrected-flux uncertainty. These uncertainties
are conditional on the supplied atmosphere, line data, masks, continuum
family, and LSF model. Use `fit_tellurics_with_systematics` to refit named
alternative model configurations and add their transmission spread to the
corrected-flux uncertainty; no universal atmosphere/LSF ensemble is assumed.
Shared multi-segment fits propagate the shared covariance into every corrected
segment. Rank-deficient fits report `NaN` covariance/output uncertainty rather
than false precision. Full product tables retain input, fit-pixel, and
corrected masks; parameter-bound flags; and machine-readable provenance. The
provenance hashes the modeled spectrum, source file, full and selected line
lists, atmospheric layers, component data, and fit configuration, and records
the numerical library versions. Wavelength correction can be a constant shift, one polynomial
shared over the full wavelength span, or independent per-segment polynomials
in the lower-level multi-segment API. Covariance rank is evaluated after
Jacobian column scaling, so the result is invariant to parameter units.

## Science-readiness status

PyMolFit has an automated cross-band, cross-instrument validation campaign,
but it does not currently pass its full scientific acceptance gate. See
[`docs/science_readiness_validation.md`](docs/science_readiness_validation.md)
for the checklist, public-data provenance, numerical thresholds, current
caveats, and reproducible evidence files. Do not describe version `0.1.0` as a
validated drop-in replacement for Molecfit.

## Physical HITRAN Example

This path does not call Molecfit, LBLRTM, LNFL, or HAPI at model-evaluation
time. It parses a HITRAN `.par` file and calculates Voigt line absorption in Python. If you provide
HITRAN isotopologue metadata and TIPS q files, line intensities use tabulated
partition sums instead of the fallback power-law approximation.

```python
from pymolfit import (
    AtmosphereProfile,
    FitConfig,
    IsotopologueMetadata,
    LineList,
    PartitionTable,
    Spectrum,
    fit_tellurics,
)

line_list = LineList.from_hitran_par(
    "hitran_h2o_lband.par",
    wavenumber_min=4200,
    wavenumber_max=4400,
)
isotopologues = IsotopologueMetadata.from_hitran_iso_meta_html("hitran_iso_meta.html")
line_list = line_list.with_isotopologue_metadata(isotopologues)
partition = PartitionTable.from_hitran_q_directory("hitran_q", isotopologues)

atmosphere = AtmosphereProfile.single_layer(
    pressure_atm=0.75,
    temperature_k=280,
    mixing_ratios={"H2O": 0.001, "CO2": 4.2e-4, "CH4": 1.9e-6},
)

spectrum = Spectrum(wavelength=wavelength_micron, flux=flux)
result = fit_tellurics(
    spectrum,
    line_list=line_list,
    config=FitConfig(
        atmosphere=atmosphere,
        airmass=1.2,
        partition_table=partition,
        continuum_order=2,
    ),
)
```

The same physical path is available through `correct_arrays` and `correct_file`.
See `examples/physical_hitran_demo.py` for a fully self-contained synthetic
HITRAN-style example.

A more realistic built-in atmosphere is also available:

```python
atmosphere = AtmosphereProfile.standard_midlatitude()
atmosphere = atmosphere.with_pwv_mm(2.0)
```

The default high-level physical workflow now uses a Molecfit-style
MIPAS+GDAS atmosphere:

```python
from pymolfit import AtmosphereProfile

atmosphere = AtmosphereProfile.from_mipas_gdas(
    observation_time="2022-01-02T05:17:35",
    observatory_altitude_m=2635.0,
    airmass=1.2,
)
```

For FITS inputs, `correct_file(..., atmosphere_mode="mipas_gdas")` reads common
FITS, ESO, and legacy Keck header keywords for observation time, airmass, site
altitude, longitude, latitude, pressure, temperature, and relative humidity.
When coordinates are absent, named Paranal, La Silla, and Keck observatories
are resolved through the packaged site registry; explicit FITS coordinates
always take precedence and the resolution source is retained in provenance.
An unknown site without complete coordinates now raises an error instead of
silently using Paranal. Supply `observatory_latitude_deg`,
`observatory_longitude_deg`, and `observatory_altitude_m`, or explicitly opt in
to the legacy fallback with `allow_default_observatory=True`. Array and plain
text inputs using MIPAS/GDAS require the same explicit geometry because they do
not carry a FITS site header.
Known barycentric and heliocentric wavelength products are moved back to the
observatory frame before telluric fitting, with the applied velocity and frame
retained in the product provenance. By default
`gdas_mode="auto"` tries to download the exact ESO GDAS tarball for the
rounded observatory coordinates, interpolates the bracketing 3-hour profiles to
the observation time, and caches both the tarball and the interpolated FITS
profile under `~/.cache/pymolfit/gdas`. If the exact profile is unavailable,
PyMolFit falls back to the same bundled six two-month average GDAS profiles
that Molecfit ships, merged with the packaged MIPAS climatology.

GDAS source control:

```python
result = correct_file(
    "spectrum.fits",
    "corrected.txt",
    hitran_par="hitran_lband.par",
    atmosphere_mode="mipas_gdas",
    gdas_mode="auto",       # exact download/cache, then monthly average fallback
    # gdas_mode="online",   # require exact download/cache
    # gdas_mode="cache",    # require exact cached tarball/profile, no download
    # gdas_mode="average",  # force bundled monthly averages
)
```

`with_pwv_mm()` scales the vertical H2O column. For manual `fit_tellurics`
calls, keep the atmosphere vertical and pass the observation airmass through
`FitConfig`. For `correct_arrays`, `correct_file`, and the CLI, pass `airmass`
there; internally built atmospheres are slanted once before fitting. If you
deliberately need pre-slanted layer paths, `standard_midlatitude` also accepts
`airmass` and uses spherical-shell geometry for its layer path lengths.

## Model-systematic sensitivity

The optimizer covariance is conditional on one chosen physical and
instrumental model. To measure sensitivity to defensible alternatives, refit
the same pixels under named configurations:

```python
from dataclasses import replace
from pymolfit import FitConfig, fit_tellurics_with_systematics

baseline = FitConfig(
    atmosphere=atmosphere,
    species=("H2O", "O2"),
    lsf_sigma_pixels=1.2,
    estimate_uncertainties=True,
)
variants = {
    "warmer_profile": replace(
        baseline,
        atmosphere=atmosphere.perturbed(temperature_offset_k=2.0),
    ),
    "broader_lsf": replace(baseline, lsf_sigma_pixels=1.3),
}
systematics = fit_tellurics_with_systematics(
    spectrum,
    line_list,
    baseline,
    variants,
)
systematics.write("fit_with_systematics.ecsv")
```

The result stores every variant transmission, the RMS systematic uncertainty,
the full envelope, and a correction whose uncertainty combines the fitted
statistical term and ensemble spread in quadrature. Perturbation sizes must
come from site, atmospheric-product, or instrument-calibration uncertainties;
they are not fitted to make one observed spectrum resemble Molecfit.

## Blocked Cross-Validation

Use blocked out-of-fold prediction to check whether a fitted configuration
generalizes to pixels that were not used by the optimizer:

```python
from dataclasses import replace
from pymolfit import cross_validate_telluric_segments

cross_validation = cross_validate_telluric_segments(
    spectra,
    line_list=line_list,
    config=replace(config, estimate_uncertainties=False),
    block_size=64,
    n_folds=2,
)
cross_validation.write("cross_validation")
print(cross_validation.metrics)
```

The split uses alternating contiguous pixel blocks and does not inspect flux
residuals, fitted transmission, or a reference correction. Every valid pixel
is predicted once by a fit that did not consume that pixel. Reported metrics
separate complete prediction coverage from correction coverage lost to
saturated transmission, and compare held-out telluric residuals against the
continuum-only prediction.

## MT_CKD H2O Continuum

PyMolFit can add the LBLRTM/MT_CKD water-vapor continuum when you provide the
AER `absco-ref_wv-mt-ckd.nc` coefficient file. The implementation reads the
reference self and foreign continuum coefficients, applies density scaling,
self-continuum temperature scaling, the radiation term, and the same cubic
interpolation formula as the reference MT_CKD H2O routine.

Python:

```python
from pymolfit import correct_file

result = correct_file(
    "spectrum.fits",
    "corrected.txt",
    hitran_par="hitran_h2o.par",
    h2o_continuum="absco-ref_wv-mt-ckd.nc",
    atmosphere_mode="standard",
    pwv_mm=2.0,
)
```

Continuum-only modelling is also supported if you are testing broad H2O
continuum effects without a line list:

```python
result = correct_file(
    "spectrum.fits",
    "corrected.txt",
    h2o_continuum="absco-ref_wv-mt-ckd.nc",
    atmosphere_mode="standard",
    pwv_mm=2.0,
)
```

Command line:

```bash
pymolfit fit spectrum.txt corrected.txt \
  --hitran-par hitran_h2o.par \
  --mtckd-h2o absco-ref_wv-mt-ckd.nc \
  --atmosphere standard \
  --pwv-mm 2.0
```

Use `--mtckd-h2o-foreign-closure` to use the optional foreign-closure
coefficient column from MT_CKD 4.2+ files.

## TIPS And Line Wings

HITRAN line intensities are weighted by natural terrestrial isotopologue
abundances. PyMolFit keeps that as the default. To model a non-terrestrial
isotopic mixture, attach `IsotopologueMetadata` and pass absolute abundance
overrides; PyMolFit applies `override / natural_abundance` internally.

```python
line_list = line_list.with_isotopologue_metadata(
    isotopologues,
    abundance_overrides={(1, 1): 0.95, (1, 2): 0.05},
)
```

The default line shape is an untruncated Voigt profile. For an LBLRTM-style
calculation window, use `line_wing_mode="lblrtm_dynamic"`. This ports the
LBLRTM `AVRAT`/`ALFV`/`ALFMAX`/`HWF3` per-line window sizing from
`oprop_voigt.f90`; HITRAN line selection also expands automatically for the
largest dynamic window implied by the wavelength grid. For simpler fixed-window
experiments, use `line_cutoff_cm`. Neither mode makes the whole calculation
bit-identical to LBLRTM by itself, but both avoid letting isolated Voigt wings
contribute indefinitely.

For experiments that should follow LBLRTM's tabulated Voigt subfunction and
panel interpolation machinery more closely, use `line_wing_mode="lblrtm_panel"`.
This builds the 0-4, 0-16, and 0-64 effective-halfwidth subfunction tables from
the LBLRTM source formula using SciPy's Faddeeva function, then applies the
source `PANEL` four-to-one interpolation coefficients. It is an expert parity
option and not guaranteed to be faster or better on every spectrum. The older
`line_wing_mode="lblrtm_table"` mode keeps only the tabulated subfunction
decomposition without panel accumulation.

```python
config = FitConfig(
    atmosphere=atmosphere,
    partition_table=partition,
    line_wing_mode="lblrtm_dynamic",
)
```

The same controls are available on the CLI:

```bash
pymolfit fit spectrum.txt corrected.txt \
  --hitran-par hitran_lband.par \
  --partition-table tips_lband.ecsv \
  --line-wing-mode lblrtm_dynamic
```

For Molecfit-style instrumental broadening, PyMolFit supports the synthetic
box/Gaussian/Lorentzian kernel controls. Molecfit's optional `kern_mode`
Voigt-approximation path is exposed as `lsf_molecfit_voigt=True` or
`--lsf-molecfit-voigt`.

Physical high-level runs default to the audited LBLRTM path: packaged
LBLRTM-12.11 TIPS tables, packaged MT_CKD/LBLRTM H2O and CO2 continuum data,
the `lblrtm_panel` line accumulator, an oversampled internal grid,
Molecfit overlap rebinning before LSF convolution, separate Gaussian and
Lorentzian kernels, and the source kernel support of 3 FWHM. Use
`line_wing_mode="lblrtm_dynamic"` for a faster approximation, or explicitly
pass `h2o_continuum="none"` / `co2_continuum="none"` to disable continua.

## Rayleigh, N2, CO2, And CIA

PyMolFit also supports external continuum and collision-induced absorption
tables, plus source-backed `contnm.f90` branches:

- `rayleigh=True` / `--rayleigh` enables the LBLRTM Rayleigh formula for
  wavenumbers above 820 cm-1.
- `n2_continuum=True` / `--n2-continuum` enables the LBLRTM N2
  rototranslational, fundamental, and first-overtone continua.
- `o2_continuum=True` / `--o2-continuum` enables the LBLRTM O2 fundamental,
  1.27 micron, 9100--11000 cm-1, A-band, and visible continua.
- `co2_continuum=` reads an Astropy-readable table with `wavenumber_cm`,
  optional `temperature_k`, and `coefficient` columns.
- `o2_cia=` and `n2_cia=` read HITRAN `.cia` files. HITRAN CIA coefficients
  are in cm5 molecule-2 and are scaled by the two collision-partner number
  densities and path length in each atmosphere layer.

Python:

```python
result = correct_file(
    "spectrum.fits",
    "corrected.txt",
    hitran_par="hitran.par",
    h2o_continuum="absco-ref_wv-mt-ckd.nc",
    co2_continuum="co2_continuum.ecsv",
    rayleigh=True,
    n2_continuum=True,
    o2_continuum=True,
    atmosphere_mode="standard",
)
```

Command line:

```bash
pymolfit fit spectrum.txt corrected.txt \
  --hitran-par hitran.par \
  --mtckd-h2o absco-ref_wv-mt-ckd.nc \
  --co2-continuum co2_continuum.ecsv \
  --rayleigh \
  --n2-continuum \
  --o2-continuum \
  --atmosphere standard
```

The source-backed N2/O2 continua and overlapping HITRAN CIA tables are
alternative representations. PyMolFit rejects combinations such as
`n2_continuum=True` with an N2-N2 CIA table instead of silently double-counting
the same collision-induced absorption.

The CO2 continuum remains table-driven rather than hardcoded from LBLRTM block
data. This keeps the package lightweight and lets users update or swap
continuum datasets without changing PyMolFit code.

## Component Architecture

The physical backend is now built from absorption components. Each component
returns species names and optical-depth basis rows on the requested wavelength
grid. Components with the same species are summed before fitting, so H2O line
absorption and H2O continuum share one fitted H2O scale factor.

```python
from pymolfit import (
    AtmosphereProfile,
    CO2ContinuumAbsorption,
    FitConfig,
    H2OContinuumAbsorption,
    HitranLineAbsorption,
    LineList,
    MTCKDH2OContinuum,
    N2RototranslationalContinuumAbsorption,
    O2CIAAbsorption,
    Spectrum,
    TabulatedContinuum,
    fit_tellurics,
)

line_list = LineList.from_hitran_par("hitran_lband.par")
h2o_continuum = MTCKDH2OContinuum.from_netcdf("absco-ref_wv-mt-ckd.nc")
co2_continuum = TabulatedContinuum.from_table("co2_continuum.ecsv")
atmosphere = AtmosphereProfile.standard_midlatitude().with_pwv_mm(2.0)

components = (
    HitranLineAbsorption(line_list),
    H2OContinuumAbsorption(h2o_continuum),
    CO2ContinuumAbsorption(co2_continuum),
    N2RototranslationalContinuumAbsorption(),
    # O2CIAAbsorption(HitranCIATable.from_hitran_cia("O2-O2_2011.cia")),
)

result = fit_tellurics(
    Spectrum(wavelength=wavelength_micron, flux=flux),
    line_list=LineList.empty_hitran(),
    config=FitConfig(
        atmosphere=atmosphere,
        airmass=1.2,
        components=components,
        continuum_order=2,
    ),
)
```

The old high-level arguments, such as `hitran_par=` and `h2o_continuum=`, still
work. They now build this component set internally.

## Command Line Example

PyMolFit can load several common 1D spectrum formats:

- whitespace numeric text: wavelength, flux, optional uncertainty
- CSV numeric text
- ECSV or Astropy-readable tables with names like `wave`, `wavelength`,
  `flux`, `err`, or `uncertainty`
- FITS binary tables with inferred wavelength/flux/error columns
- gzip-compressed `.fits.gz`/`.fit.gz` tables and images
- 1D FITS image spectra with linear `CRVAL1`/`CDELT1` wavelength WCS

For a non-scientific format smoke test with the synthetic demo line list:

```bash
pymolfit fit spectrum.txt corrected.txt --demo-lines
```

For a FITS table or 1D FITS image, supply real line data:

```bash
pymolfit fit spectrum.fits corrected.txt \
  --wavelength-unit nm \
  --hitran-par hitran_window.par
```

PyMolFit deliberately refuses to run a line fit without `--hitran-par`,
`--line-list`, an explicit continuum-only component, or `--demo-lines`. The
packaged demo list is synthetic and must not be used for scientific spectra.

With a HITRAN `.par` file:

```bash
pymolfit fit spectrum.txt corrected.txt \
  --hitran-par hitran_h2o_lband.par \
  --mtckd-h2o absco-ref_wv-mt-ckd.nc \
  --hitran-species H2O \
  --hitran-min-strength 1e-28 \
  --airmass 1.2 \
  --atmosphere standard \
  --pwv-mm 2.0 \
  --partition-table partition_sums.ecsv \
  --mixing-ratio H2O=0.001
```

The optional partition table should be readable by Astropy and contain:
`mol_id`, `iso_id`, `temperature_k`, and `q`. If a molecule/isotopologue is
not present in the table, PyMolFit falls back to its approximate internal
partition scaling for that line.

Instead of the built-in atmosphere, a profile table can be supplied:

```bash
pymolfit fit spectrum.txt corrected.txt \
  --hitran-par hitran_lband.par \
  --atmosphere-table profile.ecsv
```

The atmosphere table should be Astropy-readable and contain pressure,
temperature, path length, and volume mixing ratio columns. Supported pressure
columns are `pressure_atm`, `pressure_hpa`, `pressure_mbar`, or `pressure_pa`.
Supported path columns are `path_length_m`, `thickness_m`, or `dz_m`. Mixing
ratio columns should be named like `mix_H2O`, `mix_CO2`, or `vmr_H2O`.

To cache a filtered HITRAN file as a reusable PyMolFit line-list table:

```bash
pymolfit convert-hitran hitran_lband.par h2o_lband.ecsv \
  --species H2O \
  --wavenumber-min 4200 \
  --wavenumber-max 4400 \
  --min-strength 1e-28 \
  --max-lines 50000
```

PyMolFit can optionally acquire a newer or custom wavelength window from
HITRAN's authenticated
API v2 without installing HAPI. Create an API key in your HITRAN account, keep
it in the environment, and request all molecules needed by the fit:

```bash
export HITRAN_API_KEY='your-private-key'
pymolfit fetch-hitran \
  --species H2O --species CO2 --species CH4 \
  --wavelength-min 2.29 --wavelength-max 2.36
```

The command prints paths to a standard `.par` file, a directly reusable ECSV
line table, and a JSON manifest. They are cached under
`~/.cache/pymolfit/hitran` (or `PYMOLFIT_HITRAN_CACHE`) and include the exact
query, database edition reported by the service, line coverage, isotopologue
metadata, and SHA-256 hashes. The API key is never persisted. A verified cache
hit works offline. Existing licensed files can be imported through the same
validation/provenance path:

```bash
pymolfit cache-hitran downloaded.par \
  --species O2 --wavelength-min 0.75 --wavelength-max 0.78
```

Use the printed ECSV path as `--line-list`. Direct HITRAN API data remain
subject to HITRAN's terms and citation policy. This optional route is separate
from the official AER 3.9 catalogue used by the default workflow.

For a release checkout, exercise the real authenticated path once with a
fixed, narrow O2 request and write a credential-free validation receipt:

```bash
export HITRAN_API_KEY='your-private-key'
PYTHONPATH=src python local_tests/validate_authenticated_hitran.py
unset HITRAN_API_KEY
```

This forces a network request rather than accepting a cache hit. The receipt
contains request, database-edition, source-code, and artifact hashes, but no
API key or downloaded line records. The science-readiness campaign accepts it
only while its client-source hash matches the current checkout.

With common expert controls:

```bash
pymolfit fit spectrum_nm.txt corrected.txt \
  --wavelength-unit nm \
  --hitran-par hitran_lband.par \
  --airmass 1.3 \
  --fit-wavelength-polynomial \
  --wavelength-polynomial-order 1 \
  --wavelength-shift-bounds -0.0003 0.0003 \
  --fit-lsf-sigma \
  --lsf-molecfit-voigt \
  --fit-range 2.318:2.322 \
  --fit-range 2.330:2.334 \
  --exclude-range 2.3205:2.3212 \
  --loss soft_l1 \
  --product fit_product.ecsv \
  --plot fit_diagnostic.png
```

The main output is the corrected spectrum. The optional product table contains
the observed flux, fitted model, continuum, transmission, corrected flux,
input/fit/corrected masks, propagated uncertainties, fitted parameters,
covariance rank, parameters at active bounds, and exact model/input provenance.

To compare a corrected PyMolFit spectrum with a reference product, such as a
Molecfit-corrected spectrum:

```bash
pymolfit compare corrected.txt reference_corrected.txt --normalize
```

## Remaining Release Gates

1. Complete the independent blind packet and a reviewer-selected held-out
   spectrum with an experienced telluric-correction user. Send only
   `local_tests/science_readiness/results/independent_review_packet.zip`; its
   manifest is checked automatically and the private answer key is excluded.
2. Resolve or scientifically explain the retained direct-transmission warnings
   across X-shooter and CRIRES+ without observation-specific calibration.

The production and validation workflows both use the checksum-pinned official
AER 3.9 catalogue from Zenodo. Authenticated HITRAN validation remains an
optional check of the direct-API feature, not a requirement for normal
scientific use.
