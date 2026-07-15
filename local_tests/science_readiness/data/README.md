# Validation data provenance

The X-shooter files in `xshooter_hd53123` are public ESO Phase 3
flux-calibrated spectra of the telluric standard HD 53123 from programme
`60.A-9022(C)`. They were downloaded on 2026-07-13 through the ESO Data Portal.

| File | Arm/range | Resolving power | ESO dataset ID |
|---|---|---:|---|
| `ADP.2026-03-26T15_48_39.243.fits` | VIS, 533.66-1020.00 nm | 8935 | `ADP.2026-03-26T15:48:39.243` |
| `ADP.2026-03-26T15_48_39.204.fits` | NIR, 994.02-2101.26 nm | 5573 | `ADP.2026-03-26T15:48:39.204` |

The `uves_demo` directory contains the public Phase 3 UVES spectrum of
HD 127750 used in section 8.3.2 of the ESO Molecfit Reflex tutorial 4.4.2:

| File | Range | Resolving power | ESO dataset ID |
|---|---|---:|---|
| `ADP.2020-06-08T15_07_14.471.fits` | 472.64-683.50 nm | 74450 | `ADP.2020-06-08T15:07:14.471` |

The held-out comparison uses the tutorial's published fit intervals and the
UVES static H2O/O2 molecule selection. It does not select intervals or initial
molecular columns from GenMolFit residuals.

The `keck_hires_bd17` directory adds an independent, non-ESO instrument. It
contains two extracted orders from the public Keck/HIRES exposure of the
spectrophotometric standard BD+17 3248:

| KOAID | Orders | Range | Resolving power | Observation time |
|---|---|---|---:|---|
| `HI.20040824.18925.fits` | red CCD orders 4 and 9 | O2 B and A bands | 102700 | 2004-08-24 05:15:25.65 UTC |

The files were retrieved from the KOA level-1 API. Their SHA-256 hashes are:

```text
a4f998016c7a048b20f4fa53c7ddebd1431d663b274309edc6cec6550413111d  HI.20040824.18925_3_04_flux.fits.gz
1376baf2b311fb33ec392a808a475f663e743134f58919615f0c1874ddb116e6  HI.20040824.18925_3_09_flux.fits.gz
```

The comparison intervals are the established O2 B and A molecular bands, not
regions selected from GenMolFit residuals. MAKEE records a heliocentric vacuum
wavelength scale; the validation first applies the documented `HELIOVEL`
correction to recover topocentric vacuum wavelengths. Both programs receive
the same pixels and site weather from the source header.

KOA explicitly describes automated HIRES extractions as browse products, not
publication-quality reductions. This case therefore validates instrument and
format portability plus telluric-model behavior; it does not validate the
upstream extraction. Source documentation and acknowledgement text:

```text
https://koa.ipac.caltech.edu/UserGuide/HIRES/extracted_products.html
https://koa.ipac.caltech.edu/UserGuide/HIRES/extracted.html
```

The `kpf_vega` directory contains an independently reduced, science-ready KPF
level-1 observation of Vega (HR 7001):

| KOAID | Product | O2 order indices | DRP version | Observation time |
|---|---|---|---|---|
| `KP.20250519.55029.51.fits` | `KP.20250519.55029.51_L1.fits` | 13, 21, 22 in `RED_SCI_*1` | `v2.8.2` | 2025-05-19 15:17:09.4 UTC |

The product SHA-256 is:

```text
0b7d440c1a765eb7aab030bc84c3e36f72421903fbbb072d93efed284ff16896  KP.20250519.55029.51_L1.fits
```

The fixed intervals `0.6868--0.6905`, `0.7592--0.7625`, and
`0.7630--0.76515` micron follow the established O2 bands and KPF order
boundaries, not GenMolFit residuals. KPF L1 wavelengths are vacuum. The SCI1
slice is used without combining it with the two slightly different wavelength
solutions. The validation resolves the header-only `KECK` site to Keck
coordinates, downloads the bracketing time-local GDAS profiles, and gives the
identical interpolated source profile to both solvers.

Source documentation and archive endpoint:

```text
https://koa.ipac.caltech.edu/UserGuide/KPF/reduced_data.html
https://kpf-pipeline.readthedocs.io/en/stable/info/data_faq.html
https://koa.ipac.caltech.edu/cgi-bin/KoaAPI/nph-dnloadL1data
```

Direct archive URLs follow this pattern:

```text
https://dataportal.eso.org/dataPortal/file/<ESO dataset ID>
```

The `WAVE` arrays are standard-air wavelengths in nm (`TUCD1=em.wl;obs.atmos`).
The validation runner uses `FLUX_REDUCED`, `ERR_REDUCED`, and pixels with
`QUAL == 0`, preserving the primary headers needed for time-local GDAS.

The large AER line database is not copied here. The validation runner resolves
the same official AER 3.9 catalogue as the production workflow from:

```text
https://doi.org/10.5281/zenodo.18881607
```

The archive and extracted catalogue are verified against pinned SHA-256 hashes.
The campaign cache contains only the disjoint validation windows and seven
exercised molecules. The UVES, HIRES, and KPF comparisons request their own
AER 3.9 wavelength windows and record catalogue and selected-table hashes in
their retained products.
