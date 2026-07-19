# Tutorial data

The tutorial notebooks use reduced one-dimensional spectra already present in
the local PyMolFit validation campaign. The copies here make the notebooks
portable and prevent them from depending on a developer-specific absolute
path.

| File | Target/instrument | Wavelength convention | Purpose |
|---|---|---|---|
| `ADP.2017-04-07T01_04_41.632.fits` | Beta Pictoris / HARPS | standard air, barycentric product | broad-spectrum segmentation |
| `harps_nad_crop_air.fits` | Beta Pictoris / HARPS | standard air | Na D and nearby H2O fit |
| `harps_o2_bband_crop_air.fits` | Beta Pictoris / HARPS | standard air | O2 B-band diagnostics |
| `hires_bd17_o2_topocentric_vacuum.fits` | BD+17 3248 / Keck HIRES | topocentric vacuum | metadata and atmosphere portability |

The HARPS source is the public ESO Phase 3 product
`ADP.2017-04-07T01:04:41.632`. The compact HARPS files are deterministic
wavelength crops made from that product; they do not alter the flux values.

The HIRES file combines two public KOA extracted orders from exposure
`HI.20040824.18925.fits`, covering the established O2 B and A bands. KOA
describes automated HIRES extractions as browse products, so this example
demonstrates format and atmosphere portability rather than validating the
upstream extraction.

The large AER database is not stored here. PyMolFit resolves the official,
versioned AER 3.9 catalogue through its managed cache and records the selected
line-table provenance in each fit product.

SHA-256 checksums for the tutorial copies:

```text
03e4eb6d3736bb0eae689aefc4e883ee157c2ebd2e7847c5e9ce2e1600ddfbb0  ADP.2017-04-07T01_04_41.632.fits
d76bac756b6d77313be39b6f295867eda6e33825e0b32ada01c915587166c415  harps_nad_crop_air.fits
e481f4fa839d8d396de38971822f5c0f20bf1f1245a02bc3b7032556e6a8f63e  harps_o2_bband_crop_air.fits
209631cb3df83b35163e215f785a5af6640a6e9cb26c83970969e6b56810f430  hires_bd17_o2_topocentric_vacuum.fits
```
