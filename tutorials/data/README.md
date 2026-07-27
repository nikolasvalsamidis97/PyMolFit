# Tutorial spectra

The tutorials use two public reduced one-dimensional HARPS spectra of Beta
Pictoris:

| File | Wavelength convention | Purpose |
|---|---|---|
| `harps_nad_crop_air.fits` | standard air | minimal narrow-spectrum correction |
| `ADP.2017-04-07T01_04_41.632.fits` | standard air, barycentric product | full-spectrum correction |

The source is the public ESO Phase 3 product
`ADP.2017-04-07T01:04:41.632`. The compact Na D file is a deterministic
wavelength crop that does not alter the source flux.

SHA-256 checksums:

```text
d76bac756b6d77313be39b6f295867eda6e33825e0b32ada01c915587166c415  harps_nad_crop_air.fits
03e4eb6d3736bb0eae689aefc4e883ee157c2ebd2e7847c5e9ce2e1600ddfbb0  ADP.2017-04-07T01_04_41.632.fits
```
