# Example spectra

| File | Contents |
| --- | --- |
| `harps_nad_crop_air.fits` | Beta Pictoris HARPS crop, 588.2--590.7 nm, barycentric standard-air wavelengths. Flux and sampling are unchanged. |
| `harps_oxygen_crop_air.fits` | Optional 627.001--630.999 nm crop of the same HARPS observation. Original wavelength, flux and error samples are unchanged; only coverage metadata is updated. The speed-comparison notebook uses the full FITS file instead. |
| `ADP.2017-04-07T01_04_41.632.fits` | Original full public ESO Phase 3 product `ADP.2017-04-07T01:04:41.632`, observed 6 April 2017. |
| `beta_pic_nad.txt` | The crop's wavelength (nm) and flux (ADU), exported without normalisation or interpolation. |
| `beta_pic_template.txt` | 5600--6200 Angstrom rest-frame air subset of the user's SVO BT-NextGen (AGSS2009) model: 8000 K, log g = 4, metallicity = 0, alpha = 0. |

The template is an example for stellar exclusion generation, not an independently validated description of Beta Pictoris. It retains a broad margin around Na D for continuum estimation and broadening. Obtain a wider template before applying it to the full HARPS spectrum. Model data are attributed to the original BT-NextGen/PHOENIX authors and the [SVO Theoretical Spectra service](https://svo2.cab.inta-csic.es/theory/newov2/); PyMolFit's software licence does not relicense third-party data.

The full observation and crop were moved unchanged from the former `tutorials/data/` folder. For reproducing the array example, the original header provides UTC time `2017-04-06T00:05:02.828`, La Silla latitude -29.2584 deg, longitude -70.7345 deg, altitude 2400 m, start/end airmasses 1.231/1.236, resolving power 115000 and BERV -7.52937637262916 km/s.

SHA-256:

```text
d76bac756b6d77313be39b6f295867eda6e33825e0b32ada01c915587166c415  harps_nad_crop_air.fits
659980ec200530f47ff2128edc1e84aaeb01e15e0aaced9b8ece6ad19b780555  harps_oxygen_crop_air.fits
03e4eb6d3736bb0eae689aefc4e883ee157c2ebd2e7847c5e9ce2e1600ddfbb0  ADP.2017-04-07T01_04_41.632.fits
08d3fe876bb530a8fa4da1acf01f13b9a1db65d4fe2dc6422d42d362667db760  beta_pic_nad.txt
e81e0f14aa6a879bb4507745c82ea84694aa9a07d5046f4e082cf8cf93910f47  beta_pic_template.txt
```

Original full template SHA-256: `49f63a3a761a46bb72f317c2eb52787d8d14f32286b8edc6345e53624b51d549`. The subset preserves the numerical samples at the precision of that source.
