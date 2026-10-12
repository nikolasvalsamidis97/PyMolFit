# `TheoreticalSpectrum`

Use a rest-frame stellar template to exclude astrophysical features from the telluric fit. It does **not** replace the observed flux or directly fit the stellar spectrum during atmospheric optimisation.

```python
from pymolfit import TheoreticalSpectrum, correct

template = TheoreticalSpectrum(
    path="stellar_model.txt",
    radial_velocity_kms=20.0,
    vsini_kms=130.0,
    wavelength_unit="angstrom",
    wavelength_medium="air",
)
result = correct("spectrum.fits", theoretical_spectrum=template)
```

The velocities above illustrate the Beta Pictoris example; supply appropriate values for your target.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `path` | required | Two-column ASCII stellar spectrum; SVO model headers are supported. |
| `radial_velocity_kms` | required | Stellar barycentric radial velocity in km/s. Positive shifts the rest-frame template redward. |
| `vsini_kms` | required | Projected stellar rotation in km/s; 0 disables rotational broadening. |
| `wavelength_unit` | `"auto"` | Read recognised header units, otherwise require an explicit unit. |
| `wavelength_medium` | `"auto"` | Read recognised template convention, otherwise declare `"air"` or `"vacuum"`. SVO BT-family ASCII templates are recognised as air. |
| `resolving_power` | `None` | Initial instrumental resolution for template broadening; use observation metadata when available. This is not the subsequently fitted atmospheric LSF. |
| `limb_darkening` | `0.6` | Linear limb-darkening coefficient in the rotation kernel, between 0 and 1. |
| `mask_depth` | `"auto"` | Fractional stellar-feature threshold. Auto uses at least 0.05 and adjusts for noise; larger thresholds exclude fewer features. |
| `continuum_window_kms` | `3000.0` | Window for template pseudo-continuum estimation; choose wider than stellar features. |
| `mask_padding_kms` | `"auto"` | Extend detected exclusions to cover wings/alignment uncertainty. Auto uses half the quadrature sum of rotation and instrumental FWHM, with a 10 km/s floor. |
| `fit_velocity_offset` | `True` | Refine a small residual template alignment by cross-correlation. |
| `velocity_search_kms` | `20.0` | Symmetric search half-width for that residual alignment. |
| `confidence_weighted_masking` | `False` | Experimental: downweight predicted stellar pixels instead of binary template exclusions in `correct()`. |
| `confidence_weight_floor` | `0.05` | Minimum residual weight in that experimental mode. |
| `macroturbulence_kms` | `0.0` | Additional Gaussian stellar macroturbulent FWHM in km/s. |
| `wavelength_col` | `0` | Zero-based input wavelength column. |
| `flux_col` | `1` | Zero-based input flux column. |

PyMolFit normalises and broadens the template, aligns its wavelength medium/frame, then builds exclusions. They remain fixed during atmospheric fitting unless regenerated. Check their placement in the [selector](../user_guide/interactive_selector.md); a poor template can exclude useful telluric information or miss stellar features.
