# PyMolFit tutorials

These notebooks introduce the normal PyMolFit workflows and the complete
expert interface:

1. `01_partial_spectrum.ipynb` corrects a narrow Na D spectrum using only the
   input wavelength information and an astrophysical exclusion mask.
2. `02_full_spectrum.ipynb` corrects a broad HARPS spectrum using automatic
   segmentation and explicitly chosen fit and exclusion ranges.
3. `03_expert_mode.ipynb` shows every file-input expert control accepted by
   `correct` and explains when each one is useful.
4. `04_array_input.ipynb` constructs an `Observation` and corrects wavelength
   and flux arrays through the same `correct` API.
5. `05_telluric_region_selector.ipynb` uses the interactive AER-marked
   selector to propose, edit, save, reuse, and apply fit/exclusion regions.

Install the plotting support before running them:

```bash
python -m pip install "pymolfit[interactive]"
```

Use the same Python environment for the notebook kernel and the installation.
The first correction may download and verify the managed AER catalogue.

The notebooks intentionally leave automatic atmosphere selection, molecular
line selection, continuum fitting, instrumental broadening, segmentation, and
wavelength alignment at their package defaults. The effective choices are
printed by `correct` after each fit.
