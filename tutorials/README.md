# PyMolFit tutorials

This directory contains five independent notebooks for learning and assessing
the public PyMolFit workflow.

Run the notebooks in order:

1. `01_quickstart.ipynb` - one scientifically constrained correction
2. `02_nad_detailed_fit.ipynb` - build and diagnose a Na D-region fit
3. `03_full_echelle_spectrum.ipynb` - automatic segmentation of a broad 1D spectrum
4. `04_atmosphere_and_metadata.ipynb` - FITS metadata, MIPAS/GDAS, and explicit fallback
5. `05_diagnostics_and_troubleshooting.ipynb` - recognize and investigate common failures

## Environment

Create a clean environment and install the plotting extras before opening the
notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "pymolfit[plot]" jupyter
```

For a source checkout, replace the PyPI installation with:

```bash
python -m pip install -e ".[plot]"
```

The first scientific fit may download and verify the official AER 3.9 line
catalogue. The archive is cached outside this directory and is not duplicated
by every notebook.

All generated tables and plots are written under `outputs/`. The input files
and their provenance are described in `data/README.md`.

## Scope

These notebooks teach how to operate and assess PyMolFit. A successful
optimizer is not by itself proof of a scientifically valid correction. The
tutorials therefore use predetermined molecular bands, mask astrophysical
features, expose bound flags, and distinguish fit diagnostics from external
validation.
