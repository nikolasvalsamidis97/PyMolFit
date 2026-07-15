# LBLRTM audit notes

This folder is for source-equivalence checks, not spectrum-specific tuning.
The tests and scripts here use synthetic line data so they cannot encode a
coefficient chosen to make one observed spectrum look better.

Source anchors used from the Molecfit-bundled LBLRTM tree:

- `~/.criresflow/kits/molecfit-kit-4.4.4-9/molecfit_third_party-1.9.5/lblrtm/src/oprop_voigt.f90`
- Voigt domain constants: `HWFF1=4`, `HWFF2=16`, `HWFF3=64`, `DXFF1=0.002`, `DXFF2=0.008`, `DXFF3=0.032`.
- Dynamic line-window rule: `ALFV = AVRAT(zeta) * (ALFL + ALFAD)`, clamped by `DV` and `ALFMAX = 4 * SAMPLE * DV * 0.04 / ALFAL0`; the active extent is `HWF3 * ALFV`.
- `CNVFNV` distance sampling: nearest F1/F2/F3 table point, with linear interpolation only along the zeta axis.
- `CONVF4`: the separately accumulated F4 closure/far-wing term on a `64 * DV` grid with the standard 25 cm-1 boundary.
- Source accumulation order: add all lines to R1/R2/R3/R4 first, then apply `XINT` and `PANEL` interpolation.
- Line bookkeeping: pressure shift, temperature/intensity scaling, Doppler width, Lorentz width, and optional line-coupling terms are audited in isolated tests before comparing full spectra.

Current audit scope:

- `tests/test_physics_parity.py`: low-level source formula checks.
- `tests/test_lblrtm_audit.py`: one-line end-to-end checks through `hitran_line_optical_depth_basis`.
- `local_tests/lblrtm_audit/run_single_line_audit.py`: reproducible CSV/plot output for one synthetic H2O line.

Deliberately not included:

- Any fitted scale calibrated from Rho01, Beta Pic, CRIRES, HARPS, or Molecfit output.
- Any hand-edited coefficient that only improves one spectrum.
