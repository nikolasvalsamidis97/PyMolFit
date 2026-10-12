# PyMolFit documentation

Start with the [user guide](user_guide/README.md). It contains four small, runnable notebooks using the bundled Beta Pictoris observation.

| Need | Read |
| --- | --- |
| Correct a file or a prepared spectrum | [`correct()` and advanced options](parameter_descriptions/correct.md) |
| Prepare wavelength and flux arrays | [`Spectrum`](parameter_descriptions/Spectrum.md) |
| Supply observing conditions and reference frame | [`Observation`](parameter_descriptions/Observation.md) |
| Exclude predicted stellar features | [`TheoreticalSpectrum`](parameter_descriptions/TheoreticalSpectrum.md) |
| Choose, edit and save fitting regions | [Interactive selector](user_guide/04_interactive_selector.ipynb) |
| Read and use a saved correction product | [Output handling](user_guide/output_handling.md) |
| Compare full-spectrum correction speed and residuals | [Faster correction notebook](troubleshooting/faster_correction.ipynb) |
| Make before/after and transmission figures | [Figure examples](figure_examples/README.md) |
| Follow the correction from start to finish | [Pipeline overview](technical_details/pipeline.md) |
| Understand the calculation and its limitations | [Technical details](technical_details/README.md) |

These pages describe PyMolFit 0.8. Examples use `correct(input_path=..., advanced=...)`; deprecated flat keywords have been removed. The theoretical spectrum selects fitting exclusions; it is not subtracted from the observation. PyMolFit models atmospheric absorption, not additive sky emission.
