# Select, edit and save regions

Run the [interactive selector notebook](04_interactive_selector.ipynb) for the same workflow in executable cells.

Fitting regions choose where the observation constrains atmospheric parameters. Exclusions protect stellar features and defective pixels. Exclusions take precedence; transmission is still calculated there for the correction.

Run from `docs/user_guide/`, or change the paths. In Jupyter, first run `%matplotlib widget`; install the `interactive` extra in that kernel's environment.

```python
from pymolfit import select_telluric_regions

selector = select_telluric_regions(
    "data/harps_nad_crop_air.fits",
    automatic_region_count=100,
    automatic_region_half_width_pixels=12,
    show_interactive_window=True,
)
```

The strongest ranked catalogue transitions propose windows automatically. The requested count is a line count, not the number of merged regions. The editor is disabled unless explicitly enabled.

1. Zoom to the feature of interest.
2. Choose **Fit** or **Exclude**, enable **Draw regions**, and drag over wavelength intervals.
3. Disable drawing before panning or zooming again. **Delete**, **Undo** and **Clear** edit selections.
4. Save after editing, using **Save All** or the following cell:

```python
selector.save("regions/beta_pic.ecsv")
```

In a Python script, close the editing window before the subsequent save/correction runs. In a notebook, finish editing before running the next cell.

```python
from pymolfit import correct

result = correct(
    "data/harps_nad_crop_air.fits",
    region_file="regions/beta_pic.ecsv",
    output_path="outputs/beta_pic_regions.ecsv",
    show_plot=True,
)
```

For automatic stellar exclusions, pass `theoretical_spectrum=template` using the object from notebook 2. Add `enable_theoretical_controls=True` to edit its broadening, alignment, depth threshold and padding in the window.

## Without a window

```python
selector = select_telluric_regions("data/harps_nad_crop_air.fits", automatic_region_count=100)
selector.save("regions/beta_pic.ecsv")
```

## Reopen or regenerate

Pass `output_path="regions/beta_pic.ecsv"` and `show_interactive_window=True` to reopen saved intervals. With `reuse_existing=True` (default), saved regions take precedence over a new requested line count. To generate a fresh selection, omit an existing `output_path`, then save to your chosen destination. `reuse_existing=False` alone still starts from that file if it exists.

The editor's **Automatic** button adds proposed fitting windows; clear unwanted old regions before replacing the selection. Files store units, air/vacuum medium and reference frame, so do not manually shift their values. For reuse across a time series, `wavelength_frame="observatory"` stores telluric intervals in observatory-frame vacuum coordinates; stellar exclusions generally need regenerating for each exposure.
