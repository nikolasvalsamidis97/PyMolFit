from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.table import Table

from .aer_data import load_aer_line_window
from .atmosphere import AtmosphereProfile
from .physics import SPEED_OF_LIGHT_M_PER_S
from .spectrum import (
    Spectrum,
    air_to_vacuum_wavelength,
    normalize_wavelength_medium,
    vacuum_to_air_wavelength,
    wavelength_scale_to_micron,
)

RegionKind = Literal["fit", "exclude"]
RegionRanges = tuple[tuple[float, float], ...]

REGION_FILE_SCHEMA = "pymolfit.spectral_regions"
REGION_FILE_VERSION = 1
_REGION_COLUMNS = ("region_type", "wavelength_start", "wavelength_end")
_REGION_COLORS = {
    "fit": "#2E8B57",
    "exclude": "#C43C39",
}
_SPECIES_COLORS = {
    "H2O": "#0072B2",
    "O2": "#D55E00",
    "O3": "#009E73",
    "CO2": "#CC79A7",
    "CH4": "#E69F00",
    "CO": "#56B4E9",
    "N2O": "#666666",
}
DEFAULT_TELLURIC_MARKER_LIMIT = 10_000
DEFAULT_AUTOMATIC_REGION_COUNT = 100
DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS = 12.0


@dataclass(frozen=True)
class RegionSelection:
    """Fit and exclusion windows in a declared wavelength coordinate system.

    Regions use the wavelength unit and air/vacuum medium displayed during
    selection. Overlapping windows of the same type are merged. Use
    :meth:`converted` when applying a selection to a spectrum with different
    wavelength coordinates.
    """

    fit_ranges: RegionRanges = ()
    exclude_ranges: RegionRanges = ()
    wavelength_unit: str = "micron"
    wavelength_medium: str = "vacuum"
    output_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        wavelength_scale_to_micron(self.wavelength_unit)
        object.__setattr__(
            self,
            "wavelength_medium",
            normalize_wavelength_medium(self.wavelength_medium),
        )
        object.__setattr__(self, "fit_ranges", _normalize_ranges(self.fit_ranges))
        object.__setattr__(
            self,
            "exclude_ranges",
            _normalize_ranges(self.exclude_ranges),
        )
        object.__setattr__(
            self,
            "output_path",
            None if self.output_path is None else Path(self.output_path),
        )

    @property
    def is_empty(self) -> bool:
        return not self.fit_ranges and not self.exclude_ranges

    def converted(
        self,
        *,
        wavelength_unit: str,
        wavelength_medium: str,
    ) -> "RegionSelection":
        """Return these regions in another wavelength unit and medium."""

        target_medium = normalize_wavelength_medium(wavelength_medium)
        target_scale = wavelength_scale_to_micron(wavelength_unit)
        source_scale = wavelength_scale_to_micron(self.wavelength_unit)

        def convert_ranges(ranges: RegionRanges) -> RegionRanges:
            if not ranges:
                return ()
            values_micron = np.asarray(ranges, dtype=float) * source_scale
            if self.wavelength_medium == "air" and target_medium == "vacuum":
                values_micron = air_to_vacuum_wavelength(
                    values_micron,
                    unit="micron",
                )
            elif self.wavelength_medium == "vacuum" and target_medium == "air":
                values_micron = vacuum_to_air_wavelength(
                    values_micron,
                    unit="micron",
                )
            converted_values = values_micron / target_scale
            return tuple(
                (float(lower), float(upper))
                for lower, upper in converted_values
            )

        return RegionSelection(
            fit_ranges=convert_ranges(self.fit_ranges),
            exclude_ranges=convert_ranges(self.exclude_ranges),
            wavelength_unit=wavelength_unit,
            wavelength_medium=target_medium,
            output_path=self.output_path,
        )

    def write(self, path: str | Path) -> Path:
        """Write the selection as a versioned ECSV region file."""

        return save_region_file(self, path)


def save_region_file(selection: RegionSelection, path: str | Path) -> Path:
    """Save fit and exclusion windows as a portable ECSV table.

    The table records each interval's type and endpoints together with the
    wavelength unit and air/vacuum medium needed to apply it correctly later.
    Existing files are overwritten.
    """

    destination = Path(path)
    region_types = (
        ["fit"] * len(selection.fit_ranges)
        + ["exclude"] * len(selection.exclude_ranges)
    )
    ranges = selection.fit_ranges + selection.exclude_ranges
    starts = [region[0] for region in ranges]
    ends = [region[1] for region in ranges]
    table = Table(
        [region_types, starts, ends],
        names=_REGION_COLUMNS,
        dtype=("U7", "f8", "f8"),
    )
    table.meta.update(
        {
            "pymolfit_schema": REGION_FILE_SCHEMA,
            "schema_version": REGION_FILE_VERSION,
            "wavelength_unit": selection.wavelength_unit,
            "wavelength_medium": selection.wavelength_medium,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.write(destination, format="ascii.ecsv", overwrite=True)
    return destination


def load_region_file(path: str | Path) -> RegionSelection:
    """Load and validate a PyMolFit ECSV region file."""

    source = Path(path)
    table = Table.read(source, format="ascii.ecsv")
    missing = [name for name in _REGION_COLUMNS if name not in table.colnames]
    if missing:
        raise ValueError(
            f"region file {source} is missing columns: {', '.join(missing)}"
        )
    if table.meta.get("pymolfit_schema") != REGION_FILE_SCHEMA:
        raise ValueError(
            f"region file {source} does not declare the PyMolFit region schema"
        )
    if int(table.meta.get("schema_version", -1)) != REGION_FILE_VERSION:
        raise ValueError(
            f"unsupported region-file version in {source}: "
            f"{table.meta.get('schema_version')!r}"
        )

    wavelength_unit = table.meta.get("wavelength_unit")
    wavelength_medium = table.meta.get("wavelength_medium")
    if not wavelength_unit or not wavelength_medium:
        raise ValueError(
            f"region file {source} must declare wavelength_unit and "
            "wavelength_medium"
        )

    fit_ranges: list[tuple[float, float]] = []
    exclude_ranges: list[tuple[float, float]] = []
    for row in table:
        kind = str(row["region_type"]).strip().lower()
        interval = (
            float(row["wavelength_start"]),
            float(row["wavelength_end"]),
        )
        if kind == "fit":
            fit_ranges.append(interval)
        elif kind == "exclude":
            exclude_ranges.append(interval)
        else:
            raise ValueError(
                f"region file {source} contains unsupported region_type "
                f"{kind!r}"
            )

    return RegionSelection(
        fit_ranges=tuple(fit_ranges),
        exclude_ranges=tuple(exclude_ranges),
        wavelength_unit=str(wavelength_unit),
        wavelength_medium=str(wavelength_medium),
        output_path=source,
    )


class InteractiveRegionSelector:
    """Matplotlib interface for selecting fit and exclusion intervals.

    Zoom or pan to an interval, choose ``Fit`` or ``Exclude``, enable
    ``Draw regions``, and drag rectangles around the desired lines. Only each
    rectangle's wavelength limits are stored, so its vertical size does not
    affect the fit. Drawing remains enabled until the checkbox is cleared. In
    ``Delete`` mode, rectangles remove intersecting regions. AER catalogue
    ticks identify candidate telluric transitions by default. ``Automatic``
    proposes editable fit windows around the requested number of transitions
    with the strongest expected atmospheric absorption. By default, each
    proposal extends 12 sampled detector pixels to either side of the line so
    its core, wings, and nearby continuum can constrain alignment and the LSF.
    The side panel lists regions held in memory,
    provides an editable output filename, and includes ``Undo``, ``Clear``,
    and ``Save All`` controls.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        *,
        output_path: str | Path | None = None,
        initial_regions: RegionSelection | None = None,
        title: str | None = None,
        show_telluric_lines: bool = True,
        max_telluric_markers: int = DEFAULT_TELLURIC_MARKER_LIMIT,
        automatic_region_count: int = DEFAULT_AUTOMATIC_REGION_COUNT,
        automatic_region_half_width_pixels: float = (
            DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS
        ),
        show: bool = True,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import (
                Button,
                CheckButtons,
                RadioButtons,
                RectangleSelector,
                TextBox,
            )
        except ImportError as exc:
            raise ImportError(
                "interactive region selection requires Matplotlib; install "
                "'pymolfit[interactive]'"
            ) from exc
        if max_telluric_markers <= 0:
            raise ValueError("max_telluric_markers must be positive")
        if automatic_region_count <= 0:
            raise ValueError("automatic_region_count must be positive")
        if (
            not np.isfinite(automatic_region_half_width_pixels)
            or automatic_region_half_width_pixels <= 0
        ):
            raise ValueError(
                "automatic_region_half_width_pixels must be positive and finite"
            )

        self.spectrum = spectrum.sorted()
        self.output_path = None if output_path is None else Path(output_path)
        self._plt = plt
        self._mode: RegionKind | Literal["delete"] = "fit"
        self._records: list[tuple[RegionKind, float, float]] = []
        self._history: list[list[tuple[RegionKind, float, float]]] = []
        self._patches: list[object] = []
        self._region_labels: list[object] = []
        self._telluric_marker_count = 0
        self._telluric_marker_error: str | None = None
        self._automatic_max_lines = int(max_telluric_markers)
        self._automatic_region_half_width_pixels = float(
            automatic_region_half_width_pixels
        )

        if initial_regions is not None:
            converted = initial_regions.converted(
                wavelength_unit=self.spectrum.wavelength_unit,
                wavelength_medium=self.spectrum.wavelength_medium,
            )
            self._records.extend(
                ("fit", lower, upper)
                for lower, upper in converted.fit_ranges
            )
            self._records.extend(
                ("exclude", lower, upper)
                for lower, upper in converted.exclude_ranges
            )

        self.figure, self.axis = plt.subplots(figsize=(13, 6))
        self.figure.subplots_adjust(
            left=0.08,
            right=0.80,
            top=0.92,
            bottom=0.12,
        )
        self._plot_spectrum()
        data_xlim = self.axis.get_xlim()
        data_ylim = self.axis.get_ylim()
        if show_telluric_lines:
            self._plot_telluric_markers(max_lines=max_telluric_markers)
        self.axis.set_xlabel(
            f"{self.spectrum.wavelength_medium.capitalize()} wavelength "
            f"[{self.spectrum.wavelength_unit}]"
        )
        self.axis.set_ylabel("Flux")
        self.axis.set_title(
            title if title is not None else "Select telluric fitting regions"
        )

        mode_axis = self.figure.add_axes((0.83, 0.72, 0.14, 0.17))
        self.mode_buttons = RadioButtons(
            mode_axis,
            ("Fit", "Exclude", "Delete"),
            active=0,
        )
        self.mode_buttons.on_clicked(self._set_mode)

        draw_axis = self.figure.add_axes((0.83, 0.645, 0.14, 0.055))
        auto_count_axis = self.figure.add_axes((0.83, 0.575, 0.14, 0.05))
        auto_axis = self.figure.add_axes((0.83, 0.51, 0.14, 0.05))
        undo_axis = self.figure.add_axes((0.83, 0.445, 0.14, 0.05))
        clear_axis = self.figure.add_axes((0.83, 0.38, 0.14, 0.05))
        save_axis = self.figure.add_axes((0.83, 0.315, 0.14, 0.05))
        save_name_axis = self.figure.add_axes((0.83, 0.245, 0.14, 0.05))
        self.draw_checkbox = CheckButtons(
            draw_axis,
            ("Draw regions",),
            (False,),
        )
        self.auto_count_box = TextBox(
            auto_count_axis,
            "Lines ",
            initial=str(int(automatic_region_count)),
        )
        self.auto_button = Button(auto_axis, "Automatic")
        self.undo_button = Button(undo_axis, "Undo")
        self.clear_button = Button(clear_axis, "Clear")
        self.save_button = Button(save_axis, "Save All")
        self.save_name_box = TextBox(
            save_name_axis,
            "Filename ",
            initial=(
                self.output_path.name
                if self.output_path is not None
                else "telluric_regions.ecsv"
            ),
        )
        self.draw_checkbox.on_clicked(self._toggle_rectangle)
        self.auto_count_box.on_submit(self._set_automatic_region_count)
        self.auto_button.on_clicked(self._automatic_event)
        self.undo_button.on_clicked(self._undo_event)
        self.clear_button.on_clicked(self._clear_event)
        self.save_button.on_clicked(self._save_event)
        self.save_name_box.on_submit(self._set_output_filename)

        self.status_text = self.figure.text(
            0.83,
            0.205,
            "",
            ha="left",
            va="top",
            fontsize=8.5,
        )
        self.rectangle_selector = RectangleSelector(
            self.axis,
            self._on_rectangle,
            useblit=True,
            props={
                "facecolor": _REGION_COLORS["fit"],
                "alpha": 0.25,
            },
            button=1,
            minspanx=0.0,
            minspany=0.0,
            spancoords="data",
            interactive=False,
        )
        self.rectangle_selector.set_active(False)
        # Marker collections use axes-relative y coordinates, while
        # RectangleSelector starts with an invisible artist at x=0. Preserve
        # the spectrum-only limits so those helper artists cannot flatten
        # small physical fluxes or compress large wavelength coordinates.
        self.axis.set_xlim(data_xlim)
        self.axis.set_ylim(data_ylim)
        self.axis.set_autoscalex_on(False)
        self.axis.set_autoscaley_on(False)
        self._redraw_regions()
        if show:
            plt.show()

    @property
    def selection(self) -> RegionSelection:
        """Return the current normalized selection."""

        return RegionSelection(
            fit_ranges=tuple(
                (lower, upper)
                for kind, lower, upper in self._records
                if kind == "fit"
            ),
            exclude_ranges=tuple(
                (lower, upper)
                for kind, lower, upper in self._records
                if kind == "exclude"
            ),
            wavelength_unit=self.spectrum.wavelength_unit,
            wavelength_medium=self.spectrum.wavelength_medium,
            output_path=self.output_path,
        )

    def add_region(
        self,
        lower: float,
        upper: float,
        *,
        kind: RegionKind = "fit",
    ) -> None:
        """Add a fit or exclusion interval and redraw the plot."""

        if kind not in _REGION_COLORS:
            raise ValueError("kind must be 'fit' or 'exclude'")
        lower, upper = self._bounded_interval(lower, upper)
        self._remember()
        self._records.append((kind, lower, upper))
        self._redraw_regions()

    def delete_regions(self, lower: float, upper: float) -> None:
        """Delete every selected interval overlapping ``lower`` to ``upper``."""

        lower, upper = self._bounded_interval(lower, upper)
        retained = [
            record
            for record in self._records
            if record[2] < lower or record[1] > upper
        ]
        if len(retained) == len(self._records):
            return
        self._remember()
        self._records = retained
        self._redraw_regions()

    def mark_visible_region(self, *, kind: RegionKind | None = None) -> None:
        """Add the currently visible wavelength interval to the selection.

        This supports a zoom-then-mark workflow. When ``kind`` is omitted, the
        active Fit or Exclude mode is used. Delete mode cannot mark a region.
        """

        selected_kind = self._mode if kind is None else kind
        if selected_kind not in _REGION_COLORS:
            raise ValueError("choose Fit or Exclude before marking the view")
        lower, upper = self.axis.get_xlim()
        self.add_region(lower, upper, kind=selected_kind)

    def add_automatic_fit_regions(self, count: int | None = None) -> RegionRanges:
        """Add windows around the strongest expected telluric transitions.

        ``count`` is the number of catalogue transitions considered, not
        necessarily the final number of windows: nearby transitions are merged.
        Transitions are ranked by AER line strength times a representative
        atmospheric column for their molecule, so strengths from different
        species are compared on an atmospheric absorption scale.
        Lines falling in detector or echelle-order gaps are skipped. Proposed
        windows are ordinary fit regions and can be edited, undone, cleared, or
        saved in exactly the same way as manually drawn regions.
        """

        resolved_count = (
            self._parse_automatic_region_count(self.auto_count_box.text)
            if count is None
            else _validate_automatic_region_count(count)
        )
        ranges = _automatic_fit_regions(
            self.spectrum,
            count=resolved_count,
            max_lines=max(self._automatic_max_lines, resolved_count),
            half_width_pixels=self._automatic_region_half_width_pixels,
        )
        if not ranges:
            raise ValueError(
                "no covered positive-strength AER transitions were found"
            )
        self._remember()
        self._records.extend(("fit", lower, upper) for lower, upper in ranges)
        self._redraw_regions()
        final_count = len(self.selection.fit_ranges)
        self._update_status(
            f"Automatic: {len(ranges)} windows / {resolved_count} lines; "
            f"{final_count} fit regions total."
        )
        return ranges

    def undo(self) -> None:
        """Restore the selection state before the most recent edit."""

        if not self._history:
            return
        self._records = self._history.pop()
        self._redraw_regions()

    def clear(self) -> None:
        """Remove every fit and exclusion interval."""

        if not self._records:
            return
        self._remember()
        self._records = []
        self._redraw_regions()

    def save(self, path: str | Path | None = None) -> Path:
        """Save the current selection to ECSV."""

        if path is None:
            destination = self._output_path_from_filename(
                self.save_name_box.text
            )
        else:
            destination = Path(path)
        if destination is None:
            raise ValueError(
                "no region output path was provided; call save(path) or pass "
                "output_path to select_telluric_regions"
            )
        self.output_path = destination
        written = self.selection.write(destination)
        self._update_status(f"Saved {written.name}")
        return written

    def close(self) -> None:
        """Close the selector figure."""

        self._plt.close(self.figure)

    def _plot_spectrum(self) -> None:
        wavelength = self.spectrum.wavelength
        flux = self.spectrum.flux
        valid = np.isfinite(wavelength) & np.isfinite(flux)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            raise ValueError("cannot select regions from an empty spectrum")
        if indices.size == 1:
            sections = (indices,)
        else:
            spacing = np.diff(wavelength[indices])
            positive = spacing[np.isfinite(spacing) & (spacing > 0)]
            if positive.size:
                gap_limit = 20.0 * float(np.nanmedian(positive))
                breaks = np.flatnonzero(
                    ~np.isfinite(spacing) | (spacing > gap_limit)
                ) + 1
                sections = tuple(np.split(indices, breaks))
            else:
                sections = (indices,)
        for section in sections:
            if section.size:
                self.axis.plot(
                    wavelength[section],
                    flux[section],
                    color="#222222",
                    linewidth=0.75,
                )

    def _plot_telluric_markers(self, *, max_lines: int) -> None:
        try:
            wavelength, species = _aer_markers_for_spectrum(
                self.spectrum,
                max_lines=max_lines,
            )
        except Exception as exc:
            self._telluric_marker_error = str(exc)
            warnings.warn(
                f"AER telluric markers could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._telluric_marker_count = int(wavelength.size)
        for name in sorted(set(species.tolist())):
            selected = species == name
            self.axis.vlines(
                wavelength[selected],
                0.91,
                0.99,
                transform=self.axis.get_xaxis_transform(),
                color=_SPECIES_COLORS.get(name, "#999999"),
                alpha=0.62,
                linewidth=0.55,
                label=f"AER {name}",
            )

    def _bounded_interval(
        self,
        lower: float,
        upper: float,
    ) -> tuple[float, float]:
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("region endpoints must be finite")
        if lower > upper:
            lower, upper = upper, lower
        finite = self.spectrum.wavelength[
            np.isfinite(self.spectrum.wavelength)
        ]
        minimum = float(np.nanmin(finite))
        maximum = float(np.nanmax(finite))
        lower = max(lower, minimum)
        upper = min(upper, maximum)
        if upper <= lower:
            raise ValueError("region must have positive width inside the spectrum")
        return lower, upper

    def _remember(self) -> None:
        self._history.append(list(self._records))

    def _on_rectangle(self, click_event: object, release_event: object) -> None:
        lower = getattr(click_event, "xdata", None)
        upper = getattr(release_event, "xdata", None)
        if lower is None or upper is None or lower == upper:
            self._update_status("No region added. Draw another rectangle.")
            return
        if self._mode == "delete":
            self.delete_regions(lower, upper)
            message = "Region deleted."
        else:
            self.add_region(lower, upper, kind=self._mode)
            message = "Region added."
        self._update_status(f"{message} Draw another or clear the checkbox.")

    def _set_mode(self, label: str) -> None:
        self._mode = label.strip().lower()  # type: ignore[assignment]
        color = (
            _REGION_COLORS[self._mode]
            if self._mode in _REGION_COLORS
            else "#666666"
        )
        self.rectangle_selector.set_props(facecolor=color, alpha=0.25)
        self._update_status()

    def _toggle_rectangle(self, _label: str) -> None:
        active = bool(self.draw_checkbox.get_status()[0])
        self.rectangle_selector.set_active(active)
        message = (
            f"Draw {self._mode} rectangles in the spectrum."
            if active
            else "Rectangle drawing disabled; zoom and pan are available."
        )
        self._update_status(message)

    def _set_output_filename(self, filename: str) -> None:
        try:
            destination = self._output_path_from_filename(filename)
        except ValueError as exc:
            self._update_status(str(exc))
            return
        self.output_path = destination
        self._update_status(f"Output filename: {destination.name}")

    def _set_automatic_region_count(self, value: str) -> None:
        try:
            count = self._parse_automatic_region_count(value)
        except ValueError as exc:
            self._update_status(str(exc))
            return
        self._update_status(
            f"Automatic selection will use the {count} strongest expected "
            "telluric lines."
        )

    @staticmethod
    def _parse_automatic_region_count(value: str) -> int:
        try:
            count = int(value.strip())
        except ValueError as exc:
            raise ValueError("automatic line count must be a positive integer") from exc
        return _validate_automatic_region_count(count)

    def _output_path_from_filename(self, filename: str) -> Path:
        name = filename.strip()
        if not name:
            raise ValueError("output filename cannot be empty")
        if Path(name).name != name:
            raise ValueError("enter a filename without a directory")
        if Path(name).suffix.lower() != ".ecsv":
            name += ".ecsv"
        directory = (
            self.output_path.parent
            if self.output_path is not None
            else Path.cwd()
        )
        return directory / name

    def _undo_event(self, _event: object) -> None:
        self.undo()

    def _automatic_event(self, _event: object) -> None:
        try:
            self.add_automatic_fit_regions()
        except (OSError, ValueError) as exc:
            self._update_status(str(exc))

    def _clear_event(self, _event: object) -> None:
        self.clear()

    def _save_event(self, _event: object) -> None:
        try:
            self.save()
        except (OSError, ValueError) as exc:
            self._update_status(str(exc))

    def _redraw_regions(self) -> None:
        for patch in self._patches:
            patch.remove()  # type: ignore[attr-defined]
        self._patches = []
        for label in self._region_labels:
            label.remove()  # type: ignore[attr-defined]
        self._region_labels = []
        selection = self.selection
        region_number = 0
        for kind, ranges in (
            ("fit", selection.fit_ranges),
            ("exclude", selection.exclude_ranges),
        ):
            for lower, upper in ranges:
                region_number += 1
                patch = self.axis.axvspan(
                    lower,
                    upper,
                    facecolor=_REGION_COLORS[kind],
                    edgecolor=_REGION_COLORS[kind],
                    alpha=0.22,
                    linewidth=1.0,
                    label=(
                        "Fit region"
                        if kind == "fit" and not any(
                            record.get_label() == "Fit region"
                            for record in self._patches  # type: ignore[attr-defined]
                        )
                        else (
                            "Excluded region"
                            if kind == "exclude" and not any(
                                record.get_label() == "Excluded region"
                                for record in self._patches  # type: ignore[attr-defined]
                            )
                            else "_nolegend_"
                        )
                    ),
                )
                self._patches.append(patch)
                self._region_labels.append(
                    self.axis.text(
                        0.5 * (lower + upper),
                        0.98,
                        f"R{region_number}",
                        color=_REGION_COLORS[kind],
                        fontsize=8,
                        fontweight="bold",
                        ha="center",
                        va="top",
                        transform=self.axis.get_xaxis_transform(),
                        clip_on=True,
                    )
                )
        handles, labels = self.axis.get_legend_handles_labels()
        existing_legend = self.axis.get_legend()
        if existing_legend is not None:
            existing_legend.remove()
        if labels:
            self.axis.legend(handles, labels, loc="best")
        self._update_status()
        self.figure.canvas.draw_idle()

    def _update_status(self, message: str | None = None) -> None:
        selection = self.selection
        display_regions = (
            tuple(("fit", lower, upper) for lower, upper in selection.fit_ranges)
            + tuple(
                ("exclude", lower, upper)
                for lower, upper in selection.exclude_ranges
            )
        )
        lines = []
        if message is not None:
            lines.append(message)
        lines.extend(
            (
                f"Mode: {self._mode}",
                (
                    "Drawing: on"
                    if self.rectangle_selector.active
                    else "Drawing: off"
                ),
                f"Regions in memory: {len(display_regions)}",
            )
        )
        if self._telluric_marker_count:
            lines.append(
                f"AER markers: {self._telluric_marker_count}"
            )
        elif self._telluric_marker_error is not None:
            lines.append("AER markers unavailable")
        for index, (kind, lower, upper) in enumerate(display_regions[:5], start=1):
            lines.append(f"R{index} {kind}:")
            lines.append(f"  {lower:.6g} - {upper:.6g}")
        if len(display_regions) > 5:
            lines.append(f"... plus {len(display_regions) - 5} more")
        if display_regions:
            lines.append("Save All writes every region.")
        self.status_text.set_text("\n".join(lines))
        self.figure.canvas.draw_idle()


def select_telluric_regions(
    spectrum: Spectrum | None = None,
    *,
    wavelength: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: str = "vacuum",
    output_path: str | Path | None = None,
    initial_regions: RegionSelection | None = None,
    title: str | None = None,
    show_telluric_lines: bool = True,
    max_telluric_markers: int = DEFAULT_TELLURIC_MARKER_LIMIT,
    automatic_region_count: int = DEFAULT_AUTOMATIC_REGION_COUNT,
    automatic_region_half_width_pixels: float = (
        DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS
    ),
    reuse_existing: bool = True,
    show: bool = True,
) -> InteractiveRegionSelector | RegionSelection:
    """Open an interactive spectrum window for selecting fit regions.

    Supply either a :class:`Spectrum` or wavelength/flux arrays. Zoom or pan,
    choose fit/exclusion mode, enable ``Draw regions``, and drag rectangles;
    their horizontal limits become numbered regions. Drawing remains enabled
    until its checkbox is cleared. AER telluric-line ticks are displayed by
    default. Up to 10,000 of the strongest in-range catalogue transitions are
    shown so weaker O2 and trace-species lines are not hidden by the much more
    numerous H2O lines. Enter a line count and press ``Automatic`` to propose
    fit windows around that many transitions with the greatest expected
    atmospheric absorption. This ranking combines each AER line intensity with
    a representative atmospheric column for its molecule. Windows in detector
    gaps are skipped and overlapping windows are merged. The proposals remain
    fully editable. Edit the filename field if needed, then press ``Save All``
    to write every region together as ECSV. In Jupyter, enable
    ``%matplotlib widget`` and install ``pymolfit[interactive]``. The returned
    selector exposes ``selection`` and ``save()`` for notebook and scripted use.

    When ``output_path`` already exists and ``reuse_existing`` is ``True``, the
    saved :class:`RegionSelection` is loaded and returned without opening a new
    selector window. Set ``reuse_existing=False`` to open the interface with
    the saved regions loaded for editing. Explicit ``initial_regions`` cannot
    be combined with automatic reuse because one source would otherwise be
    silently ignored.
    """

    has_arrays = wavelength is not None or flux is not None
    if spectrum is not None and has_arrays:
        raise ValueError(
            "provide either spectrum or wavelength/flux arrays, not both"
        )
    if spectrum is None:
        if wavelength is None or flux is None:
            raise ValueError(
                "provide spectrum or both wavelength and flux arrays"
            )
        spectrum = Spectrum(
            wavelength=np.asarray(wavelength, dtype=float),
            flux=np.asarray(flux, dtype=float),
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
        )
    existing_path = _resolved_region_output_path(output_path)
    if reuse_existing and existing_path is not None and existing_path.is_file():
        if initial_regions is not None:
            raise ValueError(
                "initial_regions cannot be combined with automatic reuse of an "
                "existing output_path; set reuse_existing=False to edit it"
            )
        selection = load_region_file(existing_path)
        selection.converted(
            wavelength_unit=spectrum.wavelength_unit,
            wavelength_medium=spectrum.wavelength_medium,
        )
        return selection
    if (
        not reuse_existing
        and existing_path is not None
        and existing_path.is_file()
        and initial_regions is None
    ):
        initial_regions = load_region_file(existing_path)
    return InteractiveRegionSelector(
        spectrum,
        output_path=output_path,
        initial_regions=initial_regions,
        title=title,
        show_telluric_lines=show_telluric_lines,
        max_telluric_markers=max_telluric_markers,
        automatic_region_count=automatic_region_count,
        automatic_region_half_width_pixels=automatic_region_half_width_pixels,
        show=show,
    )


def _resolved_region_output_path(path: str | Path | None) -> Path | None:
    """Return the filename that the selector's Save All control will use."""

    if path is None:
        return None
    destination = Path(path)
    if destination.suffix.lower() != ".ecsv":
        destination = destination.with_name(destination.name + ".ecsv")
    return destination


def _aer_markers_for_spectrum(
    spectrum: Spectrum,
    *,
    max_lines: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return AER transitions in the spectrum's displayed coordinates."""

    wavelength, species, _strength = _aer_catalog_for_spectrum(
        spectrum,
        max_lines=max_lines,
    )
    return wavelength, species


def _aer_catalog_for_spectrum(
    spectrum: Spectrum,
    *,
    max_lines: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AER wavelengths, species, and strengths in display coordinates."""

    input_vacuum = spectrum.to_vacuum().to_unit("micron")
    erf_factor = _selector_erf_factor(spectrum)
    observatory_wavelength = input_vacuum.wavelength / erf_factor
    finite = observatory_wavelength[np.isfinite(observatory_wavelength)]
    if finite.size == 0:
        raise ValueError("the spectrum contains no finite wavelengths")

    artifact = load_aer_line_window(
        wavelength_min_micron=float(np.nanmin(finite)),
        wavelength_max_micron=float(np.nanmax(finite)),
        max_lines=max_lines,
    )
    line_list = artifact.line_list
    marker_micron = np.asarray(line_list.wavelength, dtype=float) * erf_factor
    if spectrum.wavelength_medium == "air":
        marker_micron = vacuum_to_air_wavelength(marker_micron, unit="micron")
    marker_wavelength = (
        marker_micron / wavelength_scale_to_micron(spectrum.wavelength_unit)
    )
    species = np.asarray(line_list.species, dtype=str)
    strength = np.asarray(line_list.strength, dtype=float)

    lower = float(np.nanmin(spectrum.wavelength))
    upper = float(np.nanmax(spectrum.wavelength))
    keep = (
        np.isfinite(marker_wavelength)
        & (marker_wavelength >= lower)
        & (marker_wavelength <= upper)
    )
    return marker_wavelength[keep], species[keep], strength[keep]


def _automatic_fit_regions(
    spectrum: Spectrum,
    *,
    count: int,
    max_lines: int,
    half_width_pixels: float,
) -> RegionRanges:
    """Propose strongest expected telluric windows on sampled pixels."""

    count = _validate_automatic_region_count(count)
    if not np.isfinite(half_width_pixels) or half_width_pixels <= 0:
        raise ValueError("half_width_pixels must be positive and finite")
    marker_wavelength, species, strength = _aer_catalog_for_spectrum(
        spectrum,
        max_lines=max(max_lines, count),
    )
    if marker_wavelength.size == 0:
        return ()

    best_distance = np.full(marker_wavelength.shape, np.inf, dtype=float)
    best_step = np.full(marker_wavelength.shape, np.nan, dtype=float)
    best_lower = np.full(marker_wavelength.shape, np.nan, dtype=float)
    best_upper = np.full(marker_wavelength.shape, np.nan, dtype=float)
    for sampled_wavelength in _selector_sampling_sections(spectrum):
        insertion = np.searchsorted(sampled_wavelength, marker_wavelength)
        right = np.clip(insertion, 0, sampled_wavelength.size - 1)
        left = np.clip(insertion - 1, 0, sampled_wavelength.size - 1)
        choose_right = (
            np.abs(sampled_wavelength[right] - marker_wavelength)
            < np.abs(sampled_wavelength[left] - marker_wavelength)
        )
        nearest = np.where(choose_right, right, left)
        local_steps = _local_selector_pixel_steps(sampled_wavelength)
        distance = np.abs(sampled_wavelength[nearest] - marker_wavelength)
        normalized_distance = distance / local_steps[nearest]
        covered = normalized_distance <= 1.5
        improved = covered & (normalized_distance < best_distance)
        best_distance[improved] = normalized_distance[improved]
        best_step[improved] = local_steps[nearest[improved]]
        best_lower[improved] = sampled_wavelength[0]
        best_upper[improved] = sampled_wavelength[-1]

    absorption_score = _expected_aer_absorption_score(strength, species)
    eligible = (
        np.isfinite(marker_wavelength)
        & np.isfinite(strength)
        & (strength > 0)
        & np.isfinite(absorption_score)
        & (absorption_score > 0)
        & np.isfinite(best_step)
    )
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return ()
    ranking = eligible_indices[
        np.argsort(-absorption_score[eligible_indices], kind="stable")
    ][:count]

    raw_ranges: list[tuple[float, float]] = []
    for index in ranking:
        half_width = half_width_pixels * best_step[index]
        lower = max(marker_wavelength[index] - half_width, best_lower[index])
        upper = min(marker_wavelength[index] + half_width, best_upper[index])
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            raw_ranges.append((float(lower), float(upper)))
    return _normalize_ranges(tuple(raw_ranges))


@lru_cache(maxsize=1)
def _representative_telluric_columns_cm2() -> tuple[tuple[str, float], ...]:
    """Return deterministic vertical columns used only for line ranking."""

    atmosphere = AtmosphereProfile.standard_midlatitude(airmass=1.0)
    return tuple(
        (species, atmosphere.total_vertical_column_cm2(species))
        for species in atmosphere.species_names
    )


def _expected_aer_absorption_score(
    strength: np.ndarray,
    species: np.ndarray,
) -> np.ndarray:
    """Estimate integrated atmospheric absorption for catalogue ranking.

    HITRAN/AER line intensity alone is not comparable across molecules because
    their atmospheric columns differ by orders of magnitude. Multiplying by a
    representative molecular column is the optically thin integrated-opacity
    scaling and gives a physical, observation-independent ordering. The actual
    correction still uses the observation-specific atmosphere and full
    radiative-transfer calculation.
    """

    strength_array = np.asarray(strength, dtype=float)
    species_array = np.asarray(species, dtype=str)
    if strength_array.shape != species_array.shape:
        raise ValueError("strength and species must have matching shapes")
    columns = dict(_representative_telluric_columns_cm2())
    molecular_column = np.asarray(
        [columns.get(name.strip().upper(), 0.0) for name in species_array],
        dtype=float,
    )
    return strength_array * molecular_column


def _selector_sampling_sections(spectrum: Spectrum) -> tuple[np.ndarray, ...]:
    """Return independently sampled wavelength sections for coverage checks."""

    valid = spectrum.valid & np.isfinite(spectrum.wavelength)
    grouped_wavelengths: list[np.ndarray] = []
    if spectrum.group_id is None:
        grouped_wavelengths.append(np.asarray(spectrum.wavelength[valid], dtype=float))
    else:
        group_id = np.asarray(spectrum.group_id)
        for value in np.unique(group_id[valid]):
            grouped_wavelengths.append(
                np.asarray(
                    spectrum.wavelength[valid & (group_id == value)],
                    dtype=float,
                )
            )

    sections: list[np.ndarray] = []
    for raw_wavelength in grouped_wavelengths:
        wavelength = np.unique(raw_wavelength[np.isfinite(raw_wavelength)])
        if wavelength.size < 2:
            continue
        steps = np.diff(wavelength)
        positive = steps[np.isfinite(steps) & (steps > 0)]
        if positive.size == 0:
            continue
        gap_limit = 20.0 * float(np.nanmedian(positive))
        boundaries = np.flatnonzero(steps > gap_limit) + 1
        sections.extend(
            section
            for section in np.split(wavelength, boundaries)
            if section.size >= 2
        )
    return tuple(sections)


def _local_selector_pixel_steps(wavelength: np.ndarray) -> np.ndarray:
    """Estimate a positive local pixel dispersion within one sampled section."""

    differences = np.diff(wavelength)
    fallback = float(np.nanmedian(differences))
    previous = np.concatenate(([np.nan], differences))
    following = np.concatenate((differences, [np.nan]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local = np.nanmedian(np.column_stack((previous, following)), axis=1)
    local[~np.isfinite(local) | (local <= 0)] = fallback
    return local


def _validate_automatic_region_count(count: int) -> int:
    if isinstance(count, bool) or int(count) != count or count <= 0:
        raise ValueError("automatic line count must be a positive integer")
    return int(count)


def _selector_erf_factor(spectrum: Spectrum) -> float:
    header = _selector_header(spectrum)
    if header is None:
        return 1.0

    # Imported lazily because workflow imports the region-file API.
    from .workflow import _spectral_frame_velocity_km_s

    frame_velocity = _spectral_frame_velocity_km_s(header)
    if frame_velocity is None:
        return 1.0
    _, velocity_km_s = frame_velocity
    return (1.0 + 1.55e-8) * (
        1.0 + velocity_km_s / (SPEED_OF_LIGHT_M_PER_S / 1_000.0)
    )


def _selector_header(spectrum: Spectrum) -> Mapping[str, object] | None:
    observation = spectrum.meta.get("observation")
    if isinstance(observation, Mapping):
        return observation

    source = spectrum.meta.get("source")
    if source is None:
        return None

    # Imported lazily because workflow imports the region-file API.
    from .workflow import _load_fits_header_if_available

    return _load_fits_header_if_available(str(source), None)


def _normalize_ranges(ranges: RegionRanges) -> RegionRanges:
    normalized: list[tuple[float, float]] = []
    for raw_lower, raw_upper in ranges:
        lower = float(raw_lower)
        upper = float(raw_upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("region endpoints must be finite")
        if lower > upper:
            lower, upper = upper, lower
        if upper <= lower:
            raise ValueError("regions must have positive width")
        normalized.append((lower, upper))
    normalized.sort()

    merged: list[tuple[float, float]] = []
    for lower, upper in normalized:
        if merged and lower <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return tuple(merged)
