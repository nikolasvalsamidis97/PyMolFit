from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from .theoretical import StellarMaskResult, TheoreticalSpectrum

RegionKind = Literal["fit", "exclude"]
RegionRanges = tuple[tuple[float, float], ...]
RegionWavelengthFrame = Literal["native", "observatory"]

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
_SELECTOR_MIN_DISPLAY_POINTS = 512
_SELECTOR_POINTS_PER_SCREEN_PIXEL = 2.0
_SELECTOR_HIDE_MARKERS_VIEW_FRACTION = 0.8
_SELECTOR_ALL_MARKERS_VIEW_FRACTION = 0.02
_SELECTOR_MARKERS_PER_SCREEN_PIXEL = 1.5
_SELECTOR_HIDE_REGION_LABELS_VIEW_FRACTION = 0.25
_SELECTOR_REGION_LABEL_SPACING_PIXELS = 55.0


@dataclass(frozen=True)
class RegionSelection:
    """Fit and exclusion windows in a declared wavelength coordinate system.

    Regions use the wavelength unit, air/vacuum medium, and velocity frame
    displayed during selection. ``native`` coordinates match the input
    spectrum as supplied and are normally exposure-specific. ``observatory``
    coordinates keep terrestrial absorption fixed and can therefore be reused
    for barycentric or heliocentric time-series spectra. Overlapping windows
    of the same type are merged. Use :meth:`converted` when applying a
    selection to a spectrum with different wavelength units or medium.
    """

    fit_ranges: RegionRanges = ()
    exclude_ranges: RegionRanges = ()
    wavelength_unit: str = "micron"
    wavelength_medium: str = "vacuum"
    wavelength_frame: RegionWavelengthFrame = "native"
    output_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        wavelength_scale_to_micron(self.wavelength_unit)
        object.__setattr__(
            self,
            "wavelength_medium",
            normalize_wavelength_medium(self.wavelength_medium),
        )
        object.__setattr__(
            self,
            "wavelength_frame",
            _normalize_region_wavelength_frame(self.wavelength_frame),
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
    ) -> RegionSelection:
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
            return tuple((float(lower), float(upper)) for lower, upper in converted_values)

        return RegionSelection(
            fit_ranges=convert_ranges(self.fit_ranges),
            exclude_ranges=convert_ranges(self.exclude_ranges),
            wavelength_unit=wavelength_unit,
            wavelength_medium=target_medium,
            wavelength_frame=self.wavelength_frame,
            output_path=self.output_path,
        )

    def write(self, path: str | Path) -> Path:
        """Write the selection as a versioned ECSV region file."""

        return save_region_file(self, path)


def save_region_file(selection: RegionSelection, path: str | Path) -> Path:
    """Save fit and exclusion windows as a portable ECSV table.

    The table records each interval's type and endpoints together with the
    wavelength unit, air/vacuum medium, and velocity frame needed to apply it
    correctly later. Existing files are overwritten.
    """

    destination = Path(path)
    region_types = ["fit"] * len(selection.fit_ranges) + ["exclude"] * len(selection.exclude_ranges)
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
            "wavelength_frame": selection.wavelength_frame,
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
        raise ValueError(f"region file {source} is missing columns: {', '.join(missing)}")
    if table.meta.get("pymolfit_schema") != REGION_FILE_SCHEMA:
        raise ValueError(f"region file {source} does not declare the PyMolFit region schema")
    if int(table.meta.get("schema_version", -1)) != REGION_FILE_VERSION:
        raise ValueError(
            f"unsupported region-file version in {source}: {table.meta.get('schema_version')!r}"
        )

    wavelength_unit = table.meta.get("wavelength_unit")
    wavelength_medium = table.meta.get("wavelength_medium")
    if not wavelength_unit or not wavelength_medium:
        raise ValueError(f"region file {source} must declare wavelength_unit and wavelength_medium")

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
            raise ValueError(f"region file {source} contains unsupported region_type {kind!r}")

    return RegionSelection(
        fit_ranges=tuple(fit_ranges),
        exclude_ranges=tuple(exclude_ranges),
        wavelength_unit=str(wavelength_unit),
        wavelength_medium=str(wavelength_medium),
        wavelength_frame=str(table.meta.get("wavelength_frame", "native")),
        output_path=source,
    )


class InteractiveRegionSelector:
    """Matplotlib interface for selecting fit and exclusion intervals.

    Zoom or pan to an interval, choose ``Fit`` or ``Exclude``, enable
    ``Draw regions``, and drag rectangles around the desired lines. Only each
    rectangle's wavelength limits are stored, so its vertical size does not
    affect the fit. Drawing remains enabled until the checkbox is cleared. In
    ``Delete`` mode, rectangles remove intersecting regions. AER catalogue
    ticks identify candidate telluric transitions by default. The displayed
    spectrum is reduced with an extrema-preserving renderer when zoomed out and
    progressively returns to the original sampling while zooming in. Catalogue
    ticks are hidden in the full-spectrum view and progressively revealed at
    narrower wavelength intervals. ``Automatic``
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
        theoretical_spectrum: TheoreticalSpectrum | None = None,
        output_path: str | Path | None = None,
        initial_regions: RegionSelection | None = None,
        title: str | None = None,
        show_telluric_lines: bool = True,
        max_telluric_markers: int = DEFAULT_TELLURIC_MARKER_LIMIT,
        automatic_region_count: int = DEFAULT_AUTOMATIC_REGION_COUNT,
        automatic_region_half_width_pixels: float = (DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS),
        enable_theoretical_controls: bool = False,
        wavelength_frame: RegionWavelengthFrame = "native",
        show: bool = True,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection
            from matplotlib.widgets import (
                Button,
                CheckButtons,
                RadioButtons,
                RectangleSelector,
                TextBox,
            )
        except ImportError as exc:
            raise ImportError(
                "interactive region selection requires Matplotlib; install 'pymolfit[interactive]'"
            ) from exc
        if max_telluric_markers <= 0:
            raise ValueError("max_telluric_markers must be positive")
        if automatic_region_count <= 0:
            raise ValueError("automatic_region_count must be positive")
        if (
            not np.isfinite(automatic_region_half_width_pixels)
            or automatic_region_half_width_pixels <= 0
        ):
            raise ValueError("automatic_region_half_width_pixels must be positive and finite")

        self.wavelength_frame = _normalize_region_wavelength_frame(wavelength_frame)
        self.spectrum = _spectrum_in_region_frame(spectrum, self.wavelength_frame).sorted()
        self.output_path = None if output_path is None else Path(output_path)
        self._plt = plt
        self._poly_collection_type = PolyCollection
        self._mode: RegionKind | Literal["delete"] = "fit"
        self._records: list[tuple[RegionKind, float, float]] = []
        self._stellar_records: list[tuple[RegionKind, float, float]] = []
        self._selection_cache: RegionSelection | None = None
        self._history: list[
            tuple[
                list[tuple[RegionKind, float, float]],
                list[tuple[RegionKind, float, float]],
            ]
        ] = []
        self._patches: list[object] = []
        self._region_labels: list[object] = []
        self._visible_region_count = 0
        self._telluric_marker_count = 0
        self._visible_telluric_marker_count = 0
        self._telluric_marker_error: str | None = None
        self._show_telluric_lines = bool(show_telluric_lines)
        self._telluric_wavelength = np.array([], dtype=float)
        self._telluric_species = np.array([], dtype=str)
        self._telluric_strength = np.array([], dtype=float)
        self._all_telluric_wavelength = np.array([], dtype=float)
        self._all_telluric_species = np.array([], dtype=str)
        self._all_telluric_strength = np.array([], dtype=float)
        self._telluric_overview_count = 0
        self._telluric_marker_artists: list[object] = []
        self._spectrum_sections: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
        self._spectrum_artist: object | None = None
        self._spectrum_data_bounds = (np.nan, np.nan)
        self._view_callback_id: int | None = None
        self._automatic_max_lines = int(max_telluric_markers)
        self._automatic_region_half_width_pixels = float(automatic_region_half_width_pixels)
        self._theoretical_spectrum = theoretical_spectrum
        self._stellar_mask_result: StellarMaskResult | None = None
        self.stellar_rv_box = None
        self.stellar_vsini_box = None
        self.stellar_resolving_power_box = None
        self.stellar_mask_depth_box = None
        self.stellar_padding_box = None
        self.stellar_limb_darkening_box = None
        self.stellar_continuum_window_box = None
        self.stellar_velocity_search_box = None
        self.stellar_alignment_checkbox = None
        self.stellar_update_button = None

        if initial_regions is not None:
            if initial_regions.wavelength_frame != self.wavelength_frame:
                raise ValueError(
                    "initial_regions use wavelength_frame="
                    f"{initial_regions.wavelength_frame!r}, but the selector uses "
                    f"{self.wavelength_frame!r}"
                )
            converted = initial_regions.converted(
                wavelength_unit=self.spectrum.wavelength_unit,
                wavelength_medium=self.spectrum.wavelength_medium,
            )
            self._records.extend(("fit", lower, upper) for lower, upper in converted.fit_ranges)
            self._records.extend(
                ("exclude", lower, upper) for lower, upper in converted.exclude_ranges
            )

        has_stellar_controls = theoretical_spectrum is not None and enable_theoretical_controls
        self.figure, self.axis = plt.subplots(figsize=(17, 8) if has_stellar_controls else (13, 6))
        self.figure.subplots_adjust(
            left=0.08,
            right=0.72 if has_stellar_controls else 0.80,
            top=0.92,
            bottom=0.12,
        )
        self._plot_spectrum()
        data_xlim = self.axis.get_xlim()
        data_ylim = self.axis.get_ylim()
        if show_telluric_lines:
            self._plot_telluric_markers(max_lines=max_telluric_markers)
        frame_label = "Observatory-frame " if self.wavelength_frame == "observatory" else ""
        self.axis.set_xlabel(
            f"{frame_label}{self.spectrum.wavelength_medium} wavelength "
            f"[{self.spectrum.wavelength_unit}]"
        )
        self.axis.set_ylabel("Flux")
        self.axis.set_title(title if title is not None else "Select telluric fitting regions")

        panel_left = 0.76 if has_stellar_controls else 0.83
        panel_width = 0.21 if has_stellar_controls else 0.14
        mode_axis = self.figure.add_axes(
            (
                panel_left,
                0.77 if has_stellar_controls else 0.72,
                panel_width,
                0.14 if has_stellar_controls else 0.17,
            )
        )
        self.mode_buttons = RadioButtons(
            mode_axis,
            ("Fit", "Exclude", "Delete"),
            active=0,
        )
        self.mode_buttons.on_clicked(self._set_mode)

        if has_stellar_controls:
            draw_axis = self.figure.add_axes((panel_left, 0.705, panel_width, 0.045))
            auto_count_axis = self.figure.add_axes((panel_left, 0.645, 0.10, 0.04))
            auto_axis = self.figure.add_axes((0.87, 0.645, 0.10, 0.04))
            undo_axis = self.figure.add_axes((panel_left, 0.25, 0.10, 0.04))
            clear_axis = self.figure.add_axes((0.87, 0.25, 0.10, 0.04))
            save_axis = self.figure.add_axes((panel_left, 0.135, panel_width, 0.04))
            save_name_axis = self.figure.add_axes((panel_left, 0.19, panel_width, 0.04))
            self.figure.text(panel_left, 0.233, "Output filename", fontsize=8)
        else:
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
            "" if has_stellar_controls else "Filename ",
            initial=(
                self.output_path.name if self.output_path is not None else "telluric_regions.ecsv"
            ),
        )
        self.draw_checkbox.on_clicked(self._toggle_rectangle)
        self.auto_count_box.on_submit(self._set_automatic_region_count)
        self.auto_button.on_clicked(self._automatic_event)
        self.undo_button.on_clicked(self._undo_event)
        self.clear_button.on_clicked(self._clear_event)
        self.save_button.on_clicked(self._save_event)
        self.save_name_box.on_submit(self._set_output_filename)

        if has_stellar_controls:
            self._create_stellar_controls(
                theoretical_spectrum,
                text_box_type=TextBox,
                check_buttons_type=CheckButtons,
                button_type=Button,
            )

        self.status_text = self.figure.text(
            panel_left,
            0.105 if has_stellar_controls else 0.205,
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
        if theoretical_spectrum is not None:
            self._replace_stellar_exclusions(
                theoretical_spectrum,
                remember=False,
            )
        self._view_callback_id = self.axis.callbacks.connect(
            "xlim_changed",
            self._on_view_changed,
        )
        self._update_view_artists()
        if show:
            plt.show()

    @property
    def selection(self) -> RegionSelection:
        """Return the current normalized selection."""

        if self._selection_cache is None or self._selection_cache.output_path != self.output_path:
            self._selection_cache = RegionSelection(
                fit_ranges=tuple(
                    (lower, upper) for kind, lower, upper in self._records if kind == "fit"
                ),
                exclude_ranges=tuple(
                    (lower, upper) for kind, lower, upper in self._records if kind == "exclude"
                ),
                wavelength_unit=self.spectrum.wavelength_unit,
                wavelength_medium=self.spectrum.wavelength_medium,
                wavelength_frame=self.wavelength_frame,
                output_path=self.output_path,
            )
        return self._selection_cache

    @property
    def theoretical_spectrum(self) -> TheoreticalSpectrum | None:
        """Return the template with the parameters currently shown in the UI."""

        return self._theoretical_spectrum

    @property
    def stellar_mask_result(self) -> StellarMaskResult | None:
        """Return the most recently prepared theoretical stellar mask."""

        return self._stellar_mask_result

    def update_stellar_mask(self) -> RegionRanges:
        """Rebuild automatic stellar exclusions from the editable controls.

        The existing automatic stellar exclusions are replaced while manually
        selected fit and exclusion regions are preserved. ``mask_padding_kms``
        controls the extra width around each detected stellar feature.
        """

        if self._theoretical_spectrum is None:
            raise ValueError("the selector was not created with theoretical_spectrum")
        if self.stellar_rv_box is None:
            raise ValueError(
                "theoretical controls are disabled; create the selector with "
                "enable_theoretical_controls=True"
            )
        candidate = replace(
            self._theoretical_spectrum,
            radial_velocity_kms=self._stellar_float(
                self.stellar_rv_box,
                "radial velocity",
            ),
            vsini_kms=self._stellar_float(
                self.stellar_vsini_box,
                "v sin(i)",
            ),
            resolving_power=self._stellar_optional_float(
                self.stellar_resolving_power_box,
                "resolving power",
            ),
            mask_depth=self._stellar_auto_or_float(
                self.stellar_mask_depth_box,
                "mask depth",
            ),
            mask_padding_kms=self._stellar_auto_or_float(
                self.stellar_padding_box,
                "mask padding",
            ),
            limb_darkening=self._stellar_float(
                self.stellar_limb_darkening_box,
                "limb darkening",
            ),
            continuum_window_kms=self._stellar_float(
                self.stellar_continuum_window_box,
                "continuum window",
            ),
            velocity_search_kms=self._stellar_float(
                self.stellar_velocity_search_box,
                "velocity search",
            ),
            fit_velocity_offset=bool(self.stellar_alignment_checkbox.get_status()[0]),
        )
        return self._replace_stellar_exclusions(candidate, remember=False)

    def _create_stellar_controls(
        self,
        theoretical_spectrum: TheoreticalSpectrum,
        *,
        text_box_type: object,
        check_buttons_type: object,
        button_type: object,
    ) -> None:
        self.figure.text(
            0.76,
            0.615,
            "Theoretical stellar mask",
            fontsize=9,
            fontweight="bold",
        )
        left = 0.76
        right = 0.875
        width = 0.095
        height = 0.035
        rows = (0.555, 0.495, 0.435, 0.375)

        def text_box(
            x_position: float,
            y_position: float,
            label: str,
            initial: str,
        ) -> object:
            self.figure.text(
                x_position,
                y_position + height + 0.003,
                label,
                fontsize=8,
            )
            return text_box_type(
                self.figure.add_axes((x_position, y_position, width, height)),
                "",
                initial=initial,
            )

        self.stellar_rv_box = text_box(
            left,
            rows[0],
            "Radial velocity [km/s]",
            f"{theoretical_spectrum.radial_velocity_kms:g}",
        )
        self.stellar_vsini_box = text_box(
            right,
            rows[0],
            "v sin(i) [km/s]",
            f"{theoretical_spectrum.vsini_kms:g}",
        )
        self.stellar_resolving_power_box = text_box(
            left,
            rows[1],
            "Resolving power R",
            _optional_parameter_text(theoretical_spectrum.resolving_power),
        )
        self.stellar_mask_depth_box = text_box(
            right,
            rows[1],
            "Minimum mask depth",
            _parameter_text(theoretical_spectrum.mask_depth),
        )
        self.stellar_padding_box = text_box(
            left,
            rows[2],
            "Extra padding [km/s]",
            _parameter_text(theoretical_spectrum.mask_padding_kms),
        )
        self.stellar_limb_darkening_box = text_box(
            right,
            rows[2],
            "Limb darkening",
            f"{theoretical_spectrum.limb_darkening:g}",
        )
        self.stellar_continuum_window_box = text_box(
            left,
            rows[3],
            "Continuum window [km/s]",
            f"{theoretical_spectrum.continuum_window_kms:g}",
        )
        self.stellar_velocity_search_box = text_box(
            right,
            rows[3],
            "Velocity search [km/s]",
            f"{theoretical_spectrum.velocity_search_kms:g}",
        )
        self.stellar_alignment_checkbox = check_buttons_type(
            self.figure.add_axes((left, 0.31, width, 0.045)),
            ("Refine velocity",),
            (theoretical_spectrum.fit_velocity_offset,),
        )
        self.stellar_update_button = button_type(
            self.figure.add_axes((right, 0.31, width, 0.045)),
            "Update mask",
        )
        self.stellar_update_button.on_clicked(self._update_stellar_mask_event)

    def _replace_stellar_exclusions(
        self,
        theoretical_spectrum: TheoreticalSpectrum,
        *,
        remember: bool,
    ) -> RegionRanges:
        result = _prepare_stellar_mask_for_selector(
            self.spectrum,
            theoretical_spectrum,
        )
        selection = result.selection_for_spectrum(self.spectrum)
        new_records = [("exclude", lower, upper) for lower, upper in selection.exclude_ranges]
        if remember:
            self._remember()
        old_stellar = set(self._stellar_records)
        self._records = [record for record in self._records if record not in old_stellar]
        self._records.extend(new_records)
        self._stellar_records = new_records
        self._invalidate_selection()
        self._theoretical_spectrum = theoretical_spectrum
        self._stellar_mask_result = result
        if self._view_callback_id is not None:
            self._redraw_regions()
        return selection.exclude_ranges

    @staticmethod
    def _stellar_float(widget: object, label: str) -> float:
        value = getattr(widget, "text", "")
        try:
            parsed = float(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a finite number") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{label} must be a finite number")
        return parsed

    @staticmethod
    def _stellar_optional_float(widget: object, label: str) -> float | None:
        value = str(getattr(widget, "text", "")).strip().lower()
        if value in {"", "auto", "none"}:
            return None
        return InteractiveRegionSelector._stellar_float(widget, label)

    @staticmethod
    def _stellar_auto_or_float(widget: object, label: str) -> str | float:
        value = str(getattr(widget, "text", "")).strip().lower()
        if value == "auto":
            return "auto"
        return InteractiveRegionSelector._stellar_float(widget, label)

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
        self._invalidate_selection()
        self._redraw_regions()

    def delete_regions(self, lower: float, upper: float) -> None:
        """Delete every selected interval overlapping ``lower`` to ``upper``."""

        lower, upper = self._bounded_interval(lower, upper)
        retained = [record for record in self._records if record[2] < lower or record[1] > upper]
        if len(retained) == len(self._records):
            return
        self._remember()
        self._records = retained
        self._stellar_records = [record for record in self._stellar_records if record in retained]
        self._invalidate_selection()
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
            raise ValueError("no covered positive-strength AER transitions were found")
        self._remember()
        self._records.extend(("fit", lower, upper) for lower, upper in ranges)
        self._invalidate_selection()
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
        self._records, self._stellar_records = self._history.pop()
        self._invalidate_selection()
        self._redraw_regions()

    def clear(self) -> None:
        """Remove every fit and exclusion interval."""

        if not self._records:
            return
        self._remember()
        self._records = []
        self._stellar_records = []
        self._invalidate_selection()
        self._redraw_regions()

    def save(self, path: str | Path | None = None) -> Path:
        """Save the current selection to ECSV."""

        if path is None:
            destination = self._output_path_from_filename(self.save_name_box.text)
        else:
            destination = Path(path)
        if destination is None:
            raise ValueError(
                "no region output path was provided; call save(path) or pass "
                "output_path to select_telluric_regions"
            )
        self.output_path = destination
        self._invalidate_selection()
        written = self.selection.write(destination)
        self._update_status(f"Saved {written.name}")
        return written

    def close(self) -> None:
        """Close the selector figure."""

        if self._view_callback_id is not None:
            self.axis.callbacks.disconnect(self._view_callback_id)
            self._view_callback_id = None
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
                breaks = np.flatnonzero(~np.isfinite(spacing) | (spacing > gap_limit)) + 1
                sections = tuple(np.split(indices, breaks))
            else:
                sections = (indices,)
        self._spectrum_sections = tuple(
            (
                np.asarray(wavelength[section], dtype=float),
                np.asarray(flux[section], dtype=float),
            )
            for section in sections
            if section.size
        )
        finite_wavelength = wavelength[indices]
        self._spectrum_data_bounds = (
            float(np.nanmin(finite_wavelength)),
            float(np.nanmax(finite_wavelength)),
        )
        (self._spectrum_artist,) = self.axis.plot(
            [],
            [],
            color="#222222",
            linewidth=0.75,
        )
        self._update_spectrum_artist(self._spectrum_data_bounds)
        self.axis.relim()
        self.axis.autoscale_view()

    def _plot_telluric_markers(self, *, max_lines: int) -> None:
        try:
            wavelength, species, strength = _aer_catalog_for_spectrum(
                self.spectrum,
                max_lines=None,
            )
        except Exception as exc:
            self._telluric_marker_error = str(exc)
            warnings.warn(
                f"AER telluric markers could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._all_telluric_wavelength = wavelength
        self._all_telluric_species = species
        self._all_telluric_strength = strength
        self._telluric_marker_count = int(wavelength.size)
        if wavelength.size > max_lines:
            score = _expected_aer_absorption_score(strength, species)
            finite_score = np.nan_to_num(score, nan=-np.inf)
            selected = np.argpartition(finite_score, -max_lines)[-max_lines:]
            wavelength = wavelength[selected]
            species = species[selected]
            strength = strength[selected]
        self._telluric_wavelength = wavelength
        self._telluric_species = species
        self._telluric_strength = strength
        self._telluric_overview_count = int(wavelength.size)

    def _on_view_changed(self, _axis: object) -> None:
        self._update_view_artists()

    def _update_view_artists(self) -> None:
        x_limits = self.axis.get_xlim()
        self._update_spectrum_artist(x_limits)
        self._update_telluric_marker_artists(x_limits)
        self._update_region_artists(x_limits)
        self._refresh_legend()
        if hasattr(self, "status_text"):
            self._update_status()
        self.figure.canvas.draw_idle()

    def _update_spectrum_artist(self, x_limits: tuple[float, float]) -> None:
        if self._spectrum_artist is None:
            return
        lower, upper = sorted((float(x_limits[0]), float(x_limits[1])))
        visible_sections: list[tuple[np.ndarray, np.ndarray]] = []
        visible_counts: list[int] = []
        for wavelength, flux in self._spectrum_sections:
            if upper < wavelength[0] or lower > wavelength[-1]:
                continue
            start = max(0, int(np.searchsorted(wavelength, lower, side="left")) - 1)
            stop = min(
                wavelength.size,
                int(np.searchsorted(wavelength, upper, side="right")) + 1,
            )
            if stop <= start:
                continue
            visible_sections.append((wavelength[start:stop], flux[start:stop]))
            visible_counts.append(stop - start)

        if not visible_sections:
            self._spectrum_artist.set_data([], [])  # type: ignore[attr-defined]
            return

        point_budget = max(
            _SELECTOR_MIN_DISPLAY_POINTS,
            round(float(self.axis.bbox.width) * _SELECTOR_POINTS_PER_SCREEN_PIXEL),
        )
        total_visible = sum(visible_counts)
        display_wavelength: list[np.ndarray] = []
        display_flux: list[np.ndarray] = []
        for index, ((wavelength, flux), count) in enumerate(
            zip(visible_sections, visible_counts, strict=True)
        ):
            section_budget = max(4, round(point_budget * count / total_visible))
            selected = _extrema_preserving_indices(
                wavelength,
                flux,
                max_points=section_budget,
            )
            if index:
                display_wavelength.append(np.array([np.nan]))
                display_flux.append(np.array([np.nan]))
            display_wavelength.append(wavelength[selected])
            display_flux.append(flux[selected])
        self._spectrum_artist.set_data(  # type: ignore[attr-defined]
            np.concatenate(display_wavelength),
            np.concatenate(display_flux),
        )

    def _update_telluric_marker_artists(self, x_limits: tuple[float, float]) -> None:
        for artist in self._telluric_marker_artists:
            artist.remove()  # type: ignore[attr-defined]
        self._telluric_marker_artists = []
        self._visible_telluric_marker_count = 0
        if not self._show_telluric_lines or self._telluric_marker_error is not None:
            return

        data_lower, data_upper = self._spectrum_data_bounds
        data_span = data_upper - data_lower
        lower = max(data_lower, min(float(x_limits[0]), float(x_limits[1])))
        upper = min(data_upper, max(float(x_limits[0]), float(x_limits[1])))
        if not np.isfinite(data_span) or data_span <= 0 or upper <= lower:
            return
        view_fraction = min(1.0, (upper - lower) / data_span)
        if view_fraction >= _SELECTOR_HIDE_MARKERS_VIEW_FRACTION:
            return

        if view_fraction <= _SELECTOR_ALL_MARKERS_VIEW_FRACTION:
            visible = (self._all_telluric_wavelength >= lower) & (
                self._all_telluric_wavelength <= upper
            )
            wavelength = self._all_telluric_wavelength[visible]
            species = self._all_telluric_species[visible]
        else:
            visible = (self._telluric_wavelength >= lower) & (self._telluric_wavelength <= upper)
            wavelength = self._telluric_wavelength[visible]
            species = self._telluric_species[visible]
            strength = self._telluric_strength[visible]
            if wavelength.size:
                progress = np.log(_SELECTOR_HIDE_MARKERS_VIEW_FRACTION / view_fraction) / np.log(
                    _SELECTOR_HIDE_MARKERS_VIEW_FRACTION / _SELECTOR_ALL_MARKERS_VIEW_FRACTION
                )
                progress = float(np.clip(progress, 0.0, 1.0))
                marker_budget = max(
                    1,
                    round(
                        float(self.axis.bbox.width) * _SELECTOR_MARKERS_PER_SCREEN_PIXEL * progress
                    ),
                )
                if wavelength.size > marker_budget:
                    score = _expected_aer_absorption_score(strength, species)
                    finite_score = np.nan_to_num(score, nan=-np.inf)
                    selected = np.argpartition(finite_score, -marker_budget)[-marker_budget:]
                    wavelength = wavelength[selected]
                    species = species[selected]

        self._visible_telluric_marker_count = int(wavelength.size)
        for name in sorted(set(species.tolist())):
            selected = species == name
            artist = self.axis.vlines(
                wavelength[selected],
                0.91,
                0.99,
                transform=self.axis.get_xaxis_transform(),
                color=_SPECIES_COLORS.get(name, "#999999"),
                alpha=0.62,
                linewidth=0.55,
                label=f"AER {name}",
            )
            self._telluric_marker_artists.append(artist)

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
        finite = self.spectrum.wavelength[np.isfinite(self.spectrum.wavelength)]
        minimum = float(np.nanmin(finite))
        maximum = float(np.nanmax(finite))
        lower = max(lower, minimum)
        upper = min(upper, maximum)
        if upper <= lower:
            raise ValueError("region must have positive width inside the spectrum")
        return lower, upper

    def _remember(self) -> None:
        self._history.append((list(self._records), list(self._stellar_records)))

    def _invalidate_selection(self) -> None:
        self._selection_cache = None

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
        color = _REGION_COLORS.get(self._mode, "#666666")
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
        self._invalidate_selection()
        self._update_status(f"Output filename: {destination.name}")

    def _set_automatic_region_count(self, value: str) -> None:
        try:
            count = self._parse_automatic_region_count(value)
        except ValueError as exc:
            self._update_status(str(exc))
            return
        self._update_status(
            f"Automatic selection will use the {count} strongest expected telluric lines."
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
        directory = self.output_path.parent if self.output_path is not None else Path.cwd()
        return directory / name

    def _undo_event(self, _event: object) -> None:
        self.undo()

    def _automatic_event(self, _event: object) -> None:
        try:
            self.add_automatic_fit_regions()
        except (OSError, ValueError) as exc:
            self._update_status(str(exc))

    def _update_stellar_mask_event(self, _event: object) -> None:
        try:
            ranges = self.update_stellar_mask()
        except (OSError, ValueError) as exc:
            self._update_status(str(exc))
            return
        self._update_status(f"Updated theoretical stellar mask: {len(ranges)} exclusions.")

    def _clear_event(self, _event: object) -> None:
        self.clear()

    def _save_event(self, _event: object) -> None:
        try:
            self.save()
        except (OSError, ValueError) as exc:
            self._update_status(str(exc))

    def _clear_region_artists(self) -> None:
        for patch in self._patches:
            patch.remove()  # type: ignore[attr-defined]
        self._patches = []
        for label in self._region_labels:
            label.remove()  # type: ignore[attr-defined]
        self._region_labels = []

    def _update_region_artists(self, x_limits: tuple[float, float]) -> None:
        """Render only viewport regions, batching every type into one artist."""

        self._clear_region_artists()
        selection = self.selection
        data_lower, data_upper = self._spectrum_data_bounds
        data_span = data_upper - data_lower
        lower = max(data_lower, min(float(x_limits[0]), float(x_limits[1])))
        upper = min(data_upper, max(float(x_limits[0]), float(x_limits[1])))
        if not np.isfinite(data_span) or data_span <= 0 or upper <= lower:
            self._visible_region_count = 0
            return

        numbered_regions = tuple(
            (index, kind, region_lower, region_upper)
            for index, (kind, region_lower, region_upper) in enumerate(
                tuple(("fit", start, stop) for start, stop in selection.fit_ranges)
                + tuple(("exclude", start, stop) for start, stop in selection.exclude_ranges),
                start=1,
            )
        )
        visible_regions = tuple(
            record for record in numbered_regions if record[3] >= lower and record[2] <= upper
        )
        self._visible_region_count = len(visible_regions)

        transform = self.axis.get_xaxis_transform()
        for kind in ("fit", "exclude"):
            ranges = tuple(
                (region_lower, region_upper)
                for _, region_kind, region_lower, region_upper in visible_regions
                if region_kind == kind
            )
            if not ranges:
                continue
            vertices = [
                (
                    (region_lower, 0.0),
                    (region_lower, 1.0),
                    (region_upper, 1.0),
                    (region_upper, 0.0),
                )
                for region_lower, region_upper in ranges
            ]
            collection = self._poly_collection_type(
                vertices,
                closed=True,
                facecolors=_REGION_COLORS[kind],
                edgecolors=_REGION_COLORS[kind],
                alpha=0.22,
                linewidths=1.0,
                transform=transform,
                label="Fit region" if kind == "fit" else "Excluded region",
                zorder=1,
            )
            self.axis.add_collection(collection, autolim=False)
            self._patches.append(collection)

        view_fraction = min(1.0, (upper - lower) / data_span)
        label_budget = max(
            1,
            int(float(self.axis.bbox.width) / _SELECTOR_REGION_LABEL_SPACING_PIXELS),
        )
        if (
            view_fraction <= _SELECTOR_HIDE_REGION_LABELS_VIEW_FRACTION
            and len(visible_regions) <= label_budget
        ):
            for region_number, kind, region_lower, region_upper in visible_regions:
                self._region_labels.append(
                    self.axis.text(
                        0.5 * (region_lower + region_upper),
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

    def _redraw_regions(self) -> None:
        self._update_region_artists(self.axis.get_xlim())
        self._refresh_legend()
        self._update_status()
        self.figure.canvas.draw_idle()

    def _refresh_legend(self) -> None:
        handles, labels = self.axis.get_legend_handles_labels()
        existing_legend = self.axis.get_legend()
        if existing_legend is not None:
            existing_legend.remove()
        if labels:
            self.axis.legend(handles, labels, loc="best")

    def _update_status(self, message: str | None = None) -> None:
        selection = self.selection
        display_regions = tuple(
            ("fit", lower, upper) for lower, upper in selection.fit_ranges
        ) + tuple(("exclude", lower, upper) for lower, upper in selection.exclude_ranges)
        lines = []
        if message is not None:
            lines.append(message)
        lines.extend(
            (
                f"Mode: {self._mode}",
                ("Drawing: on" if self.rectangle_selector.active else "Drawing: off"),
                f"Regions in memory: {len(display_regions)}",
            )
        )
        if self._telluric_marker_count:
            lines.append(f"AER markers shown: {self._visible_telluric_marker_count}")
            lines.append(f"AER lines in spectrum: {self._telluric_marker_count}")
        elif self._telluric_marker_error is not None:
            lines.append("AER markers unavailable")
        if self._stellar_mask_result is not None:
            diagnostics = self._stellar_mask_result.diagnostics
            lines.append(
                "Stellar exclusions: "
                f"{len(self._stellar_records)} "
                f"(padding {diagnostics.get('mask_padding_kms', 0.0):g} km/s)"
            )
        display_limit = 0 if self._theoretical_spectrum is not None else 5
        for index, (kind, lower, upper) in enumerate(
            display_regions[:display_limit],
            start=1,
        ):
            lines.append(f"R{index} {kind}:")
            lines.append(f"  {lower:.6g} - {upper:.6g}")
        if display_limit and len(display_regions) > display_limit:
            lines.append(f"... plus {len(display_regions) - display_limit} more")
        if display_regions and self._theoretical_spectrum is None:
            lines.append("Save All writes every region.")
        self.status_text.set_text("\n".join(lines))
        self.figure.canvas.draw_idle()


def select_telluric_regions(
    spectrum: Spectrum | None = None,
    *,
    theoretical_spectrum: TheoreticalSpectrum | None = None,
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
    automatic_region_half_width_pixels: float = (DEFAULT_AUTOMATIC_REGION_HALF_WIDTH_PIXELS),
    enable_theoretical_controls: bool = False,
    wavelength_frame: RegionWavelengthFrame | None = None,
    reuse_existing: bool = True,
    show: bool = True,
) -> InteractiveRegionSelector | RegionSelection:
    """Open an interactive spectrum window for selecting fit regions.

    Supply either a :class:`Spectrum` or wavelength/flux arrays. Zoom or pan,
    choose fit/exclusion mode, enable ``Draw regions``, and drag rectangles;
    their horizontal limits become numbered regions. Drawing remains enabled
    until its checkbox is cleared. AER telluric-line ticks are enabled by
    default but hidden in the full-spectrum view. They are progressively
    revealed as the wavelength view narrows, with every transition in the local
    catalogue window shown at close zoom. The spectrum itself uses
    extrema-preserving display reduction when zoomed out and returns to every
    original sample at close zoom; this never changes fitting data or saved
    region coordinates. Enter a line count and press ``Automatic`` to propose
    fit windows around that many transitions with the greatest expected
    atmospheric absorption. This ranking combines each AER line intensity with
    a representative atmospheric column for its molecule. Windows in detector
    gaps are skipped and overlapping windows are merged. The proposals remain
    fully editable. Edit the filename field if needed, then press ``Save All``
    to write every region together as ECSV. In Jupyter, enable
    ``%matplotlib widget`` and install ``pymolfit[interactive]``. The returned
    selector exposes ``selection`` and ``save()`` for notebook and scripted use.

    Pass ``theoretical_spectrum=TheoreticalSpectrum(...)`` to generate stellar
    exclusion regions before the window opens. Set
    ``enable_theoretical_controls=True`` to expose the template's radial
    velocity, projected rotation, resolution, mask depth, mask padding, limb
    darkening, continuum window, and residual-alignment controls. Edit values
    and press ``Update mask`` to replace the automatic stellar exclusions
    without changing selected telluric fit regions. ``Save All`` writes fit
    and exclusion intervals together in the same ECSV file, which can be
    passed directly to :func:`pymolfit.correct` as ``region_file``.

    Set ``wavelength_frame="observatory"`` when creating a reusable telluric
    region file for a barycentric or heliocentric time series. PyMolFit then
    displays the spectrum in observatory-frame vacuum coordinates and records
    that frame in the ECSV metadata. During correction, the same telluric
    intervals can be applied to every exposure without manually shifting the
    file. Static stellar exclusions are not generally reusable in this frame;
    pass a theoretical spectrum to each correction to construct those masks
    for the individual exposure.

    When ``output_path`` already exists and ``reuse_existing`` is ``True``, the
    saved :class:`RegionSelection` is loaded and returned without opening a new
    selector window. Set ``reuse_existing=False`` to open the interface with
    the saved regions loaded for editing. Explicit ``initial_regions`` cannot
    be combined with automatic reuse because one source would otherwise be
    silently ignored.
    """

    has_arrays = wavelength is not None or flux is not None
    if spectrum is not None and has_arrays:
        raise ValueError("provide either spectrum or wavelength/flux arrays, not both")
    if spectrum is None:
        if wavelength is None or flux is None:
            raise ValueError("provide spectrum or both wavelength and flux arrays")
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
        if (
            wavelength_frame is not None
            and selection.wavelength_frame != _normalize_region_wavelength_frame(wavelength_frame)
        ):
            raise ValueError(
                f"existing region file uses wavelength_frame={selection.wavelength_frame!r}, "
                f"not {wavelength_frame!r}"
            )
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
    resolved_frame = (
        initial_regions.wavelength_frame
        if wavelength_frame is None and initial_regions is not None
        else _normalize_region_wavelength_frame(wavelength_frame or "native")
    )
    return InteractiveRegionSelector(
        spectrum,
        theoretical_spectrum=theoretical_spectrum,
        output_path=output_path,
        initial_regions=initial_regions,
        title=title,
        show_telluric_lines=show_telluric_lines,
        max_telluric_markers=max_telluric_markers,
        automatic_region_count=automatic_region_count,
        automatic_region_half_width_pixels=automatic_region_half_width_pixels,
        enable_theoretical_controls=enable_theoretical_controls,
        wavelength_frame=resolved_frame,
        show=show,
    )


def _extrema_preserving_indices(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    max_points: int,
) -> np.ndarray:
    """Select ordered per-bin extrema without altering the source arrays."""

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if wavelength.ndim != 1 or flux.shape != wavelength.shape:
        raise ValueError("wavelength and flux must be matching one-dimensional arrays")
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    if wavelength.size <= max_points:
        return np.arange(wavelength.size, dtype=int)
    if max_points == 2:
        return np.array([0, wavelength.size - 1], dtype=int)
    if max_points == 3:
        interior = 1 + int(np.argmax(np.abs(flux[1:-1] - np.nanmedian(flux))))
        return np.array([0, interior, wavelength.size - 1], dtype=int)

    bin_count = max(1, (max_points - 2) // 2)
    edges = np.linspace(wavelength[0], wavelength[-1], bin_count + 1)
    starts = np.searchsorted(wavelength, edges[:-1], side="left")
    stops = np.searchsorted(wavelength, edges[1:], side="left")
    stops[-1] = wavelength.size
    selected = [0]
    for start, stop in zip(starts, stops, strict=True):
        if stop <= start:
            continue
        local_flux = flux[start:stop]
        minimum = int(start + np.argmin(local_flux))
        maximum = int(start + np.argmax(local_flux))
        selected.extend(sorted((minimum, maximum)))
    selected.append(wavelength.size - 1)
    return np.unique(np.asarray(selected, dtype=int))


def _parameter_text(value: str | float) -> str:
    return value if isinstance(value, str) else f"{value:g}"


def _optional_parameter_text(value: float | None) -> str:
    return "auto" if value is None else f"{value:g}"


def _prepare_stellar_mask_for_selector(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
) -> StellarMaskResult:
    """Prepare a template mask through the correction workflow's frame path."""

    # Import lazily because workflow imports this module to resolve region
    # files. Calling the shared helpers here keeps selector and correction
    # frame/resolution handling mathematically identical.
    from .workflow import (
        _estimate_lsf_sigma_from_resolving_power,
        _load_fits_header_if_available,
        _spectrum_to_observatory_vacuum,
        _stellar_template_frame_correction_factor,
    )

    source = spectrum.meta.get("source")
    header = None
    if source:
        hdu_value = spectrum.meta.get("hdu", 1)
        try:
            hdu = int(hdu_value)
        except (TypeError, ValueError):
            hdu = 1
        header = _load_fits_header_if_available(str(source), None, hdu=hdu)
    observatory_spectrum = _spectrum_to_observatory_vacuum(spectrum, header)
    resolution = _estimate_lsf_sigma_from_resolving_power(
        observatory_spectrum,
        header,
    )
    resolving_power = None if resolution is None else float(resolution["resolving_power"])
    return theoretical_spectrum.build_mask(
        observatory_spectrum,
        frame_correction_factor=_stellar_template_frame_correction_factor(
            observatory_spectrum,
            header,
        ),
        resolving_power=resolving_power,
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
    max_lines: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AER wavelengths, species, and strengths in display coordinates."""

    finite = spectrum.wavelength[np.isfinite(spectrum.wavelength)]
    if finite.size == 0:
        raise ValueError("the spectrum contains no finite wavelengths")
    return _aer_catalog_for_display_interval(
        spectrum,
        lower=float(np.nanmin(finite)),
        upper=float(np.nanmax(finite)),
        max_lines=max_lines,
    )


def _aer_catalog_for_display_interval(
    spectrum: Spectrum,
    *,
    lower: float,
    upper: float,
    max_lines: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AER transitions inside one interval of the displayed coordinates."""

    lower, upper = sorted((float(lower), float(upper)))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("marker interval must have finite, positive, distinct bounds")
    scale = wavelength_scale_to_micron(spectrum.wavelength_unit)
    input_vacuum_micron = np.asarray([lower, upper], dtype=float) * scale
    if spectrum.wavelength_medium == "air":
        input_vacuum_micron = air_to_vacuum_wavelength(
            input_vacuum_micron,
            unit="micron",
        )
    erf_factor = _selector_erf_factor(spectrum)
    observatory_wavelength = input_vacuum_micron / erf_factor

    artifact = load_aer_line_window(
        wavelength_min_micron=float(np.nanmin(observatory_wavelength)),
        wavelength_max_micron=float(np.nanmax(observatory_wavelength)),
        max_lines=max_lines,
    )
    line_list = artifact.line_list
    marker_micron = np.asarray(line_list.wavelength, dtype=float) * erf_factor
    if spectrum.wavelength_medium == "air":
        marker_micron = vacuum_to_air_wavelength(marker_micron, unit="micron")
    marker_wavelength = marker_micron / wavelength_scale_to_micron(spectrum.wavelength_unit)
    species = np.asarray(line_list.species, dtype=str)
    strength = np.asarray(line_list.strength, dtype=float)

    keep = (
        np.isfinite(marker_wavelength) & (marker_wavelength >= lower) & (marker_wavelength <= upper)
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
        choose_right = np.abs(sampled_wavelength[right] - marker_wavelength) < np.abs(
            sampled_wavelength[left] - marker_wavelength
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
    ranking = eligible_indices[np.argsort(-absorption_score[eligible_indices], kind="stable")][
        :count
    ]

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
        grouped_wavelengths.extend(
            [
                np.asarray(
                    spectrum.wavelength[valid & (group_id == value)],
                    dtype=float,
                )
                for value in np.unique(group_id[valid])
            ]
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
            section for section in np.split(wavelength, boundaries) if section.size >= 2
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
    if bool(spectrum.meta.get("observatory_frame_correction", False)):
        return 1.0
    header = _selector_header(spectrum)
    if header is None:
        return 1.0

    # Imported lazily because workflow imports the region-file API.
    from .workflow import _spectral_frame_velocity_km_s

    frame_velocity = _spectral_frame_velocity_km_s(header)
    if frame_velocity is None:
        return 1.0
    _, velocity_km_s = frame_velocity
    return (1.0 + 1.55e-8) * (1.0 + velocity_km_s / (SPEED_OF_LIGHT_M_PER_S / 1_000.0))


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


def _spectrum_in_region_frame(
    spectrum: Spectrum,
    wavelength_frame: RegionWavelengthFrame,
) -> Spectrum:
    """Return the spectrum in coordinates used by the selector and its file."""

    if wavelength_frame == "native":
        return spectrum

    # Imported lazily because workflow imports the region-file API.
    from .workflow import _spectrum_to_observatory_vacuum

    return _spectrum_to_observatory_vacuum(spectrum, _selector_header(spectrum))


def _normalize_region_wavelength_frame(value: str) -> RegionWavelengthFrame:
    key = str(value).strip().lower().replace("-", "_")
    aliases: dict[str, RegionWavelengthFrame] = {
        "native": "native",
        "input": "native",
        "observatory": "observatory",
        "observer": "observatory",
        "topocentric": "observatory",
        "topocent": "observatory",
        "lab": "observatory",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("wavelength_frame must be 'native' or 'observatory'") from exc


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
