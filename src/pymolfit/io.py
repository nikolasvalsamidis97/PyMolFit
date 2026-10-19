from __future__ import annotations

import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .errors import ProductFormatError, SpectrumFormatError, WavelengthMetadataError
from .observation import Observation
from .provenance import file_sha256
from .spectrum import Spectrum

if TYPE_CHECKING:
    from .fit import TelluricFitResult

WAVELENGTH_COLUMNS = (
    "wavelength",
    "wave",
    "lambda",
    "lam",
    "wl",
    "wavelength_micron",
    "wavelength_nm",
    "wavelength_angstrom",
    "wavelength_angstroms",
    "awav",
    "wave_air",
    "wavelength_air",
    "wave_vac",
    "wave_vacuum",
    "wavelength_vacuum",
)
FLUX_COLUMNS = ("flux", "flam", "fnu", "spectrum", "spec", "data", "science")
UNCERTAINTY_COLUMNS = ("uncertainty", "error", "err", "sigma", "noise", "flux_err", "flux_error")
QUALITY_COLUMNS = ("qual", "quality", "dq", "data_quality", "bpm", "bad_pixel_mask", "mask")
VALID_MASK_COLUMNS = ("valid", "input_mask", "good_pixel", "good")
ORDER_COLUMNS = ("order", "spectral_order", "echelle_order", "physical_group")
DETECTOR_COLUMNS = ("detec", "detector", "chip", "extension")


def _native_order_prefixes(names) -> tuple[str, ...]:
    """Recognize paired CRIRES order/suborder WL+SPEC columns, case-insensitively."""
    lowered = {str(name).lower() for name in names}
    prefixes = []
    for name in lowered:
        parts = name.split("_")
        if len(parts) != 3 or not all(part.isdigit() for part in parts[:2]) or parts[2] != "wl":
            continue
        prefix = "_".join(parts[:2])
        if prefix + "_spec" in lowered:
            prefixes.append(prefix)
    return tuple(sorted(prefixes))


def load_spectrum(
    path: str | Path,
    *,
    format: str | None = None,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    hdu: int | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: str | None = None,
    image_index: int | None = None,
    save_header: bool = True,
    header_output_path: str | Path | None = None,
) -> Spectrum:
    """Load a spectrum from a FITS or text-based file.

    With ``hdu=None``, search for a unique spectral table or wavelength-calibrated
    image, excluding labelled auxiliary extensions. Multiple candidates require
    an explicit ``hdu``; orders are not combined automatically. An explicit HDU
    is read exactly as requested, without falling back to another extension.

    Loading a FITS file saves its complete, formatted header to a neighboring
    ``*.header.txt`` file by default. The text file contains the primary header
    and every extension header, with aligned keywords, values, and comments.
    Set ``save_header=False`` to disable this sidecar file, or provide
    ``header_output_path`` to choose its location.

    ``wavelength_medium`` states how the input wavelength values are already
    defined; loading does not convert them between air and vacuum. For FITS
    input, ``None`` infers the medium from the selected wavelength column's
    ``TUCDn``/``TTYPEn`` metadata and other unambiguous header declarations.
    PyMolFit stops if a FITS file does not declare a reliable convention.
    FITS observing metadata are attached as ``spectrum.observation.metadata``;
    the original header keywords are retained without inventing missing values.

    :param path: Input FITS, text, CSV or ECSV file.
    :param format: Explicit file format; None infers from the filename.
    :param wavelength_col: Wavelength column name or zero-based index; None detects it.
    :param flux_col: Flux column name or zero-based index; None detects it.
    :param uncertainty_col: Flux standard-deviation column name or zero-based index;
        None detects it if available.
    :param hdu: FITS extension index; None searches for a unique spectral extension.
    :param wavelength_unit: Fallback wavelength unit where metadata are unavailable.
    :param wavelength_medium: 'air' or 'vacuum'; None infers reliable FITS metadata
        or uses the text-loader convention. This is a label, not a conversion.
    :param image_index: Row to extract from a supported 2D FITS image.
    :param save_header: Save a readable FITS-header sidecar (default True).
    :param header_output_path: Optional sidecar destination; otherwise beside the FITS.
    :return: A Spectrum with native samples and available observing metadata.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"spectrum file does not exist: {path}")
    chosen_format = infer_spectrum_format(path, format)
    if chosen_format in {"fits", "fit", "fz"}:
        with fits.open(path) as hdul:
            hdu = _select_spectrum_hdu(
                hdul,
                hdu,
                wavelength_col=wavelength_col,
                flux_col=flux_col,
                uncertainty_col=uncertainty_col,
                image_index=image_index,
            )
    resolved_wavelength_medium = wavelength_medium
    wavelength_medium_source = "user"
    if resolved_wavelength_medium is None:
        if chosen_format in {"fits", "fit", "fz"}:
            resolved_wavelength_medium = infer_fits_wavelength_medium(
                path,
                hdu=hdu,
                wavelength_col=wavelength_col,
            )
            if resolved_wavelength_medium is None:
                raise WavelengthMetadataError(
                    "PyMolFit could not determine whether the FITS wavelength "
                    "column uses air or vacuum wavelengths. Pass "
                    "wavelength_medium='air' or wavelength_medium='vacuum'."
                )
            wavelength_medium_source = "fits_header"
        else:
            resolved_wavelength_medium = "vacuum"
            wavelength_medium_source = "default"

    if chosen_format in {"txt", "dat", "csv", "ascii", "ecsv"}:
        spectrum = _load_ascii(
            path,
            format=chosen_format,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            wavelength_unit=wavelength_unit,
            wavelength_medium=resolved_wavelength_medium,
        )
    elif chosen_format in {"fits", "fit", "fz"}:
        spectrum = _load_fits(
            path,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            hdu=hdu,
            wavelength_unit=wavelength_unit,
            wavelength_medium=resolved_wavelength_medium,
            image_index=image_index,
        )
    else:
        raise SpectrumFormatError(f"unsupported spectrum format: {chosen_format}")
    header_text_path = None
    if save_header and chosen_format in {"fits", "fit", "fz"}:
        try:
            header_text_path = save_fits_header_txt(path, header_output_path)
        except OSError as exc:
            if header_output_path is not None:
                raise
            warnings.warn(
                f"could not save automatic FITS header sidecar beside {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    loaded_meta = {
        **dict(spectrum.meta),
        "name": str(path.stem),
        "source": str(path.resolve()),
        "source_file_sha256": file_sha256(path),
        "wavelength_medium_source": wavelength_medium_source,
    }
    if header_text_path is not None:
        loaded_meta["header_text_path"] = str(header_text_path.resolve())

    return replace(
        spectrum,
        meta=loaded_meta,
    )


def infer_fits_wavelength_medium(
    path: str | Path,
    *,
    hdu: int | None = None,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    image_index: int | None = None,
) -> str | None:
    """Infer air or vacuum wavelengths for the selected FITS spectral column.

    The FITS table column selected by ``wavelength_col`` is matched to its
    corresponding ``TUCDn`` card. Under the IVOA/ESO convention,
    ``em.wl;obs.atmos`` means air wavelength, while ``em.wl`` with qualifiers
    such as ``meta.main`` means vacuum wavelength. Explicit FITS declarations
    and spectral WCS codes are also honored. Conflicting declarations raise
    ``ValueError`` rather than silently choosing one.
    """

    source = Path(path)
    with fits.open(source) as hdul:
        target_index = _select_spectrum_hdu(
            hdul,
            hdu,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            image_index=image_index,
        )
        target_hdu = hdul[target_index]

        header = dict(hdul[0].header)
        header.update(dict(target_hdu.header))
    return infer_wavelength_medium_from_header(
        header,
        wavelength_col=wavelength_col,
    )


def infer_wavelength_medium_from_header(
    header: Mapping[str, object] | None,
    *,
    wavelength_col: int | str | None = None,
) -> str | None:
    """Return an unambiguous air/vacuum declaration from FITS metadata."""

    if header is None:
        return None

    declarations: set[str] = set()
    selected_index = _fits_wavelength_column_index(header, wavelength_col)
    selected_name = (
        ""
        if selected_index is None
        else str(header.get(f"TTYPE{selected_index}", "")).strip().upper()
    )

    if selected_index is not None:
        ucd = str(header.get(f"TUCD{selected_index}", "")).strip().lower()
        ucd_tokens = {token.strip() for token in ucd.split(";") if token.strip()}
        if "em.wl" in ucd_tokens:
            declarations.add("air" if "obs.atmos" in ucd_tokens else "vacuum")

    if selected_name in {"AWAV", "WAVE_AIR", "WAVELENGTH_AIR"}:
        declarations.add("air")
    elif selected_name in {"WAVE_VAC", "WAVE_VACUUM", "WAVELENGTH_VACUUM"} or (
        str(header.get("INSTRUME", "")).strip().upper() == "ESPRESSO" and selected_name == "WAVE"
    ):
        declarations.add("vacuum")

    explicit_keys = {
        "AIRORVAC",
        "AIRVAC",
        "WAVEMED",
        "WAVE_MED",
        "WAVEMEDIUM",
        "PYMOLFIT WAVE",
        "GENMOLFIT WAVE",
        "VACUUM",
    }
    for raw_key, value in header.items():
        key = str(raw_key).strip().upper()
        if key.startswith("HIERARCH "):
            key = key.removeprefix("HIERARCH ").strip()

        if key in explicit_keys:
            if key == "VACUUM":
                text = str(value).strip().lower()
                if value is True or text in {"1", "true", "t", "yes", "y"}:
                    declarations.add("vacuum")
                    continue
                if value is False or text in {"0", "false", "f", "no", "n"}:
                    declarations.add("air")
                    continue
            text = str(value).strip().lower()
            tokens = set(text.replace("-", " ").replace("_", " ").replace("/", " ").split())
            if "vacuum" in tokens or "vac" in tokens:
                declarations.add("vacuum")
            if "air" in tokens:
                declarations.add("air")

        if key.startswith(("CTYPE", "TCTYP")):
            spectral_code = str(value).strip().upper().split("-", 1)[0]
            if spectral_code == "AWAV":
                declarations.add("air")
            elif spectral_code == "WAVE":
                declarations.add("vacuum")

    if len(declarations) > 1:
        raise WavelengthMetadataError(
            "conflicting FITS metadata declares both air and vacuum wavelength "
            "conventions; pass wavelength_medium='air' or 'vacuum' explicitly"
        )
    return next(iter(declarations), None)


def _fits_wavelength_column_index(
    header: Mapping[str, object],
    wavelength_col: int | str | None,
) -> int | None:
    field_count = int(header.get("TFIELDS", 0) or 0)
    if field_count == 0:
        field_count = max(
            (
                int(str(key).strip().upper().removeprefix("TTYPE"))
                for key in header
                if str(key).strip().upper().startswith("TTYPE")
                and str(key).strip().upper().removeprefix("TTYPE").isdigit()
            ),
            default=0,
        )
    if isinstance(wavelength_col, int):
        zero_based = wavelength_col if wavelength_col >= 0 else field_count + wavelength_col
        return zero_based + 1 if 0 <= zero_based < field_count else None

    names = {
        str(header.get(f"TTYPE{index}", "")).strip().casefold(): index
        for index in range(1, field_count + 1)
    }
    if isinstance(wavelength_col, str):
        return names.get(wavelength_col.strip().casefold())

    for candidate in WAVELENGTH_COLUMNS:
        index = names.get(candidate.casefold())
        if index is not None:
            return index
    native_orders = _native_order_prefixes(names)
    if len(native_orders) == 1:
        return names[native_orders[0] + "_wl"]
    if len(native_orders) > 1:
        raise SpectrumFormatError(
            "Multiple native physical orders require an explicit wavelength_col "
            "before inferring wavelength-medium metadata."
        )
    return 1 if field_count else None


def save_fits_header_txt(
    fits_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Save all headers from a FITS file as readable, aligned plain text.

    Each HDU gets a labeled section. Blank FITS cards are omitted, while all
    keyword values, comments, ``COMMENT`` cards, and ``HISTORY`` cards are
    retained. If ``output_path`` is omitted, ``spectrum.fits`` is written as
    ``spectrum.header.txt`` beside the FITS file. Existing output is replaced.

    :param fits_path: Input FITS file.
    :param output_path: Optional destination for the formatted header text.
    :return: Path to the text file that was written.
    """

    source = Path(fits_path)
    destination = (
        Path(output_path) if output_path is not None else source.with_suffix(".header.txt")
    )
    lines: list[str] = []

    with fits.open(source) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            lines.append("=" * 80)
            lines.append(f"HDU {hdu_index}: {hdu.name}")
            lines.append("=" * 80)

            cards = [card for card in hdu.header.cards if card.keyword.strip()]
            key_width = max((len(card.keyword) for card in cards), default=10)
            for card in cards:
                comment = f"  # {card.comment}" if card.comment else ""
                lines.append(f"{card.keyword:<{key_width}} : {card.value}{comment}")
            lines.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def infer_spectrum_format(path: str | Path, format: str | None = None) -> str:
    """Return the explicit or filename-derived spectrum format.

    FITS files are commonly distributed with transparent gzip compression.
    ``Path.suffix`` only reports ``.gz`` for those files, so inspect the full
    suffix chain before falling back to the final suffix.
    """

    if format is not None:
        explicit = str(format).lower().lstrip(".")
        return "fits" if explicit == "fts" else explicit
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if suffixes[-2:] in ([".fits", ".gz"], [".fit", ".gz"], [".fts", ".gz"]):
        return "fits"
    inferred = Path(path).suffix.lower().lstrip(".")
    return "fits" if inferred == "fts" else inferred


def save_spectrum(path: str | Path, spectrum: Spectrum) -> Path:
    """Save wavelength, flux, and optional uncertainty as plain text."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [spectrum.wavelength, spectrum.flux]
    header = f"wavelength_{spectrum.wavelength_unit}_{spectrum.wavelength_medium} flux"
    if spectrum.uncertainty is not None:
        columns.append(spectrum.uncertainty)
        header += " uncertainty"
    data = np.column_stack(columns)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=path.suffix or ".txt",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savetxt(temporary, data, header=header)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def save_corrected_txt(result: TelluricFitResult, path: str | Path) -> Path:
    """Save only the corrected spectrum as a compact plain-text file.

    This output is intended for simple plotting or use in software that only
    needs the final corrected spectrum. It contains wavelength and corrected
    flux columns, plus corrected uncertainty when one is available. It does
    not contain the fitted atmospheric transmission, continuum, masks,
    parameters, or provenance; use :func:`save_fit_product_ecsv` when those
    details must be retained. Saving does not rerun or modify the fit, and an
    existing file at ``path`` is overwritten.

    :param result: Completed PyMolFit telluric-fit result returned by correct().
    :param path: Destination filename, conventionally ending in ``.txt``.
    :return: The destination path that was written.
    """

    destination = Path(path)
    save_spectrum(destination, result.corrected)
    return destination


def save_fit_product_ecsv(result: TelluricFitResult, path: str | Path) -> Path:
    """Save the complete reproducible PyMolFit result as an ECSV table.

    ECSV is Astropy's Enhanced Character-Separated Values format. The saved
    table includes input and corrected flux, fitted transmission, continuum,
    model flux, available uncertainties, fit/input/correction masks, fitted
    parameters, wavelength units and medium, diagnostics, and provenance.
    Choose this output when the fit must be inspected, reproduced, or passed
    to another Astropy workflow. Saving does not rerun or modify the fit, and
    an existing file at ``path`` is overwritten.

    :param result: Completed PyMolFit telluric-fit result returned by correct().
    :param path: Destination filename, conventionally ending in ``.ecsv``.
    :return: The destination path that was written.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".ecsv",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        result.write(temporary, format="ascii.ecsv")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def save_fit_product(result: TelluricFitResult, path: str | Path) -> Path:
    """Save the canonical, reloadable ECSV correction product.

    This format preserves the observed and corrected spectra, transmission,
    model, masks, uncertainties, fitted parameters, diagnostics, and
    provenance. It is the stable persistence format for PyMolFit results.
    """

    return save_fit_product_ecsv(result, path)


def load_fit_product(path: str | Path) -> TelluricFitResult:
    """Load a full PyMolFit ECSV product as a ``TelluricFitResult``.

    Compact text files written with :func:`save_corrected_txt` intentionally
    cannot be loaded as fit results because they do not contain the observed
    flux, transmission, model, masks, or provenance.

    :param path: Complete ECSV correction product saved by correct().
    :return: Reconstructed result; no fitting is performed.
    """

    from .fit import TelluricFitResult

    try:
        return TelluricFitResult.read(path, format="ascii.ecsv")
    except ProductFormatError:
        raise
    except Exception as exc:
        raise ProductFormatError(f"could not load PyMolFit fit product {Path(path)}") from exc


def _load_ascii(
    path: Path,
    *,
    format: str,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: str = "vacuum",
) -> Spectrum:
    if (
        format == "ecsv"
        or isinstance(wavelength_col, str)
        or isinstance(flux_col, str)
        or isinstance(uncertainty_col, str)
    ):
        table = Table.read(path)
        return _spectrum_from_table(
            table,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
        )

    delimiter = "," if format == "csv" else None
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("numeric spectra must have at least two columns: wavelength and flux")
    wavelength_index = 0 if wavelength_col is None else int(wavelength_col)
    flux_index = 1 if flux_col is None else int(flux_col)
    uncertainty = None if uncertainty_col is None else data[:, int(uncertainty_col)]
    return Spectrum(
        wavelength=data[:, wavelength_index],
        flux=data[:, flux_index],
        uncertainty=uncertainty,
        wavelength_unit=wavelength_unit,
        wavelength_medium=wavelength_medium,
    )


def _select_spectrum_hdu(
    hdul: fits.HDUList,
    hdu: int | None,
    *,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    image_index: int | None = None,
) -> int:
    """Select one supported spectrum without interpreting arbitrary tables as flux."""

    if hdu is not None:
        if isinstance(hdu, bool) or not isinstance(hdu, (int, np.integer)):
            raise SpectrumFormatError(
                "hdu must be an integer index or None for automatic selection"
            )
        if not -len(hdul) <= hdu < len(hdul):
            raise SpectrumFormatError(
                f"FITS file does not contain HDU {hdu}; it has {len(hdul)} HDUs"
            )
        index = int(hdu) % len(hdul)
        if hdul[index].data is None:
            raise SpectrumFormatError(f"FITS HDU {index} ({hdul[index].name}) contains no data")
        return index

    auxiliary_names = {
        "ERR",
        "ERROR",
        "ERRORS",
        "UNCERTAINTY",
        "SIGMA",
        "NOISE",
        "VAR",
        "VARIANCE",
        "IVAR",
        "WEIGHT",
        "WEIGHTS",
        "MASK",
        "DQ",
        "QUALITY",
        "BPM",
        "BLAZE",
        "FLAT",
        "SKY",
        "WAVE",
        "WAVELENGTH",
    }
    candidates = []
    for index, extension in enumerate(hdul):
        name_parts = set(extension.name.upper().replace("-", "_").split("_"))
        if name_parts & auxiliary_names:
            continue
        if isinstance(extension, (fits.BinTableHDU, fits.TableHDU)):
            names = extension.columns.names
            # Positional fallback remains available for explicitly selected HDUs,
            # but cannot distinguish a spectrum from an unrelated numeric table.
            single_native_order = len(_native_order_prefixes(names)) == 1
            if (
                wavelength_col is None
                and not single_native_order
                and not any(name.lower() in WAVELENGTH_COLUMNS for name in names)
            ):
                continue
            if (
                flux_col is None
                and not single_native_order
                and not any(name.lower() in FLUX_COLUMNS for name in names)
            ):
                continue
            if extension.data is None:
                continue
            table = Table(extension.data, copy=False)
            try:
                wave_name = _resolve_column(table, wavelength_col, WAVELENGTH_COLUMNS, 0)
                flux_name = _resolve_column(table, flux_col, FLUX_COLUMNS, 1)
                if wave_name == flux_name:
                    continue
                wavelength = _column_to_1d(table[wave_name])
                flux = _column_to_1d(table[flux_name])
                if wavelength.size < 2 or wavelength.shape != flux.shape:
                    continue
                if uncertainty_col is not None:
                    error_name = _resolve_column(table, uncertainty_col, UNCERTAINTY_COLUMNS, 2)
                    if _column_to_1d(table[error_name]).shape != flux.shape:
                        continue
            except (ValueError, TypeError):
                continue
        elif isinstance(extension, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
            if any(value is not None for value in (wavelength_col, flux_col, uncertainty_col)):
                continue
            header = extension.header
            if "CRVAL1" not in header or not ("CDELT1" in header or "CD1_1" in header):
                continue
            if str(header.get("CTYPE1", "")).strip().upper() not in {"", "LINEAR", "WAVE", "AWAV"}:
                continue
            if extension.data is None:
                continue
            data = np.asarray(extension.data).squeeze()
            if data.ndim not in {1, 2} or data.shape[-1] < 2:
                continue
            if (
                data.ndim == 2
                and image_index is not None
                and not -data.shape[0] <= image_index < data.shape[0]
            ):
                continue
            try:
                wavelength, _ = _wavelength_from_linear_wcs(
                    header, data.shape[-1], fallback_unit="micron"
                )
            except (ValueError, TypeError):
                continue
        else:
            continue
        finite_wave = wavelength[np.isfinite(wavelength) & (wavelength > 0)]
        if finite_wave.size >= 2 and np.ptp(finite_wave) > 0:
            candidates.append(index)

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        choices = ", ".join(f"{index} ({hdul[index].name})" for index in candidates)
        raise SpectrumFormatError(
            f"Multiple spectral HDUs found: {choices}. Specify hdu=<index> to select one; "
            "automatic selection does not combine spectra or orders."
        )
    raise SpectrumFormatError(
        "No supported spectral HDU could be identified automatically. Specify hdu=<index> "
        "and, if needed, wavelength_col/flux_col or image_index. "
        "Automatic selection requires recognised wavelength/flux columns or a linear wavelength WCS."
    )


def _load_fits(
    path: Path,
    *,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    hdu: int | None = None,
    wavelength_unit: str = "micron",
    wavelength_medium: str = "vacuum",
    image_index: int | None = None,
) -> Spectrum:
    with fits.open(path) as hdul:
        hdu = _select_spectrum_hdu(
            hdul,
            hdu,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            image_index=image_index,
        )
        target_hdu = hdul[hdu]
        data = target_hdu.data

        # Match the correction workflow's primary-header precedence.
        header = dict(target_hdu.header)
        header.update(dict(hdul[0].header))
        observation = Observation(metadata=header)
        if hasattr(data, "columns"):
            table = Table(data)
            for column in data.columns:
                if column.name in table.colnames and column.unit:
                    table[column.name].unit = column.unit
            spectrum = _spectrum_from_table(
                table,
                wavelength_col=wavelength_col,
                flux_col=flux_col,
                uncertainty_col=uncertainty_col,
                wavelength_unit=wavelength_unit,
                wavelength_medium=wavelength_medium,
            )
            return replace(
                spectrum,
                observation=observation,
                meta={**spectrum.meta, "hdu": hdu},
            )

        image = np.asarray(data, dtype=float).squeeze()
        if image.ndim == 2 and image_index is not None:
            if not -image.shape[0] <= image_index < image.shape[0]:
                raise SpectrumFormatError(
                    f"image_index {image_index} is outside the FITS image row range"
                )
            image = image[int(image_index)]
        if image.ndim != 1:
            raise SpectrumFormatError(
                "FITS image spectra must be one-dimensional, or provide image_index for a 2D image"
            )
        wavelength, unit = _wavelength_from_linear_wcs(
            target_hdu.header, image.size, fallback_unit=wavelength_unit
        )
        return Spectrum(
            wavelength=wavelength,
            flux=image,
            wavelength_unit=unit,
            wavelength_medium=wavelength_medium,
            meta={"source": str(path), "hdu": hdu, "io_type": "fits_image"},
            observation=observation,
        )


def _spectrum_from_table(
    table: Table,
    *,
    wavelength_col: int | str | None,
    flux_col: int | str | None,
    uncertainty_col: int | str | None,
    wavelength_unit: str,
    wavelength_medium: str,
) -> Spectrum:
    native_orders = _native_order_prefixes(table.colnames)
    named_generic_pair = any(name.lower() in WAVELENGTH_COLUMNS for name in table.colnames) and any(
        name.lower() in FLUX_COLUMNS for name in table.colnames
    )
    if (
        len(native_orders) > 1
        and not named_generic_pair
        and any(column is None for column in (wavelength_col, flux_col))
    ):
        raise SpectrumFormatError(
            "This native detector table contains multiple physical orders. "
            "Specify matching wavelength_col and flux_col for one WL/SPEC pair; "
            "its ERR column is optional. Selecting hdu alone does not select an order."
        )
    wave_name = _resolve_column(table, wavelength_col, WAVELENGTH_COLUMNS, 0)
    flux_name = _resolve_column(table, flux_col, FLUX_COLUMNS, 1)
    native_prefix = _matching_native_order(wave_name, flux_name)
    uncertainty_name = None
    if uncertainty_col is not None:
        uncertainty_name = _resolve_column(table, uncertainty_col, UNCERTAINTY_COLUMNS, 2)
        error_identity = _native_column_identity(uncertainty_name)
        if (
            native_prefix is not None
            and error_identity is not None
            and error_identity != (native_prefix, "err")
        ):
            raise SpectrumFormatError(
                "The native uncertainty column must match the selected WL/SPEC physical order."
            )
    elif native_prefix is not None:
        # Never substitute another order's uncertainty or a positional third column.
        by_lower = {name.lower(): name for name in table.colnames}
        uncertainty_name = by_lower.get(native_prefix + "_err")
    else:
        try:
            uncertainty_name = _resolve_column(table, None, UNCERTAINTY_COLUMNS, 2, required=False)
        except ValueError:
            uncertainty_name = None
        non_error_columns = QUALITY_COLUMNS + VALID_MASK_COLUMNS + ORDER_COLUMNS + DETECTOR_COLUMNS
        if uncertainty_name is not None and uncertainty_name.lower() in non_error_columns:
            uncertainty_name = None

    unit = _infer_wavelength_unit(table, wave_name, wavelength_unit)
    uncertainty = None if uncertainty_name is None else _column_to_1d(table[uncertainty_name])
    if uncertainty is not None and not np.any(np.isfinite(uncertainty)):
        uncertainty = None
    wavelength = _column_to_1d(table[wave_name])
    flux = _column_to_1d(table[flux_name])
    mask, quality_columns = _table_valid_mask(
        table,
        expected_size=flux.size,
        wavelength_name=wave_name,
        flux_name=flux_name,
        uncertainty_name=uncertainty_name,
    )
    group_id, group_columns = _table_physical_groups(table, flux.size)
    return Spectrum(
        wavelength=wavelength,
        flux=flux,
        uncertainty=uncertainty,
        mask=mask,
        group_id=group_id,
        wavelength_unit=unit,
        wavelength_medium=wavelength_medium,
        meta={
            **dict(table.meta),
            "io_type": "table",
            "wavelength_col": wave_name,
            "flux_col": flux_name,
            "quality_columns": quality_columns,
            "physical_group_columns": group_columns,
        },
    )


def _native_column_identity(name: str) -> tuple[str, str] | None:
    parts = name.lower().split("_")
    if (
        len(parts) == 3
        and all(part.isdigit() for part in parts[:2])
        and parts[2] in {"wl", "spec", "err"}
    ):
        return "_".join(parts[:2]), parts[2]
    return None


def _matching_native_order(wavelength_name: str, flux_name: str) -> str | None:
    """Validate native column roles and physical identity after explicit resolution."""
    wave = _native_column_identity(wavelength_name)
    flux = _native_column_identity(flux_name)
    if wave is None and flux is None:
        return None
    if wave is not None and flux is not None and wave[1] == "wl" and flux == (wave[0], "spec"):
        return wave[0]
    raise SpectrumFormatError(
        "Native wavelength and flux columns must be a matching WL/SPEC pair "
        "from the same physical order."
    )


def _table_valid_mask(
    table: Table,
    *,
    expected_size: int,
    wavelength_name: str,
    flux_name: str,
    uncertainty_name: str | None,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Return a good-pixel mask from table masks and common quality columns."""

    valid = np.ones(expected_size, dtype=bool)
    used: list[str] = []
    for name in (wavelength_name, flux_name, uncertainty_name):
        if name is None:
            continue
        column_mask = getattr(table[name], "mask", None)
        if column_mask is None or np.isscalar(column_mask):
            continue
        flattened = np.asarray(column_mask, dtype=bool).squeeze()
        if flattened.shape == valid.shape:
            valid &= ~flattened
            used.append(f"{name}:astropy_mask")

    quality_name = _resolve_named_column(table, QUALITY_COLUMNS)
    if quality_name is not None:
        quality = np.asarray(table[quality_name]).squeeze()
        if quality.shape == valid.shape and (
            np.issubdtype(quality.dtype, np.number) or quality.dtype == np.bool_
        ):
            valid &= np.asarray(quality == 0, dtype=bool)
            used.append(quality_name)

    valid_name = _resolve_named_column(table, VALID_MASK_COLUMNS)
    if valid_name is not None:
        good = np.asarray(table[valid_name]).squeeze()
        if good.shape == valid.shape:
            valid &= np.asarray(good, dtype=bool)
            used.append(valid_name)

    return (valid if used else None), tuple(used)


def _table_physical_groups(
    table: Table,
    expected_size: int,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Encode order/detector columns as stable integer physical-group IDs."""

    names = tuple(
        name
        for name in (
            _resolve_named_column(table, ORDER_COLUMNS),
            _resolve_named_column(table, DETECTOR_COLUMNS),
        )
        if name is not None
    )
    if not names:
        return None, ()
    values = [np.asarray(table[name]).squeeze() for name in names]
    if any(value.shape != (expected_size,) for value in values):
        return None, ()
    keys = np.column_stack([np.asarray(value).astype(str) for value in values])
    _, group_id = np.unique(keys, axis=0, return_inverse=True)
    return np.asarray(group_id, dtype=int), names


def _resolve_named_column(table: Table, candidates: tuple[str, ...]) -> str | None:
    by_lower = {name.lower(): name for name in table.colnames}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _column_to_1d(column: object) -> np.ndarray:
    array = np.asarray(column, dtype=float).squeeze()
    if array.ndim != 1:
        raise ValueError(
            "spectrum table columns must be one-dimensional or single-row vector columns"
        )
    return array


def _resolve_column(
    table: Table,
    requested: int | str | None,
    candidates: tuple[str, ...],
    fallback_index: int,
    *,
    required: bool = True,
) -> str:
    if isinstance(requested, int):
        try:
            return table.colnames[requested]
        except IndexError as exc:
            raise ValueError(f"column index {requested} is outside table range") from exc
    if isinstance(requested, str):
        if requested in table.colnames:
            return requested
        raise ValueError(f"column {requested!r} not found in table")

    by_lower = {name.lower(): name for name in table.colnames}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    native_orders = _native_order_prefixes(table.colnames)
    suffix = (
        "_wl"
        if candidates == WAVELENGTH_COLUMNS
        else "_spec"
        if candidates == FLUX_COLUMNS
        else "_err"
        if candidates == UNCERTAINTY_COLUMNS
        else None
    )
    if native_orders and suffix is not None:
        if len(native_orders) > 1:
            raise SpectrumFormatError(
                "Multiple native physical orders require explicit WL/SPEC/ERR column selection; "
                "positional fallback is unsafe for this table."
            )
        native_name = by_lower.get(native_orders[0] + suffix)
        if native_name is not None:
            return native_name
        raise SpectrumFormatError(f"Native order {native_orders[0]} has no {suffix[1:]} column")
    if fallback_index < len(table.colnames):
        return table.colnames[fallback_index]
    if required:
        raise ValueError(f"could not infer a column from candidates {candidates}")
    raise ValueError("optional column not found")


def _infer_wavelength_unit(table: Table, wavelength_col: str, fallback_unit: str) -> str:
    column = table[wavelength_col]
    if getattr(column, "unit", None) is not None:
        return str(column.unit)

    lowered = wavelength_col.lower()
    if "angstrom" in lowered:
        return "angstrom"
    if lowered.endswith("_nm") or "nanometer" in lowered:
        return "nm"
    if lowered.endswith("_um") or "micron" in lowered:
        return "micron"
    return fallback_unit


def _wavelength_from_linear_wcs(
    header: fits.Header, n_pixels: int, *, fallback_unit: str
) -> tuple[np.ndarray, str]:
    if "CRVAL1" not in header:
        raise ValueError("FITS image spectrum is missing CRVAL1 wavelength WCS")
    if "CDELT1" in header:
        delta = float(header["CDELT1"])
    elif "CD1_1" in header:
        delta = float(header["CD1_1"])
    else:
        raise ValueError("FITS image spectrum is missing CDELT1/CD1_1 wavelength WCS")

    crval = float(header["CRVAL1"])
    crpix = float(header.get("CRPIX1", 1.0))
    pixel = np.arange(n_pixels, dtype=float) + 1.0
    wavelength = crval + (pixel - crpix) * delta
    unit = str(header.get("CUNIT1", fallback_unit)).strip() or fallback_unit
    return wavelength, unit
