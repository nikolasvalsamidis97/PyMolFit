from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .spectrum import Spectrum
from .provenance import file_sha256

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
)
FLUX_COLUMNS = ("flux", "flam", "fnu", "spectrum", "spec", "data", "science")
UNCERTAINTY_COLUMNS = ("uncertainty", "error", "err", "sigma", "noise", "flux_err", "flux_error")
QUALITY_COLUMNS = ("qual", "quality", "dq", "data_quality", "bpm", "bad_pixel_mask", "mask")
VALID_MASK_COLUMNS = ("valid", "input_mask", "good_pixel", "good")
ORDER_COLUMNS = ("order", "spectral_order", "echelle_order", "physical_group")
DETECTOR_COLUMNS = ("detec", "detector", "chip", "extension")


def load_spectrum(
    path: str | Path,
    *,
    format: str | None = None,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    hdu: int = 1,
    wavelength_unit: str = "micron",
    wavelength_medium: str = "vacuum",
    image_index: int | None = None,
) -> Spectrum:
    path = Path(path)
    chosen_format = infer_spectrum_format(path, format)
    if chosen_format in {"txt", "dat", "csv", "ascii", "ecsv"}:
        spectrum = _load_ascii(
            path,
            format=chosen_format,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
        )
    elif chosen_format in {"fits", "fit", "fz"}:
        spectrum = _load_fits(
            path,
            wavelength_col=wavelength_col,
            flux_col=flux_col,
            uncertainty_col=uncertainty_col,
            hdu=hdu,
            wavelength_unit=wavelength_unit,
            wavelength_medium=wavelength_medium,
            image_index=image_index,
        )
    else:
        raise ValueError(f"unsupported spectrum format: {chosen_format}")
    return replace(
        spectrum,
        meta={
            **dict(spectrum.meta),
            "source": str(path.resolve()),
            "source_file_sha256": file_sha256(path),
        },
    )


def infer_spectrum_format(path: str | Path, format: str | None = None) -> str:
    """Return the explicit or filename-derived spectrum format.

    FITS files are commonly distributed with transparent gzip compression.
    ``Path.suffix`` only reports ``.gz`` for those files, so inspect the full
    suffix chain before falling back to the final suffix.
    """

    if format is not None:
        return str(format).lower().lstrip(".")
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if suffixes[-2:] in ([".fits", ".gz"], [".fit", ".gz"], [".fts", ".gz"]):
        return "fits"
    return Path(path).suffix.lower().lstrip(".")


def save_spectrum(path: str | Path, spectrum: Spectrum) -> None:
    path = Path(path)
    columns = [spectrum.wavelength, spectrum.flux]
    header = f"wavelength_{spectrum.wavelength_unit}_{spectrum.wavelength_medium} flux"
    if spectrum.uncertainty is not None:
        columns.append(spectrum.uncertainty)
        header += " uncertainty"
    data = np.column_stack(columns)
    np.savetxt(path, data, header=header)


def save_corrected_txt(result: TelluricFitResult, path: str | Path) -> Path:
    """Save only the corrected spectrum as a compact plain-text file.

    This output is intended for simple plotting or use in software that only
    needs the final corrected spectrum. It contains wavelength and corrected
    flux columns, plus corrected uncertainty when one is available. It does
    not contain the fitted atmospheric transmission, continuum, masks,
    parameters, or provenance; use :func:`save_fit_product_ecsv` when those
    details must be retained. Saving does not rerun or modify the fit, and an
    existing file at ``path`` is overwritten.

    :param result: Completed PyMolFit telluric-fit result returned by
        ``correct_file`` or ``correct_arrays``.
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

    :param result: Completed PyMolFit telluric-fit result returned by
        ``correct_file`` or ``correct_arrays``.
    :param path: Destination filename, conventionally ending in ``.ecsv``.
    :return: The destination path that was written.
    """

    destination = Path(path)
    result.write(destination, format="ascii.ecsv")
    return destination


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
    if format == "ecsv" or isinstance(wavelength_col, str) or isinstance(flux_col, str) or isinstance(uncertainty_col, str):
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


def _load_fits(
    path: Path,
    *,
    wavelength_col: int | str | None = None,
    flux_col: int | str | None = None,
    uncertainty_col: int | str | None = None,
    hdu: int = 1,
    wavelength_unit: str = "micron",
    wavelength_medium: str = "vacuum",
    image_index: int | None = None,
) -> Spectrum:
    with fits.open(path) as hdul:
        target_hdu = hdul[hdu]
        data = target_hdu.data
        if data is None and hdu != 0:
            target_hdu = hdul[0]
            data = target_hdu.data
        if data is None:
            raise ValueError(f"FITS file {path} does not contain spectrum data in HDU {hdu}")

        if hasattr(data, "columns"):
            table = Table(data)
            for column in data.columns:
                if column.name in table.colnames and column.unit:
                    table[column.name].unit = column.unit
            return _spectrum_from_table(
                table,
                wavelength_col=wavelength_col,
                flux_col=flux_col,
                uncertainty_col=uncertainty_col,
                wavelength_unit=wavelength_unit,
                wavelength_medium=wavelength_medium,
            )

        image = np.asarray(data, dtype=float).squeeze()
        if image.ndim == 2 and image_index is not None:
            image = image[int(image_index)]
        if image.ndim != 1:
            raise ValueError(
                "FITS image spectra must be one-dimensional, or provide image_index for a 2D image"
            )
        wavelength, unit = _wavelength_from_linear_wcs(target_hdu.header, image.size, fallback_unit=wavelength_unit)
        return Spectrum(
            wavelength=wavelength,
            flux=image,
            wavelength_unit=unit,
            wavelength_medium=wavelength_medium,
            meta={"source": str(path), "hdu": hdu, "io_type": "fits_image"},
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
    wave_name = _resolve_column(table, wavelength_col, WAVELENGTH_COLUMNS, 0)
    flux_name = _resolve_column(table, flux_col, FLUX_COLUMNS, 1)
    uncertainty_name = None
    if uncertainty_col is not None:
        uncertainty_name = _resolve_column(table, uncertainty_col, UNCERTAINTY_COLUMNS, 2)
    else:
        try:
            uncertainty_name = _resolve_column(table, None, UNCERTAINTY_COLUMNS, 2, required=False)
        except ValueError:
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
            "io_type": "table",
            "wavelength_col": wave_name,
            "flux_col": flux_name,
            "quality_columns": quality_columns,
            "physical_group_columns": group_columns,
        },
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
        if quality.shape == valid.shape:
            if np.issubdtype(quality.dtype, np.number) or quality.dtype == np.bool_:
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


def _wavelength_from_linear_wcs(header: fits.Header, n_pixels: int, *, fallback_unit: str) -> tuple[np.ndarray, str]:
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
