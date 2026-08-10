"""Public exception hierarchy for PyMolFit.

Applications can catch :class:`PyMolFitError` for every expected package
failure, or a more specific subclass when they can recover from one class of
problem. Configuration exceptions also inherit from :class:`ValueError` and
external-data exceptions from :class:`RuntimeError` for compatibility with
earlier PyMolFit releases.
"""

from __future__ import annotations


class PyMolFitError(Exception):
    """Base class for expected PyMolFit failures."""


class ConfigurationError(PyMolFitError, ValueError):
    """User inputs or requested options are inconsistent or unsupported."""


class SpectrumFormatError(ConfigurationError):
    """A spectrum file or table does not have a supported layout."""


class WavelengthMetadataError(ConfigurationError):
    """Wavelength units, medium, or velocity-frame metadata are ambiguous."""


class ProductFormatError(ConfigurationError):
    """A saved correction product is missing or has an incompatible schema."""


class ExternalDataError(PyMolFitError, RuntimeError):
    """Required line, atmosphere, or continuum data could not be resolved."""


class FitError(PyMolFitError, RuntimeError):
    """A telluric model could not be constructed or fitted."""
