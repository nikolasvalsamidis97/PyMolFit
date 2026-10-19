from dataclasses import replace

import numpy as np
import pytest

from pymolfit import Spectrum
from pymolfit.workflow import _split_spectrum


def _make_spectrum(sizes):
    groups = np.repeat(np.arange(len(sizes)), sizes)
    wave = np.concatenate([1.0 + i * 0.03 + np.arange(n) * 1e-4 for i, n in enumerate(sizes)])
    return Spectrum(
        wavelength=wave,
        flux=np.arange(len(wave), dtype=float),
        group_id=groups,
        uncertainty=np.arange(len(wave), dtype=float) + 1.0,
        mask=np.arange(len(wave)) % 7 != 0,
    )


@pytest.mark.parametrize("sizes", [(2, 20, 20), (20, 1), (1, 1), (3, 20)])
def test_unsupported_labeled_group_errors_before_discarding(sizes):
    original = _make_spectrum(sizes)
    original_arrays = {
        key: getattr(original, key).copy()
        for key in ("wavelength", "flux", "uncertainty", "group_id", "mask")
    }
    with pytest.raises(ValueError, match="No samples were discarded"):
        _split_spectrum(original, segment_size=0.01, minimum_points=4)
    for key, values in original_arrays.items():
        np.testing.assert_array_equal(getattr(original, key), values)


def test_valid_groups_all_samples_errors_labels_preserved_after_chunking():
    original = _make_spectrum((20, 30, 40))
    chunks = _split_spectrum(original, segment_size=0.001, minimum_points=4)
    for key in ("wavelength", "flux", "uncertainty", "group_id", "mask"):
        np.testing.assert_array_equal(
            np.concatenate([getattr(c, key) for c in chunks]), getattr(original, key)
        )
    assert {c.meta["physical_group_count"] for c in chunks} == {3}


def test_unlabeled_short_gap_islands_remain_preserved():
    original = replace(_make_spectrum((2, 20, 20)), group_id=None)
    chunks = _split_spectrum(original, segment_size=0.1, minimum_points=4)
    np.testing.assert_array_equal(
        np.concatenate([c.wavelength for c in chunks]), original.wavelength
    )


def test_group_order_and_label_permutation_do_not_hide_short_group():
    original = _make_spectrum((2, 20, 20))
    permutation = np.random.default_rng(9).permutation(len(original.flux))
    shuffled = replace(
        original,
        wavelength=original.wavelength[permutation],
        flux=original.flux[permutation],
        uncertainty=original.uncertainty[permutation],
        mask=original.mask[permutation],
        group_id=original.group_id[permutation] + 20,
    )
    with pytest.raises(ValueError, match="has 2 samples"):
        _split_spectrum(shuffled, segment_size=0.01, minimum_points=4)
