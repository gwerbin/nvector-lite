import warnings
from itertools import chain, permutations

import hypothesis
import hypothesis.strategies as st
import hypothesis.extra.numpy as st_np
import numpy as np
import pytest
from numpy.typing import NDArray

from nvector_lite import (
    _promote_shape,
    _normalize,
)


class test_promote_shape:
    example_inputs_outputs = [
        (
            np.asarray([1,2,3]),
            np.asarray([[1],[2],[3]]),
        ),
        (
            np.asarray([[1],[2],[3]]),
            np.asarray([[1],[2],[3]]),
        ),
        (
            np.asarray([[[1],[2],[3]]]),
            np.asarray([[[1],[2],[3]]]),
        ),
    ]

    example_inputs_outputs_multi = [
        tuple(zip(*perm))
        for perm in
        chain(
            permutations(example_inputs_outputs, 1),
            permutations(example_inputs_outputs, 2),
            permutations(example_inputs_outputs, 3),
            permutations(example_inputs_outputs, 4),
        )
    ]

    @pytest.mark.parametrize(("input", "output"), example_inputs_outputs_multi)
    def test_examples(self, input, output) -> None:
        result = tuple(_promote_shape(*input))
        assert len(result) == len(output)
        for res, out in zip(result, output):
            np.testing.assert_array_equal(res, out)


class test_normalize:

    # TODO: Test bigger and smaller float dtypes
    @hypothesis.given(
        v=st_np.arrays(
            np.float64,
            (5, 30, 40),
            elements=dict(
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
        )
    )
    def test_correct_nonzero_norm(self, v: NDArray[np.float64]) -> None:
        r"""Test that vectors with non-zero norm are correctly normalized to unit vectors."""

        # Filter out inputs with 0 norm for now.
        # TODO: Remove this filter and test the 0-norm handling logic.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.max(v, axis=0, keepdims=True) + np.finfo(np.float64).tiny
            n = np.linalg.norm(v / m, axis=0)
            hypothesis.assume(np.all(n != 0.0))

        # We shouldn't have any overflows or divide-by-0s at this point.
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            u = _normalize(v)

        assert u.shape == v.shape
        np.testing.assert_allclose(np.linalg.norm(u, axis=0, keepdims=True), 1.0)

