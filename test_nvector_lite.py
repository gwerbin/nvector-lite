from __future__ import annotations

import math
import warnings
from typing import Any

import hypothesis
import hypothesis.strategies as st
import hypothesis.extra.numpy as st_np
import numpy as np
import nvector
from numpy.typing import NDArray

from nvector_lite import (
    lonlat_to_nvector,
    nvector_to_lonlat,
    normalize,
    nvector_direct,
    nvector_polygon_contains_pole,
)

# "The Tau Manifesto": https://tauday.com/
π = math.pi
τ = 2.0 * π

# The ECEF rotation representing the "E" reference frame.
_frame_E = nvector.E_rotation("E")

earth_radius_avg_m = 6_371_000.0


@st.composite
def st_lonlat_radians(
    draw: st.DrawFn, /, **shape_kwargs: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Generate longitudes and latitudes."""
    shape = draw(st_np.array_shapes(**shape_kwargs))
    lon = draw(
        st_np.arrays(np.float64, shape, elements=dict(min_value=-τ, max_value=τ))
    )
    lat = draw(
        st_np.arrays(np.float64, shape, elements=dict(min_value=-π, max_value=π))
    )
    return lon, lat


class test_normalize:
    r"""Tests for `normalize`."""

    @hypothesis.given(
        v=st_np.arrays(
            np.float64,
            (5, 30, 40),
            elements=dict(allow_nan=False, allow_infinity=False, allow_subnormal=False),
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
            u = normalize(v, axis=0)

        assert u.shape == v.shape
        np.testing.assert_allclose(np.linalg.norm(u, axis=0, keepdims=True), 1.0)


class test_lonlat_to_nvector:
    r"""Tests for `lonlat_to_nvector`."""

    @hypothesis.given(lonlat=st_lonlat_radians(max_dims=1, max_side=100))
    def test_identical(
        self, lonlat: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        r"""Test that output is identical to the `nvector` library output."""
        lon, lat = lonlat

        # WARNING: The lon/lat input orders are swapped, be careful!
        nvect_actual = lonlat_to_nvector(lon, lat, radians=True)
        nvect_expected = nvector.lat_lon2n_E(lat, lon, R_Ee=_frame_E)

        np.testing.assert_allclose(nvect_actual, nvect_expected)


class test_nvector_to_lonlat:
    r"""Tests for `nvector_to_lonlat`."""

    @hypothesis.given(lonlat=st_lonlat_radians(max_dims=1, max_side=200))
    def test_identical(
        self, lonlat: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        r"""Test that output is identical to the `nvector` library output."""
        lon, lat = lonlat
        nvect = nvector.lat_lon2n_E(lat, lon, R_Ee=_frame_E)

        # WARNING: The lon/lat output orders are swapped, be careful!
        lon_actual, lat_actual = nvector_to_lonlat(nvect, radians=True)
        lat_expected, lon_expected = nvector.n_E2lat_lon(nvect, R_Ee=_frame_E)

        np.testing.assert_allclose(lon_actual, lon_expected)
        np.testing.assert_allclose(lat_actual, lat_expected)

    @hypothesis.given(lonlat=st_lonlat_radians(max_dims=10, max_side=20))
    def test_identical_nd(
        self, lonlat: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        lon, lat = lonlat

        nvect_flat = nvector.lat_lon2n_E(lat.ravel(), lon.ravel(), R_Ee=_frame_E)

        nvect = nvect_flat.reshape((3, *lon.shape))

        # WARNING: The lon/lat output orders are swapped, be careful!
        lon_actual, lat_actual = nvector_to_lonlat(nvect, radians=True)

        lat_expected_flat, lon_expected_flat = nvector.n_E2lat_lon(nvect_flat, R_Ee=_frame_E)

        lon_expected = lon_expected_flat.reshape(lon.shape)
        lat_expected = lat_expected_flat.reshape(lat.shape)

        np.testing.assert_allclose(lon_actual, lon_expected)
        np.testing.assert_allclose(lat_actual, lat_expected)


class test_nvector_direct:
    r"""Tests for `nvector_direct`."""

    @hypothesis.given(
        lonlat=st_lonlat_radians(max_dims=1, min_side=1, max_side=100),
        azimuth=st.floats(min_value=-τ, max_value=τ),
        distance=st.floats(min_value=10.0, max_value=1_000_000),
    )
    def test_equal_nvector(
        self,
        lonlat: tuple[NDArray[np.float64], NDArray[np.float64]],
        azimuth: float,
        distance: float,
    ) -> None:
        r"""Test that output is identical to `nvector.n_EA_E_distance_and_azimuth2n_EB_E`."""
        start_lon, start_lat = lonlat
        start_nvect = lonlat_to_nvector(start_lon, start_lat, radians=True)
        distance /= earth_radius_avg_m
        end_nvect_actual = nvector_direct(start_nvect, distance, azimuth)
        end_nvect_expected = nvector.n_EA_E_distance_and_azimuth2n_EB_E(
            start_nvect, distance, azimuth, R_Ee=_frame_E
        )
        np.testing.assert_allclose(end_nvect_actual, end_nvect_expected)

    @hypothesis.given(
        lonlat=st_lonlat_radians(max_dims=1, min_side=1, max_side=100),
        azimuth=st_np.arrays(
            np.float64,
            st.integers(min_value=1, max_value=10),
            elements=dict(min_value=-τ, max_value=τ),
        ),
        distance_scalar=st.booleans(),
        data=st.data(),
    )
    def test_broadcast_azimuth_outer(
        self,
        lonlat: tuple[NDArray[np.float64], NDArray[np.float64]],
        azimuth: NDArray[np.float64],
        distance_scalar: bool,
        data: st.DataObject,
    ) -> None:
        r"""Test that broadcasting can be used for a Cartesian product with azimuth."""
        center_lon, center_lat = lonlat
        center_nvect = lonlat_to_nvector(center_lon, center_lat, radians=True)

        distance: float | NDArray[np.float64]
        if distance_scalar:
            distance = data.draw(st.floats(min_value=10.0, max_value=1_000_000.0))
        else:
            distance = data.draw(
                st_np.arrays(
                    np.float64,
                    center_nvect.shape[1],
                    elements=dict(min_value=10.0, max_value=1_000_000.0),
                )
            )
        distance /= earth_radius_avg_m

        out_actual = nvector_direct(
            np.expand_dims(center_nvect, 1),
            distance,
            np.expand_dims(azimuth, (0, 2)),
        )

        out_expected = np.empty(
            (3, len(azimuth), center_nvect.shape[1]), dtype=np.float64
        )
        for ((k,), az) in np.ndenumerate(azimuth):
            out_expected[:, k, :] = nvector_direct(center_nvect, distance, az)

        np.testing.assert_array_equal(out_actual, out_expected)


class test_nvector_polygon_contains_pole:
    def test_example_northpole(self) -> None:
        polygon_lonlat = np.array([
            [0, 60],
            [45, 60],
            [90, 60],
            [135, 60],
            [180, 60],
            [-135, 60],
            [-90, 60],
            [-45, 60],
        ])
        nvect = lonlat_to_nvector(polygon_lonlat[:, 0], polygon_lonlat[:, 1])
        assert nvector_polygon_contains_pole(nvect) == (True, False)

    def test_example_southpole(self) -> None:
        polygon_lonlat = np.array([
            [0, -60],
            [-45, -60],
            [-90, -60],
            [-135, -60],
            [180, -60],
            [135, -60],
            [90, -60],
            [45, -60],
        ])
        nvect = lonlat_to_nvector(polygon_lonlat[:, 0], polygon_lonlat[:, 1])
        assert nvector_polygon_contains_pole(nvect) == (False, True)

    def test_example_neither(self) -> None:
        polygon_lonlat = np.array([
            [64, -9],
            [64, -18],
            [73, -18],
            [73, -9],
        ])
        nvect = lonlat_to_nvector(polygon_lonlat[:, 0], polygon_lonlat[:, 1])
        assert nvector_polygon_contains_pole(nvect) == (False, False)
