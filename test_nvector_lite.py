from __future__ import annotations

import math
import warnings
from typing import Any

import hypothesis
import hypothesis.strategies as st
import hypothesis.extra.numpy as st_np
import numpy as np
import nvector
import pyproj
from numpy.typing import NDArray

from nvector_lite import (
    lonlat_to_nvector,
    nvector_to_lonlat,
    _normalize,
    nvector_direct,
    nvector_polygon_contains_pole,
    nvector_cross_track_distance,
)

# "The Tau Manifesto": https://tauday.com/
π = math.pi
τ = 2.0 * π

# The ECEF rotation representing the "E" reference frame.
_frame_E = nvector.E_rotation("E")

earth_radius_avg_m = 6_371_000.0
earth_radius_avg_km = earth_radius_avg_m / 1000.0


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
            u = _normalize(v)

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


class test_cross_track_distance:
    def test_example(self) -> None:
        r"""Cross-track distance for every point in a motion track

        We are given ``N`` origin/destination pairs, and, for each pair, the motion
        track of an object that travelled from the origin to the destination.

        We would like to compute the cross-track distance of every point in each motion
        track, relative to the geodesic between its origin and destination.

        The lon/lat solution here uses the formula from https://www.movable-type.co.uk/scripts/latlong.html

        To see this data on a map, go to https://geojson.io/#map=7.97/43.423/-75.429 and enter the following:

        .. code-block:: json

            {
              "type": "FeatureCollection",
              "features": [
                {
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "coordinates": [
                      [
                        -76.2137815136364,
                        43.46001114816093
                      ],
                      [
                        -73.77150782932506,
                        43.08511132138398
                      ]
                    ],
                    "type": "LineString"
                  }
                },
                {
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "coordinates": [
                      [
                        -75.45690176584871,
                        43.193926425093935
                      ],
                      [
                        -73.66740169301255,
                        43.31073783085333
                      ]
                    ],
                    "type": "LineString"
                  }
                },
                {
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "coordinates": [
                      -75.44564704840916,
                      43.52124976781056
                    ],
                    "type": "Point"
                  }
                },
                {
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "coordinates": [
                      -75.07705505227482,
                      43.17341009622575
                    ],
                    "type": "Point"
                  }
                },
                {
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "coordinates": [
                      -74.6015432404678,
                      43.32097375630951
                    ],
                    "type": "Point"
                  }
                },
                {
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "coordinates": [
                      -74.17949133649684,
                      43.59261634178711
                    ],
                    "type": "Point"
                  }
                }
              ]
            }
        """
        origin_lons = np.asarray([-76.21378151363640, -75.456901765848710])
        origin_lats = np.asarray([ 43.46001114816093,  43.193926425093935])
        destin_lons = np.asarray([-73.77150782932506, -73.66740169301255])
        destin_lats = np.asarray([ 43.08511132138398,  43.31073783085333])
        point_lons = np.asarray([-75.44564704840916, -75.07705505227482, -74.60154324046780, -74.17949133649684])
        point_lats = np.asarray([ 43.52124976781056,  43.17341009622575,  43.32097375630951,  43.59261634178711])

        ## Nvector solution

        # Shapes: (3, 2)
        origin_nvects = lonlat_to_nvector(origin_lons, origin_lats)
        destin_nvects = lonlat_to_nvector(destin_lons, destin_lats)
        # Shape: (3, 4)
        point_nvects = lonlat_to_nvector(point_lons, point_lats)

        xt_nvect = nvector_cross_track_distance(
            # Shapes: (3, 2, 1)
            origin_nvects[:, ..., np.newaxis],
            destin_nvects[:, ..., np.newaxis],
            # Shapes: (3, 1, 4)
            point_nvects[:, np.newaxis, ...],
        )
        assert xt_nvect.shape == (2, 4)

        # Note that it isn't really necessary to multiply distances by
        # ``earth_radius_avg_km`` here; we could just use the unit sphere and everything
        # would work the same. But it's illustrative as an example of real-world usage,
        # where you typically want distances to be expressed in meaningful, familiar
        # units.
        xt_nvect *= earth_radius_avg_km

        ## Lon/Lat solution

        geod = pyproj.Geod(a=1.0, f=0.0)

        # Shapes: (2,)
        path_bearing, _, path_distance = geod.inv(
            # Shapes: (2,)
            origin_lons,
            origin_lats,
            destin_lons,
            destin_lats,
        )
        path_bearing = np.radians(path_bearing)

        # Shapes: (2, 4)
        point_bearing, _, point_distance = geod.inv(
            # Shapes: (8,)
            np.repeat(origin_lons, len(point_lons)),
            np.repeat(origin_lats, len(point_lats)),
            np.tile(point_lons, len(origin_lons)),
            np.tile(point_lats, len(origin_lats)),
        )
        out_shape = (len(origin_lons), len(point_lons))
        point_bearing = np.radians(point_bearing.reshape(out_shape))
        point_distance = point_distance.reshape(out_shape)

        xt_lonlat = np.asin(
            np.sin(point_distance) *
            np.sin(point_bearing - path_bearing[:, np.newaxis])
        )
        # As above, not necessary and shouldn't change the result,
        # but more realistic as an example of usage.
        xt_lonlat *= earth_radius_avg_km

        ## Test it

        np.testing.assert_allclose(xt_nvect, xt_lonlat)
