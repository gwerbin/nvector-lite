r"""N-Vector Lite: an efficient implementation of the n-vector horizontal position system.

About this library
==================

This implementation is hard-coded to use a spherical Earth model in the the "E"
reference frame from Gade (2010), for implementation simplicity. The choice of reference
frame does not affect any of the results of geodesic calculations. However it is
important to know the reference frame when converting between n-vector and other
representations, and when interoperating with other libraries.

An "n-vector" is a 3-vector. In our implementation, an array of shape ``(N, 3)``
represents a collection of ``N`` n-vectors.

In certain cases, only "scalar" n-vector arrays are accepted, i.e. arrays strictly of
shape ``(1, 3)``.

1-d arrays of shape ``(3,)`` represent single n-vectors, and are upgraded to 2-d arrays
of shape ``(1, 3)`` (row vectors).

Multidimensional collections of n-vectors are also supported, e.g. ``(K, M, N, 3)``
representing a K×M×N array of n-vectors.

N-vectors in spaces other than 3-dimensional Cartesian Earth-centered Earth-fixed (ECEF)
are not supported.

.. warning:: N-vectors are expected to be unit vectors. For performance, this library
  does **not** check that inputs have unit norm. N-vectors produced by
  :func:`lonlat_to_nvector` will be properly normalized, so most usage should be "safe"
  with respect to the unit-norm assumption. However, n-vector data received from
  untrusted sources should be checked for unit norm, otherwise silently incorrect
  results will be produced.


About the n-vector system
=========================

The "n-vector" representation of horizontal position represents each point on the
surface of the earth as the unit normal vector to that point on the earth ([Gade2010]_,
[GadeHome], [GadeExplained]). This representation is valid for any spherical or
ellipsoidal Earth model, as long as the normal vectors have a closed-form expression.

The actual vectors resulting from this representation are dependent on choosing
a particular coordinate reference frame. [Gade2010]_ prefers the :math:`E` reference frame,
which is an Earth-centered and Earth-fixed (ECEF) or "geocentric" frame [WikipediaECEF]_.

This :math:`E` frame is decomposed in 3 dimensions:

* The ``x`` dimension is the Earth's rotation axis.
* The ``y`` and ``z`` dimensions form the equatorial plane.
* The North Pole (90°N, undefined/arbitrary°E) is ``(1,0,0)``.
* The point 0°N, 0°E is ``(0,0,1)``
* The point 0°N, 90°E is ``(0,1,0)``.

This frame is slightly unintuitive compared to our usual mental model of a globe, where
North is "up". But it has several desirable properties for motion tracking of objects
(especially airplanes), and it is the reference frame used throughout [Gade2010]_.

Note that in the n-vector representation, geodesic distances between n-vectors are
represented as arc angles (in radians). In a spherical model, an arc angle is simply the
fraction of the Earth circumference (2π) traveled over the great-circle path. Therefore
an arc angle can be converted to a (spherical) surface distance by multiplying by the
arc angle by the sphere radius. See [MathSE1]_ and [MathSE2]_.

References
==========

.. [Gade2010] Kenneth Gade. A nonsingular horizontal position representation.
  The Journal of Navigation, 63(3):395–417, 2010.
  `DOI: 10.1017/S0373463309990415 <https://doi.org/10.1017/S0373463309990415>`_.
  `Full text PDF <http://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_,
  `Journal homepage: <https://www.cambridge.org/core/journals/journal-of-navigation/article/abs/nonsingular-horizontal-position-representation/9DA5AFB5EC91CFF0E755C18BBAA37171>`_.

.. [GadeHome] Kenneth Gade. The N-vector page.
  https://www.ffi.no/en/research/n-vector/.

.. [GadeExplained] Kenneth Gade. N-vector explained.
  https://www.ffi.no/en/research/n-vector/n-vector-explained.

.. [WikipediaECEF] Earth-centered, Earth-fixed coordinate system.
  Wikipedia.
  https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system

.. [MathSE1] Angular distance from radius of sphere and distance along circumference.
  Math StackExchange.
  https://math.stackexchange.com/q/3316069.

.. [MathSE2] Angular distance from radius of sphere and distance along circumference.
  Math StackExchange.
  https://math.stackexchange.com/q/3326426.

.. [GadeDownloads] Kenneth Gade. N-vector downloads.
  https://www.ffi.no/en/research/n-vector/n-vector-downloads.
  `Source code (Git) <https://github.com/FFI-no/n-vector>`_.

.. [Brodtkorb] Per A. Brodtkorb. nvector.
  `Documentation <https://nvector.readthedocs.io/>`_.
  `PyPI <https://pypi.org/project/nvector>`_.
  `Source code (Git) <https://github.com/pbrod/nvector>`_.

.. [Veness] Chris Veness. Vector-based spherical geodesy.
  https://www.movable-type.co.uk/scripts/latlong-vectors.html.
  `Archive 1 <https://web.archive.org/web/20241126215432/https://www.movable-type.co.uk/scripts/latlong-vectors.html>`_.
  `Archive 2 <https://archive.is/tH8Fm>`_.

.. [Brouwers] Jean Brouwers. PyGeodesy.
  `Documentation <https://mrjean1.github.io/PyGeodesy/>`_.
  `PyPI <https://pypi.org/project/PyGeodesy/>`_.
  `Source code (Git) <https://github.com/mrJean1/PyGeodesy>`_.

.. [Spinielli] Enrico Spinielli. nvctr.
  `CRAN <https://cran.r-project.org/package=nvctr>`_.
  `Source code (Git) <https://github.com/euctrl-pru/nvctr>`_.
"""

import warnings
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias, TypeVar
from typing_extensions import TypeIs

import numpy as np
from numpy.typing import NBitBase, NDArray


_B = TypeVar("_B", bound=NBitBase)
FloatArray: TypeAlias = NDArray[np.floating[_B]]
AngleArray: TypeAlias = FloatArray[_B]
NVectorArray: TypeAlias = FloatArray[_B]


class PerformanceWarning(Warning):
    pass


def _validate_dtype(*vs: NDArray[Any]) -> TypeIs[FloatArray[Any]]:
    for v in vs:
        if not np.issubdtype(v.dtype, np.floating):
            raise ValueError("Input is not a valid n-vector array dtype.")
    return True


def _validate_shape(*vs: NVectorArray[Any]) -> None:
    for v in vs:
        if v.ndim == 0 or v.shape[0] != 3:
            raise ValueError("Input is not a valid n-vector array shape.")
    return True


# def _convert_dtype(*vs: NDArray[Any]) -> FloatArray[np.float64]:
#     out = []
#     for v in vs:
#         if not np.issubdtype(v.dtype, np.floating):
#             vv = v.astype(np.float64)
#         if vv.base is not v.base:
#             warnings.warn(PerformanceWarning("DType has been promoted to float64, a copy has been created!"))
#         out.append(vv)
#     return tuple(out)


def _promote_shape(*vs: NVectorArray[_B]) -> tuple[NVectorArray[_B]]:
    out = []
    for v in vs:
        if v.ndim == 0:
            raise ValueError("Size-0 arrays are not supported and cannot be shape-promoted.")
        vv = v[:, np.newaxis] if v.ndim == 1 else v
        if vv.base is not v.base:
            warnings.warn(PerformanceWarning("A copy was created when promoting a 1-d vector!"))
        out.append(vv)
    return tuple(out)


def _preprocess(*vs: NDArray[Any]) -> NVectorArray[np.floating[Any]]:
    # N.B. these assertions can never fail
    assert _validate_dtype(*vs)
    assert _validate_shape(*vs)
    return _promote_shape(*vs)


def _dot_each(u: NVectorArray[_B], v: NVectorArray[_B]) -> NVectorArray[_B]:
    r"""Dot product of every pair of corresponding n-vectors."""
    return np.sum(u * v, axis=0)


def _cross_each(u: NVectorArray[_B], v: NVectorArray[_B]) -> NVectorArray[_B]:
    r"""Cross product of every pair of corresponding n-vectors."""
    return np.cross(u, v, axis=0)


def _norm_each(u: NVectorArray[_B]) -> NVectorArray[_B]:
    return np.linalg.norm(u, axis=0)


DTypeA = TypeVar("DTypeA", bound=type)
DTypeB = TypeVar("DTypeB", bound=type)

def _broadcast_cartesian(u: NDArray[DTypeA], v: NDArray[DTypeB]) -> tuple[NDArray[DTypeA], NDArray[DTypeB]]:
    u_slices = (slice(None),) * u.ndim + (None,) * (v.ndim - 1)
    v_slices = (slice(None),) + (None,) * (u.ndim - 1) + (slice(None),) * (v.ndim - 1)
    return u[u_slices], v[v_slices]


def _dot_cartesian(u: NVectorArray[_B], v: NVectorArray[_B]) -> NVectorArray[_B]:
    r"""Dot product of every pair of n-vectors in the Cartesian product of n-vectors."""
    return np.tensordot(u, v, axes=([0], [0]))


def _normalize(v: NVectorArray[_B], out: NVectorArray[_B] | None = None) -> NVectorArray[_B]:
    r"""Scale ("normalize") a vector to unit norm.

    :param v: The vector(s) to normalize.
    :param out: Same as ``out=`` in Numpy itself.

    :returns: Normalized vector(s) of the same dtype and shape as ``v``.

    .. warning:: This function might return ``inf`` or ``nan`` if the norm is 0 along
      the given axes.
    """
    # This function is based closely on the Python implementation by Brodtkorb.

    if out is None:
        out = np.empty_like(v)

    # Scale down before computing the norm, to avoid precision loss.
    v_max = np.max(np.abs(v), axis=0)
    # Add a tiny offset (the smallest non-subnormal) to avoid divide-by-zero.
    tiny = np.finfo(v.dtype).tiny
    if np.any(np.abs(v_max) <= tiny):
        v_max += tiny
    np.divide(v, v_max, out=out)

    # Compute the norm(s) along the given axes.
    v_norm: NVectorArray[_B] = np.linalg.norm(out, axis=0)

    # Normalize along the remaining axes.
    # NOTE: This might result in `inf` or `nan` if the norm is 0 along any axis.
    np.divide(out, v_norm, out=out)

    return out


def _as_scalar(x: FloatArray[_B]) -> FloatArray[_B] | np.float64 | float:
    return x.item() if x.size == 1 else x


def _squeezable(x: NDArray[Any]) -> bool:
    r"""Check whether "squeezing" an array would have any effect."""
    return sum(ax_len > 1 for ax_len in x.shape) == 1


def _dot_1d_scalar(x: FloatArray[_B], y: FloatArray[_B]) -> float:
    r"""Scalar dot product of two 1-D column vectors."""
    if not _squeezable(x) or not _squeezable(y):
        raise ValueError("Inputs must have exactly one non-trivial axis.")
    result: float = np.inner(x.squeeze(), y.squeeze())
    return result


def lonlat_to_nvector(
    lon: FloatArray[_B] | float,
    lat: FloatArray[_B] | float,
    radians: bool = False,
) -> NVectorArray[_B]:
    r"""Convert longitude and latitude to n-vector.

    :param lon: Longitude, in degrees or radians (see ``radians=``). Should be
      broadcast-compatible with ``lat``.
    :param lat: Latitude, in degrees or radians (see ``radians=``). Should be
      broadcast-compatible with ``lon``.
    :param radians: If true, input is expected in radians. Otherwise, input is expected
      in degrees (the default setting).

    :returns: An n-vector array of shape ``(3, *shape)`` where ``shape`` is the result of
      broadcasting ``lon`` and ``lat``. The new leading axis represents the
      n-vector coordinates in the "E" reference frame.
    """
    if not radians:
        lon = np.radians(lon)
        lat = np.radians(lat)

    lon, lat = np.atleast_1d(lon, lat)
    shape = np.broadcast_shapes(lon.shape, lat.shape)
    lon = np.broadcast_to(lon, shape)
    lat = np.broadcast_to(lat, shape)

    dtype = np.result_type(lon, lat)

    nvect: NVectorArray[_B] = np.empty((3, *shape), dtype=dtype, order="F")
    _cos_lat = np.cos(lat)
    np.stack(
        (
            # x: points to the North Pole
            np.sin(lat),
            # y: points to 90°E, 0°N
            _cos_lat * np.sin(lon),
            # z: points to 0°E, 0°N
            -_cos_lat * np.cos(lon),
        ),
        axis=0,
        out=nvect,
    )

    return nvect


def nvector_to_lonlat(nvect: NVectorArray[_B], radians: bool = False) -> tuple[FloatArray[_B], FloatArray[_B]]:
    r"""Convert n-vector to longitude and latitude.

    :param nvect: n-vector array. First dimension must be size 3.
    :param radians: If true, output is returned in radians. Otherwise, output is returned
      in degrees (the default setting).

    :returns: A pair of arrays of shape ``shape`` where ``(3, *shape)`` is the shape of
      the input. The first array is longitude, the second is latitude. Longitude at the
      North and South Poles is arbitrary, and depends on whatever Numpy ``atan2(0,0)``
      returns.
    """
    (nvect,) = _preprocess(nvect)

    lon = np.arctan2(nvect[1, ...], -nvect[2, ...])

    equatorial_component = np.sqrt(np.square(nvect[1, ...]) + np.square(nvect[2, ...]))
    lat = np.arctan2(nvect[0, ...], equatorial_component)

    if not radians:
        lon = np.degrees(lon)
        lat = np.degrees(lat)

    return lon, lat


def nvector_great_circle_normal(v1: NVectorArray[_B], v2: NVectorArray[_B]) -> NVectorArray[_B]:
    r"""Compute the unit normal vector of the great-circle plane formed by two n-vectors.

    .. warning:: For antipodal points (points on opposite sides of the Earth), the
      great-circle plane is undefined, so this function might return unstable results.
    """
    v1, v2 = _preprocess(v1, v2)
    n = _cross_each(v1, v2)
    _normalize(n, out=n)
    return n


def nvector_arc_angle(v1: NVectorArray[_B], v2: NVectorArray[_B]) -> FloatArray[_B]:
    r"""Compute the arc angle between two n-vectors.

    Note that this is the great-circle distance on the unit sphere.
    To get great-circle/geodesic distance on the surface of a non-unit sphere, multiply
    this result by the sphere radius.
    """
    v1, v2 = _preprocess(v1, v2)

    # https://math.stackexchange.com/q/1143354/117452
    # https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf
    # TODO: Assuming inputs are theoretically unit norm, how safe is it to assume `n1 == n2 == 1.0`?
    n1 = _norm_each(v1)
    n2 = _norm_each(v2)
    result = 2 * np.atan2(_norm_each(n2 * v1 - n1 * v2), _norm_each(n2 * v1 + n1 * v2))
    # result = np.atan2(_norm_each(_cross_each(v1, v2)), _dot_each(v1, v2))

    return _as_scalar(result)


def nvector_cross_track_distance(v1: NVectorArray[_B], v2: NVectorArray[_B], u: NVectorArray[_B]) -> FloatArray[_B]:
    r"""Compute the arc angle between ``u`` and its closest point along the geodesic between ``v1`` and ``v2``.

    Note that this is the great-circle distance on the unit sphere.
    To get distance on the surface of a non-unit sphere, multiply this result by the
    sphere radius.

    See N-vector Example 10: https://www.ffi.no/en/research/n-vector/#example_10
    """
    _validate(v1, v2, u)
    v1, v2, u = _promote_shape(v1, v2, u)
    n = nvector_great_circle_normal(v1, v2)
    return nvector_cross_track_distance_from_normal(n, u)


def nvector_cross_track_distance_from_normal(n: NVectorArray[_B], u: NVectorArray[_B]) -> FloatArray[_B]:
    r"""Compute the arc angle between ``u`` and its closest point the geodesic between ``v1`` and ``v2``.

    The difference between this and ``nvector_cross_track_distance`` is that here it is
    assumed that we already know the great-circle-plane normal vector ``n``. Useful to
    compute the cross-track distances of many points with respect to many tracks,
    if we want to pre-compute and save the normal vector of each track.

    To get distance on the surface of a non-unit sphere, multiply this result by the
    sphere radius.

    See N-vector Example 10: https://www.ffi.no/en/research/n-vector/#example_10
    """
    return nvector_arc_angle(n, u) - np.pi / 2


def nvector_direct(
    initial_position_nvect: NVectorArray[_B],
    distance_rad: float | FloatArray[_B],
    initial_azimuth_rad: float | FloatArray[_B],
) -> NVectorArray[_B]:
    r"""Solve the "forward" or "direct" geodesic problem.

    Computes a new location from a starting point, a distance, and an initial azimuth.

    :param initial_position_nvect: ``(3, N)``. Starting point, n-vector.
    :param distance_rad: Scalar, ``(1,)``, or ``(N,)``. Arc angle (distance), radians.
    :param initial_azimuth_rad: Scalar, ``(1,)``, ``(N,)``, or ``(K, 1, 1)``. Initial
        bearing, a.k.a. forward azimuth, radians.

    :returns: New n-vector positions computed from the initial positions, distances and
        azimuths. Shape will be ``(3, N)`` or ``(K, 3, N)``, depending on input shapes.

    .. note:: If ``initial_azimuth_rad`` is shape ``(K, 1, 1)``, the result will be an
        "outer product" over initial positions and azimuths, of shape ``(K, 3, N)``.
        Otherwise, ``initial_position_rad`` is expected to be compatible with
        ``initial_position_nvect``, and the result will have shape ``(3, N)``.

    See N-Vector Example 2: https://www.ffi.no/en/research/n-vector/#example_2
    """
    (initial_position_nvect,) = _preprocess(initial_position_nvect)
    distance_rad, initial_azimuth_rad = np.atleast_1d(distance_rad, initial_azimuth_rad)

    # The first unit basis vector in the ECEF "E" frame.
    unit_basis_0 = np.asarray([1, 0, 0])

    # Construct the initial direction vector defined by the forward azimuth.
    #  1. Use the right-hand-rule to obtain a unit vector that points exactly East.
    #  2. Use the right-hand-rule to obtain a unit vector that points exactly North.
    #     We don't need to normalize this result, because the inputs are already known
    #     to be orthogonal unit vectors, so their cross product must be also a unit
    #     vector. See e.g. https://math.stackexchange.com/a/23261.
    #  3. Compute the components of the direction vector decomposed into the East and
    #     North unit vectors, and add them to find the direction vector itself.
    # FIXME: Apparently unit_east can be numerically unstable very close to poles.
    unit_east = _normalize(_cross_each(unit_basis_0, initial_position_nvect))
    unit_north = _cross_each(initial_position_nvect, unit_east)
    initial_direction = (
        unit_north * np.cos(initial_azimuth_rad) +
        unit_east * np.sin(initial_azimuth_rad)
    )

    # Compute the location along the great-circle path of the direction vector.
    final_position: NVectorArray[_B] = (
        initial_position_nvect * np.cos(distance_rad) +
        initial_direction * np.sin(distance_rad)
    )

    return final_position


# TODO: Does this work with clockwise polygons?
# TODO: Add optional input validation.
# TODO: Contains vs. covers -- what if the pole is a vertex?
def nvector_polygon_contains_pole(polygon_nvects: FloatArray[Any]) -> tuple[bool, bool]:
    r"""Check if a polygon (with n-vector vertices) contains the North or South Pole.

    :param polygon_nvects: A "polygon" formed by an array of n-vectors, of shape
        ``(3, ...)``. The first axis corresponds to n-vector coordinates.

    :returns: A pair of Boolean values. The first value indicates if the polygon
        contains the North Pole, and the second indicates if the polygon contains the
        South Pole.

    .. warning:: The input is not actually checked for being a valid, convex, simple, or
        counter-clockwise (right-hand rule) polygon. This function only checks that each
        pole is on the same side of the great-circle plane formed by successive pairs of
        vertices.
    """

    (polygon_nvects,) = _preprocess(polygon_nvects)

    if polygon_nvects.ndim != 2:
        raise ValueError("Input must be 2-dimensional")

    if polygon_nvects.shape[0] != 3:
        raise ValueError("First axis must have length 3.")

    n_vertices = polygon_nvects.shape[1]

    # Construct n-vectors for the North and South poles in reference frame "e".
    northpole_nvect = np.eye(3, 1)
    southpole_nvect = -1.0 * northpole_nvect

    # We "accumulate" boolean values by AND-ing them together.
    # We stop when one of these accumulators becomes False.
    contains_northpole = True
    contains_southpole = True

    for k1 in range(n_vertices):
        # Wrap around: n_vertices+1 → 0 , n_vertices+2 → 1 , etc.
        k2 = (k1 + 1) % n_vertices

        # These need to be 3×1 for `nvector_great_circle_normal`.
        u1 = polygon_nvects[:, [k1]]
        u2 = polygon_nvects[:, [k2]]

        # This is the normal vector of the plane corresponding to the great circle
        # between u1 and u2. Recall that a "great circle" is defined by the plane that
        # includes u1, u2, and the center of the Earth, i.e. (0, 0, 0) in the ECEF
        # coordinate system. The great circle itself is the where this plane intersects
        # with the surface of the earth.
        edge_greatcircle_normal = nvector_great_circle_normal(u1, u2)

        # If the dot product of the pole and the great-circle-plane-normal-vector is
        # positive, then the latter points (at least slightly) in the direction of the
        # former, and therefore the former is on the "left" of the current polygon edge.
        # This is very hard to describe in words, but it makes sense as a consequence of
        # the right-hand rule for cross products.

        cos_angle_to_northpole = _dot_1d_scalar(edge_greatcircle_normal, northpole_nvect)
        normal_points_at_northpole = bool(cos_angle_to_northpole > 0)
        contains_northpole = contains_northpole and normal_points_at_northpole

        cos_angle_to_southpole = _dot_1d_scalar(edge_greatcircle_normal, southpole_nvect)
        normal_points_at_southpole = bool(cos_angle_to_southpole > 0)
        contains_southpole = contains_southpole and normal_points_at_southpole

        if not contains_northpole and not contains_southpole:
            break

    return (contains_northpole, contains_southpole)
