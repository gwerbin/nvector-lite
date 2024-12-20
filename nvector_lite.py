r"""Lightweigh n-vector functionality.

This module is heavily based on the original Nvector Python library, which is
distributed under a BSD 3-Clause license.

This implementation is hard-coded to use a spherical Earth model in the the "E"
reference frame from Gade (2010), for implementation simplicity. The choice of reference
frame does not affect any of the results of geodesic calculations. However it is
important to know the reference frame when converting between n-vector and other
representations, and when interoperating with other libraries.

An "n-vector" is a 3-vector. In our implementation, an array of shape ``(3, N)``
represents a collection of ``N`` n-vectors.

In certain cases, only "scalar" n-vector arrays are accepted, i.e. arrays strictly of
shape ``(3, 1)``.

1-d arrays of shape ``(3,)`` are considered scalar n-vectors and are upgraded to 2-d
arrays of shape ``(3,1)`` (column vectors).

Multidimensional collections of n-vectors should also work, e.g. ``(3, K, M, N)``
representing a K×M×N array of n-vectors. But that case is not tested, and should be
considered experimental at best.

N-vectors in spaces other than 3 dimensions are not supported.


Background
----------

The "n-vector" representation of horizontal position represents each point on the
surface of the earth as the unit normal vector to that point on the earth. "Height" or
"altitude" is represented by scaling these unit vectors. This representation is valid
for any spherical or ellipsoidal Earth model, as long as the normal vectors have
a closed-form expression.

The actual vectors resulting from this representation are dependent on choosing
a particular coordinate reference frame. [Gade 2010] prefers the "E" reference frame,
which is an Earth-centered and Earth-fixed (ECEF) or "geocentric" frame.

x is the Earth's rotation axis.
The North Pole (90°N, "undefined" °E) is ``(1,0,0)``.
y and z form the equatorial plane.
The point 0°N,0°E is ``(0,0,1)`` and 0°N,90°E is ``(0,1,0)``.

This frame is slightly unintuitive compared to our usual mental model of a globe, where
North is "up". But it has several desirable properties for motion tracking of objects
(especially airplanes), and it is the reference frame used throughout [Gade 2010].

See also: https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system

Note that in the n-vector representation, great-circle distances are represented as arc
angles (in radians). In a spherical model, an arc angle is simply the fraction of the
Earth circumference (2π) traveled over the great-circle path. Therefore arc angles can
be converted to surface distances by multiplying by the sphere radiug by the sphere
radius. See:

• https://math.stackexchange.com/q/3316069
• https://math.stackexchange.com/q/3326426


References
-----------

``[Gade 2010]``:
Gade, K. (2010). A Non-singular Horizontal Position Representation. Journal of Navigation, 63, 395-417.

• DOI: <https://doi.org/10.1017/S0373463309990415>
• Full text PDF: <https://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>
• Journal: <https://www.cambridge.org/core/journals/journal-of-navigation/article/abs/nonsingular-horizontal-position-representation/9DA5AFB5EC91CFF0E755C18BBAA37171>

``[Nvector 2022]``:
Brodtkorb, Per A. (2022). nvector (Python library), version 0.7.7.

• PyPI: <https://pypi.org/project/nvector>
• Source code: <https://github.com/pbrod/nvector>
• Documentation: <https://nvector.readthedocs.io/>
"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NBitBase, NDArray


_B = TypeVar("_B", bound=NBitBase)


def _validate(v: NDArray[np.floating[_B]]) -> None:
    if v.ndim == 0 or v.shape[0] != 3:
        raise ValueError("Input is not a valid n-vector array.")


def _promote_shape(v: NDArray[np.floating[_B]]) -> None:
    if v.ndim == 1:
        v = v[:, np.newaxis]
    return v


def _dot_each(u: NDArray[np.floating[_B]], v: NDArray[np.floating[_B]],) -> NDArray[np.floating[_B]]:
    r"""Dot product of every pair of corresponding n-vectors."""
    return (u * v).sum(axis=0)


def _cross_each(u: NDArray[np.floating[_B]], v: NDArray[np.floating[_B]],) -> NDArray[np.floating[_B]]:
    r"""Cross product of every pair of corresponding n-vectors."""
    return np.cross(u, v, axis=0)


def _norm_each(u: NDArray[np.floating[_B]]) -> NDArray[np.floating[_B]]:
    return np.linalg.norm(u, axis=0)


def _broadcast_cartesian(u, v):
    u_slices = (slice(None),) * u.ndim + (None,) * (v.ndim - 1)
    v_slices = (slice(None),) + (None,) * (u.ndim - 1) + (slice(None),) * (v.ndim - 1)
    return u[u_slices], v[v_slices]


def _dot_cartesian(u: NDArray[np.floating[_B]], v: NDArray[np.floating[_B]],) -> NDArray[np.floating[_B]]:
    r"""Dot product of every pair of n-vectors in the Cartesian product of n-vectors."""
    return np.tensordot(u, v, axes=([0], [0]))


def normalize(
    v: NDArray[np.floating[_B]], axis: int | Sequence[int] | None = None
) -> NDArray[np.floating[_B]]:
    r"""Scale ("normalize") a vector to unit norm.

    :param v: The vector(s) to normalize.
    :param axis: The axis to treat as individual vectors.
        This follows all the usual Numpy ``axis=`` rules.

    :returns: Normalized vector(s) of the same dtype and shape as ``v``.

    .. warning:: This function might return ``inf`` or ``nan`` if the norm is 0 along
        the given axes.
    """
    # The smallest "normal" floating-point number. Numbers smaller than this can suffer
    # serious performance degradation on most processors.
    tiny = np.finfo(v.dtype).tiny

    # Scale down before computing the norm, to avoid precision loss.
    v_max = np.max(np.abs(v), axis=axis, keepdims=True)
    # Add a tiny offset to avoid introducing divide-by-zero.
    if np.any(np.abs(v_max) <= tiny):
        v_max += tiny
    w: NDArray[np.floating[_B]] = v / v_max

    # Compute the norm(s) along the given axes.
    w_norm: NDArray[np.floating[_B]] = np.linalg.norm(  # type:ignore[assignment]
        w, axis=axis, keepdims=True                     # type:ignore[arg-type]
    )

    # Normalize along the remaining axes.
    # NOTE: This might result in `inf` or `nan` if the norm is 0 along any axis.
    u = w / w_norm

    return u


def lonlat_to_nvector(
    lon: NDArray[np.floating[_B]] | float,
    lat: NDArray[np.floating[_B]] | float,
    radians: bool = False,
) -> NDArray[np.floating[_B]]:
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
    lon, lat = np.broadcast_arrays(lon, lat)

    nvect: NDArray[np.floating[_B]] = np.stack(
        (
            # x: points to the North Pole (undefined °E, 0°N).
            np.sin(lat),
            # y: points to 90°E, 0°N.
            np.cos(lat) * np.sin(lon),
            # z: points to 0°E, 0°N.
            -np.cos(lat) * np.cos(lon),
        ),
        axis=0,
    )

    return nvect


def nvector_to_lonlat(
    nvect: NDArray[np.floating[_B]], radians: bool = False
) -> tuple[NDArray[np.floating[_B]], NDArray[np.floating[_B]]]:
    r"""Convert n-vector to longitude and latitude.

    :param nvect: n-vector array. First dimension must be size 3.
    :param radians: If true, output is returned in radians. Otherwise, output is returned
        in degrees (the default setting).

    :returns: A pair of arrays of shape ``shape`` where ``(3, *shape)`` is the shape of
        the input. The first array is longitude, the second is latitude.
    """
    _validate(nvect)
    nvect = _promote_shape(nvect)

    lon = np.arctan2(nvect[1, ...], -nvect[2, ...])

    equatorial_component = np.sqrt(np.square(nvect[1, ...]) + np.square(nvect[2, ...]))
    lat = np.arctan2(nvect[0, ...], equatorial_component)

    if not radians:
        lon = np.degrees(lon)
        lat = np.degrees(lat)

    return lon, lat


def nvector_great_circle_normal(
    v1: NDArray[np.floating[_B]],
    v2: NDArray[np.floating[_B]]
) -> NDArray[np.floating[_B]]:
    r"""Compute the unit normal vector of the great-circle plane formed by two n-vectors."""
    _validate(v1)
    _validate(v2)
    v1 = _promote_shape(v1)
    v2 = _promote_shape(v2)
    return normalize(_cross_each(v1, v2))


def nvector_arc_angle(
    v1: NDArray[np.floating[_B]],
    v2: NDArray[np.floating[_B]],
    outer: bool = False,
) -> NDArray[np.floating[_B]]:
    r"""Compute the arc angle between two n-vectors on the unit sphere.

    Note that this is the great-circle distance on the unit sphere.
    To get great-circle/geodesic distance on the surface of a non-unit sphere, multiply
    this result by the sphere radius.
    """
    _validate(v1)
    _validate(v2)
    v1 = _promote_shape(v1)
    v2 = _promote_shape(v2)
    return np.atan2(
        _norm_each(_cross_each(v1, v2)),
        _dot_each(v1, v2),
    )


def nvector_cross_track_distance(
    v1: NDArray[np.floating[_B]],
    v2: NDArray[np.floating[_B]],
    u: NDArray[np.floating[_B]],
) -> NDArray[np.floating[_B]]:
    r"""Compute the cross-track distance of ``u`` with respect to the geodesic between ``v1`` and ``v2``.

    Note that this is the great-circle distance on the unit sphere.
    To get distance on the surface of a non-unit sphere, multiply this result by the
    sphere radius.

    See N-vector Example 10: https://www.ffi.no/en/research/n-vector/#example_10
    """
    _validate(v1)
    _validate(v2)
    _validate(u)
    v1 = _promote_shape(v1)
    v2 = _promote_shape(v2)
    u = _promote_shape(u)
    n = nvector_great_circle_normal(v1, v2)
    return nvector_cross_track_distance_from_normal(n, u)


def nvector_cross_track_distance_from_normal(
    n: NDArray[np.floating[_B]],
    u: NDArray[np.floating[_B]],
) -> NDArray[np.floating[_B]]:
    r"""Compute the cross-track distance of ``u`` with respect to the geodesic between ``v1`` and ``v2``.

    The difference between this and ``nvector_cross_track_distance`` is that here it is
    assumed that we already know the great-circle-plane normal vector ``n``. Useful to
    compute the cross-track distances of many points with respect to many tracks,
    if we want to pre-compute and save the normal vector of each track.

    To get distance on the surface of a non-unit sphere, multiply this result by the
    sphere radius.

    See N-vector Example 10: https://www.ffi.no/en/research/n-vector/#example_10
    """
    # Array shapes:
    #   n: (3, *shape1)
    #   u: (3, *shape2)
    #
    # We want this to be "double vectorized", obtaining a result for each element of
    # u and each element of n. That is, something like this:
    #   for i, x in enumerate(n):
    #     for j, y in enumerate(u):
    #       result[i, j] = compute(x, y)
    return nvector_arc_angle(*_broadcast_cartesian(n, u)) - np.pi / 2


def nvector_direct(
    initial_position_nvect: NDArray[np.floating[_B]],
    distance_rad: float | NDArray[np.floating[_B]],
    initial_azimuth_rad: float | NDArray[np.floating[_B]],
) -> NDArray[np.floating[_B]]:
    r"""Solve the "forward" or "direct" geodesic problem.

    Computes a new location from a starting point, a distance, and an initial azimuth.

    :param initial_position_nvect: ``(3, N)``. Starting point, n-vector.
    :param distance_rad: Scalar, ``(1,)``, or ``(N,)``. Arc angle (distance), radians.
    :param initial_azimuth_rad: Scalar, ``(1,)``, ``(N,)``, or ``(K, 1, 1)``. Initial
        bearing, a.k.a. forward azimuth, radians.

    :returns: New n-vector positions computed from the initial positions, distances and
        azimuths. Shape will be ``(3, N)`` or ``(K, 3, N)``, depending on input shapes.

    .. warning:: n-vectors are expected to be unit vectors. This function does **not**
        check that inputs have unit norm. However, n-vectors produced by
        :func:`lonlat_to_nvector` will be properly normalized.

    .. note:: If ``initial_azimuth_rad`` is shape ``(K, 1, 1)``, the result will be an
        "outer product" over initial positions and azimuths, of shape ``(K, 3, N)``.
        Otherwise, ``initial_position_rad`` is expected to be compatible with
        ``initial_position_nvect``, and the result will have shape ``(3, N)``.
    """
    initial_position_nvect, distance_rad, initial_azimuth_rad = np.atleast_1d(
        initial_position_nvect, distance_rad, initial_azimuth_rad
    )

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
    unit_east = normalize(np.cross(unit_basis_0, initial_position_nvect, axis=0), axis=0)
    unit_north = np.cross(initial_position_nvect, unit_east, axis=0)
    initial_direction = (
        unit_north * np.cos(initial_azimuth_rad) +
        unit_east * np.sin(initial_azimuth_rad)
    )

    # Compute the location along the great-circle path of the direction vector.
    final_position: NDArray[np.floating[_B]] = (
        initial_position_nvect * np.cos(distance_rad) +
        initial_direction * np.sin(distance_rad)
    )

    return final_position


def _squeezable(x: NDArray[Any]) -> bool:
    r"""Check whether "squeezing" an array would have any effect."""
    return sum(ax_len > 1 for ax_len in x.shape) == 1


def _dot_1d(x: NDArray[np.floating[_B]], y: NDArray[np.floating[_B]]) -> float:
    r"""Scalar dot product of column vectors, or any other "squeezable" arrays."""
    if not _squeezable(x) or not _squeezable(y):
        raise ValueError("Inputs must have exactly one non-trivial axis.")
    result: float = np.dot(x.squeeze(), y.squeeze())
    return result


# TODO: Does this work with clockwise polygons?
# TODO: Add optional input validation.
def nvector_polygon_contains_pole(
    polygon_nvects: NDArray[np.floating[Any]],
) -> tuple[bool, bool]:
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

    if polygon_nvects.ndim != 2:
        raise ValueError("Input must be 2-dimensional")

    if polygon_nvects.shape[0] != 3:
        raise ValueError("First axis must have length 3.")

    n_vertices = polygon_nvects.shape[1]

    # Construct n-vectors for the North and South poles in reference frame "e".
    northpole_nvect = np.eye(3, 1)
    southpole_nvect = -1.0 * northpole_nvect

    # We "accumulate" boolean values by AND-ing them together. As soon as these
    # accumulators becomes False, we stop iterating.
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

        cos_angle_to_northpole = _dot_1d(edge_greatcircle_normal, northpole_nvect)
        normal_points_at_northpole = bool(cos_angle_to_northpole > 0)
        contains_northpole = contains_northpole and normal_points_at_northpole

        cos_angle_to_southpole = _dot_1d(edge_greatcircle_normal, southpole_nvect)
        normal_points_at_southpole = bool(cos_angle_to_southpole > 0)
        contains_southpole = contains_southpole and normal_points_at_southpole

        if not contains_northpole and not contains_southpole:
            break

    return (contains_northpole, contains_southpole)
