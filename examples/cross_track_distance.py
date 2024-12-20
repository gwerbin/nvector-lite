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

import numpy as np
import pyproj
from nvector_lite import lonlat_to_nvector, nvector_cross_track_distance


## Input data

earth_radius_km = 6371.0

origin_lons = np.asarray([-76.21378151363640, -75.456901765848710])
origin_lats = np.asarray([ 43.46001114816093,  43.193926425093935])
destin_lons = np.asarray([-73.77150782932506, -73.66740169301255])
destin_lats = np.asarray([ 43.08511132138398,  43.31073783085333])
point_lons = np.asarray([-75.44564704840916, -75.07705505227482, -74.60154324046780, -74.17949133649684])
point_lats = np.asarray([ 43.52124976781056,  43.17341009622575,  43.32097375630951,  43.59261634178711])


## Nvector solution

origin_nvects = lonlat_to_nvector(origin_lons, origin_lats)
destin_nvects = lonlat_to_nvector(destin_lons, destin_lats)
point_nvects = lonlat_to_nvector(point_lons, point_lats)

assert origin_nvects.shape == (3, 2)
assert destin_nvects.shape == (3, 2)
assert point_nvects.shape == (3, 4)

xt_nvect = nvector_cross_track_distance(
    origin_nvects,
    destin_nvects,
    point_nvects,
)
xt_nvect *= earth_radius_km

assert xt_nvect.shape == (2, 4)


## Lon/Lat solution

geod = pyproj.Geod(a=1.0, f=0.0)

path_bearing, _, path_distance = geod.inv(
    origin_lons,
    origin_lats,
    destin_lons,
    destin_lats,
)
path_bearing = np.radians(path_bearing)

point_bearing, _, point_distance = geod.inv(
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
xt_lonlat *= earth_radius_km


## Test it

np.testing.assert_allclose(xt_nvect, xt_lonlat)
