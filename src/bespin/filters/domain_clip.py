# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from shapely.geometry import Point, Polygon

import xarray as xr

from bespin.core.filter import FilterBase

class DomainClip(FilterBase):
    """ensure that observations are within a given clipping region."""

    # TODO this is slow, use a different library?
    # TODO handle cases where polygon crosses 0/360 longitude

    def __init__(self, *args):
        # parse lat/lon pairs
        points = []
        for arg in args:
            ll = arg.split(',')
            assert(len(ll) == 2)
            lat, lon = [float(x) for x in ll]
            if lon < 0.0:
                lon += 360
            points.append(Point(lat, lon))
        self.domain = Polygon(points)
        super().__init__()


    def filter(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        lat_dim = 'latitude'
        lon_dim = 'longitude'

        lats = data[lat_dim].data
        lons = data[lon_dim].data
        mask = xr.zeros_like(data[lat_dim])

        # test each point
        for i in range(len(lats)):
            point = Point(lats[i], lons[i])
            mask[i] = 1 if self.domain.contains(point) else 0

        data = data.where(mask == 1, drop=True)

        return data