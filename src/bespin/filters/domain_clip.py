# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pygeos

import xarray as xr

from bespin.core.filter import FilterBase

class DomainClip(FilterBase):
    """ensure that observations are within a given clipping region."""

    # TODO handle cases where polygon crosses 0/360 longitude

    def __init__(self, *args):
        # parse lat/lon pairs
        lons = []
        lats = []
        for arg in args:
            ll = arg.split(',')
            assert(len(ll) == 2)
            lat, lon = [float(x) for x in ll]
            if lon < 0.0:
                lon += 360
            lons.append(lon)
            lats.append(lat)
        self.domain = pygeos.polygons(pygeos.linearrings(lons, lats))
        super().__init__()


    def filter(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        lat_dim = 'latitude'
        lon_dim = 'longitude'

        lats = data[lat_dim].data
        lons = data[lon_dim].data
        points = pygeos.points(lons, lats)

        mask = xr.zeros_like(data[lat_dim])
        mask[pygeos.contains(self.domain, points)] = 1

        data = data.where(mask == 1, drop=True)

        return data