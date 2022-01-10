# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import xarray as xr

from bespin.core.filter import FilterBase


class LonWrap(FilterBase):
    """Make sure longitude is between 0 and 360.
    """

    def __init__(self):
        super().__init__()

    def filter(self, data: xr.Dataset,  **kwargs) -> xr.Dataset:
        if 'longitude' in data.variables:
            msk = data.variables['longitude'] < 0
            data.variables['longitude'][msk] += 360.0
        return data
