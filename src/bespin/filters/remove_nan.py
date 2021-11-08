# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import xarray as xr

from bespin.core.filter import FilterBase


class RemoveNan(FilterBase):

    per_variable = True

    def filter(self, data: xr.Dataset) -> xr.Dataset:
        for d in data.dims:
            data = data.dropna(d)

        return data
