# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import xarray as xr

from bespin.core.filter import FilterBase


class Copy(FilterBase):
    """copy a variable in the dataset from `var_src` to `var_dst`.
    """

    def __init__(self, var_src: str, var_dst: str):
        self.var_src = var_src
        self.var_dst = var_dst
        super().__init__()

    def filter(self, data: xr.Dataset) -> xr.Dataset:
        data = data.update({self.var_dst: data.data_vars[self.var_src]})
        return data
