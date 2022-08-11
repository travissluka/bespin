# (C) Copyright 2022-2022 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import xarray as xr

from bespin.core.filter import FilterBase


class Rename(FilterBase):
    """rename a variable in the dataset from `var_src` to `var_dst`.
    """

    per_variable = True

    def __init__(self, var_src: str, var_dst: str):
        self.var_src = var_src
        self.var_dst = var_dst
        super().__init__()

    def filter(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        var_src = self.var_src.format(**kwargs)
        var_dst = self.var_dst.format(**kwargs)
        data = data.update({var_dst: data.data_vars[var_src]})
        data = data.drop_vars(var_src)
        return data
