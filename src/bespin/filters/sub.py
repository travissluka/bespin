# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import xarray as xr

from bespin.core.filter import FilterBase


class Sub(FilterBase):
    """copy a variable in the dataset from `var_src` to `var_dst`.
    """

    per_variable = True

    def __init__(self, src1: str, src2: str, dst: str):
        self.src1 = src1
        self.src2 = src2
        self.dst = dst
        super().__init__()

    def filter(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        src1 = f"{self.src1}/{kwargs['variable']}"
        src2 = f"{self.src2}/{kwargs['variable']}"
        dst = f"{self.dst}/{kwargs['variable']}"
        data = data.update({dst: data.data_vars[src1] - data.data_vars[src2]})
#        print(f'DBG\n{src1}, {src2}. {dst}')
 #       sys.exit(1)
#        data = data.update({var_dst: data.data_vars[var_src]})
        return data
