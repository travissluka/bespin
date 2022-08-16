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

    def __init__(self, src1: str, src2: str, dst: str, *args):
        self.src1 = src1
        self.src2 = src2
        self.dst = dst
        self.quiet = False

        # optional arguments
        for arg in args:
            if arg == 'quiet':
                self.quiet=True
            else:
                raise ValueError(f'invalid optional argument "{arg}" passed to "sub"')

        super().__init__()

    def filter(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        src1 = self.src1.format(**kwargs)
        src2 = self.src2.format(**kwargs)
        dst = self.dst.format(**kwargs)

        # check to make sure the required variables are present
        for src in (src1, src2):
            if src not in data.data_vars:
                if self.quiet: # not an error, silently return
                    return data
                else:
                    raise ValueError(f'Cannot find input variable "{src}" in the dataset')

        data = data.update({dst: data.data_vars[src1] - data.data_vars[src2]})
        return data
