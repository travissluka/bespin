# (C) Copyright 2022-2022 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import xarray as xr

from bespin.core.filter import FilterBase


class Range(FilterBase):
    """make sure a variable is in a certain value range.
    """

    per_variable = True

    def __init__(self, variable: str, val_range: str):
        val_range_split = val_range.split(',')
        assert len(val_range_split) == 2
        self.range = [float(v) for v in val_range_split]
        self.variable = variable
        super().__init__()

    def filter(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        variable = self.variable.format(**kwargs)
        mask = data[variable] >= self.range[0]
        mask = np.logical_and(mask, data[variable] <= self.range[1])
        data = data.where(mask, drop=True)
        return data
