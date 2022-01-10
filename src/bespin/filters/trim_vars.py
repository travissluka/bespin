# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import List
import xarray as xr

from bespin.core.dimension import Dimension
from bespin.core.diagnostic import Diagnostic
from bespin.core.filter import FilterBase


class TrimVars(FilterBase):
    """Only keep the variables that are needed for binning.
    """

    def __init__(
            self,
            variable: str,
            diagnostics: List[Diagnostic],
            dimensions: List[Dimension]):
        super().__init__()
        self.keep = [f'{diag.name}/{variable}' for diag in diagnostics]
        self.keep += [d.name for d in dimensions]
        chan_var = 'sensor_channel'
        self.keep += [chan_var,]

    def filter(self, data: xr.Dataset,  **kwargs) -> xr.Dataset:
        keep = [s for s in self.keep if s in data.variables]
        data = data.get(keep)  # type: ignore

        return data
