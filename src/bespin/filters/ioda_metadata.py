# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Iterable

import xarray as xr

from bespin.core.filter import FilterBase


# TODO use only MetaData once we are working with proper IODAv2 files
_metaGroups = ('MetaData', 'varMetaData', 'recMetaData')


class IodaMetadata(FilterBase):
    """A filter to rename binning dimension variables.

    All IODA variables are dumped into the "data_vars" section
    of the xarray dataset. The binning process requires that these
    variables are renamed to remove the IODA groups.

    for example:

    the data variable `MetaData/latitude` would be renamed to a
    coordinate variable `latitude` if it is going to be used in the binning.
    """

    def __init__(self, bin_names: Iterable[str]):
        """Initialize Ioda2Coords filter.

        Args:
          bin_names: the names of the dimensions to be renamed/converted.
        """
        self.bin_names = bin_names
        super().__init__()

    def filter(self, data: xr.Dataset) -> xr.Dataset:
        # get list of binning dimension variables whose name does not yet
        # exist in the dataset
        dims_to_filter = set(self.bin_names).difference(data.variables.keys())

        for d in dims_to_filter:
            # get list of the metadata groups this variable exists under
            meta_group = [g for g in _metaGroups
                          if f'{g}/{d}' in data.variables]
            if len(meta_group):
                # use the first metadata group found in the list.
                # e.g. If for some reason 'latitude' exists under 'MetaData'
                # and 'varMetaData' (which definitely shouldn't happen!)
                # chose MetaData

                # rename
                data = data.rename({f'{meta_group[0]}/{d}': d})
        return data
