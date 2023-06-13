# (C) Copyright 2021-2022 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Interface to IODA netcdf obs, bypassing the annoying IODA python interface"""

from typing import Optional, List

import netCDF4 as nc
import pandas
import os
import xarray as xr

_dim_mapping = {
    'MetaData': 'nlocs',
    'RecMetaData': 'nrecs',
    'VarMetaData': 'nchans'
}

def read(filename: str, variables: Optional[List[str]] = None) -> xr.Dataset:
    """Read the given variables from the given ioda file.

    Args
    -----

      filename: the name of the file to read in

      variables: (optional) list of variables to read, otherwise all variables
        are read in.
    """

    # make sure file exists, open it
    if not os.path.exists(filename):
        raise FileExistsError(f"Unable to load file {filename}")
    ncd = nc.Dataset(filename)

    # TODO only get the exact variables required instead of loading everything?

    # get the data
    data = {}
    is2D = 'nchans' in ncd.dimensions
    for groupname in list(ncd.groups):
        group = ncd.groups[groupname]

        # determine what the dimension names should be
        if groupname in _dim_mapping:
            # if one of the "MetaData" variables, we know what their
            # dimension name should be
            var_dims = [_dim_mapping[groupname], ]
        else:
            # otherwise, assume a 1D or 2D data variable.
            var_dims = ['nlocs', ]
            if is2D:
                var_dims += ['nchans', ]

        # read the data
        for varname in list(group.variables):
            full_name = f'{groupname}/{varname}'
            d = group.variables[varname][:]
            if varname == 'dateTime':
                units = group.variables[varname].units
                # this is a hardcoded hack, find a better way to do this...
                assert(units.startswith('seconds since'))
                origin=units.split()[2][:-1]
                d = pandas.to_datetime(d, unit='s', origin=origin)
            data[full_name] = (var_dims, d)

    # place data in an xarray, and return
    xr_data = xr.Dataset(data_vars=data)
    return xr_data
