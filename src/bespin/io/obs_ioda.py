# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Interface to IODA code"""

from typing import Optional, List
import os

try:
    import ioda_obs_space as ioda  # type: ignore
    ioda_found = True
except ModuleNotFoundError as e:
    ioda_found = False
import xarray as xr


_dim_mapping = {
    'MetaData': 'nlocs',
    'RecMetaData': 'nrecs',
    'VarMetaData': 'nvars'
}


def read(filename: str, variables: Optional[List[str]] = None) -> xr.Dataset:
    """Read the given variables from the given ioda file.

    Args
    -----

      filename: the name of the file to read in

      variables: (optional) list of variables to read, otherwise all variables
        are read in.
    """
    if not ioda_found:
        raise ModuleNotFoundError('module "ioda" was not found')

    # make sure file exists, open it
    if not os.path.exists(filename):
        raise FileExistsError(f"Unable to load file {filename}")
    ios = ioda.ObsSpace(filename)

    # only get the exact variables required instead of loading everything?
    if variables is None:
        variables = ios.variables

    # get the data
    # TODO also get the attributes.
    data = {}
    is2D = 'nchans' in ios.dimensions
    for v in variables:
        # skip if not in a group
        if '/' not in v:
            continue
        groupname, _ = v.split('/')

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

        # read data
        data[v] = (var_dims, ios.Variable(v).read_data())

    # place data in an xarray, and return.
    xr_data = xr.Dataset(data_vars=data)
    return xr_data
