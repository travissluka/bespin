# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Binned stats file IO via netcdf interface."""

from typing import Iterable, List, Mapping, Any, MutableMapping, Tuple
import netCDF4 as nc  # type: ignore
import xarray as xr
import os.path

from bespin.core.dimension import Dimension
from bespin.core.diagnostic import Diagnostic


def read(filename: str) -> Tuple[Mapping[str, Any], Mapping[str, xr.Dataset]]:
    """Read an existing binned statistics file."""
    results: MutableMapping[str, Any] = {}
    filename = _add_suffix(filename)

    # Diagnostics
    data = xr.load_dataset(filename, group="Diagnostics")

    # open file for other groups to read
    root = nc.Dataset(filename, 'r')

    # global attributes
    results['global_attributes'] = {
        k: root.getncattr(k) for k in root.ncattrs()}

    # Filters
    # TODO

    # Bins
    bins: List[Dimension] = []
    for k, v in root.groups['Bins'].variables.items():
        bins.append(Dimension(k, edges=v[:]))
        # TODO get attributes
    results['bins'] = bins

    # Diagnostics attributes
    diagnostics: List[Diagnostic] = []
    group = root.groups['Diagnostics']
    diags = group.getncattr('diagnostics')
    if type(diags) is str:
        # make sure diags is in a list, even if it is just one
        diags = [diags, ]
    for diag in diags:
        stats = group.getncattr(f'{diag}.stats')
        if type(stats) is str:
            stats = [stats, ]
        diagnostics.append(Diagnostic(diag, stats))
    results['diagnostics'] = diagnostics

    root.close()
    return results, data


def write(
        filename: str,
        global_attributes: Mapping[str, str],
        bins: Iterable[Dimension],
        diagnostics: Iterable[Diagnostic],
        data: xr.Dataset,
        overwrite: bool,
        ) -> None:
    """Write binned statistic to a file."""
    filename = _add_suffix(filename)
    if not overwrite and os.path.exists(filename):
        raise FileExistsError()

    # Diagnostics
    data.to_netcdf(filename, 'w', group='Diagnostics')

    # Bins
    bin_xr = xr.Dataset(
        coords=xr.merge([b.edges_to_xarray() for b in bins]))
    bin_xr.to_netcdf(filename, 'a', group="Bins")

    # open file for writing other groups
    root = nc.Dataset(filename, 'a', format="NETCDF4")

    # global attributes
    for attr in global_attributes.items():
        root.setncattr(*attr)

    # Variable attributes
    # TODO

    # Filters
    group = root.createGroup("Filters")
    # TODO

    # Diagnostics attributes
    group = root.createGroup("Diagnostics")
    group.diagnostics = [d.name for d in diagnostics]
    for diag in diagnostics:
        group.setncattr(f'{diag.name}.stats', diag.statistics)

    root.close()


def _add_suffix(filename: str) -> str:
    # append nc suffix if not already present
    if not filename.split('.')[-1] in ('nc', 'nc4'):
        filename += ".nc4"
    return filename
