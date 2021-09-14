# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Sequence, Tuple
from collections.abc import Sized

import numpy as np
import xarray as xr

# default bounds for some common dimmension names
_default_bounds = {
    'latitude': (-90.0, 90.0),
    'longitude': (0.0, 360.0)
}

# default units for some common dimmension names
_default_units = {
    'latitude': 'degrees',
    'longitude': 'degrees'
}


class Dimension(Sized):
    """
    Contains information about a single dimension used for binning.

    Can be initialized with either a fixed resolution (by setting
    `resolution` and `range`), or a specific list of edge values (by
    setting ``edges``).

    Attributes:
        name: str name of the dimension.
        units: str description of the units.
    """

    def __init__(
            self,
            name: str,
            edges: Sequence[float] = None,
            resolution: float = None,
            bounds: Tuple[float, float] = None):
        """Create a dimension.

        Args:
            name: The name of the dimension. Certain common dimmension names
                such as latitude and longitude will have defaults available
                for 'bounds' and the metadata.

            edges: a list of the edge values for the dimension binning.
                Mutually exclusive with `resolution` and `bounds`.

            resolution: the resolution at which to create the binning. Must
                be used with `bounds` set as well.

            bounds: The starting and ending binning edges to use if
                `resolution` is set. Default values are available for some
                dimension names.

        Raises:
            ValueError: if an incorrect combination of edges, resolution, and
                bounds are given, or for bad values for those arguments.
        """
        self._bin_edges = np.array([np.nan, np.nan])
        self.name = name
        self.units = None

        # make sure correct combination of kewords provided
        if ((resolution is not None and edges is not None)
                or (edges is not None and bounds is not None)):
            raise ValueError(
                f'{self}: only one of "edges" or "resolution/bounds"'
                ' should be set')

        # define base on given bin edges
        if edges is not None:
            if len(edges) <= 1:
                raise ValueError(
                    f"{self} at least 2 edges need to be given to define bins")
            self._bin_edges = np.array(edges)

        # define based on given resolution/bounds
        if resolution is not None:
            if not bounds and self.name in _default_bounds:
                bounds = _default_bounds[name]

            # error checking
            if resolution <= 0:
                raise ValueError(f'{self} resolution for must be > 0')
            if not bounds or len(bounds) != 2:
                raise ValueError(
                    f'{self} bounds must be provided if resolution is set')

            self._bin_edges = np.arange(
                bounds[0], bounds[1] + resolution/2.0, resolution)

        # attributes
        if name in _default_units:
            self.units = _default_units[name]

    def __eq__(self, other) -> bool:
        return (
            self.name == other.name and
            self.units == other.units and
            np.array_equal(self._bin_edges, other._bin_edges))

    def __len__(self) -> int:
        """Get the number of bins (one less than the number of edges)."""
        return len(self._bin_edges)-1

    def __repr__(self) -> str:
        return (f'Dimension("{self.name}", bounds={self.bounds},'
                f' bins={len(self)})')

    @property
    def bin_centers(self) -> np.ndarray:
        """Get the values of the bin centers."""
        return (self._bin_edges[1:] + self._bin_edges[:-1]) / 2.0

    @property
    def bin_edges(self) -> np.ndarray:
        """Get the values of the bin edges."""
        return self._bin_edges

    @property
    def bounds(self) -> Tuple[float, float]:
        """Return the outer edges of the binning."""
        return (self._bin_edges[0], self._bin_edges[-1])

    def edges_to_xarray(self) -> xr.Dataset:
        """Get the bin edges as coordinates in an xarray dataset."""
        xd = xr.Dataset(
            coords={self.name: (f'{self.name}_edges', self.bin_edges)})
        xd.coords[self.name].attrs['units'] = self.units
        return xd

    def centers_to_xarray(self) -> xr.Dataset:
        """Get the bin centers as coordinates in an xarray dataset."""
        xd = xr.Dataset(coords={self.name: self.bin_centers})
        xd.coords[self.name].attrs['units'] = self.units
        return xd
