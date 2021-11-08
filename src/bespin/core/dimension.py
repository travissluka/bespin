# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Any, Sequence, Tuple, Dict
from collections.abc import Sized
import re

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

        # make sure correct combination of keywords provided
        if sum(x is not None for x in (edges, resolution)) != 1:
            raise ValueError(
                f'{self}: must set exactly ONE of "edges" or "resolution"')
        if (bounds is not None and resolution is None):
            raise ValueError(
                f'{self}: must define "resolution" if "bounds" is defined')

        # define based on given bin edges
        if edges is not None:
            if len(edges) <= 1:
                raise ValueError(
                    f"{self} at least 2 edges need to be given to define bins")
            self._bin_edges = np.array(edges)

        # define based on given resolution/bounds
        if resolution is not None:
            # use a default bounds, if none given
            if bounds is None and self.name in _default_bounds:
                bounds = _default_bounds[name]

            # error checking
            if resolution <= 0:
                raise ValueError(f'{self} resolution for must be > 0')
            if bounds is None:
                raise ValueError(
                    f'{self} bounds must be provided if resolution is set')
            if len(bounds) != 2:
                raise ValueError(
                    f'{self} bounds should be in the form [start, end].'
                    f' Incorrect value given: {bounds}')

            # calculate the bin edges based on resolution/bounds
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

    @classmethod
    def from_str(cls, string: str) -> 'Dimension':
        """Create a Dimension instance from a string representation.

        The string is expected to be in the format:

        `<name>:<arg1>:<arg2>`

        where one or two arguments are from the following:
        - resolution: `r=<float>`
        - bounds: `b=<List[float]>`
        - edges: `e=<List[float]>`

        Examples
        --------
        - latitude:r=1.0
        - longitude:r=1.0:b=0,180.0
        - depth:e=0,10,20,30,50,100,500

        See Also
        ---------
        Dimension.__init__() for description of the resolution, bounds, and
        edges arguments
        """
        args: Dict[str, Any] = {}

        try:
            # split into groups based on ":" character
            # ensure there are 2 or 3 groups
            splt = string.split(':')
            if len(splt) not in (2, 3):
                raise ValueError("Incorrect number of argument strings")

            # first group is the name of the dimension
            if not re.fullmatch(r'(\w|/)+', splt[0]):
                raise ValueError("name is not in correct format")
            args['name'] = splt[0]

            # the next group(s) are lists of numbers, or a single number.
            mapping = {
                'r': 'resolution',
                'b': 'bounds',
                'e': 'edges', }
            for s in splt[1:]:
                r = re.fullmatch(f'([{"".join(mapping.keys())}])=(.+)', s)
                if r is None:
                    raise ValueError(
                        f'Dimension argument string "{s}" is not valid.')
                k = mapping[r.group(1)]
                v = [float(x) for x in r.group(2).split(',')]
                args[k] = v[0] if len(v) == 1 else v

            # create the instance from the above parsed arguments
            dimension = cls(**args)

        except (ValueError, TypeError) as e:
            raise ValueError(
                f'Unable to parse dimension string "{string}".'
                ' See documentation for correct format.'
            ) from e

        return dimension

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
        if self.units:
            xd.coords[self.name].attrs['units'] = self.units
        return xd

    def centers_to_xarray(self) -> xr.Dataset:
        """Get the bin centers as coordinates in an xarray dataset."""
        xd = xr.Dataset(coords={self.name: self.bin_centers})
        if self.units:
            xd.coords[self.name].attrs['units'] = self.units
        return xd
