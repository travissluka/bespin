# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Iterable, Mapping, MutableMapping, Optional

import xarray as xr
from itertools import product
import numpy as np

from bespin.core.statistic import Statistic
from bespin.core.diagnostic import Diagnostic
from bespin.core.dimension import Dimension
import bespin.io.binned_netcdf as binned_io


class BinnedStatistics:
    """The main container for binned statistics.

    For current usage, see:
      get()
      read()

    Attributes:
        bins: A list of bespin.Dimension specifying how the binning is done.
        diagnostics: A list of bespin.Diagnostic specifying what is binned.
        name: A str name of the binning type.
        variables: A dict containing meta information for each variable binned.
    """

    _io = binned_io  # The file IO class to use.

    def __init__(
            self,
            name: str,
            bins: Iterable[Dimension],
            diagnostics: Iterable[Diagnostic],
            ):
        """Initialize BinnedStatistics.

        You probably don't want to use this. Look at BinnedStats.read().
        """
        self.name = name
        self.bins = bins
        self.diagnostics = {d.name: d for d in diagnostics}
        self.variables: MutableMapping[str, Mapping[str, str]] = {}
        self._data = xr.Dataset(
            coords=xr.merge([b.centers_to_xarray() for b in self.bins]))

    def __repr__(self) -> str:
        bin_names = [b.name for b in self.bins]
        return f'BinnedStatistics("{self.name}"", bins={bin_names})'

    @classmethod
    def read(cls, filename: str) -> 'BinnedStatistics':
        """Read an existing binned statistics from a file."""
        input_args, input_data = cls._io.read(filename)

        # TODO use the full set of global_attributes
        bs = BinnedStatistics(
            name=input_args['global_attributes']['binning'],
            bins=input_args['bins'],
            diagnostics=input_args['diagnostics'],
            )

        # TODO actually read in variable metadata
        bs.variables = {
            v: {} for v
            in input_args['global_attributes']['variables'].split(' ')}

        for k, v in input_data.items():
            bs._data[k] = v

        return bs

    def equivalent(self, other: 'BinnedStatistics') -> bool:
        """Return True if the shape of the diagnostics is the same.

        This checks if the binning dimensions, filter, variables, diagnostic
        types, and statistic types are all identical. The actual binned
        statistic values are allowed to differ.
        """
        eq_bins = all([b[0] == b[1] for b in zip(self.bins, other.bins)])
        eq_diags = all([b[0] == b[1] for b in zip(
            self.diagnostics, other.diagnostics)])
        return (
            self.name == other.name and
            eq_bins and
            eq_diags and
            self.variables == other.variables and
            self._data.equals(other._data))

    def get(
            self, *,
            variable=None,
            diagnostic=None,
            statistic=None,
            ignore_missing=True
            ) -> xr.Dataset:
        """Retrieve the binned statistics values.

        If no value is given for `variable`, `diagnostic`, or `statistic`,
            a default of all available names will be used for that argument.

        Args:
            variable: (optional) one or more names of variables to return.

            diagnostic: (optional) one or more names of diagnostics to return.
                If a diagnostic does not exist for a certain variable it is
                ignored.

            statistic: (optional) one or more names of statistics to return.
                If a statistic does not exist for a certain variable/diagnostic
                combination it is ignored

            ignore_missing: (Default: True) if true, a missing diagnostic/
                statistic combination for any given variable will be ignored
                and not throw an error

        Returns:
            An xarray dataset with the requested information. The returned
            variables will be in the form `<variable>.<diagnostic>.<statistic>`
        """
        # apply defaults to incoming arguments
        def _arg_default(arg, default):
            """Convert `None` or `str` into `list(str)` with given default."""
            if arg is None:
                arg = default
            elif isinstance(arg, str):
                arg = (arg,)
            return list(arg)
        variable = _arg_default(variable, self.variables.keys())
        diagnostic = ([None, ] +
                      _arg_default(diagnostic, self.diagnostics.keys()))
        statistic = _arg_default(statistic, Statistic.types())

        # create the return dataset, with proper coordinates.
        return_data = xr.Dataset(coords=self._data.coords)
        dims = list(return_data.dims)
        if len(dims) == 0:
            dims = ['None']

        # populate the dataset
        for var, diag, stat in product(variable, diagnostic, statistic):
            if diag is not None and stat == 'count':
                # TODO ugly hardcoding. figure out a cleaner way.
                # "count" gets repeated for every diag without this.
                continue

            # try to get the statistic, allow for failure
            val: Optional[np.ndarray]
            try:
                val = Statistic(stat, var, diag, self._data).value()
            except RuntimeError:
                if ignore_missing:
                    val = None
                else:
                    raise ValueError(
                        f'diag,stat= "{diag}","{stat}" does not exist for '
                        f'variable= "{var}". Try again with '
                        '"ignore_missing"=True')

            # add value to the dataset if it was calculated successfully
            if val is not None:
                db_name = '.'.join([v for v in (var, diag, stat) if v])
                return_data.update({db_name: (dims, val)})
        return return_data

    def write(self, filename: str) -> None:
        """Write the binned statistics to a file."""
        # TODO make sure binning has been run first!

        # TODO fill in the global attributes with real data.
        global_attr = {
            'binning': self.name,
            'obs_source': 'TODO: fill this in',
            'experiment': 'TODO: fill this in',
            'window_start': 'TODO: fill this in',
            'window_end': 'TODO: fill this in',
            'variables': ' '.join(self.variables.keys())
            }

        # write out to a file.
        self._io.write(
            filename,
            global_attributes=global_attr,
            bins=self.bins,
            diagnostics=self.diagnostics.values(),
            data=self._data,
            )

    def _bin_variable(self, variable: str, unbinned_data: xr.Dataset) -> None:
        # make sure variable does not already exist
        if variable in self.variables:
            raise ValueError(
                f'Cannot bin variable "{variable}", already exists.')

        # make sure variable metadata is populated
        # TODO
        self.variables[variable] = {}

        # Do other checks to make sure input xarray contains all the
        # correct data
        # TODO

        # we can currently handle at most 2D input data, with no additional
        # binning on the variables using the second dimension.
        # TODO, clean this up to better handle the 2nd input dimension
        for diag in self.diagnostics:
            var_name = f'{variable}.{diag}'

            if var_name not in unbinned_data:
                raise ValueError(f'{var_name} missing from unbinned data.')

            if len(unbinned_data[var_name].dims) > 2:
                raise ValueError(
                    f'unbinned input {var_name} has an unsupported > 2 '
                    f'dimensions.')

            if len(unbinned_data[var_name].dims) == 2:
                # 2D input data. Make sure the coordinate for the 2nd dimension
                # exists in the binned dataset.
                src_dim = unbinned_data[var_name].dims[-1]
                multichannel_dim = [
                    c for c in unbinned_data.coords.values()
                    if c.dims == (src_dim,)]
                if len(multichannel_dim) == 1:
                    coord = multichannel_dim[0]
                    if coord.name not in self._data.coords:
                        self._data.coords.update(
                            {coord.name: ((coord.name,), coord.values)})
                        self._data.set_coords(coord.name)

        # create counts
        # TODO save and reuse bin locations
        Statistic(
            'count',
            variable=variable,
            diagnostic=None,
            binned_data=self._data).calc(self.bins, unbinned_data)

        # run the bining on other stats and save results
        for diag in self.diagnostics:
            for stat in self.diagnostics[diag].statistics:
                Statistic(
                    stat,
                    variable=variable,
                    diagnostic=diag,
                    binned_data=self._data).calc(self.bins, unbinned_data)
