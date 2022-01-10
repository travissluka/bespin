# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Any, Iterable, Mapping, MutableMapping, Optional, List

import xarray as xr
import pandas
from itertools import product
import numpy as np
import copy

from bespin.core.statistic import Statistic
from bespin.core.diagnostic import Diagnostic
from bespin.core.dimension import Dimension
from bespin.core.filter import Filter, FilterBase
import bespin.io.binned_netcdf as binned_io

# modules that aren't directly accessed, but need to be imported so that
# the subclasses can be registered with the factories
import bespin.filters
import bespin.statistics


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
            **kwargs
            ):
        """Initialize BinnedStatistics.

        You probably don't want to use this. Look at BinnedStats.read() or
        BinnedStats.bin()
        """
        self.name = name
        self.bins = tuple(bins)
        self.diagnostics = {d.name: d for d in diagnostics}
        self.variables: MutableMapping[str, Mapping[str, str]] = {}
        self._data = xr.Dataset(
            coords=xr.merge([b.centers_to_xarray() for b in self.bins]))

        # TODO store following meta data in _data xarray as attributes
        self.window_start: pandas.Timestamp = kwargs.get('window_start')
        self.window_end: pandas.Timestamp = kwargs.get('window_end')
        self.sliced_dims: MutableMapping[str, Any] = {}


    def __repr__(self) -> str:
        bin_names = [b.name for b in self.bins]
        return f'BinnedStatistics("{self.name}"", bins={bin_names})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinnedStatistics):
            return False
        return self.equals(other)

    @classmethod
    def bin(cls,
            name: str,
            bins: Iterable[Dimension],
            diagnostics: Iterable[Diagnostic],
            variables: Iterable[str],
            filters: Iterable[FilterBase],
            unbinned_data: xr.Dataset) -> 'BinnedStatistics':
        """Bin the `unbinned_data` according to the options provided

        Args
        -----
            name: the user-readable name of the binning (not really used for
              anything at the moment)

            bins: zero or more dimensions used for the binning

            diagnostics: one or more diagnostics that will be binned.

            variables: one or more observation variables that will binned.

            filters: zero or more filters that will be applied to
              `unbinned_data` before being binned. Several automatic filters
              are implicitly added to this list.

            unbinned_data: the input data to bin.

        Example
        --------
           >>> BinnedStatistics(
                name = 'latlon',
                bins = (
                    Dimension("latitude", resolution=10),
                    Dimension("longitude", resolution=10)),
                diagnostics = (
                    Diagnostic('ObsValue', statistics=('count', 'sum', 'sum2', 'min', 'max')),
                    Diagnostic('omb')),
                variables = ['sea_surface_temperature',],
                unbinned_data = data)
        """
        # TODO do some sanity checks?
        # TODO the binning is complex enough that we could move this out into
        #  a separate private module.

        # create the empty BinnedStats instance
        bs = cls(
            name=name,
            bins=bins,
            diagnostics=diagnostics)

        # do some filters on the whole dataset
        filters = list(filters)

        filters_all = [
            Filter('ioda_metadata', [b.name for b in bins]),
            Filter('lon_wrap'),
            ]
        for f in filters:
            if not f.per_variable:
                filters_all.append(f)
        for f in filters_all:
            unbinned_data = f.filter(unbinned_data)

        # for each variables
        for v in variables:
            unbinned_data_var = unbinned_data.copy()

            # do some specific per-variable filtering
            filters_var = [f for f in filters if f.per_variable]
            filters_var.append(Filter('trim_vars', v, diagnostics, bins))
            #filters_var.append(Filter('remove_nan'))
            for f in filters_var:
                args = {'variable': v}
                unbinned_data_var = f.filter(unbinned_data_var, **args)
            # do the binning
            bs._bin_variable(v, unbinned_data_var)

        # pull some attributes from the unbinned_data
        for attr in ('window_start', 'window_end'):
            bs.__dict__[attr] = unbinned_data.attrs[attr]

        return bs

    @classmethod
    def read(cls, filename: str) -> 'BinnedStatistics':
        """Read an existing binned statistics from a file."""
        input_args, input_data = cls._io.read(filename)

        # TODO use the full set of global_attributes
        bs = cls(
            name=input_args['global_attributes']['binning'],
            bins=tuple(input_args['bins']),
            diagnostics=input_args['diagnostics'],
            )

        # parse date ranges
        for a in ('window_start','window_end'):
            bs.__dict__[a] = pandas.to_datetime(
                input_args['global_attributes'][a])

        # TODO actually read in variable metadata
        bs.variables = {
            v: {} for v
            in input_args['global_attributes']['variables'].split(' ')}

        for k, v in input_data.items():
            bs._data[k] = v

        return bs

    def equals(self, other: 'BinnedStatistics', tolerance=1e-14) -> bool:
        """Test equality between two binned stats.

        A tolerance provided in order to account for small rounding errors.
        """
        if not self.equivalent(other):
            return False

        diff = (other._data - self._data)/self._data
        diff = diff.reduce(np.nanmean).reduce(np.abs)
        return bool(np.all([v <= tolerance for v in diff.data_vars.values()]))

    def equivalent(self, other: 'BinnedStatistics') -> bool:
        """Return True if the shape of the diagnostics is the same.

        This checks if the binning dimensions, filter, variables, diagnostic
        types, and statistic types are all identical. The actual binned
        statistic values are allowed to differ.
        """
        return (
            self.name == other.name and
            self.bins == other.bins and
            self.diagnostics == other.diagnostics and
            self.variables == other.variables and
            bool(np.all([self._data.coords[c].equals(other._data.coords[c])
                         for c in self._data.coords]))
            )

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
        dims = list(self._data.dims)
        if len(dims) == 0:
            dims = ['None']

        # populate the dataset
        for var, diag, stat in product(variable, diagnostic, statistic):
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

        # set global attributes
        # TODO should these be global attributes on the internal xarray?
        for a in ('window_start', 'window_end'):
            return_data.attrs[a] = self.__dict__[a].isoformat()
        if len(self.sliced_dims):
            return_data.attrs['sliced_dims'] = ' '.join(self.sliced_dims)
            for k, v in self.sliced_dims.items():
                # TODO prefix name so that we don't risk overwriting something important?
                return_data.attrs[k] = v

        return return_data

    def write(self, filename: str, overwrite: bool = False) -> None:
        """Write the binned statistics to a file.

        overwrite: If true, overwrite an existing file, otherwise and exception
          will be thrown. (default: False)
        """
        # TODO make sure binning has been run first!

        # TODO fill in the global attributes with real data.
        global_attr = {
            'binning': self.name,
            'obs_source': 'TODO: fill this in',
            'experiment': 'TODO: fill this in',
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'variables': ' '.join(self.variables.keys())
            }

        # write out to a file.
        self._io.write(
            filename,
            global_attributes=global_attr,
            bins=self.bins,
            diagnostics=self.diagnostics.values(),
            data=self._data,
            overwrite=overwrite,
            )

    def merge(
            self,
            other: 'BinnedStatistics',
            ) -> 'BinnedStatistics':
        """Merge two BinnedStatistics datasets.

        A new BinnedStatistics will be returned. 'self' and 'other' will not be
        modified. The operation will be nearly identical to if all the input
        observations had been binned at the same time.

        See also: merge()
        """
        # Create a new class to hold the results, copy over the xarray coords
        merged = BinnedStatistics(
            self.name, self.bins, self.diagnostics.values())
        merged._data.coords.update(self._data.coords)

        for var in self.variables:
            merged.variables[var] = {}

            # calculate the merged stat for each diagnostic
            for diag in self.diagnostics.values():
                for stat in diag.statistics:
                    stat_class = [
                        Statistic(
                            stat,
                            variable=var,
                            diagnostic=diag.name,
                            binned_data=d._data)
                        for d in [merged, self, other]
                    ]
                    stat_class[0].merge(*stat_class[1:3])

        # merge special attributes such as obs window start/end
        merged.window_start = min(self.window_start, other.window_start)
        merged.window_end = max(self.window_end, other.window_end)
        # TODO handle other attributes

        return merged

    def select_dim(self, **dims) -> 'BinnedStatistics':
        """remove a dimension of the binning by taking a specific slice"""
        # create a record in an attribute what slice what chosen
        for k, v in dims.items():
            # TODO get the actual values, NOT the indexof the array
            self.sliced_dims[k] = v

        # TODO is a deepcopy really necessary for the xarray? (probably not)
        inst = copy.deepcopy(self)
        for d in dims:
            inst.bins = tuple(filter(lambda x: (x.name != d), inst.bins))
            inst._data = inst._data.sel({d: dims[d]}).drop_vars(d)
        return inst

    def collapse_dim(self, *dims: str) -> 'BinnedStatistics':
        def isel(self, **dims) -> 'BinnedStatistics':
            inst = copy.deepcopy(self)
            for d in dims:
                inst.bins = tuple(filter(lambda x: (x.name != d), inst.bins))
                inst._data = inst._data.isel({d: dims[d]}).drop_vars(d)
            return inst

        # NOTE this is horribly innefficient
        inst = copy.copy(self)
        for d in dims:
            collapsed = [ isel(inst, **{d:i})
                          for i in range(inst._data.dims[d])]
            inst2 = collapsed[0]
            for c in collapsed[1:]:
                inst2 = inst2.merge(c)
            inst = inst2
        return inst

    def concat(
            self,
            other: 'BinnedStatistics',
            dim: str,
            in_place=False,
            ) -> 'BinnedStatistics':
        raise NotImplementedError()

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
            var_name = f'{diag}/{variable}'

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

        # TODO save and reuse bin locations

        # run the bining on other stats and save results
        for diag in self.diagnostics:
            for stat in self.diagnostics[diag].statistics:
                Statistic(
                    stat,
                    variable=variable,
                    diagnostic=diag,
                    binned_data=self._data).calc(self.bins, unbinned_data)


def merge(binned_stats: Iterable[BinnedStatistics]) -> BinnedStatistics:
    """Merge multiple BinnedStatistics into one.

    binned_stats: a list of two or more BinnedStats.

    See also: BinnedStatistics.merge()
    """
    # TODO do in-place merges to speed up?
    itr = iter(binned_stats)
    result = copy.deepcopy(next(itr))
    for i in itr:
        result = result.merge(i)
    return result


def concat(
        binned_stats: Iterable[BinnedStatistics],
        dim: str
        ) -> BinnedStatistics:
    itr = iter(binned_stats)
    result = copy.deepcopy(next(itr))
    for i in itr:
        result.concat(i, dim=dim, in_place=True)
    return result
