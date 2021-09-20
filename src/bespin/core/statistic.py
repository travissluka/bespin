# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Statistics calculation/merging/value_getting base classes.

TODO: The binning routines should be pulled out, and reworked better for
multichannel input data.
"""

from typing import Iterable, MutableMapping, List, Type, FrozenSet
from typing import Optional, Union

import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_dd  # type: ignore
from xarray.core.coordinates import DatasetCoordinates

from bespin.core.dimension import Dimension


class StatisticBase():
    """Base class for all statistics classes.

    The methods here are not to be called directly, instead see `Statistic`.
    A sub class should provide its own copy of calc(), merge(), and value()
    where applicable.
    """

    # A list of statistics that need to be calculated before this statistic.
    depends: List[str] = []

    def __new__(cls, *args, **kwargs) -> 'StatisticBase':
        if cls == StatisticBase:
            raise RuntimeError('Cannot directly instantiate StaticBase')
        return super(StatisticBase, cls).__new__(cls)

    def __init__(
            self,
            variable: str,
            diagnostic: Optional[str],
            binned_data: xr.Dataset):
        self.type_ = str.lower(self.__class__.__name__)
        self.variable = variable
        self.diagnostic = diagnostic
        self.binned_data = binned_data

        # these are only set if binning is to be performed
        self._unbinned_data: xr.Dataset
        self._bins: Iterable[Dimension]

    def calc(self, data: np.ndarray) -> np.ndarray:
        """Calculate a binned statistic."""
        raise RuntimeError(
            f'Statistic class for "{self.type_}" does not have'
            f' a calc() method')

    def merge(
            self,
            stat1: 'StatisticBase',
            stat2: 'StatisticBase') -> np.ndarray:
        """Calculate merged binned statistics."""
        raise RuntimeError(
            f'Statistic class for "{self.type_}" does not have'
            f' a merge() method')

    def value(self) -> np.ndarray:
        """Get the binned stat value.

        By default, the value is calculated by calc() and returned from the
        existing dataset. If overridden, a derived value will be calculated and
        returned.
        """
        if self.var_name not in self.binned_data.variables:
            raise RuntimeError(
                f'Statistic for "{self.var_name}" does not exist')
        return self.binned_data.variables[self.var_name].to_numpy()

    @property
    def var_name(self) -> str:
        """Get the variable name for the binning.

        Form is of <variable>.<diagnostic>.<type_>  or just <variable>.<type_>
        if diagnostic is None.
        """
        return '.'.join([i for i in (
            self.variable,
            self.diagnostic,
            self.type_) if i])

    def _setup_binner(
            self,
            bins: Iterable[Dimension],
            unbinned_data: xr.Dataset) -> None:
        """Initialize things before _binner is called by the subclass."""
        self._bins = bins
        self._unbinned_data = unbinned_data

    def _binner(self, method: str, data: np.ndarray) -> np.ndarray:
        """Perform binning on the input data, to be called by the subclass."""
        if self._unbinned_data is None:
            raise RuntimeError(
                '_binner() cannot be called before _setup_binner()')

        if len(data.shape) > 2:
            raise ValueError("input to _binner() has too many dimensions.")

        # make sure all required binning dimensions are present in input data
        dims_unbinned = set(list(self._unbinned_data.coords.keys()))
        dims_binned = set([b.name for b in self._bins])
        if not dims_binned.issubset(dims_unbinned):
            diff = dims_binned.difference(dims_unbinned)
            raise RuntimeError(f'missing dimensions {diff}')

        # prepare the binning information
        dim_vals: Union[List[xr.Variable], np.ndarray] = [
            self._unbinned_data.variables[c].as_numpy()
            for c in [b.name for b in self._bins]]
        bins = [b.bin_edges for b in self._bins]
        if not len(bins):
            # special dummy data if no bins (i.e. global binning)
            dim_vals = np.zeros(data.shape[0])
            bins = [np.array([-1, 1])]

        # handle multidimensional input data specially
        # TODO: this needs to be done more efficiently
        if len(data.shape) == 1:
            # 1D input data
            results = binned_statistic_dd(
                sample=dim_vals,
                values=data,
                statistic=method,
                bins=bins)
            return results.statistic
        else:
            # 2D input data
            # note that we have to bin each 1D array separately, then
            # concatenate into the final result (this is innefficient, redo)
            results = None
            result_size = [len(b) for b in self._bins]
            if not len(result_size):
                result_size = []
            result_size += [0]
            result_data = np.empty(result_size)
            for i in range(data.shape[1]):
                results = binned_statistic_dd(
                    sample=dim_vals,
                    values=data[:, i],
                    statistic=method,
                    bins=bins,
                    binned_statistic_result=results)

                # give the array and empty dimension at the end, unless
                # we binned with no bins
                result_1d = results.statistic
                if len(result_data.shape) > 1:
                    result_1d = np.expand_dims(result_1d, -1)

                # concatenate to form final result array
                result_data = np.concatenate(
                    (result_data, result_1d), axis=-1)
            return result_data

    def _get(self, type_: str) -> np.ndarray:
        """Get a different statistic from the same binned DataSet."""
        try:
            val = Statistic(
                type_,
                self.variable,
                self.diagnostic,
                self.binned_data).value()
        except RuntimeError:
            raise RuntimeError(
                f'Statistic "{self.type_}" depends on "{type_}",'
                f' but it has not been calculated yet')
        return val


class Statistic():
    """A factory/wrapper for the various statistics classes."""

    __classes: MutableMapping[str, Type[StatisticBase]] = {}

    def __init__(
            self,
            type_: str,
            variable: str,
            diagnostic: Optional[str],
            binned_data: xr.Dataset):
        self.type_ = type_
        self.variable = variable
        self.diagnostic = diagnostic
        self.binned_data = binned_data

        if type_ not in self.__classes:
            raise ValueError(f'Statistic class "{type_}" does not exist')
        self._statistic = self.__classes[type_](
            variable=self.variable,
            diagnostic=self.diagnostic,
            binned_data=self.binned_data,
            )

    def __repr__(self) -> str:
        return f'Statistic("{self._statistic.var_name}")'

    @classmethod
    def register(cls, class_: Type[StatisticBase]) -> None:
        name = str.lower(class_.__name__)
        if name in cls.__classes:
            raise ValueError(f'cannot register "{name}", already exists')
        cls.__classes[name] = class_

    @classmethod
    def types(cls) -> FrozenSet[str]:
        """Get the list of valid registered statistics."""
        return frozenset(cls.__classes.keys())

    @property
    def dependencies(self) -> List[str]:
        """Get list of other statistics that must be calcuated before this."""
        return self._statistic.depends

    def calc(
            self,
            bins: Iterable[Dimension],
            unbinned_data: xr.Dataset,
            ) -> None:
        """Perform binnining of the data using the given statistics.

        The data is binned according to `bins`, using the input data from
        `unbinned_data`. The results are stored in `self.binned_data` under
        the variable name <diagnostic>.<statistic>
        """
        # create dummy data if needed (e.g. for count() statistic)
        if self.diagnostic:
            src_var = f'{self.variable}.{self.diagnostic}'
            if src_var not in unbinned_data.variables.keys():
                raise RuntimeError(
                    f'Cannot calc stat "{self.type_}" because'
                    f' "{src_var}" does not exist in the input data'
                )
            vals = unbinned_data.variables[src_var].to_numpy()
        else:
            vals = np.zeros(list(unbinned_data.dims.values()))

        # calculate the binned statistic
        self._statistic._setup_binner(bins, unbinned_data)
        result = self._statistic.calc(vals)

        # add to the dataset
        dims: Union[DatasetCoordinates, List[str]] = self.binned_data.coords
        if len(dims) == 0:
            dims = ['None']
        self.binned_data.update({self._statistic.var_name: (dims, result)})

    def merge(self, stat1: 'Statistic', stat2: 'Statistic') -> None:
        """Merge two statistics of the same type."""
        # make sure everything is the correct type
        if not self.equivalent(stat1) or not self.equivalent(stat2):
            raise ValueError(
                f'Attempting to merge statistics of different types:'
                f' self {self} \n stat1 {stat1} \n stat2 {stat2}')

        # make sure statistic has not yet been merged
        if self._statistic.var_name in self.binned_data:
            raise RuntimeError(
                f"{self._statistic.var_name} has already been calculated.")

        # merge
        result = self._statistic.merge(stat1._statistic, stat2._statistic)

        # add to the dataset
        dims: Union[DatasetCoordinates, List[str]] = self.binned_data.coords
        if len(dims) == 0:
            dims = ['None']
        self.binned_data.update({self._statistic.var_name: (dims, result)})

    def value(self) -> np.ndarray:
        return self._statistic.value()

    def equivalent(self, other: 'Statistic') -> bool:
        """Return True if two Statistic classes represent the same thing.

        The statistic type, variable, and diagnostic are checked.
        """
        return (
            self.type_ == other.type_ and
            self.variable == other.variable and
            self.diagnostic == other.diagnostic)
