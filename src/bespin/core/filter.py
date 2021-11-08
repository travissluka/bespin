# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

""" Filter base class and factory."""

from typing import MutableMapping, Type, FrozenSet
from abc import ABC,  abstractmethod
import re

import xarray as xr


class FilterBase(ABC):
    """The base class for all unbinned data Filter subclasses.

    The name of a filter is by default the lowercase/underscored version of the
    class name. However, this can be overridden by specifying `filter_name`
    when the class is defined.

    For example:
    ```
    class IodaMetadata(FilterBase):
    ```
    can be instantianced via
    ```
    f = Filter('ioda_metadata')
    ```

    whereas
    ```
    class Ioda2Coords(FilterBase, filter_name='some_filter'):
    ```
    would be instantiated via
    ```
    f = Filter('some_filter')
    ```
    """
    _filter_name: str
    per_variable = False

    def __init_subclass__(cls, filter_name=None, **kwargs) -> None:
        # Automatically register any subclasses of FilterBase.
        # The name used is a lowercase version of the class name
        # if "filter_name" is not given.
        if filter_name is None:
            filter_name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        cls._filter_name = filter_name
        Filter._register(cls)
        return super().__init_subclass__(**kwargs)  # type: ignore

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def filter(self, data: xr.Dataset) -> xr.Dataset:
        """Filter input unbinned data in some way."""
        pass


class __FilterFactory():
    _subclasses: MutableMapping[str, Type[FilterBase]] = {}

    def __call__(self, filter_name: str, *args, **kwargs) -> FilterBase:
        """Create a new unbinned data filter.

        Args:
        - filter_name: the name of one of the Filter subclasses
        - arg, kwargs: additional arguments required by the subclass
        """
        filter_name = str.lower(filter_name)
        if filter_name not in self._subclasses:
            raise ValueError(
                 f'Cannot create Filter "{filter_name}".'
                 ' It has not been registered.')
        return self._subclasses[filter_name](*args, **kwargs)

    @property
    def types(self) -> FrozenSet[str]:
        """Get a list of the registered filters."""
        return frozenset(self._subclasses.keys())

    def _register(self, cls: Type[FilterBase]) -> None:
        if cls._filter_name in self._subclasses:
            raise RuntimeError(
                f'Cannot register Filter "{cls._filter_name}".'
                ' It has already been registered.')
        self._subclasses[cls._filter_name] = cls

    def from_str(self, string: str) -> FilterBase:
        # TODO do error checking
        splt = string.split(':')

        return self(*splt)


Filter = __FilterFactory()
"""Factory for creating unbinned data filters.

Usage
------
get list of available filters:

>>> Filter.types

create a new filter:

>>> Filter(filter_name, *args, **kwargs)
"""
