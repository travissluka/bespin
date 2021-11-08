# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Iterable
import re

from .statistic import Statistic


_default_statistics = ('count', 'sum', 'sum2')


class Diagnostic:
    """Describes the diagnostic/statistic combinations used for binning.

    Attributes:
        name: str name of the diagnostic
        statistics: A list of str matching bespin.Statistic classes defining
            what statistics are calculated when performing the binning.

    TODO:
        Add meta info (source of derived diagnostic input, etc)
    """
    def __init__(
            self,
            name: str,
            statistics: Iterable[str] = _default_statistics):
        self.name = name
        self.statistics = statistics

        # check for valid name
        if not re.fullmatch(r'\w+', name):
            raise ValueError(
                f'Illegal diagnostic name: "{name}"')

        # ensure listed statistics are valid
        bad_stats = set(statistics).difference(Statistic.types())
        if len(bad_stats):
            raise ValueError(
                f'invalid statistic(s) "{bad_stats}" for Diagnostic "{name}"')

    def __eq__(self, other) -> bool:
        return (
            self.name == other.name and
            set(self.statistics) == set(other.statistics))

    def __repr__(self) -> str:
        return f'Diagnostic("{self.name}", statistics={self.statistics})'

    @classmethod
    def from_str(cls, string: str) -> 'Diagnostic':
        """Create a Diagnostic instance from a string representation.

        The string is expected to be in the format:

        `<name>:<stat1>,<stat2>,...,<statN>`

        where the stat(s) are names of one or more of the core_statistics. If
        no statistics are given, a default of "sum,sum2" is assumed, which
        should cover most cases.

        Examples
        ---------
        - "ombg"
        - "ombg:count,sum,sum2"
        - "ombg:count,sum,sum2,min,max"
        """
        try:
            # if no statistics were given, use the defaults
            if ':' not in string:
                string = f'{string}:{",".join(_default_statistics)}'

            # split into diags and statistics
            splt = string.split(':')
            if len(splt) > 2:
                raise ValueError(
                    f'Incorrect number of ":" groups in Diagnostic string')
            diag = splt[0]
            stats = splt[1].split(',')
            diagnostic = cls(name=diag, statistics=stats)

        except (ValueError) as e:
            raise ValueError(
                f'Unable to parse Diagnostic string "{string}".'
                ' See documentation for correct format.'
            ) from e

        return diagnostic
