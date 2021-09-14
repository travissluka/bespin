# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from typing import Iterable

from bespin.core.statistic import Statistic


class Diagnostic:
    """Describes the diagnostic/statistic combinations used for binning.

    Attributes:
        name: str name of the diagnostic
        statistics: A list of str matching bespin.Statistic classes defining
            what statistics are calculated when performing the binning.

    TODO:
        Add meta info (source of derived diagnostic input, etc)
    """
    def __init__(self, name: str, statistics: Iterable[str] = ('sum', 'sum2')):
        self.name = name
        self.statistics = statistics

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
