# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Core Statistic classes.

These are the classes that directly implement a calc() method and are
therefore used to do the binning.
"""

import numpy as np

from bespin.core.statistic import StatisticBase, Statistic


class Count(StatisticBase):
    """Simple counts of the number of data points."""

    def __init__(self, variable, diagnostic, binned_data):
        # Count doesn't use a diagnostic in its DataSet naming
        super().__init__(variable, None, binned_data)

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('count', data)


class Sum(StatisticBase):
    """Sumation of diagnostics"""
    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('sum', data)


class Sum2(StatisticBase):
    """ Sum of squared differences from the mean. """
    def calc(self, data: np.ndarray) -> np.ndarray:
        # compute sum of squared differences from the mean.
        # Uses the already computed sum, to speed things up a bit.
        # Calculates using offset data (subtract global mean) to help
        # numerical stability, maybe.
        count = self._get('count')
        sum1 = self._get('sum')
        offset = np.nanmean(data)
        sum2 = self._binner('sum', (data-offset)**2)
        sum2_offset = np.true_divide(
            (sum1-offset*count)**2,
            count,
            where=count > 0)
        return sum2-sum2_offset

        # # Alternate form. don't use. Here for sanity check during debugging
        # return self._bin(lambda x: np.sum( (x-np.mean(x))**2 ), data)


class Min(StatisticBase):
    """The minimum value of the bin."""
    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('min', data)


class Max(StatisticBase):
    """The maximum value of the bin."""
    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('max', data)


# Register all of the above classes with the factory.
Statistic.register(Count)
Statistic.register(Sum)
Statistic.register(Sum2)
Statistic.register(Min)
Statistic.register(Max)
