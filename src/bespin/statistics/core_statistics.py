# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Core Statistic classes.

These are the classes that directly implement a calc() method and are
therefore used to do the binning.
"""

import numpy as np
import warnings

from bespin.core.statistic import StatisticBase, Statistic


class Count(StatisticBase):
    """Simple counts of the number of data points."""

    global_diagnostic = True
    """To be computed once for all diagnostics of a variable."""

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('count', data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        return stat1._get('count') + stat2._get('count')


class Sum(StatisticBase):
    """Sumation of diagnostics."""

    depends = ['count']

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('sum', data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        return stat1._get('sum') + stat2._get('sum')


class Sum2(StatisticBase):
    """Sum of squared differences from the mean."""

    depends = ['count', 'sum']

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

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        count1 = stat1._get('count')
        count2 = stat2._get('count')
        count = count1 + count2
        delta = stat1._get('mean') - stat2._get('mean')
        delta[count1 == 0] = 0.0
        delta[count2 == 0] = 0.0
        res = np.zeros_like(delta)
        np.true_divide(
            (delta**2)*count1*count2,
            count,
            res,
            where=count > 0)
        res += stat1._get('sum2') +stat2._get('sum2')
        return res


class Min(StatisticBase):
    """The minimum value of the bin."""

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('min', data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        # numpy warns in the case of nanmin(NaN, NaN). Supress those warning
        # because the resulting NaN is indeed what I want
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmin([stat1._get('min'), stat2._get('min')], axis=0)


class Max(StatisticBase):
    """The maximum value of the bin."""

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('max', data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        # numpy warns in the case of nanmax(NaN, NaN). Supress those warning
        # because the resulting NaN is indeed what I want
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmax([stat1._get('max'), stat2._get('max')], axis=0)


# Register all of the above classes with the factory.
Statistic.register(Count)
Statistic.register(Sum)
Statistic.register(Sum2)
Statistic.register(Min)
Statistic.register(Max)
