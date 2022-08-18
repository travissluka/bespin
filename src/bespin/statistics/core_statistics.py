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


def check(data: np.ndarray, pos=False):
    """safety check to make sure NaNs and Infs don't slip in."""
    assert np.count_nonzero(np.isnan(data)) == 0
    assert np.count_nonzero(np.isinf(data)) == 0
    if pos:
        assert np.count_nonzero(data < 0) == 0


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
        sum_val = self._binner('sum', data)
        check(sum_val)
        return sum_val

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        sum1 = stat1._get('sum')
        sum2 = stat2._get('sum')
        check(sum1)
        check(sum2)

        value = sum1 + sum2
        check(value)
        return value


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

        # quick exit if input data is empty
        if not len(data):
            return np.zeros_like(count)

        offset = np.nanmean(data)
        sum2 = self._binner('sum', (data-offset)**2)
        sum2_offset = np.zeros_like(sum2)
        np.divide(
            (sum1-offset*count)**2,
            count,
            out=sum2_offset,
            where=count > 0)
        value = sum2 - sum2_offset

        # rounding errors are resulting in negative numbers when variance is small for a bin
        value[value < 0.0] = 0.0
        check(value, pos=True)
        return value

        # # Alternate form. don't use. Here for sanity check during debugging
        # return self._bin(lambda x: np.sum( (x-np.mean(x))**2 ), data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        count1 = stat1._get('count')
        count2 = stat2._get('count')
        count = count1 + count2

        mean1 = np.where(count1 > 0, stat1._get('mean'), 0.0)
        mean2 = np.where(count2 > 0, stat2._get('mean'), 0.0)
        delta = mean1 - mean2
        check(delta)

        value = np.zeros_like(delta)
        np.divide(
            (delta**2)*count1*count2,
            count,
            out=value,
            where=count > 0)
        value += stat1._get('sum2') + stat2._get('sum2')
        check(value, pos=True)
        return value


class Min(StatisticBase):
    """The minimum value of the bin."""

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('min', data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        # numpy warns in the case of nanmin(NaN, NaN). Suppress those warning
        # because the resulting NaN is indeed what I want
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmin([stat1._get('min'), stat2._get('min')], axis=0)


class Max(StatisticBase):
    """The maximum value of the bin."""

    def calc(self, data: np.ndarray) -> np.ndarray:
        return self._binner('max', data)

    def merge(self, stat1: StatisticBase, stat2: StatisticBase) -> np.ndarray:
        # numpy warns in the case of nanmax(NaN, NaN). Suppress those warning
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
