# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Derived Statistics classes.

These are the classes that do NOT implement a calc() method. They perform
no binning, and simply return a derived value from existing binned stats.
"""

import numpy as np

from bespin.core.statistic import StatisticBase, Statistic


class Mean(StatisticBase):
    """Mean bin value.

    Requires `count` and `sum` to have already been calculated.
    """

    def value(self) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            count = self._get('count')
            sum_ = self._get('sum')
            return sum_ / count


class Variance(StatisticBase):
    """Population variance of a bin.

    Requires `count`, `sum`, and `sum2` to have already been calculated.
    """

    def value(self) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            count = self._get('count')
            sum2 = self._get('sum2')
            return np.where(count > 1, sum2/count, np.nan)


class StdDev(StatisticBase):
    """Population standard deviation of a bin.

    Requires `count`, `sum`, and `sum2` to have already been calculated.
    """

    # assuming population standard deviation
    def value(self) -> np.ndarray:
        return np.sqrt(self._get('variance'))


class RMSD(StatisticBase):
    """Root mean square deviation of a bin.

    Requires `count`, `sum`, and `sum2` to have already been calculated.
    """

    def value(self) -> np.ndarray:
        mean = self._get('mean')
        var = self._get('variance')
        return np.sqrt(var + mean**2)


# Register all of the above classes with the factory.
Statistic.register(Mean)
Statistic.register(Variance)
Statistic.register(StdDev)
Statistic.register(RMSD)
