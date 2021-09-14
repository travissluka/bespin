# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Binned Experiment Statistics Package for INtegrated diagnostics.

Package for reading JEDI IODA observation diagnostic files and binning those
observations in various ways. Tools are provided to manipulate and combine
those binned statistics and plot them.
"""

from bespin.core.binned_statistics import BinnedStatistics
from bespin.core.diagnostic import Diagnostic
from bespin.core.dimension import Dimension
from bespin.core.statistic import Statistic

import bespin.statistics as statistics

__all__ = [
    'BinnedStatistics',
    'Diagnostic',
    'Dimension',
    'Statistic',
    'core',
    'statistics',
]
