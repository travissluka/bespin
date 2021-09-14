# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pytest
import numpy as np
import xarray as xr

from collections import namedtuple
import bespin as bn
from bespin.core.statistic import Statistic, StatisticBase


# all of the tests will be run twice, once for 2D-latlon binning, and once
# for 1D-latitude-only binning
@pytest.fixture(params=('lat', 'latlon'))
def bins(request):
    _bins = {
        'lat': [
            bn.Dimension("latitude", resolution=90)],
        'latlon': [
            bn.Dimension("latitude", resolution=90),
            bn.Dimension("longitude", resolution=180)],
    }
    return _bins[request.param]


@pytest.fixture
def init_args(bins):
    return {
        'variable': 'test',
        'diagnostic': 'test',
        'binned_data': xr.Dataset(
            coords=xr.merge([b.centers_to_xarray() for b in bins])
            ),
        }


@pytest.fixture
def bin_args(bins):
    """
    Carefully construct input data so that the resulting 4
    bins will contain 0,1,2, and 3 input points.
    """
    return {
        'bins': bins,
        'unbinned_data': xr.Dataset(
            coords={
                'latitude': (
                    ('nloc',),
                    np.array([10.0, -10.0, -25.0, -30.0, -40.0, -50.0])),
                'longitude': (
                    ('nloc',),
                    np.array([200.0, 95.0, 12.0, 190, 200, 210])),
                },
            data_vars={
                'test.test': (
                    ('nloc',),
                    np.array([1, 2, 3, 4, 5, 6]))
            }
        ),
    }


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def test_statistic_badinit(init_args):
    # incorrect statistic type name
    with pytest.raises(ValueError):
        Statistic('bad_type', **init_args)


def test_statistic_badvalue_nocalc(init_args):
    # calling value() before calc() was called
    with pytest.raises(RuntimeError):
        Statistic('count', **init_args).value()


def test_statistic_badcalc_nodep(init_args, bin_args):
    # calling calc() before a depenency was calculated
    with pytest.raises(RuntimeError):
        Statistic('sum2', **init_args).calc(**bin_args)


def test_statistic_badcalc_nodiag(init_args, bin_args):
    with pytest.raises(RuntimeError):
        s = Statistic('sum', **{**init_args, 'diagnostic': 'foobar'})
        s.calc(**bin_args)


# def test_statistic_badcalc_nodiag(init_args, bin_args):
#     with pytest.raises(ValueError):
#         s = Statistic('sum', **{**init_args, 'diagnostic': None})
#         s.calc(**bin_args)


# ------------------------------------------------------------------------------
# The following are the expected answers for each statistic type.
# The first set of answers is for 2D latlon binning. The second set of answers
# is for 1D latitude only binning.
# ------------------------------------------------------------------------------
StatParam = namedtuple(
    'StatParam',
    ('latlon', 'lat', 'deps'),
    defaults=(None, None))

core_params = {
    'count': StatParam(
        [[2.0, 3.0], [0.0, 1.0]],
        [5.0, 1.0],),
    'sum': StatParam(
        [[5.0, 15.0], [0.0, 1.0]],
        [20.0, 1.0],),
    'sum2': StatParam(
        [[0.5, 2.0], [0.0, 0.0]],
        [10.0, 0.0],
        ('count', 'sum',)),
    'min': StatParam(
        [[2.0, 4.0], [np.nan, 1.0]],
        [2.0, 1.0],),
    'max': StatParam(
        [[3.0, 6.0], [np.nan, 1.0]],
        [6.0, 1.0],),
}
derived_params = {
    'mean': StatParam(
        [[2.5, 5.0], [np.nan, 1.0]],
        [4.0, 1.0],
        ('count', 'sum')),
    'variance': StatParam(
        [[0.25, 2.0/3.0], [np.nan, np.nan]],
        [2.0, np.nan],
        ('count', 'sum', 'sum2')),
    'stddev': StatParam(
        [np.sqrt([0.25, 2.0/3.0]), [np.nan, np.nan]],
        [np.sqrt(2.0), np.nan],
        ('count', 'sum', 'sum2')),
    'rmsd': StatParam(
        [np.sqrt([13.0/2, 77.0/3.0]), [np.nan, np.nan]],
        [np.sqrt(18.0), np.nan],
        ('count', 'sum', 'sum2')),

}
all_params = {**core_params, **derived_params}


@pytest.fixture
def calc(init_args, bin_args):
    def _calc(type_, dependencies):
        if dependencies:
            for d in dependencies:
                s = Statistic(d, **init_args)
                s.calc(**bin_args)
        stat = Statistic(type_, **init_args)
        if type_ in core_params:
            stat.calc(**bin_args)
        return stat
    return _calc


@pytest.fixture(params=core_params.keys())
def core_statistics(request):
    return request.param


@pytest.fixture(params=all_params.keys())
def all_statistics(request):
    return request.param


@pytest.fixture
def answers(all_statistics, bins):
    if len(bins) == 2:
        return np.array(all_params[all_statistics].latlon)
    else:
        return np.array(all_params[all_statistics].lat)


def test_statistic_calc(core_statistics, calc):
    calc(core_statistics, core_params[core_statistics].deps)


def test_statistic_value(all_statistics, calc, answers):
    print(calc)
    stat = calc(all_statistics, all_params[all_statistics].deps)
    val = stat.value()
    assert np.array_equal(answers, val, equal_nan=True)
