# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pytest
import numpy as np
import xarray as xr
import copy

import bespin as bn
from bespin.core.statistic import Statistic

core_params = ['count', 'sum', 'sum2', 'min', 'max']
derived_params = ['mean', 'variance', 'stddev', 'rmsd']

answers = {
    'count': {
        '1D': [5.0, 1.0],
        '2D': [[2.0, 3.0], [0.0, 1.0]],
        },
    'sum': {
        '1D': [20.0, 1.0],
        '2D': [[5.0, 15.0], [0.0, 1.0]],
        },
    'sum2': {
        '1D': [10.0, 0.0],
        '2D': [[0.5, 2.0], [0.0, 0.0]],
        },
    'min': {
        '1D': [2.0, 1.0],
        '2D': [[2.0, 4.0], [np.nan, 1.0]],
        },
    'max': {
        '1D': [6.0, 1.0],
        '2D': [[3.0, 6.0], [np.nan, 1.0]],
        },
    'mean': {
        '1D': [4.0, 1.0],
        '2D': [[2.5, 5.0], [np.nan, 1.0]],
        },
    'variance': {
        '1D': [2.0, np.nan],
        '2D': [[0.25, 2.0/3.0], [np.nan, np.nan]],
        },
    'stddev': {
        '1D': [np.sqrt(2.0), np.nan],
        '2D': [np.sqrt([0.25, 2.0/3.0]), [np.nan, np.nan]],
        },
    'rmsd': {
        '1D': [np.sqrt(18.0), np.nan],
        '2D': [np.sqrt([13.0/2, 77.0/3.0]), [np.nan, np.nan]],
        },
}

unbinned_data = xr.Dataset(
    coords={
        'latitude': (
            ('nloc',),
            np.array([10.0, -10.0, -25.0, -30.0, -40.0, -50.0])),
        'longitude': (
            ('nloc',),
            np.array([200.0, 95.0, 12.0, 190, 200, 210])),
        },
    data_vars={
        'test/test': (
            ('nloc',),
            np.array([1, 2, 3, 4, 5, 6]))
    }
)


@pytest.fixture(params=('1D', '2D'))
def bin_dims(request):
    return request.param


@pytest.fixture
def bins(bin_dims):
    _bins = {
        '1D': [
            bn.Dimension("latitude", resolution=90)],
        '2D': [
            bn.Dimension("latitude", resolution=90),
            bn.Dimension("longitude", resolution=180)],
    }
    return _bins[bin_dims]


@pytest.fixture
def init_args(bins):
    return {
        'variable': 'test',
        'diagnostic': 'test',
        'binned_data': xr.Dataset(
            coords=xr.merge([b.centers_to_xarray() for b in bins])
            ),
        }


@pytest.fixture(params=core_params)
def core_statistics(request):
    return request.param


@pytest.fixture(params=derived_params)
def derived_statistics(request):
    return request.param


@pytest.fixture(params=core_params + derived_params)
def all_statistics(request):
    return request.param


# ------------------------------------------------------------------------------
# Test things that are supposed to break
# ------------------------------------------------------------------------------
def test_statistic_badinit(init_args):
    # incorrect statistic type name
    with pytest.raises(ValueError):
        Statistic('bad_type', **init_args)


def test_statistic_badvalue_nocalc(init_args):
    # calling value() before calc() was called
    with pytest.raises(RuntimeError):
        Statistic('count', **init_args).value()


def test_statistic_badcalc_nodep(init_args, bins):
    # calling calc() before a depenency was calculated
    with pytest.raises(RuntimeError):
        Statistic('sum2', **init_args).calc(bins, unbinned_data)


def test_statistic_badcalc_nodiag(init_args, bins):
    # calc() should fail if a non existant diagnostic is specified
    s = Statistic('sum', **{**init_args, 'diagnostic': 'foobar'})
    with pytest.raises(RuntimeError):
        s.calc(bins, unbinned_data)


def test_statistic_badcalc_duplicate(init_args, bins):
    # calc() should fail if we try calculating more than once
    Statistic('count', **init_args).calc(bins, unbinned_data)
    Statistic('sum', **init_args).calc(bins, unbinned_data)
    stat = Statistic('sum', **init_args)
    with pytest.raises(RuntimeError):
        stat.calc(bins, unbinned_data)


def test_statistic_badcalc_derived(derived_statistics, init_args, bins):
    # calc() should fail for derived statistics
    stat = Statistic(derived_statistics, **init_args)
    with pytest.raises(RuntimeError):
        stat.calc(bins, unbinned_data)


# def test_statistic_badmerge_derived(derived_statistics, init_args, bins):
#     # merge() should fail for derived statistics


def test_statistic_badmerge_duplicate(init_args, bins):
    # merge() should fail if we try calculating more than once

    # calculate "count" and "sum" on two copies of the set
    init_args_copy = [copy.deepcopy(init_args) for i in range(2)]
    c = [Statistic('count', **i) for i in init_args_copy]
    [c.calc(bins, unbinned_data) for c in c]
    s = [Statistic('sum', **i) for i in init_args_copy]
    [s.calc(bins, unbinned_data) for s in s]

    # merge "count" and "sum"
    Statistic('count', **init_args).merge(*c)
    Statistic('sum', **init_args).merge(*s)

    # merging "sum" a second time should fail
    with pytest.raises(RuntimeError):
        Statistic('sum', **init_args).merge(*s)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def test_statistic_calc(core_statistics, init_args, bins, bin_dims):
    # calc() is performed for each core statistic, and the answers
    # are checked against manually computed values.
    stat = Statistic(core_statistics, **init_args)

    # calculate dependencies
    for d in stat.dependencies:
        Statistic(d, **init_args).calc(bins, unbinned_data)

    stat.calc(bins, unbinned_data)

    assert np.array_equal(
        answers[core_statistics][bin_dims],
        stat._statistic.binned_data.variables[stat._statistic.var_name],
        equal_nan=True)


def test_statistic_value(all_statistics, init_args, bins, bin_dims):
    # value() is performed for core and derived statistics, and the answers
    # are checked against manually computed values.
    stat = Statistic(all_statistics, **init_args)

    # calculate dependencies (and self, if a core statistic)
    for d in stat.dependencies + [s for s in core_params if s == stat.type_]:
        Statistic(d, **init_args).calc(bins, unbinned_data)

    assert np.array_equal(
        answers[all_statistics][bin_dims],
        stat.value(),
        equal_nan=True)


def test_statistic_merge(core_statistics, init_args, bins, bin_dims):
    # To test the merge functionality: The unbinned input data is split into
    # two subsets (round robin style). The appropriate statistic is calculated
    # on the two subsets, and then merged. The resulting answer *should* be
    # identical to if the statistic had been calculated on the full unbinned
    # input data
    stat = Statistic(core_statistics, **init_args)

    # split the unbinned data into two subsets
    split_init_args = [copy.deepcopy(init_args) for i in range(2)]
    split_unbinned_data = [
        unbinned_data.isel(nloc=range(0, unbinned_data.dims['nloc'], 2)),
        unbinned_data.isel(nloc=range(1, unbinned_data.dims['nloc'], 2)),
        ]

    # calc stats for the split subsets
    for i, u in zip(split_init_args, split_unbinned_data):
        for d in stat.dependencies + [stat.type_]:
            Statistic(d, **i).calc(bins, u)

    # merge the two split stats
    stat.merge(*[Statistic(stat.type_, **i) for i in split_init_args])

    # make sure answers match
    assert np.array_equal(
        answers[core_statistics][bin_dims],
        stat._statistic.binned_data.variables[stat._statistic.var_name],
        equal_nan=True)
