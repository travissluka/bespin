# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import xarray as xr
import pytest
from itertools import product

import bespin as bn
from bespin.core.binned_statistics import BinnedStatistics

diagnostics = (
    bn.Diagnostic('omb', statistics=('sum', 'sum2', 'min', 'max')),
    bn.Diagnostic('omf', statistics=('sum', 'sum2')),
    )

bins = (
    bn.Dimension("latitude", resolution=10),
    bn.Dimension("longitude", resolution=10),
    )


# ------------------------------------------------------------------------------
# Pytest test fixtures.
# We will be running each test multiple times, for various combinations of
# 1) 1D vs 2D binning
# 2) non-channel and channel test data
# ------------------------------------------------------------------------------
@pytest.fixture(params=('0D', '1D', '2D'))
def init_args(request):
    """Test with a global, 1D latitude, and 2D latlon binning."""
    _bins = {
        '0D': [],
        '1D': bins[0:1],
        '2D': bins[0:2],
        }
    return {
        'bins': _bins[request.param],
        'diagnostics':  diagnostics,
    }


@pytest.fixture(params=('', 'multichannel'))
def unbinned_data(request):
    """Test with channel-less data, and multichannel test data."""
    np.random.seed(0)

    # dimensions and coordinates present in both cases
    nlocs = 5000
    dims = {'nlocs': nlocs}
    coords = {
        b.name: (
            ('nlocs',),
            np.random.uniform(
                low=b.bounds[0], high=b.bounds[1], size=(nlocs,)))
        for b in bins}

    # case specific parameters
    if request.param == 'multichannel':
        variables = ['brightness_temperature']
        channels = np.array([1, 3, 5, 6, 7, 8, 10, 12, 42, 99])
        dims['nchans'] = len(channels)
        coords['sensor_channel'] = (('nchans',), channels)
    else:
        variables = ['sea_water_temperature', 'sea_water_salinity']

    # generate fake data
    data = xr.Dataset(
        coords=coords,
        data_vars={
            f'{v}.{d.name}': (
                dims.keys(),
                100+10*np.random.normal(size=list(dims.values())))
            for v, d in product(variables, diagnostics)},
    )
    return variables, data


@pytest.fixture
def binned_stat(init_args, unbinned_data):
    """pre-generated binned data, for tests that need that."""
    bs = BinnedStatistics('binning_name', **init_args)
    variables, data = unbinned_data
    for var in variables:
        bs._bin_variable(var, data)
    return bs


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def test_binned_statistics_init(init_args):
    BinnedStatistics('binning_name', **init_args)


def test_binned_statistics_bin_bad_duplicate(init_args, unbinned_data):
    bs = BinnedStatistics('latlon', **init_args)
    variables, data = unbinned_data
    bs._bin_variable(variables[0], data)
    with pytest.raises(ValueError):
        bs._bin_variable(variables[0], data)


def test_binned_statistics_bin(binned_stat, unbinned_data):
    # if a variable dimension is present that was not in the binning
    # dimensions, that dimension is passed through due to multichannel input.
    # Make sure it is passed through unaltered
    dims = (set(binned_stat._data.coords) -
            set([b.name for b in binned_stat.bins]))
    for d in dims:
        # note: dimension name is likely different, so just check values
        assert(np.array_equal(
            binned_stat._data.coords[d].values,
            unbinned_data[1].coords[d].values))

    # TODO other tests (eventually?) to make sure binning was done correctly


def test_binned_statistics_read_write_equal(binned_stat: BinnedStatistics):
    filename = 'test_output'
    binned_stat.write(filename)
    read_stats = BinnedStatistics.read(filename)

    assert binned_stat.equivalent(read_stats)
    assert binned_stat._data.identical(read_stats._data)


def test_binned_statistics_print(binned_stat: BinnedStatistics):
    print(binned_stat)


def test_binned_statistics_get(binned_stat: BinnedStatistics):
    stats = binned_stat.get()


def test_binned_statistics_get_subset(binned_stat: BinnedStatistics):
    stats = binned_stat.get(
        diagnostic='omb',
        statistic=('count', 'rmsd', 'stddev', 'mean', 'min', 'max'),
        )
