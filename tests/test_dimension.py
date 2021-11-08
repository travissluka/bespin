# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from bespin import Dimension
import numpy as np
import pytest


def test_dimension_init_bad():
    with pytest.raises(ValueError):
        Dimension(name='latitude', edges=[1], resolution=5)
    with pytest.raises(ValueError):
        Dimension(name='latitude', edges=[1])
    with pytest.raises(ValueError):
        Dimension(name='latitude', edges=[1, 2], bounds=[1, 2])
    with pytest.raises(ValueError):
        Dimension(name="depth", resolution=5)


def test_dimension_init_resolution():
    assert len(Dimension(name='latitude', resolution=10, bounds=(0, 90))) == 9
    assert len(Dimension(name='latitude', resolution=1)) == 180
    assert len(Dimension(name='longitude', resolution=1)) == 360


def test_dimension_init_edges():
    d = Dimension(name='latitude', edges=[1, 2, 3, 4, 5])
    assert len(d) == 4


def test_dimension_centers():
    d = Dimension(name='latitude', edges=[1, 3, 5])
    assert np.array_equal(d.bin_centers, np.array([2, 4]))


def test_dimension_eq():
    d1 = Dimension(name='latitude', resolution=90)

    assert d1 == Dimension(name='latitude', edges=[-90, 0, 90])
    assert d1 != Dimension(name='latitude', resolution=10)
    assert d1 != Dimension(name='longitude', resolution=90)


def test_dimension_parse():
    good_strings = (
        'latitude:r=1',
        'latitude:r=1.0',
        'latitude:r=1:b=-10,90',
        'latitude:r=1.0:b=-10.0,90.0',
        'latitude:e=-1,0,1,2,3,4,5',
        'latitude:e=-1.0,0.0,1.0,2.0,3.0,4.0,5.0',
        'longitude:r=1',
        'longitude:r=1:b=0,180'
        )
    for s in good_strings:
        Dimension.from_str(s)

    bad_strings = (
        'latitude',                # missing all args
        'latitude:r=1.0,',         # bad arg type
        'latitude:r=1.0:f=1.0',
        'latitude:r=1.0:b=0.0,90.0:f=1.0',
        'latitude:r=1,2,3',        # bad r values
        'latitude:r=foo',
        'latitude:r=1:b=0,90,95',  # bad b values
        'latitude:r=1:b=0',
        'latitude:e=0.0',          # bad e value
        'latitude:e=1.0,foo',
        'latitude:r=1:e=0,2,3',    # incompatible combination or r,e,b
        'latitude:b=0,90',
        'latitude:r=1:b=0,90:e=0,2,3',
        'time:r=1.0',
        'latitude:b=1,2,3:e=0,2',
    )
    for s in bad_strings:
        print(f'testing: "{s}"')
        with pytest.raises(ValueError):
            d = Dimension.from_str(s)
