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
