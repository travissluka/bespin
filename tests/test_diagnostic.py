# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pytest

from bespin import Diagnostic


def test_diagnostic_create_bad():
    with pytest.raises(ValueError):
        Diagnostic('omb', statistics=('sum', 'm2', 'min', 'm'))


def test_diagnostic_create_default():
    d = Diagnostic('omb')
    assert set(d.statistics) == {'count', 'sum', 'sum2'}


def test_diagnostic_create():
    Diagnostic('omb', statistics={'count', 'sum', 'sum2', 'min', 'max'})


def test_diagnostic_eq():
    d1 = Diagnostic('omb', statistics={'count', 'sum', 'sum2', 'min', 'max'})

    assert d1 == Diagnostic('omb',
                            statistics={'count', 'sum', 'sum2', 'min', 'max'})
    assert d1 != Diagnostic('omf',
                            statistics={'count', 'sum', 'sum2', 'min', 'max'})
    assert d1 != Diagnostic('omb',
                            statistics={'count', 'sum'})


def test_diagnostic_parse():
    good_strings = (
        'omb',
        'omb:count,sum',
        'omb:count,sum,sum2',
        'omb:count,sum,sum2,min,max',
        )
    for s in good_strings:
        Diagnostic.from_str(s)

    bad_strings = (
        '',
        ':',
        'foo.bar',
        'omb:',
        'omb:foobar',
        'omb:foo:bar',
        )
    for s in bad_strings:
        print(f'testing: "{s}"')
        with pytest.raises(ValueError):
            Diagnostic.from_str(s)
