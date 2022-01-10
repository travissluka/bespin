import pytest
import subprocess as sp
import os
import bespin.io.obs_ioda as obsio

skip = pytest.mark.skipif(
    not obsio.ioda_found,
    reason='IODA library not found.')

binning = {
    'global': '',
    'lat': '-b latitude:r=5.0',
    'latlon': '-b latitude:r=1.0 -b longitude:r=1.0',
    }

for d in ('binned','merged'):
    if not os.path.exists(d):
        os.mkdir(d)

@pytest.fixture(params=binning.keys())
def binning_name(request):
    return request.param


@pytest.fixture(params=('avhrr3_metop-a','sondes'))
def sat(request):
    return request.param

@pytest.fixture(params=('03','09','15'))
def hr(request):
    return request.param

@pytest.fixture
def obs_file(sat, hr):
    return (
        f'obs/jedi_gdas_023/gfs.jedi_gdas_023.diag.PT6H.'
        f'{sat}.2020-12-15T{hr}:00:00Z.PT6H.nc4')

@skip
def test_ioda_obs_bin(obs_file, sat, hr, binning_name):
    out_fn = f'binned/{binning_name}.{sat}.{hr}.nc'
    if os.path.exists(out_fn):
        os.remove(out_fn)

    cmd=(
        f'bespin bin {obs_file}  -o {out_fn} '
        f' -d ObsValue {binning[binning_name]}')
    sp.check_call(cmd, shell=True)

    # TODO do a more meaningful check than this
    assert os.path.exists(out_fn)

@skip
def test_ioda_obs_merge(sat, binning_name):
    out_fn = f'merged/{binning_name}.{sat}.nc'
    if os.path.exists(out_fn):
        os.remove(out_fn)

    cmd=(
        f'bespin merge binned/{binning_name}.{sat}.*.nc'
        f' -o {out_fn}')
    sp.check_call(cmd, shell=True)

    # TODO do a more meaningful check than this
    assert os.path.exists(out_fn)
