import os
from unittest import mock

from nndet.io.datamodule.module import get_allowed_n_proc_DA


@mock.patch.dict(os.environ, {"det_num_threads": "32"})
def test_get_allowed_n_proc_DA_env():
    proc = get_allowed_n_proc_DA()
    assert proc == 32


@mock.patch("subprocess.getoutput", return_value="hdf19-gpu01")
@mock.patch.dict(os.environ, {}, clear=True)
def test_get_allowed_n_proc_DA_t1(mock):
    proc = get_allowed_n_proc_DA()
    assert proc == 12


@mock.patch("subprocess.getoutput", return_value="lsf22-gpu01")
@mock.patch.dict(os.environ, {}, clear=True)
def test_get_allowed_n_proc_DA_t2(mock):
    proc = get_allowed_n_proc_DA()
    assert proc == 28
