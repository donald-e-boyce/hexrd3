"""Test SparseArray ImageSeries"""
from pathlib import Path

import pytest
import numpy as np

from hexrd.imageseries import open
from hexrd.imageseries.save.sparse_array_hdf5 import SparseArrayHDF5


def test_init(tmp_path):
    sa = SparseArrayHDF5(tmp_path / "test.hdf5", "images")
    assert sa.nframes is None and sa.shape is None and sa.dtype is None
