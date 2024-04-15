"""Sparse Array imageseries"""
import numpy as np

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator
from .. save.sparse_array_hdf5 import SparseArrayHDF5


class SparseArrayImageSeriesAdapter(ImageSeriesAdapter):
    """Collection of Images in numpy array

    This is instantiated with the name or path of the HDF5 file and a keyword
    argument, `h5datagroup`, giving the DataGroup path inside the file.

    Parameters
    ----------
    fname: str or Path
       name of HDF5 file with sparse array data
    h5datagroup: str
        name or path to HDF5 DataGroup within the file
    """
    format = 'sparse-array'

    def __init__(self, fname, h5datagroup=None):
        self.fname = fname
        if h5datagroup is None:
            raise ValueError('No value was given for "h5datagroup."')
        self.sparse_h5 = SparseArrayHDF5(fname, h5datagroup)
        self._sparse_images = self.sparse_h5.get_images()
        self._metadata = self.sparse_h5.get_metadata()

    def __get_item__(self, key):
        return self.sparse_images[key]

    def __len__(self):
        return self._nframes

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def metadata(self):
        return self._metadata

    @property
    def sparse_images(self):
        return self._sparse_images
