"""Manage Data File for SparseArray Format"""
from pathlib import Path

import numpy as np
import h5py

from hexrd.imageseries.load.framecache import FrameCacheImageSeriesAdapter


class SparseArrayHDF5:
    """HDF5 Data File Manager for SparseArray

    PARAMETERS
    ----------
    h5filename: str or Path
        name of HDF5 file
    h5datagroup: str
        name or path to HDF5 DataGroup within the file
    """

    def __init__(self, h5filename, h5datagroup):
        self.h5filename = Path(h5filename)
        self.h5datagroup = h5datagroup

        # If datagroup exists, set core attributes.
        self.nframes, self.shape, self.dtype = 3 * (None,)
        if self.h5filename.exists():
            with h5py.File(self.h5filename, "r") as hf:
                if self.h5datagroup in hf:
                    self.nframes, self.shape, self.dtype = (
                        self._get_core_attrs(self.h5datagroup)
                    )

    def add_images(self, images, threshold=0, background=None):
        pass

    def get_images(self):
        """Return list of sparse images"""
        with  h5py.File(self.h5filename, "w") as hf:
            g = h5[self.h5datagroup]
            imgs = []
            for i in range(self.nframes):
                datname, indname, ptrname = SparseArrayHDF5.dataset_names(i)
                data = g[datname]
                indices = g[indname]
                indptr = g[ptrname]
                imgs.append(csr_matrix(data, indices, indptr))

        return imgs

    def copy_frame_cache(self, fcfile):
        """Write an HDF5 sparse array from a frame-cache file

        PARAMETERS
        ----------
        h5filename: str or Path
            name of HDF5 file
        h5datagroup: str
            name or path to HDF5 DataGroup within the file
        fcfile: str or Path
            name of frame-cache (npz) file
        """
        fc = FrameCacheImageSeriesAdapter(fcfile)
        with h5py.File(self.h5filename, "w") as hf:
            g = hf.create_group(self.h5datagroup)
            self._add_core_attrs(g, fc.nframes, fc.shape, fc.dtype)
            for i, frame in enumerate(fc.framelist):
                # frame is a csr_matrix
                dname, iname, pname = dataset_names(i)
                g.create_dataset(dname, data=frame.data)
                g.create_dataset(iname, data=frame.indices)
                g.create_dataset(pname, data=frame.indptr)
        # Don't forget to add the metadata

    @staticmethod
    def dataset_names(i):
        """DataSet names for frame i

        PARAMETERS
        ----------
        i: int
           index of frame

        RETURNS
        -------
        3-tuple of str
           name for data, indices, and indptr array

        """
        return f"data_{i}", f"indices_{i}", f"indptr_{i}"

    # TO DO: add metadata to core attributes (?)
    @staticmethod
    def _add_core_attrs(g, nframes, shape, dtype):
        g.attrs['_nframes'] = nframes
        g.attrs['_shape'] = shape
        g.attrs['_dtype'] = str(dtype)

    @staticmethod
    def _get_core_attrs(g):
        nframes = g.attrs['_nframes']
        shape = g.attrs['_shape']
        dtype = np.dtype(g.attrs['_dtype'])
        return nframes, shape, dtype
