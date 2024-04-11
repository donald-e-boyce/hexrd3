"""Manage Data File for SparseArray Format"""
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
        self.h5filename = h5filename
        self.h5datagroup = h5datagroup

    def add_images(self, images, threshold=0, background=None):
        pass

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
            for i, frame in enumerate(fc.framelist):
                # frame is a csr_matrix
                name = f"frame-{i}"
                g.create_dataset(f"data_{i}", data=frame.data)
                g.create_dataset(f"indices_{i}", data=frame.indices)
                g.create_dataset(f"indptr_{i}", data=frame.indptr)
        # Don't forget to add the metadata
