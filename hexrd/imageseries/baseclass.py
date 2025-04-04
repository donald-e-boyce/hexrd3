"""Base class for imageseries
"""
import numpy as np

from .imageseriesabc import ImageSeriesABC, RegionType


class ImageSeries(ImageSeriesABC):
    """collection of images

    Basic sequence class with additional properties for image shape and
    metadata (possibly None).
    """

    def __init__(self, adapter):
        """Build FrameSeries from adapter instance

        *adapter* - object instance based on abstract Sequence class with
        properties for image shape, data type and metadata.
        """
        self._adapter = adapter

        return

    def __getitem__(self, key):
        return self._adapter[key]

    def __len__(self):
        return len(self._adapter)

    def __iter__(self):
        return self._adapter.__iter__()

    @property
    def dtype(self):
        return self._adapter.dtype

    @property
    def shape(self):
        return self._adapter.shape

    @property
    def metadata(self):
        return self._adapter.metadata

    def get_region(self, frame_idx: int, region: RegionType) -> np.ndarray:
        return self._adapter.get_region(frame_idx, region)
