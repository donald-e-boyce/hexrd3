import abc
import pkgutil

from ..imageseriesabc import ImageSeriesABC
from .registry import Registry

# Metaclass for adapter registry

class _RegisterAdapterClass(abc.ABCMeta):

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)

class ImageSeriesAdapter(ImageSeriesABC, metaclass=_RegisterAdapterClass):

    format = None

# import all adapter modules

from . import (
    array, framecache, hdf5, imagefiles, rawimage, metadata, trivial,
    sparse_array
)
