import enum


class ScalarDataType(enum.Enum):
    LITHOLOGY = enum.auto()
    SCALAR_FIELD = enum.auto()
    VALUES = enum.auto()
    ALL = enum.auto()
    

class TopographyDataType(enum.Enum):
    TOPOGRAPHY = enum.auto()
    GEOMAP = enum.auto()
    SCALARS = enum.auto()  # Not implemented yet