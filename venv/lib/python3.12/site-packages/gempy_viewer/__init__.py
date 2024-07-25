import sys

from gempy_viewer.modules.plot_3d.vista import GemPyToVista
from .API import *
import gempy
__all__ = ['plot_2d', 'plot_3d', 'plot_section_traces', 'plot_topology', 'plot_stereonet']


# Assert at least pyton 3.10
assert sys.version_info[0] >= 3 and sys.version_info[1] >= 10, "GemPy requires Python 3.10 or higher"
__version__ = gempy.__version__

if __name__ == '__main__':
    pass
