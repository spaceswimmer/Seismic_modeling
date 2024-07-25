import warnings

from ._plot_2d_API import plot_2d, plot_section_traces, plot_topology 
from ._plot_LiquidEarth import plot_to_liquid_earth
# check if pyvista is installed
try:
    from ._plot_3d_API import plot_3d
except ImportError:
    warnings.warn("The pyvista package is required to plot 3D models. ")
from ._plot_others import plot_stereonet