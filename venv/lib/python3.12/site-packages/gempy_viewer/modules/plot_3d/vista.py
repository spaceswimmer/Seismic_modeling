"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    gempy is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to visualized structural geology data and
    potential fields of the regional modelling based on the potential field
    method. Tested on Windows 10

    Created on 08.04.2020

    @author: Miguel de la Varga, Bane Sullivan, Alexander Schaaf, Jan von Harten
"""
from __future__ import annotations

import warnings
from typing import Union, List, Optional

import numpy as np

import matplotlib

warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk

    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False

from gempy_viewer.optional_dependencies import require_pyvista



class GemPyToVista:

    def __init__(self, extent: Union[np.ndarray | list[float]], plotter_type: str = 'basic',
                 live_updating=False, pyvista_bounds_kwargs: Optional[dict] = None, **kwargs):
        """GemPy 3-D visualization using pyVista.

        Args:
            model (gp.Model): Geomodel instance with solutions.
            plotter_type (str): Set the plotter type. Defaults to 'basic'.
            extent (List[float], optional): Custom extent. Defaults to None.
            lith_c (pn.DataFrame, optional): Custom color scheme in the form of
                a look-up table. Defaults to None.
            live_updating (bool, optional): Toggles real-time updating of the
                plot. Defaults to False.
            **kwargs:

        """

        pv = require_pyvista()
        
        if pyvista_bounds_kwargs is None:
            pyvista_bounds_kwargs = {}

        # Override default notebook value
        pv.set_plot_theme("document")
        kwargs['notebook'] = kwargs.get('notebook', False)

        # plotting options
        self.live_updating = live_updating

        # Choosing plotter
        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
            self.p.view_isometric(negative=False)
        elif plotter_type == 'notebook':
            raise NotImplementedError
            # self.p = pv.PlotterITK()
        elif plotter_type == 'background':
            raise NotImplementedError
        else:
            raise AttributeError('Plotter type must be basic, background or notebook.')

        self.plotter_type = plotter_type

        # Default camera and bounds
        self.set_bounds(extent, **pyvista_bounds_kwargs)
        self.p.view_isometric(negative=False)

        # Actors containers
        self.surface_actors = {}
        self.surface_poly = {}

        self.regular_grid_actor = None
        self.regular_grid_mesh = None

        self.surface_points_actor = None
        self.surface_points_mesh = None
        self.surface_points_widgets = {}

        self.orientations_actor = None
        self.orientations_mesh = None
        self.orientations_widgets = {}

        # Private attributes
        col = matplotlib.colormaps['viridis'](np.linspace(0, 1, 255)) * 255
        nv = pv.convert_array(col, array_type=3)
        self._cmaps = {'viridis': nv}

        # Topology properties
        self.topo_edges = None
        self.topo_ctrs = None

    def set_bounds(self, extent: List[float], **kwargs):
        self.p.show_bounds(bounds=extent, **kwargs)

    @property
    def scalar_bar_arguments(self):
        sargs = dict(
            title_font_size=20,
            label_font_size=16,
            shadow=True,
            italic=True,
            font_family="arial",
            height=0.25,
            vertical=True,
            position_x=0.15,
            title="id",
            fmt="%.0f",
        )
        return sargs
