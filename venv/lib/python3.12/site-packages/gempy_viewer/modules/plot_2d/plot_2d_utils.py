from typing import Sequence

import numpy as np
import matplotlib.colors as mcolors

from gempy.core.data import Grid
from gempy.core.data.core_utils import calculate_line_coordinates_2points, interpolate_zvals_at_xy
from gempy.core.data.grid_modules import grid_types
from gempy.core.data.grid_modules import Sections, RegularGrid


def slice_cross_section(regular_grid: grid_types.RegularGrid, direction: str, cell_number: int or str):
    """
    Slice the 3D array (blocks or scalar field) in the specific direction selected in the plot functions

    """
    _a, _b, _c = (
        slice(0, regular_grid.resolution[0]),
        slice(0, regular_grid.resolution[1]),
        slice(0, regular_grid.resolution[2])
    )

    extent = regular_grid.extent
    if direction == "x":
        cell_number = int(regular_grid.resolution[0] / 2) if cell_number == 'mid' else cell_number
        _a, x, y, Gx, Gy = cell_number, "Y", "Z", "G_y", "G_z"
        extent_val = extent[[2, 3, 4, 5]]
    elif direction == "y":
        cell_number = int(regular_grid.resolution[1] / 2) if cell_number == 'mid' else cell_number
        _b, x, y, Gx, Gy = cell_number, "X", "Z", "G_x", "G_z"
        extent_val = extent[[0, 1, 4, 5]]
    elif direction == "z":
        cell_number = int(regular_grid.resolution[2] / 2) if cell_number == 'mid' else cell_number
        _c, x, y, Gx, Gy = cell_number, "X", "Y", "G_x", "G_y"
        extent_val = extent[[0, 1, 2, 3]]
    else:
        raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
    return _a, _b, _c, extent_val, x, y, Gx, Gy


def make_section_xylabels(sections: Sections, section_name, n=5):
    """
    @elisa heim
    Setting the axis labels to any combination of vertical crossections

    Args:
        section_name: name of a defined gempy crossection. See gempy.Model().grid.section
        n:

    Returns:

    """
    if n > 5:
        n = 3  # todo I don't know why but sometimes it wants to make a lot of xticks
    elif n < 0:
        n = 3

    j = np.where(sections.names == section_name)[0][0]
    startend = list(sections.section_dict.values())[j]
    p1, p2 = startend[0], startend[1]
    xy = calculate_line_coordinates_2points(p1, p2, n)
    if len(np.unique(xy[:, 0])) == 1:
        labels = xy[:, 1].astype(int)
        axname = 'Y'
    elif len(np.unique(xy[:, 1])) == 1:
        labels = xy[:, 0].astype(int)
        axname = 'X'
    else:
        labels = [str(xy[:, 0].astype(int)[i]) + ',\n' + str(xy[:, 1].astype(int)[i]) for i in
                  range(xy[:, 0].shape[0])]
        axname = 'X,Y'
    return labels, axname


def check_default_section(ax, section_name, cell_number, direction):
    if section_name is None:
        try:
            section_name = ax.section_name
        except AttributeError:
            pass
    if cell_number is None:
        try:
            cell_number = ax.cell_number
            direction = ax.direction
        except AttributeError:
            pass

    return section_name, cell_number, direction


def slice_topo_4_sections(grid: Grid, p1, p2, resx, method='interp2d'):
    """
    Slices topography along a set linear section

    Args:
        :param p1: starting point (x,y) of the section
        :param p2: end point (x,y) of the section
        :param resx: resolution of the defined section
        :param method: interpolation method, 'interp2d' for cubic scipy.interpolate.interp2d
                                         'spline' for scipy.interpolate.RectBivariateSpline

    Returns:
        :return: returns x,y,z values of the topography along the section
    """
    xy = calculate_line_coordinates_2points(p1, p2, resx)
    z = interpolate_zvals_at_xy(xy, grid.topography, method)
    return xy[:, 0], xy[:, 1], z


def calculate_p1p2(regular_grid: RegularGrid, direction, cell_number):
    if direction == 'y':
        cell_number = int(regular_grid.resolution[1] / 2) if cell_number == 'mid' else cell_number

        y = regular_grid.extent[2] + regular_grid.dy * cell_number
        p1 = [regular_grid.extent[0], y]
        p2 = [regular_grid.extent[1], y]

    elif direction == 'x':
        cell_number = int(regular_grid.resolution[0] / 2) if cell_number == 'mid' else cell_number

        x = regular_grid.extent[0] + regular_grid.dx * cell_number
        p1 = [x, regular_grid.extent[2]]
        p2 = [x, regular_grid.extent[3]]

    else:
        raise NotImplementedError
    return p1, p2


def get_geo_model_cmap(elements_colors: Sequence[str], reverse: bool = True) -> mcolors.ListedColormap:
    if reverse:
        return mcolors.ListedColormap(elements_colors).reversed()
    else:
        return mcolors.ListedColormap(elements_colors)


def get_geo_model_norm(number_elements: int) -> mcolors.Normalize:
    # return mcolors.Normalize(vmin=0.5, vmax=number_elements + 0.5)
    return mcolors.Normalize(vmin=0, vmax=number_elements + 0.01)
