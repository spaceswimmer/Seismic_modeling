"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.

@author: Miguel de la Varga
"""

import warnings
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

plt.style.use(['seaborn-v0_8-white', 'seaborn-v0_8-talk'])

warnings.filterwarnings("ignore", message="No contour levels were found")


# ! Missing a function? Check gempy_legacy
class Plot2D:
    # _color_lot: dict
    axes: list[plt.Axes]
    section_data_list: Optional[list] = list()

    def __init__(self):
        # TODO: Moving this to plotting options
        self.axes = list()

    @staticmethod
    def remove(ax):
        while len(ax.collections) != 0:
            list(map(lambda x: x.remove(), ax.collections))

    def create_figure(self, figsize=None, textsize=None, **kwargs):
        """
        Create the figure.

        Args:
            figsize:
            textsize:

        Returns:
            figure, list axes, subgrid values
        """
        cols = kwargs.get('cols', 1)
        rows = kwargs.get('rows', 1)

        figsize, self.ax_labelsize, _, self.xt_labelsize, self.linewidth, _ = _scale_fig_size(
            figsize,
            textsize,
            rows,
            cols
        )
        self.fig = plt.figure(figsize=figsize, constrained_layout=False)
        self.fig.is_legend = False

        return self.fig, self.axes  # , self.gs_0


def _scale_fig_size(figsize, textsize, rows=1, cols=1):
    """Scale figure properties according to rows and cols.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        fontsize
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    figsize : float or None
        Size of figure in inches
    ax_labelsize : int
        fontsize for axes label
    titlesize : int
        fontsize for title
    xt_labelsize : int
        fontsize for axes ticks
    linewidth : int
        linewidth
    markersize : int
        markersize
    """
    params = matplotlib.rcParams
    rc_width, rc_height = tuple(params["figure.figsize"])
    rc_ax_labelsize = params["axes.labelsize"]
    rc_titlesize = params["axes.titlesize"]
    rc_xt_labelsize = params["xtick.labelsize"]
    rc_linewidth = params["lines.linewidth"]
    rc_markersize = params["lines.markersize"]
    if isinstance(rc_ax_labelsize, str):
        rc_ax_labelsize = 15
    if isinstance(rc_titlesize, str):
        rc_titlesize = 16
    if isinstance(rc_xt_labelsize, str):
        rc_xt_labelsize = 14

    if figsize is None:
        width, height = rc_width, rc_height
        sff = 1 if (rows == cols == 1) else 1.2
        width = width * cols * sff
        height = height * rows * sff
    else:
        width, height = figsize

    if textsize is not None:
        scale_factor = textsize / rc_xt_labelsize
    elif rows == cols == 1:
        scale_factor = ((width * height) / (rc_width * rc_height)) ** 0.5
    else:
        scale_factor = 1

    ax_labelsize = rc_ax_labelsize * scale_factor
    titlesize = rc_titlesize * scale_factor
    xt_labelsize = rc_xt_labelsize * scale_factor
    linewidth = rc_linewidth * scale_factor
    markersize = rc_markersize * scale_factor

    return (width, height), ax_labelsize, titlesize, xt_labelsize, linewidth, markersize
