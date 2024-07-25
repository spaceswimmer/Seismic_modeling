from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FixedFormatter

from gempy_viewer.core.slicer_data import SlicerData
from gempy_viewer.core.section_data_2d import SectionData2D, SectionType
from gempy.core.data import Grid, GeoModel
from gempy.core.data.grid_modules import Sections
from gempy_viewer.modules.plot_2d.visualization_2d import Plot2D
from gempy_viewer.modules.plot_2d.plot_2d_utils import make_section_xylabels, slice_cross_section
from gempy_viewer.modules.plot_2d.drawer_input_2d import _projection_params_section, _projection_params_regular_grid


def sections_iterator(plot_2d: Plot2D, gempy_model: GeoModel, sections_names: list[str],
                      n_axis: int, n_columns: int, ve: float, projection_distance: Optional[float] = None,
                      e:int =0) -> list[SectionData2D]:
    section_data_list: list[SectionData2D] = []
    for e, sec_name in enumerate(sections_names):
        # region matplotlib configuration
        # Check if a plot that fills all pixels is plotted
        _is_filled = False
        assert e < 10, 'Reached maximum of axes'

        ax_pos = (round(n_axis / 2 + 0.1)) * 100 + n_columns + e + 1
        temp_ax = create_ax_section(
            plot_2d=plot_2d,
            gempy_grid=gempy_model.grid,
            section_name=sec_name,
            ax_pos=ax_pos,
            ve=ve
        )
        # endregion 
        slicer_data: SlicerData = _projection_params_section(
            grid=gempy_model.grid,
            orientations=gempy_model.orientations_copy.df.copy(),
            points=gempy_model.surface_points_copy.df.copy(),
            projection_distance=projection_distance,
            section_name=sec_name
        )

        section_data_2d: SectionData2D = SectionData2D(
            section_type=SectionType.SECTION,
            slicer_data=slicer_data,
            ax=temp_ax,
            section_name=sec_name,
            cell_number=None,
            direction=None
        )

        section_data_list.append(section_data_2d)

    return section_data_list


def orthogonal_sections_iterator(initial_axis: int, plot_2d: Plot2D, gempy_model: GeoModel, direction: list[str], cell_number: list[int],
                                 n_axis: int, n_columns: int, ve: float, projection_distance: Optional[float] = None) -> list[SectionData2D]:
    section_data_list: list[SectionData2D] = []
    for e in range(len(cell_number)):
        # region matplotlib configuration
        # Check if a plot that fills all pixels is plotted
        _is_filled = False
        assert e < 10, 'Reached maximum of axes'

        ax_pos = (round(n_axis / 2 + 0.1)) * 100 + n_columns + e + 1 + initial_axis
        temp_ax = create_axes_orthogonal(
            plot_2d=plot_2d,
            gempy_grid=gempy_model.grid,
            cell_number=cell_number[e],
            direction=direction[e],
            ax_pos=ax_pos,
            ve=ve
        )
        
        # endregion 
        slicer_data: SlicerData = _projection_params_regular_grid(
            regular_grid=gempy_model.grid.regular_grid,
            orientations=gempy_model.orientations_copy.df.copy(),
            points=gempy_model.surface_points_copy.df.copy(),
            projection_distance=projection_distance,
            cell_number=cell_number[e],
            direction=direction[e]
        )

        section_data_2d: SectionData2D = SectionData2D(
            section_type=SectionType.ORTHOGONAL,
            slicer_data=slicer_data,
            ax=temp_ax,
            section_name=None,
            cell_number=cell_number[e],
            direction=direction[e]
        )

        section_data_list.append(section_data_2d)

    return section_data_list


def create_ax_section(plot_2d: Plot2D, gempy_grid: Grid, section_name, ax=None, ax_pos=111, ve=1.):
    if ax is None:
        ax = plot_2d.fig.add_subplot(ax_pos)

    if section_name is not None:
        if section_name == 'topography':
            _setup_topography_section(ax)
            extent_val = gempy_grid.topography.extent
        else:
            # Check if section is in the grid and if not raise an error
            if section_name not in gempy_grid.sections.df.index:
                raise ValueError('Section name not in grid')
            dist = gempy_grid.sections.df.loc[section_name, 'dist']
            _setup_section(
                ax=ax,
                sections=gempy_grid.sections,
                section_name=section_name,
                dist=dist
            )
            extent_val = [0, dist, gempy_grid.regular_grid.extent[4], gempy_grid.regular_grid.extent[5]]

    if extent_val[3] < extent_val[2]:  # correct vertical orientation of plot
        ax.invert_yaxis()
    plot_2d._aspect = (extent_val[3] - extent_val[2]) / (extent_val[1] - extent_val[0]) / ve
    ax.set_xlim(extent_val[0], extent_val[1])
    ax.set_ylim(extent_val[2], extent_val[3])

    ax.set_aspect('equal')
    ax.set_aspect(ve)
    
    # Adding some properties to the axes to make easier to plot
    ax.section_name = section_name
    ax.tick_params(axis='x', labelrotation=30)
    plot_2d.axes = np.append(plot_2d.axes, ax)
    plot_2d.fig.tight_layout()

    return ax


def create_axes_orthogonal(plot_2d: Plot2D, gempy_grid: Grid, cell_number,
                           direction, ax=None, ax_pos=111, ve=1.):
    if ax is None:
        ax = plot_2d.fig.add_subplot(ax_pos)

    extent_val = _setup_orthogonal_section(ax, cell_number, direction, gempy_grid)

    if extent_val[3] < extent_val[2]:  # correct vertical orientation of plot
        ax.invert_yaxis()
    ax.set_xlim(extent_val[0], extent_val[1])
    ax.set_ylim(extent_val[2], extent_val[3])

    # Adding some properties to the axes to make easier to plot
    ax.cell_number = cell_number
    ax.direction = direction
    ax.tick_params(axis='x', labelrotation=30)

   
    plot_2d.axes = np.append(plot_2d.axes, ax)
    plot_2d.fig.tight_layout()

    return ax


def _setup_orthogonal_section(ax, cell_number, direction, gempy_grid):
    _a, _b, _c, extent_val, x, y = slice_cross_section(
        regular_grid=gempy_grid.regular_grid,
        direction=direction,
        cell_number=cell_number
    )[:-2]  # * This requires the grid object
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set(title='Cell Number: ' + str(cell_number) + ' Direction: ' + str(direction))
    return extent_val


def _setup_section(ax: Axes, sections: Sections, section_name: str, dist: float):
    labels, axname = make_section_xylabels(
        sections=sections,
        section_name=section_name,
        n=len(ax.get_xticklabels()) - 2
    )
    pos_list = np.linspace(0, dist, len(labels))
    ax.xaxis.set_major_locator(FixedLocator(nbins=len(labels), locs=pos_list))
    ax.xaxis.set_major_formatter(FixedFormatter((labels)))
    ax.set(title=section_name, xlabel=axname, ylabel='Z')


def _setup_topography_section(ax):
    ax.set_title('Geological map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
