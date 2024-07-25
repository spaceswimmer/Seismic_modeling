from typing import Union, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from gempy.core.data import GeoModel
from gempy.core.data.grid_modules import RegularGrid
from gempy_viewer.API._plot_2d_sections_api import plot_sections
from gempy_viewer.core.data_to_show import DataToShow
from gempy_viewer.core.section_data_2d import SectionData2D
from gempy_viewer.modules.plot_2d.multi_axis_manager import sections_iterator, orthogonal_sections_iterator
from gempy_viewer.modules.plot_2d.visualization_2d import Plot2D


# noinspection t
def plot_2d(model: GeoModel,
            n_axis=None,
            section_names: list = None,
            cell_number: Optional[Union[int | list[int] | str | list[str]]] = None,
            direction: Optional[Union[str | list[str]]] = 'y',
            series_n: Union[int, List[int]] = 0,
            legend: bool = True,
            ve=1,
            block=None,
            override_regular_grid=None,
            kwargs_topography=None,
            kwargs_lithology=None,
            kwargs_scalar_field=None,
            **kwargs) -> Plot2D:
    """Plot 2-D sections of the geomodel.

    This function plots cross-sections either based on custom section traces or cell numbers
    in the xyz directions. Options are provided to plot lithology blocks, scalar fields, or
    rendered surface lines. Input data and topography can be included.

    Args:
        model (GeoModel): Geomodel object with solutions.
        n_axis (Optional[int]): Subplot axis for multiple sections.
        section_names (Optional[List[str]]): Names of predefined custom section traces.
        cell_number (Optional[Union[int, List[int], str, List[str]]]): Position of the array to plot.
        direction (Optional[Union[str, List[str]]]): Cartesian direction to be plotted (xyz).
        series_n (Union[int, List[int]]): Number of the scalar field.
        legend (bool): If True, plot legend. Defaults to True.
        ve (float): Vertical exaggeration. Defaults to 1.
        block (Optional[np.ndarray]): Deprecated. Use regular grid instead.
        override_regular_grid (Optional[np.ndarray]): Numpy array of the size of model.grid.regular_grid.
            If provided, the regular grid will be overridden by this array.
        kwargs_topography (Optional[dict]): Additional keyword arguments for topography.
            * fill_contour: Fill contour flag.
            * hillshade (bool): Calculate and add hillshading using elevation data.
            * azdeg (float): Azimuth of sun for hillshade.
            - altdeg (float): Altitude in degrees of sun for hillshade.
        kwargs_lithology (Optional[dict]): Additional keyword arguments for lithology.
        kwargs_scalar_field (Optional[dict]): Additional keyword arguments for scalar field.

    Keyword Args:
        show_block (bool): If True and the model has been computed, plot cross section of the final model.
        show_values (bool): If True and the model has been computed, plot cross section of the value.
        show (bool): Call matplotlib show. Defaults to True.
        show_data (bool): Show original input data. Defaults to True.
        show_results (bool): If False, override show lithology, scalar field, and values. Defaults to True.
        show_lith (bool): Show lithological block volumes. Defaults to True.
        show_scalar (bool): Show scalar field isolines. Defaults to False.
        show_boundaries (bool): Show surface boundaries as lines. Defaults to True.
        show_topography (bool): Show topography on plot. Defaults to False.
        show_section_traces (bool): Show section traces. Defaults to True.

    Returns:
        gempy.plot.visualization_2d.Plot2D: Plot2D object.
    """

    if kwargs_lithology is None:
        kwargs_lithology = dict()
    if kwargs_topography is None:
        kwargs_topography = dict()
    if kwargs_scalar_field is None:
        kwargs_scalar_field = dict()

    if section_names is None and cell_number is None and direction is not None:
        cell_number = ['mid']

    show = kwargs.get('show', True)

    if block is not None:
        import warnings
        regular_grid = block
        warnings.warn('block is going to be deprecated. Use regular grid instead',
                      DeprecationWarning)

    section_names = [] if section_names is None else section_names
    section_names = np.atleast_1d(section_names)
    if cell_number is None:
        cell_number = []
    elif cell_number == 'mid':
        cell_number = ['mid']
    direction = [] if direction is None else direction

    if type(cell_number) != list:
        cell_number = [cell_number]

    if type(direction) != list:
        direction = [direction] * len(cell_number)

    if n_axis is None:
        n_axis = len(section_names) + len(cell_number)

    if type(series_n) is int:
        series_n = [series_n] * n_axis

    # * Grab from kwargs all the show arguments and create the proper class. This is for backwards compatibility
    can_show_results = model.solutions is not None
    data_to_show = DataToShow(
        n_axis=n_axis,
        show_data=kwargs.get('show_data', True),
        _show_results=kwargs.get('show_results', can_show_results),
        show_surfaces=kwargs.get('show_surfaces', True),
        show_lith=kwargs.get('show_lith', True),
        show_scalar=kwargs.get('show_scalar', False),
        show_boundaries=kwargs.get('show_boundaries', True),
        show_topography=kwargs.get('show_topography', False),
        show_section_traces=kwargs.get('show_section_traces', True),
        show_values=kwargs.get('show_values', False),
        show_block=kwargs.get('show_block', False)
    )

    # is 10 and 10 because in the ax pos is the second digit
    n_columns_ = 1 if len(section_names) + len(cell_number) < 2 else 2
    n_columns = n_columns_ * 10  # This is for the axis location syntax
    n_rows = (len(section_names) + len(cell_number)) / n_columns_

    n_columns_ = np.max([n_columns_, 1])
    n_rows = np.max([n_rows, 1])

    p = Plot2D()
    p.create_figure(cols=n_columns_, rows=n_rows, **kwargs)  # * This creates fig and axes
    section_data_list: list[SectionData2D] = sections_iterator(
        plot_2d=p,
        gempy_model=model,
        sections_names=section_names,
        n_axis=n_axis,
        n_columns=n_columns,
        ve=ve,
        projection_distance=kwargs.get('projection_distance', 0.2 * model.input_transform.isometric_scale)
    )

    orthogonal_section_data_list: list[SectionData2D] = orthogonal_sections_iterator(
        initial_axis=len(section_data_list),
        plot_2d=p,
        gempy_model=model,
        direction=direction,
        cell_number=cell_number,
        n_axis=n_axis,
        n_columns=n_columns,
        ve=ve,
        projection_distance=kwargs.get('projection_distance', 0.2 * model.input_transform.isometric_scale)
    )

    section_data_list.extend(orthogonal_section_data_list)
    p.section_data_list = section_data_list

    plot_sections(
        gempy_model=model,
        sections_data=section_data_list,
        data_to_show=data_to_show,
        ve=ve,
        series_n=series_n,
        override_regular_grid=override_regular_grid,
        legend=legend,
        kwargs_topography=kwargs_topography,
        kwargs_scalar_field=kwargs_scalar_field,
        kwargs_lithology=kwargs_lithology
    )
    if show is True:
        p.fig.show()

    return p


def plot_section_traces(model: GeoModel, section_names: list[str] = None):
    """
    Plot section traces of section grid in 2-D topview (xy).
    """
    from gempy_viewer.modules.plot_2d.drawer_traces_2d import plot_section_traces as pst
    plot: Plot2D = plot_2d(
        model=model,
        n_axis=1,
        direction=['z'],
        cell_number=[-1],
        show_data=False,
        show_boundaries=False,
        show_lith=False,
        show=False
    )
    pst(
        gempy_model=model,
        ax=plot.axes[0],
        section_names=section_names
    )
    
    plot.fig.show()
    return pst


def plot_topology(regular_grid: RegularGrid, edges, centroids, direction="y", ax=None, scale=True,
                  label_kwargs=None, edge_kwargs=None):
    """Plot the topology adjacency graph in 2-D.

        Args:
            geo_model ([type]): GemPy geomodel instance.
            edges (Set[Tuple[int, int]]): Set of topology edges.
            centroids (Dict[int, Array[int, 3]]): Dictionary of topology id's and
                their centroids.
            direction (Union["x", "y", "z", optional): Section direction.
                Defaults to "y".
            label_kwargs (dict, optional): Keyword arguments for topology labels.
                Defaults to None.
            edge_kwargs (dict, optional): Keyword arguments for topology edges.
                Defaults to None.

        """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        
    res = regular_grid.resolution
    if direction == "y":
        c1, c2 = (0, 2)
        e1 = regular_grid.extent[1] - regular_grid.extent[0]
        e2 = regular_grid.extent[5] - regular_grid.extent[4]
        d1 = regular_grid.extent[0]
        d2 = regular_grid.extent[4]
        # ? (miguel, Oct 23) When is this condition used?
        if len(list(centroids.items())[0][1]) == 2:
            c1, c2 = (0, 1)
        r1 = res[0]
        r2 = res[2]
    elif direction == "x":
        c1, c2 = (1, 2)
        e1 = regular_grid.extent[3] - regular_grid.extent[2]
        e2 = regular_grid.extent[5] - regular_grid.extent[4]
        d1 = regular_grid.extent[2]
        d2 = regular_grid.extent[4]
        r1 = res[1]
        r2 = res[2]
    elif direction == "z":
        c1, c2 = (0, 1)
        e1 = regular_grid.extent[1] - regular_grid.extent[0]
        e2 = regular_grid.extent[3] - regular_grid.extent[2]
        d1 = regular_grid.extent[0]
        d2 = regular_grid.extent[2]
        r1 = res[0]
        r2 = res[1]

    tkw = {
        "color"              : "white",
        "fontsize"           : 13,
        "ha"                 : "center",
        "va"                 : "center",
        "weight"             : "ultralight",
        "family"             : "monospace",
        "verticalalignment"  : "center",
        "horizontalalignment": "center",
        "bbox"               : dict(boxstyle='round', facecolor='black', alpha=1),
    }
    if label_kwargs is not None:
        tkw.update(label_kwargs)

    lkw = {
        "linewidth": 1,
        "color"    : "black"
    }
    if edge_kwargs is not None:
        lkw.update(edge_kwargs)

    for a, b in edges:
        # plot edges
        x = np.array([centroids[a][c1], centroids[b][c1]])
        y = np.array([centroids[a][c2], centroids[b][c2]])
        if scale:
            x = x * e1 / r1 + d1
            y = y * e2 / r2 + d2
        ax.plot(x, y, **lkw)

    for node in np.unique(list(edges)):
        x = centroids[node][c1]
        y = centroids[node][c2]
        if scale:
            x = x * e1 / r1 + d1
            y = y * e2 / r2 + d2
        ax.text(x, y, str(node), **tkw)
    
    plt.show()
