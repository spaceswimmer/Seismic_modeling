from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from gempy.core.data import GeoModel
from gempy_viewer.core.data_to_show import DataToShow
from gempy_viewer.core.scalar_data_type import ScalarDataType, TopographyDataType
from gempy_viewer.modules.plot_2d.plot_2d_utils import get_geo_model_cmap
from gempy_viewer.modules.plot_3d.vista import GemPyToVista

try:
    import pyvista as pv
    from gempy_viewer.modules.plot_3d._vista import Vista as Vista

    PYVISTA_IMPORT = True
except ImportError:
    PYVISTA_IMPORT = False

try:
    import mplstereonet

    mplstereonet_import = True
except ImportError:
    mplstereonet_import = False


# noinspection t
def plot_3d(
        model: GeoModel,
        plotter_type: str = 'basic',
        active_scalar_field: Optional[str] = None,
        ve: Optional[float] = None,
        topography_scalar_type: TopographyDataType = TopographyDataType.GEOMAP,
        kwargs_pyvista_bounds: Optional[dict] = None,
        kwargs_plot_structured_grid: Optional[dict] = None,
        kwargs_plot_topography: Optional[dict] = None,
        kwargs_plot_data: Optional[dict] = None,
        kwargs_plotter: Optional[dict] = None,
        kwargs_plot_surfaces: Optional[dict] = None,
        image: bool = False,
        show: bool = True,
        transformed_data: bool = False,
        **kwargs
) -> GemPyToVista:
    """Plot 3-D geomodel.

    Args:
        model (GeoModel): Geomodel object with solutions.
        plotter_type (str): Type of plotter to use. Defaults to 'basic'.
        active_scalar_field (Optional[str]): Active scalar field for the plot.
        ve (Optional[float]): Vertical exaggeration.
        topography_scalar_type (TopographyDataType): Type of topography scalar data. Defaults to TopographyDataType.GEOMAP.
        kwargs_pyvista_bounds (Optional[dict]): Additional keyword arguments for PyVista bounds.
        kwargs_plot_structured_grid (Optional[dict]): Additional keyword arguments for plotting the structured grid.
        kwargs_plot_topography (Optional[dict]): Additional keyword arguments for plotting the topography.
        kwargs_plot_data (Optional[dict]): Additional keyword arguments for plotting data.
        kwargs_plotter (Optional[dict]): Additional keyword arguments for the plotter.
        kwargs_plot_surfaces (Optional[dict]): Additional keyword arguments for plotting surfaces.
        image (bool): If True, saves the plot as an image. Defaults to False.
        show (bool): If True, displays the plot. Defaults to True.
        transformed_data (bool): If True, uses transformed data for plotting. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        GemPyToVista: Object for 3D plotting in GemPy.
    """

    from gempy_viewer.modules.plot_3d.drawer_input_3d import plot_data
    from gempy_viewer.modules.plot_3d.drawer_structured_grid_3d import plot_structured_grid
    from gempy_viewer.modules.plot_3d.drawer_surfaces_3d import plot_surfaces
    from gempy_viewer.modules.plot_3d.drawer_topography_3d import plot_topography_3d
    from gempy_viewer.modules.plot_3d.plot_3d_utils import set_scalar_bar
    
    # * Grab from kwargs all the show arguments and create the proper class. This is for backwards compatibility
    can_show_results = model.solutions is not None  # and model.solutions.lith_block.shape[0] != 0
    data_to_show = DataToShow(
        n_axis=1,
        show_data=kwargs.get('show_data', True),
        _show_results=kwargs.get('show_results', can_show_results),
        show_surfaces=kwargs.get('show_surfaces', True),
        show_lith=kwargs.get('show_lith', True),
        show_scalar=kwargs.get('show_scalar', False),
        show_boundaries=kwargs.get('show_boundaries', True),
        show_topography=kwargs.get('show_topography', True),
        show_section_traces=kwargs.get('show_section_traces', True),
        show_values=kwargs.get('show_values', False),
        show_block=kwargs.get('show_block', False)
    )
    kwargs_plot_topography = kwargs_plot_topography or {}
    kwargs_plot_structured_grid = kwargs_plot_structured_grid or {}
    kwargs_plot_data = kwargs_plot_data or {}
    kwargs_plotter = kwargs_plotter or {}
    kwargs_plot_surfaces = kwargs_plot_surfaces or {}
    kwargs_pyvista_bounds = kwargs_pyvista_bounds or {}

    if image is True:
        show = True
        kwargs_plotter['off_screen'] = True
        plotter_type = 'basic'

    if model.solutions is None:
        data_to_show.show_results = False
        solutions_raw_arrays = None
    else:
        solutions_raw_arrays = model.solutions.raw_arrays

    extent: np.ndarray = model.grid.extent

    gempy_vista = GemPyToVista(
        extent=extent,
        plotter_type=plotter_type,
        pyvista_bounds_kwargs=kwargs_pyvista_bounds,
        **kwargs_plotter
    )

    if data_to_show.show_topography[0] is True and model.grid.topography is not None:
        plot_topography_3d(
            gempy_vista=gempy_vista,
            topography=model.grid.topography,
            solution=solutions_raw_arrays,
            topography_scalar_type=topography_scalar_type,
            elements_colors=model.structural_frame.elements_colors[::-1],
            contours=kwargs_plot_topography.get('contours', True),
            **kwargs_plot_topography
        )
        
    if data_to_show.show_boundaries[0] is True and len(solutions_raw_arrays.vertices) != 0:
        if transformed_data:
            surfaces_transform = model.input_transform
            grid_transform = model.grid.transform
        else:
            surfaces_transform = None
            grid_transform = None
            
        plot_surfaces(
            gempy_vista=gempy_vista,
            structural_elements_with_solution=model.structural_frame.structural_elements,
            input_transform=surfaces_transform,
            grid_transform=grid_transform,
            **kwargs_plot_surfaces
        )

    if data_to_show.show_data[0] is True:
        arrow_size = kwargs_plot_data.get('arrow_size', 10)
        min_axes = np.min(np.diff(extent)[[0, 2, 4]])

        plot_data(
            gempy_vista=gempy_vista,
            model=model,
            arrows_factor=arrow_size / (100 / min_axes),
            transformed_data=transformed_data,
            **kwargs_plot_data
        )

    if transformed_data:
        vtk_formated_regular_mesh = model.regular_grid_coordinates_transformed
    else:
        vtk_formated_regular_mesh = model.regular_grid_coordinates

    if data_to_show.show_lith[0] is True:
        plot_structured_grid(
            gempy_vista=gempy_vista,
            vtk_formated_regular_mesh=vtk_formated_regular_mesh,
            resolution=model.grid.regular_grid.resolution,
            scalar_data_type=ScalarDataType.LITHOLOGY,
            active_scalar_field="lith",
            solution=solutions_raw_arrays,
            cmap=get_geo_model_cmap(model.structural_frame.elements_colors_volumes),
            **kwargs_plot_structured_grid
        )

    if data_to_show.show_scalar[0] is True:
        plot_structured_grid(
            gempy_vista=gempy_vista,
            vtk_formated_regular_mesh=vtk_formated_regular_mesh,
            resolution=model.grid.regular_grid.resolution,
            scalar_data_type=ScalarDataType.SCALAR_FIELD,
            active_scalar_field=active_scalar_field,
            solution=solutions_raw_arrays,
            cmap='viridis',
            **kwargs_plot_structured_grid
        )

    if True: 
        set_scalar_bar(
            gempy_vista=gempy_vista,
            elements_names = model.structural_frame.elements_names,
            surfaces_ids=model.structural_frame.elements_ids - 1
        )

    if ve is not None:
        gempy_vista.p.set_scale(zscale=ve)

    fig_path: str = kwargs.get('fig_path', None)
    if fig_path is not None:
        gempy_vista.p.show(screenshot=fig_path)

    if image is True:
        show = _plot_in_matplotlib(gempy_vista, show)

    if show is True:
        gempy_vista.p.show()

    return gempy_vista


def _plot_in_matplotlib(gempy_vista, show):
    gempy_vista.p.show(screenshot=True)
    img = gempy_vista.p.last_image
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)
    gempy_vista.p.close()
    show = False
    return show
