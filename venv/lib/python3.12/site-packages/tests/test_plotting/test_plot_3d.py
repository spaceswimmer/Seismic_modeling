import numpy as np
import pytest

import gempy_viewer as gpv
import gempy as gp
from gempy.core.data import GeoModel
from gempy_viewer.core.scalar_data_type import TopographyDataType
from tests.conftest import _one_fault_model_generator


class TestPlot3dInputData:
    def test_plot_3d_input_data(self, one_fault_model_no_interp):
        gpv.plot_3d(one_fault_model_no_interp, image=True)


class TestPlot3DSolutions:
    def test_plot_3d_solutions_default(self, one_fault_model_topo_solution):
        gpv.plot_3d(one_fault_model_topo_solution, image=True)
    
    def test_plot_3d_solutions(self, one_fault_model_topo_solution):
        gpv.plot_3d(
            model=one_fault_model_topo_solution,
            show_scalar=False,
            show_lith=True,
            show_data=True,
            show_boundaries=True,
            image=True
        )
    
    def test_plot_3d_scalar_field(self, one_fault_model_topo_solution):
        gpv.plot_3d(
            model=one_fault_model_topo_solution,
            active_scalar_field="sf_1",
            show_scalar=True,
            show_lith=False,
            image=True
        )
    
    def test_plot_3d_solutions_topography(self, one_fault_model_topo_solution):
        gpv.plot_3d(
            model=one_fault_model_topo_solution,
            show_topography=True,
            topography_scalar_type=TopographyDataType.TOPOGRAPHY,
            image=True
        )
    
    def test_plot_3d_solutions_topography_geological_map(self, one_fault_model_topo_solution):
        gpv.plot_3d(
            model=one_fault_model_topo_solution,
            show_lith=True,
            show_topography=True,
            topography_scalar_type=TopographyDataType.GEOMAP,
            image=True
        )


class TestPlot2DSolutionsOctrees:
    @pytest.fixture(scope='class')
    def one_fault_model_topo_solution_octrees(self) -> GeoModel:
        one_fault_model = _one_fault_model_generator()
        one_fault_model.grid.regular_grid.resolution = np.array([2, 4, 2])

        one_fault_model.interpolation_options.number_octree_levels = 5

        # TODO: Test octree regular grid with everything else combined
        gp.set_section_grid(
            grid=one_fault_model.grid,
            section_dict={'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                          'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
        )

        gp.set_topography_from_random(
            grid=one_fault_model.grid,
            fractal_dimension=1.2,
            d_z=np.array([600, 2000]),
            topography_resolution=np.array([60, 60])
        )

        gp.compute_model(one_fault_model)
        return one_fault_model


    def test_plot_3d_solutions_default(self, one_fault_model_topo_solution_octrees):
        gpv.plot_3d(one_fault_model_topo_solution_octrees, image=True)
