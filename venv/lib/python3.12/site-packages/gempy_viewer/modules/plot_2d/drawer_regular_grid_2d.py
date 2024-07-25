import matplotlib.colors as mcolors
import numpy as np

from gempy.core.data import GeoModel, Grid
from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution
from gempy_viewer.core.slicer_data import SlicerData


# TODO: This name seems bad. This is plotting area basically?
def plot_regular_grid_area(ax, slicer_data: SlicerData, block: np.ndarray, resolution: iter,
                           cmap: mcolors.Colormap, norm: mcolors.Normalize, imshow_kwargs: dict = None):
    if imshow_kwargs is None:
        imshow_kwargs = dict()

    plot_grid = imshow_kwargs.pop('plot_grid', False)

    plot_block = block.reshape(resolution)
    image = plot_block[
        slicer_data.regular_grid_x_idx,
        slicer_data.regular_grid_y_idx,
        slicer_data.regular_grid_z_idx].T

    im = ax.imshow(
        image,
        origin='lower',
        zorder=-100,
        cmap=cmap,
        norm=norm,
        extent=[*ax.get_xlim(), *ax.get_ylim()],
        **imshow_kwargs
    )

    if plot_grid:
        ticks_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], image.shape[1] + 1)
        ticks_y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], image.shape[0] + 1)
        ax.set_xticks(ticks_x, minor=True)
        ax.set_yticks(ticks_y, minor=True)
        ax.grid(which="minor", linestyle='-', linewidth='3', color='grey', alpha=0.7)

    ax.figure.colorbar(im, ax=ax, shrink=0.5)

    return im


def plot_section_area(gempy_model: GeoModel, ax, section_name: str, cmap: mcolors.Colormap, norm: mcolors.Normalize):
    image = _prepare_section_image(gempy_model, section_name)
    ax.imshow(
        image,
        origin='lower',
        zorder=-100,
        cmap=cmap,
        norm=norm,
        extent=[*ax.get_xlim(), *ax.get_ylim()]
    )

    return ax


def _prepare_section_image(gempy_model: GeoModel, section_name: str):
    legacy_solutions: RawArraysSolution = gempy_model.solutions.raw_arrays
    grid: Grid = gempy_model.grid

    if section_name == 'topography':
        try:
            image = legacy_solutions.geological_map.reshape(grid.topography.values_2d[:, :, 2].shape).T
        except AttributeError:
            raise AttributeError('Geological map not computed. Activate the topography grid.')
    else:
        assert type(section_name) == str or type(
            section_name) == np.str_, 'section name must be a string of the name of the section'
        assert legacy_solutions.sections is not None, 'no sections for plotting defined'

        l0, l1 = grid.sections.get_section_args(section_name)
        shape = grid.sections.df.loc[section_name, 'resolution']
        image = legacy_solutions.sections[l0:l1].reshape(shape[0], shape[1]).T
    return image
