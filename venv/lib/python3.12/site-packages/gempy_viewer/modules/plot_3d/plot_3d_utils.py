from typing import Union

import numpy as np
import pandas as pd

from gempy_viewer.modules.plot_3d.vista import GemPyToVista


# ? Is this used?
def select_surfaces_data(data_df: pd.DataFrame, surfaces: Union[str, list[str]] = 'all') -> pd.DataFrame:
    """Select the surfaces that has to be plot.

    Args:
        data_df (pd.core.frame.DataFrame): GemPy data df that contains
            surface property. E.g Surfaces, SurfacePoints or Orientations.
        surfaces: If 'all' select all the active data. If a list of surface
            names or a surface name is passed, plot only those.
    """
    if surfaces == 'all':
        geometric_data = data_df
    else:
        geometric_data = pd.concat([data_df.groupby('surface').get_group(group) for group in surfaces])
    return geometric_data


def set_scalar_bar(gempy_vista: GemPyToVista, elements_names: list[str], surfaces_ids: np.ndarray):
    import pyvista as pv
    
    # Get mapper actor 
    if gempy_vista.surface_points_actor is not None:
        mapper_actor: pv.Actor = gempy_vista.surface_points_actor
    elif gempy_vista.regular_grid_actor is not None:
        mapper_actor = gempy_vista.regular_grid_actor
    else:
        return None  # * Not a good mapper for the scalar bar
    
    annotations = {}
    for e, name in enumerate(elements_names):
        annotations[e] = name

    mapper_actor.mapper.lookup_table.annotations = annotations
    
    sargs = gempy_vista.scalar_bar_arguments
    sargs["mapper"] = mapper_actor.mapper
    
    gempy_vista.p.add_scalar_bar(**sargs)
    gempy_vista.p.update_scalar_bar_range((surfaces_ids.min(), surfaces_ids.max())) # * This has to be here to now screw the colors with the volumes
