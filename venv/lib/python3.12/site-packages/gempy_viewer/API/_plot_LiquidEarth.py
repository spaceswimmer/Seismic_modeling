from typing import Optional

import gempy
from gempy_viewer.optional_dependencies import require_liquid_earth_api


def plot_to_liquid_earth(
        geo_model: gempy.data.GeoModel, space_name: str,
        file_name: str = "gempy_model", user_token: Optional[str] = None, grab_link=True,
        make_new_space: bool = False
):
    # if user_token is None Try to grab it from the environment
    liquid_earth_api = require_liquid_earth_api()  # ! Order matters

    if user_token is None:
        import os
        user_token = os.environ.get('LIQUID_EARTH_API_TOKEN')
        if user_token is None:
            raise ValueError("No user token provided and no token found in the environment")

    if make_new_space:
        result = liquid_earth_api.upload_mesh_to_new_space(
            space_name=space_name,
            data=geo_model.solutions.raw_arrays.meshes_to_subsurface(),
            file_name=file_name,
            token=user_token,
            grab_link=grab_link
        )
    else:
        result = liquid_earth_api.upload_mesh_to_existing_space(
            space_name=space_name,
            data=geo_model.solutions.raw_arrays.meshes_to_subsurface(),
            file_name=file_name,
            token=user_token,
            grab_link=grab_link
        )
    return result
