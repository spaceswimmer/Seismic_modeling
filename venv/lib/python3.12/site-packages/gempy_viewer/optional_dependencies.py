def require_liquid_earth_api():
    try:
        import liquid_earth_api
    except ImportError:
        raise ImportError("The liquid_earth_api package is required to run this function.")
    return liquid_earth_api

def require_gempy_plugins():
    try:
        import gempy.plugins
    except ImportError:
        raise ImportError("The gempy.plugins package is required to run this function.")
    return gempy.plugins


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("The pyvista package is required to run this function.")
    return pv

def require_scipy():
    try:
        import scipy
    except ImportError:
        raise ImportError("The scipy package is required to run this function.")
    return scipy

def require_skimage():
    try:
        import skimage
    except ImportError:
        raise ImportError("The skimage package is required to run this function.")
    return skimage
    

def require_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("The pandas package is required to run this function.")
    return pd