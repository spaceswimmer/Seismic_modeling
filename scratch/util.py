import numpy as np
from matplotlib import pyplot as plt
from examples.seismic import SeismicModel, plot_velocity, demo_model
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

def plot_hist_pars(el_pars, ignore_zero=False):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    
    for (k, v), ax in zip(el_pars.items(), axs):
        data = v.data.to_numpy().flatten()
        if ignore_zero:
            data = data[data != 0]
        ax.set_title(k)
        ax.hist(data, bins=20)
        ax.set_ylim([0, 20])
    plt.show()

def CreateSeismicModel(vp,vs,rho, origin, spacing, shape, so, nbl, bcs='damp'):

    model = demo_model(preset='layers-elastic', nlayers=3, shape=shape, spacing=spacing,
                   space_order=so, origin = origin, nbl = nbl,)

    rho_data_nozero = np.where(rho == 0, 1, rho)
    
    model.update('vp', vp)
    model.update('vs', vs)
    model.update('b', 1/rho_data_nozero)

    model._initialize_physics(vp=vp,
                              vs=vs,
                              b=model.b.data[nbl:-nbl, nbl:-nbl],
                              space_order=so
                             )
    return model

def nn_interp_coords(data: np.ndarray, origin: tuple, domain_size : tuple, spacing : tuple, dim_vectors : tuple):
    X, Z = [np.arange(o, ds+sp, step=sp, dtype='float') for o, ds, sp in zip(origin, domain_size, spacing)]
    Z, X = np.meshgrid(Z, X)
    interp = RegularGridInterpolator(dim_vectors, data,
                                     method='nearest',
                                     # method='linear', # Linear артефачит
                                    )
    new_value = interp((X, Z))

    return new_value