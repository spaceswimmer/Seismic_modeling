import numpy as np
from matplotlib import pyplot as plt
from examples.seismic import SeismicModel, plot_velocity, demo_model, source
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
    
    # model.update('vp', vp)
    # model.update('vs', vs)
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
    
def plot_rec_src(model: SeismicModel, data_type: str, src, rec: source.PointSource, xrange: tuple = None, yrange: tuple = None):
    cmap = "jet"
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    
    match data_type:
        case 'vp':
            field = model.vp.data[slices]
        case 'vs':
            field = model.vs.data[slices]
        case 'b':
            field = model.b.data[slices]
        case 'lam':
            field = model.lam.data[slices]
        case 'mu':
            field = model.mu.data[slices]
        case _:
            raise ValueError('No such data in layers-elastic model')
            
    
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=np.min(field), vmax=np.max(field),
                      extent=extent, aspect='auto')
    
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')
    
    plt.scatter(1e-3*rec.coordinates.data[:, 0], 1e-3*rec.coordinates.data[:, 1],
                        s=15, c='green', marker='D')
    plt.scatter(1e-3*src.coordinates.data[:, 0], 1e-3*src.coordinates.data[:, 1],
                        s=15, c='red', marker='D')
        
    plt.colorbar(plot)
    
    plt.xlim(xrange) # Пределы можно регулировать
    plt.ylim(yrange)
    plt.show()

def plot_seis_data(rec_coordinates, rec_data, t0: float, tn: float):
    #NBVAL_SKIP
    # Pressure (txx + tzz) data at sea surface
    extent = [rec_coordinates[0, 0], rec_coordinates[-1, 0], 1e-3*tn, t0]
    aspect = rec_coordinates[-1, 0]/(1e-3*tn)
    vminmax = np.max(np.abs(rec_data))
    gain = 2e1
    sc = vminmax/gain
    plt.figure(figsize=(15, 15))
    plt.imshow(rec_data[::5,:], vmin=-sc, vmax=sc, cmap="seismic",
               interpolation='bilinear', extent=extent, aspect=aspect)
    plt.ylabel("Time (s)", fontsize=20)
    plt.xlabel("Receiver position (m)", fontsize=20)
    plt.show()