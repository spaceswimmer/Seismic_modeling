import numpy as np
from devito import div, grad, diag, solve
from devito import configuration, VectorTimeFunction, TensorTimeFunction, Eq, Operator
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from examples.seismic import SeismicModel, plot_velocity, demo_model, source
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

def run_on_gpu():
    configuration['platform'] = 'nvidiaX'
    configuration['compiler'] = 'pgcc'
    configuration['language'] = 'openacc'

def CreateSeismicModelElastic(vp, vs, rho, origin, spacing, shape, so, nbl, bcs='damp'):

    model = demo_model(preset='layers-elastic', nlayers=3, shape=shape, spacing=spacing,
                   space_order=so, origin = origin, nbl = nbl)

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

def CreateSeismicModelAcoustic(vp, rho, origin, spacing, shape, so, nbl, bcs='damp'):

    model = demo_model(preset='layers-isotropic', nlayers=3, shape=shape, spacing=spacing,
                   space_order=so, origin = origin, nbl = nbl, density = True)

    rho_data_nozero = np.where(rho == 0, 1, rho)
    
    model.update('vp', vp)
    model.update('b', 1/rho_data_nozero)

    model._initialize_physics(vp=vp,
                              b=1/rho_data_nozero,
                              space_order=so
                             )
    
    return model

def elastic_solver(model, time_range, f0, src_coords, rec_coords):
    src = RickerSource(name='src', grid = model.grid, f0=f0, time_range=time_range, npoint=1)
    src.coordinates.data[:] = src_coords
    
    v = VectorTimeFunction(name='v', grid=model.grid,
                           space_order=4, time_order=2)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             space_order=4, time_order=2)

    lam, mu, b = model.lam, model.mu, model.b

    eq_v = v.dt - b * div(tau)
    # Stress
    e = (grad(v.forward) + grad(v.forward).transpose(inner=False))
    eq_tau = tau.dt - lam * diag(div(v.forward)) - mu * e

    u_v = Eq(v.forward, model.damp * solve(eq_v, v.forward))
    u_t = Eq(tau.forward, model.damp * solve(eq_tau, tau.forward))

    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet

    rec = Receiver(name="rec", grid=model.grid, npoint=rec_coords.shape[0], time_range=time_range)
    rec.coordinates.data[:,0] = rec_coords[:,0]
    rec.coordinates.data[:,1] = rec_coords[:,1]

    
    rec_term = rec.interpolate(expr=v[1])
    # The source injection term
    # src_expr = src.inject(tau.forward[1, 1], expr=src * s)
    src_expr = src.inject(v.forward[1], expr=src * s)

    srcrec = src_expr + rec_term
    op = Operator([u_v] + [u_t] + srcrec, subs=model.spacing_map, name="ForwardElastic",
                  # opt=('noop', {'openmp': True}),
                 )
    return op, rec, v, tau