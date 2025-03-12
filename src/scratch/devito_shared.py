import numpy as np
import scipy
from devito import div, grad, diag, solve
from devito import configuration, VectorTimeFunction, TensorTimeFunction, Eq, Operator
from examples.seismic.source import RickerSource, Receiver, TimeAxis, WaveletSource
from examples.seismic import SeismicModel, plot_velocity, demo_model, source
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

class SweepSource(WaveletSource):
    """
    Symbolic object that encapsulates a set of sources with a
    pre-defined Ricker wavelet:

    Parameters
    ----------
    name : str
        Name for the resulting symbol.
    grid : Grid
        The computational domain.
    f0 : float
        Peak frequency for Ricker wavelet in kHz.
    time : TimeAxis
        Discretized values of time in ms.
    sweep_f : tuple or list 
        Tuple or list of frequencies

    Returns
    ----------
    Sweep source
    """
    def __init_finalize__(self, *args, **kwargs):
        self.sweep_f = kwargs.get('sweep_f', None)
        # Store other kwargs if needed
        self.kwargs = kwargs
        super().__init_finalize__(*args, **kwargs)
           
    
    def test(self):
        t0 = self.t0 or 1 / self.f0
        a = self.a or 1
        r = (np.pi * self.f0 * (self.time_values - t0))
        return a * (1-2.*r**2)*np.exp(-r**2)
    
    @property
    def wavelet(self):
        #generating ricker
        #I'm really confused on why this and np.convolve cause problems
        # length = 1/self.f0
        # t = np.arange(-length, (length) + self.dt, self.dt)
        # ricker = (1 - 2 * (np.pi ** 2) * (self.f0 ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (self.f0 ** 2) * (t ** 2))

        #generates ones
        t_max = self.time_values[-1]
        dt = self.time_values[1] - self.time_values[0]
        f_grad = (self.sweep_f[1] - self.sweep_f[0]) / t_max
        f = self.sweep_f[0] + f_grad * self.time_values
        trace = np.zeros_like(self.time_values)
        
        t_c = 0
        while t_c < t_max:
            t_ind = int(t_c / dt)
            trace[t_ind] = 1
            t_c += 1 / f[t_ind]
        return trace

def run_on_gpu():
    configuration['platform'] = 'nvidiaX'
    configuration['compiler'] = 'pgcc'
    configuration['language'] = 'openacc'

def ricker_wavelet(frequency, dt):
    length = 2/frequency
    t = np.arange(-length / 2, (length / 2) + dt, dt)
    y = (1 - 2 * (np.pi ** 2) * (frequency ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (frequency ** 2) * (t ** 2))
    return y

def gen_sweep(time_values, sweep_f):
    t_max = time_values[-1]
    dt = time_values[1] - time_values[0]
    f_grad = (sweep_f[1] - sweep_f[0]) / t_max
    f = sweep_f[0] + f_grad * time_values
    trace = np.zeros_like(time_values)
    
    t_c = 0
    while t_c < t_max:
        t_ind = int(t_c / dt)
        trace[t_ind] = 1
        t_c += 1 / f[t_ind]
    return trace

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

def elastic_solver(model, time_range, f0, src, rec_coordinates):
    
    print(src.data.shape)
    
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
    rec = Receiver(name="rec", grid=model.grid, npoint=rec_coordinates.shape[0], time_range=time_range)
    rec.coordinates.data[:,0] = rec_coordinates[:,0]
    rec.coordinates.data[:,1] = rec_coordinates[:,1]
    
    rec_term = rec.interpolate(expr=v[1])
    # The source injection term
    # src_expr = src.inject(tau.forward[1, 1], expr=src * s)
    src_expr = src.inject(v.forward[1], expr=src * s)

    srcrec = src_expr + rec_term
    op = Operator([u_v] + [u_t] + srcrec, subs=model.spacing_map, name="ForwardElastic",
                  # opt=('noop', {'openmp': True}),
                 )
    return op, rec