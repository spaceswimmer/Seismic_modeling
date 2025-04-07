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

def acoustic_solver(u, model, t0, tn, f0, src_pos=[[0,0]], rec_pos=None, dt=None):
    if dt == None:
        dt=model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    #Sources
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)
    src.coordinates.data[:, 0] = src_pos[:, 0]
    src.coordinates.data[:, 1] = src_pos[:, 1]
    src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

    #Receivers
    if rec_pos != None:
        rec = Receiver(name='rec', grid=model.grid, npoint=rec_pos.shape[0], time_range=time_range)

        # Prescribe even spacing for receivers along the x-axis
        rec.coordinates.data[:, 0] = rec_pos[:, 0]
        rec.coordinates.data[:, 1] = rec_pos[:, 1]
        src_term += rec.interpolate(expr=u.forward)
    
    #Snapshots
    nsnaps = time_range.num
    factor = round(time_range.num / nsnaps)
    time_subsampled = ConditionalDimension('t_sub', parent=model.grid.time_dim, factor=factor)
    usave = TimeFunction(name='usave', grid=model.grid, time_order=2, space_order=4,
                         save=nsnaps, time_dim=time_subsampled)

    #Operator
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))
    op = Operator([stencil] + src_term + [Eq(usave, u)],
                subs=model.spacing_map)  # operator with snapshots
    return op, rec, usave, time_range, dt

def tti_solver(p, q, b, pp, model, t0, tn, f0, src_pos, rec_pos=None, dt=None):
    # Get symbols from model
    theta = model.theta
    delta = model.delta
    epsilon = model.epsilon
    m = model.m

    # Use trigonometric functions from Devito
    costheta = cos(theta)
    sintheta = sin(theta)
    cos2theta = cos(2 * theta)
    sin2theta = sin(2 * theta)
    sin4theta = sin(4 * theta)

    # Values used to compute the time sampling
    epsilonmax = np.max(np.abs(epsilon.data[:]))
    deltamax = np.max(np.abs(delta.data[:]))
    etamax = max(epsilonmax, deltamax)
    vmax = model._max_vp
    max_cos_sin = np.amax(
        np.abs(np.cos(theta.data[:]) - np.sin(theta.data[:])))
    dvalue = min(model.spacing)

    if dt is None:
        dt = (dvalue / (np.pi * vmax)) * \
            np.sqrt(1 / (1 + etamax * (max_cos_sin) ** 2))
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Main equations
    term1_p = (
        1 + 2 * delta * (sintheta**2) * (costheta**2) +
        2 * epsilon * costheta**4
    ) * q.dx4
    term2_p = (
        1 + 2 * delta * (sintheta**2) * (costheta**2) +
        2 * epsilon * sintheta**4
    ) * q.dy4
    term3_p = (
        2
        - delta * (sin2theta) ** 2
        + 3 * epsilon * (sin2theta) ** 2
        + 2 * delta * (cos2theta) ** 2
    ) * ((q.dy2).dx2)
    term4_p = (delta * sin4theta - 4 * epsilon *
               sin2theta * costheta**2) * ((q.dy).dx3)
    term5_p = (-delta * sin4theta - 4 * epsilon * sin2theta * sintheta**2) * (
        (q.dy3).dx
    )

    stencil_p = solve(
        m * p.dt2 - (term1_p + term2_p + term3_p +
                     term4_p + term5_p), p.forward
    )
    update_p = Eq(p.forward, stencil_p)

    # Create stencil and boundary condition expressions
    x, z = model.grid.dimensions
    t = model.grid.stepping_dim

    update_q = Eq(
        pp[t + 1, x, z],
        (
            (pp[t, x + 1, z] + pp[t, x - 1, z]) * z.spacing**2
            + (pp[t, x, z + 1] + pp[t, x, z - 1]) * x.spacing**2
            - b[x, z] * x.spacing**2 * z.spacing**2
        )
        / (2 * (x.spacing**2 + z.spacing**2)),
    )

    bc = [Eq(pp[t + 1, x, 0], 0.0)]
    bc += [Eq(pp[t + 1, x, model.shape[1] + 2 * model.nbl - 1], 0.0)]
    bc += [Eq(pp[t + 1, 0, z], 0.0)]
    bc += [Eq(pp[t + 1, model.shape[0] - 1 + 2 * model.nbl, z], 0.0)]

    # set source and receivers
    src = RickerSource(
        name="src", grid=model.grid, f0=f0, npoint=1, time_range=time_range
    )
    src.coordinates.data[:, 0] = src_pos[:, 0]
    src.coordinates.data[:, 1] = src_pos[:, 1]
    # Define the source injection
    src_term = src.inject(field=p.forward, expr=src * dt**2 / m)

    # Returning operators
    # first: optime, second: oppres
    optime = Operator([update_p] + src_term,
                      opt=('advanced', {'gpu-fit': [p, q]})
                      )
    oppres = Operator([update_q] + bc,
                      opt=('advanced', {'gpu-fit': [pp, b]}), 
                      )

    return optime, oppres, time_range