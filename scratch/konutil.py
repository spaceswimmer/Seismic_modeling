#util functions for pykonal
import pykonal
import numpy as np

def sph2xyz(rtp, x0=0, y0=0):
    x = x0 + rtp[..., 0] * np.cos(rtp[..., 2])
    y = y0 + rtp[..., 0] * np.sin(rtp[..., 2])
    xyz = np.stack([x, y, np.zeros_like(x)], axis=-1)
    return (xyz)

def xyz2sph(xyz, x0=0, y0=0):
    dx = xyz[..., 0] - x0
    dy = xyz[..., 1] - y0
    r = np.sqrt(np.square(dx) + np.square(dy))
    p = np.arctan2(dy, dx)
    rtp = np.stack([r, np.pi/2 * np.ones_like(r), p], axis=-1)
    return (rtp)

def propagate_wavefront(src, vmodel):
    # Define the far-field grid
    ff = pykonal.EikonalSolver(coord_sys="cartesian")
    ff.vv.min_coords = vmodel.min_coords
    ff.vv.node_intervals = vmodel.node_intervals
    ff.vv.npts = vmodel.npts
    ff.vv.values = vmodel.values


    # Interpolate velocity from the far-field grid to the near-field grid
    xyz = sph2xyz(nf.vv.nodes, x0=src[0], y0=src[1])
    nf.vv.values = ff.vv.resample(xyz.reshape(-1, 3)).reshape(xyz.shape[:-1])

    # Initialize the near-field narrow-band
    for ip in range(nf.tt.npts[2]):
        idx = (0, 0, ip)
        vv = nf.vv.values[idx]
        if not np.isnan(vv):
            nf.tt.values[idx] = nf.tt.node_intervals[0] / vv
            nf.unknown[idx] = False
            nf.trial.push(*idx)

    # Propagate the wavefront across the near-field grid
    nf.solve()

    # Map the traveltime values from the near-field grid onto the far-field grid
    rtp = xyz2sph(ff.tt.nodes, x0=src[0], y0=src[1])
    rtp = rtp.reshape(-1, 3)
    tt = nf.tt.resample(rtp)
    tt = tt.reshape(ff.tt.npts)

    idxs = np.nonzero(np.isnan(tt))
    tt[idxs] = np.inf

    ff.tt.values = tt

    # Initialize far-field narrow band
    for idx in np.argwhere(~np.isinf(ff.tt.values)):
        idx = tuple(idx)
        ff.unknown[idx] = False
        ff.trial.push(*idx)

    # Propagate the wavefront across the remainder of the far field.
    ff.solve()
    
    return (ff)