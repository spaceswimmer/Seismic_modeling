import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse
import matplotlib.pyplot as plt
from scratch.devito_shared import run_on_gpu, CreateSeismicModelElastic, elastic_solver, SweepSource, ricker_wavelet
from examples.seismic.source import Receiver, TimeAxis, RickerSource
from examples.seismic import plot_velocity

def define_model():
    # сетка
    spacing = (0.2, 0.2)
    shape = (int(500//spacing[0]), int(500//spacing[0]))
    origin = (0, 0)
    nbl = 100
    so = 4

    # Данные
    rho_data=np.ones(shape)
    vp_data=np.ones(shape)
    vs_data=np.ones(shape)

    # Layer 1
    l1=int(2//spacing[0])
    rho_data[:,0:l1] = 1.8
    vp_data[:,0:l1] = 0.5
    vs_data[:,0:l1] = 0.13
    # Layer 2
    l2=int(4.8//spacing[0])
    rho_data[:,l1:l2] = 2
    vp_data[:,l1:l2] = 1.7
    vs_data[:,l1:l2] = 0.14
    # Layer 3
    l3=int(21.8//spacing[0])
    rho_data[:,l2:l3] = 2
    vp_data[:,l2:l3] = 1.7
    vs_data[:,l2:l3] = 0.16
    # Layer 4
    rho_data[:,l3:] = 2.4
    vp_data[:,l3:] = 1.7
    vs_data[:,l3:] = 0.55
    
    model = CreateSeismicModelElastic(origin=origin,
                        spacing=spacing,
                        shape=shape,
                        vp=vp_data,
                        vs=vs_data,
                        rho=rho_data,
                        so=so,
                        nbl=nbl,
                        bcs='damp'
                        )
    
    model.damp.data[:, :100] = model.damp.data[:, 101][:, None]
    
    return model
def model_creation():
    spacing = (1, 1)
    shape = (int(110//spacing[0]), int(114//spacing[1]))
    origin = (0, 0)
    nbl = 50
    so = 4

    vp = np.ones(shape)
    vs = np.ones(shape)
    rho = np.ones(shape)
    # vs примем просто 0.5 от vp

    #layer1
    l1 = int(35//spacing[1])
    vp[:,0:l1] = 1.45
    vs[:,0:l1] = 1.45/2**0.5
    rho[:,0:l1] = 1.6
    #layer2
    l2 = int(70//spacing[1])
    vp[:,l1:l2] = 1.55
    vs[:,l1:l2] = 1.55/2**0.5
    rho[:,l1:l2] = 1.6
    #layer3
    vp[:,l2:] = 1.87
    vs[:,l2:] = 1.87/2**0.5
    rho[:,l2:] = 1.6
    
    model = CreateSeismicModelElastic(
                    origin=origin,
                    spacing=spacing,
                    shape=shape,
                    vp=vp,
                    vs=vs,
                    rho=rho,
                    so=so,
                    nbl=nbl,
                    bcs='damp'
                    )
    print(model.shape, model.spacing)
    
    # model.damp.data[:, :100] = model.damp.data[:, 101][:, None]
    
    return model

def plot_seis_data(rec):
    plt.imshow(rec.data[:], aspect='auto', vmin=-.5*1e-3, vmax=.5*1e-3)
    plt.show

def main():
    parser = argparse.ArgumentParser(description="vibroseis elastic modeling")
    parser.add_argument("sx", type=float, help="Source X position")
    parser.add_argument("sz", type=float, help="Source Z position")
    parser.add_argument("rec", help="path to file with rec coordinates")
    parser.add_argument("-gpu", action="store_true", help = "Run modeling on GPU")
    parser.add_argument("-r", "--output", help="Output file", required=False)
    args = parser.parse_args()

    if args.gpu:
        run_on_gpu()

    model = model_creation()
    
    # геометрия
    t0=0.
    tn=20000.
    dt = model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    f0=0.06

    nsrc = 1
    src_coordinates = np.empty((nsrc, 2))
    src_coordinates[:, 0] = args.sx
    src_coordinates[:, 1] = args.sz
    src = RickerSource(name='src', grid = model.grid, f0=f0, time_range=time_range, npoint=1)
    # src = SweepSource(name='src', grid = model.grid, f0=f0, sweep_f=[0.02,0.05], time_range=time_range, npoint=1)
    # src.data[:,0] = np.convolve(src.data[:,0], ricker_wavelet(f0, dt))[:src.data.shape[0]]
    src.coordinates.data[:] = src_coordinates

    rec_coordinates = np.loadtxt(args.rec)


    op, rec = elastic_solver(model, time_range, f0, src, rec_coordinates)
    summary = op.apply(dt=dt)
    dt_r = 0.5
    rec = rec.resample(dt=dt_r)
    src = src.resample(dt=dt_r)

    np.savez_compressed(args.r, data=rec.data, dt = dt_r, sweep = src.data[:,0])


if __name__ == "__main__":
    main()