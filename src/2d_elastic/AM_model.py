import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import segyio
import argparse
import numpy as np
from tqdm import tqdm
from devito import configuration, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag, solve

from examples.seismic import AcquisitionGeometry, plot_velocity, demo_model
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from devito import configuration, VectorTimeFunction, TensorTimeFunction, Eq, Operator

from scratch.devito_shared import run_on_gpu, CreateSeismicModelElastic, elastic_solver

configuration['ignore-unknowns'] = True

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
 
def main():
    parser = argparse.ArgumentParser(description="Preset model acoustic modeling")
    parser.add_argument("sx", type=float, help="Source X position")
    parser.add_argument("sz", type=float, help="Source Z position")
    parser.add_argument("f0", type=float, help="Source frequency")
    parser.add_argument("rec", help="path to file with rec coordinates")
    parser.add_argument("-gpu", action="store_true", help = "Run modeling on GPU")
    parser.add_argument("-r", "--output", help="Output file", required=True)
    args = parser.parse_args()

    if args.gpu:
        run_on_gpu()

    # геометрия
    model = define_model()
    t0=0.
    tn=2000.
    dt = model.critical_dt/2
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    f0=args.f0

    src = RickerSource(name='src', grid = model.grid, f0=f0, time_range=time_range, npoint=1)
    src.coordinates.data[:, 0] = args.sx
    src.coordinates.data[:, 1] = args.sz
    

    rec_coordinates = np.loadtxt(args.rec)

    op, rec = elastic_solver(model, time_range, f0, src, rec_coordinates)
    op.apply(dt=dt)

    dt_r = 0.5
    rec = rec.resample(dt=dt_r)

    if args.output:
        segyio.tools.from_array2D(args.output +'/2d_AM_SRC-'+str(int(src.coordinates.data[0]))+'.sgy', rec.data.T, dt=dt_r*10**3)
        with segyio.open(args.output+'/2d_AM_SRC-'+str(int(src.coordinates.data[0]))+'.sgy', 'r+') as f:
            for j in range(len(f.header)):
                f.header[j] = {segyio.TraceField.SourceGroupScalar : -100,
                                segyio.TraceField.SourceX : int(src.coordinates.data[0]*100),
                                segyio.TraceField.GroupX : np.array(rec.coordinates.data[j,0], dtype = int)
                                }
if __name__ == "__main__":
    main()