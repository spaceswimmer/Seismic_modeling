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

from scratch.util import CreateSeismicModelElastic, nn_interp_coords

configuration['ignore-unknowns'] = True
#Закоменти, чтобы выполнять на CPU
from devito import configuration
#(a) using one GPU
#(b) using multi-GPUs with MPI (requires turning the notebook into a python script)
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'pgcc'
configuration['language'] = 'openacc'

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
    parser.add_argument("h", type=float, help="Well spacing")
    parser.add_argument("-r", "--output", help="Output file", required=True)
    args = parser.parse_args()


if __name__ == "__main__":
    main()