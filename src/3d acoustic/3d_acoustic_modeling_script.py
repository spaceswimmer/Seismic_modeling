from devito import *
import gc
from tqdm import tqdm
import segyio
import numpy as np
import matplotlib.pyplot as plt
from examples.seismic import SeismicModel
from examples.seismic.source import RickerSource, Receiver, TimeAxis


from devito import configuration
#(a) using one GPU
#(b) using multi-GPUs with MPI (requires turning the notebook into a python script)
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'pgcc'
configuration['language'] = 'openacc'

#Open vp segy
dstpath = '../../Data/Meshdurechenskaya/Model_2_3d/Vp 3D Big_tilted_trim.sgy'
src = segyio.open(dstpath, mode='r', endian='big', ignore_geometry=True)
vp = segyio.tools.collect(src.trace[:])
cdpx = []
cdpy = []
# print(cdpx.shape)
for i, th in enumerate(src.header[:]):
    cdpx.append(th[segyio.TraceField.CDP_X])
    cdpy.append(th[segyio.TraceField.CDP_Y])
src.close()
xu = np.unique(cdpx)
yu = np.unique(cdpy)
vp = np.reshape(vp, (yu.size, xu.size, vp.shape[-1])).transpose(1,0,2)[:,240:351,:]

#Open rho segy
dstpath = '../../Data/Meshdurechenskaya/Model_1_3d/Rho 3D Big_tilted_trim.sgy'
src = segyio.open(dstpath, mode='r', endian='big', ignore_geometry=True)
rho = segyio.tools.collect(src.trace[:])
cdpx = []
cdpy = []
# print(cdpx.shape)
for i, th in enumerate(src.header[:]):
    cdpx.append(th[segyio.TraceField.CDP_X])
    cdpy.append(th[segyio.TraceField.CDP_Y])
src.close()
xu = np.unique(cdpx)
yu = np.unique(cdpy)
rho = np.reshape(rho, (yu.size, xu.size, rho.shape[-1])).transpose(1,0,2)[:,240:351,:]
rho_data_nozero = np.where(rho == 0, 1, rho)

#Зададим модель
shape = rho.shape
spacing = (5., 5., 5) 
origin = (0, 0, 0)
nbl = 170
so = 8
model = SeismicModel(origin=origin,
                    spacing=spacing,
                    shape=shape,
                    vp=vp,
                    b=1/rho_data_nozero,
                    space_order=so,
                    nbl=nbl,
                    bcs='damp',
                    )

t0, tn = 0., 1500.
dt = model.critical_dt 
time_range = TimeAxis(start=t0, stop=tn, step=dt)

# The source 
nsou = 5*(1000/25+1)
src_coordinates = np.zeros((int(nsou),3))
for i in range(5):
    src_coordinates[i*41:(i+1)*41, 0] = np.arange(2000,3001,25)
    src_coordinates[i*41:(i+1)*41, 1] = 175 + i*50
    src_coordinates[i*41:(i+1)*41, 2] = 0

# The receiver
nrec = 12012

rec = Receiver(name="rec", grid=model.grid, npoint=nrec, time_range=time_range)
for i in range(12):
    rec.coordinates.data[i*1001:i*1001+1001,0] = np.linspace(0, 5000, 1001)
    rec.coordinates.data[i*1001:i*1001+1001,1] = i*50
    rec.coordinates.data[i*1001:i*1001+1001,2] = 0

# We need some initial conditions
    V_p = 1.5
    density = 1.

    ro = 1/density
    l2m = V_p*V_p*density

#solver
pbar = tqdm(src_coordinates[7:])
for i, src_coords in enumerate(pbar):
    pbar.set_description('Source %s' % str(src_coords))
    # Now we create the velocity and pressure fields
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=4)

    src = RickerSource(name='src', grid = model.grid, f0=0.025, time_range=time_range)
    src.coordinates.data[:] = src_coords

    src_term = src.inject(field=u.forward, expr=src)
    rec_term = rec.interpolate(expr=u.forward)

    rec.data.fill(0.)

    #Operator
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))
    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
    op(time=time_range.num-1, dt=model.critical_dt)

    #Segy output
    rec_res = rec.resample(dt=0.5)
    rec_res = rec_res.data.T.reshape(12,1001,rec_res.shape[0])
        
    segyio.tools.from_array3D('Results/Meshdurechenskaya/2.5d_modeling/3d_vankor_SRC-'+str(src_coords[:2])+'.sgy', rec_res, dt=0.5*10**3)
    with segyio.open('Results/Meshdurechenskaya/2.5d_modeling/3d_vankor_SRC-'+str(src_coords[:2])+'.sgy', 'r+') as f:
        for j in range(len(f.header)):
            f.header[j] = {segyio.TraceField.SourceX  : int(src_coords[0]),
                           segyio.TraceField.SourceY  : int(src_coords[1]),
                           segyio.TraceField.GroupX  : int(rec.coordinates.data[j, 0]),
                           segyio.TraceField.GroupY  : int(rec.coordinates.data[j, 1]),
                           }
    del op, stencil, pde, rec_res, u, src, src_term, rec_term
    gc.collect()