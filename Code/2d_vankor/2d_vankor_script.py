import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))

import glob
import gc
import segyio
import segysak
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from devito import configuration, VectorTimeFunction, TensorTimeFunction
from examples.seismic import AcquisitionGeometry, plot_velocity
from examples.seismic.elastic import ElasticWaveSolver
from scratch.util import CreateSeismicModelElastic, nn_interp_coords

configuration['ignore-unknowns'] = True
#Закоменти, чтобы выполнять на CPU
from devito import configuration
#(a) using one GPU
#(b) using multi-GPUs with MPI (requires turning the notebook into a python script)
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'pgcc'
configuration['language'] = 'openacc'

sc_path = 'Data/2D_Scenarios'
scenarios = glob.glob(sc_path+'/sc_1*')

# df_ins = pd.read_csv(sc_path+'/instruments.txt', sep='\t')
# df_ins['Z'] += 2
# df_ins['X'] = np.linspace(0,7950, 319)

constraints = {"Vp": 1800, "Vs": 750, "Rho" : 1500}
for i, scenario in enumerate(scenarios[:1]): #single check
# for i, scenario in enumerate(scenarios): #whole
    readsgy = lambda x : xr.open_dataset(x,
                                         dim_byte_fields={"cdp" : 1},
                                         extra_byte_fields={'cdp_x':181, 'cdp_y':185}
                                        )
    el_pars = {file.split('/')[-1].split(' ')[0] : readsgy(file) for file in glob.glob(scenario+'/*.sgy')}
    print(el_pars.keys())
    # plot_hist_pars(el_pars, ignore_zero=True) # гистограммы параметров перед корректировкой
    # for k, v in el_pars.items():
    #     el_pars[k] = el_pars[k].where(((el_pars[k] > constraints[k]) | (el_pars[k].samples>100) | (el_pars[k] == 0)), constraints[k])
    # plot_hist_pars(el_pars, ignore_zero=True)  # гистограммы параметров после корректировки
    print('read the sgy file')
    cdp_x = el_pars['Vp'].cdp_x.to_numpy()
    cdp_y = el_pars['Vp'].cdp_y.to_numpy()
    el_pars['x'] = np.cumsum(np.sqrt(np.diff(cdp_x, prepend=cdp_x[0])**2+np.diff(cdp_y, prepend=cdp_y[0])**2))/10
    el_pars['z'] = el_pars['Rho'].samples.data
    # привычный формат
    rho_data = (el_pars["Rho"].data/1000).to_numpy()
    vp_data = (el_pars["Vp"].data/1000).to_numpy()
    vs_data = (el_pars["Vs"].data/1000).to_numpy()

    # Верхний слой со свойствами воды
    # rho_data[rho_data==0] = 1.
    # vp_data[vp_data==0] = 1.5
    # vs_data[vs_data==0] = 0.0

    # сетка
    dim_vectors = (el_pars['x'], el_pars['z'])
    spacing = (2.5, 2.5) # z из header_rho.loc["TRACE_SAMPLE_INTERVAL", "mean"]
    origin = (0, 0)
    nbl = 40
    so = 4
    
    # инт данные
    rho_data_int = nn_interp_coords(rho_data, origin, (el_pars['x'].max(), el_pars['z'].max()), spacing, dim_vectors)
    vp_data_int = nn_interp_coords(vp_data, origin, (el_pars['x'].max(), el_pars['z'].max()), spacing, dim_vectors)
    vs_data_int = nn_interp_coords(vs_data, origin, (el_pars['x'].max(), el_pars['z'].max()), spacing, dim_vectors)
    rho_data_int = rho_data_int[:,50:]
    vp_data_int = vp_data_int[:,50:]
    vs_data_int = vs_data_int[:,50:]

    # модель
    model = CreateSeismicModelElastic(origin=origin,
                           spacing=spacing,
                           shape=rho_data_int.shape,
                           vp=vp_data_int,
                           vs=vs_data_int,
                           rho=rho_data_int,
                           so=so,
                           nbl=nbl,
                           bcs='mask',
                          )
    
    # геометрия
    t0=0.
    tn=1000.
    f0=0.025

    nsrc = 20
    src_coordinates = np.empty((nsrc, 2))
    src_coordinates[:, 0] = np.arange(3500,4500, 50)
    src_coordinates[:, 1] = 0
    
    # nsrc = 245
    # src_coordinates = np.empty((nsrc, 2))
    # src_coordinates[:, 0] = np.arange(1000,7125, 25)
    # src_coordinates[:, 1] = 0

    nrec = int(el_pars['x'].max()/5) #df_ins.shape[0]
    rec_coordinates = np.empty((nrec+1, 2))
    rec_coordinates[:,0] = np.arange(0, int(el_pars['x'].max()), 5) #df_ins['X']
    rec_coordinates[:,1] = 0 #df_ins['Z']
    print(rho_data_int.shape, rho_data.shape)
    plot_velocity(model, source=src_coordinates, receiver=rec_coordinates)
    # тензоры
    v = VectorTimeFunction(name='v', grid=model.grid, space_order=so, time_order=2)
    tau = TensorTimeFunction(name='t', grid=model.grid, space_order=so, time_order=2)

    for i, src_coords in enumerate(tqdm(src_coordinates)):
        print('Source - ', i, '; Coordinate - ', src_coords)
        geometry = AcquisitionGeometry(model, rec_coordinates, src_coords, t0, tn, f0=f0, src_type='Ricker')
        
        # солвер
        solver = ElasticWaveSolver(model, geometry, space_order=so, v=v, tau=tau)
        
        # оператор
        print('Starting operator')
        rec_p, rec_v, v, _, _ = solver.forward() # tau summary
        print('finished operator')
        # выгрузка в sgy
        dt_r = 0.5
        # inheader = segysak.segy.segy_header_scrape(scenario+'/Vs_smooth_2D')
        rec_v = rec_v.resample(dt=dt_r)
        print(np.unique(rec_v.data))
        path = 'Results/2d_vankor'
        segyio.tools.from_array2D(path +'/2d_vankor_SRC-'+str(int(src_coords[0]))+'.sgy', rec_v.data.T, dt=dt_r*10**3)
        with segyio.open(path+'/2d_vankor_SRC-'+str(int(src_coords[0]))+'.sgy', 'r+') as f:
            for j in range(len(f.header)):
                f.header[j] = {segyio.TraceField.SourceGroupScalar : -100,
                               segyio.TraceField.SourceX : int(src_coords[0]*100),
                               segyio.TraceField.GroupX : np.array(rec_v.coordinates.data[j,0], dtype = int)
                    # segyio.TraceField.CDP: j,
                    #            segyio.TraceField.CDP_X: np.array(inheader['CDP_X'][j]*10),
                    #            segyio.TraceField.CDP_Y: np.array(inheader['CDP_Y'][j]*10),
                               # segyio.TraceField.ReceiverGroupElevation: np.array(df_ins['Z'][j], dtype = int),
                               # segyio.TraceField.ElevationScalar : 1,
                              }
        gc.collect()