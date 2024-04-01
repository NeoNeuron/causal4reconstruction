#%%
import os

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16

from causal4.utils import force_refractory, save2bin
from causal4.Causality import CausalityEstimator
from causal4.figrc import fig_path, data_path
from multiprocessing import Pool

import pickle
import h5py

import warnings
warnings.filterwarnings('ignore')
#%%
session_id = 715093703
with open(f"../allen_data/preprocessed_allen_data_session_{session_id:d}.pkl", 'rb') as f:
    data_pickle = pickle.load(f)
# run causality measures
stimulus_names = data_pickle['stimulus']
stimulus_names = np.append(stimulus_names, 'drifting_gratings-compact')
stimulus_group = {
    'gratings': ['drifting_gratings', 'static_gratings'],
    'natural_movie': ['natural_movie_one', 'natural_movie_three'],
    'natural': ['natural_scenes', 'natural_movie_one', 'natural_movie_three'],
    'all': ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie_one', 'natural_movie_three'],
}
stimulus_names = np.append(stimulus_names, list(stimulus_group.keys()))
#%%
# further data selection according to refractory periods
t_ref = 5.0    # msecond
gap_width = 250

def fnaming(name):
    return f"session_{session_id:d}_{name:s}_" \
        + f"ref={int(t_ref):d}_" \
        + f"gap={int(gap_width):d}"

n_unit = data_pickle['index'].shape[0]
#%%
filtered_spks = {}
force_regen = False
hf = h5py.File('allen_data/metadata.h5','a')
data_group = {key:None for key in stimulus_group.keys()}
fig, ax = plt.subplots(14,1, figsize=(10,18), sharex=True, 
    gridspec_kw=dict(top=0.95,bottom=0.05, left=0.1, right=0.95, hspace=0.5))
[axi.spines['top'].set_visible(False) for axi in ax]
[axi.spines['right'].set_visible(False) for axi in ax]
for idx, stimulus in enumerate(np.append(data_pickle['stimulus'], 'drifting_gratings-compact')):
    if force_regen or not os.path.isfile(f"../allen_data/data/EE/N={n_unit:d}/{fnaming(stimulus):s}_spike_train.dat"):
        if 'compact' in stimulus:
            data_out = data_pickle['spike_times'][stimulus.split('-')[0]].copy()
        else:
            data_out = data_pickle['spike_times'][stimulus].copy()
        # change index
        for new_id, old_id in enumerate(data_pickle['index']):
            data_out[data_out[:,1]==old_id, 1] = new_id

        for key, val in stimulus_group.items():
            if stimulus in val:
                if data_group[key] is None:
                    data_group[key] = data_out.copy() 
                else:
                    data_group[key] = np.vstack((data_group[key],data_out))

        # align temporally with the first spike
        data_out[:,0] -= data_out[0,0]

        # change the width of big gap in raster
        if gap_width > 0:
            spk_diff = np.diff(data_out[:,0])
            if 'compact' in stimulus:
                raster_gaps, = np.nonzero((spk_diff > 0.5)*(spk_diff < 10))
                for gap in raster_gaps:
                    data_out[gap+1:,0] -= data_out[gap+1,0] - data_out[gap,0] - 1e-2
            raster_gaps, = np.nonzero(spk_diff > 100)
            for gap in raster_gaps:
                data_out[gap+1:,0] -= data_out[gap+1,0] - data_out[gap,0] - gap_width

        data_out_filtered = force_refractory(data_out, t_ref)

        T = np.ceil(data_out_filtered[-1,0])
        # caluculate mean firing rates after downsampling with forced refractory
        if 'compact' in stimulus:
            rate = np.array([
                # np.sum(data_out_filtered[:,1]==i)*1000.0/T
                np.sum(data_out_filtered[:,1]==i)*1.0/data_pickle['stimulus_present_time'][stimulus.split('-')[0]]
                for i in range(data_pickle['index'].shape[0])
            ])
        else:
            rate = np.array([
                # np.sum(data_out_filtered[:,1]==i)*1000.0/T
                np.sum(data_out_filtered[:,1]==i)*1.0/data_pickle['stimulus_present_time'][stimulus]
                for i in range(data_pickle['index'].shape[0])
            ])
        if fnaming(stimulus) in hf:
            hds = hf[fnaming(stimulus)]
            hds[:] = rate
        else:
            hds = hf.create_dataset(fnaming(stimulus), data=rate)
        hds.attrs['session']=session_id
        hds.attrs['stimulus']=str(stimulus)
        hds.attrs['t_ref']=t_ref
        hds.attrs['gap_width']=gap_width
        hds.attrs['T']=T

        save2bin(
            f"../allen_data/data/EE/N={n_unit:d}/{fnaming(stimulus):s}_spike_train.dat",
            data_out_filtered
            )
    else:
        data_out_filtered = np.fromfile(
            f"../allen_data/data/EE/N={n_unit:d}/{fnaming(stimulus):s}_spike_train.dat",
            dtype=float).reshape(-1,2)
        filtered_spks[stimulus] = data_out_filtered

    print(f"{stimulus:20s} : T = {hf[fnaming(stimulus)].attrs['T']/1e3:.3f} seconds")
    ax[idx].plot(data_out_filtered[:,0]/1000., data_out_filtered[:,1], '|')
    ax[idx].set_title(stimulus)
    if idx%2 == 0:
        ax[idx].set_ylabel('Neuronal Indices')

# process the grouped data
# ----
counter = 10
for key in data_group.keys():
    if force_regen or not os.path.isfile(f"../allen_data/data/EE/N={n_unit:d}/{fnaming(key):s}_spike_train.dat"):
        data_group[key] = data_group[key][np.argsort(data_group[key][:,0], axis=0),:]
        data_group[key][:,0] -= data_group[key][0,0]
        # change the width of big gap in raster
        if gap_width > 0:
            spk_diff = np.diff(data_group[key][:,0])
            raster_gaps, = np.nonzero(spk_diff > 100)
            for gap in raster_gaps:
                data_group[key][gap+1:,0] -= data_group[key][gap+1,0] - data_group[key][gap,0] - gap_width
        data_out_filtered = force_refractory(data_group[key], t_ref)

        T = np.ceil(data_out_filtered[-1,0])
        # caluculate mean firing rates after downsampling with forced refractory
        stimulus_present_time = np.sum([data_pickle['stimulus_present_time'][stimulus] for stimulus in stimulus_group[key]])
        rate = np.array([
            # np.sum(data_out_filtered[:,1]==i)*1000.0/T
            np.sum(data_out_filtered[:,1]==i)*1.0/stimulus_present_time
            for i in range(data_pickle['index'].shape[0])
        ])
        if fnaming(key) in hf:
            hds = hf[fnaming(key)]
            hds[:] = rate
        else:
            hds = hf.create_dataset(fnaming(key), data=rate)
        hds.attrs['session']=session_id
        hds.attrs['stimulus']=key
        hds.attrs['t_ref']=t_ref
        hds.attrs['gap_width']=gap_width
        hds.attrs['T']=T

        save2bin(
            f"../allen_data/data/EE/N={n_unit:d}/{fnaming(key):s}_spike_train.dat",
            data_out_filtered
            )
    else:
        data_out_filtered = np.fromfile(
            f"../allen_data/data/EE/N={n_unit:d}/{fnaming(key):s}_spike_train.dat",
            dtype=float).reshape(-1,2)
    print(f"{key:20s} : T = {hf[fnaming(key)].attrs['T']/1e3:.3f} seconds")
    ax[counter].plot(data_out_filtered[:,0]/1000., data_out_filtered[:,1], '|')
    ax[counter].set_title(key)
    if key == 'all':
        ax[counter].set_xlim(0)
    if counter%2 == 0:
        ax[counter].set_ylabel('Neuronal Indices')
    counter += 1

hf.close()
fig.savefig(fig_path/f"{fnaming('raster'):s}.png", dpi=200)
#%%
with h5py.File('../allen_data/metadata.h5', 'r') as hf:
    def printname(data):
        print(data)
    hf.visit(printname)
    # for key, val in hf.items():
    #     print(key, val.attrs['T'])
# %%
# setup TGIC configurations
TGIC_cfg = dict(
    order = (1,5),
    dt = 1,
    delay = 0,
    suffix = 0,
)

TGIC_prefix = f"K={TGIC_cfg['order'][0]:d}_{TGIC_cfg['order'][1]:d}" \
            + f"bin={TGIC_cfg['dt']:.2f}" \
            + f"delay={TGIC_cfg['delay']:.2f}"

def long_fnaming(name):
    return f"sfx={TGIC_cfg['suffix']:d}-{fnaming(name):s}"

#%%
hf = h5py.File('../allen_data/metadata.h5','r')
stimuli_buff = stimulus_names.copy()
# stimuli_buff = ['drifting_gratings-with-gray', 'drifting_gratings-compact',]
pm = dict(
    spk_fname = fnaming(stimulus_names[0]),
    N = n_unit,
    order = TGIC_cfg['order'],
    T = hf[fnaming(stimulus_names[0])].attrs['T'] + TGIC_cfg['suffix']*1e3,
    DT = 2e5,
    dt = TGIC_cfg['dt'],
    delay = TGIC_cfg['delay'],
    path = f'../allen_data/data/EE/N={n_unit:d}/',
)
estimator = CausalityEstimator(**pm, n_thread=60)
for stimulus in stimuli_buff:
    estimator.spk_fname=fnaming(stimulus)
    estimator.T = hf[fnaming(stimulus)].attrs['T'] + TGIC_cfg['suffix']*1e3
    estimator.fetch_data(new_run=True)

# for stimulus in stimuli_buff:
#     os.rename(
#         pm['path']+'-'.join(['TGIC2.0',TGIC_prefix+f"T={hf[fnaming(stimulus)].attrs['T'] + TGIC_cfg['suffix']*1e3:.2e}",fnaming(stimulus)+'.dat']),
#         pm['path']+'-'.join(['TGIC2.0',TGIC_prefix,long_fnaming(stimulus)+'.dat']),
#         )
hf.close()
#%%
# fig, ax = plt.subplots(5,1, figsize=(10,12), sharex=True, 
#     gridspec_kw=dict(top=0.95,bottom=0.05, left=0.1, right=0.95, hspace=0.5))
# [axi.spines['top'].set_visible(False) for axi in ax]
# [axi.spines['right'].set_visible(False) for axi in ax]
# counter = 0
# for key, value in filtered_spks.items():
#     if key in ['drifting_gratings-with-gray', 'drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie_three']:
#         ax[counter].plot(value[:15000,0]/1000., value[:15000,1], '|')
#         ax[counter].set_title(key)
#         if key == 'all':
#             ax[counter].set_xlim(0)
#         if counter%2 == 0:
#             ax[counter].set_ylabel('Neuronal Indices')
#         counter += 1
# ax[-1].set_xlabel('Time (seconds)')
# ax[-1].set_xlim(0,10)
# [ax[0].axvline(val, ls='-', ymin=-1.5,color='orange').set_clip_on(False) for val in [2,3,5,6,8,9]]
# fig.savefig(fig_path/"raster.png", dpi=200)
# # %%
# # =======================
# #! Binary time series
# from causal4.utils import spk2bin
# for key in ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie_one', 'natural_movie_three']:
#     spk_data = filtered_spks[key]
#     N = int(spk_data[:,1].max()+1)
#     spk_bin_buff = []
#     for i in range(N):
#         buff = spk2bin(spk_data[spk_data[:,1]==i,0], dt=1)
#         spk_bin_buff.append(buff)
#     T = spk_bin_buff[int(spk_data[-1,1])].shape[0]
#     spk_bin = np.zeros((N, T))
#     for i in range(N):
#         spk_bin[i, :spk_bin_buff[i].shape[0]] = spk_bin_buff[i]

#     cov = (spk_bin @ spk_bin.T)/T
#     cov[np.eye(N, dtype=bool)] = 0
#     mean = spk_bin.mean(1)
#     mean = np.sqrt(np.outer(mean, mean))
#     syn_mat = cov/mean
#     print(f"{key:20s}: {syn_mat.mean():.3e}")
# # %%
# def spk2rate(spikes, window, stride):
#     dt = 1
#     spk_bin = np.zeros(np.ceil(spikes.max()/dt).astype(int)+1)
#     spk_bin[(spikes/dt).astype(int)] = 1
#     return spk_bin

# key = 'drifting_gratings'
# spk_data = filtered_spks[key]
# N = int(spk_data[:,1].max()+1)
# spk_bin_buff = []
# for i in range(N):
#     buff = spk2bin(spk_data[spk_data[:,1]==i,0], dt=5)
#     spk_bin_buff.append(buff)
# T = spk_bin_buff[int(spk_data[-1,1])].shape[0]
# spk_bin = np.zeros((N, T))
# for i in range(N):
#     spk_bin[i, :spk_bin_buff[i].shape[0]] = spk_bin_buff[i]
# #%%
# mfr = spk_bin.sum(0)/N*1000
# # %%
# from scipy.ndimage.filters import gaussian_filter1d
# tt = np.arange(mfr.shape[0])/1e3
# mfr_smoothen = gaussian_filter1d(mfr, 500)
# #%%
# fft = np.abs(np.fft.fft(mfr))
# fft_freq = np.fft.fftfreq(T, 0.001)
# mask = (fft_freq>0)*(fft_freq<20)
# plt.semilogy(fft_freq[mask], fft[mask])
# %%
