# %% [markdown]
# # Visual Coding - Neuropixels
""" Replace code in allensdk.brain_observatory.ecephys.ecephys_session.presentationwise_spike_times()

        ```python
        presentation_times = np.zeros([stimulus_presentations.shape[0] * 2])
        presentation_times[::2] = np.array(stimulus_presentations['start_time'])
        presentation_times[1::2] = np.array(stimulus_presentations['stop_time'])
        ```

    with:
        ```python
        sep_idx = np.nonzero(np.diff(stimulus_presentation_ids)>1)[0]
        sep_idx = np.append(sep_idx, sep_idx+1)
        sep_idx = np.append(sep_idx, [0, stimulus_presentation_ids.shape[0]-1])
        sep_idx = np.sort(sep_idx.flatten())
        presentation_times = np.zeros(sep_idx.shape[0])
        presentation_times[::2] = np.array(stimulus_presentations['start_time'].iloc[sep_idx[::2]])
        presentation_times[1::2] = np.array(stimulus_presentations['stop_time'].iloc[sep_idx[1::2]] )
        ```
    And rename the new function as presentationwise_spike_times_modified

"""
# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import struct
import pickle
from causal4.figrc import fig_path, data_path

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
data_directory = '../data/neuropixel/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session_id = 715093703
nwb_path = f'session_{session_id:d}.nwb'
# sessions = cache.get_session_table()
# session = cache.get_session_data(session_id)
session = EcephysSession.from_nwb_path(data_directory+nwb_path, api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf
    })

# print arguments
# print([attr_or_method for attr_or_method in dir(session) if attr_or_method[0] != '_'])

# %%
units = cache.get_units()
len(units)
# %%
units = cache.get_units(amplitude_cutoff_maximum = np.inf,
                        presence_ratio_minimum = -np.inf,
                        isi_violations_maximum = np.inf)

len(units)
# %%
from scipy.ndimage.filters import gaussian_filter1d
plt.rcParams.update({'font.size': 14})

def plot_metric(data, bins, x_axis_label, color, max_value=-1):
    
    h, b = np.histogram(data, bins=bins, density=True)

    x = b[:-1]
    y = gaussian_filter1d(h, 1)

    plt.plot(x, y, color=color)
    plt.xlabel(x_axis_label)
    plt.gca().get_yaxis().set_visible(False)
    [plt.gca().spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]
    if max_value < np.max(y) * 1.1:
        max_value = np.max(y) * 1.1
    plt.ylim([0, max_value])
    
    return max_value
region_dict = {'cortex' : ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal','VISmma','VISmmp','VISli'],
               'thalamus' : ['LGd','LD', 'LP', 'VPM', 'TH', 'MGm','MGv','MGd','PO','LGv','VL',
                             'VPL','POL','Eth','PoT','PP','PIL','IntG','IGL','SGN','VPL','PF','RT'],
               'hippocampus' : ['CA1', 'CA2','CA3', 'DG', 'SUB', 'POST','PRE','ProS','HPF'],
               'midbrain': ['MB','SCig','SCiw','SCsg','SCzo','PPT','APN','NOT','MRN','OP','LT','RPF','CP']}

color_dict = {'cortex' : '#08858C',
              'thalamus' : '#FC6B6F',
              'hippocampus' : '#7ED04B',
              'midbrain' : '#FC9DFE'}

bins = np.linspace(-3,2,100)
max_value = -np.inf

for idx, region in enumerate(region_dict.keys()):
    
    data = np.log10(units[units.ecephys_structure_acronym.isin(region_dict[region])]['firing_rate'])
    
    max_value = plot_metric(data, bins, 'log$_{10}$ firing rate (Hz)', color_dict[region], max_value)
    
_ = plt.legend(region_dict.keys())
# %%
print(f'{session.units.shape[0]} units total')
units_high_snr = session.units[session.units['snr'] > 4]
units_high_fr = session.units[session.units['firing_rate'] > 0.05]

units_chosen = session.units[(session.units['snr'] > 4)]# * (session.units['firing_rate'] > 0.05)]
# drop abnormal unit
units_chosen = units_chosen.drop(950942603)
print(f'{units_high_snr.shape[0]} units have snr > 4')
print(f'{units_high_fr.shape[0]} units have firing_rate > 0.05')

# grab an arbitrary (though high-snr!) unit (we made units_with_high_snr above)
# high_snr_unit_ids = units_with_very_high_snr.index.values
high_snr_unit_ids = units_chosen.index.values
unit_id = high_snr_unit_ids[0]

# print(f"{len(session.spike_times[unit_id])} spikes were detected for unit {unit_id} at times:")
# session.spike_times[unit_id]
# %%
stimulus_names = [
    'drifting_gratings-with-gray',
    'drifting_gratings',
    'static_gratings',
    'natural_scenes',
    'natural_movie_one',
    'natural_movie_three',
    'flashes',
    'gabors',
    'spontaneous',
]
indices = np.zeros((5,193))
stimulus_count = 0
# data_out = np.zeros((5,194,2))
stimulus_duration = np.zeros(len(stimulus_names))
data_pickle = {
    'index' : units_chosen.index.values,
    'rate_raw' : units_chosen.firing_rate.values,
    'area_name' : units_chosen.ecephys_structure_acronym.values,
    'stimulus' : stimulus_names,
    'stimulus_present_time' : {},
    'rate_tight' : np.zeros((len(stimulus_names),units_chosen.shape[0])),
    'rate_loose' : np.zeros((len(stimulus_names),units_chosen.shape[0])),
    'spike_times': {},
}
#%%
fig, ax = plt.subplots(len(stimulus_names),1, figsize=(12, len(stimulus_names)*3), sharex=True)
for idx, stimulus in enumerate(stimulus_names):
    # get spike times from the first block of drifting gratings presentations 
    if 'with-gray' in stimulus: # fill in stimulus-free period with spontaneous activities
        stimulus_presentation_ids = session.stimulus_presentations.loc[
            (session.stimulus_presentations['stimulus_name'] == stimulus.split('-')[0])
        ].index.values

        # select all spikes responding for drift gratings
        times = session.presentationwise_spike_times_modified(
            stimulus_presentation_ids=stimulus_presentation_ids,
            unit_ids=units_chosen.index.values
        )
    else:
        stimulus_presentation_ids = session.stimulus_presentations.loc[
            (session.stimulus_presentations['stimulus_name'] == stimulus)
        ].index.values

        # select all spikes responding for drift gratings
        times = session.presentationwise_spike_times(
            stimulus_presentation_ids=stimulus_presentation_ids,
            unit_ids=units_chosen.index.values
        )
    # output to *.dat files
    spike_times = np.vstack((times.index.values, times.unit_id.values)).T
    data_pickle['spike_times'][stimulus] = spike_times.copy()
    print(f"{np.unique(times.unit_id.values).shape[0]:d} units in total for {stimulus:s} trials.")
    ax[idx].plot(spike_times[:,0], spike_times[:,1], '|')
    ax[idx].set_title(stimulus)
    ax[idx].set_ylabel('Neuronal Indices')

    # calculate from the data
    stimulus_duration[stimulus_count] = session.stimulus_presentations.loc[stimulus_presentation_ids, 'duration'].sum()
    data_pickle['stimulus_present_time'][stimulus] = stimulus_duration[stimulus_count]
    for i, unit_id in enumerate(data_pickle['index']):
        data_pickle['rate_tight'][stimulus_count, i] = np.sum(spike_times[:,1] == unit_id)/stimulus_duration[stimulus_count]
        data_pickle['rate_loose'][stimulus_count, i] = np.sum(spike_times[:,1] == unit_id)

    T = spike_times[-1,0] - spike_times[0,0]
    data_pickle['rate_loose'][stimulus_count,:] /= T
    stimulus_count +=1
ax[-1].set_xlabel('Time (seconds)', )
plt.tight_layout()
plt.savefig(fig_path/'allen_raster.png', dpi=300)
# %%
if not os.path.isdir(f"../allen_data/data/EE/N={units_chosen.index.shape[0]:d}"):
    os.mkdir(f"../allen_data/data/EE/N={units_chosen.index.shape[0]:d}")
with open(f'../allen_data/preprocessed_allen_data_session_{session_id:d}.pkl', 'wb') as f:
    pickle.dump(data_pickle, f)
#%%