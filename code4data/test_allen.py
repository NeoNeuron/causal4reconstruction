#%%
import pickle
import h5py

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

from causal4.Causality import CausalityEstimator
from causal4.utils import Gaussian, match_features, reconstruction_analysis
from causal4.figrc import line_rc, c_inv, fig_path, data_path

import warnings
warnings.filterwarnings('ignore')
key_map = {'TE': 'TE', 'MI': 'sum(MI)', 'CC': 'sum(CC2)', 'GC': 'GC'}
#%%
session_id = 715093703
with open(f"../allen_data/preprocessed_allen_data_session_{session_id:d}.pkl", 'rb') as f:
    data_pickle = pickle.load(f)
    n_unit = data_pickle['index'].shape[0]
    stimulus_names = data_pickle['stimulus']
    # high_rate_mask = data_pickle['rate_tight'] >= 0.05
    # unit_rate_mask_union = data_pickle['rate_raw'] >= 0.05
    del data_pickle
# run causality measures
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

def fnaming(name, gap=gap_width):
    return f"session_{session_id:d}_{name:s}_" \
        + f"ref={int(t_ref):d}_" \
        + f"gap={int(gap):d}"

# setup configurations
TGIC_cfg = dict(
    order = (1,5),
    dt = 1,
    delay = 0,
    suffix = 0,
)

TGIC_prefix = f"K={TGIC_cfg['order'][0]:d}_{TGIC_cfg['order'][1]:d}" \
            + f"bin={TGIC_cfg['dt']:.2f}" \
            + f"delay={TGIC_cfg['delay']:.2f}"

def long_fnaming(name, gap=gap_width, sfx=TGIC_cfg['suffix']):
    return f"sfx={sfx:d}-{fnaming(name, gap):s}"

# %%
#! ====================
#! Draw histogram of causal values for each stimuli
#! ====================
# stimulus_names_plot = stimulus_names.copy()
# fig_sfx = '_all9'
stimulus_names_plot = ['drifting_gratings-with-gray', 'static_gratings', 'natural_scenes', 'natural_movie']
# stimulus_names_plot = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie', 'gabors', 'spontaneous']
# stimulus_names_plot = ['flashes', 'gabors', 'spontaneous']
# fig_sfx = '_new3' 
gap_vals = np.ones(len(stimulus_names_plot),dtype=int)*gap_width
# gap_vals = [1, 250, 250, 250,]
sfx = np.ones(len(stimulus_names_plot), dtype=int)*TGIC_cfg['suffix']
# sfx[-2] = 500
fig_sfx = '_raw4'

hf = h5py.File('../allen_data/metadata.h5','r')
new_rate = np.array([hf[fnaming(stimulus)][:] for stimulus in ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie']])
# new_rate = np.array([hf[fnaming(stimulus)][:] for stimulus in stimulus_names_plot])
high_rate_mask = new_rate >= 0.08#0.335
unit_rate_mask_union = high_rate_mask.sum(0) == high_rate_mask.shape[0]
print(f">> {np.sum(unit_rate_mask_union):d} units are under mask")
#%%
N = n_unit
pm = dict(
    spk_fname = fnaming(stimulus_names_plot[0], gap_vals[0]),
    N = n_unit,
    order = TGIC_cfg['order'],
    T = hf[fnaming(stimulus_names_plot[0])].attrs['T'] + TGIC_cfg['suffix']*1e3,
    DT = 2e5,
    dt = TGIC_cfg['dt'],
    delay = TGIC_cfg['delay'],
    path = f'../allen_data/data/EE/N={n_unit:d}/',
)
estimator = CausalityEstimator(**pm, n_thread=60)

new_N = int(np.sum(unit_rate_mask_union))
chosen_unit_set = np.nonzero(unit_rate_mask_union)[0]
fig, ax = plt.subplots(2,2, figsize=(16,14))
causal_values = {key: np.zeros((new_N*new_N-new_N, len(stimulus_names_plot))) for key in ('TE', 'GC', 'MI', 'CC')}
th_conditions = {stimuli: {} for stimuli in stimulus_names_plot}
allen_data = {}
data_recon_list = []
log_norm_fit_pval = {}

for i, axi in enumerate(ax.flatten()[:len(stimulus_names_plot)]):
    # fetch causality data
    estimator.spk_fname = fnaming(stimulus_names_plot[i], gap_vals[i])
    estimator.T = hf[fnaming(stimulus_names_plot[i])].attrs['T'] + TGIC_cfg['suffix']*1e3
    data = estimator.fetch_data()
    data = data[(data['pre_id'].isin(chosen_unit_set)) & (data['post_id'].isin(chosen_unit_set))].copy()
    data_matched = match_features(data, N=n_unit)
    vrange=(-8,-2)
    data_recon, data_fig = reconstruction_analysis(data_matched, nbins=100, hist_range=vrange, fit_p0=[0.5,-5.8,-4.2,1,1])
    data_fig = data_fig.dropna(axis=1, how='all')
    allen_data[stimulus_names_plot[i]] = data_fig.copy()
    data_recon['stimulus'] = stimulus_names_plot[i]
    data_recon_list.append(data_recon.copy())

    ax_hist = inset_axes(axi, width="100%", height="100%",
                    bbox_to_anchor=(.35, .35, .50, .50),
                    bbox_transform=axi.transAxes, loc='center', axes_kwargs={'facecolor':[1,1,1,0]})

    log_norm_fit_pval[stimulus_names_plot[i]] = {}
    for key in ('CC', 'MI', 'GC', 'TE'):
        causal_values[key][:, i] = data_recon['log-'+key].to_numpy()
        ax_hist.plot(data_fig['edges'][key], data_fig['hist'][key], **line_rc[key])
        ax_hist.axvline(data_fig['th_gauss'][key], color=line_rc[key]['color'], ls='--')
        fpr, tpr = data_fig['roc_blind'][key]
        axi.plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=line_rc[key]['label'])[0].set_clip_on(False)

        log_norm_fit_pval[stimulus_names_plot[i]][key] = data_fig['log_norm_fit_pval'][key].copy()
        th_conditions[stimulus_names_plot[i]][key] = data_fig['th_gauss'][key]
        print(f"{key:s}: {data_fig['auc_gauss'][key]:.3f}", end='\t')
    ax_hist.set_xlim(*vrange)
    xticks = np.arange(-8, -1, 2)
    ax_hist.set_xticks(xticks)
    ax_hist.set_xticklabels([r"$10^{%.0f}$"%val for val in ax_hist.get_xticks()])
    print('')
    ax_hist.set_ylim(0)
    axi.legend(loc='upper right')
    if '-' in stimulus_names_plot[i]:
        arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
    else:
        arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
    axins = inset_axes(axi, width="100%", height="100%",
                    bbox_to_anchor=(.05, .05, .23, .23),
                    bbox_transform=axi.transAxes, loc='center')

    axins.imshow(arr_image)
    axins.axis('off')
    ax_hist.set_ylabel('probability density')
    ax_hist.set_xlabel('causal value')
    axi.set_xlim(0,1)
    axi.set_ylim(0,1)
[axi.set_ylabel('True Positive Rate', fontsize=30) for axi in ax[:,0]]
[axi.set_xlabel('False Positive Rate', fontsize=30) for axi in ax[-1,:]]

data_recon = pd.concat(data_recon_list)

plt.tight_layout()
plt.savefig(fig_path/f"histogram_of_TE_allen-{TGIC_prefix:s}-{long_fnaming('all'):s}{fig_sfx:s}.pdf")

#%%
#! ====================
#! Draw histogram of causal values for each stimuli filtered with inconsistent masks
#! ====================
tmp = pd.DataFrame(
    data_recon.groupby(['pre_id', 'post_id']).sum()[
        ['recon-gauss-CC', 'recon-gauss-MI', 'recon-gauss-GC', 'recon-gauss-TE']])
for key in ('CC', 'MI', 'GC', 'TE'):
    print(f"{key:s}: {tmp['recon-gauss-'+key].eq(2).mean()*100:>5.2f} %")

inconsist_hist = {key:{} for key in stimulus_names_plot}
for key in ('CC', 'MI', 'GC', 'TE'):
    mask = tmp['recon-gauss-'+key].eq(2)
    selected_data = pd.merge(tmp[mask], data_recon, how='left', left_index=True, right_on=['pre_id', 'post_id'])
    fig, ax = plt.subplots(2,2, figsize=(16,14))
    for stim, axi in zip(stimulus_names_plot, ax.flatten()):
        vrange=(-8,-2)
        counts, bins = np.histogram(selected_data[selected_data['stimulus'].eq(stim)]['log-'+key], bins=100, range=vrange, density=True)
        pop_ratio = tmp['recon-gauss-'+key].eq(2).mean()
        axi.plot(bins[:-1], gaussian_filter1d(counts*pop_ratio, 1), lw=6, color='#00C2A0', label=line_rc[key]['label'], zorder=1)
        inconsist_hist[stim][key] = counts*pop_ratio,
        popt = allen_data[stim]['log_norm_fit_pval'][key]
        axi.plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3])*(1-popt[0]), lw=4, alpha=1.0, color=line_rc[key]['color'],zorder=0)
        axi.plot(bins[:-1], Gaussian(bins[:-1], popt[2], popt[4])*popt[0], lw=4, alpha=1.0, color=c_inv[line_rc[key]['color']],zorder=0)
        # calculate threshold
        axi.axvline(allen_data[stim]['th_gauss'][key], color='k', ls='-')

        axi.set_xlim(*vrange)
        xticks = np.arange(-8, -1, 2)
        axi.set_xticks(xticks)
        axi.set_xticklabels([r"$10^{%.0f}$"%val for val in axi.get_xticks()])
        axi.set_ylim(0)
        if '-' in stim:
            arr_image = plt.imread('../'+stim.split('-')[0]+'.png', format='png')
        else:
            arr_image = plt.imread('../'+stim+'.png', format='png')
        axins = inset_axes(axi, width="100%", height="100%",
                        bbox_to_anchor=(.05, .75, .23, .23),
                        bbox_transform=axi.transAxes, loc='center')

        axins.imshow(arr_image)
        axins.axis('off')
    [axi.set_ylabel('probability density', fontsize=30) for axi in ax[:,0]]
    [axi.set_xlabel(key, fontsize=30) for axi in ax[-1,:]];

    plt.tight_layout()
    plt.savefig(fig_path/f"histogram_of_{key:s}_allen-{TGIC_prefix:s}-{long_fnaming('all'):s}{fig_sfx:s}_fitted13.pdf")

for key in allen_data.keys():
    allen_data[key] = allen_data[key].merge(
        pd.DataFrame(inconsist_hist[key], index=['hist_inconsist']).T,
        left_index=True, right_index=True, how='left')
#%%
#! Draw the heatmap of correlation coefficient matrix
#! ----
allen_data['consistency']={}
allen_data['consistency_binary']={}
for idx, key in enumerate(('CC', 'MI', 'GC', 'TE')):
    tmp = data_recon[['stimulus', 'pre_id', 'post_id', 'log-'+key, 'recon-gauss-'+key]]
    tmp = tmp.sort_values(by=['stimulus', 'pre_id', 'post_id'])
    data = np.corrcoef(tmp['log-'+key].to_numpy().reshape(4,-1))
    allen_data['consistency'][key]=data
    allen_data['consistency_binary'][key]=np.corrcoef(tmp['recon-gauss-'+key].to_numpy().reshape(4,-1))
    mask = np.triu(np.ones_like(data, dtype=bool),k=1)
    fig, g = plt.subplots(1,1, figsize=(10,10), dpi=200, 
        gridspec_kw=dict(bottom=0.2, left=0.2, top=0.95, right=0.95))
    g = sns.heatmap(data, mask=mask,
        vmin=0, vmax=1, 
        cmap=plt.cm.OrRd, 
        square=True,
        lw=.5,
        ax=g,
        annot=True,
        annot_kws={"fontsize":25}
        )

    g.set_xticklabels([])
    g.set_yticklabels([])

    length = 1./data.shape[0]-0.01
    # draw x-axis
    for i in range(len(stimulus_names_plot)):
        if '-' in stimulus_names_plot[i]:
            arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
        else:
            arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
        axins = inset_axes(g, width="100%", height="100%",
                        bbox_to_anchor=(.005+i*(length+0.01), -length-0.01, length, length),
                        bbox_transform=g.transAxes, loc='center')

        axins.imshow(arr_image)
        axins.axis('off')

    # draw y-axis
    for i in range(len(stimulus_names_plot)):
        if '-' in stimulus_names_plot[i]:
            arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
        else:
            arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
        axins = inset_axes(g, width="100%", height="100%",
                        bbox_to_anchor=(-length-0.01, 1-length-0.005-i*(length+0.01), length, length),
                        bbox_transform=g.transAxes, loc='center')

        axins.imshow(arr_image)
        axins.axis('off')

    plt.savefig(fig_path/f"allen_recon_rsa4_{key:s}-{TGIC_prefix:s}-{long_fnaming('all'):s}_annot{fig_sfx:s}.pdf")
    print(f"Minimum coincidence rate : {data.min():6.3f}")
    print(f"Maximum coincidence rate : {np.sort(np.unique(data))[-2]:6.3f}")
    print(data)

#%%
#! ============================================================
#! Draw the histogram of dp and Delta p for each stimuli
#! ============================================================
N = n_unit
new_N = int(np.sum(unit_rate_mask_union))
fig, ax = plt.subplots(2,2, figsize=(10,8))
for i, axi in enumerate(ax.flatten()[:len(stimulus_names_plot)]):
    bins = allen_data[stimulus_names_plot[i]]['edges']['dp']
    counts = allen_data[stimulus_names_plot[i]]['hist']['dp']
    axi.plot(bins, counts, label=r'$\Delta_p$', lw=4)
    axi.axvline(0, ls=':', color='r')
    axi.set_ylim(0)
    axi.set_xlim(-2,1)
    xticks = np.array([-2,-1,0,1])
    axi.set_xticks(xticks)
    axi.set_xticklabels([r'$10^{%d}$'%val for val in xticks])

    if '-' in stimulus_names_plot[i]:
        arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
    else:
        arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
    axins = inset_axes(axi, width="100%", height="100%",
                    bbox_to_anchor=(.05, .3, .23, .23),
                    bbox_transform=axi.transAxes, loc='center')

    axins.imshow(arr_image)
    axins.axis('off')
[axi.set_xlabel(r'$|\Delta p_m|$ values', fontsize=30) for axi in ax[-1,:]]
[axi.set_ylabel('probability density', fontsize=24) for axi in ax[:,0]]
plt.tight_layout()
plt.savefig(fig_path/f"histogram_of_dp_Delta_p-{TGIC_prefix:s}-{long_fnaming('all'):s}_allen{fig_sfx:s}.pdf")

with open(data_path/'allen_data.pkl', 'wb') as f:
    pickle.dump(allen_data, f)

# %%
# drifting_gratings :   1884568 ms
# static_gratings :     1503250 ms
# natural_scenes :      1490757 ms
# natural_movie_one :   601496  ms
# natural_movie_three : 1202032 ms
# 1884568+1503250+1490757+601496+1202032