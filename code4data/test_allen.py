#%%
import pickle
import h5py

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16

from causal4.Causality import CausalityEstimator
from causal4.utils import Gaussian, Double_Gaussian, Double_Gaussian_Analysis, match_features
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
allen_data = {key: {'roc':{}, 'hist': {}, 'hist_inconsist': {}} for key in stimulus_names_plot}
log_norm_fit_pval = {}

new_columns_map = {
    'log-TE': 'TE',
    'log-GC': 'GC',
    'log-MI': 'sum(MI)',
    'log-CC': 'sum(CC2)',
    'log-dp': 'Delta_p',
}
for i, axi in enumerate(ax.flatten()[:len(stimulus_names_plot)]):
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(False)
    ax_hist = inset_axes(axi, width="100%", height="100%",
                    bbox_to_anchor=(.35, .35, .50, .50),
                    bbox_transform=axi.transAxes, loc='center', axes_kwargs={'facecolor':[1,1,1,0]})
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    estimator.spk_fname = fnaming(stimulus_names_plot[i], gap_vals[i])
    estimator.T = hf[fnaming(stimulus_names_plot[i])].attrs['T'] + TGIC_cfg['suffix']*1e3
    # fetch data and transform causality value into log-scale
    data = estimator.fetch_data()
    for key, val in new_columns_map.items():
        data[key] = np.log10(np.abs(data[val]))
        if key in ('log-TE', 'log-MI'):
            data[key] += np.log10(2)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    data = data[(data['pre_id'].isin(chosen_unit_set)) & (data['post_id'].isin(chosen_unit_set))].copy()
    log_norm_fit_pval[stimulus_names_plot[i]] = {}
    for key in ('CC', 'MI', 'GC', 'TE'):
        data_buff = data['log-'+key].to_numpy()
        causal_values[key][:, i] = data_buff
        vrange=(-8,-2)
        counts, bins = np.histogram(data_buff, bins=100, range=vrange, density=True)
        allen_data[stimulus_names_plot[i]]['hist'][key] = counts.copy()
        # counts, bins = np.histogram(log_cau[~np.isinf(log_cau)], bins=200, range=(-8,-2), density=True)
        popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts, bins, p0=[0.5, 0.5, -5, -4, 1, 1])
        # ax_hist.plot(bins[:-1], Double_Gaussian(bins[:-1], *popt), lw=4, ls=':', color=line_rc[key]['color'], )
        # ax_hist.plot(bins[:-1], Gaussian(bins[:-1], popt[0], popt[2], popt[4], ), '-o', lw=2, ms=1, color=line_rc[key]['color'], )
        # ax_hist.plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3], popt[5], ), '-d', lw=2, ms=1, color=line_rc[key]['color'], )
        ax_hist.plot(bins[:-1], counts, **line_rc[key])
        ax_hist.axvline(threshold, color=line_rc[key]['color'], ls='--')
        axi.plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=line_rc[key]['label'])[0].set_clip_on(False)

        log_norm_fit_pval[stimulus_names_plot[i]][key] = popt
        th_conditions[stimulus_names_plot[i]][key] = threshold
        print(f"{key:s}: {-np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2):.3f}", end='\t')
        allen_data[stimulus_names_plot[i]]['roc'][key] = np.vstack((fpr, tpr))
    ax_hist.set_xlim(*vrange)
    xticks = np.arange(-8, -1, 2)
    ax_hist.set_xticks(xticks)
    ax_hist.set_xticklabels([r"$10^{%.0f}$"%val for val in ax_hist.get_xticks()])
    print('')
    # ax_hist.set_title(stimulus_names_plot[i].replace('_', ' '))
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
    ax_hist.set_ylabel('Probability Density')
    ax_hist.set_xlabel('Causal value')
    axi.set_xlim(0,1)
    axi.set_ylim(0,1)
[axi.set_ylabel('True Positive Rate', fontsize=30) for axi in ax[:,0]]
[axi.set_xlabel('False Positive Rate', fontsize=30) for axi in ax[-1,:]]
# ax[0].set_ylabel('Probability Density')
# [ax[i].set_xlabel('Causal value') for i in range(3)]

allen_data['edges'] = bins[:-1].copy()
allen_data['log_norm_fit_pval'] = log_norm_fit_pval
allen_data['opt_th'] = th_conditions

plt.tight_layout()
plt.savefig(fig_path/f"histogram_of_TE_allen-{TGIC_prefix:s}-{long_fnaming('all'):s}{fig_sfx:s}.pdf")

#%%
#! ====================
#! Draw histogram of causal values for each stimuli filtered with inconsistent masks
#! ====================
inconsist_range = (1,3)
inconsist_mask = {}
for key in ('CC', 'MI', 'GC', 'TE'):
    ths = np.array([th_conditions[stimuli][key] for stimuli in stimulus_names_plot])
    # if key in ('TE', 'MI'):
    #     ths -= np.log10(2)
    bin_recon_buff = (causal_values[key]>=ths).astype(int).sum(1)
    inconsist_mask[key] = (bin_recon_buff>inconsist_range[0])*(bin_recon_buff<inconsist_range[1])
    print(f"{key:s}: {np.sum(inconsist_mask[key])/new_N/(new_N-1)*100:>5.2f} %")


for key in ('CC', 'MI', 'GC', 'TE'):
    fig, ax = plt.subplots(2,2, figsize=(16,14))
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    for i, ax_hist in enumerate(ax.flatten()[:len(stimulus_names_plot)]):
        log_cau = causal_values[key][:, i]
        # if key in ('TE', 'MI'):
        #     log_cau += np.log10(2)
        vrange=(-8,-2)
        counts, bins = np.histogram(log_cau[inconsist_mask[key]], bins=100, range=vrange, density=True)
        pop_ratio = np.sum(inconsist_mask[key])/len(log_cau)
        ax_hist.plot(bins[:-1], gaussian_filter1d(counts*pop_ratio, 1), lw=6, color='#00C2A0', label=line_rc[key]['label'], zorder=1)
        allen_data[stimulus_names_plot[i]]['hist_inconsist'][key] = counts*pop_ratio
        # counts, bins = np.histogram(log_cau[~np.isinf(log_cau)], bins=200, range=(-8,-2), density=True)
        # ax_hist.plot(bins[:-1], Double_Gaussian(bins[:-1], *popt), lw=4, ls=':', color=line_rc[key]['color'], )
        popt = log_norm_fit_pval[stimulus_names_plot[i]][key]
        ax_hist.plot(bins[:-1], Gaussian(bins[:-1], popt[0], popt[2], popt[4], ), lw=4, alpha=1.0, color=line_rc[key]['color'],zorder=0)
        ax_hist.plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3], popt[5], ), lw=4, alpha=1.0, color=c_inv[line_rc[key]['color']],zorder=0)
        # calculate threshold
        ax_hist.axvline(th_conditions[stimulus_names_plot[i]][key], color='k', ls='-')

        ax_hist.set_xlim(*vrange)
        xticks = np.arange(-8, -1, 2)
        ax_hist.set_xticks(xticks)
        ax_hist.set_xticklabels([r"$10^{%.0f}$"%val for val in ax_hist.get_xticks()])
        # ax_hist.set_title(stimulus_names_plot[i].replace('_', ' '))
        ax_hist.set_ylim(0)
        axi.legend(loc='upper right')
        if '-' in stimulus_names_plot[i]:
            arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
        else:
            arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
        axins = inset_axes(ax_hist, width="100%", height="100%",
                        bbox_to_anchor=(.05, .75, .23, .23),
                        bbox_transform=ax_hist.transAxes, loc='center')

        axins.imshow(arr_image)
        axins.axis('off')
        axi.set_xlim(0,1)
        axi.set_ylim(0,1)
    [axi.set_ylabel('probability density', fontsize=30) for axi in ax[:,0]]
    [axi.set_xlabel(key, fontsize=30) for axi in ax[-1,:]];

    plt.tight_layout()
    plt.savefig(fig_path/f"histogram_of_{key:s}_allen-{TGIC_prefix:s}-{long_fnaming('all'):s}{fig_sfx:s}_fitted13.pdf")

#%%
#! Draw the heatmap of correlation coefficient matrix
#! ----
allen_data['consistency']={}
allen_data['consistency_bin']={}
adj_recon = {}
for idx, key in enumerate(('CC', 'MI', 'GC', 'TE')):
    data = np.corrcoef(causal_values[key], rowvar=False)
    allen_data['consistency'][key]=data
    ths = np.array([th_conditions[stimuli][key] for stimuli in stimulus_names_plot])
    adj_recon[key] = (causal_values[key]>=ths)
    allen_data['consistency_bin'][key]=np.corrcoef(adj_recon[key].astype(float), rowvar=False)
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

    # plt.savefig(fig_path/f"allen_recon_rsa4_{key:s}-{TGIC_prefix:s}-{long_fnaming('all'):s}_annot{fig_sfx:s}.pdf")
    print(f"Minimum coincidence rate : {data.min():6.3f}")
    print(f"Maximum coincidence rate : {np.sort(np.unique(data))[-2]:6.3f}")
    print(data)

#%%
#! ============================================================
#! Draw the histogram of dp and Delta p for each stimuli
#! ============================================================
N = n_unit
new_N = int(np.sum(unit_rate_mask_union))
fig, ax = plt.subplots(2,2, figsize=(16,14))
for i, axi in enumerate(ax.flatten()[:len(stimulus_names_plot)]):
    # for var, label in zip(('Delta_p', 'dp'), (r'$\Delta p_m$', r'$\delta p$')):
    for var, label in zip(('Delta_p',), (r'$\Delta p_m$',)):
        estimator.spk_fname = fnaming(stimulus_names_plot[i], gap_vals[i])
        estimator.T = hf[fnaming(stimulus_names_plot[i])].attrs['T'] + TGIC_cfg['suffix']*1e3
        # fetch data and transform causality value into log-scale
        data = estimator.fetch_data()
        data['log-dp'] = np.log10(np.abs(data['Delta_p']))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        data = data[(data['pre_id'].isin(chosen_unit_set)) & (data['post_id'].isin(chosen_unit_set))].copy()
        counts, bins = np.histogram(data['log-dp'], bins=100, density=True, range=(-2,1))
        allen_data[stimulus_names_plot[i]][var+'_hist'] = {}
        allen_data[stimulus_names_plot[i]][var+'_hist']['counts'] = counts.copy()
        allen_data[stimulus_names_plot[i]][var+'_hist']['edges'] = bins[:-1].copy()
        axi.plot(bins[:-1], counts, label=label,lw=4)
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
[axi.set_xlabel(r'$\Delta p_m$ or $\delta_p$ values', fontsize=30) for axi in ax[-1,:]]
[axi.set_ylabel('probability density', fontsize=30) for axi in ax[:,0]]
plt.tight_layout()
plt.savefig(fig_path/f"histogram_of_dp_Delta_p-{TGIC_prefix:s}-{long_fnaming('all'):s}_allen{fig_sfx:s}.pdf")

with open(data_path/'allen_data.pkl', 'wb') as f:
    pickle.dump(allen_data, f)

#%%
# dp = cau.load_from_single(long_fnaming(stimulus_names[0]), 'dp')[unit_rate_mask_union,:][:,unit_rate_mask_union]
# Dp = cau.load_from_single(long_fnaming(stimulus_names[0]), 'Delta_p')[unit_rate_mask_union,:][:,unit_rate_mask_union]
# px = cau.load_from_single(long_fnaming(stimulus_names[0]), 'px')[unit_rate_mask_union,:][:,unit_rate_mask_union]
# py = cau.load_from_single(long_fnaming(stimulus_names[0]), 'py')[unit_rate_mask_union,:][:,unit_rate_mask_union]
# %%
# drifting_gratings :   1884568 ms
# static_gratings :     1503250 ms
# natural_scenes :      1490757 ms
# natural_movie_one :   601496  ms
# natural_movie_three : 1202032 ms
# 1884568+1503250+1490757+601496+1202032