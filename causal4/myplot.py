"""
    Library for plotting function used in analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .figrc import line_rc, roc_formatter
from .utils import Gaussian, load_spike_data
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, MaxNLocator
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def get_conn_fpath(pm_causal):
    N = pm_causal['Ne']+pm_causal['Ni']
    fname_conn = pm_causal['path_output']
    if pm_causal['Ne']*pm_causal['Ni'] > 0:
        fname_conn += f"EI/N={N:d}/"
    elif pm_causal['Ne'] > 0:
        fname_conn += f"EE/N={N:d}/"
    elif pm_causal['Ni'] > 0:
        fname_conn += f"II/N={N:d}/"
    else:
        raise ValueError('No neuron!')
    fname_conn += pm_causal['con_mat']
    return fname_conn

def format_xticks(ax, hist_range):
    ax.set_xlim(hist_range[0], hist_range[-1])
    if hist_range[-1]-hist_range[0]<=5:
        xticks = np.arange(hist_range[0], hist_range[-1]+1)
        ax.set_xticks(xticks)
    else:
        if hist_range[0] % 2:
            xticks = np.arange(hist_range[0]+1, hist_range[-1]+2)
        else:
            xticks = np.arange(hist_range[0], hist_range[-1]+1)
        ax.set_xticks(xticks[::2])
    ax.xaxis.set_major_formatter(sci_formatter)

def hist_causal_with_conn_mask(data_fig:pd.DataFrame, hist_range:tuple=None)->None:
    """Histogram of causality masked by ground truth.

    Args:
        pm_causal (dict): dict of causality parameters.
        hist_range (tuple): range of the histogram. Default None.
    """

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    print(data_fig['acc_kmeans'].dropna())

    if 'conn' in data_fig.index:
        axins = inset_axes(ax[0], width="100%", height="100%",
                        bbox_to_anchor=(.05, .55, .4, .3),
                        bbox_transform=ax[0].transAxes, loc='center')
        axins.spines['left'].set_visible(False)

        axins.bar(data_fig['edges']['conn'], data_fig['hist']['conn'],
                  width=data_fig['edges']['conn'][1]-data_fig['edges']['conn'][0])
        if data_fig['edges']['conn'][0] < 0:
            axins.xaxis.set_major_formatter(sci_formatter)
        axins.set_xticklabels(axins.get_xticks(), fontsize=12)
        axins.set_title('Structural Conn', fontsize=12)
        axins.set_yticks([])

    for key in ('CC', 'MI', 'GC', 'TE'):
        for conn_key, linestyle in zip(('hist_conn', 'hist_disconn'), ('-', ':')):
            ax[0].plot(data_fig['edges'][key], data_fig[conn_key][key], ls=linestyle, **line_rc[key])
        ax[0].axvline(data_fig['th_kmeans'][key], color=line_rc[key]['color'], ls='--')
        fpr, tpr = data_fig['roc_gt'][key]
        label = line_rc[key]['label'] + f" : {data_fig['auc_kmeans'][key]:.2f}"
        ax[1].plot(fpr, tpr, lw=line_rc[key]['lw'], color=line_rc[key]['color'], label=label)[0].set_clip_on(False)
    ax[0].set_ylabel('Probability distribution')
    ax[0].set_xlabel('Causal value')
    ax[0].set_ylim(0)
    if hist_range is None:
        hist_range = (data_fig['edges']['TE'][0], data_fig['edges']['TE'][-1])
    else:
        format_xticks(ax[0], hist_range)
    ax[1].legend()
    ax[1]=roc_formatter(ax[1])
    plt.tight_layout()

    # fig.savefig('image/'+f"histogram_of_causal_with_conn_mask_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")
    return fig

def hist_dp(data:pd.DataFrame, **kwargs)->None:
    """Histogram of Delta pm and dp without masking.

    Args:
        data (pd.DataFrame): causality data or matched causality data

    """

    fig, ax = plt.subplots(1,1,figsize=(5,4))
    if 'bins' not in kwargs:
        kwargs['bins'] = 100
    if 'log-dp' in data:
        sns.histplot(data['log-dp'], kde=True, stat='density', ec='none', ax=ax, label=r'$\Delta p_m$',**kwargs)
    else:
        sns.histplot(np.log10(data['Delta_p']), kde=True, stat='density', ec='none', ax=ax, label=r'$\Delta p_m$',**kwargs)
    sns.histplot(np.log10(data['dp1']), kde=True, stat='density', ec='none', ax=ax, label=r'$\delta p$',**kwargs)
    ax.legend()
    ax.set_ylim(0)
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.set_xlabel(r'$\Delta p_m$ or $\delta_p$ values')
    ax.set_ylabel('probability density')
    ax.axvline(0, ls='--', color='r')
    plt.tight_layout()

    #    fig.savefig('image/'+f"histogram_of_dp_Delta_p_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")
    return fig

def ReconstructionFigure(
    data, sc_hist=False, causal_hist_with_gt=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,7))

    ax_hist = inset_axes(ax, width="100%", height="100%",
                bbox_to_anchor=(.30, .20, .60, .40),
                bbox_transform=ax.transAxes, loc='center', 
                axes_kwargs={'facecolor':[1,1,1,0]})
    ax_hist.spines['left'].set_visible(False)
    ax_hist.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax_hist.xaxis.set_major_formatter(sci_formatter)
    # plot the distribution of connectivity weight if exist
    if sc_hist and 'conn' in data['hist'] and hasattr(data['hist']['conn'], '__len__'):
        ax_conn = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(.05, .40, .45, .3),
                    bbox_transform=ax.transAxes, loc='center',)
        ax_conn.spines['left'].set_visible(False)

        edges = data['edges']['conn']
        counts = data['hist']['conn']
        ax_conn.bar(edges, counts, width=edges[1]-edges[0])
        ax_conn.set_title('Structural Conn', fontsize=12)
        ax_conn.set_yticks([])
    else:
        ax_conn = None
    if causal_hist_with_gt:
        th_key = 'th_kmeans'
        auc_key = 'auc_kmeans'
        acc_key = 'acc_kmeans'
        ppv_key = 'ppv_kmeans'
        roc_key = 'roc_gt'
    else:
        th_key = 'th_gauss'
        auc_key = 'auc_gauss'
        acc_key = 'acc_gauss'
        ppv_key = 'ppv_gauss'
        roc_key = 'roc_blind'
    for key in ('CC', 'MI', 'GC', 'TE'):
        edges = data['edges'][key]

        if data[th_key][key] not in (np.nan, None):
            ax_hist.axvline(data[th_key][key], ymax=0.9, lw=1, color=line_rc[key]['color'])

        fpr, tpr = data[roc_key][key]
        auc = data[auc_key][key]
        label=line_rc[key]['label'] + f" : {auc:.3f}"
        ax.plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=label)[0].set_clip_on(False)

        if causal_hist_with_gt:
            # KMeans clustering causal values
            for linestyle, hist_key in zip(('-', ':'), ('hist_conn', 'hist_disconn')):
                counts = data[hist_key][key]
                ax_hist.plot(edges, counts, ls=linestyle, **line_rc[key])
        else:
            # Double Gaussian Anaylsis
            ax_hist.plot(edges, data['hist'][key], **line_rc[key])

            # plot double Gaussian based ROC
            popt = data['log_norm_fit_pval'][key]
            # print(popt)
            if not hasattr(popt, '__len__'):
                popt = None
            if popt is not None:
                ax_hist.plot(edges, Gaussian(edges, popt[1], popt[3])*(1-popt[0]), '-.', lw=1, color=line_rc[key]['color'])
                ax_hist.plot(edges, Gaussian(edges, popt[2], popt[4]) * (popt[0]), '-.', lw=1, color=line_rc[key]['color'])

    # print acc, ppv
    if isinstance(data, dict):  # data as dict of dict
        acc_mean = np.nanmean(list(data[acc_key].values()))
        ppv_mean = np.nanmean(list(data[ppv_key].values()))
    else:   # data as pandas dataframe
        acc_mean = np.nanmean(data[acc_key].values)
        ppv_mean = np.nanmean(data[ppv_key].values)
    texts = []
    texts.append(ax.text(0.15, 0.90, f"acc={acc_mean*100:.2f}%", fontsize=18, transform=ax.transAxes))
    texts.append(ax.text(0.15, 0.83, f"ppv={ppv_mean*100:.2f}%", fontsize=18, transform=ax.transAxes))

    format_xticks(ax_hist, (edges[0], edges[-1]))
    ax_hist.set_ylim(0)
    ax_hist.set_yticks([])
    ax_hist.set_xlabel('Causal value')
    ax=roc_formatter(ax)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25)
    ax.set_xlabel(ax.get_xlabel(), fontsize=25)
    ax.legend(loc='upper left', bbox_to_anchor=(0.55, 0.95), fontsize=14)

    plt.tight_layout()
    return ax, ax_hist, ax_conn, texts

def reconstruction_illustration(fig_data:pd.DataFrame):
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    ReconstructionFigure(fig_data, sc_hist=True, causal_hist_with_gt=True,  ax=ax[0])
    ReconstructionFigure(fig_data, sc_hist=True, causal_hist_with_gt=False, ax=ax[1])
    return fig

def plot_raster(pm_causal:dict, xrange:tuple=(0,1000), return_spk:bool=False, ax=None, **kwargs):
    """plot sample raster plot given network parameters

    Args:
        pm_causal (dict): paramter dictionary
        xrange (tuple, optional): range of xaxis. Defaults to (0,1000).
        return_spk (bool, optional): return spike data. Defaults to False.
        ax (optional): matplotlib axis. Defaults to None.

    Returns:
        matplotlib.Figure: figure containing raster plot
    """
    spk_file_path = pm_causal['path'] + pm_causal['spk_fname'] + '_spike_train.dat'
    spk_data = load_spike_data(spk_file_path, xrange)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,3))
    else:
        fig = ax.get_figure()
    ax.plot(spk_data[:, 0], spk_data[:, 1], '|', **kwargs)
    ax.set_xlim(*xrange)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuronal Indices')
    plt.tight_layout()
    fig.savefig('image/'+f"raster_{pm_causal['spk_fname']:s}.pdf")
    if return_spk:
        return fig, spk_data
    else:
        return fig