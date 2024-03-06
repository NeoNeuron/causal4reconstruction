"""
    Library for plotting function used in analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .Causality import CausalityIO, CausalityAPI
from .figrc import line_rc, roc_formatter
from .utils import Gaussian, Double_Gaussian_Analysis
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
from sklearn.metrics import roc_auc_score, roc_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, MaxNLocator
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
import struct
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

def kmeans_1d(X:np.ndarray, init='kmeans++', return_label:bool=False):
    """Perform kmeans clustering on 1d data.

    Args:
        X (np.ndarray): data to cluster, 1d.
        return_label (bool, optional): _description_. Defaults to False.

    Returns:
        threshold (float): clustering boundary of 1d data X.
        labels (np.ndarray, optional): kmeans label.
    """
    kmeans = KMeans(n_clusters=2, max_iter=1, init=init).fit(X.reshape(-1,1))
    km_center_order = np.argsort(kmeans.cluster_centers_.flatten())
    threshold = 0.5*(X[kmeans.labels_==km_center_order[0]].max() + X[kmeans.labels_==km_center_order[1]].min())
    if return_label:
        return threshold, kmeans.labels_
    else:
        return threshold

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


def hist_causal_with_conn_mask_linear(pm_causal:dict, hist_range:tuple=None)->None:
    """Histogram of causality masked by ground truth.

    Args:
        pm_causal (dict): dict of causality parameters.
        hist_range (tuple): range of the histogram. Default None.
    """

    N = pm_causal['Ne']+pm_causal['Ni']
    fname_conn = get_conn_fpath(pm_causal)
    conn = np.fromfile(fname_conn, dtype=float
        )[:int(N*N)].reshape(N,N).astype(bool)
    fname = pm_causal['fname']
    causal_dtype = fname.split('=')[0][:-1]
    cau = CausalityIO(dtype=causal_dtype, N=(pm_causal['Ne'], pm_causal['Ni']), **pm_causal)

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ths = cau.load_from_single(fname, 'th')
    print(cau.load_from_single(fname, 'acc'))
    inf_mask = ~np.eye(N, dtype=bool)

    conn_weight = np.fromfile(fname_conn, dtype=float)[int(N*N):]
    if conn_weight.shape[0] > 0:
        conn_weight= conn_weight.reshape(N,N)
        axins = inset_axes(ax[0], width="100%", height="100%",
                        bbox_to_anchor=(.05, .55, .4, .3),
                        bbox_transform=ax[0].transAxes, loc='center')
        axins.spines['left'].set_visible(False)

        zero_mask = conn_weight>0
        if 'LN-' in causal_dtype:
            axins.hist(np.log10(conn_weight[inf_mask*zero_mask]), bins=40)
            axins.xaxis.set_major_formatter(sci_formatter)
        else:
            axins.hist(conn_weight[inf_mask*zero_mask], bins=40)
        axins.set_xticklabels(axins.get_xticks(), fontsize=12)
        axins.set_title('Structural Conn', fontsize=12)
        axins.set_yticks([])

    if hist_range is None:
        # determine histogram value range
        data_buff = cau.load_from_single(fname, 'CC')
        hist_range = (0, data_buff.max())

    ratio = np.sum(conn[inf_mask])/np.sum(inf_mask)
    for counter, key in enumerate(('CC',)):# 'MI', 'GC', 'TE')):
        data_buff = cau.load_from_single(fname, key)
        if np.any(data_buff<0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<0] = False
        log_TE = data_buff
        if key in ('TE', 'MI'):
            log_TE *= 2
        for mask, linestyle, ratio_ in zip((conn, ~conn), ('-', ':'), (ratio, 1-ratio)):
            log_TE_buffer = log_TE[mask*inf_mask]
            # log_TE_buffer = log_TE_buffer[~np.isinf(log_TE_buffer)*~np.isnan(log_TE_buffer)]
            counts, bins = np.histogram(log_TE_buffer, bins=100, range=hist_range,density=True)
            ax[0].semilogy(bins[:-1], counts*ratio_, ls=linestyle, **line_rc[key])
        ax[0].axvline(ths[counter], color=line_rc[key]['color'], ls='--')
        fpr, tpr, _ = roc_curve(conn[inf_mask], log_TE[inf_mask])
        label = line_rc[key]['label'] + f" : {roc_auc_score(conn[inf_mask], log_TE[inf_mask]):.2f}"
        ax[1].plot(fpr, tpr, lw=line_rc[key]['lw'], color=line_rc[key]['color'], label=label)[0].set_clip_on(False)
    ax[0].set_ylabel('Probability distribution')
    ax[0].set_xlabel('Causal value')
    ax[0].set_ylim(0)
    ax[0].ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    ax[1].legend()
    ax[1]=roc_formatter(ax[1])
    plt.tight_layout()

    fig.savefig('image/'+f"histogram_of_causal_with_conn_mask_linear_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")

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


def hist_causal(pm_causal:dict, hist_range:tuple=None)->None:
    """Histogram of causality without masking.

    Args:
        pm_causal (dict): dict of causality parameters.
        hist_range (tuple): range of the histogram. Default None.
    """

    N = pm_causal['Ne']+pm_causal['Ni']
    fname_conn = get_conn_fpath(pm_causal)
    conn = np.fromfile(fname_conn, dtype=float
        )[:int(N*N)].reshape(N,N).astype(bool)
    fname = pm_causal['fname']
    causal_dtype = fname.split('=')[0][:-1]
    cau = CausalityIO(dtype=causal_dtype, N=(pm_causal['Ne'], pm_causal['Ni']), **pm_causal)

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ths = cau.load_from_single(fname, 'th')
    print(cau.load_from_single(fname, 'acc'))

    inf_mask = ~np.eye(N, dtype=bool)
    if hist_range is None:
        # determine histogram value range
        data_buff = cau.load_from_single(fname, 'TE')
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        data_buff = np.log10(data_buff[inf_mask])
        hist_range = (np.floor(data_buff.min())+1, np.ceil(data_buff.max())+1)

    for counter, key in enumerate(('CC', 'MI', 'GC', 'TE')):
        data_buff = cau.load_from_single(fname, key)
        log_TE = np.log10(data_buff)
        if key in ('TE', 'MI'):
            log_TE += np.log10(2)
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        log_TE_buffer = log_TE[inf_mask]
        # log_TE_buffer = log_TE_buffer[~np.isinf(log_TE_buffer)*~np.isnan(log_TE_buffer)]
        counts, bins = np.histogram(log_TE_buffer, bins=100, range=hist_range,density=True)
        ax[0].plot(bins[:-1], counts, **line_rc[key])
        ax[0].axvline(np.log10(ths[counter]), color=line_rc[key]['color'], ls='--')
        fpr, tpr, _ = roc_curve(conn[inf_mask], log_TE[inf_mask])
        label = line_rc[key]['label'] + f" : {roc_auc_score(conn[inf_mask], log_TE[inf_mask]):.2f}"
        ax[1].plot(fpr, tpr, lw=line_rc[key]['lw'], color=line_rc[key]['color'], label=label)[0].set_clip_on(False)
    ax[0].set_ylabel('Probability distribution')
    ax[0].set_xlabel('Causal value')
    ax[0].set_ylim(0)
    format_xticks(ax[0], hist_range)
    ax[1].legend()
    ax[1]=roc_formatter(ax[1])
    plt.tight_layout()

    fig.savefig('image/'+f"histogram_of_causal_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")

def hist_causal_linear(pm_causal:dict, hist_range:tuple=None)->None:
    """Histogram of causality without masking.

    Args:
        pm_causal (dict): dict of causality parameters.
        hist_range (tuple): range of the histogram. Default None.
    """

    N = pm_causal['Ne']+pm_causal['Ni']
    fname_conn = get_conn_fpath(pm_causal)
    conn = np.fromfile(fname_conn, dtype=float
        )[:int(N*N)].reshape(N,N).astype(bool)
    fname = pm_causal['fname']
    causal_dtype = fname.split('=')[0][:-1]
    cau = CausalityIO(dtype=causal_dtype, N=(pm_causal['Ne'], pm_causal['Ni']), **pm_causal)

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ths = cau.load_from_single(fname, 'th')
    print(cau.load_from_single(fname, 'acc'))

    inf_mask = ~np.eye(N, dtype=bool)
    if hist_range is None:
        # determine histogram value range
        data_buff = cau.load_from_single(fname, 'TE')[inf_mask]
        hist_range = (0, data_buff.max())

    for counter, key in enumerate(('CC', 'MI', 'GC', 'TE')):
        data_buff = cau.load_from_single(fname, key)
        log_TE = data_buff
        if key in ('TE', 'MI'):
            log_TE *= 2
        log_TE_buffer = log_TE[inf_mask]
        # log_TE_buffer = log_TE_buffer[~np.isinf(log_TE_buffer)*~np.isnan(log_TE_buffer)]
        counts, bins = np.histogram(log_TE_buffer, bins=100, range=hist_range,density=True)
        ax[0].plot(bins[:-1], counts, **line_rc[key])
        ax[0].axvline(ths[counter], color=line_rc[key]['color'], ls='--')
        fpr, tpr, _ = roc_curve(conn[inf_mask], log_TE[inf_mask])
        label = line_rc[key]['label'] + f" : {roc_auc_score(conn[inf_mask], log_TE[inf_mask]):.2f}"
        ax[1].plot(fpr, tpr, lw=line_rc[key]['lw'], color=line_rc[key]['color'], label=label)[0].set_clip_on(False)
    ax[0].set_ylabel('Probability distribution')
    ax[0].set_xlabel('Causal value')
    ax[0].set_ylim(0)
    ax[0].ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    ax[1].legend()
    ax[1]=roc_formatter(ax[1])
    plt.tight_layout()

    fig.savefig('image/'+f"histogram_of_causal_linear_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")

def hist_causal_blind(pm_causal, hist_range:tuple=None, fit_p0 = None, return_data=False):
    # ====================
    # Draw histogram of causal values
    # ====================
    N = int(pm_causal['Ne'] + pm_causal['Ni'])
    fname = pm_causal['fname']
    cau = CausalityAPI(dtype=pm_causal['path_input'].split('/')[-3], N=N, **pm_causal)
    fig, ax = plt.subplots(1,1, figsize=(8,7))
    causal_values = {key: np.zeros(N*N-N) for key in ('TE', 'GC', 'MI', 'CC')}
    th_conditions = {}
    fig_data = {'roc':{}, 'hist': {}, 'hist_inconsist': {}}
    log_norm_fit_pval = {}

    inf_mask = ~np.eye(N, dtype=bool)
    if hist_range is None:
        # determine histogram value range
        data_buff = cau.load_from_single(fname, 'TE')
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        data_buff = np.log10(data_buff[inf_mask])
        hist_range = (np.floor(data_buff.min())+1, np.ceil(data_buff.max())+1)

    ax_hist = inset_axes(ax, width="100%", height="100%",
                bbox_to_anchor=(.20, .20, .80, .40),
                bbox_transform=ax.transAxes, loc='center', axes_kwargs={'facecolor':[1,1,1,0]})
    ax_hist.spines['left'].set_visible(False)
    ax_hist.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax_hist.xaxis.set_major_formatter(sci_formatter)
    fname_conn = pm_causal['path_output']+f"EE/N={N:d}/" + pm_causal['con_mat']
    conn = np.fromfile(fname_conn, dtype=float
    )[:int(N*N)].reshape(N,N).astype(bool)

    # plot the distribution of connectivity weight
    conn_weight = np.fromfile(fname_conn, dtype=float)[int(N*N):]
    if conn_weight.shape[0] > 0:
        conn_weight= conn_weight.reshape(N,N)
        axins = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(.05, .40, .45, .3),
                    bbox_transform=ax.transAxes, loc='center',)
        axins.spines['left'].set_visible(False)

        zero_mask = conn_weight>0
        axins.hist(conn_weight[inf_mask*zero_mask], bins=40)
        axins.set_title('Structural Conn', fontsize=12)
        axins.set_yticks([])

    data_var = cau.load_from_single(fname, 'Delta_p')
    data_var = np.log10(np.abs(data_var))
    data_var_mask = (~np.isinf(data_var))*(~np.isnan(data_var))*((data_var<0)+(data_var>0))
    acc = ppv = 0.
    for key in ('CC', 'MI', 'GC', 'TE'):
        data_buff = cau.load_from_single(fname, key)
        causal_values[key] = data_buff[~np.eye(data_buff.shape[0], dtype=bool)].flatten()
        log_cau = np.log10(data_buff)
        if key in ('TE', 'MI'):
            log_cau += np.log10(2)
        counts, bins = np.histogram(log_cau[data_var_mask], bins=100, range=hist_range, density=True)
        fig_data['hist'][key] = counts.copy()
        if fit_p0 is None:
            popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts, bins)
        else:
            popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts, bins, p0=fit_p0)
        # ax_hist.plot(bins[:-1], Double_Gaussian(bins[:-1], *popt), lw=4, ls=':', color=line_rc[key]['color'], )
        ax_hist.plot(bins[:-1], Gaussian(bins[:-1], popt[0], popt[2], popt[4], ), '-.', lw=1, color=line_rc[key]['color'], )
        ax_hist.plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3], popt[5], ), '-.', lw=1, color=line_rc[key]['color'], )
        ax_hist.plot(bins[:-1], counts, lw=line_rc[key]['lw'], color=line_rc[key]['color'], label=line_rc[key]['label'])
        ax_hist.axvline(threshold, ymax=0.75, color=line_rc[key]['color'], ls='--')
        # plot double Gaussian based ROC
        ax.plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=line_rc[key]['label'])[0].set_clip_on(False)
        print(f"{key:s}: {-np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2):.3f}", end='\t')

        log_norm_fit_pval[key] = popt
        th_conditions[key] = threshold
        fig_data['roc'][key] = np.vstack((fpr, tpr))

        # calculate reconstruction accuracy
        conn_recon = log_cau >= threshold
        recon = np.logical_xor(conn[inf_mask], conn_recon[inf_mask])
        acc += 100-100*recon.sum()/len(recon)
        ppv += 100*(conn_recon[inf_mask]*conn[inf_mask]).sum()/conn_recon[inf_mask].sum()

    # print acc, 
    ax.text(0.6, 0.90, f"acc={acc/4.:.2f}%", fontsize=18, transform=ax.transAxes)
    ax.text(0.6, 0.83, f"ppv={ppv/4.:.2f}%", fontsize=18, transform=ax.transAxes)

    format_xticks(ax_hist, hist_range)
    print('')
    ax_hist.set_ylim(0)
    ax.legend(loc='upper left', fontsize=14)

    ax_hist.set_yticks([])
    ax_hist.set_xlabel('Causal value')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_ylabel('True Positive Rate', fontsize=30)
    ax.set_xlabel('False Positive Rate', fontsize=30)

    fig_data['edges'] = bins[:-1].copy()
    fig_data['log_norm_fit_pval'] = log_norm_fit_pval
    fig_data['opt_th'] = th_conditions

    plt.tight_layout()
    fig.savefig(f"image/histogram_of_causal_blind_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")
    if return_data:
        return fig_data

def _ReconstructionAnalysis(pm_causal, hist_range:tuple=None,
                            fit_p0 = None, EI_mask=None, nbins=100):
    # ====================
    # Histogram of causal values
    # ====================
    N = int(pm_causal['Ne']), int(pm_causal['Ni'])
    fname = pm_causal['fname']
    cau = CausalityAPI(dtype=pm_causal['path_input'].split('/')[-3], N=N, **pm_causal)
    fig_data_keys = ['raw_data', 'roc_gt', 'roc_blind', 
        'hist', 'hist_conn', 'hist_disconn', 'hist_error', 'edges', 
        'log_norm_fit_pval', 'th_gauss', 'th_kmeans', 
        'acc_gauss', 'acc_kmeans', 'ppv_gauss', 'ppv_kmeans',
        'auc_gauss', 'auc_kmeans',
    ]
    data = {key: {} for key in fig_data_keys}

    N = N[0]+N[1]
    # load causal data to determine the shape of mask
    data_buff = cau.load_from_single(fname, 'TE')
    if len(data_buff.shape) == 1:
        inf_mask = ~np.isinf(data_buff)
    elif len(data_buff.shape) == 2:
        inf_mask = ~np.eye(N, dtype=bool)
        if EI_mask == 'E':
            inf_mask[int(pm_causal['Ne']):, :]=False
        elif EI_mask == 'I':
            inf_mask[:int(pm_causal['Ne']), :]=False
    else:
        raise ValueError('data shape error!')

    if hist_range is None:
        # determine histogram value range
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        data_buff = np.log10(data_buff[inf_mask])
        hist_range = (np.floor(data_buff.min())+1, np.ceil(data_buff.max())+1)

    # load connectivity matrix 
    if pm_causal['Ne'] * pm_causal['Ni'] > 0:
        neuron_type = 'EI'
    elif pm_causal['Ne'] > 0:
        neuron_type = 'EE'
    elif pm_causal['Ni'] > 0:
        neuron_type = 'II'
    else:
        raise ValueError('Neuron number error!')
    fname_conn = pm_causal['path_output']+f"{neuron_type:s}/N={N:d}/" + pm_causal['con_mat']
    conn_raw = np.fromfile(fname_conn, dtype=float)
    if conn_raw.shape[0] >= N*N:
        print('>> Load dense matrix:')
        conn = conn_raw[:int(N*N)].reshape(N,N).astype(bool)
        if conn_raw.shape[0] > N*N:
            conn_weight = conn_raw[int(N*N):].reshape(N,N)
        else:
            conn_weight = None
    else:
        print('>> Load sparse matrix:')
        conn_raw = conn_raw.reshape(2,-1)
        conn = np.zeros((N,N), dtype=bool)
        conn[conn_raw[0].astype(int), conn_raw[1].astype(int)] = True
        conn_weight = None

    if 'mask_file' in pm_causal:
        mask = np.fromfile(pm_causal['path_output']+f"EE/N={N:d}/"+pm_causal['mask_file'], dtype=float).reshape(2,-1)
        mask = mask.astype(int)
        conn = conn[mask[0], mask[1]]

    ratio = np.sum(conn[inf_mask])/np.sum(inf_mask)
    # plot the distribution of connectivity weight if exist
    if conn_weight is not None:
        zero_mask = conn_weight>0
        if 'LN' in fname_conn:
            counts, bins = np.histogram(
                np.log10(conn_weight[inf_mask*zero_mask]), bins=40)
        else:
            counts, bins = np.histogram(
                conn_weight[inf_mask*zero_mask], bins=40)
        for key in data.keys():
            if key == 'hist':
                data[key]['conn']  = counts.copy()
            elif key == 'edges':
                data[key]['conn'] = bins[:-1].copy()
            elif key == 'raw_data':
                data[key]['conn'] = conn_weight.copy()
            else:
                data[key]['conn'] = np.nan
    else:
        for key in data.keys():
            if key == 'raw_data':
                data[key]['conn'] = conn.astype(float)
            else:
                data[key]['conn'] = np.nan

    data_dp_raw = cau.load_from_single(fname, 'Delta_p')
    data_dp = np.log10(np.abs(data_dp_raw))
    nan_mask = (~np.isinf(data_dp))*(~np.isnan(data_dp))
    counts, bins = np.histogram(data_dp[inf_mask*nan_mask], bins=100, density=True)
    for key in data.keys():
        if key == 'hist':
            data[key]['dp']  = counts.copy()
        elif key == 'edges':
            data[key]['dp'] = bins[:-1].copy()
        elif key == 'raw_data':
            data[key]['dp'] = data_dp_raw.copy()
        else:
            data[key]['dp'] = np.nan
    for key in ('CC', 'MI', 'GC', 'TE'):
        data_buff = cau.load_from_single(fname, key)
        log_cau = np.log10(data_buff)
        data['raw_data'][key] = log_cau.copy()
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        if key in ('TE', 'MI'):
            log_cau += np.log10(2)
        counts_total = np.zeros(nbins)
        nan_mask = (~np.isinf(log_cau))*(~np.isnan(log_cau))
        for mask, ratio_, hist_key in zip((conn, ~conn), (ratio, 1-ratio), ('hist_conn', 'hist_disconn')):
            log_TE_buffer = log_cau[mask*inf_mask*nan_mask]
            counts, bins = np.histogram(log_TE_buffer, bins=nbins, range=hist_range,density=True)
            counts_total += counts*ratio_
            data[hist_key][key] = counts*ratio_
        data['edges'][key] = bins[:-1].copy()
        data['hist'][key] = counts_total.copy()

        # KMeans clustering causal values
        if fit_p0 is None:
            disconn_peak_id = data['hist_disconn'][key].argmax()
            conn_peak_id = data['hist_conn'][key].argmax()
            fit_p0 = [0.5, 0.5, bins[disconn_peak_id], bins[conn_peak_id], 1, 1]
        # th_idx = dict(TE=0, GC=1, MI=2, CC=3)
        # th_km = np.log10(cau.load_from_single(fname, 'th')[th_idx[key]])
        # TODO: Fix inaccurate double peak separation for some cases.
        try:
            th_km = kmeans_1d(log_cau[inf_mask*nan_mask], np.array([[fit_p0[2]],[fit_p0[3]]]))
            data['th_kmeans'][key] = th_km
            conn_recon = log_cau >= th_km
            error_mask = np.logical_xor(conn, conn_recon)
            data['acc_kmeans'][key] = 1-error_mask[inf_mask].sum()/len(error_mask[inf_mask])
            data['ppv_kmeans'][key] = (conn_recon[inf_mask]*conn[inf_mask]).sum()/conn_recon[inf_mask].sum()
        except:
            data['th_kmeans'][key] = np.nan
            data['acc_kmeans'][key] = np.nan
            data['ppv_kmeans'][key] = np.nan

        # Double Gaussian Anaylsis
        try:
            popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts_total, bins, p0=fit_p0)
            data['th_gauss'][key] = threshold
            data['log_norm_fit_pval'][key] = popt
            data['roc_blind'][key] = np.vstack((fpr, tpr))
        except:
            popt = None
            data['th_gauss'][key] = np.nan
            data['log_norm_fit_pval'][key] = np.nan
            data['roc_blind'][key] = np.nan
        if popt is not None:
            # plot double Gaussian based ROC
            auc = -np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)
            data['auc_gauss'][key] = auc

            # calculate reconstruction accuracy
            conn_recon = log_cau >= threshold
            error_mask = np.logical_xor(conn, conn_recon)
            data['acc_gauss'][key] = 1-error_mask[inf_mask].sum()/len(error_mask[inf_mask])
            data['ppv_gauss'][key] = (conn_recon[inf_mask]*conn[inf_mask]).sum()/conn_recon[inf_mask].sum()
            counts_error, _ = np.histogram(log_cau[error_mask*inf_mask], bins=nbins, range=hist_range,density=True)
            data['hist_error'][key] = counts_error.copy()
        else:
            data['acc_gauss'][key] = np.nan
            data['ppv_gauss'][key] = np.nan
            data['hist_error'][key] = np.nan

        fpr, tpr, _ = roc_curve(conn[inf_mask*nan_mask], log_cau[inf_mask*nan_mask])
        data['roc_gt'][key] = np.vstack((fpr, tpr))
        auc = roc_auc_score(conn[inf_mask*nan_mask], log_cau[inf_mask*nan_mask])
        data['auc_kmeans'][key] = auc
    return pd.DataFrame(data)


# match cell-type, connectivity, masking to causality data
def match_features(data:pd.DataFrame, N:int, conn_file:str,
        Ni:int=0, EI_types:np.ndarray=None, mask_file:str=None):
    """match causality data other network features, 

    Args:
        data (pd.DataFrame): causality data.
        N (int): number of (excitatory) neurons.
        conn_file (str): filename of connectivity matrix, both dense and sparse matrix supported.
        Ni (int, optional): number of inhibitory neurons, if nonzero, the total number of neuron N+Ni.
            Defaults to 0.
        EI_types (dict, optional): {'neuron_id':[...], 'EI_type':[...]}. Defaults to None.
        mask_file (str, optional): filename of masking used in causality calculation. Defaults to None.

    Returns:
        data_match: DataFrame of integrated data.
    """
    
    data_match = data.copy()
    
    # Transform causality column in data to log10-scale
    new_columns_map = {
        'log-TE': 'TE',
        'log-GC': 'GC',
        'log-MI': 'sum(MI)',
        'log-CC': 'sum(CC2)',
        'log-dp': 'Delta_p',
    }
    for key, val in new_columns_map.items():
        data_match[key] = np.log10(np.abs(data_match[val]))
        if key in ('log-TE', 'log-MI'):
            data_match[key] += np.log10(2)

    N = N + Ni

    # match causality mask
    if mask_file is not None:
        mask = np.fromfile(mask_file, dtype=float).reshape(2,-1)
        mask_pre_id, mask_post_id = mask.astype(int)
        mask = pd.DataFrame({'pre_id':mask_pre_id, 'post_id':mask_post_id}).astype(int)
        # Select rows in data according to mask
        data_match = data_match.merge(mask, how='inner', on=['pre_id', 'post_id'])

    # match EI types
    if EI_types is None:
        data_match['pre_cell_type'] = ['E' if ele else 'I' for ele in data_match['pre_id'] < N-Ni]
        data_match['post_cell_type'] = ['E' if ele else 'I' for ele in data_match['post_id'] < N-Ni]
    else:
        assert isinstance(EI_types, dict), 'EI_types should be a dict!'
        df_tmp = pd.DataFrame({'pre_id':EI_types['neuron_id'], 'pre_cell_type':EI_types['EI_type']})
        data_match = data_match.merge(df_tmp, how='left', on='pre_id')
        df_tmp = pd.DataFrame({'post_id':EI_types['neuron_id'], 'post_cell_type':EI_types['EI_type']})
        data_match = data_match.merge(df_tmp, how='left', on='post_id')

    # load connectivity matrix 
    conn_raw = np.fromfile(conn_file, dtype=float)
    conn_weight = None
    if conn_raw.shape[0] >= N*N:
        print('>> Load dense matrix:')
        conn = conn_raw[:int(N*N)].reshape(N,N).astype(bool)
        pre_id, post_id = np.where(conn)

        if conn_raw.shape[0] > N*N:
            conn_weight = conn_raw[int(N*N):].astype(float)
    else:
        print('>> Load sparse matrix:')
        conn_raw = conn_raw.reshape(2,-1)
        pre_id, post_id = conn_raw

    conn_pairs = pd.DataFrame(
        {'pre_id':pre_id, 'post_id':post_id, 'connection':np.ones_like(pre_id, dtype=int)})
    # Merge conn_pairs with data
    data_match = data_match.merge(conn_pairs, how='left', on=['pre_id', 'post_id'])
    # Fill missing rows in 'connection' column with 0
    data_match['connection'].fillna(0, inplace=True)

    # merge connection strength if exist
    if conn_weight is not None:
        weight_pairs = pd.DataFrame(
            {'pre_id':pre_id, 'post_id':post_id, 'weight':weight_pairs})
        data_match = data_match.merge(conn_pairs, how='left', on=['pre_id', 'post_id'])
        data_match['weight'].fillna(0, inplace=True)

    # drop inf and nan rows, mainly self-connections
    data_match.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_match.dropna(inplace=True)
    return data_match

def _ReconstructionAnalysis_new(data:pd.DataFrame,
        hist_range:tuple=None, nbins=100, fit_p0 = None, EI_mask:str=None, weight_hist_type:str='linear'):
    fig_data_keys = ['roc_gt', 'roc_blind', 
        'hist', 'hist_conn', 'hist_disconn', 'hist_error', 'edges', 
        'log_norm_fit_pval', 'th_gauss', 'th_kmeans', 
        'acc_gauss', 'acc_kmeans', 'ppv_gauss', 'ppv_kmeans',
        'auc_gauss', 'auc_kmeans',
    ]
    data_fig = {key: {} for key in fig_data_keys}
    data_recon = data.copy()
    
    if EI_mask is not None:
        if EI_mask == 'E':
            data_recon = data_recon[data_recon['pre_cell_type'].eq('E')]
        elif EI_mask == 'I':
            data_recon = data_recon[data_recon['pre_cell_type'].eq('E')]
        else:
            raise ValueError('EI_mask error!')

    # merge connection strength if exist
    if 'weight' in data_recon:
        if weight_hist_type == 'log':
            counts, bins = np.histogram(
                np.log10(data_recon[data_recon['connection'].eq(1)]), bins=40)
        elif weight_hist_type == 'linear':
            counts, bins = np.histogram(
                data_recon[data_recon['connection'].eq(1)], bins=40)
        else:
            raise ValueError('weight_hist_type error!')
        for key in data_fig.keys():
            if key == 'hist':
                data[key]['conn']  = counts.copy()
            elif key == 'edges':
                data[key]['conn'] = bins[:-1].copy()
            else:
                data[key]['conn'] = np.nan

    # drop inf and nan rows, mainly self-connections
    ratio = data_recon['connection'].mean()

    if hist_range is None:
        # determine histogram value range
        hist_range = (np.floor(data_recon[['log-TE', 'log-GC', 'log-MI', 'log-CC']].min().min()),
                      np.ceil(data_recon[['log-TE', 'log-GC', 'log-MI', 'log-CC']].max()).max())

    counts, bins = np.histogram(data_recon['log-dp'], bins=100, density=True)
    for key in data_fig.keys():
        if key == 'hist':
            data_fig[key]['dp']  = counts.copy()
        elif key == 'edges':
            data_fig[key]['dp'] = bins[:-1].copy()
        else:
            data_fig[key]['dp'] = np.nan

    for key in ('CC', 'MI', 'GC', 'TE'):
        counts_total = np.zeros(nbins)
        for i, hist_key in enumerate(('hist_disconn', 'hist_conn')):
            buffer = data_recon[data_recon['connection'].eq(i)][f'log-{key}']
            counts, bins = np.histogram(buffer, bins=nbins, range=hist_range, density=True)
            counts *= np.abs(1-i-ratio)
            counts_total += counts.copy()
            data_fig[hist_key][key] = counts.copy()
        data_fig['edges'][key] = bins[:-1].copy()
        data_fig['hist'][key] = counts_total.copy()

        # KMeans clustering causal values
        if fit_p0 is None:
            disconn_peak_id = data_fig['hist_disconn'][key].argmax()
            conn_peak_id = data_fig['hist_conn'][key].argmax()
            fit_p0 = [0.5, 0.5, bins[disconn_peak_id], bins[conn_peak_id], 1, 1]
        try:
            th_kmeans = kmeans_1d(data_recon[f'log-{key}'].to_numpy(), np.array([[fit_p0[2]],[fit_p0[3]]]))
            data_fig['th_kmeans'][key] = th_kmeans
            data_recon[f'recon-kmeans-{key}'] = (data_recon[f'log-{key}'] >= th_kmeans).astype(int)
            error_mask = np.logical_xor(data_recon['connection'], data_recon[f'recon-kmeans-{key}'])
            data_fig['acc_kmeans'][key] = 1-error_mask.sum()/len(error_mask)
            data_fig['ppv_kmeans'][key] = (data_recon['connection']*data_recon[f'recon-kmeans-{key}']).sum()/data_recon[f'recon-kmeans-{key}'].sum()
        except:
            print("Warning: KMeans clustering failed!")
            data_fig['th_kmeans'][key] = np.nan
            data_fig['acc_kmeans'][key] = np.nan
            data_fig['ppv_kmeans'][key] = np.nan

        # Double Gaussian Anaylsis
        try:
            popt, th_gauss, fpr, tpr = Double_Gaussian_Analysis(counts_total, bins, p0=fit_p0)
            data_fig['th_gauss'][key] = th_gauss
            data_fig['log_norm_fit_pval'][key] = popt
            data_fig['roc_blind'][key] = np.vstack((fpr, tpr))
        except:
            print("Warning: Double Gaussian Anaylsis failed!")
            popt = None
            data_fig['th_gauss'][key] = np.nan
            data_fig['log_norm_fit_pval'][key] = np.nan
            data_fig['roc_blind'][key] = np.nan
        if popt is not None:
            # plot double Gaussian based ROC
            auc = -np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)
            data_fig['auc_gauss'][key] = auc

            # calculate reconstruction accuracy
            data_recon[f'recon-gauss-{key}'] = (data_recon[f'log-{key}'] >= th_gauss).astype(int)
            error_mask = np.logical_xor(data_recon['connection'], data_recon[f'recon-gauss-{key}'])
            data_fig['acc_gauss'][key] = 1-error_mask.sum()/len(error_mask)
            data_fig['ppv_gauss'][key] = (data_recon['connection']*data_recon[f'recon-gauss-{key}']).sum()/data_recon[f'recon-gauss-{key}'].sum()
            counts_error, _ = np.histogram(data_recon[f'log-{key}'][error_mask], bins=nbins, range=hist_range, density=True)
            data_fig['hist_error'][key] = counts_error.copy()
        else:
            data_fig['acc_gauss'][key] = np.nan
            data_fig['ppv_gauss'][key] = np.nan
            data_fig['hist_error'][key] = np.nan

        fpr, tpr, _ = roc_curve(data_recon['connection'], data_recon[f'log-{key}'])
        data_fig['roc_gt'][key] = np.vstack((fpr, tpr))
        try:
            auc = roc_auc_score(data_recon['connection'], data_recon[f'log-{key}'])
        except ValueError:
            print("Warning: only one class, AUC calculation error!")
            auc = np.nan
        data_fig['auc_kmeans'][key] = auc

    # Reorder columns in data
    new_columns = ['pre_id', 'post_id', 'connection', 'recon-kmeans-TE', 'recon-gauss-TE', 'log-TE', 'log-GC', 'log-MI', 'log-CC', 'log-dp']
    other_columns = [col for col in data_recon.columns if col not in new_columns]
    data_recon = data_recon[new_columns + other_columns]
        
    return data_recon, pd.DataFrame(data_fig)

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
                ax_hist.plot(edges, Gaussian(edges, popt[0], popt[2], popt[4], ), '-.', lw=1, color=line_rc[key]['color'], )
                ax_hist.plot(edges, Gaussian(edges, popt[1], popt[3], popt[5], ), '-.', lw=1, color=line_rc[key]['color'], )

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

def ReconstructionAnalysis_new(
        data:pd.DataFrame, hist_range:tuple=None,
        fit_p0 = None, EI_mask=None, fig_toggle=True, nbins=100,
    ):
    data_recon, fig_data = _ReconstructionAnalysis_new(
        data, hist_range=hist_range, fit_p0=fit_p0, EI_mask=EI_mask, nbins=nbins)
    if fig_toggle:
        fig, ax = plt.subplots(1,2, figsize=(13,6))
        ReconstructionFigure(fig_data, sc_hist=True, causal_hist_with_gt=True,  ax=ax[0])
        ReconstructionFigure(fig_data, sc_hist=True, causal_hist_with_gt=False, ax=ax[1])
        
        return fig_data, fig
    else:
        return fig_data

def ReconstructionAnalysis(
        pm_causal, hist_range:tuple=None, fit_p0 = None,
        EI_mask=None, fig_toggle=True, nbins=100,
    ):
    fig_data = _ReconstructionAnalysis(pm_causal, hist_range, fit_p0, EI_mask, nbins)
    if fig_toggle:
        fig, ax = plt.subplots(1,2, figsize=(13,6))
        ReconstructionFigure(fig_data, sc_hist=True, causal_hist_with_gt=True,  ax=ax[0])
        ReconstructionFigure(fig_data, sc_hist=True, causal_hist_with_gt=False, ax=ax[1])
        
        return fig_data, fig
    else:
        return fig_data

def load_spike_data(spk_file_path:str, xrange:tuple=(0,1000), verbose=True, **kwargs):
    """plot sample raster plot given network parameters

    Args:
        spk_file_path (str): relative path of spike files
        xrange (tuple, optional): range of xaxis. Defaults to (0,1000).

    Returns:
        spike_data: np.ndarray, (num_spikes, 2)
    """
    n_time = 1000
    with open(spk_file_path, 'rb') as f:
        data_buff = f.read(8*2*n_time)
        data_len = int(len(data_buff)/16)
        spk_data = np.array(struct.unpack('d'*2*data_len, data_buff)).reshape(-1,2)
        while spk_data[-1,0] < xrange[1]:
            data_buff = f.read(8*2*n_time)
            data_len = int(len(data_buff)/16)
            if data_len == 0:
                print('reaching end of file')
                break
            spk_data_more = np.array(struct.unpack('d'*2*data_len, data_buff)).reshape(-1,2)
            spk_data = np.concatenate((spk_data, spk_data_more), axis=0)

    mask = (spk_data[:,0] > xrange[0]) * (spk_data[:,0] < xrange[1])
    if verbose:
        N = np.unique(spk_data[:,1]).shape[0]
        print(f"mean firing rate is {spk_data.shape[0]/spk_data[-1,0]/N*1000.:.3f} Hz")
    return spk_data[mask, :]

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