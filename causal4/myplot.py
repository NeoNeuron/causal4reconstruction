"""
    Library for plotting function used in analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
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

def hist_causal_with_conn_mask(pm_causal:dict, hist_range:tuple=None)->None:
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
        data_buff = cau.load_from_single(fname, 'TE')
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        data_buff = np.log10(data_buff[inf_mask])
        hist_range = (np.floor(data_buff.min())+1, np.ceil(data_buff.max())+1)

    ratio = np.sum(conn[inf_mask])/np.sum(inf_mask)
    for counter, key in enumerate(('CC', 'MI', 'GC', 'TE')):
        data_buff = cau.load_from_single(fname, key)
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        log_TE = np.log10(data_buff)
        if key in ('TE', 'MI'):
            log_TE += np.log10(2)
        for mask, linestyle, ratio_ in zip((conn, ~conn), ('-', ':'), (ratio, 1-ratio)):
            log_TE_buffer = log_TE[mask*inf_mask]
            # log_TE_buffer = log_TE_buffer[~np.isinf(log_TE_buffer)*~np.isnan(log_TE_buffer)]
            counts, bins = np.histogram(log_TE_buffer, bins=100, range=hist_range,density=True)
            ax[0].plot(bins[:-1], counts*ratio_, ls=linestyle, **line_rc[key])
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

    fig.savefig('image/'+f"histogram_of_causal_with_conn_mask_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")


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

def hist_dp(pm_causal:dict, hist_range:tuple=None)->None:
    """Histogram of Delta pm and dp without masking.

    Args:
        pm_causal (dict): dict of causality parameters.
    """

    N = pm_causal['Ne']+pm_causal['Ni']
    fname = pm_causal['fname']
    causal_dtype = fname.split('=')[0][:-1]
    cau = CausalityIO(dtype=causal_dtype, N=(pm_causal['Ne'], pm_causal['Ni']), **pm_causal)
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    inf_mask = ~np.eye(N, dtype=bool)
    for var, label in zip(('Delta_p', 'dp'), (r'$\Delta p_m$', r'$\delta p$')):
        data_var = cau.load_from_single(fname, var)
        data_var = np.log10(np.abs(data_var[inf_mask]))
        counts, bins = np.histogram(
            data_var[~np.isinf(data_var)*(~np.isnan(data_var))], 
            bins=100, density=True, range=hist_range)
        ax.plot(bins[:-1], counts, label=label)
        ax.legend()
    ax.set_ylim(0)
    if hist_range is not None:
        ax.set_xlim(*hist_range)
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.set_xlabel(r'$\Delta p_m$ or $\delta_p$ values')
    ax.set_ylabel('Probability density')
    ax.axvline(0, ls='--', color='r')
    plt.tight_layout()

    if 'T' in pm_causal:
        fig.savefig('image/'+f"histogram_of_dp_Delta_p_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")
    else:
        fig.savefig('image/'+f"histogram_of_dp_Delta_p_bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")


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
        ppv += 100*(conn_recon[inf_mask]*conn[inf_mask]).sum()/conn[inf_mask].sum()

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


def ReconstructionAnalysis(pm_causal, hist_range:tuple=None, fit_p0 = None):
    # ====================
    # Histogram of causal values
    # ====================
    N = int(pm_causal['Ne'] + pm_causal['Ni'])
    fname = pm_causal['fname']
    cau = CausalityAPI(dtype=pm_causal['path_input'].split('/')[-3], N=N, **pm_causal)
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    fig_data = {
        'raw_data':{},
        'roc_gt':{},
        'roc_blind':{},
        'hist': {},
        'hist_conn': {},
        'hist_disconn':{},
        'hist_error':{},
        'edges':{},
        'log_norm_fit_pval': {},
        'opt_th': {},
        'kmean_th': {},
        'acc_gauss': {},
        'acc_kmeans': {},
        'ppv_gauss': {},
        'ppv_kmeans': {},
    }

    inf_mask = ~np.eye(N, dtype=bool)
    if hist_range is None:
        # determine histogram value range
        data_buff = cau.load_from_single(fname, 'TE')
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        data_buff = np.log10(data_buff[inf_mask])
        hist_range = (np.floor(data_buff.min())+1, np.ceil(data_buff.max())+1)

    ax_hist = [inset_axes(axi, width="100%", height="100%",
                bbox_to_anchor=(.30, .20, .60, .40),
                bbox_transform=axi.transAxes, loc='center', 
                axes_kwargs={'facecolor':[1,1,1,0]}) for axi in ax]
    for axi in ax_hist:
        axi.spines['left'].set_visible(False)
        axi.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        axi.xaxis.set_major_formatter(sci_formatter)

    # load connectivity matrix 
    fname_conn = pm_causal['path_output']+f"EE/N={N:d}/" + pm_causal['con_mat']
    conn_raw = np.fromfile(fname_conn, dtype=float)
    if conn_raw.shape[0] >= N*N:
        conn = conn_raw[:int(N*N)].reshape(N,N).astype(bool)
        if conn_raw.shape[0] > N*N:
            conn_weight = conn_raw[int(N*N):].reshape(N,N)
        else:
            conn_weight = None
    else:
        conn_raw = conn_raw.reshape(3,-1)
        conn = np.zeros((N,N), dtype=bool)
        conn_weight = np.zeros((N,N), dtype=float)
        conn[conn_raw[0].astype(int), conn_raw[1].astype(int)] = True
        conn_weight[conn_raw[0].astype(int), conn_raw[1].astype(int)] = conn_raw[2]

    ratio = np.sum(conn[inf_mask])/np.sum(inf_mask)
    # plot the distribution of connectivity weight if exist
    if conn_weight is not None:
        conn_weight= conn_weight.reshape(N,N)
        axins = inset_axes(ax[0], width="100%", height="100%",
                    bbox_to_anchor=(.05, .40, .45, .3),
                    bbox_transform=ax[0].transAxes, loc='center',)
        axins.spines['left'].set_visible(False)

        zero_mask = conn_weight>0
        if 'LN' in fname_conn:
            counts, bins = np.histogram(
                np.log10(conn_weight[inf_mask*zero_mask]), bins=40)
        else:
            counts, bins = np.histogram(
                conn_weight[inf_mask*zero_mask], bins=40)
        axins.bar(bins[:-1], counts, width=bins[1]-bins[0])
        axins.set_title('Structural Conn', fontsize=12)
        axins.set_yticks([])
        for key in fig_data.keys():
            if key == 'hist':
                fig_data[key]['conn']  = counts.copy()
            elif key == 'edges':
                fig_data[key]['conn'] = bins[:-1].copy()
            elif key == 'raw_data':
                fig_data[key]['conn'] = conn_weight.copy()
            else:
                fig_data[key]['conn'] = np.nan
    else:
        for key in fig_data.keys():
            if key == 'raw_data':
                fig_data[key]['conn'] = conn.astype(float)
            else:
                fig_data[key]['conn'] = np.nan

    data_dp = cau.load_from_single(fname, 'Delta_p')
    data_dp = np.log10(np.abs(data_dp))
    nan_mask = (~np.isinf(data_dp))*(~np.isnan(data_dp))
    counts, bins = np.histogram(data_dp[inf_mask*nan_mask], bins=100, density=True)
    for key in fig_data.keys():
        if key == 'hist':
            fig_data[key]['dp']  = counts.copy()
        elif key == 'edges':
            fig_data[key]['dp'] = bins[:-1].copy()
        elif key == 'raw_data':
            fig_data[key]['dp'] = data_dp.copy()
        else:
            fig_data[key]['dp'] = np.nan
    for key in ('CC', 'MI', 'GC', 'TE'):
        data_buff = cau.load_from_single(fname, key)
        log_cau = np.log10(data_buff)
        fig_data['raw_data'][key] = log_cau.copy()
        if np.any(data_buff[inf_mask]<=0):
            print('WARNING: some negative entities occurs!')
            inf_mask[data_buff<=0] = False
        if key in ('TE', 'MI'):
            log_cau += np.log10(2)
        counts_total = np.zeros(100)
        nan_mask = (~np.isinf(log_cau))*(~np.isnan(log_cau))
        for mask, linestyle, ratio_, hist_key in zip((conn, ~conn), ('-', ':'), (ratio, 1-ratio), ('hist_conn', 'hist_disconn')):
            log_TE_buffer = log_cau[mask*inf_mask*nan_mask]
            counts, bins = np.histogram(log_TE_buffer, bins=100, range=hist_range,density=True)
            counts_total += counts*ratio_
            fig_data[hist_key][key] = counts*ratio_
            ax_hist[0].plot(bins[:-1], counts*ratio_, ls=linestyle, **line_rc[key])
        fig_data['edges'][key] = bins[:-1].copy()
        fig_data['hist'][key] = counts_total.copy()

        # KMeans clustering causal values
        if fit_p0 is None:
            fit_p0 = [0.5, 0.5, -7, -5, 1, 1]
        # th_idx = dict(TE=0, GC=1, MI=2, CC=3)
        # th_km = np.log10(cau.load_from_single(fname, 'th')[th_idx[key]])
        # TODO: Fix inaccurate double peak separation for some cases.
        th_km = kmeans_1d(log_cau[inf_mask*nan_mask], np.array([[fit_p0[2]],[fit_p0[3]]]))
        fig_data['kmean_th'][key] = th_km
        ax_hist[0].axvline(th_km, ymax=0.9, lw=1, color=line_rc[key]['color'])
        conn_recon = log_cau >= th_km
        error_mask = np.logical_xor(conn, conn_recon)
        fig_data['acc_kmeans'][key] = 1-error_mask[inf_mask].sum()/len(error_mask[inf_mask])
        fig_data['ppv_kmeans'][key] = (conn_recon[inf_mask]*conn[inf_mask]).sum()/conn[inf_mask].sum()

        # Double Gaussian Anaylsis
        ax_hist[1].plot(bins[:-1], counts_total, **line_rc[key])
        popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts_total, bins, p0=fit_p0)
        fig_data['opt_th'][key] = threshold
        fig_data['log_norm_fit_pval'][key] = popt
        fig_data['roc_blind'][key] = np.vstack((fpr, tpr))
        if popt is not None:
            ax_hist[1].plot(bins[:-1], Gaussian(bins[:-1], popt[0], popt[2], popt[4], ), '-.', lw=1, color=line_rc[key]['color'], )
            ax_hist[1].plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3], popt[5], ), '-.', lw=1, color=line_rc[key]['color'], )
            ax_hist[1].axvline(threshold, ymax=0.9, color=line_rc[key]['color'], ls='--')
            # plot double Gaussian based ROC
            label=line_rc[key]['label'] + f" : {-np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2):.3f}"
            ax[1].plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=label)[0].set_clip_on(False)

            # calculate reconstruction accuracy
            conn_recon = log_cau >= threshold
            error_mask = np.logical_xor(conn, conn_recon)
            fig_data['acc_gauss'][key] = 1-error_mask[inf_mask].sum()/len(error_mask[inf_mask])
            fig_data['ppv_gauss'][key] = (conn_recon[inf_mask]*conn[inf_mask]).sum()/conn[inf_mask].sum()
            counts_error, _ = np.histogram(log_cau[error_mask*inf_mask], bins=100, range=hist_range,density=True)
            fig_data['hist_error'][key] = counts_error.copy()
        else:
            fig_data['acc_gauss'][key] = np.nan
            fig_data['ppv_gauss'][key] = np.nan
            fig_data['hist_error'][key] = np.nan

        fpr, tpr, _ = roc_curve(conn[inf_mask*nan_mask], log_cau[inf_mask*nan_mask])
        fig_data['roc_gt'][key] = np.vstack((fpr, tpr))
        label = line_rc[key]['label'] + f" : {roc_auc_score(conn[inf_mask*nan_mask], log_cau[inf_mask*nan_mask]):.3f}"
        ax[0].plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=label)[0].set_clip_on(False)

    # print acc, 
    acc_kmeans_mean = np.nanmean(list(fig_data['acc_kmeans'].values())) 
    acc_gauss_mean  = np.nanmean(list(fig_data['acc_gauss'].values())) 
    ppv_kmeans_mean = np.nanmean(list(fig_data['ppv_kmeans'].values())) 
    ppv_gauss_mean  = np.nanmean(list(fig_data['ppv_gauss'].values())) 
    ax[0].text(0.15, 0.90, f"acc={acc_kmeans_mean*100:.2f}%", fontsize=18, transform=ax[0].transAxes)
    ax[0].text(0.15, 0.83, f"ppv={ppv_kmeans_mean*100:.2f}%", fontsize=18, transform=ax[0].transAxes)
    ax[1].text(0.15, 0.90, f"acc={acc_gauss_mean*100:.2f}%", fontsize=18, transform=ax[1].transAxes)
    ax[1].text(0.15, 0.83, f"ppv={ppv_gauss_mean*100:.2f}%", fontsize=18, transform=ax[1].transAxes)

    for axi, ax_histi in zip(ax, ax_hist):
        format_xticks(ax_histi, hist_range)
        ax_histi.set_ylim(0)
        ax_histi.set_yticks([])
        ax_histi.set_xlabel('Causal value')
        axi=roc_formatter(axi)
        axi.set_ylabel(axi.get_ylabel(), fontsize=25)
        axi.set_xlabel(axi.get_xlabel(), fontsize=25)
        axi.legend(loc='upper left', bbox_to_anchor=(0.55, 0.95), fontsize=14)

    plt.tight_layout()
    fig.savefig(f"image/histogram_of_causal_recon_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")
    return fig_data

import struct
def plot_raster(pm_causal:dict, xrange:tuple=(0,1000), return_spk:bool=False, ax=None):
    """plot sample raster plot given network parameters

    Args:
        pm_causal (dict): paramter dictionary
        xrange (tuple, optional): range of xaxis. Defaults to (0,1000).
        return_spk (bool, optional): return spike data. Defaults to False.
        ax (optional): matplotlib axis. Defaults to None.

    Returns:
        matplotlib.Figure: figure containing raster plot
    """
    N = int(pm_causal['Ne']+pm_causal['Ni'])
    spk_fname = f"{pm_causal['path_input']}EE/N={N:d}/{pm_causal['fname']:s}_spike_train.dat"
    n_time = 1000
    with open(spk_fname, 'rb') as f:
        data_buff = f.read(8*2*N*n_time)
        data_len = int(len(data_buff)/16)
        spk_data = np.array(struct.unpack('d'*2*data_len, data_buff)).reshape(-1,2)
        while spk_data[-1,0] < xrange[1]:
            data_buff = f.read(8*2*N*n_time)
            data_len = int(len(data_buff)/16)
            if data_len == 0:
                print('reaching end of file')
                break
            spk_data_more = np.array(struct.unpack('d'*2*data_len, data_buff)).reshape(-1,2)
            spk_data = np.concatenate((spk_data, spk_data_more), axis=0)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,3))
    else:
        fig = ax.get_figure()
    mask = (spk_data[:,0] > xrange[0]) * (spk_data[:,0] < xrange[1])
    ax.plot(spk_data[mask, 0], spk_data[mask, 1], '|')
    ax.set_xlim(*xrange)
    ax.set_xlabel('Time (ms)')
    print(f"mean firing rate is {spk_data.shape[0]/spk_data[-1,0]/N*1000.:.3f} Hz")
    plt.tight_layout()
    prefix = spk_fname.split('/')[-1].replace('_spike_train.dat', '')
    fig.savefig('image/'+f'raster_{prefix:s}.pdf')
    if return_spk:
        return fig, spk_data
    else:
        return fig