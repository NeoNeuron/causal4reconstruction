"""
    Library for plotting function used in analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from .Causality import CausalityIO, CausalityAPI
from .figrc import line_rc
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
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('True Negative Rate')
    ax[1].set_xticks([0,0.5,1])
    ax[1].set_yticks([0,0.5,1])
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1)
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
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('True Negative Rate')
    ax[1].set_xticks([0,0.5,1])
    ax[1].set_yticks([0,0.5,1])
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1)
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
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('True Negative Rate')
    ax[1].set_xticks([0,0.5,1])
    ax[1].set_yticks([0,0.5,1])
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1)
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
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('True Negative Rate')
    ax[1].set_xticks([0,0.5,1])
    ax[1].set_yticks([0,0.5,1])
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1)
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
    causal_values = {key: np.zeros(N*N-N) for key in ('TE', 'GC', 'MI', 'CC')}
    th_conditions = {}
    fig_data = {'roc_gt':{}, 'roc_blind':{}, 'hist': {}, 'hist_conn': {}, 'hist_disconn':{}, 'hist_error':{}}
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

    ax_hist = [inset_axes(axi, width="100%", height="100%",
                bbox_to_anchor=(.30, .20, .60, .40),
                bbox_transform=axi.transAxes, loc='center', axes_kwargs={'facecolor':[1,1,1,0]}) for axi in ax]
    for axi in ax_hist:
        axi.spines['left'].set_visible(False)
        axi.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        axi.xaxis.set_major_formatter(sci_formatter)

    # load connectivity matrix 
    fname_conn = pm_causal['path_output']+f"EE/N={N:d}/" + pm_causal['con_mat']
    conn = np.fromfile(fname_conn, dtype=float)[:int(N*N)].reshape(N,N).astype(bool)
    ratio = np.sum(conn[inf_mask])/np.sum(inf_mask)
    # plot the distribution of connectivity weight if exist
    conn_weight = np.fromfile(fname_conn, dtype=float)[int(N*N):]
    if conn_weight.shape[0] > 0:
        conn_weight= conn_weight.reshape(N,N)
        axins = inset_axes(ax[0], width="100%", height="100%",
                    bbox_to_anchor=(.05, .40, .45, .3),
                    bbox_transform=ax[0].transAxes, loc='center',)
        axins.spines['left'].set_visible(False)

        zero_mask = conn_weight>0
        axins.hist(conn_weight[inf_mask*zero_mask], bins=40)
        axins.set_title('Structural Conn', fontsize=12)
        axins.set_yticks([])

    data_dp = cau.load_from_single(fname, 'Delta_p')
    data_dp = np.log10(np.abs(data_dp))
    counts, bins = np.histogram(data_dp[inf_mask], bins=100, density=True)
    fig_data['hist_dp']  = counts.copy()
    fig_data['edges_dp'] = bins[:-1].copy()
    acc = ppv = 0.
    for key in ('CC', 'MI', 'GC', 'TE'):
        data_buff = cau.load_from_single(fname, key)
        causal_values[key] = data_buff[~np.eye(data_buff.shape[0], dtype=bool)].flatten()
        log_cau = np.log10(data_buff)
        if key in ('TE', 'MI'):
            log_cau += np.log10(2)
        counts_total = np.zeros(100)
        for mask, linestyle, ratio_, hist_key in zip((conn, ~conn), ('-', ':'), (ratio, 1-ratio), ('hist_conn', 'hist_disconn')):
            log_TE_buffer = log_cau[mask*inf_mask]
            counts, bins = np.histogram(log_TE_buffer, bins=100, range=hist_range,density=True)
            counts_total += counts*ratio_
            fig_data[hist_key][key] = counts*ratio_
            ax_hist[0].plot(bins[:-1], counts*ratio_, ls=linestyle, **line_rc[key])
        fig_data['hist'][key] = counts_total.copy()
        ax_hist[1].plot(bins[:-1], counts_total, **line_rc[key])
        if fit_p0 is None:
            popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts_total, bins)
        else:
            popt, threshold, fpr, tpr = Double_Gaussian_Analysis(counts_total, bins, p0=fit_p0)
        # ax_hist.plot(bins[:-1], Double_Gaussian(bins[:-1], *popt), lw=4, ls=':', color=line_rc[key]['color'], )
        ax_hist[1].plot(bins[:-1], Gaussian(bins[:-1], popt[0], popt[2], popt[4], ), '-.', lw=1, color=line_rc[key]['color'], )
        ax_hist[1].plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3], popt[5], ), '-.', lw=1, color=line_rc[key]['color'], )
        ax_hist[1].axvline(threshold, ymax=0.9, color=line_rc[key]['color'], ls='--')
        # plot double Gaussian based ROC
        label=line_rc[key]['label'] + f" : {-np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2):.3f}"
        ax[1].plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=label)[0].set_clip_on(False)

        log_norm_fit_pval[key] = popt
        th_conditions[key] = threshold
        fig_data['roc_blind'][key] = np.vstack((fpr, tpr))

        # calculate reconstruction accuracy
        conn_recon = log_cau >= threshold
        error_mask = np.logical_xor(conn, conn_recon)
        acc += 1-error_mask[inf_mask].sum()/len(error_mask[inf_mask])
        ppv += (conn_recon[inf_mask]*conn[inf_mask]).sum()/conn[inf_mask].sum()
        counts_error, _ = np.histogram(log_cau[error_mask*inf_mask], bins=100, range=hist_range,density=True)
        fig_data['hist_error'] = counts_error.copy()

        fpr, tpr, _ = roc_curve(conn[inf_mask], log_cau[inf_mask])
        label = line_rc[key]['label'] + f" : {roc_auc_score(conn[inf_mask], log_cau[inf_mask]):.3f}"
        ax[0].plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=label)[0].set_clip_on(False)
        fig_data['roc_gt'][key] = np.vstack((fpr, tpr))

    # print acc, 
    ax[1].text(0.15, 0.90, f"acc={acc/4.*100:.2f}%", fontsize=18, transform=ax[1].transAxes)
    ax[1].text(0.15, 0.83, f"ppv={ppv/4.*100:.2f}%", fontsize=18, transform=ax[1].transAxes)

    for axi, ax_histi in zip(ax, ax_hist):
        format_xticks(ax_histi, hist_range)
        ax_histi.set_ylim(0)
        ax_histi.set_yticks([])
        ax_histi.set_xlabel('Causal value')
        axi.set_xlim(0,1)
        axi.set_ylim(0,1)
        axi.set_xticks([0,0.5,1])
        axi.set_yticks([0,0.5,1])
        axi.set_ylabel('True Positive Rate',  fontsize=25)
        axi.set_xlabel('False Positive Rate', fontsize=25)
        axi.legend(loc='upper left', bbox_to_anchor=(0.55, 0.95), fontsize=14)

    fig_data['edges'] = bins[:-1].copy()
    fig_data['log_norm_fit_pval'] = log_norm_fit_pval
    fig_data['opt_th'] = th_conditions

    plt.tight_layout()
    fig.savefig(f"image/histogram_of_causal_recon_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}delay={pm_causal['delay']:.2f}_{fname:s}.pdf")
    return fig_data