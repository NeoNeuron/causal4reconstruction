"""
    Library for plotting function used in analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from .Causality import CausalityIO
from .figrc import line_rc
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
from sklearn.metrics import roc_auc_score, roc_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
            axins.set_xticklabels([r'$10^{%d}$'%val for val in axins.get_xticks()], fontsize=12)
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
    xticks = np.arange(hist_range[0], hist_range[-1]+1)
    ax[0].set_xticks(xticks[::2])
    ax[0].set_xlim(xticks[0], xticks[-1])
    ax[0].set_xticklabels([r'$10^{%d}$'%val for val in xticks[::2]])
    ax[1].legend()
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('True Negative Rate')
    ax[1].set_xticks([0,0.5,1])
    ax[1].set_yticks([0,0.5,1])
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1)
    plt.tight_layout()

    fig.savefig('image/'+f"histogram_of_causal_with_conn_mask_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}_{fname:s}.pdf")


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
            axins.set_xticklabels([r'$10^{%d}$'%val for val in axins.get_xticks()], fontsize=12)
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

    fig.savefig('image/'+f"histogram_of_causal_with_conn_mask_linear_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}_{fname:s}.pdf")

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
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    if hist_range is not None:
        ax.set_xlim(*hist_range)
    ax.set_xticklabels([r'$10^{%.0f}$'%val for val in xticks])
    ax.set_xlabel(r'$\Delta p_m$ or $\delta_p$ values')
    ax.set_ylabel('Probability density')
    ax.axvline(0, ls='--', color='r')
    plt.tight_layout()

    if 'T' in pm_causal:
        fig.savefig('image/'+f"histogram_of_dp_Delta_p_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}_{fname:s}.pdf")
    else:
        fig.savefig('image/'+f"histogram_of_dp_Delta_p_bin={pm_causal['bin']:.3f}_{fname:s}.pdf")


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
        data_buff = cau.load_from_single(fname, 'TE')[inf_mask]
        data_buff = np.log10(data_buff)
        hist_range = (np.floor(data_buff.min())+1, np.ceil(data_buff.max())+1)

    for counter, key in enumerate(('CC', 'MI', 'GC', 'TE')):
        data_buff = cau.load_from_single(fname, key)
        log_TE = np.log10(data_buff)
        if key in ('TE', 'MI'):
            log_TE += np.log10(2)
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
    xticks = np.arange(hist_range[0], hist_range[-1]+1)
    ax[0].set_xticks(xticks[::2])
    ax[0].set_xlim(xticks[0], xticks[-1])
    ax[0].set_xticklabels([r'$10^{%d}$'%val for val in xticks[::2]])
    ax[1].legend()
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('True Negative Rate')
    ax[1].set_xticks([0,0.5,1])
    ax[1].set_yticks([0,0.5,1])
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1)
    plt.tight_layout()

    fig.savefig('image/'+f"histogram_of_causal_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}_{fname:s}.pdf")


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

    fig.savefig('image/'+f"histogram_of_causal_linear_T={pm_causal['T']:0.0e}bin={pm_causal['bin']:.3f}_{fname:s}.pdf")