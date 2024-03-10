import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.ndimage.filters import gaussian_filter1d
from struct import pack, _clearcache
from scipy.optimize import curve_fit
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Manager, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple
from .Causality import CausalityIO, run
from joblib import Parallel, delayed
from sklearn.cluster import KMeans


def spk2bin(spike_train:np.ndarray, dt:float)->np.ndarray:
    spk_bin = np.zeros(np.ceil(spike_train.max()/dt).astype(int)+1)
    spk_bin[(spike_train/dt).astype(int)] = 1
    return spk_bin

def save2bin(fname, data, fmode='wb', verbose=False, clearcache=False):
    with open(fname, fmode) as f:
        f.write(pack("d"*data.shape[0]*data.shape[1], *data.flatten()))
    if clearcache:
        _clearcache()
    if verbose:
        print(f">> save to {str(fname):s}")

def prepare_save_spikes(ts, spk0, *args, **kwargs):
    """convert a binary spiking raster to an array of spike 
        evnets data of network. Desiged for compatibility 
        with BrainPy simulation.

    Args:
        ts (array): time sequence of simulation
        spk0 (array): bool array of spiking raster of network

    Returns:
        array: spiking events of the network
    """
    spk0 = np.array(spk0)
    spk_idx = np.array(np.where(spk0))
    spk_time_total = np.zeros_like(spk_idx, dtype=float)
    spk_time_total[0,:] = ts[spk_idx[0,:]]
    spk_time_total[1,:] = spk_idx[1,:]
    if args:
        n = spk0.shape[1]
        for spk in args:
            spk = np.array(spk)
            spk_idx = np.array(np.where(spk))
            spk_time = np.zeros_like(spk_idx, dtype=float)
            spk_time[0,:] = ts[spk_idx[0,:]]
            spk_time[1,:] = spk_idx[1,:] + n
            spk_time_total = np.concatenate([spk_time_total, spk_time], axis=1)
            n += spk.shape[1]
        spk_time_total = spk_time_total[:,np.argsort(spk_time_total[0])]
    return spk_time_total

def conn2bin(net, sparse=True):
    """convert the connectviity matrix in EI network based on BrainPy
        to a whole connectivity matrix

    Args:
        net (bp.Networks): EI network built with BrainPy
        sparse (bool, optional): store in sparse matrix. Defaults to True.

    Returns:
        conn_mat (array): connectivity matrix
    """
    if not hasattr(net, 'I'):
        E2E = net.E2E
        if not sparse:
            conn_mat = E2E.conn.require('conn_mat').astype(float)
        else:
            pre_ids_all, post_ids_all = E2E.conn.require('pre_ids', 'post_ids')
            weights_all = np.ones_like(pre_ids_all)*net.w_e2e
            conn_mat = np.vstack([pre_ids_all, post_ids_all, weights_all])
    else:
        E2E, E2I, I2E, I2I = net.E2E, net.E2I, net.I2E, net.I2I
        if not sparse:
            Erows = np.concatenate([E2E.conn.require('conn_mat').astype(float), E2I.conn.require('conn_mat').astype(float)], axis=1)
            Irows = np.concatenate([I2E.conn.require('conn_mat').astype(float), I2I.conn.require('conn_mat').astype(float)], axis=1)
            conn_mat = np.concatenate([Erows, Irows], axis=0).astype(float)
        else:
            pre_ids_all, post_ids_all = E2E.conn.require('pre_ids', 'post_ids')
            weights_all = np.ones_like(pre_ids_all)*net.w_e2e
            pre_ids, post_ids = E2I.conn.require('pre_ids', 'post_ids')
            weights = np.ones_like(pre_ids)*net.w_e2i
            post_ids += net.num_e
            pre_ids_all = np.concatenate([pre_ids_all, pre_ids])
            post_ids_all = np.concatenate([post_ids_all, post_ids])
            weights_all = np.concatenate([weights_all, weights])
            pre_ids, post_ids = I2E.conn.require('pre_ids', 'post_ids')
            weights = np.ones_like(pre_ids)*(-net.w_i2e)
            pre_ids += net.num_e
            pre_ids_all = np.concatenate([pre_ids_all, pre_ids])
            post_ids_all = np.concatenate([post_ids_all, post_ids])
            weights_all = np.concatenate([weights_all, weights])
            pre_ids, post_ids = I2I.conn.require('pre_ids', 'post_ids')
            weights = np.ones_like(pre_ids)*(-net.w_i2i)
            pre_ids += net.num_e
            post_ids += net.num_e
            pre_ids_all = np.concatenate([pre_ids_all, pre_ids])
            post_ids_all = np.concatenate([post_ids_all, post_ids])
            weights_all = np.concatenate([weights_all, weights])
            conn_mat = np.vstack([pre_ids_all, post_ids_all, weights_all])
    return conn_mat

def plot_spk_fft(spk_data:np.ndarray, dt:float=0.5, ax:Axes=None,
    label:str=None)->Axes:

    signal = spk2bin(spk_data, dt=dt).astype(float)
    # if signal length longer than 100 seconds
    if signal.shape[0]>1e5:
        fourier = np.fft.fft(signal[:100000])
        n = 100000
    else:
        fourier = np.fft.fft(signal)
        n = signal.size
    timestep = dt/1000.
    freq = np.fft.fftfreq(n, d=timestep)
    freq_mask = (freq<=200)*(freq>0)
    ax.plot(
        freq[freq_mask], 
        gaussian_filter1d(np.abs(fourier[freq_mask])**2,sigma=50),
        label=label,
        )
    if label is not None:
        ax.legend(fontsize=14)
    return ax 

def Linear_R2(x:np.ndarray, y:np.ndarray, pval:np.ndarray)->float:
    """Compute R-square value for poly fitting.

    Args:
        x (np.ndarray): variable of function
        y (np.ndarray): image of function
        pval (np.ndarray): parameter of linear fitting

    Returns:
        float: R square value
    """
    mask = ~np.isnan(x)*~np.isnan(y)*~np.isinf(x)*~np.isinf(y)# filter out nan
    deg = len(pval)
    if deg < 2:
        raise ValueError(f'len(pval) must be greater than 2, (len(pval) = {deg:d})')
    y_predict = np.zeros_like(y[mask])
    for i in np.arange(deg, dtype=int):
        y_predict += pval[i]*x[mask]**(deg-i-1)
    R = np.corrcoef(y[mask], y_predict)[0,1]
    return R**2

def polyR2(x, y, pval)->float:
    return Linear_R2(x, y, pval)

def force_refractory(spike_time:np.ndarray, t_ref:float)->np.ndarray:
    """Using forced-refractory period to down sample given spike train

    Args:
        spike_time (np.ndarray): (n,2) array of spike sequence.
            First entry is spike time, and second one is index of spiking neurons
            in each row.
        t_ref (float): refractory period time, unit ms.

    Returns:
        np.ndarray: downsampled spike train (spike times).
    """

    node_ids = np.unique(spike_time[:,1])
    # n_unit = node_ids.shape[0]

    data_out = spike_time.copy()
    data_out[:,0] *= 1000.0 # convert second to ms

    for unit_id in node_ids:
        buffer = data_out[data_out[:,1] == unit_id,0].copy()
        previous_spike = buffer[0]
        for i in range(1, buffer.shape[0]):
            if buffer[i]-previous_spike >= t_ref:
                previous_spike = buffer[i]
            else:
                buffer[i] = np.nan
        data_out[data_out[:,1] == unit_id,0] = buffer.copy()
    
    return data_out[~np.isnan(data_out[:,0]),:].copy()

# Calculate spike trigered activities
def STA(xdata:np.ndarray, spk_data:np.ndarray, pair:np.ndarray, data_len:int, half_window:int):
    """Calculate spike triggered activities.

    Args:
        xdata (np.ndarray): continuous-valued time series.
        spk_data (np.ndarray): spike train data.
        pair (np.ndarray): indices of pair of neurons 
        data_len (int): length of data
        half_window (int): size of half of the sliding window.

    Returns:
        delay: array of relative delays
        sta: array of sta
    """
    sta = np.zeros(2*half_window+1)
    delay = (np.arange(2*half_window+1)-half_window)*0.01
    for spk in spk_data[spk_data[:,1]==pair[0],0]:
        spk_time_id = int(spk/0.01)
        if spk_time_id-half_window < 0:
            pass
        elif spk_time_id+half_window+1>data_len:
            pass
        else:
            sta += xdata[spk_time_id-half_window:spk_time_id+half_window+1,pair[1]+1]
    sta /= np.sum(spk_data[:,1]==1)
    return delay,sta

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

def Gaussian(x, a, mu, sigma2):
    """Standand Gaussian template function.

    Args:
        x (array-like): 1-d input data.
        a (float): positive contrained Gaussian amplitude
        mu (float): mean
        sigma2 (float): variance

    Returns:
        array-like: function value
    """
    return np.abs(a)*np.exp(-(x-mu)**2/sigma2)

def Double_Gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2):
    """Double Gaussian like template function.

    Args:
        x (array-like): 1d input data
        a1 (float): amplitude of first Gaussian
        a2 (float): amplitude of second Gaussian
        mu1 (float): mean of first Gaussian
        mu2 (float): mean of second Gaussian
        sigma1 (float): variance of first Gaussian
        sigma2 (float): variance of second Gaussian

    Returns:
        array-like: function value
    """
    return Gaussian(x, a1, mu1, sigma1) + Gaussian(x, a2, mu2, sigma2)

def Double_Gaussian_Analysis(counts, bins, p0=None):
    if p0 is None:
        p0 = [0.5, 0.5, -7, -5, 1, 1]
    try:
        popt, _ = curve_fit(Double_Gaussian, bins[:-1], counts, p0=p0)
        # calculate threshold, find crossing point of two Gaussian
        if popt[2] < popt[3]:
            x_grid = np.linspace(popt[2]-0.02, popt[3]+0.02, 10000)
        else:
            x_grid = np.linspace(popt[3]-0.02, popt[2]+0.02, 10000)
        th_ = x_grid[np.argmin(np.abs(Gaussian(x_grid, popt[0], popt[2], popt[4], )-Gaussian(x_grid, popt[1], popt[3], popt[5], )))]

        true_cumsum = np.cumsum(Gaussian(bins[:-1], popt[0], popt[2], popt[4], ))
        true_cumsum /= true_cumsum[-1]
        false_cumsum = np.cumsum(Gaussian(bins[:-1], popt[1], popt[3], popt[5], ))
        false_cumsum /= false_cumsum[-1]
        if popt[2] < popt[3]:
            true_cumsum, false_cumsum = false_cumsum, true_cumsum
        fpr = 1-false_cumsum
        tpr = 1-true_cumsum
        return popt, th_, fpr, tpr
    except:
        return None, None, None, None

def mp_logger(q:Queue, logfile:Path):
    """write log to file in multiprocessing.Pool.

    Args:
        q (Queue): multiprocessing.Manager().Queue().
        logfile (Path): filename of log file.
    """
    with logfile.open('w') as out:
        while True:
            val = q.get()
            if val is None: break
            if isinstance(val, list):
                for item in val:
                    out.write('['+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+']:\t')
                    out.write(item + '\n')
            else:
                out.write('['+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+']:\t')
                out.write(val + '\n')
            out.write('-'*10+'\n')

def shrink_datafile(path:Path, f:float, fu:float, T:float, log_queue:Queue=None, dry_run:bool=True):
    """delete spike events with spike time greater than given value.

    Args:
        path (Path): folder path of spike data.
        f (float): Poisson feedforward strength.
        fu (float): products of Poisson strength and frequency.
        T (float): The upper bound of spike events to shrink.
        log_queue (Queue, optional): multiprocessing.Manager().Queue() to buffer logging info. Defaults to None.
        dry_run (bool, optional): Flag to dry_run, generating logging without any other operation. Defaults to True.
    """
    filename = Path(f"HHp=0.25s=0.020f={f:.3f}u={fu/f:.3f}_spike_train.dat")
    raw_data = np.fromfile(path/filename, dtype=float).reshape(-1,2)
    if raw_data[-1, 0] > T:
        new_data = raw_data[raw_data[:,0]<=T,:]
        if log_queue is not None:
            log_queue.put([
                "processing file: '" + str(path/filename) + "' ...",
                f"raw file contains {raw_data.shape[0]:d} events, and last spike time is {raw_data[-1,0]:f}.",
                f"new file contains {new_data.shape[0]:d} events, and last spike time is {new_data[-1,0]:f}.",
                ])
        if not dry_run:
            save2bin(path/filename, new_data, clearcache=True)

# cutting off the reduandent spike trains
def compress_data(path:Path=Path('./HH/data/EE/N=3/'), T:float=1e7, dry_run:bool=True):
    f_list = np.arange(16)*0.01 + 0.05
    fu_list = np.arange(21)*0.002 + 0.01
    ff, fufu = np.meshgrid(f_list, fu_list)

    m = Manager()
    queue = m.Queue()
    log_process = Process(target=mp_logger, args=(queue, path/'compress_data.log'))
    log_process.start()
    Parallel(n_jobs=8)(
        delayed(shrink_datafile)(path, f, fu, T, queue, dry_run) for f, fu in zip(ff.flatten(), fufu.flatten()))
    queue.put(None)
    log_process.join()
    log_process.close()

def scan_fu_process(dtype:str, f:float, fu:float, pm_causal:dict,
    run_times:int, shuffle:bool, use_exists:bool=True):
    _pm_causal = pm_causal.copy()
    _pm_causal['fname']=_pm_causal['fname'].split('f=')[0]+f'f={f:.3f}u={fu/f:.3f}'
    N = _pm_causal['Ne']+_pm_causal['Ni']
    cau = CausalityIO(dtype='HH', N=N, **_pm_causal)
    fname_buff = get_fname(dtype, f'data/EE/N={N:d}/', spk_fname=_pm_causal['fname'], **_pm_causal)
    arr = np.zeros((run_times, N, N), dtype=float)
    for idx in range(run_times):
        if Path(fname_buff).exists() and use_exists:
            pass
        else:
            run(False, shuffle, **_pm_causal)
        if shuffle:
            arr[idx]=cau.load_from_single(_pm_causal['fname']+'_shuffle', 'TE')
        else:
            arr[idx]=cau.load_from_single(_pm_causal['fname'], 'TE')
    return arr

from .Causality import get_fname

def scan_s_process(dtype:str, s:float, pm_causal:dict,
    run_times:int, shuffle:bool, use_exists:bool=False):
    _pm_causal = pm_causal.copy()
    fname = _pm_causal['fname']
    begin = fname.find('s=')
    end = fname.find('f=')
    _pm_causal['fname'] = fname.replace(fname[begin:end], f's={s:.3f}')
    N = _pm_causal['Ne']+_pm_causal['Ni']
    arr = np.zeros((run_times, N, N), dtype=float)
    cau = CausalityIO(dtype=dtype, N=N, **_pm_causal)
    fname_buff = get_fname(dtype, f'data/EE/N={N:d}/', spk_fname=_pm_causal['fname'], **_pm_causal)
    for idx in range(run_times):
        if Path(fname_buff).exists() and use_exists:
            pass
        else:
            run(False, shuffle, **_pm_causal)
        if shuffle:
            arr[idx]=cau.load_from_single(_pm_causal['fname']+'_shuffle', 'TE')
        else:
            arr[idx]=cau.load_from_single(_pm_causal['fname'], 'TE')
    return arr

from scipy import signal
from typing import Union
import struct

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

def spk_power_spectrum(pm:dict, fs: float=1000, idx:Union[int, list]=None,
                       T_range:tuple=(0,1000)) -> np.ndarray:
    """
    Calculates the power spectrum of a binary spike train.

    Args:
        pm (dict): Parameter dictionary.
        fs (float): Sampling frequency of the spike train.
        idx (Union[int, list]): 
            Index of the neuron to calculate the power spectrum.
            None for calculating the average power spectrum.
            Defaults to None.
        T_range (tuple): Time range of the spike train.

    Returns:
        freq (np.ndarray): Frequency array.
        psd (np.ndarray): Power spectrum.
    """
    N = int(pm['Ne']+pm['Ni'])
    spk_fname = f"{pm['path_input']}EE/N={N:d}/{pm['fname']:s}_spike_train.dat"
    n_time = 100
    with open(spk_fname, 'rb') as f:
        data_buff = f.read(8*2*N*n_time)
        spk_data = np.frombuffer(data_buff, dtype=np.float64).reshape(-1,2)
        while spk_data[-1,0] < T_range[1]:
            data_buff = f.read(8*2*N*n_time)
            data_len = int(len(data_buff)/16)
            if data_len == 0:
                print('reaching end of file')
                break
            spk_data_more = np.frombuffer(data_buff, dtype=np.float64).reshape(-1,2)
            spk_data = np.concatenate((spk_data, spk_data_more), axis=0)
    T_mask = (spk_data[:,0]>=T_range[0]) & (spk_data[:,0]<=T_range[1])
    spk_data = spk_data[T_mask,:]
    
    if idx is None:
        idx = np.arange(N)
    if hasattr(idx, '__len__'):
        for i in idx:
            binary_spk = np.zeros(int(spk_data[-1,0]/1e3*fs)+1, dtype=float)
            binary_spk[(spk_data[spk_data[:,1]==i,0]/1e3*fs).astype(int)] = 1
            # Calculate the Fourier transform of the spike train
            freq, _psd = signal.welch(
                binary_spk, fs, 
                nperseg=binary_spk.shape[0]/2, noverlap=binary_spk.shape[0]/4)
            psd = _psd if i==idx[0] else psd+_psd
        return freq, psd/len(idx)
    elif isinstance(idx, int):
        binary_spk = np.zeros(int(spk_data[-1,0]/1e3*fs)+1, dtype=float)
        binary_spk[(spk_data[spk_data[:,1]==idx,0]/1e3*fs).astype(int)] = 1
        # Calculate the Fourier transform of the spike train
        return signal.welch(
            binary_spk, fs,
            nperseg=binary_spk.shape[0]/2, noverlap=binary_spk.shape[0]/4)
    else:
        raise TypeError('idx must be int or list of int.')

# match cell-type, connectivity, masking to causality data
def match_features(data:pd.DataFrame, N:int, conn_file:str=None,
        Ni:int=0, EI_types:np.ndarray=None):
    """match causality data other network features, and transform causal values to log10-scale.
        TE and TDMI are modified according to the quantitative relations.

    Args:
        data (pd.DataFrame): causality data.
        N (int): number of (excitatory) neurons.
        conn_file (str): filename of connectivity matrix.
            Both dense and sparse matrix supported. If None, then connection property won't be added.
        Ni (int, optional): number of inhibitory neurons, if nonzero, the total number of neuron N+Ni.
            Defaults to 0.
        EI_types (dict, optional): {'neuron_id':[...], 'EI_type':[...]}. Defaults to None.

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
    if conn_file is not None:
        conn_raw = np.fromfile(conn_file, dtype=float)
        conn_weight = None
        if conn_raw.shape[0] >= N*N:
            print('>> matching: dense connectivity matrix.')
            conn = conn_raw[:int(N*N)].reshape(N,N).astype(bool)
            pre_id, post_id = np.where(conn)

            if conn_raw.shape[0] > N*N:
                conn_weight = conn_raw[int(N*N):].reshape(N,N).astype(float)
                conn_weight = conn_weight[pre_id, post_id]
        else:
            print('>> matching: sparse connectivity matrix.')
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
                {'pre_id':pre_id, 'post_id':post_id, 'weight':conn_weight})
            data_match = data_match.merge(weight_pairs, how='left', on=['pre_id', 'post_id'])
            data_match['weight'].fillna(0, inplace=True)

    # drop inf and nan rows, mainly self-connections
    data_match.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_match.dropna(inplace=True)
    return data_match

from sklearn.metrics import roc_auc_score, roc_curve
def reconstruction_analysis(data: pd.DataFrame,
                            hist_range: tuple = None,
                            nbins: int = 100,
                            fit_p0: list = None,
                            EI_mask: str = None,
                            weight_hist_type: str = 'linear') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform reconstruction analysis on the given data.

    Args:
        data (pd.DataFrame): The input data for analysis.
        hist_range (tuple, optional): The range of histogram value. Defaults to None.
        nbins (int, optional): The number of bins for histogram. Defaults to 100.
        fit_p0 (list, optional): The initial parameters for 2-pop kmeans and bimodal Gaussian fitting.
                                 Defaults to None.
        EI_mask (str, optional): The mask for filtering data based on cell E/I type.
                                 Defaults to None.
        weight_hist_type (str, optional): The type of histogram for connection strength.
                                          Defaults to 'linear'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing the reconstructed data and the analysis results.
            Note: if ground truth is not available, then the reconstruction performance won't be evaluated.

    Raises:
        ValueError: If the EI_mask is invalid.

    """
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
                np.log10(data_recon[data_recon['connection'].eq(1)]['weight']), bins=40)
        elif weight_hist_type == 'linear':
            counts, bins = np.histogram(
                data_recon[data_recon['connection'].eq(1)]['weight'], bins=40)
        else:
            raise ValueError('weight_hist_type error!')
        for key in data_fig.keys():
            if key == 'hist':
                data_fig[key]['conn']  = counts.copy()
            elif key == 'edges':
                data_fig[key]['conn'] = bins[:-1].copy()
            else:
                data_fig[key]['conn'] = np.nan

    if 'connection' in data_recon:
        ratio = data_recon['connection'].mean()

    if hist_range is None:
        # determine histogram value range
        hist_range = (np.floor(data_recon[['log-TE', 'log-GC', 'log-MI', 'log-CC']].min().min()),
                      np.ceil(data_recon[['log-TE', 'log-GC', 'log-MI', 'log-CC']].max()).max())

    counts, bins = np.histogram(data_recon['log-dp'], bins=nbins, density=True)
    for key in data_fig.keys():
        if key == 'hist':
            data_fig[key]['dp']  = counts.copy()
        elif key == 'edges':
            data_fig[key]['dp'] = bins[:-1].copy()
        else:
            data_fig[key]['dp'] = np.nan

    for key in ('CC', 'MI', 'GC', 'TE'):
        if 'connection' in data_recon:
            counts_total = np.zeros(nbins)
            for i, hist_key in enumerate(('hist_disconn', 'hist_conn')):
                buffer = data_recon[data_recon['connection'].eq(i)][f'log-{key}']
                if len(buffer)>0:
                    counts, bins = np.histogram(buffer, bins=nbins, range=hist_range, density=True)
                else:
                    counts, bins = np.histogram(buffer, bins=nbins, range=hist_range)
                    counts = counts.astype(float)
                counts *= np.abs(1-i-ratio)
                counts_total += counts.copy()
                data_fig[hist_key][key] = counts.copy()
            data_fig['edges'][key] = bins[:-1].copy()
            data_fig['hist'][key] = counts_total.copy()
        else:
            counts_total, bins = np.histogram(data_recon[f'log-{key}'], bins=nbins, range=hist_range, density=True)
            data_fig['edges'][key] = bins[:-1].copy()
            data_fig['hist'][key] = counts_total.copy()
            data_fig['hist_conn'][key] = np.nan
            data_fig['hist_disconn'][key] = np.nan

        # KMeans clustering causal values
        if fit_p0 is None:
            if 'connection' in data_recon:
                disconn_peak_id = data_fig['hist_disconn'][key].argmax()
                conn_peak_id = data_fig['hist_conn'][key].argmax()
                fit_p0 = [0.5, 0.5, bins[disconn_peak_id], bins[conn_peak_id], 1, 1]
            else:
                fit_p0 = [0.5, 0.5, bins[data_fig['hist'][key].argmax()], bins[data_fig['hist'][key].argmax()], 1, 1]
        try:
            th_kmeans = kmeans_1d(data_recon[f'log-{key}'].to_numpy(), np.array([[fit_p0[2]],[fit_p0[3]]]))
            data_fig['th_kmeans'][key] = th_kmeans
            data_recon[f'recon-kmeans-{key}'] = (data_recon[f'log-{key}'] >= th_kmeans).astype(int)
            if 'connection' in data_recon:
                error_mask = np.logical_xor(data_recon['connection'], data_recon[f'recon-kmeans-{key}'])
                data_fig['acc_kmeans'][key] = 1-error_mask.sum()/len(error_mask)
                data_fig['ppv_kmeans'][key] = (data_recon['connection']*data_recon[f'recon-kmeans-{key}']).sum()/data_recon[f'recon-kmeans-{key}'].sum()
            else:
                data_fig['acc_kmeans'][key] = np.nan
                data_fig['ppv_kmeans'][key] = np.nan
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
            if 'connection' in data_recon:
                error_mask = np.logical_xor(data_recon['connection'], data_recon[f'recon-gauss-{key}'])
                data_fig['acc_gauss'][key] = 1-error_mask.sum()/len(error_mask)
                data_fig['ppv_gauss'][key] = (data_recon['connection']*data_recon[f'recon-gauss-{key}']).sum()/data_recon[f'recon-gauss-{key}'].sum()
                counts_error, _ = np.histogram(data_recon[f'log-{key}'][error_mask], bins=nbins, range=hist_range, density=True)
                data_fig['hist_error'][key] = counts_error.copy()
            else:
                data_fig['acc_gauss'][key] = np.nan
                data_fig['ppv_gauss'][key] = np.nan
                data_fig['hist_error'][key] = np.nan
        else:
            data_fig['auc_gauss'][key] = np.nan
            data_fig['acc_gauss'][key] = np.nan
            data_fig['ppv_gauss'][key] = np.nan
            data_fig['hist_error'][key] = np.nan

        if 'connection' in data_recon:
            fpr, tpr, _ = roc_curve(data_recon['connection'], data_recon[f'log-{key}'])
            data_fig['roc_gt'][key] = np.vstack((fpr, tpr))
            try:
                auc = roc_auc_score(data_recon['connection'], data_recon[f'log-{key}'])
            except ValueError:
                print("Warning: only one class, AUC calculation error!")
                auc = np.nan
        else:
            data_fig['roc_gt'][key] = np.nan
            auc = np.nan
        data_fig['auc_kmeans'][key] = auc

    # Reorder columns in data
    if 'connection' in data_recon:
        new_columns = ['pre_id', 'post_id', 'connection', 'recon-kmeans-TE', 'recon-gauss-TE', 'log-TE', 'log-GC', 'log-MI', 'log-CC', 'log-dp']
    else:
        new_columns = ['pre_id', 'post_id', 'recon-kmeans-TE', 'recon-gauss-TE', 'log-TE', 'log-GC', 'log-MI', 'log-CC', 'log-dp']
    other_columns = [col for col in data_recon.columns if col not in new_columns]
    data_recon = data_recon[new_columns + other_columns]
        
    return data_recon, pd.DataFrame(data_fig)


if __name__ == '__main__':
    pass