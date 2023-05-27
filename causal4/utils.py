import numpy as np
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
            x_grid = np.linspace(popt[2]-0.2, popt[3]+0.2, 10000)
        else:
            x_grid = np.linspace(popt[3]-0.2, popt[2]+0.2, 10000)
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

if __name__ == '__main__':
    pass