import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import gaussian_filter1d
import struct
from scipy.optimize import curve_fit

def spk2bin(spike_train:np.ndarray, dt:float)->np.ndarray:
    spk_bin = np.zeros(np.ceil(spike_train.max()/dt).astype(int)+1)
    spk_bin[(spike_train/dt).astype(int)] = 1
    return spk_bin

def save2bin(fname, data, fmode='wb', verbose=False):
    with open(fname, fmode) as f:
        f.write(struct.pack("d"*data.shape[0]*data.shape[1], *data.flatten()))
    if verbose:
        print(f">> save to {fname:s}")

def plot_spk_fft(spk_data:np.ndarray, dt:float=0.5, ax:mpl.axes.Axes=None,
    label:str=None)->mpl.axes.Axes:

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

# calculate the cross correlation
from multiprocessing import Pool
def TDCC(data, delay, mp=None):
    N = data.shape[0]
    ddelay = delay[1]-delay[0]
    dt = data[1,0]-data[0,0]
    ref_indices = ddelay/dt*np.arange(delay.shape[0]) + delay[0]/dt
    ref_indices = ref_indices.astype(int)
    print(ref_indices)
    if mp is None:
        tdcc = [np.corrcoef(data[0:N-i, 1], data[i:N,2])[0,1] for i in ref_indices]
    else:
        p = Pool(mp)
        result = [
            p.apply_async(
            func = np.corrcoef, args=(data[0:N-i, 1], data[i:N,2],),
            ) for i in ref_indices
        ]
        p.close()
        p.join()
        tdcc = [res.get()[0,1] for res in result]
    return np.array(tdcc)

def MI(x, y, bins = None):
    """mutual information for 0-1 binary time series
    :param x: first series
    :type x: int of ndarray
    :param y: second series
    :type y: int of ndarray
    :return: mutual information

    """
    N = len(x)
    if bins is None:
        bins = np.sqrt(N).astype(int)
    pxy,_,_ = np.histogram2d(x, y, bins)
    pxx = np.tile(pxy.sum(1), (bins, 1)).T
    pyy = np.tile(pxy.sum(0), (bins, 1))
    mask = pxy!=0
    mi_val = np.sum(pxy[mask]*np.log(pxy[mask]/pxx[mask]/pyy[mask]))
    return mi_val / N + np.log(N)

def DMI(x, y, delay, bins=None):
    if delay == 0:
        return MI(x, y, bins)
    elif delay < 0:
        return MI(x[-delay:],y[:delay], bins)
    elif delay > 0:
        return MI(x[:-delay],y[delay:], bins)

def TDMI(x, y, time_range, bins=None):
    return np.array([DMI(x,y,delay,bins) for delay in time_range])

from scipy.stats import chi2
def create_structure_array(x:np.ndarray, order:int)->np.ndarray:
    '''
    Prepare structure array for regression analysis.

    Args:
    x         : original time series
    order     : regression order

    Return:
    x_array   : structure array with shape (len(x)-order) by (order).

    '''
    N = len(x) - order
    x_array = np.zeros((N, order))
    for i in range(order):
        x_array[:, i] = x[-i-1-N:-i-1]
    return x_array

def auto_reg(x, order)->np.ndarray:
    '''
    Auto regression analysis of time series.

    Args:
    x         : original time series
    order     : regression order

    Return:
    res       : residual vector

    '''
    reg_array = create_structure_array(x, order)
    coef = np.linalg.lstsq(reg_array, x[order:], rcond=None)[0]
    res = x[order:] - reg_array @ coef
    return res

def joint_reg(x, y, order)->np.ndarray:
    '''
    Joint regression analysis of time series.

    Args:
    x         : original time series 1
    y         : original time series 2
    order     : regression order

    Return:
    res       : residual vector

    '''
    reg_array_x = create_structure_array(x, order)
    reg_array_y = create_structure_array(y, order)
    reg_array = np.hstack((reg_array_x, reg_array_y))
    coef = np.linalg.lstsq(reg_array, x[order:], rcond=None)[0]
    res = x[order:] - reg_array @ coef
    return res

def GC(x, y, order):
    '''
    Granger Causality from y to x

    Args:
    x         : original time series (dest)
    y         : original time series (source)
    order     : regression order

    Return:
    GC_value  : residual vector

    '''
    res_auto = auto_reg(x, order)
    res_joint = joint_reg(x, y, order)
    GC_value = 2.*np.log(res_auto.std()/res_joint.std())
    return GC_value

def GC_SI(p, order, length):
    '''
    Significant level of GC value.

    Args
    p       : p-value
    order   : parameter of chi^2 distribution
    length  : length of data.

    Return:
    significant level of null hypothesis (GC
        between two independent time seies)

    '''
    return chi2.ppf(1-p, order)/length

def DGC(x, y, delay, order=10):
    if delay == 0:
        return GC(x, y, order)
    elif delay < 0:
        return GC(x[-delay:],y[:delay], order)
    elif delay > 0:
        return GC(x[:-delay],y[delay:], order)

def TDGC(x, y, time_range, order=10):
    return np.array([DGC(x,y,delay,order) for delay in time_range])

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

def Double_Gaussian_Analysis(counts, bins, p0=[0.5, 0.5, -7, -5, 1, 1]):
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

if __name__ == '__main__':
    pass