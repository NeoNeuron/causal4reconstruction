import numpy as np
from numba import njit, prange

#! Time-delayed cross correlation
def DCC(data:np.ndarray, delay:int)->np.ndarray:
    """Delayed cross correlation coefficient

    Args:
        data (np.ndarray): 2-D array data. Each row is a variable and each column is a observation.
        delay (int): number of delay steps.

    Returns:
        np.ndarray: 2-D array of correlation coeffcient of delayed data.
    """
    N, L = data.shape
    tdcc = np.corrcoef(data[:,0:L-int(delay)], data[:,int(delay):L])[:N,N:]
    return tdcc

@njit(nogil=True, parallel=True)
def TDCC_pair(x:np.ndarray, y:np.ndarray, delay:np.ndarray)->np.ndarray:
    """Time-delayed correlation coefficient

    Args:
        x (np.ndarray): 1-D data
        y (np.ndarray): 1-D data
        delay (np.ndarray): array of delayed indices.

    Returns:
        np.ndarray: array of correlation coefficient with different delay indices.
    """
    N = x.shape[0]
    tdcc = np.zeros(len(delay))
    for i in prange(len(delay)):
        tdcc[i] = np.corrcoef(x[0:N-int(delay[i])], y[int(delay[i]):N])[0,1]
    return tdcc

#! Time-delayed mutual information
@njit(nogil=True, parallel=False)
def hist2d_numba_seq(tracks:np.ndarray, bins:np.ndarray, ranges:np.ndarray)->np.ndarray:
    H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1
    return H

@njit(nogil=True)
def MI(x:np.ndarray, y:np.ndarray, bins:np.ndarray|tuple|list)->float:
    """mutual information estimator. Joint PDF is estimated using uniform binning.

    Args:
        x (np.ndarray): 1-D data
        y (np.ndarray): 1-D data
        bins (np.ndarray | tuple | list): number of bins for histogram2d estimation.

    Returns:
        float: value of mutual information
    """
    N = x.shape[0]
    ranges = np.array([[x.min(), x.max()], [y.min(), y.max()]])
    dat = np.vstack((x,y))
    _bins = np.array(bins)
    counts_xy = hist2d_numba_seq(dat, bins=_bins, ranges=ranges)
    x_mar = counts_xy.sum(1)
    y_mar = counts_xy.sum(0)
    mi_val = 0.0
    for i in range(_bins):
        for j in range(_bins):
            if counts_xy[i,j] > 0:
                mi_val += counts_xy[i,j]*np.log(counts_xy[i,j]/(x_mar[i]*y_mar[j]))
    return mi_val / N + np.log(N)

@njit(nogil=True, parallel=False)
def DMI(x:np.ndarray, y:np.ndarray, delay:int, bins:int):
    if delay == 0:
        return MI(x, y, [bins, bins])
    elif delay < 0:
        return MI(x[-delay:],y[:delay], [bins, bins])
    elif delay > 0:
        return MI(x[:-delay],y[delay:], [bins, bins])

@njit(nogil=True, parallel=True)
def TDMI(x:np.ndarray, y:np.ndarray, time_range:np.ndarray, bins:np.ndarray=None)->np.ndarray:
    tdmi = np.zeros(len(time_range))
    for i in prange(len(time_range)):
        tdmi[i] = DMI(x,y,time_range[i],bins)
    return tdmi

#! Granger Causality
def create_structure_array(x:np.ndarray, order:int)->np.ndarray:
    '''
    Prepare structure array for regression analysis.

    Args:
        x       : original time series
        order   : regression order

    Return:
        x_array : structure array with shape (len(x)-order) by (order).

    '''
    N = len(x) - order
    x_array = np.zeros((N, order))
    for i in range(order):
        x_array[:, i] = x[-i-1-N:-i-1]
    return x_array

def auto_reg(x:np.ndarray, order:int)->np.ndarray:
    '''
    Auto regression analysis of time series.

    Args:
        x     : original time series
        order : regression order

    Return:
        res   : residual vector

    '''
    reg_array = create_structure_array(x, order)
    coef = np.linalg.lstsq(reg_array, x[order:], rcond=None)[0]
    res = x[order:] - reg_array @ coef
    return res

def joint_reg(x:np.ndarray, y:np.ndarray, order:int)->np.ndarray:
    '''
    Joint regression analysis of time series.

    Args:
        x     : original time series 1
        y     : original time series 2
        order : regression order

    Return:
        res   : residual vector

    '''
    reg_array_x = create_structure_array(x, order)
    reg_array_y = create_structure_array(y, order)
    reg_array = np.hstack((reg_array_x, reg_array_y))
    coef = np.linalg.lstsq(reg_array, x[order:], rcond=None)[0]
    res = x[order:] - reg_array @ coef
    return res

def GC(x:np.ndarray, y:np.ndarray, order:int)->float:
    '''
    Granger Causality from y to x

    Args:
        x        : original time series (dest)
        y        : original time series (source)
        order    : regression order

    Return:
        GC_value : residual vector

    '''
    res_auto = auto_reg(x, order)
    res_joint = joint_reg(x, y, order)
    GC_value = 2.*np.log(res_auto.std()/res_joint.std())
    return GC_value

from scipy.stats import chi2
def GC_SI(p:float, order:int, length:int)->float:
    '''
    Significant level of GC value.

    Args
        p      : p-value
        order  : parameter of chi^2 distribution
        length : length of data.

    Return:
        significant level of null hypothesis (GC
            between two independent time seies)

    '''
    return chi2.ppf(1-p, order)/length

def DGC(x:np.ndarray, y:np.ndarray, delay:int, order:int=10)->float:
    if delay == 0:
        return GC(x, y, order)
    elif delay < 0:
        return GC(x[-delay:],y[:delay], order)
    elif delay > 0:
        return GC(x[:-delay],y[delay:], order)

def TDGC(x:np.ndarray, y:np.ndarray, time_range:np.ndarray, order:int=10)->np.ndarray:
    return np.array([DGC(x,y,delay,order) for delay in time_range])


@njit(nogil=True, parallel=False)
def hist3d_numba_seq(tracks, bins, ranges):
    H = np.zeros((bins[0], bins[1], bins[2]), dtype=np.uint64)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        k = (tracks[2, t] - ranges[2, 0]) * delta[2]
        if 0 <= i < bins[0] and 0 <= j < bins[1] and 0 <= k < bins[2]:
            H[int(i), int(j), int(k)] += 1
    return H


@njit(nogil=True, parallel=False)
def TE11_nb(x, y, bins = None):
    """Transfer entropy with order k=1, l=1 (Numba accelerated version).

    Args:
        x (numpy.array): time series of input-node
        y (numpy.array): time series of output-node
        bins (int, optional): number of bins of histogram. Defaults to None.

    Returns:
        TE: transfer entropy
    """
    N = x.shape[0]-1
    data = np.vstack((y[1:], y[:-1], x[:-1]))
    ranges = np.array([[y[1:].min(), y[1:].max()], [y[:-1].min(), y[:-1].max()], [x[:-1].min(), x[:-1].max()]])
    bins3 = np.array([bins, bins, bins])
    pyyx = hist3d_numba_seq(data, bins3, ranges)
    py  = pyyx.sum(2)
    py  = py.sum(0)
    pyx = pyyx.sum(0)
    pyy = pyyx.sum(2)
    te_val = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if pyyx[i,j,k]>0:
                    te_val += pyyx[i,j,k]*np.log((pyyx[i,j,k]*py[j])/(pyx[j,k]*pyy[i,j]))
    return te_val / N

def TE11(x:np.ndarray, y:np.ndarray, bins:int=None)->float:
    """Transfer entropy with order k=1, l=1

    Args:
        x (numpy.array): time series of input-node
        y (numpy.array): time series of output-node
        bins (int, optional): number of bins of histogram. Defaults to None.

    Returns:
        TE: transfer entropy
    """
    N = len(x)-1
    data = np.vstack((y[1:], y[:-1], x[:-1])).T

    pyyx,_ = np.histogramdd(data, (bins, bins, bins))
    py  = np.tile(pyyx.sum((0,2), keepdims=True), (bins, 1, bins))
    pyx = np.tile(pyyx.sum(0, keepdims=True), (bins, 1, 1))
    pyy = np.tile(pyyx.sum(2, keepdims=True), (1, 1, bins))
    mi_val = np.nansum(pyyx*np.log((pyyx*py)/(pyx*pyy)))
    return mi_val / N

@njit(nogil=True, parallel=False)
def DTE11_nb(x:np.ndarray, y:np.ndarray, delay:int, bins:int=None)->float:
    if delay == 0:
        return TE11_nb(x, y, bins)
    elif delay < 0:
        return TE11_nb(x[-delay:],y[:delay], bins)
    elif delay > 0:
        return TE11_nb(x[:-delay],y[delay:], bins)
