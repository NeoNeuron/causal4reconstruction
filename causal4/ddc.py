"""
    Author: Kai Chen
    Modified from https://github.com/yschen13/DDC
"""
# %%
import numpy as np
from struct import unpack

def dCov(x, dt, cov_kws:dict={}):
    N = x.shape[0]
    dx = (x[:,2:] - x[:,:-2])/dt/2
    dx = np.hstack((dx.mean(1).reshape(-1,1), dx, dx.mean(1).reshape(-1,1)))
    return np.cov(dx, x, **cov_kws)[:N,N:]
    
def DDC(x, dt, cov_kws:dict={}):
    cov = np.cov(x,**cov_kws)
    dcov = dCov(x, dt, cov_kws)
    return dcov @ np.linalg.inv(cov)

def DDC_long(fname:str, N:int, batch:int, cov_kws:dict={}):
    cov = np.zeros((N,N))
    dcov = np.zeros((N,N))
    for _ in range(10):
        with open(fname, 'rb') as f:
            dat = np.array(unpack('d'*(N+1)*batch, f.read(8*batch*(N+1)))).reshape(-1,N+1)
            dt = dat[1,0]-dat[0,0]
            cov += np.cov(dat[:,1:].T,**cov_kws)
            dcov += dCov(dat[:,1:].T, dt, **cov_kws)
            del dat
    cov /= 10
    dcov /= 10
    return dcov @ np.linalg.inv(cov)


def dReLU(x, dt, theta, cov_kws:dict={}):
    N = x.shape[0]
    cov = np.cov(np.maximum(x, theta), x,**cov_kws)[:N,N:]
    dcov = dCov(x, dt, cov_kws)
    return dcov @ np.linalg.inv(cov)

def c_sensitivity(GT, FC):
    TP_value = np.abs(FC.flatten()[GT.flatten()!=0])
    FP_value = np.abs(FC.flatten()[GT.flatten()==0])
    FP_95q = np.quantile(FP_value,0.95)
    return np.sum(TP_value>FP_95q)/len(TP_value)